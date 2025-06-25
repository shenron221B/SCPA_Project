#include "../include/cuda_hll_spmv.h"
#include <cstdio>
#include <cuda_runtime.h>

#define CUDA_HLL_CHECK(err) { \
    if (err != cudaSuccess) { \
        fprintf(stderr, "cuda error in %s at line %d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
        return err; \
    } \
}

// kernel for HLL SpMV - one common strategy is one thread per row
__global__ void spmv_hll_kernel(int total_rows, int total_cols, int hack_size, int num_hll_blocks,
                                const ELLPACKBlock_device* d_ell_blocks_meta, // array of block metadata
                                const int* const* d_JA_block_arrays,     // array of pointers to JA arrays for each block
                                const float* const* d_AS_block_arrays,   // array of pointers to AS arrays for each block
                                const float* __restrict__ d_x,
                                float* __restrict__ d_y) {
    int global_row_idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (global_row_idx < total_rows) {
        int block_idx = global_row_idx / hack_size;
        int r_block = global_row_idx % hack_size;

        // these checks might be redundant if grid is perfectly sized, but good for safety
        if (block_idx >= num_hll_blocks || r_block >= d_ell_blocks_meta[block_idx].num_rows_in_block) {
            d_y[global_row_idx] = 0.0f; // out of bounds for actual matrix data
            return;
        }

        const ELLPACKBlock_device* current_block_meta = &d_ell_blocks_meta[block_idx];
        const int*   current_JA_ell = d_JA_block_arrays[block_idx];
        const float* current_AS_ell = d_AS_block_arrays[block_idx];

        if (current_block_meta->max_nz_per_row == 0 || !current_JA_ell || !current_AS_ell) {
            d_y[global_row_idx] = 0.0f;
            return;
        }

        float sum_row = 0.0f;
        for (int k_ell = 0; k_ell < current_block_meta->max_nz_per_row; ++k_ell) {
            int ja_flat_idx = r_block * current_block_meta->max_nz_per_row + k_ell;
            float val = current_AS_ell[ja_flat_idx];
            if (val != 0.0f) {
                int col = current_JA_ell[ja_flat_idx];
                 if (col >= 0 && col < total_cols) { // boundary check
                    sum_row += val * d_x[col];
                }
            }
        }
        d_y[global_row_idx] = sum_row;
    }
}

/**
 * @brief host wrapper function for HLL SpMV using CUDA.
 * returns the kernel execution time via kernel_time_s.
 *
 * @param h_A_hll pointer to the host HLLMatrix structure.
 * @param h_x pointer to the host input vector x.
 * @param h_y pointer to the host output vector y.
 * @param threads_per_block_dim desired number of threads per CUDA block (1D dimension).
 * @param kernel_time_s pointer to a double to store the kernel execution time in seconds.
 * @return cudaSuccess (0) on success, or a non-zero CUDA error code.
 */
int cuda_spmv_hll_wrapper(const HLLMatrix *h_A_hll, const float *h_x, float *h_y, int threads_per_block_dim, double *kernel_time_s) {
    if (kernel_time_s) *kernel_time_s = -1.0; // initialize to error/default

    if (!h_A_hll || !h_x || !h_y || h_A_hll->num_blocks < 0) {
        fprintf(stderr, "error [cuda_spmv_hll_wrapper]: invalid host arguments.\n");
        return cudaErrorInvalidValue;
    }
    if (h_A_hll->total_rows == 0) {
        if (kernel_time_s) *kernel_time_s = 0.0;
        return cudaSuccess;
    }
    if (h_A_hll->total_cols == 0 && h_A_hll->total_rows > 0) {
        fprintf(stderr, "error [cuda_spmv_hll_wrapper]: matrix has rows but 0 columns.\n");
        return cudaErrorInvalidValue;
    }

    cudaError_t err;
    float *d_x_gpu = NULL, *d_y_gpu = NULL;
    ELLPACKBlock_device *h_ell_blocks_meta_temp = NULL; // host array for metadata
    ELLPACKBlock_device *d_ell_blocks_meta_gpu = NULL;  // device array for metadata
    int   **h_JA_dev_ptrs_temp = NULL; // host array of device pointers to JA_ell
    float **h_AS_dev_ptrs_temp = NULL; // host array of device pointers to AS_ell
    int   **d_JA_block_gpu_ptrs = NULL; // device array of device pointers to JA_ell
    float **d_AS_block_gpu_ptrs = NULL; // device array of device pointers to AS_ell

    cudaEvent_t start_event, stop_event;

    // allocate and copy x and y vectors
    err = cudaMalloc((void **)&d_x_gpu, h_A_hll->total_cols * sizeof(float)); CUDA_HLL_CHECK(err);
    err = cudaMalloc((void **)&d_y_gpu, h_A_hll->total_rows * sizeof(float)); CUDA_HLL_CHECK(err);
    err = cudaMemcpy(d_x_gpu, h_x, h_A_hll->total_cols * sizeof(float), cudaMemcpyHostToDevice); CUDA_HLL_CHECK(err);
    err = cudaMemset(d_y_gpu, 0, h_A_hll->total_rows * sizeof(float)); CUDA_HLL_CHECK(err);


    if (h_A_hll->num_blocks > 0) {
        // allocate host-side temporary arrays for metadata and device pointers
        h_ell_blocks_meta_temp = (ELLPACKBlock_device *)malloc(h_A_hll->num_blocks * sizeof(ELLPACKBlock_device));
        h_JA_dev_ptrs_temp = (int**)malloc(h_A_hll->num_blocks * sizeof(int*));
        h_AS_dev_ptrs_temp = (float**)malloc(h_A_hll->num_blocks * sizeof(float*));
        if (!h_ell_blocks_meta_temp || !h_JA_dev_ptrs_temp || !h_AS_dev_ptrs_temp) {
            perror("error [cuda_spmv_hll_wrapper]: malloc failed for host temp arrays");
            if(d_x_gpu) cudaFree(d_x_gpu); if(d_y_gpu) cudaFree(d_y_gpu);
            if(h_ell_blocks_meta_temp) free(h_ell_blocks_meta_temp);
            if(h_JA_dev_ptrs_temp) free(h_JA_dev_ptrs_temp);
            if(h_AS_dev_ptrs_temp) free(h_AS_dev_ptrs_temp);
            return cudaErrorMemoryAllocation;
        }


        // for each HLL block, allocate its JA and AS arrays on device and copy data
        for (int i = 0; i < h_A_hll->num_blocks; ++i) {
            const ELLPACKBlock *h_block = &h_A_hll->blocks[i];
            h_ell_blocks_meta_temp[i].num_rows_in_block = h_block->num_rows_in_block;
            h_ell_blocks_meta_temp[i].max_nz_per_row = h_block->max_nz_per_row;
            h_ell_blocks_meta_temp[i].d_JA_ell = NULL; // will be set by kernel from d_JA_block_gpu_ptrs
            h_ell_blocks_meta_temp[i].d_AS_ell = NULL; // " "

            if (h_block->num_rows_in_block > 0 && h_block->max_nz_per_row > 0) {
                size_t ja_size_bytes = (size_t)h_block->num_rows_in_block * h_block->max_nz_per_row * sizeof(int);
                err = cudaMalloc((void **)&(h_JA_dev_ptrs_temp[i]), ja_size_bytes); CUDA_HLL_CHECK(err); // Add proper cleanup on error
                err = cudaMemcpy(h_JA_dev_ptrs_temp[i], h_block->JA_ell, ja_size_bytes, cudaMemcpyHostToDevice); CUDA_HLL_CHECK(err);

                size_t as_size_bytes = (size_t)h_block->num_rows_in_block * h_block->max_nz_per_row * sizeof(float);
                err = cudaMalloc((void **)&(h_AS_dev_ptrs_temp[i]), as_size_bytes); CUDA_HLL_CHECK(err);
                err = cudaMemcpy(h_AS_dev_ptrs_temp[i], h_block->AS_ell, as_size_bytes, cudaMemcpyHostToDevice); CUDA_HLL_CHECK(err);
            } else {
                h_JA_dev_ptrs_temp[i] = NULL;
                h_AS_dev_ptrs_temp[i] = NULL;
            }
        }

        // allocate memory on GPU for the array of ELLPACKBlock_device metadata structs
        err = cudaMalloc((void **)&d_ell_blocks_meta_gpu, h_A_hll->num_blocks * sizeof(ELLPACKBlock_device)); CUDA_HLL_CHECK(err);
        // copy the metadata array from host to device
        err = cudaMemcpy(d_ell_blocks_meta_gpu, h_ell_blocks_meta_temp, h_A_hll->num_blocks * sizeof(ELLPACKBlock_device), cudaMemcpyHostToDevice); CUDA_HLL_CHECK(err);

        // allocate memory on GPU for the array of JA pointers and copy them
        err = cudaMalloc((void ***)&d_JA_block_gpu_ptrs, h_A_hll->num_blocks * sizeof(int*)); CUDA_HLL_CHECK(err);
        err = cudaMemcpy(d_JA_block_gpu_ptrs, h_JA_dev_ptrs_temp, h_A_hll->num_blocks * sizeof(int*), cudaMemcpyHostToDevice); CUDA_HLL_CHECK(err);

        // allocate memory on GPU for the array of AS pointers and copy them
        err = cudaMalloc((void ***)&d_AS_block_gpu_ptrs, h_A_hll->num_blocks * sizeof(float*)); CUDA_HLL_CHECK(err);
        err = cudaMemcpy(d_AS_block_gpu_ptrs, h_AS_dev_ptrs_temp, h_A_hll->num_blocks * sizeof(float*), cudaMemcpyHostToDevice); CUDA_HLL_CHECK(err);
    }

    // --- create CUDA events for timing ---
    err = cudaEventCreate(&start_event); CUDA_HLL_CHECK(err);
    err = cudaEventCreate(&stop_event); CUDA_HLL_CHECK(err);

    // --- kernel launch configuration ---
    if (threads_per_block_dim <= 0 || threads_per_block_dim > 1024) threads_per_block_dim = 256;
    dim3 threads_per_block(threads_per_block_dim);
    dim3 num_hll_kernel_blocks((h_A_hll->total_rows + threads_per_block.x - 1) / threads_per_block.x);

    // record start event (just before kernel launch)
    err = cudaEventRecord(start_event, 0); CUDA_HLL_CHECK(err);

    spmv_hll_kernel<<<num_hll_kernel_blocks, threads_per_block>>>(
        h_A_hll->total_rows, h_A_hll->total_cols, h_A_hll->hack_size, h_A_hll->num_blocks,
        d_ell_blocks_meta_gpu,      // array of metadata structs (on device)
        d_JA_block_gpu_ptrs,      // array of JA_ell pointers (on device)
        d_AS_block_gpu_ptrs,      // array of AS_ell pointers (on device)
        d_x_gpu, d_y_gpu
    );
    err = cudaGetLastError(); CUDA_HLL_CHECK(err);

    // record stop event and synchronize
    err = cudaEventRecord(stop_event, 0); CUDA_HLL_CHECK(err);
    err = cudaEventSynchronize(stop_event); CUDA_HLL_CHECK(err);

    // calculate kernel elapsed time
    float kernel_elapsed_ms = 0;
    err = cudaEventElapsedTime(&kernel_elapsed_ms, start_event, stop_event); CUDA_HLL_CHECK(err);
    if (kernel_time_s) *kernel_time_s = (double)kernel_elapsed_ms / 1000.0;

    // destroy events
    cudaEventDestroy(start_event);
    cudaEventDestroy(stop_event);

    // copy result y back to host
    err = cudaMemcpy(h_y, d_y_gpu, h_A_hll->total_rows * sizeof(float), cudaMemcpyDeviceToHost); CUDA_HLL_CHECK(err);

    // --- cleanup GPU memory ---
    if (d_ell_blocks_meta_gpu) { // this is the array of metadata structs on device
        // free the individual JA_ell and AS_ell arrays for each block
        if (h_A_hll->num_blocks > 0 && h_JA_dev_ptrs_temp && h_AS_dev_ptrs_temp) { // check if host temp ptrs were alloc
            for (int i = 0; i < h_A_hll->num_blocks; ++i) {
                if (h_JA_dev_ptrs_temp[i]) cudaFree(h_JA_dev_ptrs_temp[i]); // use the host-side copy of device pointers
                if (h_AS_dev_ptrs_temp[i]) cudaFree(h_AS_dev_ptrs_temp[i]);
            }
        }
        cudaFree(d_ell_blocks_meta_gpu); // free the array of metadata structs
        cudaFree(d_JA_block_gpu_ptrs);  // free the array of JA pointers
        cudaFree(d_AS_block_gpu_ptrs);  // free the array of AS pointers
    }
    if (h_ell_blocks_meta_temp) free(h_ell_blocks_meta_temp);
    if (h_JA_dev_ptrs_temp) free(h_JA_dev_ptrs_temp);
    if (h_AS_dev_ptrs_temp) free(h_AS_dev_ptrs_temp);

    cudaFree(d_x_gpu);
    cudaFree(d_y_gpu);

    return cudaSuccess;
}