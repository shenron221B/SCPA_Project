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


int cuda_spmv_hll_wrapper(const HLLMatrix *h_A_hll, const float *h_x, float *h_y, int threads_per_block_dim) {
    if (!h_A_hll || !h_x || !h_y || h_A_hll->num_blocks < 0) {
        fprintf(stderr, "error [cuda_spmv_hll_wrapper]: invalid host arguments.\n");
        return cudaErrorInvalidValue;
    }
    if (h_A_hll->total_rows == 0) { // empty matrix
        return cudaSuccess; // nothing to do, y should remain as is (or be zeroed by caller)
    }

    cudaError_t err;
    float *d_x_gpu, *d_y_gpu;

    // allocate and copy x and y vectors
    err = cudaMalloc((void **)&d_x_gpu, h_A_hll->total_cols * sizeof(float)); CUDA_HLL_CHECK(err);
    err = cudaMalloc((void **)&d_y_gpu, h_A_hll->total_rows * sizeof(float)); CUDA_HLL_CHECK(err);
    err = cudaMemcpy(d_x_gpu, h_x, h_A_hll->total_cols * sizeof(float), cudaMemcpyHostToDevice); CUDA_HLL_CHECK(err);
    // it's good practice to initialize d_y_gpu to 0 if the kernel doesn't guarantee writing all y entries
    // or if some rows might be skipped.
    err = cudaMemset(d_y_gpu, 0, h_A_hll->total_rows * sizeof(float)); CUDA_HLL_CHECK(err);


    // prepare device representation for HLL blocks
    ELLPACKBlock_device *d_ell_blocks_meta_gpu = NULL;
    int **d_JA_block_arrays_gpu_ptrs = NULL;   // array of pointers (on GPU) to JA arrays (on GPU)
    float **d_AS_block_arrays_gpu_ptrs = NULL; // array of pointers (on GPU) to AS arrays (on GPU)

    if (h_A_hll->num_blocks > 0) {
        // 1. Allocate memory on GPU for the array of metadata structs
        d_ell_blocks_meta_gpu = (ELLPACKBlock_device *)malloc(h_A_hll->num_blocks * sizeof(ELLPACKBlock_device)); // HOST temp
        if (!d_ell_blocks_meta_gpu) { perror("malloc d_ell_blocks_meta_gpu host"); cudaFree(d_x_gpu); cudaFree(d_y_gpu); return 1;}

        // 2. Allocate memory on GPU for the arrays of pointers
        err = cudaMalloc((void***)&d_JA_block_arrays_gpu_ptrs, h_A_hll->num_blocks * sizeof(int*)); CUDA_HLL_CHECK(err);
        err = cudaMalloc((void***)&d_AS_block_arrays_gpu_ptrs, h_A_hll->num_blocks * sizeof(float*)); CUDA_HLL_CHECK(err);

        // Temporary host arrays to hold device pointers before copying the whole array of pointers to device
        int** h_temp_JA_device_ptrs = (int**)malloc(h_A_hll->num_blocks * sizeof(int*));
        float** h_temp_AS_device_ptrs = (float**)malloc(h_A_hll->num_blocks * sizeof(float*));
        if(!h_temp_JA_device_ptrs || !h_temp_AS_device_ptrs) {
             perror("malloc h_temp ptrs"); /* cleanup */ return 1;
        }


        // 3. For each block, allocate its JA_ell and AS_ell on GPU, copy data, and store device ptr
        for (int i = 0; i < h_A_hll->num_blocks; ++i) {
            const ELLPACKBlock *h_block = &h_A_hll->blocks[i];
            d_ell_blocks_meta_gpu[i].num_rows_in_block = h_block->num_rows_in_block;
            d_ell_blocks_meta_gpu[i].max_nz_per_row = h_block->max_nz_per_row;

            if (h_block->max_nz_per_row > 0 && h_block->num_rows_in_block > 0) {
                size_t blockSizeBytesJA = (size_t)h_block->num_rows_in_block * h_block->max_nz_per_row * sizeof(int);
                size_t blockSizeBytesAS = (size_t)h_block->num_rows_in_block * h_block->max_nz_per_row * sizeof(float);

                err = cudaMalloc((void **)&(h_temp_JA_device_ptrs[i]), blockSizeBytesJA); CUDA_HLL_CHECK(err);
                err = cudaMemcpy(h_temp_JA_device_ptrs[i], h_block->JA_ell, blockSizeBytesJA, cudaMemcpyHostToDevice); CUDA_HLL_CHECK(err);

                err = cudaMalloc((void **)&(h_temp_AS_device_ptrs[i]), blockSizeBytesAS); CUDA_HLL_CHECK(err);
                err = cudaMemcpy(h_temp_AS_device_ptrs[i], h_block->AS_ell, blockSizeBytesAS, cudaMemcpyHostToDevice); CUDA_HLL_CHECK(err);
            } else {
                h_temp_JA_device_ptrs[i] = NULL;
                h_temp_AS_device_ptrs[i] = NULL;
            }
        }
        // 4. Copy the array of device pointers (JA) from host to device
        err = cudaMemcpy(d_JA_block_arrays_gpu_ptrs, h_temp_JA_device_ptrs, h_A_hll->num_blocks * sizeof(int*), cudaMemcpyHostToDevice); CUDA_HLL_CHECK(err);
        // 5. Copy the array of device pointers (AS) from host to device
        err = cudaMemcpy(d_AS_block_arrays_gpu_ptrs, h_temp_AS_device_ptrs, h_A_hll->num_blocks * sizeof(float*), cudaMemcpyHostToDevice); CUDA_HLL_CHECK(err);

        // also copy the metadata array itself to device
        ELLPACKBlock_device* d_ell_blocks_meta_final_gpu;
        err = cudaMalloc((void**)&d_ell_blocks_meta_final_gpu, h_A_hll->num_blocks * sizeof(ELLPACKBlock_device)); CUDA_HLL_CHECK(err);
        err = cudaMemcpy(d_ell_blocks_meta_final_gpu, d_ell_blocks_meta_gpu, h_A_hll->num_blocks * sizeof(ELLPACKBlock_device), cudaMemcpyHostToDevice); CUDA_HLL_CHECK(err);


        free(d_ell_blocks_meta_gpu); // free host temporary array for metadata
        free(h_temp_JA_device_ptrs);
        free(h_temp_AS_device_ptrs);
        d_ell_blocks_meta_gpu = d_ell_blocks_meta_final_gpu; // use the one on device
    }


    // kernel launch configuration
    if (threads_per_block_dim <= 0 || threads_per_block_dim > 1024) threads_per_block_dim = 256;
    dim3 threads_per_block(threads_per_block_dim);
    dim3 num_hll_kernel_blocks((h_A_hll->total_rows + threads_per_block.x - 1) / threads_per_block.x);

    spmv_hll_kernel<<<num_hll_kernel_blocks, threads_per_block>>>(
        h_A_hll->total_rows, h_A_hll->total_cols, h_A_hll->hack_size, h_A_hll->num_blocks,
        d_ell_blocks_meta_gpu, // this is the array of structs on device
        d_JA_block_arrays_gpu_ptrs,
        d_AS_block_arrays_gpu_ptrs,
        d_x_gpu, d_y_gpu
    );
    err = cudaGetLastError(); CUDA_HLL_CHECK(err);
    err = cudaDeviceSynchronize(); CUDA_HLL_CHECK(err);

    // copy result y back to host
    err = cudaMemcpy(h_y, d_y_gpu, h_A_hll->total_rows * sizeof(float), cudaMemcpyDeviceToHost); CUDA_HLL_CHECK(err);

    // cleanup
    if (d_ell_blocks_meta_gpu) { // this is the array of ELLPACKBlock_device structs on GPU
        // To free the JA_ell and AS_ell arrays pointed to by d_JA_block_arrays_gpu_ptrs and d_AS_block_arrays_gpu_ptrs:
        // We need to get these pointers back to the host to iterate and free one by one.
        // This is a bit complex. A simpler model might be to allocate one giant slab for all JAs and one for all ASs.
        // For this model (array of pointers):
        if (h_A_hll->num_blocks > 0) {
            int** h_JA_ptrs_to_free = (int**)malloc(h_A_hll->num_blocks * sizeof(int*));
            float** h_AS_ptrs_to_free = (float**)malloc(h_A_hll->num_blocks * sizeof(float*));

            cudaMemcpy(h_JA_ptrs_to_free, d_JA_block_arrays_gpu_ptrs, h_A_hll->num_blocks * sizeof(int*), cudaMemcpyDeviceToHost);
            cudaMemcpy(h_AS_ptrs_to_free, d_AS_block_arrays_gpu_ptrs, h_A_hll->num_blocks * sizeof(float*), cudaMemcpyDeviceToHost);

            for (int i = 0; i < h_A_hll->num_blocks; ++i) {
                if (h_JA_ptrs_to_free[i]) cudaFree(h_JA_ptrs_to_free[i]);
                if (h_AS_ptrs_to_free[i]) cudaFree(h_AS_ptrs_to_free[i]);
            }
            free(h_JA_ptrs_to_free);
            free(h_AS_ptrs_to_free);

            cudaFree(d_JA_block_arrays_gpu_ptrs);
            cudaFree(d_AS_block_arrays_gpu_ptrs);
        }
        cudaFree(d_ell_blocks_meta_gpu); // Free the array of metadata structs itself
    }
    cudaFree(d_x_gpu);
    cudaFree(d_y_gpu);

    return cudaSuccess;
}