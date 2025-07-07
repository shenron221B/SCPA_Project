#include "../include/cuda_hll_spmv.h"
#include <cstdio>
#include <cuda_runtime.h>

#define CUDA_HLL_CHECK(err) { \
    if (err != cudaSuccess) { \
        fprintf(stderr, "cuda error in %s at line %d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
        return err; \
    } \
}

// texture memory for vector x
texture<float, cudaTextureType1D, cudaReadModeElementType> x_tex_hll;

/**
 * @brief CUDA kernel for SpMV on a matrix in the optimized HLL format.
 *
 * This kernel implements the "one thread per row" strategy. It is designed to work with
 * a "flattened" data structure, where all ELLPACK block data is stored in contiguous
 * global memory arrays. This approach, combined with a column-major data layout within
 * each logical ELLPACK block, ensures coalesced memory access to the matrix data, which
 * is critical for high performance on GPUs. Access to the input vector 'x' is optimized
 * using the texture cache.
 *
 * @param total_rows        The total number of rows in the entire matrix.
 * @param hack_size         The number of rows per HLL block, used to map a global row
 *                          index to a specific block and a local row within that block.
 * @param d_ell_blocks_meta A device pointer to an array of metadata structures. Each
 *                          structure contains information for one HLL block, including its
 *                          dimensions and the starting offsets into the flattened data arrays.
 * @param d_JA_all_blocks   A device pointer to a single, large ("flattened") array containing
 *                          the column indices for all HLL blocks, stored contiguously.
 * @param d_AS_all_blocks   A device pointer to the flattened array of non-zero values for
 *                          all HLL blocks.
 * @param d_y               A device pointer to the output vector y.
 */
__global__ void spmv_hll_kernel(int total_rows, int hack_size,
                                const ELLPACKBlock_device* d_ell_blocks_meta, // array of block metadata
                                const int* d_JA_all_blocks,     // unique array for JA
                                const float* d_AS_all_blocks,   // unique array for AS
                                float* __restrict__ d_y) {
    int global_row_idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (global_row_idx < total_rows) {
        int block_idx = global_row_idx / hack_size;
        int r_block = global_row_idx % hack_size;

        const ELLPACKBlock_device* current_block_meta = &d_ell_blocks_meta[block_idx];

        // check if row is valid for this block
        if (r_block >= current_block_meta->num_rows_in_block) {
            d_y[global_row_idx] = 0.0f;
            return;
        }

        float sum_row = 0.0f;
        int max_nz = current_block_meta->max_nz_per_row;
        int rows_in_block = current_block_meta->num_rows_in_block;

        // pointers for data of this block
        const int*   block_JA = d_JA_all_blocks + current_block_meta->ja_start_offset;
        const float* block_AS = d_AS_all_blocks + current_block_meta->as_start_offset;

        for (int k_ell = 0; k_ell < max_nz; ++k_ell) {
            // calculate index as: k_ell * row_in_block + local_row
            int flat_idx = k_ell * rows_in_block + r_block;

            float val = block_AS[flat_idx];
            if (val != 0.0f) {
                int col = block_JA[flat_idx];
                sum_row += val * tex1Dfetch(x_tex_hll, col);
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
    ELLPACKBlock_device *d_ell_blocks_meta_gpu = NULL;
    int *d_JA_all_blocks = NULL;
    float *d_AS_all_blocks = NULL;

    // allocation and copy of x and y
    err = cudaMalloc((void **)&d_x_gpu, h_A_hll->total_cols * sizeof(float)); CUDA_HLL_CHECK(err);
    err = cudaMalloc((void **)&d_y_gpu, h_A_hll->total_rows * sizeof(float)); CUDA_HLL_CHECK(err);
    err = cudaMemcpy(d_x_gpu, h_x, h_A_hll->total_cols * sizeof(float), cudaMemcpyHostToDevice); CUDA_HLL_CHECK(err);

    // binding texture of x
    err = cudaBindTexture(NULL, x_tex_hll, d_x_gpu, h_A_hll->total_cols * sizeof(float)); CUDA_HLL_CHECK(err);

    if (h_A_hll->num_blocks > 0) {
        // 1. calculate necessary total dimension for all blocks JA e AS
        size_t total_ja_elements = 0;
        size_t total_as_elements = 0;
        for (int i = 0; i < h_A_hll->num_blocks; ++i) {
            size_t elements_in_block = (size_t)h_A_hll->blocks[i].num_rows_in_block * h_A_hll->blocks[i].max_nz_per_row;
            total_ja_elements += elements_in_block;
            total_as_elements += elements_in_block;
        }

        // 2. allocate a unique buffer for JA and for AS
        err = cudaMalloc((void**)&d_JA_all_blocks, total_ja_elements * sizeof(int)); CUDA_HLL_CHECK(err);
        err = cudaMalloc((void**)&d_AS_all_blocks, total_as_elements * sizeof(float)); CUDA_HLL_CHECK(err);

        // 3. prepare metadata on host and copy data in flat buffers
        ELLPACKBlock_device* h_ell_blocks_meta_temp = (ELLPACKBlock_device*)malloc(h_A_hll->num_blocks * sizeof(ELLPACKBlock_device));
        size_t ja_current_offset = 0;
        size_t as_current_offset = 0;

        for (int i = 0; i < h_A_hll->num_blocks; ++i) {
            const ELLPACKBlock *h_block = &h_A_hll->blocks[i];
            size_t elements_in_block = (size_t)h_block->num_rows_in_block * h_block->max_nz_per_row;

            h_ell_blocks_meta_temp[i].num_rows_in_block = h_block->num_rows_in_block;
            h_ell_blocks_meta_temp[i].max_nz_per_row = h_block->max_nz_per_row;
            h_ell_blocks_meta_temp[i].ja_start_offset = ja_current_offset;
            h_ell_blocks_meta_temp[i].as_start_offset = as_current_offset;

            if (elements_in_block > 0) {
                // copy data of block at correct offset in the buffer
                err = cudaMemcpy(d_JA_all_blocks + ja_current_offset, h_block->JA_ell, elements_in_block * sizeof(int), cudaMemcpyHostToDevice); CUDA_HLL_CHECK(err);
                err = cudaMemcpy(d_AS_all_blocks + as_current_offset, h_block->AS_ell, elements_in_block * sizeof(float), cudaMemcpyHostToDevice); CUDA_HLL_CHECK(err);
            }

            ja_current_offset += elements_in_block;
            as_current_offset += elements_in_block;
        }

        // 4. copy array of metadata (with offset) on GPU
        err = cudaMalloc((void**)&d_ell_blocks_meta_gpu, h_A_hll->num_blocks * sizeof(ELLPACKBlock_device)); CUDA_HLL_CHECK(err);
        err = cudaMemcpy(d_ell_blocks_meta_gpu, h_ell_blocks_meta_temp, h_A_hll->num_blocks * sizeof(ELLPACKBlock_device), cudaMemcpyHostToDevice); CUDA_HLL_CHECK(err);

        free(h_ell_blocks_meta_temp);
    }

    // timing and run kernel
    cudaEvent_t start_event, stop_event;
    err = cudaEventCreate(&start_event); CUDA_HLL_CHECK(err);
    err = cudaEventCreate(&stop_event); CUDA_HLL_CHECK(err);

    if (threads_per_block_dim <= 0 || threads_per_block_dim > 1024) threads_per_block_dim = 256;
    dim3 threads_per_block(threads_per_block_dim);
    dim3 num_hll_kernel_blocks((h_A_hll->total_rows + threads_per_block.x - 1) / threads_per_block.x);

    err = cudaEventRecord(start_event, 0); CUDA_HLL_CHECK(err);

    spmv_hll_kernel<<<num_hll_kernel_blocks, threads_per_block>>>(
        h_A_hll->total_rows, h_A_hll->hack_size,
        d_ell_blocks_meta_gpu,
        d_JA_all_blocks,
        d_AS_all_blocks,
        d_y_gpu
    );
    err = cudaGetLastError(); CUDA_HLL_CHECK(err);

    err = cudaEventRecord(stop_event, 0); CUDA_HLL_CHECK(err);
    err = cudaEventSynchronize(stop_event); CUDA_HLL_CHECK(err);

    float kernel_elapsed_ms = 0;
    cudaEventElapsedTime(&kernel_elapsed_ms, start_event, stop_event);
    if(kernel_time_s) *kernel_time_s = (double)kernel_elapsed_ms / 1000.0;

    cudaEventDestroy(start_event);
    cudaEventDestroy(stop_event);

    err = cudaMemcpy(h_y, d_y_gpu, h_A_hll->total_rows * sizeof(float), cudaMemcpyDeviceToHost); CUDA_HLL_CHECK(err);

    // cleanup
    cudaUnbindTexture(x_tex_hll);
    cudaFree(d_y_gpu);
    cudaFree(d_x_gpu);
    cudaFree(d_ell_blocks_meta_gpu);
    cudaFree(d_JA_all_blocks);
    cudaFree(d_AS_all_blocks);

    return cudaSuccess;
}