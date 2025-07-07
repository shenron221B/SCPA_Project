#include "../include/openmp_spmv.h"
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>

/**
 * @brief Pre-calculates the block boundaries for each thread for HLL format.
 *
 * This function partitions the HLL blocks into chunks for static scheduling in OpenMP.
 * It assigns an approximately equal number of HLL blocks to each thread. This is a
 * simple and effective strategy for static load balancing when the primary unit of
 * parallel work is an entire block. The calculation is parallelized to be efficient.
 *
 * @param A_hll       A pointer to the input HLLMatrix structure.
 * @param num_threads The target number of threads.
 * @return            A dynamically allocated integer array of size (num_threads + 1)
 *                    containing the block boundaries. The caller must free this memory.
 */
int* prepare_hll_bounds(const HLLMatrix *A_hll, int num_threads) {
    if (num_threads <= 0 || !A_hll) return NULL;

    int* block_bounds = (int*)malloc((num_threads + 1) * sizeof(int));
    if (!block_bounds) {
        perror("Failed to allocate block_bounds");
        return NULL;
    }

    // set the start and end points of the entire range
    block_bounds[0] = 0;
    block_bounds[num_threads] = A_hll->num_blocks;
    // calculate the number of blocks per thread using ceiling division
    int blocks_per_thread = (A_hll->num_blocks + num_threads - 1) / num_threads;

    // parallelize the calculation of the intermediate boundaries
    #pragma omp parallel for
    for (int i = 1; i < num_threads; ++i) {
        int bound = i * blocks_per_thread;
        block_bounds[i] = (bound > A_hll->num_blocks) ? A_hll->num_blocks : bound;
    }
    return block_bounds;
}

/**
 * @brief Computes the sparse matrix-vector product (y = Ax) for a matrix in HLL format using OpenMP.
 *
 * This function parallelizes the SpMV operation by partitioning the HLL blocks among
 * the available threads. Each thread is assigned a contiguous chunk of blocks to process.
 * This static, block-level partitioning strategy is chosen for its stability and low overhead,
 * especially when dynamic scheduling proves problematic. The data within each ELLPACK block
 * is assumed to be in column-major layout for cache efficiency.
 *
 * @param A_hll         Pointer to the input HLLMatrix structure (const).
 * @param x             Pointer to the input vector x (const).
 * @param y             Pointer to the output vector y, which will be filled with the result.
 * @param num_threads   The number of OpenMP threads to use for the computation. If <= 0,
 *                      OpenMP's default number of threads will be used.
 * @param block_bounds  An array of size (num_threads + 1) that defines the start and end
 *                      block indices for each thread. `thread[tid]` processes blocks from
 *                      `block_bounds[tid]` to `block_bounds[tid + 1] - 1`.
 */
void openmp_spmv_hll(const HLLMatrix *A_hll, const float *x, float *y, int num_threads, const int* block_bounds) {
    // input validation to prevent null pointer
    if (!A_hll || !x || !y || !block_bounds) return;

    // set the number of threads for the parallel region
    if (num_threads > 0) {
        omp_set_num_threads(num_threads);
    }

    // start a parallel region. The work (HLL blocks) is partitioned manually
    #pragma omp parallel
    {
        // each thread identifies itself
        int tid = omp_get_thread_num();

        // each thread gets its assigned range of blocks from the pre-computed bounds
        int start_block = block_bounds[tid];
        int end_block = block_bounds[tid + 1];

        // each thread iterates over its assigned chunk of HLL blocks
        for (int block_idx = start_block; block_idx < end_block; ++block_idx) {
            const ELLPACKBlock *block = &A_hll->blocks[block_idx];
            for (int r_block = 0; r_block < block->num_rows_in_block; ++r_block) {
                // calculate the global row index to write to the correct position in the output vector y
                int global_row_idx = block_idx * A_hll->hack_size + r_block;
                // boundary check to handle the last block, which might not be full
                if (global_row_idx >= A_hll->total_rows) continue;
                float sum_row = 0.0f;
                int max_nz = block->max_nz_per_row;
                int rows_in_block = block->num_rows_in_block;
                // loop over the padded width (max_nz) of the ELLPACK block
                for (int k_ell = 0; k_ell < max_nz; ++k_ell) {
                    // calculate the 1D index for column-major access
                    int ja_idx = k_ell * rows_in_block + r_block;
                    float val = block->AS_ell[ja_idx];
                    // padded elements have a value of 0, so this check skips unnecessary multiplications
                    if (val != 0.0f) {
                        int col = block->JA_ell[ja_idx];
                        if (col >= 0 && col < A_hll->total_cols) {
                            sum_row += val * x[col];
                        }
                    }
                }
                // write the final result for the current global row
                y[global_row_idx] = sum_row;
            }
        }
    }
}