#include "../include/openmp_spmv.h"
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>

int* prepare_hll_bounds(const HLLMatrix *A_hll, int num_threads) {
    if (num_threads <= 0 || !A_hll) return NULL;

    int* block_bounds = (int*)malloc((num_threads + 1) * sizeof(int));
    if (!block_bounds) {
        perror("Failed to allocate block_bounds");
        return NULL;
    }

    block_bounds[0] = 0;
    block_bounds[num_threads] = A_hll->num_blocks;
    int blocks_per_thread = (A_hll->num_blocks + num_threads - 1) / num_threads;

    #pragma omp parallel for
    for (int i = 1; i < num_threads; ++i) {
        int bound = i * blocks_per_thread;
        block_bounds[i] = (bound > A_hll->num_blocks) ? A_hll->num_blocks : bound;
    }
    return block_bounds;
}

void openmp_spmv_hll(const HLLMatrix *A_hll, const float *x, float *y, int num_threads, const int* block_bounds) {
    if (!A_hll || !x || !y || !block_bounds) return;

    if (num_threads > 0) {
        omp_set_num_threads(num_threads);
    }

    #pragma omp parallel
    {
        int tid = omp_get_thread_num();
        int start_block = block_bounds[tid];
        int end_block = block_bounds[tid + 1];

        for (int block_idx = start_block; block_idx < end_block; ++block_idx) {
            // ... il resto del codice è IDENTICO a prima ...
            const ELLPACKBlock *block = &A_hll->blocks[block_idx];
            for (int r_block = 0; r_block < block->num_rows_in_block; ++r_block) {
                int global_row_idx = block_idx * A_hll->hack_size + r_block;
                if (global_row_idx >= A_hll->total_rows) continue;
                float sum_row = 0.0f;
                int max_nz = block->max_nz_per_row;
                int rows_in_block = block->num_rows_in_block;
                for (int k_ell = 0; k_ell < max_nz; ++k_ell) {
                    int ja_idx = k_ell * rows_in_block + r_block;
                    float val = block->AS_ell[ja_idx];
                    if (val != 0.0f) {
                        int col = block->JA_ell[ja_idx];
                        if (col >= 0 && col < A_hll->total_cols) {
                            sum_row += val * x[col];
                        }
                    }
                }
                y[global_row_idx] = sum_row;
            }
        }
    }
}