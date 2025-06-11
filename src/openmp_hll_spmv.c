// File: src/openmp_hll_spmv.c
#include "../include/openmp_spmv.h"
#include <omp.h>
#include <stdio.h>

void openmp_spmv_hll(const HLLMatrix *A_hll, const float *x, float *y, int num_threads) {
    if (!A_hll || !x || !y) {
        fprintf(stderr, "error [openmp_spmv_hll]: null pointer argument.\n");
        return;
    }

    if (num_threads > 0) {
        omp_set_num_threads(num_threads);
    }

    // parallelize the outer loop over blocks, or over rows.
    // parallelizing over rows is often simpler to map directly.
    // each thread will calculate a range of global rows.
    // y must be initialized to 0 before this parallel region if sum_row is added to y[global_row_idx]
    // or each y[global_row_idx] assigned directly.
    // for simplicity, assign directly, so no prior y init needed for correctness here.
    #pragma omp parallel for schedule(static)
    for (int global_row_idx = 0; global_row_idx < A_hll->total_rows; ++global_row_idx) {
        // determine which block and which row within the block this global_row_idx corresponds to
        int block_idx = global_row_idx / A_hll->hack_size;
        int r_block = global_row_idx % A_hll->hack_size;

        // ensure block_idx is valid (especially for the last few rows if total_rows is not a multiple of hack_size)
        if (block_idx >= A_hll->num_blocks) {
            // this should not happen if logic is correct, but as a safeguard
            // or if a thread gets an out-of-bounds global_row_idx due to scheduling
            y[global_row_idx] = 0.0f; // or handle error
            continue;
        }

        const ELLPACKBlock *block = &A_hll->blocks[block_idx];

        // check if this specific row (r_block) is valid for this block
        if (r_block >= block->num_rows_in_block) {
             // this can happen if the last block is smaller than hack_size
             y[global_row_idx] = 0.0f;
             continue;
        }

        if (block->max_nz_per_row == 0 && block->num_rows_in_block > 0) {
            y[global_row_idx] = 0.0f;
            continue;
        }
        if (!block->JA_ell || !block->AS_ell) {
            y[global_row_idx] = 0.0f;
            continue;
        }

        float sum_row = 0.0f;
        for (int k_ell = 0; k_ell < block->max_nz_per_row; ++k_ell) {
            int ja_idx = r_block * block->max_nz_per_row + k_ell;
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