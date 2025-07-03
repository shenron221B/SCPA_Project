#include "../include/openmp_spmv.h"
#include <omp.h>
#include <stdio.h>

void openmp_spmv_hll(const HLLMatrix *A_hll, const float *x, float *y, int num_threads) {
    if (!A_hll || !x || !y) {
        fprintf(stderr, "error [openmp_spmv_hll]: null pointer argument.\n");
        return;
    }

    if (num_threads <= 0) {
        omp_set_num_threads(omp_get_max_threads());
    } else {
        omp_set_num_threads(num_threads);
    }

    #pragma omp parallel
    {
        int tid = omp_get_thread_num();
        int nthreads = omp_get_num_threads();

        // partition on HLL block, not row
        int blocks_per_thread = (A_hll->num_blocks + nthreads - 1) / nthreads;
        int start_block = tid * blocks_per_thread;
        int end_block = (tid + 1) * blocks_per_thread;

        if (end_block > A_hll->num_blocks) {
            end_block = A_hll->num_blocks;
        }

        // each thread iterate on its set of blocks
        for (int block_idx = start_block; block_idx < end_block; ++block_idx) {
            const ELLPACKBlock *block = &A_hll->blocks[block_idx];

            // internal loop on rows
            for (int r_block = 0; r_block < block->num_rows_in_block; ++r_block) {

                // calculate global index of row to write on y[]
                int global_row_idx = block_idx * A_hll->hack_size + r_block;

                if (global_row_idx >= A_hll->total_rows) continue;

                if (block->max_nz_per_row == 0 || !block->JA_ell || !block->AS_ell) {
                    y[global_row_idx] = 0.0f;
                    continue;
                }

                float sum_row = 0.0f;
                int max_nz = block->max_nz_per_row;
                int rows_in_block = block->num_rows_in_block;

                for (int k_ell = 0; k_ell < max_nz; ++k_ell) {
                    int ja_idx = k_ell * rows_in_block + r_block;
                    float val = block->AS_ell[ja_idx];
                    if (val != 0.0f) {
                        int col = block->JA_ell[ja_idx];
                        // check on columns of x
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