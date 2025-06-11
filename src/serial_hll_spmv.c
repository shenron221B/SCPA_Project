// File: src/serial_hll_spmv.c
#include "../include/serial.h" // Will declare serial_spmv_hll
#include <stdio.h>

void serial_spmv_hll(const HLLMatrix *A_hll, const float *x, float *y) {
    if (!A_hll || !x || !y) {
        fprintf(stderr, "error [serial_spmv_hll]: null pointer argument.\n");
        return;
    }

    // initialize y to zero
    for (int i = 0; i < A_hll->total_rows; ++i) {
        y[i] = 0.0f;
    }

    int current_global_row = 0;
    for (int block_idx = 0; block_idx < A_hll->num_blocks; ++block_idx) {
        const ELLPACKBlock *block = &A_hll->blocks[block_idx];
        if (block->max_nz_per_row == 0 && block->num_rows_in_block > 0) { // block with only empty rows
            current_global_row += block->num_rows_in_block; // y is already 0.0f
            continue;
        }
        if (!block->JA_ell || !block->AS_ell) { // skip if block arrays are null (e.g. max_nz_per_row was 0)
            current_global_row += block->num_rows_in_block;
            continue;
        }


        for (int r_block = 0; r_block < block->num_rows_in_block; ++r_block) {
            if (current_global_row >= A_hll->total_rows) break; // safety break

            float sum_row = 0.0f;
            // iterate through elements in this row of the ELLPACK block
            for (int k_ell = 0; k_ell < block->max_nz_per_row; ++k_ell) {
                int ja_idx = r_block * block->max_nz_per_row + k_ell;
                float val = block->AS_ell[ja_idx];
                if (val != 0.0f) { // process only actual non-zeros (ELLPACK pads with 0s)
                    int col = block->JA_ell[ja_idx];
                    // boundary check for col might be needed if JA padding is not robust
                    if (col >= 0 && col < A_hll->total_cols) { // ensure col is valid
                        sum_row += val * x[col];
                    }
                }
            }
            y[current_global_row] = sum_row;
            current_global_row++;
        }
    }
}