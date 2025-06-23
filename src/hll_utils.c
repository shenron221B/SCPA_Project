#include <stdio.h>
#include <stdlib.h>
#include <string.h> // for memset, memcpy
#include <math.h>   // for ceil
#include "../include/hll_utils.h"

// helper function to find max non-zeros in a range of rows for a CSR matrix
static int find_max_nz_in_row_range(const CSRMatrix *csr_matrix, int start_row, int end_row) {
    int max_nz = 0;
    for (int i = start_row; i < end_row && i < csr_matrix->nrows; ++i) {
        int count = csr_matrix->IRP[i + 1] - csr_matrix->IRP[i];
        if (count > max_nz) {
            max_nz = count;
        }
    }
    return max_nz;
}

HLLMatrix csr_to_hll(const CSRMatrix *csr_matrix, int hack_size) {
    HLLMatrix hll_matrix;
    if (!csr_matrix || hack_size <= 0) {
        fprintf(stderr, "error [csr_to_hll]: invalid input csr_matrix or hack_size.\n");
        hll_matrix.num_blocks = 0; // indicate an invalid HLL matrix
        return hll_matrix;
    }

    hll_matrix.total_rows = csr_matrix->nrows;
    hll_matrix.total_cols = csr_matrix->ncols;
    hll_matrix.total_nnz = csr_matrix->nnz; // store original nnz for reference
    hll_matrix.hack_size = hack_size;
    hll_matrix.num_blocks = (csr_matrix->nrows + hack_size - 1) / hack_size; // ceiling division

    if (hll_matrix.num_blocks == 0 && csr_matrix->nrows > 0) { // e.g. if nrows < hack_size but nrows > 0
        hll_matrix.num_blocks = 1;
    } else if (csr_matrix->nrows == 0) {
        hll_matrix.num_blocks = 0;
    }


    if (hll_matrix.num_blocks > 0) {
        hll_matrix.blocks = (ELLPACKBlock *)malloc(hll_matrix.num_blocks * sizeof(ELLPACKBlock));
        if (!hll_matrix.blocks) {
            perror("error [csr_to_hll]: failed to allocate memory for HLL blocks");
            hll_matrix.num_blocks = 0;
            return hll_matrix;
        }
    } else {
        hll_matrix.blocks = NULL;
        return hll_matrix; // if no rows, return empty HLL
    }


    for (int block_idx = 0; block_idx < hll_matrix.num_blocks; ++block_idx) {
        ELLPACKBlock *current_block = &hll_matrix.blocks[block_idx];
        int first_row_in_block = block_idx * hack_size;
        int last_row_exclusive = first_row_in_block + hack_size;
        if (last_row_exclusive > csr_matrix->nrows) {
            last_row_exclusive = csr_matrix->nrows;
        }
        current_block->num_rows_in_block = last_row_exclusive - first_row_in_block;

        if (current_block->num_rows_in_block <= 0) { // should not happen if num_blocks is calculated correctly
            current_block->max_nz_per_row = 0;
            current_block->JA_ell = NULL;
            current_block->AS_ell = NULL;
            continue;
        }

        current_block->max_nz_per_row = find_max_nz_in_row_range(csr_matrix, first_row_in_block, last_row_exclusive);

        // if a block has rows but all rows are empty, max_nz_per_row could be 0
        if (current_block->max_nz_per_row == 0) {
            // still allocate to represent empty rows, or handle differently
            // for simplicity, if max_nz_per_row is 0, JA_ell and AS_ell can be NULL
            // or small dummy arrays, but SpMV logic must handle this.
            // a common approach is to ensure max_nz_per_row is at least 1 if num_rows_in_block > 0
            // to avoid zero-size allocations if all rows in block are empty.
            // however, if all rows truly are empty, 0 is correct.
            // let's assume SpMV will handle max_nz_per_row = 0 correctly (inner loop won't run)
            current_block->JA_ell = NULL; // Or calloc(current_block->num_rows_in_block * 1, sizeof(int)) if min 1
            current_block->AS_ell = NULL; // Or calloc(current_block->num_rows_in_block * 1, sizeof(float))
        } else {
            current_block->JA_ell = (int *)malloc(current_block->num_rows_in_block * current_block->max_nz_per_row * sizeof(int));
            current_block->AS_ell = (float *)malloc(current_block->num_rows_in_block * current_block->max_nz_per_row * sizeof(float));

            if (!current_block->JA_ell || !current_block->AS_ell) {
                perror("error [csr_to_hll]: failed to allocate memory for ELLPACK block arrays");
                // cleanup previously allocated blocks
                for (int j = 0; j < block_idx; ++j) {
                    if (hll_matrix.blocks[j].JA_ell) free(hll_matrix.blocks[j].JA_ell);
                    if (hll_matrix.blocks[j].AS_ell) free(hll_matrix.blocks[j].AS_ell);
                }
                free(hll_matrix.blocks);
                hll_matrix.num_blocks = 0;
                return hll_matrix;
            }

            // initialize with a padding value (e.g., 0 for values, -1 or last valid col for indices if needed)
            // memset(current_block->AS_ell, 0, current_block->num_rows_in_block * current_block->max_nz_per_row * sizeof(float));
            // for JA_ell, padding needs careful consideration. often, if value is 0, JA index doesn't matter.
            // for simplicity, let's fill with 0.0f and a sentinel for JA_ell if needed.
            // the problem description implies padding with appropriate coefficients (0 for AS)
            // and for JA "the corresponding index... is fixed to the last valid index encountered along the row"
            // this is tricky if a row is shorter than max_nz_per_row from the start.
            // a simpler ELLPACK pads JA with a valid column index (e.g., 0) if AS is 0.

            for (int r_block = 0; r_block < current_block->num_rows_in_block; ++r_block) {
                int global_row_idx = first_row_in_block + r_block;
                int nz_in_this_row_count = 0;
                int last_valid_col_idx = 0; // default if row is empty

                // fill with actual non-zeros from CSR
                for (int csr_ptr = csr_matrix->IRP[global_row_idx]; csr_ptr < csr_matrix->IRP[global_row_idx + 1]; ++csr_ptr) {
                    if (nz_in_this_row_count < current_block->max_nz_per_row) {
                        current_block->JA_ell[r_block * current_block->max_nz_per_row + nz_in_this_row_count] = csr_matrix->JA[csr_ptr];
                        current_block->AS_ell[r_block * current_block->max_nz_per_row + nz_in_this_row_count] = csr_matrix->AS[csr_ptr];
                        last_valid_col_idx = csr_matrix->JA[csr_ptr];
                        nz_in_this_row_count++;
                    } else {
                        // this should not happen if max_nz_per_row was calculated correctly for the block
                        fprintf(stderr, "warning: row %d has more non-zeros than block's max_nz_per_row. data truncated.\n", global_row_idx);
                        break;
                    }
                }

                // pad the rest of the row in ELLPACK block
                for (int j_ell = nz_in_this_row_count; j_ell < current_block->max_nz_per_row; ++j_ell) {
                    current_block->JA_ell[r_block * current_block->max_nz_per_row + j_ell] = last_valid_col_idx; // or 0, or a specific sentinel
                    current_block->AS_ell[r_block * current_block->max_nz_per_row + j_ell] = 0.0f;
                }
            }
        }
    }
    return hll_matrix;
}

void free_hll_matrix(HLLMatrix *hll_matrix) {
    if (hll_matrix && hll_matrix->blocks) {
        for (int i = 0; i < hll_matrix->num_blocks; ++i) {
            if (hll_matrix->blocks[i].JA_ell) {
                free(hll_matrix->blocks[i].JA_ell);
                hll_matrix->blocks[i].JA_ell = NULL;
            }
            if (hll_matrix->blocks[i].AS_ell) {
                free(hll_matrix->blocks[i].AS_ell);
                hll_matrix->blocks[i].AS_ell = NULL;
            }
        }
        free(hll_matrix->blocks);
        hll_matrix->blocks = NULL;
    }
    hll_matrix->num_blocks = 0;
    hll_matrix->total_rows = 0;
}