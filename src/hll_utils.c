#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
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

        if (current_block->num_rows_in_block <= 0) {
            current_block->max_nz_per_row = 0;
            current_block->JA_ell = NULL;
            current_block->AS_ell = NULL;
            continue;
        }

        current_block->max_nz_per_row = find_max_nz_in_row_range(csr_matrix, first_row_in_block, last_row_exclusive);

        // if a block has rows but all rows are empty, max_nz_per_row could be 0
        if (current_block->max_nz_per_row == 0) {
            current_block->JA_ell = NULL;
            current_block->AS_ell = NULL;
        } else {
            size_t block_size_elements = (size_t)current_block->num_rows_in_block * current_block->max_nz_per_row;
            current_block->JA_ell = (int *)malloc(block_size_elements * sizeof(int));
            current_block->AS_ell = (float *)malloc(block_size_elements * sizeof(float));

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

            int rows_in_block = current_block->num_rows_in_block;
            int max_nz_in_block = current_block->max_nz_per_row;

            for (int r_block = 0; r_block < current_block->num_rows_in_block; ++r_block) {
                int global_row_idx = first_row_in_block + r_block;
                int nz_in_this_row_count = 0;
                int last_valid_col_idx = 0;

                // 1. read data from CSR and put in HLL
                int csr_start = csr_matrix->IRP[global_row_idx];
                int csr_end = csr_matrix->IRP[global_row_idx + 1];

                // fill with actual non-zeros from CSR
                for (int csr_ptr = csr_start; csr_ptr < csr_end; ++csr_ptr) {
                    if (nz_in_this_row_count < max_nz_in_block) {
                        int col_idx = csr_matrix->JA[csr_ptr];
                        float val = csr_matrix->AS[csr_ptr];

                        // index for COLUMN-MAJOR layout
                        int ell_idx = nz_in_this_row_count * rows_in_block + r_block;

                        current_block->JA_ell[ell_idx] = col_idx;
                        current_block->AS_ell[ell_idx] = val;

                        last_valid_col_idx = col_idx;
                        nz_in_this_row_count++;
                    } else {
                        break;
                    }
                }
                // 2. padding for the rest of logical row
                for (int j_ell = nz_in_this_row_count; j_ell < max_nz_in_block; ++j_ell) {
                    int ell_idx = j_ell * rows_in_block + r_block;

                    current_block->JA_ell[ell_idx] = last_valid_col_idx; // padding
                    current_block->AS_ell[ell_idx] = 0.0f;               // padding
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