#ifndef HLL_UTILS_H
#define HLL_UTILS_H

#include "mm_reader.h"

// --- ELLPACK Block Structure ---
typedef struct {
    int num_rows_in_block;    // actual number of rows in this block (<= hack_size)
    int max_nz_per_row;       // maximum non-zeros per row *within this specific block*
    int *JA_ell;              // column indices: 2D array flattened (num_rows_in_block * max_nz_per_row)
    float *AS_ell;            // values: 2D array flattened (num_rows_in_block * max_nz_per_row)
} ELLPACKBlock;

// --- HLL Matrix Structure ---
// represents the entire matrix partitioned into ELLPACK blocks.
typedef struct {
    int total_rows;           // total rows in the original matrix (M)
    int total_cols;           // total columns in the original matrix (N)
    long long total_nnz;      // total non-zeros in the original matrix (for reference, SpMV uses block data)
    int hack_size;            // the HLL parameter for partitioning rows
    int num_blocks;           // total number of ELLPACK blocks
    ELLPACKBlock *blocks;     // array of ELLPACK blocks
} HLLMatrix;

#ifdef __cplusplus // this header is included in a C++ file (main.cpp)
extern "C" {
#endif

HLLMatrix csr_to_hll(const CSRMatrix *csr_matrix, int hack_size);

void free_hll_matrix(HLLMatrix *hll_matrix);

#ifdef __cplusplus
}
#endif

#endif