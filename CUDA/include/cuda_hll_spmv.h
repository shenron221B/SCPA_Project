#ifndef CUDA_HLL_SPMV_H
#define CUDA_HLL_SPMV_H

#include "../../include/hll_utils.h"

// device representation of an ELLPACK block (struct of offset)
typedef struct {
    int num_rows_in_block;
    int max_nz_per_row;
    size_t ja_start_offset;
    size_t as_start_offset;
} ELLPACKBlock_device;

// device representation of HLL matrix
typedef struct {
    int total_rows;
    int total_cols;
    int hack_size;
    int num_blocks;
    ELLPACKBlock_device *d_blocks_meta; // array of metadata for blocks (num_rows, max_nz)
    int **d_JA_blocks_ptrs; // array of device pointers, each to a block's JA_ell
    float **d_AS_blocks_ptrs; // array of device pointers, each to a block's AS_ell
} HLLMatrix_device_handle;

int cuda_spmv_hll_wrapper(const HLLMatrix *h_A_hll, const float *h_x, float *h_y, int threads_per_block_dim, double *kernel_time_s);

#endif