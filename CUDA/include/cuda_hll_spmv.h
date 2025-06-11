// File: CUDA/include/cuda_hll_spmv.h
#ifndef CUDA_HLL_SPMV_H
#define CUDA_HLL_SPMV_H

#include "../../include/hll_utils.h" // for HLLMatrix host structure

// device representation of HLL might be slightly different or involve
// copying block data arrays contiguously. for a simple start,
// we can have an array of device ELLPACKBlock pointers, or flatten.
// for now, assume HLLMatrix structure is mostly mirrored, with device pointers.

// device representation of an ELLPACK block
typedef struct {
    int num_rows_in_block;
    int max_nz_per_row;
    int *d_JA_ell;
    float *d_AS_ell;
} ELLPACKBlock_device;

// device representation of HLL matrix
typedef struct {
    int total_rows;
    int total_cols;
    int hack_size;
    int num_blocks;
    ELLPACKBlock_device *d_blocks_meta; // array of metadata for blocks (num_rows, max_nz)
    // and pointers d_JA_ell, d_AS_ell would be part of this
    // or a more complex structure:
    // or a single large allocation for all JA_ell and AS_ell with offsets
    int **d_JA_blocks_ptrs; // array of device pointers, each to a block's JA_ell
    float **d_AS_blocks_ptrs; // array of device pointers, each to a block's AS_ell
    // for simplicity, let's try to pass the host HLLMatrix and copy its components.
    // the wrapper will manage copying individual block arrays.
} HLLMatrix_device_handle; // this is more of a conceptual handle

int cuda_spmv_hll_wrapper(const HLLMatrix *h_A_hll, const float *h_x, float *h_y, int threads_per_block_dim);

#endif // CUDA_HLL_SPMV_H