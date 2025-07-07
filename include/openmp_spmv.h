#ifndef SCPA_PROJECT_OPENMP_SPMV_H
#define SCPA_PROJECT_OPENMP_SPMV_H

#include "mm_reader.h" // for CSRMatrix
#include "hll_utils.h" // for HLLMatrix

#ifdef __cplusplus // this header is included in a C++ file (main.cpp)
extern "C" {
#endif

// calculate limits of rows for CSR based on NNZ
int* prepare_csr_bounds_by_nnz(const CSRMatrix *A, int num_threads);

// calculate limits of block for HLL
int* prepare_hll_bounds(const HLLMatrix *A_hll, int num_threads);

// SpMV for CSR format
void openmp_spmv_csr(const CSRMatrix *A, const float *x, float *y, int num_threads, const int* row_bounds);

// SpMV for HLL format
void openmp_spmv_hll(const HLLMatrix *A_hll, const float *x, float *y, int num_threads, const int* block_bounds);

#ifdef __cplusplus
}
#endif

#endif