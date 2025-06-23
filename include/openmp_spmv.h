#ifndef SCPA_PROJECT_OPENMP_SPMV_H
#define SCPA_PROJECT_OPENMP_SPMV_H

#include "mm_reader.h" // for CSRMatrix
#include "hll_utils.h" // for HLLMatrix // NUOVO

#ifdef __cplusplus // this header is included in a C++ file (main.cpp)
extern "C" {
#endif

// SpMV for CSR format
void openmp_spmv_csr(const CSRMatrix *A, const float *x, float *y, int num_threads);

// SpMV for HLL format
void openmp_spmv_hll(const HLLMatrix *A_hll, const float *x, float *y, int num_threads);

#ifdef __cplusplus
}
#endif

#endif