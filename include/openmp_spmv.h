// File: include/openmp_spmv.h
#ifndef SCPA_PROJECT_OPENMP_SPMV_H
#define SCPA_PROJECT_OPENMP_SPMV_H

#include "mm_reader.h" // for CSRMatrix
#include "hll_utils.h" // for HLLMatrix // NUOVO

// SpMV for CSR format
void openmp_spmv_csr(const CSRMatrix *A, const float *x, float *y, int num_threads);

// NUOVO: SpMV for HLL format
void openmp_spmv_hll(const HLLMatrix *A_hll, const float *x, float *y, int num_threads);

#endif //SCPA_PROJECT_OPENMP_SPMV_H