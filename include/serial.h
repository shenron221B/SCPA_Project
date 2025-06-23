#ifndef SERIAL_H
#define SERIAL_H

#include "mm_reader.h" // For CSRMatrix
#include "hll_utils.h" // For HLLMatrix // NUOVO

#ifdef __cplusplus // this header is included in a C++ file (main.cpp)
extern "C" {
#endif

// SpMV for CSR format
void serial_spmv_csr(const CSRMatrix *A, const float *x, float *y);

#ifdef __cplusplus
}
#endif

#endif