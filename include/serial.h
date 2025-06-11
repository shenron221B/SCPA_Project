// File: include/serial.h
#ifndef SERIAL_H
#define SERIAL_H

#include "mm_reader.h" // For CSRMatrix
#include "hll_utils.h" // For HLLMatrix // NUOVO

// SpMV for CSR format
void serial_spmv_csr(const CSRMatrix *A, const float *x, float *y);

// NUOVO: SpMV for HLL format
void serial_spmv_hll(const HLLMatrix *A_hll, const float *x, float *y);

#endif