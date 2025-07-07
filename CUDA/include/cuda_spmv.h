#ifndef SCPA_PROJECT_CUDA_SPMV_H
#define SCPA_PROJECT_CUDA_SPMV_H

#include "../../include/mm_reader.h"

// structure to contain pointers for data matrix on GPU
typedef struct {
    int nrows;
    int ncols;
    long long nnz;
    int *d_IRP;
    int *d_JA;
    float *d_AS;
} CSRMatrix_device;

int cuda_spmv_csr_wrapper(const CSRMatrix *h_A, const float *h_x, float *h_y, int block_size, double *kernel_time_s);

#endif