#ifndef MM_READER_H
#define MM_READER_H

typedef struct {
    int nrows;
    int ncols;
    long long nnz;
    int *IRP;
    int *JA;
    float *AS;
} CSRMatrix;

#ifdef __cplusplus // this header is included in a C++ file (main.cpp)
extern "C" {
#endif

CSRMatrix read_matrix_market_to_csr(const char *filename);

#ifdef __cplusplus
}
#endif

#endif
