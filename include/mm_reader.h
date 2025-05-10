#ifndef MM_READER_H
#define MM_READER_H

typedef struct {
    int nrows;
    int ncols;
    int nnz;
    int *IRP;
    int *JA;
    float *AS;
} CSRMatrix;

CSRMatrix read_matrix_market_to_csr(const char *filename);

#endif
