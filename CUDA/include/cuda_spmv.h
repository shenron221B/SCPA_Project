#ifndef SCPA_PROJECT_CUDA_SPMV_H
#define SCPA_PROJECT_CUDA_SPMV_H

#include "../../include/mm_reader.h" // Percorso relativo per mm_reader.h

// Struttura per contenere i puntatori ai dati della matrice sulla GPU
typedef struct {
    int nrows;
    int ncols;
    long long nnz; // Manteniamo long long per coerenza con mm_reader
    int *d_IRP;     // Puntatore a IRP su device (GPU)
    int *d_JA;      // Puntatore a JA su device
    float *d_AS;    // Puntatore a AS su device
} CSRMatrix_device;

/**
 * @brief function to perform sparse matrix-vector multiplication (SpMV) y = Ax
 *        for a CSR matrix using CUDA.
 *
 * this function will:
 * 1. allocate memory on the GPU for the matrix (A) and vectors (x, y).
 * 2. copy matrix A and vector x from host (CPU) memory to device (GPU) memory.
 * 3. launch the CUDA kernel to perform SpMV on the GPU.
 * 4. copy the resulting vector y from device memory back to host memory.
 * 5. free memory allocated on the GPU.
 *
 * @param h_A pointer to the host CSRMatrix structure.
 * @param h_x pointer to the host input vector x.
 * @param h_y pointer to the host output vector y (will be filled with result).
 * @param block_size the size of the thread block to use for the CUDA kernel (e.g., 256, 512).
 * @return 0 on success, non-zero on CUDA error.
 */
int cuda_spmv_csr_wrapper(const CSRMatrix *h_A, const float *h_x, float *h_y, int block_size);

#endif //SCPA_PROJECT_CUDA_SPMV_H