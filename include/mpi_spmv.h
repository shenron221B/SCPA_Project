
#ifndef SCPA_PROJECT_MPI_SPMV_H
#define SCPA_PROJECT_MPI_SPMV_H

#include "mm_reader.h" // Per CSRMatrix
#include <mpi.h>

// Struttura per contenere i dati locali di ogni processo MPI
typedef struct {
    int local_nrows;    // Numero di righe gestite da questo processo
    int global_ncols;   // Numero totale di colonne (uguale per tutti)
    int local_nnz;      // Numero di non-zeri nelle righe locali
    int *IRP_local;     // Puntatori alle righe locali (ribasato, IRP_local[0]=0)
    int *JA_local;      // Indici di colonna locali
    float *AS_local;    // Valori dei coefficienti locali
} CSRMatrix_local;

void distribute_matrix_mpi_csr(const CSRMatrix *A_global, CSRMatrix_local *A_local, int rank, int size, MPI_Comm comm);
void free_csr_local(CSRMatrix_local *A_local);
void mpi_spmv_csr(const CSRMatrix_local *A_local, const float *x_global_or_local, float *y_local_buff, float *y_global, int rank, int size, MPI_Comm comm);

#endif //SCPA_PROJECT_MPI_SPMV_H

