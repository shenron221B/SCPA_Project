#ifndef SCPA_PROJECT_OPENMP_SPMV_H
#define SCPA_PROJECT_OPENMP_SPMV_H

#include "mm_reader.h" // Per CSRMatrix

/**
 * @brief Calcola il prodotto matrice-vettore y = Ax per una matrice sparsa in formato CSR
 *        utilizzando OpenMP per la parallelizzazione.
 *
 * @param A Puntatore alla matrice CSR (CSRMatrix). La matrice non viene modificata.
 * @param x Puntatore al vettore di input x. Non viene modificato.
 * @param y Puntatore al vettore di output y. Verrà riempito con il risultato.
 * @param num_threads Numero di thread OpenMP da utilizzare per il calcolo.
 */
void openmp_spmv_csr(const CSRMatrix *A, const float *x, float *y, int num_threads);

#endif //SCPA_PROJECT_OPENMP_SPMV_H