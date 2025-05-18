#include "../include/serial.h"

/**
 * @brief Calcola il prodotto matrice-vettore y = Ax in modo seriale
 *        per una matrice sparsa A memorizzata in formato CSR
 *
 * Per ogni riga i della matrice A:
 *   y[i] = somma di (A[i,j] * x[j]) per tutti i j tali che A[i,j] è non-zero
 *
 * @param A Puntatore (const) alla matrice CSR
 * @param x Puntatore (const) al vettore di input x
 * @param y Puntatore al vettore di output y, che verrà riempito con il risultato
 */

void serial_spmv(const CSRMatrix *A, const float *x, float *y) {

    // todo aggiungere controlli per puntatori nulli e per dimensioni di matrici e vettori

    // for each row of matrix A (from 0 to A -> nrows - 1)
    for (int i = 0; i < A->nrows; i++) {
        float sum = 0.0f; // accumulator for element y[i]

        // for each non - zero elements of row 'i'
        // A->IRP[i]: index (in the array JA and AS) of the first non-zero element of row 'i'
        // A->IRP[i+1]: index of the first non-zero element of the next row (i+1)
        // A->IRP[i+1]-1: index of the last non-zero element of row 'i'
        for (int k = A->IRP[i]; k < A->IRP[i + 1]; k++) {
            // A->JA[k] contain the index of column 'j' of the current non - zero element
            // A->AS[k] contain the value A[i,j] of the current non - zero element
            // x[A->JA[k]] is the element x[j] of the vector x
            sum += A->AS[k] * x[A->JA[k]]; // execute moltiplication and update sum
        }
        y[i] = sum;
    }
}