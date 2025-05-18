#include <stdio.h>
#include <stdlib.h>
#include "../include/csr_utils.h"

/**
 * @brief Libera la memoria allocata dinamicamente per una matrice in formato CSR.
 * È fondamentale chiamare questa funzione quando una matrice CSR non è più necessaria
 * per evitare memory leak (perdita di memoria).
 *
 * @param A Puntatore alla struttura CSRMatrix la cui memoria deve essere liberata.
 *          I puntatori IRP, JA, e AS all'interno della struttura verranno liberati,
 *          e idealmente dovrebbero poi essere impostati a NULL dal chiamante se la
 *          struttura CSRMatrix stessa non viene distrutta immediatamente.
 */

void free_csr(CSRMatrix *A) {
    // check null pointer
    if (A == NULL) {
        return;
    }

    // free memory for the array of row pointer (IRP)
    if (A->IRP != NULL) {
        free(A->IRP);
        A->IRP = NULL;
    }

    // free memory for the array of column index (JA)
    if (A->JA != NULL) {
        free(A->JA);
        A->JA = NULL;
    }

    // free memory for the array of values (AS)
    if (A->AS != NULL) {
        free(A->AS);
        A->AS = NULL;
    }

    // Nota: La struttura CSRMatrix stessa (A) non viene liberata qui se è stata allocata
    // dinamicamente (es. CSRMatrix *mat = malloc(sizeof(CSRMatrix))).
    // Questa funzione libera solo gli array INTERNI alla struttura.
    // Se la struttura A è una variabile locale (sullo stack), non c'è bisogno di liberarla.
    // Se A è un puntatore a una struttura allocata con malloc, il chiamante è responsabile
    // di fare free(A) DOPO aver chiamato free_csr(A).
    // Nel main.c attuale, `matrix_global` è una variabile locale, quindi non serve `free(&matrix_global)`.
}

// print element of a vector on standard output
void print_vector(const float *v, int size) {
    // check for null pointer or invalid dimension
    if (v == NULL || size <= 0) {
        printf("null vector or invalid dimension\n");
        return;
    }

    for (int i = 0; i < size; i++) {
        printf("%.2f ", v[i]);
    }
    printf("\n");
}