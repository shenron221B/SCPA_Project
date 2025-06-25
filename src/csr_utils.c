#include <stdio.h>
#include <stdlib.h>
#include "../include/csr_utils.h"

/**
 * @brief Frees dynamically allocated memory for a CSR-formatted matrix.
 * It is essential to call this function when a CSR matrix is no longer needed
 * to avoid memory leaks.
 *
 * @param A Pointer to the CSRMatrix structure whose memory is to be freed.
 *          The IRP, JA, and AS pointers within the structure will be freed,
 *          and ideally should then be set to NULL by the caller if the
 *          CSRMatrix structure itself is not destroyed immediately.
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