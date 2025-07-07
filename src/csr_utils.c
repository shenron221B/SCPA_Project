#include <stdio.h>
#include <stdlib.h>
#include "../include/csr_utils.h"

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