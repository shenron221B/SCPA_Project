#include <stdio.h>
#include <stdlib.h>
#include "../include/csr_utils.h"

void free_csr(CSRMatrix *A) {
    free(A->IRP);
    free(A->JA);
    free(A->AS);
}

void print_vector(const float *v, int size) {
    for (int i = 0; i < size; i++) {
        printf("%.2f ", v[i]);
    }
    printf("\n");
}
