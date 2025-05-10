#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "../include/mm_reader.h"

int compare_row_col(const void *a, const void *b) {
    int *ia = (int *)a;
    int *ib = (int *)b;
    if (ia[0] != ib[0]) return ia[0] - ib[0];
    return ia[1] - ib[1];
}

CSRMatrix read_matrix_market_to_csr(const char *filename) {
    FILE *f = fopen(filename, "r");
    if (!f) {
        perror("Errore apertura file");
        exit(1);
    }

    // Ignora commenti e righe vuote
    char line[1024];
    while (1) {
        if (!fgets(line, sizeof(line), f)) {
            fprintf(stderr, "Errore: fine file raggiunto prima delle dimensioni\n");
            exit(EXIT_FAILURE);
        }

        // Salta righe vuote o che iniziano con '%'
        char *trim = line;
        while (*trim == ' ' || *trim == '\t') trim++;  // Rimuove spazi iniziali
        if (*trim == '%' || *trim == '\n' || *trim == '\0') continue;
        break;
    }

    int m, n, nnz;

    int matched = sscanf(line, "%d %d %d", &m, &n, &nnz);
    if (matched != 3) {
        fprintf(stderr, "Ignorato file non valido (dimensioni mancanti o incomplete): %s\n", filename);
        // Restituisce una matrice vuota
        CSRMatrix empty = {0, 0, 0, NULL, NULL, NULL};
        return empty;
    }

    if (m <= 0 || n <= 0 || nnz < 0) {
        fprintf(stderr, "Dimensioni non valide: m=%d, n=%d, nnz=%d\n", m, n, nnz);
        exit(EXIT_FAILURE);
    }

    int **triplets = malloc(nnz * sizeof(int *));
    float *values = malloc(nnz * sizeof(float));

    // Ordina per riga
    int *JA = malloc(nnz * sizeof(int));
    float *AS = malloc(nnz * sizeof(float));
    int *IRP = calloc(m + 1, sizeof(int));

    int *counter = calloc(m, sizeof(int));

    if (!triplets || !values || !JA || !AS || !IRP || !counter) {
        fprintf(stderr, "Errore allocazione memoria per file %s\n", filename);
        exit(EXIT_FAILURE);
    }

    printf("Inizio lettura %d triplet...\n", nnz);

    for (int i = 0; i < nnz; i++) {
        triplets[i] = malloc(2 * sizeof(int));

        int ret = fscanf(f, "%d %d %f", &triplets[i][0], &triplets[i][1], &values[i]);
        if (ret != 3) {
            fprintf(stderr, "Errore lettura triplet alla riga %d nel file %s\n", i+1, filename);
            exit(EXIT_FAILURE);
        }

        triplets[i][0]--; // MatrixMarket è 1-based
        triplets[i][1]--;

        if (triplets[i][0] < 0 || triplets[i][0] >= m || triplets[i][1] < 0 || triplets[i][1] >= n) {
            fprintf(stderr, "Indice fuori dai limiti alla riga %d: row=%d, col=%d\n", i+1, triplets[i][0]+1, triplets[i][1]+1);
            exit(EXIT_FAILURE);
        }

    }
    fclose(f);

    // Conta elementi per riga
    for (int i = 0; i < nnz; i++) IRP[triplets[i][0] + 1]++;

    for (int i = 0; i < m; i++) IRP[i + 1] += IRP[i]; // scan

    for (int i = 0; i < nnz; i++) {
        int row = triplets[i][0];
        int dest = IRP[row] + counter[row];
        JA[dest] = triplets[i][1];
        AS[dest] = values[i];
        counter[row]++;
    }

    for (int i = 0; i < nnz; i++) free(triplets[i]);
    free(triplets);
    free(values);
    free(counter);

    CSRMatrix A = {m, n, nnz, IRP, JA, AS};
    return A;
}
