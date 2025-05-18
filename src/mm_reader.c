#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#include "../include/mm_reader.h"

// La funzione compare_row_col non serve se convertiamo direttamente in CSR
// senza prima memorizzare tutti i triplet e ordinarli.
// Se la tua strategia di conversione a CSR richiede l'ordinamento dei triplet,
// allora mantienila. Per ora, la commento assumendo una conversione diretta.
/*
int compare_row_col(const void *a, const void *b) {
    int *ia = (int *)a;
    int *ib = (int *)b;
    if (ia[0] != ib[0]) return ia[0] - ib[0];
    return ia[1] - ib[1];
}
*/

/**
 * @brief Legge una matrice sparsa da un file in formato MatrixMarket e la converte in formato CSR.
 *
 * Il formato MatrixMarket per matrici "coordinate" ha:
 * 1. Una riga "banner" che inizia con "%%MatrixMarket matrix <format> <type> <structure>".
 *    - <format>: deve essere "coordinate".
 *    - <type>: può essere "real", "integer", "pattern", "complex". Questo codice supporta "real", "integer", "pattern".
 *    - <structure>: può essere "general", "symmetric", "skew-symmetric", "hermitian".
 *                   La gestione dell'espansione simmetrica non è ancora implementata qui.
 * 2. Righe di commento opzionali che iniziano con '%'.
 * 3. Una riga con tre interi: M (numero di righe), N (numero di colonne), NNZ (numero di non-zeri nel file).
 * 4. NNZ righe di dati:
 *    - Per tipo "real" o "integer": <riga> <colonna> <valore>
 *    - Per tipo "pattern": <riga> <colonna> (il valore è implicitamente 1.0)
 * Gli indici di riga e colonna nel file sono 1-based e vengono convertiti a 0-based.
 *
 * @param filename Il nome del file .mtx da leggere.
 * @return Una struttura CSRMatrix contenente la matrice. Se si verifica un errore o
 *         la matrice non è valida/supportata, restituisce una CSRMatrix con nrows=0, nnz=0
 *         e puntatori NULL (o IRP allocato e azzerato per matrici esplicitamente vuote).
 */

CSRMatrix read_matrix_market_to_csr(const char *filename) {
    FILE *f = fopen(filename, "r"); // open file in read mode
    if (!f) {
        fprintf(stderr, "error to open file: %s ", filename);
        perror("details");
        // return empty matrix to continue with others file
        CSRMatrix empty = {0, 0, 0, NULL, NULL, NULL};
        return empty;
    }

    char line[1024]; // buffer to read row of text from file
    char banner[1024]; // buffer for row banner "%%MatrixMarket..."
    char matrix_format[64], matrix_type[64], matrix_structure[64]; // strings for banner's fields
    int m, n, nnz_from_file; // size of matrix and number of non-zero read from file

    // read banner's row %%MatrixMarket
    if (!fgets(banner, sizeof(banner), f)) {
        fprintf(stderr, "error: file %s is empty or the banner row is impossible to read\n", filename);
        fclose(f);
        CSRMatrix empty = {0, 0, 0, NULL, NULL, NULL}; return empty;
    }

    // try to extract format, type and structure of the banner
    if (sscanf(banner, "%%%%MatrixMarket matrix %s %s %s",
               matrix_format, matrix_type, matrix_structure) != 3) {
        fprintf(stderr, "error: header MatrixMarket is not valid in the file %s: %s\n", filename, banner);
        fclose(f);
        CSRMatrix empty = {0, 0, 0, NULL, NULL, NULL}; return empty;
    }

    // check if format is 'coordinate'
    if (strcmp(matrix_format, "coordinate") != 0) {
        fprintf(stderr, "error: only coordinate format is supported, found '%s' in %s\n", matrix_format, filename);
        fclose(f);
        CSRMatrix empty = {0, 0, 0, NULL, NULL, NULL}; return empty;
    }

    // determine the type of data and if the matrix is 'pattern'
    int is_pattern = (strcmp(matrix_type, "pattern") == 0);
    int is_real = (strcmp(matrix_type, "real") == 0);
    int is_integer = (strcmp(matrix_type, "integer") == 0);
    // int is_symmetric = (strcmp(matrix_structure, "symmetric") == 0); // todo flag for symmetric matrix

    // check if data type is supported
    if (!is_pattern && !is_real && !is_integer) {
        fprintf(stderr, "error: only 'real', 'integer' or 'pattern' types are supported, found '%s' in %s\n", matrix_type, filename);
        fclose(f);
        CSRMatrix empty = {0, 0, 0, NULL, NULL, NULL}; return empty;
    }

    // skip comment and search row of dimension
    while (fgets(line, sizeof(line), f)) {
        char *trim = line;
        while (*trim != '\0' && isspace((unsigned char)*trim)) trim++; // skip white space
        if (*trim == '%' || *trim == '\0' || *trim == '\n') continue; // skip comment and empty row
        break;
    }
    // loop is end before find the row of dimension (EOF or I/O error)
    if (feof(f) || ferror(f)) { // Se siamo arrivati alla fine del file senza trovare le dimensioni
         fprintf(stderr, "error: end of file or I/O error before dimension in %s\n", filename);
         fclose(f);
         CSRMatrix empty = {0, 0, 0, NULL, NULL, NULL}; return empty;
    }

    // read dimension: M, N, NNZ
    int matched = sscanf(line, "%d %d %d", &m, &n, &nnz_from_file);
    if (matched != 3) {
        fprintf(stderr, "ignored invalid file (dimensions are null or incomplete after the header): %s. Row: '%s'\n", filename, line);
        fclose(f);
        CSRMatrix empty = {0, 0, 0, NULL, NULL, NULL}; return empty;
    }

    // check the read dimensions
    if (m <= 0 || n <= 0 || nnz_from_file < 0) {
        fprintf(stderr, "invalid dimensions in %s: m=%d, n=%d, nnz=%d\n", filename, m, n, nnz_from_file);
        fclose(f);
        CSRMatrix empty = {0, 0, 0, NULL, NULL, NULL}; return empty;
    }
    // if matrix is empty (declared nnz = 0)
    if (nnz_from_file == 0) {
        printf("matrix %s have 0 non-zeros declared\n", filename);
        fclose(f);
        // return empty valid matrix (IRP, JA, AS are empty if not allocated)
        int *IRP_empty = calloc(m + 1, sizeof(int)); // IRP should be [0,0,0,...]
        CSRMatrix A = {m, n, 0, IRP_empty, NULL, NULL};
        return A;
    }

    // todo gestione della matrice simmetrica
    // Per ora, non gestiamo l'espansione simmetrica, quindi nnz_effettivo = nnz_from_file
    // Quando implementerai la simmetria, nnz_effettivo potrebbe diventare ~2*nnz_from_file
    long long effective_nnz = nnz_from_file;

    // allocation for CSR array (IRP, JA, AS)
    // NOTA: Se dovessimo gestire l'espansione simmetrica, questa allocazione basata su nnz_from_file
    // potrebbe non essere sufficiente. Una strategia a due passate o riallocazione sarebbe necessaria.
    // Per ora, assumiamo che non stiamo espandendo matrici simmetriche in questa fase.
    int *JA = malloc(effective_nnz * sizeof(int)); // array of column index
    float *AS = malloc(effective_nnz * sizeof(float)); // array of non-zero value
    int *IRP = calloc(m + 1, sizeof(int)); // array of row pointer (m + 1 element) -> calloc initialized to 0

    // check allocation
    if (!JA || !AS || !IRP) {
        fprintf(stderr, "error in allocation memory for CSR for file %s\n", filename);
        if (JA) free(JA);
        if (AS) free(AS);
        if (IRP) free(IRP);
        fclose(f);
        CSRMatrix empty = {0, 0, 0, NULL, NULL, NULL}; return empty;
    }

    printf("starting to read %d triplet/couple for %s (M=%d, N=%d). Type: %s\n",
           nnz_from_file, filename, m, n, matrix_type);

    // Struttura dati temporanea per leggere i dati prima di popolare IRP
    // Questo è necessario perché per costruire IRP correttamente con il metodo a due passate
    // (prima conta, poi popola), dobbiamo sapere prima tutte le righe.
    // Se il file non è ordinato per righe, questa strategia è più robusta.
    // read data (triplet or couples) in a buffer
    typedef struct { int r, c; float val; } Triplet;
    Triplet* temp_triplets = malloc(nnz_from_file * sizeof(Triplet));
    if (!temp_triplets) { // check allocation
        fprintf(stderr, "error in allocation memory for temporary triplet for file %s\n", filename);
        free(JA); free(AS); free(IRP);
        fclose(f);
        CSRMatrix empty = {0, 0, 0, NULL, NULL, NULL}; return empty;
    }

    long long actual_elements_read = 0; // counter of effectively read elements
    // loop to read nnz_from_file row of data
    for (int k = 0; k < nnz_from_file; k++) {
        int r_in, c_in;
        float v_in = 1.0f; // default value for 'pattern' matrix

        if (is_pattern) { // if matrix is 'pattern'
            // read only row and column
            if (fscanf(f, "%d %d", &r_in, &c_in) != 2) {
                fprintf(stderr, "read error couple (pattern) at data row %d in the file %s\n", k + 1, filename);
                goto cleanup_error;
            }
        } else if (is_real) { // if matrix is 'real'
            // read row, column and float value
            if (fscanf(f, "%d %d %f", &r_in, &c_in, &v_in) != 3) {
                fprintf(stderr, "read error triplet (real) at data row %d in the file %s\n", k + 1, filename);
                goto cleanup_error;
            }
        } else if (is_integer) { // if matrix is integer
            // read row, column and integer value
            int int_val;
            if (fscanf(f, "%d %d %d", &r_in, &c_in, &int_val) != 3) {
                fprintf(stderr, "read error triplet (integer) at data row %d in the file %s\n", k + 1, filename);
                goto cleanup_error;
            }
            v_in = (float)int_val; // convert integer to float
        }

        // save read data in the buffer and convert index to 0-based (MatrixMarket is 1-based)
        temp_triplets[actual_elements_read].r = r_in - 1;
        temp_triplets[actual_elements_read].c = c_in - 1;
        temp_triplets[actual_elements_read].val = v_in;

        // check if index are valid (< n and < m after conversion)
        if (temp_triplets[actual_elements_read].r < 0 || temp_triplets[actual_elements_read].r >= m ||
            temp_triplets[actual_elements_read].c < 0 || temp_triplets[actual_elements_read].c >= n) {
            fprintf(stderr, "index out of limits at data row %d: row=%d (0-based), col=%d (0-based) in %s\n",
                    k + 1, temp_triplets[actual_elements_read].r, temp_triplets[actual_elements_read].c, filename);
            goto cleanup_error;
        }
        actual_elements_read++; // increment the counter of effectively read element
    }
    fclose(f); // all data are read or there is an error -> close the file

    // check if the effectively read element is equal to nnz declared in the header
    if (actual_elements_read != nnz_from_file) {
        fprintf(stderr, "attention: nnz declared (%d) is different from read elements (%lld) in %s\n",
                nnz_from_file, actual_elements_read, filename);
        // we could exit, but for the moment continue using actual_elements_read
        effective_nnz = actual_elements_read;
        // in the future, if effective_nnz < nnz_from_file, we could do realloc for JA and AS for save memory
    }


    // construct CSR from read triplet
    // count element by row to populate IRP
    for (long long k = 0; k < effective_nnz; k++) {
        IRP[temp_triplets[k].r + 1]++;
    }

    // calculate cumulative scan for IRP to obtain the start index
    // IRP[0] is already 0 for calloc
    for (int i = 0; i < m; i++) {
        IRP[i + 1] += IRP[i];
    }

    // populate JA (column indices) and AS (values)
    // È necessario un array 'counter' o 'current_pos_in_row' per sapere dove inserire
    // il prossimo elemento all'interno dello spazio allocato per ciascuna riga in JA/AS.
    // Questo array deve essere delle dimensioni di m e inizializzato a zero.
    // IRP (prima di questo passo) contiene gli offset di inizio per ogni riga.
    // Dopo questo passo, IRP conterrà gli stessi offset, ma useremo una copia
    // per tenere traccia delle posizioni correnti.
    int *current_pos_in_row = malloc(m * sizeof(int));
    if(!current_pos_in_row) {
        fprintf(stderr, "error allocation memory for current_pos_in_row for file %s\n", filename);
        goto cleanup_error_after_temp_triplets;
    }
    // copy the start offset of IRP in current_pos_in_row
    memcpy(current_pos_in_row, IRP, m * sizeof(int)); // only the first m element is copied, because IRP[m] id the offset after the last row

    for (long long k = 0; k < effective_nnz; k++) {
        int row = temp_triplets[k].r;
        int col = temp_triplets[k].c;
        float val = temp_triplets[k].val;

        // determine destination index in JA/SA using current_pos_in_row for the current row
        int dest_idx = current_pos_in_row[row];
        JA[dest_idx] = col; // save column index
        AS[dest_idx] = val; // save the value
        current_pos_in_row[row]++; // increment pointer for the next entry in this row
    }

    free(temp_triplets);
    free(current_pos_in_row);

    // create and return the final CSRMatrix structure
    CSRMatrix A = {m, n, effective_nnz, IRP, JA, AS};
    return A;

    // cleanup solution in case of error during the read of data
cleanup_error:
    // free all memory allocated and close the file
    if (f) fclose(f); // f could be already closed if the error occur after fclose
    free(JA);
    free(AS);
    free(IRP);
    free(temp_triplets); // temp_triplets could be not allocated if error is previous
    CSRMatrix empty_err = {0, 0, 0, NULL, NULL, NULL}; return empty_err;

cleanup_error_after_temp_triplets: // used if error occur after the allocation of temp_triplets
    if (f) fclose(f);
    free(JA);
    free(AS);
    free(IRP);
    free(temp_triplets);
    // if (current_pos_in_row) free(current_pos_in_row);
    CSRMatrix empty_err2 = {0, 0, 0, NULL, NULL, NULL}; return empty_err2;
}