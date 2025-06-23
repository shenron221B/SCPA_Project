#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#include "../include/mm_reader.h"


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
        CSRMatrix empty = {0, 0, 0LL, NULL, NULL, NULL};
        return empty;
    }

    char line[1024]; // buffer to read row of text from file
    char banner[1024]; // buffer for row banner "%%MatrixMarket..."
    char matrix_format[64], matrix_type[64], matrix_structure[64]; // strings for banner's fields
    int m, n;// size of matrix
    long long nnz_from_file_header; // number of non-zero read from file

    // read banner's row %%MatrixMarket
    if (!fgets(banner, sizeof(banner), f)) {
        fprintf(stderr, "error: file %s is empty or the banner row is impossible to read\n", filename);
        fclose(f);
        CSRMatrix empty = {0, 0, 0LL, NULL, NULL, NULL};
        return empty;
    }

    // try to extract format, type and structure of the banner
    if (sscanf(banner, "%%%%MatrixMarket matrix %s %s %s",
               matrix_format, matrix_type, matrix_structure) != 3) {
        fprintf(stderr, "error: header MatrixMarket is not valid in the file %s: %s\n", filename, banner);
        fclose(f);
        CSRMatrix empty = {0, 0, 0LL, NULL, NULL, NULL};
        return empty;
    }

    // check if format is 'coordinate'
    if (strcmp(matrix_format, "coordinate") != 0) {
        fprintf(stderr, "error: only coordinate format is supported, found '%s' in %s\n", matrix_format, filename);
        fclose(f);
        CSRMatrix empty = {0, 0, 0LL, NULL, NULL, NULL};
        return empty;
    }

    // determine the type of data and if the matrix is 'pattern'
    int is_pattern = (strcmp(matrix_type, "pattern") == 0);
    int is_real = (strcmp(matrix_type, "real") == 0);
    int is_integer = (strcmp(matrix_type, "integer") == 0);
    int is_symmetric = (strcmp(matrix_structure, "symmetric") == 0);

    // check if data type is supported
    if (!is_pattern && !is_real && !is_integer) {
        fprintf(stderr, "error: only 'real', 'integer' or 'pattern' types are supported, found '%s' in %s\n", matrix_type, filename);
        fclose(f);
        CSRMatrix empty = {0, 0, 0LL, NULL, NULL, NULL};
        return empty;
    }

    // skip comment and search row of dimension
    while (fgets(line, sizeof(line), f)) {
        char *trim = line;
        while (*trim != '\0' && isspace((unsigned char)*trim)) trim++; // skip white space
        if (*trim == '%' || *trim == '\0' || *trim == '\n') continue; // skip comment and empty row
        break;
    }
    // loop is end before find the row of dimension (EOF or I/O error)
    if (feof(f) || ferror(f)) {
         fprintf(stderr, "error: end of file or I/O error before dimension in %s\n", filename);
         fclose(f);
         CSRMatrix empty = {0, 0, 0LL, NULL, NULL, NULL};
        return empty;
    }

    // read dimension: M, N, NNZ
    if (sscanf(line, "%d %d %lld", &m, &n, &nnz_from_file_header) != 3) {
        fprintf(stderr, "ignored invalid file (dimensions are null or incomplete after the header): %s. Row: '%s'\n", filename, line);
        fclose(f);
        CSRMatrix empty = {0, 0, 0LL, NULL, NULL, NULL};
        return empty;
    }

    // check the read dimensions
    if (m <= 0 || n <= 0 || nnz_from_file_header < 0) {
        fprintf(stderr, "invalid dimensions in %s: m=%d, n=%d, nnz=%lld\n", filename, m, n, nnz_from_file_header);
        fclose(f);
        CSRMatrix empty = {0, 0, 0LL, NULL, NULL, NULL};
        return empty;
    }

    if (m == 0 || n == 0 || nnz_from_file_header == 0) { // if matrix is declared empty or has no rows/cols
        if (nnz_from_file_header > 0 && (m==0 || n==0)) { // nnz > 0 but no rows/cols is an error
            fprintf(stderr, "error: matrix %s has nnz > 0 but m or n is 0.\n", filename);
            fclose(f);
            CSRMatrix empty_err = {0,0,0LL,0,NULL,NULL};
            return empty_err;
        }
        printf("info: matrix %s has 0 non-zeros declared in file or m/n is 0.\n", filename);
        fclose(f);
        int *IRP_empty = NULL;
        if (m > 0) { // only allocate IRP if there are rows
            IRP_empty = (int*)calloc(m + 1, sizeof(int));
            if (!IRP_empty && m > 0) { // check allocation only if m > 0
                fprintf(stderr, "error: calloc failed for IRP_empty in %s\n", filename);
                CSRMatrix empty_err = {0,0,0LL,0,NULL,NULL};
                return empty_err;
            }
        }
        // nnz in CSRMatrix struct should be long long to match effective_nnz
        CSRMatrix A = {m, n, 0LL, IRP_empty, NULL, NULL};
        return A;
    }

    if (is_symmetric && m != n) {
        fprintf(stderr, "error: symmetric matrix must be square (M=%d, N=%d) in %s\n", m, n, filename);
        fclose(f); CSRMatrix empty = {0,0,0LL,NULL,NULL,NULL};
        return empty;
    }

    // temporary storage for triplets read from file
    typedef struct { int r, c; float val; } FileTriplet;
    FileTriplet* file_triplets = (FileTriplet*)malloc(nnz_from_file_header * sizeof(FileTriplet));
    if (!file_triplets) {
        perror("error allocating memory for file_triplets");
        fclose(f); CSRMatrix empty = {0,0,0LL,NULL,NULL,NULL};
        return empty;
    }

    long long elements_actually_in_file = 0;
    for (long long k = 0; k < nnz_from_file_header; k++) {
        int r_in, c_in;
        float v_in = 1.0f; // default for pattern
        int read_count = 0;
        char entry_line[256]; // buffer for reading each data line

        if (!fgets(entry_line, sizeof(entry_line), f)) {
            fprintf(stderr, "error: premature end of file or read error while reading data entry %lld for %s\n", k + 1, filename);
            free(file_triplets);
            CSRMatrix empty = {0,0,0LL,NULL,NULL,NULL}; return empty;
        }

        // remove newline character at the end of entry_line
        entry_line[strcspn(entry_line, "\n")] = 0;
        entry_line[strcspn(entry_line, "\r")] = 0; // for Windows/DOS file

        if (is_pattern) {
            read_count = sscanf(entry_line, "%d %d", &r_in, &c_in);
        } else if (is_real) {
            read_count = sscanf(entry_line, "%d %d %f", &r_in, &c_in, &v_in);
        } else if (is_integer) {
            int int_val;
            read_count = sscanf(entry_line, "%d %d %d", &r_in, &c_in, &int_val);
            if (read_count == 3) v_in = (float)int_val;
        }

        // common error check after fscanf attempts
        if ((is_pattern && read_count != 2) || (!is_pattern && read_count != 3)) {
            fprintf(stderr, "error reading data at entry %lld for file %s\n", k + 1, filename);
            free(file_triplets);
            fclose(f);
            CSRMatrix empty = {0,0,0LL,NULL,NULL,NULL};
            return empty;
        }

        file_triplets[elements_actually_in_file].r = r_in - 1; // 0-based
        file_triplets[elements_actually_in_file].c = c_in - 1; // 0-based
        file_triplets[elements_actually_in_file].val = v_in;

        if (file_triplets[elements_actually_in_file].r < 0 || file_triplets[elements_actually_in_file].r >= m ||
            file_triplets[elements_actually_in_file].c < 0 || file_triplets[elements_actually_in_file].c >= n) {
            fprintf(stderr, "error: index out of bounds at data entry %lld for file %s: r=%d, c=%d (0-based)\n",
                    k + 1, filename, file_triplets[elements_actually_in_file].r, file_triplets[elements_actually_in_file].c);
            free(file_triplets);
            fclose(f);
            CSRMatrix empty = {0,0,0LL,NULL,NULL,NULL}; return empty;
            }
        elements_actually_in_file++;
    }

    if (elements_actually_in_file != nnz_from_file_header) {
        fprintf(stderr, "warning: nnz in header (%lld) differs from elements read (%lld) in %s. using elements read.\n",
                nnz_from_file_header, elements_actually_in_file, filename);
        nnz_from_file_header = elements_actually_in_file; // adjust to what was actually read
    }

    // --- Expand symmetric matrix and determine final_effective_nnz ---
    // estimate max possible nnz after symmetric expansion
    long long estimated_max_expanded_nnz = nnz_from_file_header; // start with nnz from file
    if (is_symmetric) {
        long long off_diagonal_count = 0;
        for (long long k = 0; k < nnz_from_file_header; k++) {
            if (file_triplets[k].r != file_triplets[k].c) {
                off_diagonal_count++;
            }
        }
        estimated_max_expanded_nnz = nnz_from_file_header + off_diagonal_count; // each off-diagonal adds one counterpart
    }

    // use 'Triplet' for the expanded list for clarity, can be same as FileTriplet
    typedef struct { int r, c; float val; } Triplet;
    Triplet* expanded_triplets = (Triplet*)malloc(estimated_max_expanded_nnz * sizeof(Triplet));
    if (!expanded_triplets) {
        perror("error allocating memory for expanded_triplets");
        free(file_triplets);
        fclose(f);
        CSRMatrix empty = {0,0,0LL,NULL,NULL,NULL};
        return empty;
    }

    long long current_expanded_idx = 0;
    for (long long k = 0; k < nnz_from_file_header; k++) {
        int r = file_triplets[k].r;
        int c = file_triplets[k].c;
        float v = file_triplets[k].val;

        // add the original triplet
        expanded_triplets[current_expanded_idx].r = r;
        expanded_triplets[current_expanded_idx].c = c;
        expanded_triplets[current_expanded_idx].val = v;
        current_expanded_idx++;

        // if symmetric and not a diagonal element, add the counterpart
        if (is_symmetric && r != c) {
            // ensure not to exceed allocated space (should be fine with the new estimate)
            if (current_expanded_idx < estimated_max_expanded_nnz) {
                expanded_triplets[current_expanded_idx].r = c; // swapped row and col
                expanded_triplets[current_expanded_idx].c = r;
                expanded_triplets[current_expanded_idx].val = v;
                current_expanded_idx++;
            } else {
                // this case indicates an issue with estimated_max_expanded_nnz, or too many off-diagonals
                fprintf(stderr, "warning: ran out of estimated space during symmetric expansion for %s. this is unexpected.\n", filename);
                // might need to realloc expanded_triplets here if this happens, or improve initial estimate
            }
        }
    }
    free(file_triplets); // no longer needed

    long long final_effective_nnz = current_expanded_idx; // this is the final number of non-zeros for CSR

    // if nnz_effettivo is 0 after expansion (e.g. original nnz_from_file was 0 and it wasn't caught earlier)
    if (final_effective_nnz == 0) { // if, after all, matrix is empty but has dimensions
        printf("info: matrix %s results in 0 effective non-zeros after processing.\n", filename);
        int *IRP_empty = NULL;
        if (m > 0) {
            IRP_empty = (int*)calloc(m + 1, sizeof(int));
            if (!IRP_empty) {
                /* error */
                free(expanded_triplets);
                fclose(f);
                CSRMatrix empty_err = {0,0,0LL,0,NULL,NULL};
                return empty_err;
            }
        }
        free(expanded_triplets); // free this as well
        fclose(f); // close file if not already closed
        CSRMatrix A = {m, n, 0, IRP_empty, NULL, NULL};
        return A;
    }

    printf("info: building CSR for %s (M=%d, N=%d, Effective NNZ=%lld). Type: %s, Structure: %s\n",
           filename, m, n, final_effective_nnz, matrix_type, matrix_structure);

    // allocation for CSR array (IRP, JA, AS)
    int *JA = malloc(final_effective_nnz * sizeof(int)); // array of column index
    float *AS = malloc(final_effective_nnz * sizeof(float)); // array of non-zero value
    int *IRP = calloc(m + 1, sizeof(int)); // array of row pointer (m + 1 element) -> calloc initialized to 0

    // check allocation
    if (!JA || !AS || !IRP) {
        fprintf(stderr, "error allocating memory for final CSR arrays for file %s\n", filename);
        if(JA) free(JA); if(AS) free(AS); if(IRP) free(IRP);
        free(expanded_triplets); // free the temporary expanded list
        fclose(f); // ensure file is closed
        CSRMatrix empty = {0, 0, 0LL, NULL, NULL, NULL};
        return empty;
    }

    // construct CSR from 'expanded_triplets'
    // step 1: count elements per row to populate IRP (as counts initially)
    for (long long k = 0; k < final_effective_nnz; k++) {
        IRP[expanded_triplets[k].r + 1]++; // increment count for the row
    }

    // step 2: calculate cumulative sum for IRP to get row start pointers
    // IRP[0] is already 0 due to calloc
    for (int i = 0; i < m; i++) {
        IRP[i + 1] += IRP[i];
    }

    // step 3: populate JA (column indices) and AS (values)
    // current_pos_in_row will track the current position to insert for each row in JA/AS
    int *current_pos_in_row = (int*)malloc(m * sizeof(int));
    if(!current_pos_in_row) {
        fprintf(stderr, "error allocating memory for current_pos_in_row for file %s\n", filename);
        free(JA); free(AS); free(IRP); free(expanded_triplets);
        fclose(f);
        CSRMatrix empty = {0,0,0LL,NULL,NULL,NULL};
        return empty;
    }
    // initialize current_pos_in_row with the starting offsets from IRP
    memcpy(current_pos_in_row, IRP, m * sizeof(int));

    for (long long k = 0; k < final_effective_nnz; k++) {
        int row = expanded_triplets[k].r;
        int dest_idx = current_pos_in_row[row];
        JA[dest_idx] = expanded_triplets[k].c;
        AS[dest_idx] = expanded_triplets[k].val;
        current_pos_in_row[row]++;
    }

    free(expanded_triplets);    // free the temporary expanded list
    free(current_pos_in_row);   // free the helper array
    fclose(f);

    // create and return the final CSRMatrix structure
    CSRMatrix A = {m, n, final_effective_nnz, IRP, JA, AS};

    return A;
}