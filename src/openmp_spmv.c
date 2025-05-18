#include "../include/openmp_spmv.h"
#include <omp.h>
#include <stdio.h>

/**
 * @brief Calcola il prodotto matrice-vettore y = Ax per una matrice sparsa in formato CSR
 *        utilizzando OpenMP per la parallelizzazione del ciclo sulle righe.
 *
 * Dettagli dell'implementazione:
 * - La parallelizzazione avviene sul ciclo esterno che itera sulle righe della matrice A.
 *   Questo è un approccio comune per SpMV perché le computazioni per ogni riga di y sono indipendenti.
 * - Ogni thread OpenMP gestisce un sottoinsieme di righe. La distribuzione delle righe
 *   ai thread è gestita dalla direttiva OpenMP e dalla clausola 'schedule'.
 * - La variabile 'sum' (usata per accumulare il prodotto scalare per una singola riga y[i])
 *   è dichiarata privata ('private(sum)') per ogni thread all'interno della regione parallela.
 *   Questo è cruciale per evitare race condition, dove più thread tentano di modificare
 *   la stessa variabile 'sum' contemporaneamente, portando a risultati errati.
 * - Si utilizza una schedule 'static' ('schedule(static)') per la distribuzione delle iterazioni (righe)
 *   ai thread. Con 'static', le N righe vengono divise in blocchi di dimensione circa (N / numero_thread)
 *   e ogni blocco viene assegnato a un thread all'inizio del loop. Questa schedule ha basso overhead.
 *   Per matrici dove il numero di non-zeri per riga varia molto (matrici "sbilanciate"),
 *   altre schedule come 'dynamic' o 'guided' potrebbero offrire un miglior bilanciamento del carico
 *   a costo di un overhead maggiore, ma 'static' è spesso un buon punto di partenza.
 *
 * @param A Puntatore alla matrice CSR (CSRMatrix). La matrice non viene modificata (const).
 * @param x Puntatore al vettore di input x. Non viene modificato (const).
 * @param y Puntatore al vettore di output y. Verrà riempito con il risultato del prodotto Ax.
 * @param num_threads Numero di thread OpenMP che l'utente desidera utilizzare per il calcolo.
 *                    Se num_threads <= 0, OpenMP utilizzerà il numero di thread di default
 *                    (spesso determinato dal sistema, come il numero di core disponibili).
 */

void openmp_spmv_csr(const CSRMatrix *A, const float *x, float *y, int num_threads) {
    // check if pointer are not null
    if (!A || !A->IRP || !A->JA || !A->AS || !x || !y) {
        fprintf(stderr, "Errore [openmp_spmv_csr]: uno o più argomenti sono NULL.\n");
        return;
    }
    // check consistency of dimension for x and y
    // if (A->ncols <= 0 || A->nrows <= 0) {
    //     fprintf(stderr, "Errore [openmp_spmv_csr]: dimensioni della matrice non valide.\n");
    //     return;
    // }


    // set the number of threads to use, if is specified and valid
    // if num_threads is <= 0, OpenMP use the default number
    if (num_threads > 0) {
        omp_set_num_threads(num_threads);
    }

    // --- REGIONE PARALLELA OPENMP ---
    // La direttiva '#pragma omp parallel for' istruisce il compilatore a parallelizzare
    // il ciclo 'for' immediatamente seguente. Un team di thread verrà creato (o riutilizzato).
    //
    // Clausole della direttiva:
    // - 'shared(A, x, y)': Le variabili A (puntatore alla matrice), x (puntatore al vettore input),
    //                      e y (puntatore al vettore output) sono condivise tra tutti i thread.
    //                      Questo significa che tutti i thread vedono e accedono alle stesse istanze
    //                      di queste variabili in memoria.
    //                      'A' e 'x' sono letti in modo concorrente.
    //                      'y' è scritto in modo concorrente, ma ogni thread scrive in una posizione
    //                      diversa (y[i]), quindi non c'è race condition su 'y' stesso.
    // - 'private(i, k, sum)': Le variabili 'i' (contatore del ciclo esterno, per le righe),
    //                         'k' (contatore del ciclo interno, per gli elementi non-zero) e 'sum'
    //                         sono dichiarate private. Ogni thread avrà la sua copia locale di queste
    //                         variabili. Questo è essenziale per 'i' e 'k' perché sono variabili
    //                         di loop, e per 'sum' per evitare che i thread sovrascrivano
    //                         l'accumulo parziale degli altri.
    // - 'schedule(static)': Specifica come le iterazioni del loop (le righe da 0 a A->nrows-1)
    //                       vengono distribuite ai thread. 'static' senza un argomento 'chunk_size'
    //                       divide le iterazioni in blocchi di dimensione circa uguale e le assegna
    //                       ai thread in modo fisso.
    #pragma omp parallel for shared(A, x, y) schedule(static)
    for (int i = 0; i < A->nrows; i++) {
        float sum = 0.0f; // 'sum' is private for each thread

        // the second 'for' calculate the product for row 'i' (y_i = A_i * x)
        // A->IRP[i]: index of JA and AS of the first non-zero element of the row 'i'
        // A->IRP[i+1]-1: index of the last non-zero element of row 'i'
        for (int k = A->IRP[i]; k < A->IRP[i + 1]; k++) {
            // A->JA[k]: column index of the current non-zero element
            // A->AS[k]: value of current non-zero element
            // x[A->JA[k]]: corresponding element of the vector x
            sum += A->AS[k] * x[A->JA[k]]; // accumulate product
        }
        y[i] = sum; // each thread write on its element y[i]
    }
    // at the end of #pragma construct, there is an implicit barrier: all trheads wait that the others has completed
}