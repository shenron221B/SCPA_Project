import os
import re
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from itertools import cycle

# --- CONFIGURAZIONE ---
RESULTS_DIR = "server_results"
PLOTS_ROOT_DIR = "plots"
CSV_DIR = "csv_results"

# --- REGEX DI PARSING ---
FILENAME_RE = re.compile(r"results_(?P<Format>\w+)_(?P<Mode>\w+)\.txt")

PERF_LINE_RE = re.compile(
    r"\[PERF\]\s+.*?"
    r"Threads:(?P<Threads>[-0-9]+),\s+"
    r"BlockSize:(?P<BlockSize>[-0-9]+),\s+"
    r"HackSize:(?P<HackSize>[-0-9]+),\s+"
    r"Time_s:(?P<Time_s>[0-9.]+),\s+"
    r"MFLOPS:(?P<MFLOPS>[-0-9.]+),\s+"
    r"NNZ:(?P<NNZ>[0-9]+),\s*"
    r"Matrix:(?P<Matrix>[\w.-]+)"
)


def ensure_dir(directory_path):
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)
    return directory_path


def parse_perf_logs():
    """Analizza tutti i file di log nella directory dei risultati."""
    if not os.path.exists(RESULTS_DIR):
        print(f"ERRORE: La directory dei risultati '{RESULTS_DIR}' non è stata trovata.")
        return pd.DataFrame()

    print(f"--- Inizio parsing dei file in '{RESULTS_DIR}' ---")
    data_collected = []

    for filename in sorted(os.listdir(RESULTS_DIR)):
        if not filename.startswith("results_"):
            continue

        file_match = FILENAME_RE.match(filename)
        if not file_match:
            print(f"  [ATTENZIONE] Nome file non riconosciuto, lo salto: {filename}")
            continue

        metadata = file_match.groupdict()
        file_format = metadata['Format'].upper()
        raw_mode = metadata['Mode'].upper()

        # --- LOGICA DI PARSING ---
        file_mode = raw_mode
        hack_size_from_file = -1

        if raw_mode.startswith('OPENMP'):
            file_mode = 'OpenMP'
            # Estrai numero dopo OPENMP se presente (per HLL)
            hack_str = raw_mode.replace('OPENMP', '')
            if hack_str: hack_size_from_file = int(hack_str)

        elif raw_mode.startswith('CUDA'):
            file_mode = 'CUDA'
            # Estrai numero dopo CUDA se presente (per HLL)
            hack_str = raw_mode.replace('CUDA', '')
            if hack_str: hack_size_from_file = int(hack_str)

        elif raw_mode == 'SERIAL':
            file_mode = 'SerialRef'

        print(
            f"  Analisi di: {filename} (Formato: {file_format}, Modalità: {file_mode}, HackSize: {hack_size_from_file})")

        filepath = os.path.join(RESULTS_DIR, filename)
        with open(filepath, 'r', errors='ignore') as f:
            for line in f:
                if line.startswith("[PERF]"):
                    perf_match = PERF_LINE_RE.search(line)
                    if perf_match:
                        row_data = perf_match.groupdict()
                        row_data['Format'] = file_format
                        row_data['Mode'] = file_mode
                        if hack_size_from_file != -1:
                            row_data['HackSize'] = hack_size_from_file
                        data_collected.append(row_data)

    if not data_collected:
        return pd.DataFrame()

    print(f"--- Parsing completato. Trovati {len(data_collected)} record [PERF]. ---")

    df = pd.DataFrame(data_collected)
    numeric_cols = ['Threads', 'BlockSize', 'HackSize', 'Time_s', 'MFLOPS', 'NNZ']
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col])

    initial_rows = len(df)
    df = df[df['MFLOPS'] >= 0]
    filtered_count = initial_rows - len(df)
    if filtered_count > 0:
        print(f"Filtrate {filtered_count} righe con MFLOPS < 0.")

    df['GFLOPS'] = df['MFLOPS'] / 1000

    initial_rows = len(df)
    df = df[~df['Matrix'].str.match(r'.*(_b|_x)$')]
    filtered_count = initial_rows - len(df)
    if filtered_count > 0:
        print(f"Filtrate {filtered_count} righe per matrici terminanti in '_b' o '_x'.")

    return df


def plot_line_series(df, x_col, y_col, plot_subdir, title_prefix, x_label, y_label, palette, xticks=None):
    all_matrices = sorted(df['Matrix'].unique())
    if not all_matrices:
        return

    mid_point = (len(all_matrices) + 1) // 2
    matrix_chunks = [all_matrices[:mid_point], all_matrices[mid_point:]]

    for i, chunk in enumerate(matrix_chunks):
        if not chunk: continue
        df_chunk = df[df['Matrix'].isin(chunk)]

        plt.figure(figsize=(16, 9))
        sns.lineplot(data=df_chunk, x=x_col, y=y_col, hue='Matrix',
                     marker='o', palette=palette, errorbar=None)

        title = f"{title_prefix} - Parte {i + 1}"
        plt.title(title, fontsize=16)
        plt.xlabel(x_label, fontsize=12)
        plt.ylabel(y_label, fontsize=12)
        plt.grid(True, which='both', linestyle='--')

        if xticks is not None and len(xticks) > 0:
            plt.xticks(xticks)

        if 'Efficiency' in y_col:
            plt.ylim(0, 1.2)
            plt.axhline(1.0, color='r', linestyle='--', label='Efficienza Ideale (1.0)')

        plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left')
        plt.tight_layout(rect=[0, 0, 0.9, 1])

        plot_path = os.path.join(plot_subdir, f"{y_col.lower()}_vs_{x_col.lower()}_part{i + 1}.png")
        plt.savefig(plot_path, bbox_inches='tight')
        plt.close()
        print(f"  Grafico salvato: {plot_path}")


def plot_bar_comparison(df_best, y_col, plot_subdir, title_prefix, y_label):
    all_matrices = sorted(df_best['Matrix'].unique())
    if not all_matrices:
        return

    mid_point = (len(all_matrices) + 1) // 2
    matrix_chunks = [all_matrices[:mid_point], all_matrices[mid_point:]]

    for i, chunk in enumerate(matrix_chunks):
        if not chunk: continue
        df_chunk = df_best[df_best['Matrix'].isin(chunk)]

        plt.figure(figsize=(18, 9))
        # ax = sns.barplot(data=df_chunk, x="Matrix", y=y_col, hue="Format")
        ax = sns.barplot(data=df_chunk, x="Matrix", y=y_col, hue="Format", errorbar=None)

        if 'Efficiency' in y_col:
            ax.set_ylim(0, 1.2)
            ax.axhline(1.0, color='r', linestyle='--', label='Efficienza Ideale (1.0)')

        plt.title(f"{title_prefix} - Parte {i + 1}", fontsize=16)
        plt.ylabel(y_label, fontsize=12)
        plt.xlabel("Matrice", fontsize=12)
        plt.xticks(rotation=45, ha="right")
        plt.grid(axis='y', linestyle='--')
        plt.tight_layout()
        plot_path = os.path.join(plot_subdir, f"comparison_{y_col.lower()}_part{i + 1}.png")
        plt.savefig(plot_path, bbox_inches='tight')
        plt.close()
        print(f"  Grafico salvato: {plot_path}")


def generate_openmp_plots(df_omp_csr, df_omp_hll, serial_ref_time, palette):
    print("\n--- Generazione Grafici OpenMP ---")
    if serial_ref_time.empty:
        print("Dati SerialRef non trovati, Speedup/Efficiency non calcolati per OpenMP.")
        return

    for df in [df_omp_csr, df_omp_hll]:
        if not df.empty:
            df['Speedup'] = df.apply(
                lambda row: serial_ref_time.get(row["Matrix"], np.nan) / row["Time_s"] if row['Time_s'] > 0 else 0,
                axis=1)
            df['Efficiency'] = df.apply(lambda row: row['Speedup'] / row['Threads'] if row['Threads'] > 0 else 0,
                                        axis=1)

    plot_dir_omp_csr = ensure_dir(os.path.join(PLOTS_ROOT_DIR, 'OpenMP', 'CSR'))
    print(f"\nGenerando grafici per OpenMP-CSR in '{plot_dir_omp_csr}'...")
    plot_line_series(df_omp_csr, 'Threads', 'GFLOPS', plot_dir_omp_csr, 'Performance OpenMP CSR', 'Numero di Thread',
                     'GFLOPS', palette)
    plot_line_series(df_omp_csr, 'Threads', 'Speedup', plot_dir_omp_csr, 'Speedup OpenMP CSR', 'Numero di Thread',
                     'Speedup (vs SerialRef)', palette)
    plot_line_series(df_omp_csr, 'Threads', 'Efficiency', plot_dir_omp_csr, 'Efficienza OpenMP CSR', 'Numero di Thread',
                     'Efficienza', palette)

    for hack_size in sorted(df_omp_hll['HackSize'].unique()):
        if hack_size == -1: continue
        df_hack = df_omp_hll[df_omp_hll['HackSize'] == hack_size].copy()
        plot_dir_omp_hll = ensure_dir(os.path.join(PLOTS_ROOT_DIR, 'OpenMP', 'HLL', f'HackSize_{int(hack_size)}'))
        print(f"\nGenerando grafici per OpenMP-HLL (HackSize={int(hack_size)}) in '{plot_dir_omp_hll}'...")

        plot_line_series(df_hack, 'Threads', 'GFLOPS', plot_dir_omp_hll,
                         f'Performance OpenMP HLL (HackSize={int(hack_size)})', 'Numero di Thread', 'GFLOPS', palette)
        plot_line_series(df_hack, 'Threads', 'Speedup', plot_dir_omp_hll,
                         f'Speedup OpenMP HLL (HackSize={int(hack_size)})', 'Numero di Thread',
                         'Speedup (vs SerialRef)', palette)
        plot_line_series(df_hack, 'Threads', 'Efficiency', plot_dir_omp_hll,
                         f'Efficienza OpenMP HLL (HackSize={int(hack_size)})', 'Numero di Thread', 'Efficienza',
                         palette)

    df_omp_all = pd.concat([df_omp_csr, df_omp_hll]).dropna(subset=['GFLOPS', 'Speedup', 'Efficiency'])
    if not df_omp_all.empty:
        best_omp = df_omp_all.loc[df_omp_all.groupby(['Matrix', 'Format', 'HackSize'])['GFLOPS'].idxmax()]
        plot_dir_omp_comp = ensure_dir(os.path.join(PLOTS_ROOT_DIR, 'OpenMP', 'Comparisons'))
        print(f"\nGenerando grafici di confronto per OpenMP in '{plot_dir_omp_comp}'...")
        plot_bar_comparison(best_omp, 'GFLOPS', plot_dir_omp_comp, 'Confronto Performance di Picco OpenMP', 'GFLOPS')
        plot_bar_comparison(best_omp, 'Speedup', plot_dir_omp_comp, 'Confronto Speedup di Picco OpenMP', 'Speedup')
        plot_bar_comparison(best_omp, 'Efficiency', plot_dir_omp_comp, 'Confronto Efficienza di Picco OpenMP',
                            'Efficienza')


def generate_cuda_plots(df_cuda_csr, df_cuda_hll, serial_ref_time, palette):
    print("\n--- Generazione Grafici CUDA ---")
    if serial_ref_time.empty:
        print("Dati SerialRef non trovati, Speedup non calcolato per CUDA.")
    else:
        for df in [df_cuda_csr, df_cuda_hll]:
            if not df.empty:
                df['Speedup'] = df.apply(
                    lambda row: serial_ref_time.get(row["Matrix"], np.nan) / row["Time_s"] if row['Time_s'] > 0 else 0,
                    axis=1)

    plot_dir_cuda_csr = ensure_dir(os.path.join(PLOTS_ROOT_DIR, 'CUDA', 'CSR'))
    print(f"\nGenerando grafici per CUDA-CSR in '{plot_dir_cuda_csr}'...")
    xticks_csr = sorted(df_cuda_csr['BlockSize'].unique())
    plot_line_series(df_cuda_csr, 'BlockSize', 'GFLOPS', plot_dir_cuda_csr, 'Performance CUDA CSR', 'Dimensione Blocco',
                     'GFLOPS', palette, xticks=xticks_csr)
    plot_line_series(df_cuda_csr, 'BlockSize', 'Speedup', plot_dir_cuda_csr, 'Speedup CUDA CSR', 'Dimensione Blocco',
                     'Speedup (vs SerialRef)', palette, xticks=xticks_csr)

    for hack_size in sorted(df_cuda_hll['HackSize'].unique()):
        if hack_size == -1: continue
        df_hack = df_cuda_hll[df_cuda_hll['HackSize'] == hack_size]
        plot_dir_cuda_hll = ensure_dir(os.path.join(PLOTS_ROOT_DIR, 'CUDA', 'HLL', f'HackSize_{int(hack_size)}'))
        print(f"\nGenerando grafici per CUDA-HLL (HackSize={int(hack_size)}) in '{plot_dir_cuda_hll}'...")

        xticks_hll = sorted(df_hack['BlockSize'].unique())
        plot_line_series(df_hack, 'BlockSize', 'GFLOPS', plot_dir_cuda_hll,
                         f'Performance CUDA HLL (HackSize={int(hack_size)})', 'Dimensione Blocco', 'GFLOPS', palette,
                         xticks=xticks_hll)
        plot_line_series(df_hack, 'BlockSize', 'Speedup', plot_dir_cuda_hll,
                         f'Speedup CUDA HLL (HackSize={int(hack_size)})', 'Dimensione Blocco', 'Speedup (vs SerialRef)',
                         palette, xticks=xticks_hll)

    df_cuda_all = pd.concat([df_cuda_csr, df_cuda_hll]).dropna(subset=['GFLOPS', 'Speedup'])
    if not df_cuda_all.empty:
        best_cuda = df_cuda_all.loc[df_cuda_all.groupby(['Matrix', 'Format', 'HackSize'])['GFLOPS'].idxmax()]
        plot_dir_cuda_comp = ensure_dir(os.path.join(PLOTS_ROOT_DIR, 'CUDA', 'Comparisons'))
        print(f"\nGenerando grafici di confronto per CUDA in '{plot_dir_cuda_comp}'...")
        plot_bar_comparison(best_cuda, 'GFLOPS', plot_dir_cuda_comp, 'Confronto Performance di Picco CUDA', 'GFLOPS')
        plot_bar_comparison(best_cuda, 'Speedup', plot_dir_cuda_comp, 'Confronto Speedup di Picco CUDA', 'Speedup')


def main():
    """Funzione principale che orchestra il processo di analisi e plotting."""
    df_all = parse_perf_logs()
    if df_all.empty:
        print("Nessun dato valido trovato. Lo script termina.")
        return

    df_serial_ref = df_all[df_all['Mode'] == 'SerialRef'].copy()
    df_omp_csr = df_all[(df_all['Mode'] == 'OpenMP') & (df_all['Format'] == 'CSR')].copy()
    df_omp_hll = df_all[(df_all['Mode'] == 'OpenMP') & (df_all['Format'] == 'HLL')].copy()
    df_cuda_csr = df_all[(df_all['Mode'] == 'CUDA') & (df_all['Format'] == 'CSR')].copy()
    df_cuda_hll = df_all[(df_all['Mode'] == 'CUDA') & (df_all['Format'] == 'HLL')].copy()

    ensure_dir(CSV_DIR)
    df_omp_csr.to_csv(os.path.join(CSV_DIR, "openmp_csr_perf.csv"), index=False)
    df_omp_hll.to_csv(os.path.join(CSV_DIR, "openmp_hll_perf.csv"), index=False)
    df_cuda_csr.to_csv(os.path.join(CSV_DIR, "cuda_csr_perf.csv"), index=False)
    df_cuda_hll.to_csv(os.path.join(CSV_DIR, "cuda_hll_perf.csv"), index=False)
    print(f"\nFile CSV salvati nella cartella '{CSV_DIR}'.")

    serial_ref_time = df_serial_ref.groupby('Matrix')['Time_s'].mean()

    all_matrices = sorted(df_all['Matrix'].unique())
    colors = cycle(sns.color_palette("tab20", n_colors=20) + sns.color_palette("tab20b", n_colors=20))
    matrix_palette = {matrix: next(colors) for matrix in all_matrices}

    generate_openmp_plots(df_omp_csr, df_omp_hll, serial_ref_time, matrix_palette)
    generate_cuda_plots(df_cuda_csr, df_cuda_hll, serial_ref_time, matrix_palette)

    print("\n--- Analisi e generazione grafici completata! ---")


if __name__ == '__main__':
    main()