#!/bin/bash

# --- Configuration ---
# This script launches benchmark suites. The main C++ program will handle parameter sweeps.
EXECUTABLE="./build/spmv_exec"
BASE_OUTPUT_DIR="../benchmark_results_$(date +%Y%m%d_%H%M%S)"
MAIN_LOG_FILE="${BASE_OUTPUT_DIR}/master_execution_log.txt"

# HLL HackSizes to test
HLL_HACK_SIZES_TO_TEST=(16 32 64)

# --- Setup ---
echo "Creating output directory: ${BASE_OUTPUT_DIR}"
mkdir -p "${BASE_OUTPUT_DIR}"
exec > >(tee -a "${MAIN_LOG_FILE}") 2>&1

echo "Starting all SpMV benchmark suites at $(date)"
echo "----------------------------------------------------"

# Function to run a single benchmark suite
run_benchmark_suite() {
    local benchmark_mode_arg=$1  # es. csr_openmp, hll_cuda
    local extra_args_for_exec=$2 # es. "32" for hll_hack_size

    # Filename will be based on the benchmark mode itself
    local output_filename="results_${benchmark_mode_arg}${extra_args_for_exec}.txt"
    local output_path="${BASE_OUTPUT_DIR}/${output_filename}"

    local cmd_to_run="${EXECUTABLE} ${benchmark_mode_arg} ${extra_args_for_exec}"

    echo "=== Running Benchmark Suite: ${cmd_to_run} ==="
    echo "Outputting all [PERF] data for this suite to: ${output_path}"

    ${cmd_to_run} > "${output_path}" 2>&1

    if [ $? -eq 0 ]; then
        echo "Benchmark Suite [${benchmark_mode_arg} ${extra_args_for_exec}] COMPLETED successfully."
    else
        echo "Benchmark Suite [${benchmark_mode_arg} ${extra_args_for_exec}] FAILED. Check ${output_path}."
    fi
    echo "----------------------------------------"
}

# --- Execute the Benchmark Suites ---
# Ensure BENCHMARK=1 is set in main.cpp before compiling

# 1. Serial CSR (Reference Run)
run_benchmark_suite "csr_serial" ""

# 2. OpenMP CSR (sweeps threads 1-40 internally)
run_benchmark_suite "csr_openmp" ""

# 3. CUDA CSR (sweeps block sizes internally)
run_benchmark_suite "csr_cuda" ""

# 4. OpenMP HLL (one run for each hack_size, sweeping threads internally)
for hack_val in "${HLL_HACK_SIZES_TO_TEST[@]}"; do
    run_benchmark_suite "hll_openmp" "${hack_val}"
done

# 5. CUDA HLL (one run for each hack_size, sweeping block sizes internally)
for hack_val in "${HLL_HACK_SIZES_TO_TEST[@]}"; do
    run_benchmark_suite "hll_cuda" "${hack_val}"
done

echo "All benchmark suites finished at $(date)"