#!/bin/bash

set -euo pipefail

echo "========================================="
echo "===   STARTING COMPILE & RUN PROCESS  ==="
echo "========================================="
echo "current hour: $(date)"

# --- 1. load module ---
echo "1. loading necessary module (cuda/11.8)..."
module purge
module load cuda/11.8
echo "module loaded:"
module list

export OMP_PROC_BIND=close
export OMP_PLACES=cores
echo "OpenMP affinity: OMP_PROC_BIND=${OMP_PROC_BIND}, OMP_PLACES=${OMP_PLACES}"

# --- 2. compilation process ---
echo "2. start compilation process..."

# go to the script directory (/data/lpompili/SCPA_Project/)
cd "$(dirname "$0")"

# create a new clean build
if [ -d "build" ]; then
    echo "   removing old 'build' directory..."
    rm -rf build
fi
echo "   creating new 'build' directory..."
mkdir build
cd build

# execute CMake and Make
echo "   execution of CMake..."
cmake ..
echo "   execution of Make (compile with 4 jobs)..."
make -j4

echo "compile successfully."

# --- 3. execution of tests ---
echo "3. starting benchmark suite..."

# move on root directory of the project
cd ..

# ensure that script is executable
chmod +x ./run_all_tests.sh

# execute script of tests
./run_all_tests.sh

echo "========================================="
echo "===   BUILD & RUN PROCESS COMPLETE   ==="
echo "========================================="