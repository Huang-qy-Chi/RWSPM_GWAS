#!/bin/bash
# ============================================================
# Compile cball_ext.so for Linux
# Requires: gcc
#
# Run from anywhere:  bash compile_linux.sh
# (the script cd's to its own directory first)
#
# NOTE: To use OpenMP parallel speedup, add -fopenmp to the gcc
# command below — but only if libgomp is installed
# (check: ldconfig -p | grep libgomp). Without it, the code
# falls back to single-threaded automatically.
# ============================================================
set -e
cd "$(dirname "$0")"

if ! command -v gcc >/dev/null 2>&1; then
    echo "ERROR: gcc not found. Install it first, e.g.:"
    echo "  Ubuntu/Debian: sudo apt install gcc"
    echo "  CentOS/RHEL:   sudo yum install gcc"
    exit 1
fi

echo "Compiling cball_ext.so ..."
gcc -shared -O3 -fPIC \
    cball_wrapper.c bdd_matrix.c utilities.c median.c \
    -I. -lm \
    -o cball_ext.so

echo "SUCCESS: cball_ext.so created"
ls -lh cball_ext.so
