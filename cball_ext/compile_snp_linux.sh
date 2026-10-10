#!/bin/bash
# Build the SNP-specialised Ball Covariance kernel.
#
# Linux servers: no OpenMP needed, no Python headers needed -- the kernel is a
# plain shared library loaded through ctypes.
#
#   cd cball_ext && bash compile_snp_linux.sh
#
# NOTE: the output name is deliberately NOT cball_ext.so -- that file is the
# existing Ball BDD kernel built by compile_linux.sh, and overwriting it would
# break bcov._bdd_matrix_bias_c.
#
set -e
cd "$(dirname "$0")"

CC=${CC:-gcc}
OUT=${1:-csnpbcov.so}

$CC -O3 -fPIC -shared -o "$OUT" snp_bcov.c -lm
echo "built $OUT"
nm -D --defined-only "$OUT" 2>/dev/null | grep -E 'snp_bcov' || true
