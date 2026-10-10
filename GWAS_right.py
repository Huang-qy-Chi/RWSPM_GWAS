"""
GWAS_right.py — Region-wise GWAS on the RIGHT hippocampus (ADNI) via RW-SPM.

The pipeline is identical to GWAS_left.py (see its docstring for the full
method description): region partitioning (150x100, window h=11 -> 1488
subregions), LQD embedding (L=21), covariate projection adjustment, BCov
permutation tests per SNP x subregion (Gamma p-values), and RW-MTCCT
cluster-level aggregation (alpha0 = 0.001; SNP-level threshold 1e-5).

The only differences from GWAS_left.py are the hemisphere-specific data:

    right_hipp_ori.rdata   ->  "right_hipp_inf1_3"
    output directory       ->  results_gwas_right/

This script reuses GWAS_left.py's implementation: it imports the module,
points its hemisphere settings at the right hippocampus, and calls its
command-line entry point.  Use it exactly like GWAS_left.py:

    python GWAS_right.py                          # 1st 2 SNPs, n_perm=2 (demo)
    python GWAS_right.py --n_snp 5 --n_perm 10
    python GWAS_right.py --snp_start 100 --n_snp 50 --n_process 8
"""

import os
import sys

# Resolve paths relative to this file's directory, then import the shared
# implementation from GWAS_left.py (sibling module in the same folder).
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import GWAS_left as _gl  # noqa: E402

# --- Hemisphere-specific settings (override GWAS_left's module defaults) ----
_gl.SIDE = "right"
_gl.HIPP_FILE = "right_hipp_ori.rdata"
_gl.HIPP_KEY = "right_hipp_inf1_3"
_gl.OUT_DIR = "results_gwas_right"

if __name__ == "__main__":
    _gl.main()
