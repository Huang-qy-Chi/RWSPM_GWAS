"""
smoke_test_GWAS.py — End-to-end smoke test of the RW-SPM region-wise GWAS
pipeline for BOTH hemispheres (GWAS_left.py and GWAS_right.py) WITHOUT real
ADNI data.

What it does
------------
1. Generates synthetic data in the SAME format as the real ADNI inputs:
     - hippocampus image   : (n_subj, M1*M2) float64  -- here M1=30, M2=20
       (instead of the real 150 x 100), a smooth blob + per-subject noise
     - SNP matrix          : (n_subj, 1) float64  -- 1 SNP (binomial dosage)
     - covariates          : (n_subj, 9)  -- [1, age, sex, APOE4, PC1..PC5]
2. Verifies the GWAS_right.py wiring: it must import GWAS_left.py and override
   the hemisphere settings (SIDE="right", HIPP_FILE/HIPP_KEY of the right
   hippocampus, OUT_DIR="results_gwas_right").
3. Monkey-patches GWAS_left.py so that its settings point at the synthetic
   data (small image, 1 SNP) and runs the full pipeline once per hemisphere:
       partition (slide_width_step) -> LQD (L=21) -> covariate projection
       adjustment -> per-region BCov permutation test (Gamma p-value)
       -> RW-MTCCT connected-region aggregation (alpha0 = 0.001).
4. Verifies each hemisphere produces sane results and prints the same summary
   block as the real GWAS: average time per region, per-SNP time, MTCCT
   region count, and the extrapolated full 494,465-SNP scan time.

Usage
-----
    python smoke_test_GWAS.py                 # 1 SNP/side, n_perm=10, 2 workers
    python smoke_test_GWAS.py --n_perm 25 --n_process 4 --n_subj 120

Exit code 0 = PASS (both hemispheres ran end-to-end and outputs are sane).
"""

import argparse
import os
import sys
import time

import numpy as np

# ---------------------------------------------------------------------------
# Make GWAS_left (and its helpers rwspm/lqd/bcov/seed) importable
# ---------------------------------------------------------------------------
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import GWAS_left as gl   # noqa: E402
import GWAS_right as gr  # noqa: E402  (shares the GWAS_left module; verifies wiring)


# ---------------------------------------------------------------------------
# Synthetic data generator (same shape/format contract as the real files)
# ---------------------------------------------------------------------------
def make_synthetic_data(n_subj=60, M1=30, M2=20, seed=7):
    """Return (adni, cov, hipp) with the same layout as load_data():

    adni : (n_subj, n_snp)      genotype matrix (here 1 SNP)
    cov  : (n_subj, 9)          [1, age, sex, APOE4, PC1..PC5]
    hipp : (n_subj, M1*M2)      flattened hippocampus image
    """
    rng = np.random.default_rng(seed)

    # --- smooth base field (shared anatomy) -------------------------------
    yy, xx = np.mgrid[0:M1, 0:M2]
    blob = np.exp(-((xx - M2 * 0.45) ** 2 + (yy - M1 * 0.55) ** 2)
                  / (2 * (min(M1, M2) * 0.18) ** 2))
    grad = (xx / M2 + yy / M1) * 0.15
    base = 0.85 * blob + 0.15 * grad

    # --- per-subject images: anatomy + subject variation + noise ----------
    hipp = np.empty((n_subj, M1 * M2))
    for i in range(n_subj):
        subj_var = 0.1 * np.sin(2 * np.pi * (xx / M2) + i * 0.7) \
                 + 0.1 * np.cos(2 * np.pi * (yy / M1) + i * 0.3)
        img = base + subj_var + 0.05 * rng.standard_normal((M1, M2))
        hipp[i] = img.ravel()

    # --- 1 SNP: binomial dosage {0,1,2}, MAF ~ 0.3 ------------------------
    adni = rng.binomial(2, 0.3, size=(n_subj, 1)).astype(np.float64)

    # --- covariates: [1, age, sex, APOE4, PC1..PC5] ------------------------
    cov = np.column_stack([
        np.ones(n_subj),
        rng.normal(75, 5, n_subj),            # age
        rng.binomial(1, 0.5, n_subj),         # sex
        rng.binomial(1, 0.3, n_subj),         # APOE4
        rng.normal(0, 1, (n_subj, 5)),        # PC1..PC5
    ]).astype(np.float64)

    return adni, cov, hipp


# ---------------------------------------------------------------------------
# Per-side runner: point the shared GWAS_left module at synthetic data and
# run the full pipeline for one hemisphere (left or right).
# ---------------------------------------------------------------------------
def run_side(side, n_subj, n_perm, n_process, seed):
    """Run the full pipeline for one hemisphere; return (results, out_dir)."""
    gl.SIDE = side
    gl.M1, gl.M2 = 30, 20                  # image 150x100 -> 30x20
    gl.OUT_DIR = f"results_gwas_smoke_{side}"
    gl.TOTAL_SNP_GWAS = 494465             # keep the real extrapolation basis

    def _fake_load_data():
        return make_synthetic_data(n_subj=n_subj,
                                   M1=gl.M1, M2=gl.M2, seed=seed)

    gl.load_data = _fake_load_data        # never touches real .rdata/.npy

    print("=" * 72)
    print(f"SMOKE TEST [{side.upper()}] - synthetic data, NO real ADNI files used")
    print("=" * 72)
    results = gl.run_gwas(snp_start=0, n_snp=1, n_perm=n_perm,
                          method="gamma", n_process=n_process,
                          seed=seed, save_region_pvals=False)
    return results, os.path.join(HERE, gl.OUT_DIR)


# ---------------------------------------------------------------------------
# PASS / FAIL checks
# ---------------------------------------------------------------------------
def check_right_wiring():
    """GWAS_right.py must reuse GWAS_left's implementation with the
    hemisphere-specific overrides applied."""
    ok = True
    checks = [
        ("GWAS_right shares the GWAS_left module", gr._gl is gl),
        ("GWAS_right SIDE -> 'right'", gr._gl.SIDE == "right"),
        ("GWAS_right HIPP_FILE -> 'right_hipp_ori.rdata'",
         gr._gl.HIPP_FILE == "right_hipp_ori.rdata"),
        ("GWAS_right HIPP_KEY -> 'right_hipp_inf1_3'",
         gr._gl.HIPP_KEY == "right_hipp_inf1_3"),
        ("GWAS_right OUT_DIR -> 'results_gwas_right'",
         gr._gl.OUT_DIR == "results_gwas_right"),
    ]
    for name, passed in checks:
        ok &= bool(passed)
        print(f"  [{'PASS' if passed else 'FAIL'}] {name}")
    return ok


def check_side(side, results, out_dir):
    """Sanity checks for one hemisphere's pipeline run."""
    ok = True
    checks = []
    if len(results) == 1:
        r = results[0]
        checks.append(("pipeline returned 1 SNP result", True))
        checks.append(("MTCCT p in [0,1]",
                       isinstance(r["p_agg"], float) and 0.0 <= r["p_agg"] <= 1.0))
        checks.append(("MTCCT region count >= 0", int(r["B"]) >= 0))
        checks.append(("sig regions >= 0", int(r["n_sig"]) >= 0))
        checks.append(("per-SNP time > 0", r["time_snp"] > 0))
        checks.append(("per-region time > 0", r["time_per_region"] > 0))
        checks.append(("summary.csv written",
                       os.path.isfile(os.path.join(out_dir, "summary.csv"))))
        checks.append(("summary.npz written",
                       os.path.isfile(os.path.join(out_dir, "summary.npz"))))
    else:
        checks.append(("pipeline returned 1 SNP result", False))
    for name, passed in checks:
        ok &= bool(passed)
        print(f"  [{'PASS' if passed else 'FAIL'}] [{side}] {name}")
    return ok


def main(argv=None):
    multiprocessing_guard()  # noqa: F821 -- defined below (avoids spawn issues)
    parser = argparse.ArgumentParser(
        description="RW-SPM GWAS smoke test (left + right) on synthetic "
                    "30x20 data, 1 SNP per side"
    )
    parser.add_argument("--n_subj", type=int, default=60)
    parser.add_argument("--n_perm", type=int, default=10)
    parser.add_argument("--n_process", type=int, default=2)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args(argv)

    print("=" * 72)
    print("WIRING CHECK - GWAS_right.py must reuse GWAS_left with overrides")
    print("=" * 72)
    ok = check_right_wiring()

    # --- run the pipeline once per hemisphere ------------------------------
    t0 = time.time()
    results, out_dirs = {}, {}
    for side in ("left", "right"):
        results[side], out_dirs[side] = run_side(
            side, n_subj=args.n_subj, n_perm=args.n_perm,
            n_process=args.n_process, seed=args.seed,
        )
    wall = time.time() - t0

    # --- PASS / FAIL checks per hemisphere ---------------------------------
    for side in ("left", "right"):
        print(f"\n--- {side.upper()} hemisphere checks ---")
        ok &= check_side(side, results[side], out_dirs[side])

    print(f"\nSmoke test {'PASSED' if ok else 'FAILED'} "
          f"in {wall:.1f}s total wall time.")
    return 0 if ok else 1


def multiprocessing_guard():
    """Required on Windows spawn: called inside the __main__-guarded path."""
    import multiprocessing
    multiprocessing.freeze_support()


if __name__ == "__main__":
    sys.exit(main())
