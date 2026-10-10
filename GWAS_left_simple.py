"""
GWAS_left_simple.py — Minimal timing harness for the region-wise BCov test.

Simplified from GWAS_left.py:
  * only the FIRST SNP (index 0) is tested,
  * only the first X of the 1488 regions are processed (default X = 1),
  * no RW-MTCCT cluster aggregation, no multiprocessing (serial, 1 core),
  * per-region timing = distance-matrix build (DY) + bcov_perm_test_gwas,
    exactly the workload of GWAS_left.py's compute_one_region.

Output
------
  * BCov statistic (and Gamma p-value) of the first region,
  * wall time of each of the first X regions (plus total / average).

Usage
-----
    python GWAS_left_simple.py                    # X=1, n_perm=2, method=gamma
    python GWAS_left_simple.py --X 3 --n_perm 10
    python GWAS_left_simple.py --X 2 --method limit   # requires cball_ext compiled
"""

import argparse
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from GWAS_left import (                       # noqa: E402  (reuse loaders/settings)
    SIDE, M1, M2, WINDOW_WIDTH, STEP_DIVISOR, N_QUANTILE,
    load_data, slide_width_step, LQD, bcov_perm_test_gwas, set_seed,
)


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Time the region-wise BCov test on SNP 0 (first X regions)"
    )
    parser.add_argument("--X", type=int, default=1,
                        help="number of regions to process/time (default 1)")
    parser.add_argument("--n_perm", type=int, default=2,
                        help="BCov permutations (default 2)")
    parser.add_argument("--method", type=str, default="gamma",
                        choices=["gamma", "permu", "limit"])
    parser.add_argument("--seed", type=int, default=1145)
    args = parser.parse_args(argv)
    X = max(1, args.X)
    set_seed(args.seed)

    t_total0 = time.time()

    # ---------------- data -------------------------------------------------
    print(f"[{SIDE}] Loading data ...")
    adni, cov, hipp = load_data()
    n_subj, n_snp_total = adni.shape
    print(f"  SNPs: {adni.shape}, covariates: {cov.shape}, "
          f"hippocampus: {hipp.shape}")

    # ---------------- region partition (window h=11 -> 1488 regions) -------
    b_step = int(np.ceil(WINDOW_WIDTH / STEP_DIVISOR))
    idx_dict = slide_width_step(b_step, WINDOW_WIDTH, M1, M2)
    sub_idx = idx_dict["idx"]
    m_all = sub_idx.shape[0]
    width_x, width_y = idx_dict["width_x"], idx_dict["width_y"]
    X = min(X, m_all)
    print(f"  Partition: window={WINDOW_WIDTH}, step={b_step} -> "
          f"{width_x} x {width_y} = {m_all} subregions; "
          f"processing the first {X}")

    # ---------------- LQD + covariate adjustment (only first X regions) ----
    t = np.linspace(0, 1, N_QUANTILE)
    t_lqd0 = time.time()
    G_list = LQD(hipp, sub_idx[:X], t=t)              # (X, n_subj, L)
    t_lqd1 = time.time()
    print(f"  LQD transform on {X} region(s): {t_lqd1 - t_lqd0:.1f}s")
    del hipp

    n, p = cov.shape
    P_cov = (np.identity(n)
             - cov @ np.linalg.inv(cov.T @ cov + 1e-8 * np.identity(p)) @ cov.T)
    G_tilde = np.stack([P_cov @ G_j for G_j in G_list])   # (X, n_subj, L)
    del cov, G_list
    print(f"  Adjusted LQD features: {G_tilde.shape}")

    # ---------------- BCov test on SNP 0, region by region -----------------
    X_j = np.asarray(adni[:, 0], dtype=float).reshape(-1, 1)
    is_const = bool(np.all(X_j == X_j[0]))
    print(f"  SNP 0 genotype: {X_j.shape}, constant={is_const} "
          f"(n_perm={args.n_perm}, method={args.method})")

    per_region = []
    for r in range(X):
        Gt = G_tilde[r]
        t_dy0 = time.time()
        DY = np.linalg.norm(Gt[:, None, :] - Gt[None, :, :], axis=-1)  # (n, n)
        t_dy1 = time.time()

        t_test0 = time.time()
        if is_const:                      # same guard as GWAS_left.py
            stat, pval = 0.0, 1.0
        else:
            stat, pval = bcov_perm_test_gwas(
                X_j, DY, n_perm=args.n_perm, method=args.method)
            if not np.isfinite(pval) or pval < 0 or pval > 1:
                pval = 1.0
        t_test1 = time.time()

        dt_dy = t_dy1 - t_dy0
        dt_test = t_test1 - t_test0
        dt_reg = t_test1 - t_dy0
        per_region.append({"region": r, "stat": float(stat), "pval": float(pval),
                           "t_dy": dt_dy, "t_test": dt_test, "t_total": dt_reg})
        print(f"  region {r}: BCov stat = {stat:.6e}  p = {pval:.6e}  "
              f"time = {dt_reg:.3f}s  (DY {dt_dy:.3f}s + test {dt_test:.3f}s)")

    # ---------------- summary ---------------------------------------------
    total = sum(rr["t_total"] for rr in per_region)
    avg = total / X
    print("\n" + "=" * 72)
    print(f"SUMMARY - SNP 0, first {X} of {m_all} regions, "
          f"n_perm={args.n_perm}, method={args.method}, 1 core")
    print(f"  region 0 (of {m_all}) BCov statistic : {per_region[0]['stat']:.6e}")
    print(f"  region 0 (of {m_all}) p-value        : {per_region[0]['pval']:.6e}")
    print(f"  total time for {X} region(s)         : {total:.3f} s")
    print(f"  average time per region              : {avg:.3f} s")
    print(f"  est. 1 SNP x all {m_all} regions (1 core): {avg * m_all / 60:.2f} min")
    print(f"  est. {m_all} regions x {n_snp_total} SNPs (1 core)   : "
          f"{avg * m_all * n_snp_total / 3600:.2f} h")
    print(f"  total wall time (incl. preprocessing): {time.time() - t_total0:.1f} s")
    print("=" * 72)

    return {"X": X, "m_all": m_all, "per_region": per_region,
            "total_region_time": total, "avg_per_region": avg}


if __name__ == "__main__":
    main()
