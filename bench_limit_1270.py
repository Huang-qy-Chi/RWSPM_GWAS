"""
bench_limit_1270.py -- memory-lean benchmark of the limit p-value path at n=1270.

Why a second script
-------------------
The first attempt passed the whole ``G_tilde`` (1488, 1270, 21) array to every
worker.  Windows starts workers with *spawn*, so each one unpickles its own
0.3 GiB copy; on top of that the limit path needs an n x n distance matrix, two
n x n BDD kernels and an n x n eigendecomposition workspace.  With only ~10.8 GB
free that dies, sometimes without a traceback.

Fix: the features are cached to disk once, and every worker **memory-maps the
cache and materialises only the single region it is currently processing**.
Peak per worker is then ~100-200 MB instead of ~1 GiB.

What is compared (p-value method = limit, so no permutation is needed by either)
-------------------------------------------------------------------------------
ORIGINAL : bcov_perm_test_gwas(x, DY, n_perm=N_PERM, method='limit')
           -- builds an n x n DX, runs N_PERM wasted permutations of the
              pure-python _bcov_from_dm, then rebuilds the SNP-side BDD kernel
              inside _bcov_limit_pvalue for every region.
SNP + C  : C-built RY + the validated C statistic + Kx reused across regions +
           no permutations.

Usage
-----
    python bench_limit_1270.py --n_process 3 --n_regions 3
    python bench_limit_1270.py --n_process 3 --n_regions 1488 --n_original 20
"""
import argparse
import multiprocessing
import os
import sys
import time

for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import bcov as B                       # noqa: E402
from seed import set_seed              # noqa: E402

M1, M2 = 150, 100
WINDOW_WIDTH, STEP_DIVISOR, N_QUANTILE = 11, 4, 21
N_SUBJ = 1270
CACHE = os.path.join(HERE, f"_gtilde_cache_n{N_SUBJ}_m1488.npy")

# worker state
_SNP = None
_MM = None
_N_PERM = 1
_DX = None
_KX_C = None
_N = None


def _init(cache_path, snp, n_perm):
    """Worker init: mmap the feature cache, build the SNP-side pieces once."""
    global _SNP, _MM, _N_PERM, _DX, _KX_C, _N
    _SNP = snp
    _MM = np.load(cache_path, mmap_mode="r")
    _N_PERM = n_perm
    _N = snp.shape[0]
    _DX = np.abs(snp[:, None] - snp[None, :])
    _KX_C = B._center_bdd_matrix(B._bdd_matrix_bias_c(_DX, "constant"))


def _region_dist(region_idx):
    """Distance matrix of one region, materialised from the mmap on demand."""
    Gt = np.asarray(_MM[region_idx], dtype=np.float64)
    return np.linalg.norm(Gt[:, None, :] - Gt[None, :, :], axis=-1)


def region_original(region_idx):
    """ORIGINAL path, untouched library call."""
    DY = _region_dist(region_idx)
    _, p = B.bcov_perm_test_gwas(_SNP.reshape(-1, 1), DY,
                                 n_perm=_N_PERM, method="limit")
    return float(p)


def region_snp_c(region_idx):
    """NEW path; same arithmetic as _bcov_limit_pvalue but Kx/stat supplied."""
    Gt = np.asarray(_MM[region_idx], dtype=np.float64)
    RY = B.build_rank_matrix(Gt, backend="c", dtype=np.int64)
    stat = B.bcov_snp_stat(_SNP, RY, backend="c")
    del RY
    DY = np.linalg.norm(Gt[:, None, :] - Gt[None, :, :], axis=-1)
    Ky_c = B._center_bdd_matrix(B._bdd_matrix_bias_c(DY, "constant"))
    ev = np.linalg.eigh(_KX_C * Ky_c)[0]
    ev = ev[ev > 0] / _N
    return float(B._hbe(ev, _N * stat))


def _time_pool(ctx, fn, idxs, n_process, label):
    t0 = time.perf_counter()
    with ctx.Pool(n_process, initializer=_init,
                  initargs=(CACHE, _SNP_MAIN, _N_PERM_MAIN)) as pool:
        out = np.array(pool.map(fn, idxs, chunksize=1))
    dt = time.perf_counter() - t0
    print(f"  {label}: wall {dt:.1f}s for {len(idxs)} regions "
          f"({dt / len(idxs):.2f} s/region)", flush=True)
    return dt, out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_process", type=int, default=3)
    ap.add_argument("--n_perm", type=int, default=1)
    ap.add_argument("--n_regions", type=int, default=3,
                    help="regions timed on the NEW path")
    ap.add_argument("--n_original", type=int, default=20,
                    help="regions timed on the ORIGINAL path, then extrapolated")
    ap.add_argument("--full_new", action="store_true",
                    help="time the NEW path on all 1488 regions")
    args = ap.parse_args(argv)

    global _SNP_MAIN, _N_PERM_MAIN
    _SNP_MAIN = None
    _N_PERM_MAIN = args.n_perm

    print("=" * 78)
    print("limit-path benchmark at n=1270  (one SNP, 1488 regions)")
    print(f"  workers {args.n_process} | original path n_perm={args.n_perm}")
    print("=" * 78, flush=True)

    if not os.path.exists(CACHE):
        print(f"ERROR: feature cache missing: {CACHE}")
        return 1
    MM = np.load(CACHE, mmap_mode="r")
    n_reg, n, L = MM.shape
    print(f"  cached features {MM.shape} on disk "
          f"({os.path.getsize(CACHE) / 2 ** 20:.0f} MiB)", flush=True)

    adni = np.load(os.path.join(HERE, "ADNI_1_3_sampled_25000.npy"),
                   allow_pickle=True)
    snp = np.asarray(adni, dtype=np.float64)[:, 0]
    _SNP_MAIN = snp
    del adni
    print(f"  SNP dosages {np.bincount(snp.astype(int))}", flush=True)

    ctx = multiprocessing.get_context("spawn")

    # sanity: the two paths must agree on a region
    _init(CACHE, snp, args.n_perm)
    p_o = region_original(0)
    p_n = region_snp_c(0)
    print(f"\n[sanity] region 0: original p={p_o:.10f}  SNP+C p={p_n:.10f}  "
          f"|diff|={abs(p_o - p_n):.3e}", flush=True)

    idxs_new = list(range(n_reg if not args.full_new else n_reg))
    print("\n[NEW path] SNP + C", flush=True)
    t_new, p_new = _time_pool(ctx, region_snp_c, idxs_new, args.n_process,
                              "SNP + C")

    n_orig = min(args.n_original, n_reg)
    print(f"\n[ORIGINAL path] generic Ball BCov (first {n_orig} regions)",
          flush=True)
    t_old, p_old = _time_pool(ctx, region_original, list(range(n_orig)),
                              args.n_process, "original")

    per_new = t_new / len(idxs_new)
    per_old = t_old / n_orig
    print("\n" + "=" * 78)
    print("RESULT")
    print("=" * 78)
    print(f"  per region  ORIGINAL : {per_old:>8.2f} s")
    print(f"  per region  SNP + C  : {per_new:>8.2f} s")
    print(f"  speed-up             : {per_old / per_new:>8.1f} x")
    print()
    full_new = per_new * n_reg
    full_old = per_old * n_reg
    print(f"  one SNP, all {n_reg} regions, {args.n_process} workers:")
    print(f"    ORIGINAL : {full_old:>8.1f} s  ({full_old / 60:.1f} min)")
    print(f"    SNP + C  : {full_new:>8.1f} s  ({full_new / 60:.2f} min)")
    print()
    core_old = per_old * n_reg
    core_new = per_new * n_reg
    print(f"  CPU-seconds per SNP (1-core equivalent): "
          f"ORIGINAL {core_old:.0f} s | SNP+C {core_new:.0f} s")
    print()
    print("  Extrapolation (this per-SNP CPU cost, 1488 regions/SNP):")
    for label, cpu in (("ORIGINAL", core_old), ("SNP + C", core_new)):
        print(f"    {label:<9}: 5000 SNP {cpu * 5000 / 3600:>9.0f} core-h "
              f"| 494465 SNP {cpu * 494465 / 86400:>8.0f} core-days")
    print()
    agree = np.isclose(p_old, p_new[:n_orig], rtol=1e-9, atol=1e-12)
    print(f"  p-value agreement on {n_orig} regions: "
          f"{int(agree.sum())}/{n_orig} identical, max|diff|="
          f"{np.max(np.abs(p_old - p_new[:n_orig])):.3e}")
    return 0


if __name__ == "__main__":
    multiprocessing.freeze_support()
    sys.exit(main())
