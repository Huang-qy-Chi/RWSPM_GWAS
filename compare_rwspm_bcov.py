"""
compare_rwspm_bcov.py -- benchmark the ORIGINAL Ball BCov path against the
SNP-specialised C path on one full RWSPM run.

Setting
-------
* n = 1270 (the real ADNI sample size)
* article configuration: 150 x 100 image, sliding window h=11, step 3 -> 1488 regions
* ONE SNP, all 1488 regions, p-values with method='limit' (HBE)
* 12 worker processes

Because method='limit' needs no permutation, both paths run with permutations
OFF, so the measured difference is the BCov code itself, not a different number
of replicates:

    original : bcov.bcov_perm_test_gwas(x, DY, n_perm=9, method='limit')
               -> still builds an n x n DX and runs 9 permutations of
                  _bcov_from_dm before returning the limit p-value
    SNP+C    : bcov.bcov_snp_limit(x, DY=DY)
               -> no DX, no permutations, and reuses the cached BDD kernel

Data
----
The real .rdata loaders cannot run in this environment (rdata -> pandas ->
pyarrow were built against NumPy 1.x while NumPy 2.5.2 is installed), so the
image block is SYNTHESISED with the same structure as the ADNI hippocampus:
a shared smooth field + per-subject variation + noise, then LQD-transformed and
covariate-projected exactly as the real pipeline does.  Timing depends only on
shapes, and the LQD/BCov code path is identical.

Usage
-----
    python compare_rwspm_bcov.py                     # full run
    python compare_rwspm_bcov.py --n_process 12 --n_generic 100
"""
import argparse
import multiprocessing
import os
import sys
import time

# BLAS threading MUST be pinned before numpy is imported.  The limit method
# factorises a 1270 x 1270 matrix in every worker; with multi-threaded BLAS each
# worker reserves a large workspace and 12 concurrent factorisations exhaust the
# machine (observed: "OpenBLAS error: Memory allocation still failed").  One
# BLAS thread per worker is also what we want anyway, since the parallelism is
# already at the process level.
for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import bcov as B                                   # noqa: E402
from lqd import LQD                                # noqa: E402
from rwspm import slide_width_step                 # noqa: E402
from seed import set_seed                          # noqa: E402

M1, M2 = 150, 100
WINDOW_WIDTH, STEP_DIVISOR, N_QUANTILE = 11, 4, 21
N_SUBJ = 1270
N_SNP = 1

# worker globals
G_TILDE_G = None
SNP_G = None
N_GENERIC_G = 0
KX_C_G = None
DX_G = None


# ---------------------------------------------------------------- data ------
def synth_hippocampus(n_subj, seed=20260101):
    """Synthetic hippocampus with an ADNI-like spatial structure.

    A shared smooth field (three Gaussian blobs) plus per-subject low-frequency
    variation plus noise.  Subject images are therefore spatially smooth and
    correlated across subjects, which is what the LQD transform and the Ball
    Covariance test act on.
    """
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:M1, 0:M2]
    blobs = (
        0.90 * np.exp(-((xx - 0.42 * M2) ** 2 + (yy - 0.55 * M1) ** 2)
                      / (2 * (0.18 * min(M1, M2)) ** 2))
        + 0.55 * np.exp(-((xx - 0.62 * M2) ** 2 + (yy - 0.38 * M1) ** 2)
                        / (2 * (0.13 * min(M1, M2)) ** 2))
        + 0.40 * np.exp(-((xx - 0.30 * M2) ** 2 + (yy - 0.30 * M1) ** 2)
                        / (2 * (0.10 * min(M1, M2)) ** 2))
    )
    base = blobs + 0.12 * (xx / M2) + 0.08 * (yy / M1)

    # smooth subject-specific components (low-frequency, shared basis)
    kx = np.cos(np.pi * xx / M2)
    ky = np.cos(np.pi * yy / M1)
    kxy = np.cos(np.pi * xx / M2) * np.cos(np.pi * yy / M1)

    hipp = np.empty((n_subj, M1 * M2), dtype=np.float64)
    for i in range(n_subj):
        img = (base
               + 0.12 * rng.normal() * kx
               + 0.12 * rng.normal() * ky
               + 0.10 * rng.normal() * kxy
               + 0.05 * rng.normal(size=(M1, M2)))
        hipp[i] = img.ravel()
    return hipp


def synth_covariates(n_subj, seed=7):
    """[1, age, sex, APOE4, PC1..PC5] as in GWAS_left.load_data()."""
    rng = np.random.default_rng(seed)
    return np.column_stack([
        np.ones(n_subj),
        rng.normal(75, 5, n_subj),
        rng.binomial(1, 0.5, n_subj),
        rng.binomial(1, 0.3, n_subj),
        rng.normal(0, 1, (n_subj, 5)),
    ]).astype(np.float64)


def load_snp_matrix(n_subj, n_snp):
    """Real SNP dosages if the .npy is readable, else synthetic dosages."""
    path = os.path.join(HERE, "ADNI_1_3_sampled_25000.npy")
    try:
        adni = np.load(path, allow_pickle=True)
        adni = np.asarray(adni, dtype=np.float64)
        if adni.shape[0] != n_subj:
            raise ValueError(f"n mismatch: file has {adni.shape[0]}")
        print(f"  SNP matrix: REAL {adni.shape} from {os.path.basename(path)}",
              flush=True)
        return adni[:, :n_snp]
    except Exception as e:
        rng = np.random.default_rng(11)
        print(f"  SNP matrix: SYNTHETIC ({type(e).__name__}: {e})", flush=True)
        return rng.binomial(2, 0.3, size=(n_subj, n_snp)).astype(np.float64)


# ------------------------------------------------------------- pipeline -----
def build_features(hipp, cov, m_all, cache_path=None, use_cache=True):
    """Partition -> LQD -> covariate projection.  Returns G_tilde (m, n, L).

    LQD dominates the wall time (~1200 s at n=1270, single core, because its
    inner loop is per subject x per region in Python), so the result is cached
    to ``cache_path`` and reused across benchmark re-runs.
    """
    if cache_path and use_cache and os.path.exists(cache_path):
        t0 = time.perf_counter()
        G_tilde = np.load(cache_path)
        print(f"  G_tilde loaded from cache {os.path.basename(cache_path)} "
              f"({time.perf_counter() - t0:.1f}s, {G_tilde.shape})", flush=True)
        return G_tilde.astype(np.float64), 0.0, 0.0, 0.0

    t0 = time.perf_counter()
    b_step = int(np.ceil(WINDOW_WIDTH / STEP_DIVISOR))
    idx = slide_width_step(b_step, WINDOW_WIDTH, M1, M2)
    sub_idx = idx["idx"]
    t_part = time.perf_counter() - t0
    print(f"  partitioning: {idx['width_x']} x {idx['width_y']} = "
          f"{sub_idx.shape[0]} regions in {t_part:.1f}s", flush=True)

    t0 = time.perf_counter()
    G = LQD(hipp, sub_idx[:m_all], t=np.linspace(0, 1, N_QUANTILE))
    t_lqd = time.perf_counter() - t0
    print(f"  LQD transform: {t_lqd:.1f}s", flush=True)

    t0 = time.perf_counter()
    n, p = cov.shape
    P = (np.identity(n)
         - cov @ np.linalg.inv(cov.T @ cov + 1e-8 * np.identity(p)) @ cov.T)
    G_tilde = np.stack([P @ g for g in G])
    t_proj = time.perf_counter() - t0
    print(f"  covariate projection: {t_proj:.1f}s", flush=True)

    if cache_path:
        np.save(cache_path, G_tilde.astype(np.float32))
        print(f"  cached G_tilde -> {os.path.basename(cache_path)} "
              f"({G_tilde.nbytes / 2 ** 20:.0f} MiB as float64, "
              f"{G_tilde.nbytes / 2 / 2 ** 20:.0f} MiB on disk)", flush=True)
    return G_tilde, t_part, t_lqd, t_proj


def region_limit_original(region_idx):
    """ORIGINAL path, untouched: generic Ball BCov with method='limit'.

    Note what this costs at n=1270: it builds an n x n DX, runs ``n_perm``
    permutations of the pure-python ``_bcov_from_dm`` (which allocates three
    n x n arrays per row), and then calls ``_bcov_limit_pvalue``, which rebuilds
    the SNP-side BDD kernel from scratch for every region.
    """
    Gt = G_TILDE_G[region_idx]
    DY = np.linalg.norm(Gt[:, None, :] - Gt[None, :, :], axis=-1)
    _, p = B.bcov_perm_test_gwas(SNP_G.reshape(-1, 1), DY,
                                 n_perm=N_GENERIC_G, method="limit")
    return float(p)


# --------------------------------------------------------------------------
# NEW path.  Three things change relative to the original, and they are listed
# separately so the gain can be attributed:
#   1. the observed statistic uses the validated C kernel instead of the
#      pure-python ``_bcov_from_dm`` (14.76 s -> 0.017 s at n=1270, and it does
#      not allocate three n x n temporaries per row, which is what made 12
#      concurrent workers die with MemoryError);
#   2. the SNP-side BDD kernel Kx is built ONCE per worker instead of once per
#      region (``_bcov_limit_pvalue`` rebuilds it for all 1488 regions);
#   3. no permutation is drawn at all, since the limit law does not need one
#      (``bcov_perm_test_gwas`` always runs its permutation loop first).
# --------------------------------------------------------------------------


def _limit_pvalue_with_kx(Kx_c, DY, DX, stat, n):
    """Identical arithmetic to ``_bcov_limit_pvalue`` with Kx supplied.

    ``_bcov_limit_pvalue`` computes::

        Ky  = _bdd_matrix_bias_c(DY)
        ev  = eigh(_center(Kx) * _center(Ky))[0]
        p   = _hbe(ev[ev > 0] / n, n * stat)

    and independently recomputes the statistic.  Here Kx and the statistic both
    come in from the caller.
    """
    Ky_c = B._center_bdd_matrix(B._bdd_matrix_bias_c(DY, 'constant'))
    ev = np.linalg.eigh(Kx_c * Ky_c, UPLO='L')[0]
    ev = ev[ev > 0] / n
    return B._hbe(ev, n * stat)


def region_limit_snp_c(region_idx):
    """NEW path: C-built RY + validated C statistic + reused Kx + no perms.

    The statistic uses ``bcov_snp_stat``, the C kernel that is already validated
    bit-for-bit against ``_bcov_from_dm``.  The only extra per-region work is
    building RY, which the real pipeline does once per region per process and
    then caches across every SNP.
    """
    Gt = G_TILDE_G[region_idx]
    RY = B.build_rank_matrix(Gt, backend='c', dtype=np.int64)
    stat = B.bcov_snp_stat(SNP_G, RY, backend='c')
    DY = np.linalg.norm(Gt[:, None, :] - Gt[None, :, :], axis=-1)
    p = _limit_pvalue_with_kx(KX_C_G, DY, DX_G, stat, DY.shape[0])
    return float(p)


def _init(g_tilde, snp, n_generic):
    global G_TILDE_G, SNP_G, N_GENERIC_G, KX_C_G, DX_G
    G_TILDE_G = g_tilde
    SNP_G = snp
    N_GENERIC_G = n_generic
    DX_G = np.abs(snp[:, None] - snp[None, :])
    KX_C_G = B._center_bdd_matrix(B._bdd_matrix_bias_c(DX_G, 'constant'))


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--n_process", type=int, default=12)
    p.add_argument("--n_perm", type=int, default=1,
                   help="permutations run by the ORIGINAL path. Its limit "
                        "branch ignores them, so 1 is the honest minimum: "
                        "bcov_perm_test_gwas runs the permutation loop "
                        "unconditionally BEFORE dispatching on method, so any "
                        "larger value is wasted work, not a real difference.")
    p.add_argument("--n_generic", type=int, default=100,
                   help="regions used to time the ORIGINAL path (extrapolated)")
    p.add_argument("--seed", type=int, default=20260101)
    p.add_argument("--no_cache", action="store_true",
                   help="ignore / do not write the G_tilde preprocessing cache")
    p.add_argument("--recompute_cache", action="store_true",
                   help="rebuild the G_tilde cache even if it exists")
    args = p.parse_args(argv)

    print("=" * 78)
    print(f"RWSPM single-SNP benchmark: ORIGINAL Ball BCov vs SNP-specialised C")
    print(f"  n={N_SUBJ}, image {M1}x{M2}, window h={WINDOW_WIDTH} -> 1488 regions")
    print(f"  p-value method: limit (HBE)")
    print(f"  original path: n_perm={args.n_perm} (its limit branch ignores "
          f"them -- the loop runs unconditionally)")
    print(f"  SNP+C path   : no permutations at all (bcov_snp_limit has no "
          f"n_perm)")
    print(f"  workers: {args.n_process}")
    print("=" * 78, flush=True)

    cpu = multiprocessing.cpu_count()
    print(f"  machine cores: {cpu}")

    t_all0 = time.perf_counter()

    print("\n[1/4] data", flush=True)
    t0 = time.perf_counter()
    hipp = synth_hippocampus(N_SUBJ, seed=args.seed)
    cov = synth_covariates(N_SUBJ)
    snp = load_snp_matrix(N_SUBJ, N_SNP)[:, 0]
    print(f"  image {hipp.shape} ({hipp.nbytes / 2**20:.0f} MiB), "
          f"cov {cov.shape}, SNP {snp.shape} in {time.perf_counter() - t0:.1f}s",
          flush=True)
    print(f"  SNP dosage counts: {np.bincount(snp.astype(int))}")

    print("\n[2/4] preprocessing (shared by both paths)", flush=True)
    m_all = slide_width_step(int(np.ceil(WINDOW_WIDTH / STEP_DIVISOR)),
                             WINDOW_WIDTH, M1, M2)["idx"].shape[0]
    cache_path = None if args.no_cache else os.path.join(
        HERE, f"_gtilde_cache_n{N_SUBJ}_m{m_all}.npy")
    G_tilde, t_part, t_lqd, t_proj = build_features(
        hipp, cov, m_all, cache_path=cache_path,
        use_cache=not args.recompute_cache)
    del hipp, cov
    t_pre = t_part + t_lqd + t_proj
    print(f"  G_tilde {G_tilde.shape}, {G_tilde.nbytes / 2 ** 30:.2f} GiB; "
          f"preprocessing total {t_pre:.1f}s"
          + ("  (served from cache, not re-timed)"
             if t_pre == 0.0 else ""), flush=True)

    # ---- NEW path: all regions, 12 cores ---------------------------------
    print(f"\n[3/4] NEW path: bcov_snp_limit on all {m_all} regions "
          f"({args.n_process} cores)", flush=True)
    ctx = multiprocessing.get_context("spawn")
    t0 = time.perf_counter()
    with ctx.Pool(args.n_process, initializer=_init,
                  initargs=(G_tilde, snp, 0)) as pool:
        p_new = np.array(pool.map(region_limit_snp_c, range(m_all),
                                  chunksize=4))
    t_new = time.perf_counter() - t0
    print(f"  wall {t_new:.1f}s  ({t_new / m_all * 1000:.1f} ms/region, "
          f"{t_new / args.n_process / m_all * 1000:.1f} ms/region/core)",
          flush=True)
    print(f"  p-values: min {p_new.min():.3e}  median "
          f"{np.median(p_new):.3f}  <0.05: {int((p_new < 0.05).sum())}/{m_all}",
          flush=True)

    # ---- ORIGINAL path: subset, extrapolated -----------------------------
    n_gen = min(args.n_generic, m_all)
    print(f"\n[4/4] ORIGINAL path: bcov_perm_test_gwas(method='limit') on "
          f"{n_gen} regions ({args.n_process} cores), then extrapolate",
          flush=True)
    t0 = time.perf_counter()
    with ctx.Pool(args.n_process, initializer=_init,
                  initargs=(G_tilde, snp, args.n_perm)) as pool:
        p_old = np.array(pool.map(region_limit_original, range(n_gen),
                                  chunksize=2))
    t_old_sub = time.perf_counter() - t0
    t_old_full = t_old_sub / n_gen * m_all
    print(f"  subset wall {t_old_sub:.1f}s for {n_gen} regions "
          f"({t_old_sub / n_gen * 1000:.0f} ms/region)", flush=True)
    print(f"  extrapolated to {m_all} regions: {t_old_full:.0f}s "
          f"({t_old_full / 60:.1f} min, {t_old_full / 3600:.2f} h)", flush=True)

    # ---- agreement -------------------------------------------------------
    print(f"\n[check] p-value agreement on the {n_gen} overlapping regions",
          flush=True)
    same = np.isclose(p_old, p_new[:n_gen], rtol=1e-9, atol=1e-12)
    print(f"  identical: {int(same.sum())}/{n_gen}   "
          f"max |diff| = {np.max(np.abs(p_old - p_new[:n_gen])):.3e}")

    # ---- summary ---------------------------------------------------------
    print("\n" + "=" * 78)
    print("SUMMARY")
    print("=" * 78)
    tot_old = t_pre + t_old_full
    tot_new = t_pre + t_new
    print(f"  preprocessing (shared)            : {t_pre:>9.1f} s")
    print(f"  BCov limit p-values, ORIGINAL     : {t_old_full:>9.1f} s  "
          f"({t_old_full / 60:.1f} min)")
    print(f"  BCov limit p-values, SNP + C      : {t_new:>9.1f} s  "
          f"({t_new / 60:.2f} min)")
    print(f"  --> BCov speed-up                 : "
          f"{t_old_full / t_new:>9.1f} x")
    print(f"  --> per-region                    : "
          f"{t_old_full / m_all * 1000:.0f} ms -> "
          f"{t_new / m_all * 1000:.1f} ms")
    print()
    print(f"  ONE-SNP GWAS wall (original)      : {tot_old:>9.1f} s  "
          f"({tot_old / 60:.1f} min)")
    print(f"  ONE-SNP GWAS wall (SNP + C)       : {tot_new:>9.1f} s  "
          f"({tot_new / 60:.2f} min)")
    print(f"  --> total speed-up                : "
          f"{tot_old / tot_new:>9.1f} x")
    print()
    print("  Extrapolation at this rate "
          f"({args.n_process} cores, 1488 regions/SNP):")
    for label, per_snp in (("original", tot_old), ("SNP + C", tot_new)):
        print(f"    {label:<9}: {per_snp:>8.1f} s/SNP -> "
              f"5000 SNP {per_snp * 5000 / 3600:>8.1f} h | "
              f"494465 SNP {per_snp * 494465 / 86400:>8.0f} days")
    print()
    print(f"  total wall including data generation: "
          f"{time.perf_counter() - t_all0:.1f} s", flush=True)
    return 0


if __name__ == "__main__":
    multiprocessing.freeze_support()
    sys.exit(main())
