"""
GWAS_left.py — Region-wise GWAS on the LEFT hippocampus (ADNI) via RW-SPM.

Pipeline (following run_simu_parallel.py's region-wise BCov + connected-region
aggregation workflow, and Main_GWAS_left.py's ADNI data usage; settings per
RWSPM-article0730.pdf §2):

  (S1) Region partitioning : the 150 x 100 hippocampus image is swept by a
       sliding window of width h = 11 with step h/4 = 3, yielding
       48 x 31 = 1488 subregions (matches the article's ADNI application).
  (S2) Distribution aggregation: each subregion's within-region voxel
       distribution is embedded with the LQD transform on L = 21 quantile
       points.
  (S3) Covariate adjustment: residualize LQD features with the projection
       matrix P_cov = I - X (X'X + 1e-8 I)^{-1} X', where X = [1, age,
       gender, APOE-4, PC1..PC5] (1270 x 9, as in Main_GWAS_left.py).
  (S4) Region-wise independence test: for each SNP x subregion, the BCov
       permutation test (Gamma-approximated p-value, bcov.bcov_perm_test_gwas)
       is run against the pre-computed pairwise distance matrix of the
       adjusted LQD features.
  (S5) Cluster-level aggregation (RW-MTCCT): significant subregions
       (p < alpha0 = 0.001) that are spatially connected to the most
       significant subregion are combined with the Truncated Cauchy
       Combination Test; the final SNP-level significance threshold is
       alpha_snp = 1e-5 (suggestive, per the article).

Reported for every SNP:
  - aggregate (MTCCT) p-value,
  - number of regions used by MTCCT (size of the connected cluster),
  - number of preliminarily significant regions (p < alpha0),
  - significant-region counts at the fixed report thresholds
    REPORT_THRESHOLDS = (0.001, 1e-5, 5e-8), independent of alpha0,
  - per-SNP wall time and average per-region time,
  - extrapolated scan time for the full 494,465-SNP GWAS.

Windows / Linux compatibility
-----------------------------
  * .rdata files are parsed with the pure-Python ``rdata`` package
    (no R / pyreadr backend required) — the keys match the file contents:
      left_hipp_ori.rdata      -> "left_hipp_inf1_3"
      cov_inf_0711_1_3.rdata   -> "cov_inf1_3"
  * all file paths are resolved relative to this file's directory, so the
    script runs from any working directory;
  * multiprocessing uses the standard ``multiprocessing.Pool`` initializer
    pattern with ``freeze_support()`` and works under both Windows (spawn)
    and Linux (fork).

Usage
-----
    python GWAS_left.py                          # 1st 2 SNPs, n_perm=2 (demo)
    python GWAS_left.py --n_snp 5 --n_perm 10    # 1st 5 SNPs, 10 perms
    python GWAS_left.py --snp_start 100 --n_snp 50
    python GWAS_left.py --n_process 8 --alpha0 0.05
"""

import argparse
import multiprocessing
import os
import sys
import time
import warnings
from collections import OrderedDict

import numpy as np
from scipy import stats

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Path setup — resolve every input/output relative to this file's directory
# ---------------------------------------------------------------------------
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from rwspm import slide_width_step          # noqa: E402  (region partition)
from lqd import LQD                         # noqa: E402  (LQD transform)
from bcov import (                          # noqa: E402  (BCov permutation test)
    bcov_snp_stat,
    bcov_snp_gamma_sample,
    make_gamma_sample_plan,
    bcov_snp_null,
    bcov_snp_limit,
    bcov_perm_test_snp_gwas,
    build_rank_matrix,
    make_permutations,
    snp_backend_available,
)
from gammafit import gamma_moment_match     # noqa: E402
from seed import set_seed                   # noqa: E402

# ---------------------------------------------------------------------------
# Settings (RWSPM-article0730.pdf ADNI application)
# ---------------------------------------------------------------------------
M1, M2 = 150, 100            # hippocampus image dimensions (rows, cols)
WINDOW_WIDTH = 11            # optimal sliding-window width h (article)
STEP_DIVISOR = 4             # step size = ceil(h / 4)
N_QUANTILE = 21              # LQD interpolation points L
ALPHA0 = 0.001               # primary region screening threshold (article §2.4)
ALPHA_SNP = 1e-5             # final SNP-level suggestive threshold (article)
TOTAL_SNP_GWAS = 494465      # full GWAS scan size (article: 494,465 SNPs)

# Fixed thresholds for reporting per-SNP significant-region counts, independent
# of the MTCCT screening threshold alpha0 (0.001, suggestive 1e-5, genome-wide
# 5e-8 as in GWAS conventions).
REPORT_THRESHOLDS = (0.001, 1e-5, 5e-8)

# ---------------------------------------------------------------------------
# Hemisphere-specific data (GWAS_right.py overrides these before run_gwas)
# ---------------------------------------------------------------------------
SIDE = "left"
HIPP_FILE = "left_hipp_ori.rdata"
HIPP_KEY = "left_hipp_inf1_3"
COV_FILE = "cov_inf_0711_1_3.rdata"
COV_KEY = "cov_inf1_3"
SNP_FILE = "ADNI_1_3_sampled_25000.npy"
OUT_DIR = "results_gwas_left"

# ---------------------------------------------------------------------------
# Worker globals (set once per process by Pool initializer)
# ---------------------------------------------------------------------------
ADNI_global = None
G_TILDE_global = None
N_PERM_global = 10
METHOD_global = "gamma"
BACKEND_global = "auto"
CACHE_global = True
# Shared permutation set for every region of the current SNP.  Hoisting the
# permutations out of the per-region call is what lets the null distribution be
# computed once per SNP instead of once per (SNP, region) pair.
PERMS_global = None
DISTANCE_BACKEND_global = "cdist"
DISTANCE_BLOCK_SIZE_global = 128
# Cached observed statistic for the current SNP, keyed by region index.  It is
# the same for the region-major null pass and the statistic pass, so it is
# computed once and reused.
STAT0_CACHE = {}

# Per-process cache of the image rank matrices, one entry per sub-region.
# RY is a property of the image alone, so it is identical for every SNP.  The
# pool dispatches work region-major within each SNP, so consecutive SNPs
# revisit the same regions and the cache is reused instead of rebuilt.
#
# The cache is BOUNDED: with P worker processes each process only ever handles
# about m/P distinct regions, so caching every region would waste a factor P in
# memory (18 GiB per process at n=1270, m=1488).  An LRU cache sized to the
# actual working set keeps the reuse without the blow-up.
_RY_CACHE = OrderedDict()
_RY_CACHE_MAX = 0
# Distance-matrix cache, used only by the 'limit' method (which needs DY, not RY).
_DY_CACHE = OrderedDict()
_CACHE_STATS = {"hit": 0, "miss": 0, "evict": 0}


def estimate_ry_cache_mb(n_subj, n_regions):
    """Memory needed to cache RY for every sub-region, in MiB."""
    return n_subj * n_subj * 8 * n_regions / 2 ** 20


def init_worker(ADNI, G_tilde, n_perm, method, seed,
                backend="auto", cache=True, cache_max=0,
                distance_backend="cdist", distance_block_size=128,
                ratio=0.1, perm_seed=20260101, share_perms=True):
    """Pool initializer — copy-on-write sharing of read-only data (fork) or
    per-process copy (spawn).  Also gives each worker a distinct RNG stream."""
    global ADNI_global, G_TILDE_global, N_PERM_global, METHOD_global
    global BACKEND_global, CACHE_global, _RY_CACHE_MAX
    global DISTANCE_BACKEND_global, DISTANCE_BLOCK_SIZE_global
    global SAMPLE_RATIO, SAMPLE_SEED, SAMPLE_SHARE, SAMPLE_PLAN, SAMPLE_SNP
    SAMPLE_RATIO, SAMPLE_SEED, SAMPLE_SHARE = ratio, perm_seed, share_perms
    SAMPLE_PLAN, SAMPLE_SNP = None, None
    ADNI_global = ADNI
    G_TILDE_global = G_tilde
    N_PERM_global = n_perm
    METHOD_global = method
    BACKEND_global = backend
    CACHE_global = cache
    _RY_CACHE_MAX = max(0, int(cache_max))
    DISTANCE_BACKEND_global = distance_backend
    DISTANCE_BLOCK_SIZE_global = distance_block_size
    _RY_CACHE.clear()
    _DY_CACHE.clear()
    STAT0_CACHE.clear()
    _CACHE_STATS.update(hit=0, miss=0, evict=0)
    np.random.seed(seed + (os.getpid() % 100000))


def get_region_rank(region_idx):
    """Rank matrix RY of one sub-region, cached per process (LRU).

    RY depends only on the image features of the region, so every SNP reuses the
    same matrix.  It is built directly from the region feature matrix by the C
    kernel (`build_rank_matrix`), which computes the distances row by row: the
    n x n float64 distance matrix is never materialised, so peak memory is O(n)
    instead of O(n^2) beyond RY itself.
    """
    if CACHE_global and _RY_CACHE_MAX:
        RY = _RY_CACHE.get(region_idx)
        if RY is not None:
            _RY_CACHE.move_to_end(region_idx)
            _CACHE_STATS["hit"] += 1
            return RY

    _CACHE_STATS["miss"] += 1
    RY = build_rank_matrix(G_TILDE_global[region_idx], backend=BACKEND_global,
                           dtype=np.int64)

    if CACHE_global and _RY_CACHE_MAX:
        _RY_CACHE[region_idx] = RY
        if len(_RY_CACHE) > _RY_CACHE_MAX:
            _RY_CACHE.popitem(last=False)
            _CACHE_STATS["evict"] += 1
    return RY


def _snp_vector(snp_idx):
    return np.asarray(ADNI_global[:, snp_idx], dtype=float).reshape(-1)


def pairwise_region_distance(features, backend="cdist", block_size=128):
    """Euclidean DY without full n x n x L broadcast temporaries.

    cdist is the default. numpy_blocked retains the original per-pair NumPy
    reduction, using at most block_size rows of broadcast temporaries.
    Both return the full float64 n x n matrix required by the limit test.
    """
    features = np.asarray(features, dtype=np.float64)
    if features.ndim != 2:
        raise ValueError("features must be a two-dimensional array")
    if backend not in ("cdist", "numpy_blocked"):
        raise ValueError("distance backend must be cdist or numpy_blocked")
    if not isinstance(block_size, (int, np.integer)) or block_size < 1:
        raise ValueError("distance block_size must be a positive integer")
    n = features.shape[0]
    distance = np.empty((n, n), dtype=np.float64)
    if backend == "cdist":
        from scipy.spatial.distance import cdist
        cdist(features, features, metric="euclidean", out=distance)
    else:
        for start in range(0, n, block_size):
            stop = min(start + block_size, n)
            distance[start:stop] = np.linalg.norm(
                features[start:stop, None, :] - features[None, :, :], axis=-1)
    return distance


def _region_distance(region_idx):
    """Distance matrix of one sub-region, cached per process.

    Only the ``limit`` method needs this (the permutation paths use RY instead),
    so it is kept separate from the rank cache and shares the same LRU bound.
    """
    DY = _DY_CACHE.get(region_idx)
    if DY is not None:
        _DY_CACHE.move_to_end(region_idx)
        return DY
    Gt = G_TILDE_global[region_idx]
    DY = pairwise_region_distance(Gt, backend=DISTANCE_BACKEND_global,
                                  block_size=DISTANCE_BLOCK_SIZE_global)
    if _RY_CACHE_MAX:
        _DY_CACHE[region_idx] = DY
        if len(_DY_CACHE) > _RY_CACHE_MAX:
            _DY_CACHE.popitem(last=False)
    return DY


def null_for_region(args):
    """Permutation null statistics for one (SNP, region) pair.

    Exposed for callers that want the raw null (e.g. to score many SNPs against
    one region); the GWAS loop uses :func:`compute_one_region` directly.
    """
    snp_idx, region_idx = args
    X_j = _snp_vector(snp_idx)
    if np.all(X_j == X_j[0]):
        return np.zeros(0, dtype=np.float64)         # degenerate SNP marker
    RY = get_region_rank(region_idx)
    return bcov_snp_null(X_j, RY, PERMS_global, backend=BACKEND_global)


def stat_for_region(args):
    """Observed statistic (0 permutations) for one (SNP, region) pair.

    This is the cheap quantity used for pre-screening.
    """
    snp_idx, region_idx = args
    X_j = _snp_vector(snp_idx)
    if np.all(X_j == X_j[0]):
        return np.nan
    RY = get_region_rank(region_idx)
    return bcov_snp_stat(X_j, RY, backend=BACKEND_global)


def _install_perms(args):
    """Worker-side: install the shared permutation set for the current SNP.

    Every worker generates the set from the SAME seed, so all workers obtain an
    identical ``(n_perm, n)`` matrix locally.  That avoids shipping an
    ``n_perm x n`` int32 array to every worker once per SNP -- 3.6 MB per worker
    at n=10000, n_perm=9, times the number of workers, on every iteration.
    """
    global PERMS_global
    n = ADNI_global.shape[0]
    PERMS_global = make_permutations(n, N_PERM_global, seed=int(args))
    return 1


def _clear_perms(args):
    """Worker-side: drop the shared permutation set (single-pass mode)."""
    global PERMS_global
    PERMS_global = None
    return 0


def pvalue_from_null(stat, null, method=None):
    """Gamma-approximated (default) or permutation p-value from a null sample."""
    method = METHOD_global if method is None else method
    if method == 'gamma':
        fit = gamma_moment_match(null)
        return float(1.0 - stats.gamma.cdf(stat, a=fit['shape'],
                                           scale=fit['scale']))
    return float((np.sum(null >= stat) + 1.0) / (null.size + 1.0))


def compute_one_region(args):
    """BCov permutation test for one (SNP, region) pair.

    Uses the shared permutation set ``PERMS_global`` when one is installed, so
    the null draw is identical for every region of the current SNP; otherwise it
    falls back to freshly drawn permutations.  RY comes from the per-region LRU
    cache, so it is built at most once per region per process.

    Returns
    -------
    p : float
        Gamma-approximated (default) or permutation BCov p-value.
        1.0 for a degenerate/constant SNP or a non-finite p-value.
    """
    snp_idx, region_idx = args
    X_j = _snp_vector(snp_idx)
    if METHOD_global == 'gamma_sample':
        global SAMPLE_PLAN, SAMPLE_SNP
        if SAMPLE_SNP != snp_idx:
            SAMPLE_PLAN = make_gamma_sample_plan(X_j,SAMPLE_RATIO,N_PERM_global,SAMPLE_SEED+snp_idx)
            SAMPLE_SNP = snp_idx
        plan = SAMPLE_PLAN
        if not SAMPLE_SHARE:
            plan = dict(plan)
            plan['perm_seed'] = SAMPLE_SEED+snp_idx+1000003*(region_idx+1)
            plan['perms'] = make_permutations(plan['m'],N_PERM_global,seed=plan['perm_seed'])
        ry = None if len(plan['levels']) == 1 else get_region_rank(region_idx)
        stat,p,details = bcov_snp_gamma_sample(X_j,RY=ry,n_perm=N_PERM_global,
                                              ratio=SAMPLE_RATIO,backend=BACKEND_global,
                                              plan=plan,return_details=True)
        return {'region':region_idx,'stat':stat,'p':p,**details}
    if np.all(X_j == X_j[0]):                 # degenerate SNP -> no signal
        return 1.0

    if METHOD_global == 'limit':
        # HBE limit law: no permutations, no rank matrix -- just the two
        # distance matrices and an n x n eigendecomposition (O(n^3)).  This is
        # the method the original RWSPM pipeline used.
        return bcov_snp_limit(X_j, DY=_region_distance(region_idx))[1]

    RY = get_region_rank(region_idx)

    if PERMS_global is None:
        # single-pass mode: draw permutations here, one statistic each
        _, p = bcov_perm_test_snp_gwas(X_j, None, n_perm=N_PERM_global,
                                       method=METHOD_global, RY=RY,
                                       backend=BACKEND_global)
    else:
        # shared-permutation mode: the null depends only on (n, RY, perms), so
        # it is computed once here and reused for every region and every SNP
        stat = bcov_snp_stat(X_j, RY, backend=BACKEND_global)
        null = bcov_snp_null(X_j, RY, PERMS_global, backend=BACKEND_global)
        p = pvalue_from_null(stat, null)

    if not np.isfinite(p) or p < 0 or p > 1:
        return 1.0
    return float(p)


# ---------------------------------------------------------------------------
# RW-MTCCT with region count (mirrors rwspm.RW_MTCCT, also returns the number
# of subregions used by the cluster test — i.e. |A_e| in the article's eq. 8)
# ---------------------------------------------------------------------------
def rw_mtcct_report(pvec, num_region_r, num_region_c, alpha0=ALPHA0):
    """Connected-region MTCCT aggregation.

    Parameters
    ----------
    pvec : ndarray, shape (n_regions,)
        Region-wise p-values (ordering of slide_width_step: x slowest).
    num_region_r : int
        Number of regions along the row (x) direction (48 here).
    num_region_c : int
        Number of regions along the column (y) direction (31 here).
    alpha0 : float
        Primary screening threshold for "significant" regions.

    Returns
    -------
    (pval, n_cluster_regions, n_sig_regions) :
        MTCCT combined p-value; number of connected significant regions used
        by the test (B = |A_e|); total number of regions with p < alpha0.
    """
    from scipy.spatial.distance import cdist
    import networkx as nx
    from rwspm import CCT_chisq

    pvec = np.asarray(pvec, dtype=float)
    n_region = pvec.size
    sig_mask = pvec < alpha0
    n_sig = int(np.sum(sig_mask))
    if n_sig == 0:
        return 1.0, 0, 0

    # most significant (smallest p) among significant regions
    sig_idx = np.where(sig_mask)[0]
    min_pos = int(sig_idx[np.argmin(pvec[sig_idx])])

    # slide_width_step is row-major: (num_region_r, num_region_c).
    # The column count is the stride for both coordinate conversions.
    min_x = (min_pos // num_region_c) + 1        # 1-based row
    min_y = (min_pos % num_region_c) + 1         # 1-based column

    o_matrix = pvec.reshape(num_region_r, num_region_c)
    sig_coords = np.column_stack(np.where(o_matrix < alpha0))  # (row=x, col=y)
    sig_coords[:, 0] += 1
    sig_coords[:, 1] += 1

    o_dist = cdist(sig_coords, sig_coords, metric="euclidean")
    G = nx.Graph()
    G.add_nodes_from(range(sig_coords.shape[0]))
    adj_pairs = np.argwhere(np.abs(o_dist - 1.0) < 1e-6)
    for r, c in adj_pairs:
        if r < c:
            G.add_edge(int(r), int(c))
    comps = list(nx.connected_components(G))

    target_coord = np.array([[min_x, min_y]])
    dist_to_target = cdist(sig_coords, target_coord, metric="euclidean")
    min_dist_idx = int(np.argmin(dist_to_target))

    comp_id = None
    for idx, comp in enumerate(comps):
        if min_dist_idx in comp:
            comp_id = idx
            break
    if comp_id is None:
        return 1.0, 0, n_sig

    comp_vertices = list(comps[comp_id])
    detect_index = (
        (sig_coords[comp_vertices, 0] - 1) * num_region_c
        + sig_coords[comp_vertices, 1]
    )  # 1-based linear region index
    B = len(detect_index)

    non_sig_count = n_region - n_sig
    cct_region_count = non_sig_count + B
    weight_cct = np.zeros(n_region)
    for idx in detect_index:
        weight_cct[int(idx) - 1] = 1.0 / cct_region_count

    pval = CCT_chisq(pvec, weights=weight_cct)
    return float(pval), B, n_sig


# ---------------------------------------------------------------------------
# Data loaders (pure-Python ``rdata`` -> Windows/Linux portable)
# ---------------------------------------------------------------------------
def load_rdata_matrix(filepath, key):
    """Parse an .rdata file with the pure-Python rdata package.

    Handles both xarray DataArray and pandas DataFrame objects returned by
    ``rdata.conversion.convert`` (their ``to_numpy`` signatures differ), so
    the same code works across rdata versions on Windows and Linux.
    """
    import rdata
    parsed = rdata.parser.parse_file(filepath)
    converted = rdata.conversion.convert(parsed)
    if key not in converted:
        raise KeyError(
            f"Key '{key}' not found in {os.path.basename(filepath)}; "
            f"available: {list(converted.keys())}"
        )
    arr = converted[key].to_numpy()
    return np.asarray(arr, dtype=np.float64)


def load_data():
    """Load SNPs, covariates and the hippocampus image (as float64)."""
    adni = np.load(os.path.join(HERE, SNP_FILE), allow_pickle=True)
    adni = np.asarray(adni, dtype=np.float64)
    cov = load_rdata_matrix(os.path.join(HERE, COV_FILE), COV_KEY)
    hipp = load_rdata_matrix(os.path.join(HERE, HIPP_FILE), HIPP_KEY)
    return adni, cov, hipp


# ---------------------------------------------------------------------------
# LQD worker (region-level parallelism for the preprocessing step)
# ---------------------------------------------------------------------------
_LQD_ARGS = None


def _lqd_init(y, sub_idx, t):
    global _LQD_ARGS
    _LQD_ARGS = (y, sub_idx, t)


def _lqd_one_region(j):
    y, sub_idx, t = _LQD_ARGS
    return LQD(y, sub_idx[j:j + 1], t=t)[0]


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------
def run_gwas(snp_start=0, n_snp=2, n_perm=2, method="gamma",
             n_process=None, alpha0=ALPHA0, seed=1145,
             max_regions=None, save_region_pvals=True,
             backend="auto", cache_rank=True, cache_budget_mb=8000.0,
             share_perms=True, perm_seed=20260101,
             distance_backend="cdist", distance_block_size=128, ratio=0.1):
    """Run the region-wise GWAS on the configured hemisphere.

    Parameters
    ----------
    snp_start, n_snp : int
        SNP range to scan (indices into the sampled 25000-SNP matrix).
    n_perm : int
        Permutations for the BCov Gamma null (default 2 for the demo).
    method : str
        "gamma" (default) or "permu".  The limit-distribution (HBE) method is
        not offered by the SNP-specialised test.
    n_process : int or None
        Number of worker processes (default cpu_count - 4).
    alpha0 : float
        Primary region screening threshold for RW-MTCCT (default 0.001).
    seed : int
        Master random seed (permutation streams are worker-specific).
    max_regions : int or None
        If set, only the first ``max_regions`` regions are processed (testing
        only); untested regions get p = 1.0.
    save_region_pvals : bool
        Save per-SNP region p-value vectors as .npy files.
    backend : {"auto", "c", "numpy"}
        Which BCov implementation to use.  ``"auto"`` (default) takes the
        compiled C kernel when it is loadable and falls back to numpy
        otherwise.  ``"numpy"`` forces the pure-numpy path.
    cache_rank : bool
        Cache the per-region image rank matrix RY across SNPs (default True).
        RY depends only on the image, so every SNP reuses it; this removes the
        per-region O(n^2 log n) ranking from all but the first SNP.
    cache_budget_mb : float
        Memory budget (MiB) per worker for the RY cache.  If caching every
        region would exceed it, caching is disabled automatically with a
        warning, because an unbounded cache would be swapped out or OOM.
    share_perms : bool
        Use ONE permutation set for all regions of a SNP (default True).  The
        null distribution of a region depends only on ``(n, RY, permutations)``,
        so a single draw serves every region and the ``n_perm`` kernel calls are
        hoisted out of the innermost loop.  Set False for the original
        behaviour, where each (SNP, region) pair draws its own permutations.

        NOTE: with sharing, the region p-values of a SNP become dependent on the
        same draw, so the p-values are correlated across SNPs.  They stay valid
        per SNP (each is a proper Monte Carlo p-value for that SNP) but the
        RW-MTCCT combination assumes less dependence, so the aggregated p-values
        should be validated against ``share_perms=False`` on a small subset
        before being reported.
    perm_seed : int
        Seed from which every worker derives the shared permutation set, so all
        workers agree without shipping the array over IPC.
    distance_backend : {"cdist", "numpy_blocked"}
        Region feature distances for limit: cdist avoids three-dimensional
        temporaries; numpy_blocked preserves the original NumPy reduction.
    distance_block_size : int
        Positive row block size for numpy_blocked (default 128).
    ratio : float
        Fixed stratified sample fraction for gamma_sample only (default 0.1).
        All present genotypes retained. Calibrate m*BCov_m^2 against n*BCov_n^2.
    """
    t_total0 = time.time()
    if distance_backend not in ("cdist", "numpy_blocked"):
        raise ValueError("distance_backend must be cdist or numpy_blocked")
    if not isinstance(distance_block_size, (int, np.integer)) or distance_block_size < 1:
        raise ValueError("distance_block_size must be a positive integer")

    if n_process is None:
        n_process = max(1, multiprocessing.cpu_count() - 4)
    cpu_total = multiprocessing.cpu_count()
    print(f"[{SIDE}] CPU cores: {cpu_total}, worker processes: {n_process}")

    # ---------------- load data -------------------------------------------
    print(f"[{SIDE}] Loading data ...")
    adni, cov, hipp = load_data()
    n_subj, n_snp_total = adni.shape
    if method == 'gamma_sample':
        # Validate before starting worker processes or expensive preprocessing.
        make_gamma_sample_plan(adni[:,snp_start],ratio,n_perm,perm_seed+snp_start)
    print(f"[{SIDE}] SNPs: {adni.shape}, covariates: {cov.shape}, "
          f"hippocampus: {hipp.shape}")

    # ---------------- region partitioning (article: h=11 -> 1488) ----------
    b_step = int(np.ceil(WINDOW_WIDTH / STEP_DIVISOR))
    idx_dict = slide_width_step(b_step, WINDOW_WIDTH, M1, M2)
    sub_idx = idx_dict["idx"]
    m_all = sub_idx.shape[0]
    width_x, width_y = idx_dict["width_x"], idx_dict["width_y"]
    m = min(m_all, max_regions) if max_regions else m_all
    print(f"[{SIDE}] Partition: window={WINDOW_WIDTH}, step={b_step} -> "
          f"{width_x} x {width_y} = {m_all} subregions "
          f"(using {m} for this run)")

    # ---------------- LQD per region (parallel over regions) ---------------
    t = np.linspace(0, 1, N_QUANTILE)
    print(f"[{SIDE}] LQD transform on {m} regions (L={N_QUANTILE}) ...")
    t_lqd0 = time.time()
    if n_process > 1 and m > 1:
        from concurrent.futures import ProcessPoolExecutor
        with ProcessPoolExecutor(
            max_workers=min(n_process, m),
            initializer=_lqd_init,
            initargs=(hipp, sub_idx[:m], t),
        ) as pool:
            G_list = list(pool.map(_lqd_one_region, range(m)))
    else:
        G_list = LQD(hipp, sub_idx[:m], t=t)
    t_lqd1 = time.time()
    print(f"[{SIDE}] LQD done in {t_lqd1 - t_lqd0:.1f}s")
    del hipp

    # ---------------- covariate adjustment --------------------------------
    # P_cov = I - X (X'X + 1e-8 I)^{-1} X'  (as in Main_GWAS_left.py)
    print(f"[{SIDE}] Covariate adjustment (projection on {cov.shape[1]} cols) ...")
    n, p = cov.shape
    P_cov = (np.identity(n)
             - cov @ np.linalg.inv(cov.T @ cov + 1e-8 * np.identity(p)) @ cov.T)
    G_tilde = np.stack([P_cov @ G_j for G_j in G_list])   # (m, n, L)
    del cov, G_list
    print(f"[{SIDE}] Adjusted LQD features: {G_tilde.shape}")

    # ---------------- region-wise BCov tests -------------------------------
    end_snp = min(snp_start + n_snp, n_snp_total)

    # Size the per-worker RY cache.  RY is n x n int64 per region.  The GWAS
    # loop is SNP-major and region-minor, so every process sweeps the SAME
    # region indices for each SNP: an LRU holding the region set it revisits is
    # enough to serve every SNP from memory.  m + 8 entries covers a full sweep;
    # the budget check below disables caching when that would not fit.
    est_mb = estimate_ry_cache_mb(n_subj, m)
    cache_max = 0
    if cache_rank:
        cache_max = m + 8
        per_worker_mb = estimate_ry_cache_mb(n_subj, cache_max)
        if per_worker_mb > cache_budget_mb:
            print(f"[{SIDE}] WARNING: caching the {m}-region working set needs "
                  f"{per_worker_mb:.0f} MiB per worker > budget "
                  f"{cache_budget_mb:.0f} MiB -> RY caching DISABLED "
                  f"(raise --cache_budget_mb to enable)")
            cache_rank = False
            cache_max = 0
    be = backend
    if be == "auto":
        be = "c" if snp_backend_available() else "numpy"
    print(f"[{SIDE}] Testing SNPs {snp_start}..{end_snp - 1} "
          f"({n_snp} SNPs x {m} regions, n_perm={n_perm}, method={method})")
    print(f"[{SIDE}] BCov backend: {be}"
          + (f" (requested '{backend}')" if backend != be else "")
          + (f", RY cache: LRU {cache_max} regions/worker "
             f"(~{estimate_ry_cache_mb(n_subj, cache_max):.0f} MiB)"
             if cache_max else ", RY cache: off")
          + f"  [full set would be {est_mb:.0f} MiB]")
    print(f"[{SIDE}] Region distance backend: {distance_backend}, "
          f"NumPy block size: {distance_block_size}")
    print(f"[{SIDE}] permutations: "
          + (f"SHARED across regions (one draw per SNP, seed {perm_seed})"
             if share_perms else "independent per (SNP, region)"))

    set_seed(seed)
    per_snp_time = []
    results = []
    snp_t0 = time.time()
    with multiprocessing.Pool(
        processes=n_process,
        initializer=init_worker,
        initargs=(adni, G_tilde, n_perm, method, seed, be, cache_rank,
                  cache_max, distance_backend, distance_block_size,ratio,perm_seed,share_perms),
    ) as pool:
        for j in range(snp_start, end_snp):
            t_snp0 = time.time()

            if method == 'gamma_sample':
                import json
                plan=make_gamma_sample_plan(adni[:,j],ratio,n_perm,perm_seed+j)
                os.makedirs(os.path.join(HERE,OUT_DIR),exist_ok=True)
                np.savez(os.path.join(HERE,OUT_DIR,f'gamma_sample_plan_snp_{j:05d}.npz'),
                         indices=plan['indices'],perms=plan['perms'])
                with open(os.path.join(HERE,OUT_DIR,f'gamma_sample_plan_snp_{j:05d}.json'),'w') as f:
                    json.dump({k:v for k,v in plan.items() if k not in ('indices','perms')},f,indent=2)
            elif share_perms:
                # One permutation set per SNP: the null distribution of a region
                # depends only on (n, RY, permutations), so a single draw serves
                # every region -- this is what removes n_perm kernel calls from
                # the per-region cost.
                installed = pool.map(_install_perms,
                                     [perm_seed + j] * n_process)
            else:
                installed = pool.map(_clear_perms, range(n_process))

            region_outputs=pool.map(compute_one_region,[(j,r) for r in range(m)])
            if method == 'gamma_sample':
                pvals=np.array([r['p'] for r in region_outputs])
                with open(os.path.join(HERE,OUT_DIR,f'gamma_sample_diagnostics_snp_{j:05d}.json'),'w') as f:
                    json.dump({'ratio':ratio,'method':method,'regions':region_outputs},f,indent=2)
            else:
                pvals=np.array(region_outputs)
            t_snp1 = time.time()
            dt_snp = t_snp1 - t_snp0

            # untested regions -> p = 1.0 (conservative) so MTCCT sees 1488
            pvec_full = np.ones(m_all)
            pvec_full[:m] = pvals

            p_agg, B, n_sig = rw_mtcct_report(
                pvec_full, width_x, width_y, alpha0=alpha0
            )
            per_snp_time.append(dt_snp)
            results.append({
                "snp_idx": j,
                "p_agg": p_agg,
                "B": B,
                "n_sig": n_sig,
                "p_min": float(np.min(pvals)),
                "time_snp": dt_snp,
                "time_per_region": dt_snp / m,
                **{f"n_sig_lt_{t:g}": int(np.sum(pvec_full < t))
                   for t in REPORT_THRESHOLDS},
            })
            sig_extra = "  ".join(
                f"sig(p<{t:g})={int(np.sum(pvec_full < t)):d}"
                for t in REPORT_THRESHOLDS)
            print(f"[{SIDE}] SNP {j}: MTCCT p={p_agg:.6e}  "
                  f"regions_used={B}  sig_regions(p<{alpha0:g})={n_sig}  "
                  f"{sig_extra}  "
                  f"min_region_p={np.min(pvals):.6e}  "
                  f"time={dt_snp:.1f}s ({dt_snp / m:.3f}s/region)")

            if save_region_pvals:
                os.makedirs(os.path.join(HERE, OUT_DIR), exist_ok=True)
                np.save(os.path.join(HERE, OUT_DIR,
                                     f"p_values_snp_{j:05d}.npy"), pvec_full)

    snp_loop_time = time.time() - snp_t0

    # ---------------- summary & extrapolation ------------------------------
    mean_per_snp = float(np.mean(per_snp_time))
    mean_per_region = mean_per_snp / m
    est_full = mean_per_snp * TOTAL_SNP_GWAS          # at n_process workers
    est_full_1core = mean_per_snp * n_process * TOTAL_SNP_GWAS  # 1-core equiv
    est_sampled = mean_per_snp * n_snp_total          # 25000-SNP sample file

    print("\n" + "=" * 72)
    print(f"[{SIDE}] SUMMARY (SNPs {snp_start}..{end_snp - 1}, "
          f"{m} regions, n_perm={n_perm}, {n_process} workers)")
    print("=" * 72)
    for r in results:
        flag = " <-- suggestive (p<1e-5)" if r["p_agg"] < ALPHA_SNP else ""
        sig_extra = "  ".join(
            f"sig(p<{t:g})={r[f'n_sig_lt_{t:g}']}" for t in REPORT_THRESHOLDS)
        print(f"  SNP {r['snp_idx']:>5d}: MTCCT p = {r['p_agg']:.6e}"
              f"   regions_used_by_MTCCT = {r['B']}"
              f"   sig_regions(p<{alpha0:g}) = {r['n_sig']}"
              f"   {sig_extra}"
              f"   time = {r['time_snp']:.1f}s ({r['time_per_region']:.3f}s/region)"
              f"{flag}")
    print(f"\n  Average time per region      : {mean_per_region:.3f} s")
    print(f"  Average time per SNP         : {mean_per_snp:.1f} s "
          f"({mean_per_snp / 60:.1f} min)")
    print(f"  SNP-loop total wall time     : {snp_loop_time:.1f} s "
          f"({snp_loop_time / 60:.1f} min)")
    print(f"  Estimated full scan time     : {mean_per_snp * TOTAL_SNP_GWAS / 3600:.1f} h "
          f"({mean_per_snp * TOTAL_SNP_GWAS / 86400:.1f} days) "
          f"for {TOTAL_SNP_GWAS} SNPs @ {n_process} workers")
    print(f"    (1-core equivalent         : {est_full_1core / 3600:.1f} h)")
    print(f"  Estimated time for the 25000-SNP sample file: "
          f"{est_sampled / 3600:.1f} h")
    print(f"  Total wall time (incl. preprocessing): "
          f"{time.time() - t_total0:.1f} s")

    # ---------------- save summary ----------------------------------------
    os.makedirs(os.path.join(HERE, OUT_DIR), exist_ok=True)
    thresh_cols = [f"n_sig_lt_{t:g}" for t in REPORT_THRESHOLDS]
    summ = np.array([
        [r["snp_idx"], r["p_agg"], r["B"], r["n_sig"],
         *[r[c] for c in thresh_cols],
         r["p_min"], r["time_snp"], r["time_per_region"]]
        for r in results
    ])
    header = ("snp_idx,p_agg_mtcct,n_regions_used_mtcct,n_sig_regions,"
              + ",".join(thresh_cols) + ","
              + "min_region_p,time_snp_s,time_per_region_s")
    np.savetxt(os.path.join(HERE, OUT_DIR, "summary.csv"),
               summ, delimiter=",", header=header, comments="", fmt="%.8e")
    np.savez(os.path.join(HERE, OUT_DIR, "summary.npz"),
             results=summ,
             sig_thresholds=np.asarray(REPORT_THRESHOLDS, dtype=np.float64),
             mean_per_region=mean_per_region,
             mean_per_snp=mean_per_snp,
             est_full_gwas_h=mean_per_snp * TOTAL_SNP_GWAS / 3600,
             est_sampled_h=est_sampled / 3600,
             n_process=n_process, n_perm=n_perm, alpha0=alpha0,ratio=ratio,method=method,
             m_regions=m, m_all_regions=m_all, side=SIDE)
    print(f"[{SIDE}] Summary saved to {os.path.join(HERE, OUT_DIR)}")
    return results


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main(argv=None):
    """Command-line entry point (also used by GWAS_right.py)."""
    multiprocessing.freeze_support()
    parser = argparse.ArgumentParser(
        description=f"RW-SPM region-wise GWAS — {SIDE} hippocampus (ADNI)"
    )
    parser.add_argument("--snp_start", type=int, default=0,
                        help="first SNP index (default 0)")
    parser.add_argument("--n_snp", type=int, default=2,
                        help="number of SNPs to scan (default 2)")
    parser.add_argument("--n_perm", type=int, default=2,
                        help="BCov permutations (default 2)")
    parser.add_argument("--method", type=str, default="gamma",
                        choices=["gamma", "permu", "limit", "gamma_sample"],
                        help="p-value method: gamma (permutation null + Gamma "
                             "fit), permu (plain permutation), or limit (HBE "
                             "limit law, no permutations, O(n^3)), or gamma_sample "
                             "(fixed stratified subsample Gamma calibration)")
    parser.add_argument('--ratio',type=float,default=0.1,
                        help='fixed stratified subsample fraction for gamma_sample (default 0.1)')
    parser.add_argument("--backend", type=str, default="auto",
                        choices=["auto", "c", "numpy"],
                        help="BCov implementation: 'auto' uses the compiled C "
                             "kernel when available (default)")
    parser.add_argument("--no_cache_rank", action="store_true",
                        help="disable the cross-SNP cache of the per-region "
                             "image rank matrix RY")
    parser.add_argument("--cache_budget_mb", type=float, default=8000.0,
                        help="per-worker memory budget for the RY cache in "
                             "MiB; caching is disabled automatically if the "
                             "whole region set would not fit (default 8000)")
    parser.add_argument("--no_share_perms", action="store_true",
                        help="draw permutations independently for every "
                             "(SNP, region) pair instead of sharing one set per "
                             "SNP (slower; use to validate the shared mode)")
    parser.add_argument("--perm_seed", type=int, default=20260101,
                        help="seed for the shared permutation set (default "
                             "20260101)")
    parser.add_argument("--n_process", type=int, default=None,
                        help="worker processes (default cpu_count-4)")
    parser.add_argument("--distance_backend", choices=["cdist", "numpy_blocked"],
                        default="cdist", help="region Euclidean distance implementation")
    parser.add_argument("--distance_block_size", type=int, default=128,
                        help="positive row block size for numpy_blocked (default 128)")
    parser.add_argument("--alpha0", type=float, default=ALPHA0,
                        help="region screening threshold for MTCCT (default 0.001)")
    parser.add_argument("--seed", type=int, default=1145)
    parser.add_argument("--max_regions", type=int, default=None,
                        help="testing only: process first k regions")
    parser.add_argument("--no_save_pvals", action="store_true",
                        help="do not save per-SNP region p-value vectors")
    args = parser.parse_args(argv)

    run_gwas(
        snp_start=args.snp_start,
        n_snp=args.n_snp,
        n_perm=args.n_perm,
        method=args.method,
        n_process=args.n_process,
        alpha0=args.alpha0,
        seed=args.seed,
        max_regions=args.max_regions,
        save_region_pvals=not args.no_save_pvals,
        backend=args.backend,
        cache_rank=not args.no_cache_rank,
        cache_budget_mb=args.cache_budget_mb,
        share_perms=not args.no_share_perms,
        perm_seed=args.perm_seed,
        distance_backend=args.distance_backend,
        distance_block_size=args.distance_block_size,
        ratio=args.ratio,
    )


if __name__ == "__main__":
    main()
