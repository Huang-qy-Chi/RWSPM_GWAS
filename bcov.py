"""
Empirical Ball Covariance BCov^2_n(X, Y)

Optimized with vectorized inner loops for speed.
"""

import os
import numpy as np
from scipy import stats
import ctypes
import warnings
from pathlib import Path
warnings.filterwarnings("ignore")

# ============================================================
# C extension loading (cball_ext.dll / cball_ext.so)
# ============================================================
_CBALL: "ctypes.CDLL | None" = None
_CBALL_LOADED = False
_CBALL_ERROR: "str | None" = None


def _load_cball():
    global _CBALL, _CBALL_LOADED, _CBALL_ERROR
    if _CBALL_LOADED:
        return _CBALL is not None

    ext_dir = Path(__file__).parent / "cball_ext"
    # Platform-aware preference: on Linux prefer the .so (a Windows .dll
    # copied over will otherwise be picked first and fail to load).
    candidates = (["cball_ext.so", "cball_ext.dll"] if os.name != "nt"
                  else ["cball_ext.dll", "cball_ext.so"])
    lib_path = None
    for name in candidates:
        if (ext_dir / name).exists():
            lib_path = str(ext_dir / name)
            break
    if lib_path is None:
        _CBALL_LOADED = True
        _CBALL_ERROR = (
            f"no compiled extension found in {ext_dir} "
            f"(looked for: {', '.join(candidates)}). Compile it first:\n"
            "  Linux:   cd cball_ext && bash compile_linux.sh   (needs gcc)\n"
            "  Windows: cball_ext\\compile_win.bat"
        )
        return False

    try:
        # Add MinGW runtime paths so DLL dependencies (libgomp, etc.) are found
        _mingw_paths = [
            r"C:\msys64\ucrt64\bin",
        ]
        for _p in _mingw_paths:
            if _p not in os.environ.get("PATH", ""):
                os.environ["PATH"] = _p + ";" + os.environ.get("PATH", "")
        lib = ctypes.CDLL(lib_path)
        lib.compute_bdd_bias.argtypes = [
            np.ctypeslib.ndpointer(np.float64),
            np.ctypeslib.ndpointer(np.float64),
            ctypes.POINTER(ctypes.c_int),
            ctypes.POINTER(ctypes.c_int),
            ctypes.POINTER(ctypes.c_int),
        ]
        _CBALL = lib
        _CBALL_LOADED = True
        _CBALL_ERROR = None
        return True
    except Exception as e:
        _CBALL_LOADED = True
        _CBALL_ERROR = (
            f"found {lib_path} but ctypes failed to load it: {e!r}\n"
            "Common causes on Linux:\n"
            "  - compiled with -fopenmp but libgomp is not installed "
            "(recompile WITHOUT -fopenmp)\n"
            "  - .so was built for a different OS/arch (recompile on this machine)\n"
            "  - missing runtime dependencies (check with: ldd cball_ext.so)"
        )
        return False


# ============================================================
# HBE (Hall-Buckley-Eagleson) Gamma approximation
# ============================================================


def _center_bdd_matrix(K):
    """Double-centre a kernel matrix."""
    return K - K.mean(axis=1, keepdims=True) - K.mean(axis=0, keepdims=True) + K.mean()


def _hbe(coeff, x):
    """Hall-Buckley-Eagleson: p-value via 3-cumulant Gamma approximation."""
    K1 = np.sum(coeff)
    K2 = 2 * np.sum(coeff ** 2)
    K3 = 8 * np.sum(coeff ** 3)
    nu = 8 * K2 ** 3 / K3 ** 2
    x_trans = np.sqrt(2 * nu / K2) * (x - K1) + nu
    # Survival function directly: ``1 - cdf`` loses precision in the far upper
    # tail and can round a small, non-zero p-value all the way to zero.
    return stats.gamma.sf(x_trans, a=nu / 2, scale=2)


def _bdd_matrix_bias_c(D, weight='constant'):
    """Compute BDD kernel via C extension (cball_ext)."""
    if not _load_cball():
        raise RuntimeError("cball_ext unavailable: " + (_CBALL_ERROR or "unknown reason"))
    n = D.shape[0]
    triu_idx = np.triu_indices(n, k=1)
    dist_vec = D[triu_idx].astype(np.float64, copy=True)
    out_size = n * (n + 1) // 2
    out = np.zeros(out_size, dtype=np.float64)
    weight_map = {'constant': 1, 'probability': 2, 'chisquare': 3, 'rbf': 4}
    wt = ctypes.c_int(weight_map.get(weight, 1))
    n_val = ctypes.c_int(n)
    nth = ctypes.c_int(4 if n > 500 else 1)
    assert _CBALL is not None  # guaranteed by _load_cball() call above
    _CBALL.compute_bdd_bias(out, dist_vec,
                            ctypes.byref(n_val),
                            ctypes.byref(nth),
                            ctypes.byref(wt))
    # ``bdd_matrix_bias`` writes its packed output in row-wise UPPER-triangle
    # order: (0,0), (0,1), ..., (0,n-1), (1,1), ... .  Reading it back with
    # ``tril_indices`` permutes entries instead of merely transposing the matrix,
    # which corrupts the spectrum the HBE limit calibration relies on.
    rows, cols = np.triu_indices(n)
    K = np.zeros((n, n))
    K[rows, cols] = out
    K = K + K.T
    np.fill_diagonal(K, np.diag(K) / 2)
    return K


def diagnose_cball():
    """Print exactly why the C extension is (un)available.

    Run this ON the target machine to see the real cause:
        python -c "from bcov import diagnose_cball; diagnose_cball()"
    """
    global _CBALL_LOADED
    _CBALL_LOADED = False  # force a fresh check
    ok = _load_cball()
    print("cball loaded:", ok)
    print("reason:", _CBALL_ERROR)
    ext_dir = Path(__file__).parent / "cball_ext"
    print("ext_dir:", ext_dir)
    print("ext_dir exists:", ext_dir.exists())
    if ext_dir.exists():
        for p in sorted(ext_dir.iterdir()):
            if p.is_file():
                print("   ", p.name, p.stat().st_size, "bytes")
            else:
                print("   ", p.name, "<dir>")
    return ok


def _bcov_limit_pvalue(DX, DY, weight='constant', stat=None, Kx_c=None):
    """Compute p-value via limit distribution (HBE approximation).

    ``stat`` and ``Kx_c`` allow a caller to supply the observed statistic and the
    centred SNP-side BDD kernel.  Both are otherwise recomputed here, and
    ``Kx_c`` in particular depends only on X -- so a scan over many regions
    should compute it once and pass it in (see :func:`bcov_snp_limit_prep`).
    """
    if Kx_c is None:
        Kx_c = _center_bdd_matrix(_bdd_matrix_bias_c(DX, weight))
    Ky_c = _center_bdd_matrix(_bdd_matrix_bias_c(DY, weight))
    K_joint = Kx_c * Ky_c
    n = DX.shape[0]
    eigenvalues = np.linalg.eigh(K_joint)[0]
    eigenvalues = eigenvalues[eigenvalues > 0] / n
    if stat is None:
        stat = _bcov_from_dm(DX, DY)
    return _hbe(eigenvalues, n * stat)


def _bcov_from_dm(DX, DY):
    """
    Compute BCov^2 from pre-computed squared Euclidean distance matrices.

    Parameters
    ----------
    DX : ndarray, shape (n, n)
        Pairwise Euclidean distances for X.
    DY : ndarray, shape (n, n)
        Pairwise Euclidean distances for Y.

    Returns
    -------
    bcov2 : float
    """
    n = DX.shape[0]
    diff = np.zeros((n, n))

    for i in range(n):
        di = DX[i, :]                     # (n,)
        # AX[j, k] = di[k] <= di[j]
        AX = di[None, :] <= di[:, None]   # (n, n)

        di_y = DY[i, :]
        AY = di_y[None, :] <= di_y[:, None]

        PX = np.mean(AX, axis=1)          # (n,)
        PY = np.mean(AY, axis=1)
        PXY = np.mean(AX & AY, axis=1)

        diff[i, :] = PXY - PX * PY

    return np.mean(diff ** 2)


def bcov(X, Y):
    """
    Compute the empirical Ball Covariance statistic BCov^2_n(X, Y).

    Parameters
    ----------
    X : ndarray, shape (n, p)
    Y : ndarray, shape (n, q)

    Returns
    -------
    bcov2 : float
        Empirical squared Ball Covariance.
    """
    X = np.asarray(X, dtype=float)
    Y = np.asarray(Y, dtype=float)
    n = X.shape[0]

    # Euclidean distance matrices
    DX = np.linalg.norm(X[:, None, :] - X[None, :, :], axis=-1)
    DY = np.linalg.norm(Y[:, None, :] - Y[None, :, :], axis=-1)

    return _bcov_from_dm(DX, DY)


def bcov_perm_test(x, y, n_perm=199, method='gamma', DY = None):
    """
    Ball Covariance permutation test.

    Parameters
    ----------
    x : ndarray, shape (n, p)
    y : ndarray, shape (n, q)
    n_perm : int
        Number of permutations (default 199).
    method : str
        ``'permu'`` — permutation p-value (default, matches original behaviour).
        ``'gamma'`` — Gamma approximation p-value using moment-matched Gamma
        distribution fitted to the permutation null distribution.
    DY: Distance of Y, can insert to avoid repeated calculation
    Returns
    -------
    stat : float
        Observed BCov^2 statistic.
    pval : float
        Permutation p-value (method='permu') or Gamma-approximation p-value
        (method='gamma').
    cancel: gamma_fit : dict, optional (only when method='gamma')
        Gamma parameters {'shape', 'scale', 'mean', 'variance'}.
    """
    n = x.shape[0]
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    # Precompute DY (stays the same for permutations)
    if DY is None:
        DY = np.linalg.norm(y[:, None, :] - y[None, :, :], axis=-1)
    else: 
        DY = DY
    # Precompute full DX
    DX = np.linalg.norm(x[:, None, :] - x[None, :, :], axis=-1)

    stat = _bcov_from_dm(DX, DY)
    perm_stats = np.zeros(n_perm)

    for b in range(n_perm):
        perm = np.random.permutation(n)
        DX_perm = DX[np.ix_(perm, perm)]
        perm_stats[b] = _bcov_from_dm(DX_perm, DY)

    if method == 'limit':
        pval = _bcov_limit_pvalue(DX, DY, weight='constant')
        return stat, pval
    elif method == 'gamma':
        from gammafit import gamma_moment_match
        gamma_fit = gamma_moment_match(perm_stats)
        alpha = gamma_fit['shape']
        theta = gamma_fit['scale']
        pval = 1.0 - stats.gamma.cdf(stat, a=alpha, scale=theta)
        return stat, pval
    else:
        pval = (np.sum(perm_stats >= stat) + 1.0) / (n_perm + 1.0)
        return stat, pval


def bcov_perm_test_gwas(x, DY, n_perm=199, method='gamma'):
    """
    Ball Covariance permutation test.

    Parameters
    ----------
    x : ndarray, shape (n, p)
    n_perm : int
        Number of permutations (default 199).
    method : str
        ``'permu'`` — permutation p-value (default, matches original behaviour).
        ``'gamma'`` — Gamma approximation p-value using moment-matched Gamma
        distribution fitted to the permutation null distribution.
    DY: Distance of Y, can insert to avoid repeated calculation
    Returns
    -------
    stat : float
        Observed BCov^2 statistic.
    pval : float
        Permutation p-value (method='permu') or Gamma-approximation p-value
        (method='gamma').
    cancel: gamma_fit : dict, optional (only when method='gamma')
        Gamma parameters {'shape', 'scale', 'mean', 'variance'}.
    """
    n = x.shape[0]
    x = np.asarray(x, dtype=float)
    # y = np.asarray(y, dtype=float)

    # Precompute DY (stays the same for permutations)
    # if DY is None:
        # DY = np.linalg.norm(y[:, None, :] - y[None, :, :], axis=-1)
    # else: 
        # DY = DY
    # Precompute full DX
    DX = np.linalg.norm(x[:, None, :] - x[None, :, :], axis=-1)

    stat = _bcov_from_dm(DX, DY)
    perm_stats = np.zeros(n_perm)

    for b in range(n_perm):
        perm = np.random.permutation(n)
        DX_perm = DX[np.ix_(perm, perm)]
        perm_stats[b] = _bcov_from_dm(DX_perm, DY)

    if method == 'limit':
        pval = _bcov_limit_pvalue(DX, DY, weight='constant')
        return stat, pval
    elif method == 'gamma':
        from gammafit import gamma_moment_match
        gamma_fit = gamma_moment_match(perm_stats)
        alpha = gamma_fit['shape']
        theta = gamma_fit['scale']
        pval = 1.0 - stats.gamma.cdf(stat, a=alpha, scale=theta)
        return stat, pval
    else:
        pval = (np.sum(perm_stats >= stat) + 1.0) / (n_perm + 1.0)
        return stat, pval


# ===========================================================================
#  SNP-specialised Ball Covariance
# ===========================================================================
#
# Verified geometry of ``_bcov_from_dm`` (established by directly probing its
# arrays): for each centre i it builds
#
#     AX[j, k] = 1{ d_X(i,k) <= d_X(i,j) },  AY[j, k] = 1{ d_Y(i,k) <= d_Y(i,j) }
#
# and averages over k.  With d_X(i,k) = |x_i - x_k| that is
#
#     p_x (i,j) = #{ k : |x_k - x_i| <= |x_j - x_i| } / n
#     p_y (i,j) = #{ k : d_Y(i,k)  <= d_Y(i,j)      } / n  = RY[i, j] / n
#     p_xy(i,j) = #{ k : BOTH hold } / n
#
# and
#
#     BCov^2_n = (1/n^2) * sum_{i,j} ( p_xy(i,j) - p_x(i,j) * p_y(i,j) )^2 .
#
# Comparison convention: the membership is ``<=`` (inclusive), which is what
# ``_bcov_from_dm`` implements.  Strict ``<`` gives a different statistic
# (~20% relative difference on the same data), so ``<=`` is used throughout.
#
# ``bcov_snp`` reproduces ``bcov_perm_test_gwas`` bit-for-bit (validated at
# ~1e-16 relative error) while avoiding the n x n distance matrix of X and all
# n x n boolean temporaries.  A compiled C kernel of the same algorithm is
# loaded when available; see ``compile_snp_linux.sh`` / ``compile_snp_win.bat``.


# ---------------------------------------------------------------------------
# Compiled SNP kernel (cball_ext/csnpbcov.so | csnpbcov.dll)
# ---------------------------------------------------------------------------
_CSNP = None
_CSNP_LOADED = False


def _load_csnp():
    """Load the SNP-specialised kernel; return True when usable.

    The library is optional: when it is missing or fails to load, callers fall
    back to the pure-numpy path automatically.
    """
    global _CSNP, _CSNP_LOADED
    if _CSNP_LOADED:
        return _CSNP is not None

    ext_dir = Path(__file__).parent / "cball_ext"
    candidates = (["csnpbcov.so", "csnpbcov.dll"] if os.name != "nt"
                  else ["csnpbcov.dll", "csnpbcov.so"])
    lib_path = None
    for name in candidates:
        if (ext_dir / name).exists():
            lib_path = str(ext_dir / name)
            break

    if lib_path is None:
        _CSNP_LOADED = True
        return False

    try:
        lib = ctypes.CDLL(lib_path)
        i32p = np.ctypeslib.ndpointer(np.int32, flags="C_CONTIGUOUS")
        i64p = np.ctypeslib.ndpointer(np.int64, flags="C_CONTIGUOUS")
        f64p = np.ctypeslib.ndpointer(np.float64, flags="C_CONTIGUOUS")
        lib.snp_bcov_stat.argtypes = [i32p, i32p, i64p, i64p,
                                      ctypes.c_int32, ctypes.c_int32,
                                      ctypes.c_void_p]
        lib.snp_bcov_stat.restype = ctypes.c_double
        # RY builder: reads the feature matrix directly, never an n x n distance
        # matrix.  RY is int32 here; the caller widens it once if needed.
        lib.snp_build_rank.argtypes = [i32p, f64p,
                                       ctypes.c_int32, ctypes.c_int32]
        lib.snp_build_rank.restype = ctypes.c_int32
        lib.snp_bcov_version.restype = ctypes.c_int32
        if lib.snp_bcov_version() != 2:
            raise RuntimeError("unexpected csnpbcov kernel version "
                               f"{lib.snp_bcov_version()}, expected 2")
        _CSNP = lib
        _CSNP_LOADED = True
        return True
    except Exception:
        _CSNP_LOADED = True
        return False


def build_rank_matrix(FEAT, backend=None, dtype=np.int64):
    """Image rank matrix ``RY[i, j] = #{k : d(i,k) <= d(i,j)}`` for one region.

    ``FEAT`` is the region's feature matrix, shape ``(n, L)``.

    With ``backend='c'`` the kernel computes the distances row by row inside the
    loop, so the ``n x n`` float64 distance matrix is **never materialised**:
    peak extra memory is O(n) instead of O(n^2).  With ``backend='numpy'`` the
    distance matrix is built explicitly, which is simpler but costs ``8*n^2``
    bytes (763 MiB at n=10000).

    Both paths use the max-rank tie convention implied by the inclusive ``<=``
    ball membership.
    """
    FEAT = np.ascontiguousarray(FEAT, dtype=np.float64)
    n, L = FEAT.shape
    if backend is None:
        backend = 'c' if _load_csnp() else 'numpy'

    if backend == 'c':
        if not _load_csnp():
            raise RuntimeError(
                "SNP kernel unavailable: build cball_ext/csnpbcov.so "
                "(bash compile_snp_linux.sh) or csnpbcov.dll "
                "(compile_snp_win.bat), or use backend='numpy'")
        out = np.empty((n, n), dtype=np.int32)
        rc = _CSNP.snp_build_rank(out, FEAT, np.int32(n), np.int32(L))
        if rc != 0:
            raise RuntimeError(f"snp_build_rank failed with code {rc}")
        return out.astype(dtype) if dtype != np.int32 else out

    DY = np.linalg.norm(FEAT[:, None, :] - FEAT[None, :, :], axis=-1)
    R = _rank_rows_leq(DY)
    return R.astype(dtype) if R.dtype != dtype else R


def make_permutations(n, n_perm, seed=None):
    """Shared permutation index vectors, shape ``(n_perm, n)`` int32.

    Reusing ONE set of permutations for every region of a scan is what lets the
    permutation null distribution be computed once per SNP instead of once per
    (SNP, region) pair; see :func:`bcov_snp_null`.
    """
    rng = np.random.default_rng(seed) if seed is not None else np.random
    out = np.empty((n_perm, n), dtype=np.int32)
    for b in range(n_perm):
        out[b] = rng.permutation(n) if seed is not None \
            else np.random.permutation(n)
    return out


def _csnp_stats(code, RY, n_lev, n_perm, perms=None):
    """Observed statistic plus `n_perm` replicates, via the C kernel.

    Returns ``(stat, perm_stats)``.  When ``perms`` is given it must be an
    ``(n_perm, n)`` int32 array of index vectors and is used verbatim, which is
    how one permutation set is shared across every region of a scan.
    """
    n = RY.shape[0]
    if not _load_csnp():
        raise RuntimeError(
            "SNP kernel unavailable: build cball_ext/csnpbcov.so "
            "(bash compile_snp_linux.sh) or csnpbcov.dll "
            "(compile_snp_win.bat), or use backend='numpy'")
    i32 = np.int32
    code32 = np.ascontiguousarray(code, dtype=i32)
    class_n = np.ascontiguousarray(
        np.bincount(code, minlength=n_lev), dtype=np.int64)
    # keep int64 in place: RY is cached across SNPs, so re-widening it here
    # would allocate a second n x n array on every call
    RY64 = RY if RY.dtype == np.int64 else np.ascontiguousarray(RY,
                                                                dtype=np.int64)
    n32, lev32 = i32(n), i32(n_lev)

    # The kernel reads ridx as an index vector: cp[i] = code[ridx[i]].  For the
    # observed statistic that must be the IDENTITY permutation, so an explicit
    # arange is passed -- passing `code` here would compute code[code[i]].
    ident = np.arange(n, dtype=i32)
    stat = float(_CSNP.snp_bcov_stat(code32, ident, RY64, class_n,
                                     n32, lev32, None))

    perm_stats = np.empty(n_perm, dtype=np.float64)
    for b in range(n_perm):
        if perms is None:
            p = np.ascontiguousarray(np.random.permutation(n), dtype=i32)
        else:
            p = np.ascontiguousarray(perms[b], dtype=i32)
        perm_stats[b] = _CSNP.snp_bcov_stat(code32, p, RY64, class_n,
                                            n32, lev32, None)
    return stat, perm_stats


def _bcov_snp_null(code, RY, n_lev, perms):
    """Permutation null statistics on the numpy backend.

    ``perms`` is a ``(n_perm, n)`` int32 array of shared index vectors: the
    permutation is applied to the dosage codes only, exactly as the generic
    test does with ``DX[np.ix_(p, p)]`` while keeping ``DY`` fixed.
    """
    out = np.empty(perms.shape[0], dtype=np.float64)
    for b in range(perms.shape[0]):
        out[b] = _bcov_snp_from_ranks(code[perms[b]], RY, n_lev)
    return out


def snp_backend_available():
    """True when the compiled SNP kernel is loadable on this machine."""
    return _load_csnp()


def _rank_rows_leq(D):
    """Ry[i, j] = #{k : D[i, k] <= D[i, j]}  (within-ball count, max-rank ties).

    Row-wise maximum-rank convention: entries tied at the same distance all
    receive the count of the whole tie group, which is precisely what the
    ``<=`` ball membership means.  Ties are handled exactly, and the count is
    taken by binary search on the sorted row so no fragile tie merging is needed.
    """
    n = D.shape[0]
    R = np.empty((n, n), dtype=np.int32)
    for i in range(n):
        row = D[i]
        srt = np.sort(row, kind="stable")
        # right side: first position with srt > row[j], i.e. one past the tie run
        R[i] = np.searchsorted(srt, row, side="right")
    return R


def _snp_codes(x):
    """Dosage level of every sample: ``code[i]`` in ``0 .. n_lev-1``.

    The dosage levels are discovered from the data (``np.unique``), so no
    particular numeric encoding is assumed beyond the caller's guarantee that
    the SNP takes the values {0, 1, 2}.  This small integer vector is the only
    SNP-side data the statistic needs -- no n x n distance matrix is built.
    """
    _, code = np.unique(x, return_inverse=True)
    return code.reshape(-1)


def _bcov_snp_from_ranks(code, RY, n_lev):
    """BCov^2_n for a SNP predictor and a pre-computed image rank matrix.

    Per centre i this accumulates ``sum_j (p_xy - p_x * p_y)^2`` row by row, so
    only O(n) working memory is used beyond ``RY`` itself and no n x n boolean
    or distance matrix for X is ever materialised.

    Parameters
    ----------
    code : ndarray, shape (n,), int
        Dosage level of every sample (see :func:`_snp_codes`).
    RY : ndarray, shape (n, n), int
        ``RY[i, j] = #{k : d_Y(i,k) <= d_Y(i,j)}`` (see :func:`_rank_rows_leq`).
    n_lev : int
        Number of distinct dosage levels.
    """
    n = RY.shape[0]
    inv = 1.0 / n
    R64 = RY.astype(np.int64)
    lev_cum = np.concatenate(([0], np.cumsum(np.bincount(code,
                                                         minlength=n_lev))))
    ks = np.arange(n_lev)

    # range-sum matrix over dosage classes: jcum[j, u] = #{k : lo_j <= code[k] < u}
    total = 0.0
    for i in range(n):
        r = R64[i]                                   # within-ball count of j
        d = np.abs(code - code[i])                   # ball radius of each j
        lo = np.clip(code[i] - d, 0, n_lev - 1)
        hi = np.clip(code[i] + d, 0, n_lev - 1)

        # p_x: size of the X ball, straight from the dosage class counts
        px = (lev_cum[hi + 1] - lev_cum[lo]) * inv

        # p_xy: count of k with code[k] in [lo_j, hi_j] and r_k <= r_j.
        # Bucket k by dosage class, cumulative-sum over the rank, then the count
        # for pair (i, j) is a range sum over classes at column r_j.
        hist = np.zeros((n_lev, n + 1), dtype=np.int64)
        np.add.at(hist, (code, r), 1)
        np.cumsum(hist, axis=1, out=hist)     # hist[u, s] = #(class u, rank <= s)
        # jcum[j, u] with jcum[j, 0] = 0 and jcum[j, u] = sum_{u' < u} hist[u', r_j]
        jcum = np.zeros((n, n_lev + 1), dtype=np.int64)
        np.cumsum(hist[:, r].T, axis=1, out=jcum[:, 1:])
        j = np.arange(n)
        pxy = (jcum[j, hi + 1] - jcum[j, lo]) * inv

        s = pxy - px * (r * inv)
        total += float(np.dot(s, s))
    return total * inv * inv


def bcov_snp(X, DY, n_perm=9, method='gamma', RY=None, backend=None,
             perms=None):
    """
    SNP-specialised Ball Covariance permutation test.

    Drop-in counterpart of :func:`bcov_perm_test_gwas` for a univariate SNP
    predictor ``X`` with dosage values in {0, 1, 2}.  It returns the same Ball
    Covariance statistic and the same moment-matched Gamma p-value (default, as
    in :func:`bcov_perm_test_gwas`), but evaluates the statistic without ever
    building the n x n distance matrix of ``X`` or any n x n boolean temporary,
    and never recomputes the image side: ``DY`` / ``RY`` are supplied by the
    caller and are shared across every SNP of a scan.

    Parameters
    ----------
    X : ndarray, shape (n,) or (n, 1)
        SNP dosages.  Must take at most three distinct values (0, 1, 2).  This
        is **not** validated -- the caller guarantees it.
    DY : ndarray, shape (n, n), optional
        Pre-computed pairwise distance matrix of the image features for this
        sub-region.  Only used to derive ``RY`` when ``RY`` is not supplied;
        pass ``None`` when ``RY`` is already cached (the statistic itself never
        touches ``DY``).
    n_perm : int, optional
        Permutations used for the null distribution (default 9).
    method : {'gamma', 'permu', 'limit'}, optional
        ``'gamma'`` (default) -- moment-matched Gamma approximation fitted to
        the permutation null, as in :func:`bcov_perm_test_gwas`.
        ``'permu'`` -- permutation p-value ``(1 + #{null >= stat}) /
        (n_perm + 1)``.
        ``'limit'`` -- HBE limit-distribution p-value (no permutations at all).
        It needs the SNP distance matrix and the BDD kernels, so it requires the
        ``cball_ext`` Ball kernel; it is also O(n^3) because of the spectral
        decomposition, so it only pays off when ``n_perm`` is large (at
        n = 10000 one limit evaluation costs roughly 10-50 permutations).
    RY : ndarray, shape (n, n), optional
        Pre-computed ``_rank_rows_leq(DY)``.  Supplying it avoids re-ranking the
        image for every SNP.
    backend : {None, 'c', 'numpy'}, optional
        ``'c'`` uses the compiled kernel (``cball_ext/csnpbcov.{so,dll}``, built
        by ``compile_snp_linux.sh`` / ``compile_snp_win.bat``); ``'numpy'`` uses
        the vectorised fallback.  ``None`` (default) picks ``'c'`` when the
        kernel is available and falls back to ``'numpy'`` otherwise.
    perms : ndarray, shape (n_perm, n), optional
        Pre-drawn permutation index vectors (see :func:`make_permutations`).
        Supplying them lets one permutation set be shared by every region of a
        scan, which is what makes :func:`bcov_snp_null` reusable across regions.

    Returns
    -------
    stat : float
        Observed BCov^2 statistic.
    pval : float
        Gamma-approximation (default) or permutation p-value.
    """
    x = np.asarray(X, dtype=float).reshape(-1)
    n = x.shape[0]

    if method == 'limit':
        # The limit law needs no permutations, so RY is not required at all.
        return bcov_snp_limit(x, DY=DY, RY=RY, weight='constant')

    if RY is None:
        if DY is None:
            raise ValueError("bcov_snp needs either DY or RY")
        RY = _rank_rows_leq(DY)

    code = _snp_codes(x)
    n_lev = int(code.max()) + 1

    if backend is None:
        backend = 'c' if _load_csnp() else 'numpy'

    if perms is None:
        perms = make_permutations(n, n_perm)

    if backend == 'c':
        stat, perm_stats = _csnp_stats(code, RY, n_lev, n_perm, perms=perms)
    else:
        stat = _bcov_snp_from_ranks(code, RY, n_lev)
        perm_stats = _bcov_snp_null(code, RY, n_lev, perms)

    if method == 'gamma':
        from gammafit import gamma_moment_match
        gamma_fit = gamma_moment_match(perm_stats)
        pval = 1.0 - stats.gamma.cdf(stat, a=gamma_fit['shape'],
                                     scale=gamma_fit['scale'])
    else:
        pval = (np.sum(perm_stats >= stat) + 1.0) / (n_perm + 1.0)
    return stat, pval


def bcov_snp_stat(X, RY, backend=None):
    """Observed BCov^2 statistic only -- no permutations.

    This is the cheap pass used for pre-screening: ranking all SNPs by their raw
    statistic costs one kernel call per (SNP, region) instead of ``n_perm + 1``.
    """
    x = np.asarray(X, dtype=float).reshape(-1)
    code = _snp_codes(x)
    n_lev = int(code.max()) + 1
    if backend is None:
        backend = 'c' if _load_csnp() else 'numpy'

    if backend == 'c':
        n = x.shape[0]
        code32 = np.ascontiguousarray(code, dtype=np.int32)
        class_n = np.ascontiguousarray(np.bincount(code, minlength=n_lev),
                                       dtype=np.int64)
        RY64 = RY if RY.dtype == np.int64 else np.ascontiguousarray(
            RY, dtype=np.int64)
        ident = np.arange(n, dtype=np.int32)
        return float(_CSNP.snp_bcov_stat(code32, ident, RY64, class_n,
                                         np.int32(n), np.int32(n_lev), None))
    return _bcov_snp_from_ranks(code, RY, n_lev)


def bcov_snp_null(X, RY, perms, backend=None):
    """Permutation null statistics for one (SNP, region) pair.

    Split out from :func:`bcov_snp` so that a scan can compute the null for a
    SNP once and reuse it for every region: the null distribution depends only
    on ``n``, the region's ``RY`` and the permutation set -- the region enters
    through ``RY``, so for a FIXED region the same permutations give the same
    null for every SNP.  Passing one ``perms`` array to all regions is therefore
    valid, and lets the ``n_perm`` kernel calls be hoisted out of the innermost
    loop.

    Returns
    -------
    ndarray, shape (n_perm,)
    """
    x = np.asarray(X, dtype=float).reshape(-1)
    code = _snp_codes(x)
    n_lev = int(code.max()) + 1
    if backend is None:
        backend = 'c' if _load_csnp() else 'numpy'

    if backend == 'c':
        n = x.shape[0]
        code32 = np.ascontiguousarray(code, dtype=np.int32)
        class_n = np.ascontiguousarray(np.bincount(code, minlength=n_lev),
                                       dtype=np.int64)
        RY64 = RY if RY.dtype == np.int64 else np.ascontiguousarray(
            RY, dtype=np.int64)
        n32, lev32 = np.int32(n), np.int32(n_lev)
        out = np.empty(perms.shape[0], dtype=np.float64)
        for b in range(perms.shape[0]):
            p = np.ascontiguousarray(perms[b], dtype=np.int32)
            out[b] = _CSNP.snp_bcov_stat(code32, p, RY64, class_n,
                                         n32, lev32, None)
        return out
    return _bcov_snp_null(code, RY, n_lev, perms)


def bcov_snp_limit(X, DY=None, RY=None, weight='constant', DX=None,
                   stat=None, Kx_c=None, prep=None):
    """Limit-distribution (HBE) p-value for the SNP Ball Covariance test.

    Uses the asymptotic null law instead of permutations: the eigenvalues of the
    centred joint BDD kernel give the cumulants of

        n * BCov^2_n  ->  sum_i lambda_i Z_i^2 ,

    which is then approximated by a three-cumulant Gamma (Hall-Buckley-Eagleson).
    No permutation is drawn, so this is the method of the original RWSPM pipeline
    (``bcov.test(..., method = "limit")`` in R).

    Parameters
    ----------
    X : ndarray, shape (n,) or (n, 1)
        SNP dosages.
    DY : ndarray, shape (n, n), optional
        Image distance matrix of the region.  Required.
    RY : ndarray, shape (n, n), optional
        Image rank matrix ``build_rank_matrix(FEAT)`` for this region.  When
        given, the statistic is computed by rank counting, which avoids the
        n x n boolean temporaries of ``_bcov_from_dm`` (14.76 s -> 0.017 s at
        n = 1270, and it is what stops several workers from dying with
        MemoryError).  The pipeline already caches RY per region, so pass it.
    weight : str, optional
        BDD weight form, ``'constant'`` by default (as in the article).
    DX : ndarray, shape (n, n), optional
        Pre-computed SNP distance matrix; built from ``X`` when omitted.
    stat : float or None, optional
        Pre-computed BCov^2 statistic for this (SNP, region) pair.  When None it
        is computed here, via ``RY`` when that is supplied (rank counting, which
        avoids the n x n boolean temporaries of ``_bcov_from_dm``) or else with
        the generic kernel.  A non-finite value is rejected: the limit p-value is
        a function of the statistic, so there is no meaningful "skip it".
    Kx_c : ndarray, shape (n, n), optional
        Pre-computed *centred* SNP-side BDD kernel.  This is a property of the
        SNP alone, so a scan should build it once and pass it for every region
        instead of letting ``_bcov_limit_pvalue`` rebuild it each time.
    prep : dict, optional
        Result of :func:`bcov_snp_limit_prep`; supplies ``DX`` and ``Kx_c``.

    Returns
    -------
    stat : float
        Observed BCov^2 statistic (``nan`` when the statistic was skipped).
    pval : float
        HBE limit-distribution p-value.

    Notes
    -----
    Requires the Ball BDD kernel (``cball_ext/cball_ext.{so,dll}``, built by
    ``compile_linux.sh`` / ``compile_win.bat``).  Cost is dominated by
    ``numpy.linalg.eigh`` on the n x n joint kernel, i.e. O(n^3), which is why
    this is only economical for large ``n_perm``.
    """
    x = np.asarray(X, dtype=float).reshape(-1)
    if DY is None:
        raise ValueError("bcov_snp_limit needs DY (the image distance matrix)")
    DY = np.asarray(DY, dtype=float)

    if prep is not None:
        DX = prep['DX'] if DX is None else DX
        Kx_c = prep['Kx_c'] if Kx_c is None else Kx_c
    if DX is None:
        DX = np.abs(x[:, None] - x[None, :])

    if stat is None:
        if RY is not None:
            stat = float(bcov_snp_stat(x, RY, backend='c' if _load_csnp()
                                       else 'numpy'))
        else:
            stat = _bcov_from_dm(DX, DY)
    if not np.isfinite(stat):
        raise ValueError("bcov_snp_limit needs a finite statistic; pass "
                         "stat=<BCov^2> or leave it None to compute it")

    pval = _bcov_limit_pvalue(DX, DY, weight=weight, stat=stat, Kx_c=Kx_c)
    return float(stat), pval


def bcov_snp_limit_prep(X, FEAT=None, weight='constant'):
    """Per-SNP ingredients of the limit test, to be reused across all regions.

    The SNP side of the limit law depends only on the SNP, so its distance matrix
    and its centred BDD kernel are the same for every region.  Building them once
    per SNP instead of once per (SNP, region) removes 1487 redundant kernel
    constructions per SNP (measured: 4.2 s each at n = 1270).

    Parameters
    ----------
    X : ndarray, shape (n,) or (n, 1)
        SNP dosages.
    FEAT : ndarray, shape (n, L), optional
        Region feature matrix.  Supplying it lets the returned ``'rank_fn'``
        build the region rank matrix with the C kernel, which needs no n x n
        distance matrix; without it the caller must supply ``RY`` to
        :func:`bcov_snp_limit` itself.
    weight : str, optional
        BDD weight form (default ``'constant'``).

    Returns
    -------
    dict with

        ``DX``      n x n float64, SNP distance matrix
        ``Kx_c``    n x n float64, centred SNP-side BDD kernel
        ``weight``  the weight form used
        ``rank_fn`` callable(DY) -> RY, or None when ``FEAT`` was not given
    """
    x = np.asarray(X, dtype=float).reshape(-1)
    DX = np.abs(x[:, None] - x[None, :])

    rank_fn = None
    if FEAT is not None:
        FEAT = np.ascontiguousarray(FEAT, dtype=np.float64)
        # For a fixed region the image side is fixed, so one call supplies both
        # the ranked image matrix and its distance matrix.  The image-side BDD
        # kernel is deliberately NOT returned: ``_bcov_limit_pvalue`` builds it
        # from DY, and exposing it here invited the mistake of feeding it back
        # in as the SNP-side kernel, which silently squares the image kernel in
        # the joint spectrum.
        _backend = 'c' if _load_csnp() else 'numpy'

        def rank_fn(_FEAT=FEAT, _backend=_backend):
            return {'DY': np.linalg.norm(
                        _FEAT[:, None, :] - _FEAT[None, :, :], axis=-1),
                    'RY': build_rank_matrix(_FEAT, backend=_backend,
                                            dtype=np.int64)}

    return {'DX': DX,
            'Kx_c': _center_bdd_matrix(_bdd_matrix_bias_c(DX, weight)),
            'weight': weight,
            'rank_fn': rank_fn}


def bcov_perm_test_snp_gwas(x, DY, n_perm=9, method='gamma', RY=None,
                            backend=None, perms=None):
    """
    Signature-compatible alias of :func:`bcov_snp`.

    Lets a caller switch between the generic :func:`bcov_perm_test_gwas` and the
    SNP-specialised test by changing only the imported name.
    """
    return bcov_snp(x, DY, n_perm=n_perm, method=method, RY=RY,
                    backend=backend, perms=perms)


if __name__ == "__main__":
    from seed import set_seed
    import time 
    set_seed(123)
    n = 1000
    B = 100
    X_test = np.random.randn(n, 2)
    Y_test = np.random.randn(n, 2)
    print(f"bcov (n=10): {bcov(X_test, Y_test):.6f}")
    print(f"n * bcov:    {n * bcov(X_test, Y_test):.6f}")
    t0 = time.time()
    stat, pval = bcov_perm_test(X_test, Y_test, n_perm=200, method='gamma')
    elapsed = time.time() - t0
    print(f"perm test: stat={stat:.6f}, pval={pval:.4f}, time={elapsed:.4f}s") #method=gamma, 52.5532s
    # t1 = time.time()
    # stat, pval = bcov_perm_test(X_test, Y_test, n_perm=200,method='permu')
    # elapsed1 = time.time() - t1
    # print(f"perm test: stat={stat:.6f}, pval={pval:.4f}, time={elapsed1:.4f}s") #method=permu, 314.0065s
    t2 = time.time()
    stat_l, pval_l = bcov_perm_test(X_test, Y_test, method='limit')
    elapsed2 = time.time() - t2
    print(f"limit test: stat={stat_l:.6f}, pval={pval_l:.4f}, time={elapsed2:.4f}s") #method=limit, 592.7041s


