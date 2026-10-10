/*
 * snp_bcov.c -- SNP-specialised Ball Covariance kernels.
 *
 * Two kernels, both for a SNP predictor with a few dosage levels:
 *
 *   1. snp_bcov_stat / snp_bcov_perm
 *        BCov^2_n = (1/n^2) * sum_{i,j} ( p_xy - p_x * p_y )^2
 *      with inclusive balls (matching the reference `_bcov_from_dm`):
 *        p_x (i,j) = #{ k : |x_k - x_i| <= |x_j - x_i| } / n
 *        p_y (i,j) = RY[i, j] / n
 *        p_xy(i,j) = #{ k : both } / n
 *      O(n) working memory, O(n^2) time.  Only X is permuted between
 *      replicates, so the image rank matrix RY is passed in and never permuted.
 *
 *   2. snp_build_rank
 *        Builds RY straight from the region feature matrix, one row at a time:
 *        for each centre i it computes the n distances, sorts them and writes
 *        the within-ball counts.  The n x n DISTANCE matrix is never
 *        materialised -- working memory is O(n) beyond the output RY -- which
 *        is what makes large n feasible in RAM.
 *      RY[i*n + j] = #{ k : d(i,k) <= d(i,j) }   (max-rank tie convention)
 *      Distances are compared SQUARED (monotone, so the ranking is identical)
 *      to avoid n^2 square roots.
 *
 * Design notes
 * ------------
 * * Single threaded by design: the caller parallelises over regions and SNPs
 *   (one process per core), so nested threading would only oversubscribe.
 * * No OpenMP, no Python headers, no external BLAS: plain C99 + libm only.
 *
 * Build (Linux):            gcc -O3 -fPIC -shared -o csnpbcov.so  snp_bcov.c -lm
 * Build (Windows / MinGW):  gcc -O3 -shared        -o csnpbcov.dll snp_bcov.c -lm
 */

#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#if defined(_WIN32) || defined(_WIN64)
#define EXPORT __declspec(dllexport)
#else
#define EXPORT
#endif

/* ------------------------------------------------------------------ */
/* Scratch layout in int64 units                                      */
/*   hist : n_lev * (n + 1)              class-by-rank counts         */
/*   jcum : (n_lev + 1)                  prefix sums over CLASSES     */
/*   cbuf : ceil(n * sizeof(int32) / 8)  permuted dosage codes        */
/* ------------------------------------------------------------------ */
#define SCRATCH_I64(n, n_lev) \
    ((size_t)(n_lev) * ((size_t)(n) + 1) + ((size_t)(n_lev) + 1) \
     + (((size_t)(n) * sizeof(int32_t) + sizeof(int64_t) - 1) / sizeof(int64_t)))

/* ================================================================== */
/*  Sorting helpers for the (key, index) pair arrays                  */
/* ================================================================== */

/* swap key and the accompanying index (used by the partitions) */
static void kswap_i(double *key, int32_t *idx, int32_t a, int32_t b)
{
    double tk = key[a];
    int32_t ti = idx[a];
    key[a] = key[b];
    idx[a] = idx[b];
    key[b] = tk;
    idx[b] = ti;
}

static void insertion_sort(double *key, int32_t *idx, int32_t lo, int32_t hi)
{
    int32_t i, j;
    for (i = lo + 1; i <= hi; i++) {
        double k = key[i];
        int32_t v = idx[i];
        for (j = i - 1; j >= lo && key[j] > k; j--) {
            key[j + 1] = key[j];
            idx[j + 1] = idx[j];
        }
        key[j + 1] = k;
        idx[j + 1] = v;
    }
}

/* Median-of-three pivot selection, done on the KEYS only.
 * key[] is a scratch copy owned by the caller, so shuffling it here cannot
 * affect anything downstream -- the index permutation is rebuilt by the
 * partitions below.  Returns the pivot VALUE (never the index). */
static double pick_pivot(double *key, int32_t lo, int32_t hi)
{
    int32_t mid = lo + ((hi - lo) >> 1);
    double a = key[lo], b = key[mid], c = key[hi], t;
    if (a > b) { t = a; a = b; b = t; }
    if (b > c) { t = b; b = c; c = t; }
    if (a > b) { t = a; a = b; b = t; }
    return b;
}

/* Sort key[] ascending, carrying idx[] along, so that afterwards
 * key[k] is the k-th smallest and idx[k] its ORIGINAL position.
 * Ties are broken by original position, which makes the result
 * deterministic and independent of the algorithm's internals. */
static void argsort(double *key, int32_t *idx, int32_t n, int32_t maxdepth)
{
    int32_t *lo_stk, *hi_stk;
    int32_t sp = 0;

    lo_stk = (int32_t *)malloc((size_t)maxdepth * sizeof(int32_t));
    hi_stk = (int32_t *)malloc((size_t)maxdepth * sizeof(int32_t));
    if (!lo_stk || !hi_stk) {           /* fall back to plain insertion sort */
        insertion_sort(key, idx, 0, n - 1);
        free(lo_stk); free(hi_stk);
        return;
    }

    lo_stk[sp] = 0;
    hi_stk[sp] = n - 1;
    sp++;

    while (sp > 0) {
        int32_t lo, hi, i, j;
        double pivot;

        sp--;
        lo = lo_stk[sp];
        hi = hi_stk[sp];

        if (hi - lo <= 15) {
            if (hi > lo) insertion_sort(key, idx, lo, hi);
            continue;
        }

        pivot = pick_pivot(key, lo, hi);
        i = lo;
        j = hi;
        while (i <= j) {
            while (key[i] < pivot) i++;
            while (key[j] > pivot) j--;
            if (i <= j) {
                kswap_i(key, idx, i, j);
                i++;
                j--;
            }
        }

        /* push the larger side, loop on the smaller: the stack stays shallow */
        if (j - lo < hi - i) {
            if (i < hi && sp < maxdepth) { lo_stk[sp] = i; hi_stk[sp] = hi; sp++; }
            if (lo < j && sp < maxdepth) { lo_stk[sp] = lo; hi_stk[sp] = j; sp++; }
        } else {
            if (lo < j && sp < maxdepth) { lo_stk[sp] = lo; hi_stk[sp] = j; sp++; }
            if (i < hi && sp < maxdepth) { lo_stk[sp] = i; hi_stk[sp] = hi; sp++; }
        }
    }

    free(lo_stk);
    free(hi_stk);
}

/* ================================================================== */
/*  Kernel 1: BCov^2 statistic                                        */
/* ================================================================== */
EXPORT double snp_bcov_stat(const int32_t *code,
                            const int32_t *ridx,
                            const int64_t *RY,
                            const int64_t *class_n,
                            int32_t n,
                            int32_t n_lev,
                            int64_t *work)
{
    int64_t *hist, *jcum, *own = NULL;
    int64_t *class_cum;
    int32_t *cp;
    double total = 0.0;
    const double inv = 1.0 / (double)n;
    const int32_t n1 = n + 1;
    int32_t i, j, u;

    if (!code || !RY || !class_n || n <= 0 || n_lev <= 0) return 0.0;

    if (work) {
        hist = work;
        jcum = hist + (size_t)n_lev * n1;
        cp   = (int32_t *)(jcum + (size_t)n_lev + 1);
    } else {
        own = (int64_t *)malloc(SCRATCH_I64(n, n_lev) * sizeof(int64_t));
        if (!own) return 0.0;
        hist = own;
        jcum = hist + (size_t)n_lev * n1;
        cp   = (int32_t *)(jcum + (size_t)n_lev + 1);
    }

    class_cum = (int64_t *)malloc((size_t)(n_lev + 1) * sizeof(int64_t));
    if (!class_cum) {
        if (!work) free(own);
        return 0.0;
    }
    class_cum[0] = 0;
    for (u = 0; u < n_lev; u++) class_cum[u + 1] = class_cum[u] + class_n[u];

    for (i = 0; i < n; i++) cp[i] = ridx ? code[ridx[i]] : code[i];

    for (i = 0; i < n; i++) {
        const int64_t *Ri = RY + (size_t)i * n;
        const int32_t ci = cp[i];
        double row = 0.0;

        /* hist[u][r] = #{ j : cp[j] == u and RY[i,j] == r }, then cumulate
         * along the RANK axis: p_xy needs "rank <= RY[i,j]", not "== ". */
        memset(hist, 0, (size_t)n_lev * n1 * sizeof(int64_t));
        for (j = 0; j < n; j++) hist[(size_t)cp[j] * n1 + (int32_t)Ri[j]] += 1;
        for (u = 0; u < n_lev; u++) {
            int64_t *hu = hist + (size_t)u * n1;
            int64_t run = 0;
            for (j = 0; j <= n; j++) { run += hu[j]; hu[j] = run; }
        }

        for (j = 0; j < n; j++) {
            const int32_t r = (int32_t)Ri[j];
            const int32_t d = cp[j] - ci;
            const int32_t ad = (d < 0) ? -d : d;
            int32_t lo = ci - ad;
            int32_t hi = ci + ad;
            double px, pxy, s;

            if (lo < 0) lo = 0;
            if (hi > n_lev - 1) hi = n_lev - 1;

            jcum[0] = 0;
            for (u = 0; u < n_lev; u++)
                jcum[u + 1] = jcum[u] + hist[(size_t)u * n1 + r];

            pxy = (double)(jcum[hi + 1] - jcum[lo]) * inv;
            px  = (double)(class_cum[hi + 1] - class_cum[lo]) * inv;
            s = pxy - px * ((double)r * inv);
            row += s * s;
        }
        total += row;
    }

    free(class_cum);
    if (!work) free(own);
    return total * inv * inv;
}

EXPORT void snp_bcov_perm(double *stat_out,
                          const int32_t *code,
                          const int64_t *RY,
                          const int64_t *class_n,
                          int32_t n,
                          int32_t n_lev,
                          const int32_t *perm,
                          int32_t n_perm)
{
    int32_t b;
    stat_out[0] = snp_bcov_stat(code, NULL, RY, class_n, n, n_lev, NULL);
    for (b = 0; b < n_perm; b++)
        stat_out[b + 1] = snp_bcov_stat(code, perm + (size_t)b * n, RY,
                                        class_n, n, n_lev, NULL);
}

/* ================================================================== */
/*  Kernel 2: build the image rank matrix, O(n) working memory        */
/* ================================================================== */

/*
 * Build RY (n x n int32, row major) from the region features FEAT (n x L,
 * row major float64).
 *
 * For every centre i:
 *     d_k = sum_l (FEAT[i*L+l] - FEAT[k*L+l])^2      (squared Euclidean)
 *     sort (d, k) ascending, then
 *     RY[i*n + k] = number of entries equal to d_k counted at the END of its
 *                   tie run, i.e. #{ k' : d(i,k') <= d(i,k) }
 * which is exactly the `<=` ball membership with the max-rank tie convention.
 *
 * Peak extra memory: one double[n] + one int32[n] + a small stack.
 *
 * Returns 0 on success, -1 on bad arguments, -2 on allocation failure.
 */
EXPORT int32_t snp_build_rank(int32_t *RY,
                              const double *FEAT,
                              int32_t n,
                              int32_t L)
{
    double *key;
    int32_t *idx;
    int32_t i, k, l, maxdepth;

    if (!RY || !FEAT || n <= 0 || L <= 0) return -1;

    key = (double *)malloc((size_t)n * sizeof(double));
    idx = (int32_t *)malloc((size_t)n * sizeof(int32_t));
    if (!key || !idx) { free(key); free(idx); return -2; }

    maxdepth = 4 * (int32_t)ceil(log2((double)(n > 1 ? n : 2))) + 16;
    if (maxdepth < 64) maxdepth = 64;

    for (i = 0; i < n; i++) {
        const double *fi = FEAT + (size_t)i * L;
        int32_t *Ri = RY + (size_t)i * n;
        int32_t run;

        for (k = 0; k < n; k++) {
            const double *fk = FEAT + (size_t)k * L;
            double d = 0.0;
            for (l = 0; l < L; l++) {
                double t = fi[l] - fk[l];
                d += t * t;
            }
            key[k] = d;
            idx[k] = k;
        }

        argsort(key, idx, n, maxdepth);

        /* a run of equal keys all receive the count at the end of the run */
        run = 0;
        while (run < n) {
            int32_t end = run;
            double v = key[run];
            while (end + 1 < n && key[end + 1] == v) end++;
            for (k = run; k <= end; k++) Ri[idx[k]] = end + 1;
            run = end + 1;
        }
    }

    free(key);
    free(idx);
    return 0;
}

/* Convenience wrapper: build RY for a batch of regions.
 * FEATS is (n_regions, n, L) row major; RY_OUT is (n_regions, n, n) int32.
 * Returns the index of the first region that failed, or -1 on success. */
EXPORT int32_t snp_build_rank_batch(int32_t *RY_OUT,
                                    const double *FEATS,
                                    int32_t n_regions,
                                    int32_t n,
                                    int32_t L)
{
    int32_t r;
    for (r = 0; r < n_regions; r++) {
        int32_t rc = snp_build_rank(RY_OUT + (size_t)r * n * n,
                                    FEATS + (size_t)r * n * L, n, L);
        if (rc != 0) return r;
    }
    return -1;
}

/* Reported so the Python side can sanity-check the build it loaded. */
EXPORT int32_t snp_bcov_version(void) { return 2; }
