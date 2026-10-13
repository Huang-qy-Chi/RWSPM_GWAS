# ADNI · RWSPM

[![简体中文](docs/language-zh.svg)](README.md)
[![English](docs/language-en.svg)](README.en.md)

Regional-window SNP association analysis for hippocampal images:
**sliding windows → LQD → covariate adjustment → Ball Covariance → RW-MTCCT aggregation**.

This repository packages an experimental source snapshot dated 2026-10-10:
Python pipelines, SNP-specialised C kernels, the BDD extension, benchmarks and
validation tools. ADNI inputs, resampled datasets, experiment outputs, logs and
compiled binaries are excluded. The code is intended for research and
reproduction; performance checks do not establish statistical calibration.

## Contents

- [Installation and compilation](#installation-and-compilation)
- [Quick checks without ADNI](#quick-checks-without-adni)
- [Data contract and GWAS](#data-contract-and-gwas)
- [Large-sample benchmarks](#large-sample-benchmarks)
- [Methods and current limitations](#methods-and-current-limitations)
- [Source layout](#source-layout)

## Installation and compilation

Use Python 3.10+; the server experiments ran with Python 3.12. See
[requirements.txt](requirements.txt). The existing `seed.py` imports `torch`, so
PyTorch is required even without a GPU. `rdata` reads private R data files without
an R interpreter.

```bash
python -m venv .venv
source .venv/bin/activate                 # Windows: .venv\Scripts\activate
python -m pip install -r requirements.txt
```

Prefer **Linux** for large parallel runs. Benchmark monitoring uses `fork`,
`/proc`, and optional cgroup/systemd interfaces. Core code and small tests can run
on Windows, but its `spawn` memory behaviour differs from Linux estimates.

Limit BLAS threads when using process-level parallelism:

```bash
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
```

Build the two distinct shared libraries with GCC:

```bash
bash cball_ext/compile_snp_linux.sh       # csnpbcov.so: SNP statistics and image ranks
bash cball_ext/compile_linux.sh           # cball_ext.so: BDD kernels for limit
python -c "from bcov import snp_backend_available; assert snp_backend_available()"
python -c "from bcov import diagnose_cball; diagnose_cball()"
```

On Windows, use MinGW-w64 and run `compile_snp_win.bat` and `compile_win.bat`
inside `cball_ext`. Do not mix Linux `.so` and Windows `.dll` files, or overwrite
the BDD library with the SNP library. Default builds do not enable OpenMP.

## Quick checks without ADNI

```bash
python test_generate_adni_data.py
python test_mtcct_indexing.py
python smoke_test_GWAS.py --n_subj 30 --n_perm 19 --n_process 2
```

The synthetic smoke test exercises hemisphere wiring, LQD, covariate adjustment,
C statistics, shared permutations and aggregation outputs. Nineteen permutations
are a quick-test setting, not a recommendation for inference. Both Linux C builds
and these checks passed during packaging; no large ADNI experiment was rerun.

## Data contract and GWAS

Obtain authorised data separately and place it beside the scripts. All matrices
must use the same participant order.

| File | Object/format | Shape |
|---|---|---|
| `ADNI_1_3_sampled_25000.npy` | NumPy, SNP dosages 0/1/2 | `(n, p)` |
| `cov_inf_0711_1_3.rdata` | `cov_inf1_3` | `(n, 9)` |
| `left_hipp_ori.rdata` | `left_hipp_inf1_3` | `(n, 15000)` |
| `right_hipp_ori.rdata` | `right_hipp_inf1_3` | `(n, 15000)` |

Defaults are 150×100 images, window width 11, step 3 and 1488 regions. LQD uses
21 points including endpoints 0 and 1. Covariate adjustment uses the projection
implemented in the code. File and object names are defined in `GWAS_left.py`;
the right entry point reuses it and overrides hemisphere settings.

Start with a small SNP/region subset:

```bash
python GWAS_left.py --method gamma --backend c --n_perm 199 \
  --n_snp 1 --max_regions 16 --n_process 2 --no_cache_rank
python GWAS_right.py --method gamma --backend c --n_perm 199 \
  --n_snp 1 --max_regions 16 --n_process 2 --no_cache_rank
python GWAS_left.py --help
```

Permutation indices are shared across regions for each SNP by default; use
`--no_share_perms` for a comparison. Sharing indices does not mean sharing null
statistics across regions or SNPs. CLI options also control caches, memory
budgets, SNP offsets and distance backends. Outputs go to `results_gwas_left/`
and `results_gwas_right/`, which are ignored by Git. Demo defaults use few
permutations; set the count explicitly for research runs.

## Large-sample benchmarks

These commands reproduce **one batch of the first 16 regions on Linux**, not a
complete 1488-region GWAS. The SNP is column 0. Each output directory must be new;
create its parent first.

1. Generate 10000 aligned bootstrap rows:

```bash
mkdir -p runs
python generate_adni_data.py --n_samples 10000 --n_snp 1 --seed 1145 \
  --output_dir simulated_adni_n10000_snp0_seed1145
```

SNPs, covariates and left-hippocampal images use the same resampling index. The
generator records mappings and a manifest and supports read-only memory maps.
Original ADNI files are required. Duplicated participants make this a performance
dataset, not an independent 10000-subject null simulation.

2. Run `limit`, sharing a SNP kernel prepared once in the parent:

```bash
python bench_n10000_batch.py monitor \
  --data_dir simulated_adni_n10000_snp0_seed1145 \
  --output_dir runs/limit --processes 16 --expected_n 10000
```

3. Run Gamma with shared seed-1145 permutation indices and separate regional null
statistics:

```bash
python bench_n10000_gamma_batch.py monitor \
  --data_dir simulated_adni_n10000_snp0_seed1145 \
  --output_dir runs/gamma199 --limit_output runs/limit \
  --processes 16 --expected_n 10000 --n_perm 199

python bench_n10000_gamma_batch.py monitor \
  --data_dir simulated_adni_n10000_snp0_seed1145 \
  --output_dir runs/gamma999 --limit_output runs/limit --prior_output runs/gamma199 \
  --processes 16 --expected_n 10000 --n_perm 999
```

The 999-permutation run verifies the first 199 against the earlier run and
generates parameter/p-value stability and 499-training/500-held-out diagnostics.
Gamma benchmarks use `gamma.sf()`; logarithmic differences use `gamma.logsf()`.

Monitoring records stage wall time, task-tree PSS/RSS at five-second intervals,
available system memory, status and worker results. Completion generates method
comparisons and a single-SNP extrapolation: measured BCov batch wall time ×1488/16,
plus one-time preparation and extrapolated LQD. Full covariate-adjustment peaks,
aggregation and sustained throughput were not measured. Sampled PSS can miss
brief peaks; cgroup peaks are a different measure.

Benchmarks allow 1–24 workers, default 16. Software guards terminate this task's
child processes if available memory falls below 48 GiB or task PSS exceeds
220 GiB. **These Python commands do not install a 240 GiB hard limit or scheduled
notifications.** Configure cgroup/systemd separately if required; the historical
protocol used MemoryMax=240G, MemoryHigh=200G, MemorySwapMax=0, OOMPolicy=stop and
a 48-hour timeout. Those budgets come from a large-memory server and are not
suitable defaults for an ordinary workstation. `check_n10000_batch.py` is a
read-only systemd status tool; pass `--output_dir` and `--unit` explicitly.

## Methods and current limitations

| Method | Computation | Limitation |
|---|---|---|
| `gamma` | Two-parameter Gamma matched to permutation mean/sample variance | Extreme tails are model extrapolations; more permutations do not guarantee a correct distributional fit |
| `permu` | `(1 + exceedances)/(B + 1)` | Resolution is `1/(B+1)`; appropriate exchangeability is required |
| `limit` | SNP/image BDD kernels, joint spectrum, three-cumulant HBE approximation | Expensive construction/decomposition; asymptotic calibration still needs validation |

In the measured n=10000 batch, Gamma199 and Gamma999 were substantially cheaper
than limit. Image BDD construction dominated limit costs, not the observed
statistic. Extreme Gamma p-values were sensitive to estimated moments. Held-out
checks only provide conditional-null evidence; they do not establish extreme
tail or final RWSPM type-I error calibration, or show that limit is more accurate.
Raw experimental results are intentionally excluded from this source package.

The snapshot retains differences between the formal pipeline and benchmarks:

- The formal `GWAS_left.py` limit branch does not yet reuse a parent SNP kernel
  or explicitly pass precomputed `RY/stat`. Use the benchmark scripts above to
  reproduce the optimised large-sample path.
- Some formal Gamma branches in `bcov.py` and `GWAS_left.py` still use
  `1-gamma.cdf()`, whereas benchmarks use `gamma.sf()`. Benchmark tail precision
  is not a claim about the older formal path.
- `cdist` computes distances between regional LQD feature vectors;
  `numpy_blocked` is the compatibility option. LQD itself is not replaced by
  `cdist`. Gamma ranks are built directly by C without a full image distance matrix.
- `LQD()` is serial within a region; GWAS and benchmarks parallelise regions.
  Placeholder p-values in `rwspm.RWSPM()` are not a formal GWAS test.
- Gamma999 requires completed limit and Gamma199 baseline directories. Other
  sizes can be checked with `--expected_n`, but image geometry remains 150×100.

## Source layout

| Files | Purpose |
|---|---|
| `GWAS_left.py`, `GWAS_right.py` | Formal hemisphere GWAS and RW-MTCCT |
| `bcov.py`, `gammafit.py` | BCov, C bindings, permutations, Gamma/HBE |
| `lqd.py`, `rwspm.py`, `seed.py` | LQD, windows/aggregation, seeding |
| `cball_ext/` | C sources, headers and build scripts |
| `generate_adni_data.py`, `test_generate_adni_data.py` | Aligned bootstrap and contract checks |
| `bench_n10000_batch.py`, `bench_n10000_gamma_batch.py` | Large-sample limit/Gamma batch benchmarks |
| `compare_gamma_stability.py` | Nested fits and conditional held-out checks |
| `benchmark_adni44.py`, `bench_large_window.py` | Historical full-region benchmark, monitoring, large-window benchmark |
| `compare_adni44_distance.py`, `summarize_large_window.py` | Analysis of completed resource measurements |
| `compare_rwspm_bcov.py`, `bench_limit_1270.py` | Generic/specialised comparison and cache benchmarks |
| `smoke_test_GWAS.py`, `probe_distance_memory.py`, `validate_generated_adni.py` | Synthetic smoke, distance-memory probe, private-input validation |
| `GWAS_left_simple.py` | Retained simplified comparison entry point |

Historical analysis scripts expect specific output structures, which are not
included. Use `gammafit.py` as a module: its standalone demo references an absent
`permutation_null` module; this does not affect pipeline imports.

## Provenance and third-party notices

[SOURCE_MANIFEST.json](SOURCE_MANIFEST.json) records source checksums and changes.
On 2026-10-13, both MTCCT entry points were fixed to interpret region vectors in
row-major `(width_x, width_y)` order, using `width_y` as the coordinate/linear-index
stride. The default 48×31 grid is no longer interpreted as 31×48. All 225 grid
regression cases and both hemisphere smoke tests passed. BCov statistics, C
kernels, permutations, Gamma fits and CCT weighting rules were unchanged.
Historical archives are preserved; corrected results may differ when the selected
connected component changes.

Ball-related author and GPL notices are preserved. See
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) and
[COPYING.GPL-3.0.txt](COPYING.GPL-3.0.txt). No new blanket license has been assigned
to other original files. Input data remain subject to their own access agreements.
