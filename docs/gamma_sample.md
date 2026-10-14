# 固定分层子样本 Gamma / Fixed stratified-subsample Gamma

## 简体中文

`method=gamma_sample` 是第四种、实验性的 p 值方法，不改变原 `gamma`、`permu`、`limit` 路径。参数 `ratio` 默认0.1，仅对新方法生效。

对每个 SNP，以 `perm_seed + snp_idx` 为种子抽取一次固定子样本；各区域复用相同样本索引。抽样无放回，按原样本中存在的0/1/2类别分配整数配额，每个非空类别至少抽取一个；不存在的类别不补造。样本量为 `min(n, max(ceil(ratio*n), 非空类别数, 2))`。整数配额接近原比例但不保证严格相等，计划中记录类别人数及比例误差。NaN、-9等缺失值需先显式清理，不自动插补。

每区域仍在全样本上计算观测 `BCov_n²`，固定子样本内置换 SNP，影像不置换。全样本排名子矩阵先重新计算子样本内最大秩，不能直接传给 C 内核。

- Gamma拟合输入：`m * BCov_m²` 的置换统计量。
- 上尾评价输入：`n * BCov_n²`，使用 `scipy.stats.gamma.sf()`。
- `ratio=1` 使用全样本排名，不再子抽样；与完整置换Gamma的sf版本数学等价。
- 子样本零分布方差为零时，回退全样本置换；若全样本零分布也退化，则用经验置换p值，记录原因，不拟合无定义Gamma。
- 单态SNP返回统计量0、p值1，记录`constant_snp`。

默认所有区域共用子样本置换。`--no_share_perms` 保持同一个SNP子样本，但每区域使用不同的确定性置换种子。抽样不依赖PID或worker任务分配。

```bash
python GWAS_left.py --method gamma_sample --ratio 0.1 --n_perm 199 --n_process 2 --n_snp 1
python GWAS_right.py --method gamma_sample --ratio 0.1 --n_perm 199 --n_process 2 --n_snp 1
python smoke_test_GWAS.py --method gamma_sample --ratio 0.1 --backend c --n_subj 60 --n_perm 199 --n_process 2
python test_gamma_sample.py
```

每个SNP输出抽样计划`.npz/.json`及分区域诊断`.json`，记录requested/effective ratio、类别配额、样本与校准样本量、Gamma矩和回退原因。已有区域p值及MTCCT汇总继续输出。C源码不需要修改。

固定子样本仅估计其条件零分布，外推到全样本依赖渐近尺度及代表性；罕见类别被强制保留可能改变比例。冒烟和一致性验证不证明极端尾部、Ⅰ类错误率或最终RWSPM校准。全样本排名/统计量仍有成本，不能把ratio²当作全流程加速比。

## English

`gamma_sample` is a fourth, experimental method. Existing `gamma`, `permu` and
`limit` paths are preserved. `ratio=0.1` applies only to the new method.

One fixed, without-replacement stratified sample is drawn per SNP with seed
`perm_seed + snp_idx`. Every present 0/1/2 class receives at least one member;
absent classes are not invented. The sample size is
`min(n, max(ceil(ratio*n), number_of_present_classes, 2))`. Integer quotas
approximate the original proportions; exact equality is generally impossible.
Missing values such as NaN or -9 require explicit preprocessing.

The observed statistic remains full-sample. Full ranks restricted to the subset
are re-ranked with maximum-rank ties before C evaluation. A Gamma is fitted to
`m * BCov_m²` permutation replicates, then evaluated at `n * BCov_n²` with `sf`.
At ratio=1 no subsampling/re-ranking is needed. Zero subset variance triggers a
full-permutation fallback; an invariant full null uses an empirical permutation
p-value with a recorded reason. Constant SNPs return statistic 0 and p-value 1.

Regions share the sample and, by default, permutations. `--no_share_perms` keeps
the same SNP sample but assigns deterministic region-specific permutation seeds.
Plan and diagnostic files record class counts, proportion errors, effective
calibration size and fallback reasons. No C source changes are required.

This estimates a conditional subset null and transfers it approximately to the
full-sample scale. Numerical/smoke tests do not establish extreme-tail or RWSPM
calibration. Full-sample rank/statistic costs remain.
