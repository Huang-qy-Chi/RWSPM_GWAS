# ADNI · RWSPM

[![简体中文](docs/language-zh.svg)](README.md)
[![English](docs/language-en.svg)](README.en.md)

基于海马影像的区域窗口建模与 SNP 关联分析：**滑动窗口 → LQD → 协变量校正 → Ball Covariance → RW-MTCCT 聚合**。

本仓库整理自 2026-10-10 的实验代码快照，包含 Python 流程、SNP 特化 C 内核、BDD C 扩展、性能基准及验证脚本。真实 ADNI 数据、重采样数据、实验结果、日志和编译后的二进制文件均不随仓库提供。代码用于研究与复现；性能验证不等于统计校准验证。

## 目录导航

- [环境与编译](#环境与编译)
- [不依赖 ADNI 的快速验证](#不依赖-adni-的快速验证)
- [数据约定与 GWAS](#数据约定与-gwas)
- [大样本基准复现](#大样本基准复现)
- [方法差异与已知限制](#方法差异与已知限制)
- [代码结构](#代码结构)

## 环境与编译

建议使用 Python 3.10+；服务器运行环境为 Python 3.12。依赖见 [requirements.txt](requirements.txt)。`torch` 是现有 `seed.py` 的直接依赖，即使不使用 GPU 也需要安装；`rdata` 用于读取私有 `.rdata` 输入，不需要 R 解释器。

```bash
python -m venv .venv
source .venv/bin/activate                 # Windows: .venv\Scripts\activate
python -m pip install -r requirements.txt
```

大型并行实验优先使用 **Linux**。基准脚本依赖 `fork`、`/proc` 和可选的 cgroup/systemd；Windows 可运行核心代码与小型测试，但其 `spawn` 内存行为不能直接套用 Linux 的估计。

固定 BLAS 线程，避免进程并行与线程并行叠加：

```bash
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
```

编译两套不同的 C 扩展（需要 GCC）：

```bash
bash cball_ext/compile_snp_linux.sh       # csnpbcov.so：SNP统计量与影像排名
bash cball_ext/compile_linux.sh           # cball_ext.so：BDD核，limit需要
python -c "from bcov import snp_backend_available; assert snp_backend_available()"
python -c "from bcov import diagnose_cball; diagnose_cball()"
```

Windows 使用 MinGW-w64，在 `cball_ext` 目录运行 `compile_snp_win.bat` 和 `compile_win.bat`。不要混用 `.dll` 与 `.so`，也不要把 SNP 内核编译到 BDD 内核的输出文件名。默认构建脚本不启用 OpenMP。

## 不依赖 ADNI 的快速验证

```bash
python test_generate_adni_data.py
python smoke_test_GWAS.py --n_subj 30 --n_perm 19 --n_process 2
```

第二条命令测试左右海马入口、LQD、协变量校正、C统计量、共享置换和聚合输出，使用合成数据。19 次置换仅用于快速测试，不是正式推断的推荐值。小样本冒烟及两套 C 扩展的 Linux 重新编译已通过；没有因此重新执行大型 ADNI 实验。

## 数据约定与 GWAS

自行取得有权使用的数据，并放在脚本所在目录；各矩阵必须按同一受试者顺序对齐。

| 输入文件 | R对象名/格式 | 形状 |
|---|---|---|
| `ADNI_1_3_sampled_25000.npy` | NumPy，SNP剂量0/1/2 | `(n, p)` |
| `cov_inf_0711_1_3.rdata` | `cov_inf1_3` | `(n, 9)` |
| `left_hipp_ori.rdata` | `left_hipp_inf1_3` | `(n, 15000)` |
| `right_hipp_ori.rdata` | `right_hipp_inf1_3` | `(n, 15000)` |

默认影像为 150×100，窗宽11、步长3，得到1488个区域；LQD使用21个点（包含0、1端点）。协变量校正使用代码中的投影矩阵。读取文件名和对象名位于 `GWAS_left.py`，右侧入口复用左侧实现并覆盖影像设置。

先运行少量 SNP、少量区域：

```bash
python GWAS_left.py --method gamma --backend c --n_perm 199 \
  --n_snp 1 --max_regions 16 --n_process 2 --no_cache_rank
python GWAS_right.py --method gamma --backend c --n_perm 199 \
  --n_snp 1 --max_regions 16 --n_process 2 --no_cache_rank
python GWAS_left.py --help
```

默认共享每个 SNP 的置换索引，可用 `--no_share_perms` 作对照；共享的是索引，不是所有区域或所有 SNP 的零统计量。区域缓存、内存预算、SNP起始位置和距离计算选项均可通过 CLI 调整。输出写入 `results_gwas_left/`、`results_gwas_right/`，包含区域 p 值和 SNP 汇总，默认由 `.gitignore` 排除。演示默认置换数较小，正式运行需显式给出。

## 大样本基准复现

以下为 **Linux下首16区域的单批测试**，不等同于完整1488区域GWAS。当前命令对应第0列SNP；输出目录必须不存在，父目录应先创建。

1. 按同一个受试者索引联合重采样，生成10000行：

```bash
mkdir -p runs
python generate_adni_data.py --n_samples 10000 --n_snp 1 --seed 1145 \
  --output_dir simulated_adni_n10000_snp0_seed1145
```

生成器同时重采样 SNP、协变量和左海马影像，保留映射与 manifest，并支持只读内存映射。需要原始 ADNI 文件；这不是独立10000人的模拟零假设数据，重复受试者只用于性能测试。

2. `limit`：父进程仅构造一次 SNP 核，16个worker各测试一个区域：

```bash
python bench_n10000_batch.py monitor \
  --data_dir simulated_adni_n10000_snp0_seed1145 \
  --output_dir runs/limit --processes 16 --expected_n 10000
```

3. `gamma`：各区域共享seed1145的置换索引，各自计算零统计量：

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

999次实验会核验前199次与上次完全一致，并生成参数、p值稳定性及499训练/500留出诊断。两个 Gamma 基准使用 `gamma.sf()`；对数差异使用 `gamma.logsf()`。

`monitor` 记录分阶段墙钟、5秒采样的进程树PSS/RSS、系统可用内存、任务状态与worker结果。完成后自动生成单SNP外推和方法比较：BCov实测批次墙钟×1488/16，另计一次预处理和外推LQD。完整协变量校正峰值、聚合和长期吞吐未实测。采样PSS可能遗漏短暂峰值；cgroup峰值与PSS不能直接等同。

基准仅允许1–24个worker，默认16；软保护为可用内存低于48 GiB或任务PSS高于220 GiB时终止本任务子进程。**启动这些 Python 命令不会自动建立240 GiB硬限制或定时通知**。硬限制需要在外部配置 cgroup/systemd（既有协议：MemoryMax=240G、MemoryHigh=200G、MemorySwapMax=0、OOMPolicy=stop、48小时超时）；这些预算源自大内存测试机，不适合直接套到普通电脑。`check_n10000_batch.py` 为只读systemd状态工具，使用时显式给出 `--output_dir` 与 `--unit`。

## 方法差异与已知限制

| 方法 | 当前计算路径 | 主要限制 |
|---|---|---|
| `gamma` | 置换均值、样本方差拟合二参数Gamma | 极端尾部依赖模型外推，增加置换不保证消除拟合误差 |
| `permu` | `(1 + 超过次数)/(B + 1)` | p值分辨率为`1/(B+1)`；依赖适当的可交换性条件 |
| `limit` | SNP与影像BDD核、联合谱、三累积量HBE近似 | 核构造与分解昂贵，渐近近似仍需要校准 |

实际n=10000单批实验中，Gamma199和Gamma999的计算负担明显低于limit；limit的主要瓶颈为影像BDD核而非统计量。Gamma极小p值对置换矩估计较敏感，留出诊断仅提供条件零分布线索，不能证明其极端尾部或最终RWSPMⅠ类错误率已校准，也不能据p值大小判定limit更准确。原始性能结果不纳入此源码包。

本快照保留了正式入口与基准入口的差异：

- `GWAS_left.py` 的正式 `limit` 分支未整合父进程共享SNP核，也未显式传入已计算的`RY/stat`；大样本优化复现应使用上述基准脚本。
- 正式 `bcov.py` 和 `GWAS_left.py` 的部分Gamma分支仍为 `1-gamma.cdf()`；基准为 `gamma.sf()`。不要将基准的极小p值精度直接归于旧正式路径。
- `cdist` 作用于LQD特征之间的区域距离计算；`numpy_blocked` 是兼容选项。这不是把LQD变换本身改为`cdist`。Gamma排名直接由C内核计算，不需要完整影像距离矩阵。
- 单区域`LQD()`本身是串行函数；GWAS及基准在区域之间并行。`rwspm.RWSPM()`中的占位p值不能替代正式GWAS检验入口。
- Gamma999基准需要已完成的limit和Gamma199目录以作对照；其他样本量可用`--expected_n`调整验证，但默认区域结构仍为150×100。

## 代码结构

| 文件/目录 | 用途 |
|---|---|
| `GWAS_left.py`, `GWAS_right.py` | 正式左右海马GWAS与RW-MTCCT |
| `bcov.py`, `gammafit.py` | BCov、C绑定、置换与Gamma/HBE计算 |
| `lqd.py`, `rwspm.py`, `seed.py` | LQD、分区/聚合、随机种子 |
| `cball_ext/` | C源码、头文件、Linux/Windows构建脚本 |
| `generate_adni_data.py`, `test_generate_adni_data.py` | 联合重采样及契约验证 |
| `bench_n10000_batch.py`, `bench_n10000_gamma_batch.py` | limit/Gamma大样本单批基准 |
| `compare_gamma_stability.py` | 嵌套置换拟合与条件留出诊断 |
| `benchmark_adni44.py`, `bench_large_window.py` | 历史全区基准、资源监控、大窗基准 |
| `compare_adni44_distance.py`, `summarize_large_window.py` | 已完成实验的资源结果分析 |
| `compare_rwspm_bcov.py`, `bench_limit_1270.py` | 通用/特化实现对照、缓存基准 |
| `smoke_test_GWAS.py`, `probe_distance_memory.py`, `validate_generated_adni.py` | 合成冒烟、距离内存探针、私有数据验证 |
| `GWAS_left_simple.py` | 保留的简化对照入口 |

部分历史汇总脚本要求特定输出结构；它们不会因为没有随包附带结果而直接运行成功。`gammafit.py`应作为模块使用，其演示入口引用了本包未包含的`permutation_null`；不影响正式导入。

## 来源与第三方许可

源文件校验和及本地整理变更见 [SOURCE_MANIFEST.json](SOURCE_MANIFEST.json)。除了输出基线路径可配置化，复制时未改变统计方法或默认p值路径。

保留了Ball相关C源码的作者与GPL声明，见 [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) 和 [COPYING.GPL-3.0.txt](COPYING.GPL-3.0.txt)。本整理没有为其他原创文件额外指定整体许可证。输入数据的取得与使用须遵循其各自的数据协议。
