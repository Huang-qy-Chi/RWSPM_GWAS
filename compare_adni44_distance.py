"""Compare two completed, identically configured ADNI one-SNP benchmarks."""
import csv, json, pathlib, statistics, sys
import numpy as np

baseline, current = map(lambda p:pathlib.Path(p).resolve(),sys.argv[1:3])
def read(root):
    summary = json.loads((root/'resource_summary.json').read_text())
    assert summary['exit_code']==0
    rows = [json.loads(line) for file in root.glob('worker_*.jsonl')
            for line in file.read_text().splitlines()]
    assert len(rows)==1488 and sorted(r['region'] for r in rows)==list(range(1488))
    assert all(np.isfinite(r['p']) and 0<=r['p']<=1 for r in rows)
    p = np.load(root/'results'/'p_values_snp_00000.npy')
    assert p.shape==(1488,)
    pipeline_s=sum(s['wall_s'] for s in summary['stages'] if s['phase'] not in ('imports','finished'))
    stages={s['phase']:s for s in summary['stages']}
    result=next(csv.DictReader((root/'results'/'summary.csv').open()))
    return {'summary':summary,'rows':rows,'p':p,'pipeline_s':pipeline_s,'stages':stages,'result':result}

a,b=read(baseline),read(current)
def hashes(root):
    return dict((line.split()[1],line.split()[0])
                for line in (root/'source_sha256.txt').read_text().splitlines())
ha,hb=hashes(baseline),hashes(current)
for name in ('bcov.py','lqd.py','cball_ext/cball_ext.so','cball_ext/csnpbcov.so'):
    assert ha[name]==hb[name], 'Unrelated source changed: '+name
np.testing.assert_allclose(b['p'],a['p'],rtol=1e-11,atol=1e-12)
sa=np.array([r['stat'] for r in sorted(a['rows'],key=lambda r:r['region'])])
sb=np.array([r['stat'] for r in sorted(b['rows'],key=lambda r:r['region'])])
np.testing.assert_array_equal(sa,sb)
data={
    'baseline_dir':str(baseline),'current_dir':str(current),'regions':1488,
    'baseline_pipeline_s':a['pipeline_s'],'cdist_pipeline_s':b['pipeline_s'],
    'baseline_peak_pss_gib':a['summary']['peak_pss_gib'],
    'cdist_peak_pss_gib':b['summary']['peak_pss_gib'],
    'peak_pss_reduction_pct':100*(1-b['summary']['peak_pss_gib']/a['summary']['peak_pss_gib']),
    'p_max_abs_difference':float(np.max(np.abs(b['p']-a['p']))),
    'p_bitwise_equal':bool(np.array_equal(b['p'],a['p'])),
    'stat_bitwise_equal':True,
    'bcov_lqd_and_c_kernels_unchanged':True,
    'baseline_mean_distance_s':statistics.mean(r['distance_s'] for r in a['rows']),
    'cdist_mean_distance_s':statistics.mean(r['distance_s'] for r in b['rows']),
    'baseline_mean_limit_s':statistics.mean(r['limit_s'] for r in a['rows']),
    'cdist_mean_limit_s':statistics.mean(r['limit_s'] for r in b['rows']),
    'baseline_mean_eigh_s':statistics.mean(r['eigh_s'] for r in a['rows']),
    'cdist_mean_eigh_s':statistics.mean(r['eigh_s'] for r in b['rows']),
    'result':b['result']}
(current/'distance_comparison.json').write_text(json.dumps(data,indent=2))
lines=['# cdist 距离计算：单 SNP 44 进程复测', '',
       '两次测试均为 n=1270、左海马 1488 脑区、21 维 LQD、ADNI_1_3_sampled_25000.npy 第 0 列 SNP、44 进程、单线程 BLAS、SNP 专用 C 统计量和 limit/HBE。',
       'LQD 仍按脑区并行；无跨 SNP RY/DY 缓存。主要实现变化是区域 DY 从完整 NumPy 广播改为 cdist。',
       'SNP 侧核由父进程构造并 fork 共享，两次配置相同；没有在本次实验引入其他性能优化。', '',
       '| 指标 | 修改前 | cdist |','|---|---:|---:|',
       f"| RWSPM 总墙钟 / 秒 | {a['pipeline_s']:.2f} | {b['pipeline_s']:.2f} |",
       f"| 整体采样峰值 PSS / GiB | {a['summary']['peak_pss_gib']:.3f} | {b['summary']['peak_pss_gib']:.3f} |",
       f"| 距离计算平均秒/脑区（并发中的 worker 墙钟） | {data['baseline_mean_distance_s']:.5f} | {data['cdist_mean_distance_s']:.5f} |", '',
       f"整体峰值 PSS 降低 {data['peak_pss_reduction_pct']:.2f}%。这是两次单次运行的采样对比，未做重复基准，不把总耗时差异全部归因于 cdist。", '',
       '| 步骤 | 原耗时/秒 | 新耗时/秒 | 原峰值 PSS/GiB | 新峰值 PSS/GiB |',
       '|---|---:|---:|---:|---:|']
names={'load_data':'数据读取','partition':'脑区划分','lqd_parallel':'并行 LQD',
       'covariate_adjustment':'协变量校正','snp_kernel_prep':'SNP 侧核预处理',
       'bcov_parallel':'区域 BCov（含进程池启动）','rw_mtcct_aggregation':'RW-MTCCT 聚合',
       'snp_result_save_and_pool_shutdown':'SNP 输出和进程池退出','summary_and_save':'汇总输出'}
for key,name in names.items():
    x,y=a['stages'][key],b['stages'][key]
    mx=f"{x['peak_pss_gib']:.3f}" if x['samples'] else '未采到'
    my=f"{y['peak_pss_gib']:.3f}" if y['samples'] else '未采到'
    lines.append(f"| {name} | {x['wall_s']:.4f} | {y['wall_s']:.4f} | {mx} | {my} |")
lines += ['', 'PSS 为主进程和全部后代进程之和，避免共享页面重复计算；监控本身不计入。',
          '采样设定 0.5 秒等待，实际还包含读取各进程 smaps_rollup 的时间；这是非原子采样峰值，可能遗漏瞬时峰值。',
          '短阶段未采到内存不等于零。采样方式与旧实验相同。', '',
          f"完整性：两次各 1488 个脑区，无重复或遗漏；统计量逐位一致。p 值最大绝对差 {data['p_max_abs_difference']:.3g}，逐位一致：{data['p_bitwise_equal']}。",
          'p 值一致性校验阈值：rtol=1e-11、atol=1e-12；真实 ADNI 全部区域已检查。', '',
          '## 修改与兼容选项', '',
          '`GWAS_left.py` 默认采用 `--distance_backend cdist`，右半球入口复用该实现。',
          '兼容路径：`--distance_backend numpy_blocked --distance_block_size 128`，保留原来的 NumPy 按特征求和顺序。',
          '该改变位于 LQD 特征之后的区域欧氏距离阶段，lqd.py 的 LQD 变换未修改。',
          '远程原文件备份：GWAS_left.py.pre_cdist_20261008。', '',
          '距离阶段的 n=10000 合成单进程检查已通过，输出矩阵为 0.745 GiB。',
          '完整 limit 仍包含 RY、BDD、上三角索引及特征分解工作区；此次 n=1270 的降低不能直接证明 n=10000 完整流程可开 44 进程。',
          '后续应先测量 n=10000 单脑区完整 limit 峰值，再按资源预算增加并发。', '',
          '原始证据：resource_summary.json、memory.jsonl、phases.jsonl、worker_*.jsonl、run.log、results/。']
(current/'distance_comparison.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
print(json.dumps(data,indent=2))
