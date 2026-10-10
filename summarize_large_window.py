"""Summarize measured n=3000 costs and explicitly qualified n=10000 estimates."""
import json, pathlib, sys
root=pathlib.Path(sys.argv[1]).resolve()
summary=json.loads((root/'resource_summary.json').read_text())
config=json.loads((root/'configuration.json').read_text())
rows=json.loads((root/'region_results.json').read_text())
assert summary['exit_code']==0 and len(rows)==config['m_tested']==1
stages={s['phase']:s for s in summary['stages']}
bcov_keys=['snp_distance','snp_bdd_and_center']+[k for k in stages if k.startswith('region_')]
peak=max(stages[k]['peak_pss_gib'] for k in bcov_keys)
base=stages['imports']['peak_pss_gib']
ratio=10000/config['n']
estimated_worker=base+(peak-base)*ratio**2
planning_worker=estimated_worker*1.5
row=rows[0]
snp_prep=stages['snp_bdd_and_center']['wall_s']+stages['snp_distance']['wall_s']
region_total=sum(row[k] for k in ('rank_s','distance_s','stat_c_s','image_bdd_s','joint_s','eigh_s','hbe_s'))
lqd=stages['lqd']['wall_s']
dominant=row['image_bdd_s']+row['eigh_s']
estimated_region=dominant*ratio**3+(region_total-dominant)*ratio**2
estimated_prep=snp_prep*ratio**3
data={'configuration':config,'baseline_pss_gib':base,'bcov_peak_pss_gib':peak,
      'total_peak_pss_gib':summary['peak_pss_gib'],'region_bcov_s':region_total,
      'snp_prep_s':snp_prep,'lqd_s':lqd,'measured_result':row,
      'estimated_n10000_worker_gib':estimated_worker,
      'planning_worker_gib_with_50pct_margin':planning_worker,
      'estimated_n10000_region_bcov_s':estimated_region,
      'estimated_n10000_snp_prep_s':estimated_prep,
      'estimated_n10000_lqd_s':lqd*ratio,
      'n10000_not_run':True}
(root/'feasibility_summary.json').write_text(json.dumps(data,indent=2)+'\n')
names={'imports':'解释器和模块导入','load_data':'映射输入','partition':'分区',
       'lqd':'单区域 LQD','covariate_adjustment':'协变量校正',
       'snp_distance':'SNP 距离 DX','snp_bdd_and_center':'SNP BDD 与中心化',
       'region_0_rank_c':'C 秩矩阵 RY','region_0_distance_cdist':'cdist 距离 DY',
       'region_0_stat_c':'SNP 专用 C 统计量','region_0_image_bdd_and_center':'影像 BDD 与中心化',
       'region_0_joint_kernel':'联合核','region_0_eigh':'特征分解','region_0_hbe':'HBE p 值',
       'finished':'监控收尾'}
lines=['# 大窗单区域 n=3000 实测与 n=10000 估计','',
       '服务器 AMSS-ser-159；左海马 150×100；窗宽100、步长50；每区10000像素；LQD 21维；首 SNP；cdist + SNP专用C统计量 + 原Ball BDD + eigh/HBE；单进程、BLAS单线程。',
       '输入是从1270名 ADNI受试者联合重采样的3000行，1154个唯一来源受试者。',
       '原分区函数返回2×2=4个窗口，起始像素索引[0,0,5000,5000]，只有2个唯一窗口。本次只测试第一个唯一窗口。',
       '没有修改分区函数；没有启动n=10000完整计算。', '',
       '| 阶段 | 墙钟秒 | 采样峰值 PSS/GiB |','|---|---:|---:|']
for stage in summary['stages']:
    mem=f"{stage['peak_pss_gib']:.3f}" if stage['samples'] else '未采到'
    lines.append(f"| {names.get(stage['phase'],stage['phase'])} | {stage['wall_s']:.4f} | {mem} |")
lines += ['',f"包含导入和监控收尾的总墙钟：{summary['total_wall_s']:.2f}秒；整体峰值PSS {summary['peak_pss_gib']:.3f} GiB。",
          f"单区域 BCov（不含可复用的 SNP 侧预处理）：{region_total:.2f}秒。SNP 侧预处理：{snp_prep:.2f}秒。",
          'PSS是单个运行进程的采样值，监控进程不计入；短暂峰值可能遗漏，短步骤无采样不写为零。',
          'KDE 使用原 lqd.py；limit 计算直接调用原内部原语，按原顺序保留 DX/Kx、RY/DY、Ky和联合核等矩阵。', '',
          '## n=10000：容量估计', '',
          f"n²倍率={ratio**2:.3f}；n³倍率={ratio**3:.3f}。BCov测得峰值={peak:.3f} GiB，固定导入基线约{base:.3f} GiB。",
          f"按 固定基线+(峰值−基线)×n²倍率 粗估，n=10000单进程约 **{estimated_worker:.2f} GiB**。",
          f"为规划并发另加50%余量，按 **{planning_worker:.2f} GiB/进程** 预算；余量是工程选择，不是实测置信界或峰值保证。", '',
          '| 同时计算的区域数 | 含50%余量的进程预算/GiB |','|---|---:|']
for workers in (1,2,4,8,10):
    lines.append(f'| {workers} | {planning_worker*workers:.1f} |')
lines += ['', '服务器总内存约377 GiB、现有可用约370 GiB。还应为父进程、共享输入、系统和其他任务留余量。',
          '本次单 SNP 输入；n=10000若同时保留25000列float64 SNP数据，还需约1.86 GiB共享输入。',
          '测算没有跨区域/跨SNP RY与DY长期缓存；后续应关闭或限制这些缓存，避免累积内存。',
          '本次大窗只有2个唯一区域，按上述预算少量区域并行具有充足容量余量；不能据此证明44个BCov进程安全。',
          '单进程预算包括当前保留的协变量投影及SNP核，fork共享时部分不应重复计入；表中按各进程完整预算保守相加。',
          'n=10000时每张float64/int64 n×n矩阵为0.745 GiB。LQD大窗的KDE临时数组主要随窗内像素数量变化；影像输入随n线性增长，而BCov矩阵按n²增长。', '',
          '## n=10000：时间估计（未经实测）', '',
          f"同一大窗单区域LQD按n线性外推：{lqd*ratio/60:.1f}分钟。",
          f"SNP侧核一次性预处理按n³外推：{estimated_prep/3600:.2f}小时。",
          f"单区域BCov：影像BDD及eigh按n³、其余近似按n²外推，约 **{estimated_region/3600:.2f}小时/区域**。",
          '这些只说明规划量级。缓存、内存带宽、矩阵阶数、CPU频率、距离并列及特征分解的谱结构都会影响外推。',
          '重采样带来的重复行可能改变耗时，不能将此次速度等同于独立真实受试者的速度。',
          '少量大窗区域可以减少任务总数量，但改变了区域划分与检验目标，不能与原1488小窗流程视为同一分析。', '',
          '原始文件：configuration.json、resource_summary.json、region_results.json、memory.jsonl、phases.jsonl、source_sha256.txt、run.log。']
(root/'feasibility_report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
print(json.dumps(data,indent=2))
