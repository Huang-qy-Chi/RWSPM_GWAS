"""Gamma performance batch: one region per worker, shared permutation indices."""
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[key]='1'
import argparse,hashlib,json,multiprocessing as mp,pathlib,sys,time
import benchmark_adni44 as bm
OUT=None
PERMS=None
BARRIER=None
EXPECTED=None

def write_json(path,value):
    temp=path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(value,indent=2),encoding='utf-8')
    temp.replace(path)

def worker_init(adni,features,barrier):
    global BARRIER
    import GWAS_left as g
    BARRIER=barrier
    g.init_worker(adni,features,len(PERMS),'gamma',1145,backend='c',cache=False,
                  cache_max=0,distance_backend='cdist')
    write_json(OUT/f'worker_initialized_{os.getpid()}.json',
               {'pid':os.getpid(),'time':time.time(),
                'permutations_sha256':hashlib.sha256(PERMS.tobytes()).hexdigest()})

def region_task(r):
    import numpy as np
    from scipy import stats
    import bcov as b
    import GWAS_left as g
    from gammafit import gamma_moment_match
    write_json(OUT/f'task_started_{r}.json',{'pid':os.getpid(),'region':r,'time':time.time()})
    BARRIER.wait(timeout=300)
    start=time.perf_counter()
    x=g._snp_vector(0)
    ry=g.get_region_rank(r)
    t1=time.perf_counter()
    stat=b.bcov_snp_stat(x,ry,backend='c')
    t2=time.perf_counter()
    null=b.bcov_snp_null(x,ry,PERMS,backend='c')
    t3=time.perf_counter()
    assert len(null)==len(PERMS) and np.isfinite(null).all()
    fit=gamma_moment_match(null)
    assert all(np.isfinite(fit[k]) and fit[k]>0 for k in ('shape','scale','variance'))
    p=float(stats.gamma.sf(stat,a=fit['shape'],scale=fit['scale']))
    end=time.perf_counter()
    assert np.isfinite(stat) and np.isfinite(p) and 0<=p<=1
    if EXPECTED is not None:
        np.testing.assert_allclose(stat,EXPECTED[r],rtol=1e-10,atol=1e-14)
    else:
        # Small-sample preflight: verify split timers reproduce the public gamma API.
        ref_stat,ref_p=b.bcov_snp(x,None,n_perm=len(PERMS),method='gamma',RY=ry,backend='c',perms=PERMS)
        np.testing.assert_allclose(stat,ref_stat,rtol=1e-12)
        np.testing.assert_allclose(p,ref_p,rtol=1e-9,atol=1e-14)
    np.save(OUT/f'null_stats_{r}.npy',null,allow_pickle=False)
    row={'region':r,'pid':os.getpid(),'rank_s':t1-start,'stat_c_s':t2-t1,
         'permutation_s':t3-t2,'gamma_fit_s':end-t3,'gamma_s':end-t2,
         'total_s':end-start,'n_perm':len(PERMS),'stat':float(stat),'p':p,**fit}
    write_json(OUT/f'task_done_{r}.json',row)
    return row

def run(args):
    global OUT,PERMS,EXPECTED
    import numpy as np
    import bcov as b
    import GWAS_left as g
    from generate_adni_data import load_generated_adni
    from concurrent.futures import ProcessPoolExecutor
    OUT=pathlib.Path(args.output_dir).resolve()
    assert b.snp_backend_available()
    bm.event('load_data')
    adni,cov,hipp=load_generated_adni(args.data_dir)
    assert adni.shape==(args.expected_n,1)
    bm.event('partition')
    part=g.slide_width_step(3,11,g.M1,g.M2)
    assert len(part['idx'])==1488
    selected=part['idx'][:args.processes]
    assert len(np.unique(selected,axis=0))==args.processes
    config={'n':args.expected_n,'processes':args.processes,'tested_regions':args.processes,
            'm_all':1488,'window':11,'step':3,'method':'gamma','n_perm':args.n_perm,
            'permutation_seed':1145,'shared_permutations':True,'cache':False,
            'source_data':str(pathlib.Path(args.data_dir).resolve()),
            'limit_output':str(pathlib.Path(args.limit_output).resolve()),
            'prior_output':str(pathlib.Path(args.prior_output).resolve()),
            'pvalue_implementation':'gamma_moment_match plus scipy.stats.gamma.sf'}
    write_json(OUT/'configuration.json',config)
    ctx=mp.get_context('fork')
    bm.event('lqd_parallel')
    with ProcessPoolExecutor(max_workers=args.processes,mp_context=ctx,
                             initializer=g._lqd_init,
                             initargs=(hipp,selected,np.linspace(0,1,21))) as pool:
        features=list(pool.map(g._lqd_one_region,range(args.processes)))
    del hipp
    bm.event('covariate_adjustment')
    n,q=cov.shape
    P_cov=np.eye(n)-cov@np.linalg.inv(cov.T@cov+1e-8*np.eye(q))@cov.T
    adjusted=np.stack([P_cov@f for f in features])
    del cov,features
    bm.event('checkpoint_features')
    np.save(OUT/'adjusted_features.npy',adjusted,allow_pickle=False)
    bm.event('shared_permutation_prep')
    PERMS=b.make_permutations(n,args.n_perm,seed=1145)
    assert PERMS.shape==(args.n_perm,n) and PERMS.dtype==np.int32
    if args.expected_n==10000 and args.n_perm==999:
        prior=pathlib.Path(args.prior_output)
        assert np.array_equal(PERMS[:199],np.load(prior/'shared_permutations.npy',mmap_mode='r'))
        config['first_199_permutations_verified']=True
    for row in PERMS:
        assert np.array_equal(np.sort(row),np.arange(n))
    PERMS.flags.writeable=False
    np.save(OUT/'shared_permutations.npy',PERMS,allow_pickle=False)
    config['permutations_sha256']=hashlib.sha256(PERMS.tobytes()).hexdigest()
    if args.expected_n==10000:
        baseline=json.loads((pathlib.Path(args.limit_output)/'batch_results.json').read_text())
        EXPECTED={row['region']:row['stat'] for row in baseline}
    bm.event('bcov_parallel')
    tick=time.perf_counter()
    barrier=ctx.Barrier(args.processes)
    with ctx.Pool(args.processes,initializer=worker_init,initargs=(adni,adjusted,barrier)) as pool:
        rows=pool.map_async(region_task,range(args.processes),chunksize=1).get(timeout=40*3600)
    config['batch_wall_s']=time.perf_counter()-tick
    assert len({row['pid'] for row in rows})==args.processes
    write_json(OUT/'configuration.json',config)
    write_json(OUT/'batch_results.json',rows)
    bm.event('finished')

def capture_cgroup(root):
    for line in pathlib.Path('/proc/self/cgroup').read_text().splitlines():
        if line.startswith('0::'):
            cg=pathlib.Path('/sys/fs/cgroup'+line[3:])
            values={name:(cg/name).read_text().strip() for name in
                    ('memory.peak','memory.current','memory.events','memory.max','memory.high','memory.swap.max') if (cg/name).exists()}
            write_json(root/'cgroup_resources.json',values)

def summarize(output):
    import numpy as np
    root=pathlib.Path(output)
    config=json.loads((root/'configuration.json').read_text())
    resource=json.loads((root/'resource_summary.json').read_text())
    rows=json.loads((root/'batch_results.json').read_text())
    workers=[json.loads(p.read_text()) for p in root.glob('worker_initialized_*.json')]
    p=config['processes']
    assert resource['exit_code']==0 and len(rows)==p
    assert sorted(x['region'] for x in rows)==list(range(p)) and len({x['pid'] for x in rows})==p
    assert len(workers)==p and all(x['permutations_sha256']==config['permutations_sha256'] for x in workers)
    assert all(x['n_perm']==config['n_perm'] and np.isfinite(x['stat']) and np.isfinite(x['p']) and 0<=x['p']<=1 for x in rows)
    stages={x['phase']:x for x in resource['stages']}
    scan=config['batch_wall_s']*1488/p
    total=scan+stages['lqd_parallel']['wall_s']*1488/p+stages['covariate_adjustment']['wall_s']+stages['shared_permutation_prep']['wall_s']
    result={'n':config['n'],'processes':p,'method':'gamma','n_perm':config['n_perm'],
            'batch_wall_s':config['batch_wall_s'],'measured_total_wall_s':resource['total_wall_s'],
            'peak_pss_gib':resource['peak_pss_gib'],'estimated_region_scan_s':scan,
            'estimated_single_snp_total_s':total,'estimated_single_snp_days':total/86400,
            'mean_rank_s':float(np.mean([x['rank_s'] for x in rows])),
            'mean_stat_c_s':float(np.mean([x['stat_c_s'] for x in rows])),
            'mean_permutation_s':float(np.mean([x['permutation_s'] for x in rows])),
            'mean_gamma_fit_s':float(np.mean([x['gamma_fit_s'] for x in rows])),
            'note':'One measured parallel batch extrapolated by 1488/P. Full covariate adjustment, aggregation and long-run throughput unmeasured; aligned-bootstrap input, not independent subjects. Full features add about 2.30 GiB shared memory.'}
    cgfile=root/'cgroup_resources.json'
    if cgfile.exists():
        cg=json.loads(cgfile.read_text())
        if 'memory.peak' in cg: result['cgroup_memory_peak_gib']=int(cg['memory.peak'])/2**30
    write_json(root/'single_snp_estimate.json',result)
    baseline_path=pathlib.Path(config['limit_output'])/'single_snp_estimate.json'
    if baseline_path.exists() and config['n']==10000:
        baseline=json.loads(baseline_path.read_text())
        baseline_resource=json.loads((baseline_path.parent/'resource_summary.json').read_text())
        comparison={'limit':baseline,'gamma':result,
                    'batch_speedup':baseline['batch_wall_s']/result['batch_wall_s'],
                    'pss_reduction_fraction':1-result['peak_pss_gib']/baseline['peak_pss_gib']}
        write_json(root/'method_comparison.json',comparison)
        lines=[f'# n=10000、16进程：limit 与 gamma({config["n_perm"]}) 实测比较','',
               f'相同输入、第0个SNP、首16区域、窗宽11步长3、无区域缓存、C统计量、BLAS单线程；gamma各区共用{config["n_perm"]}次置换索引。','',
               f'| 指标 | limit | gamma（{config["n_perm"]}） |','|---|---:|---:|',
               f'| 16区BCov批次墙钟/秒 | {baseline["batch_wall_s"]:.2f} | {result["batch_wall_s"]:.2f} |',
               f'| 整次实验墙钟/秒 | {baseline_resource["total_wall_s"]:.2f} | {result["measured_total_wall_s"]:.2f} |',
               f'| 峰值进程树PSS/GiB | {baseline["peak_pss_gib"]:.3f} | {result["peak_pss_gib"]:.3f} |',
               f'| 1488区域单SNP外推/小时 | {baseline["estimated_single_snp_total_s"]/3600:.2f} | {total/3600:.2f} |','',
               f'实测BCov批次加速比：{comparison["batch_speedup"]:.2f}。PSS下降：{comparison["pss_reduction_fraction"]:.1%}。','',
               '外推采用批次墙钟×93，并计入各方法的一次预处理及LQD外推；完整协变量校正、RWSPM聚合和长期吞吐未实测。进程树PSS为5秒间隔采样，短阶段峰值可能遗漏。',
               'Gamma使用置换零统计量的均值与样本方差做二参数矩匹配，以sf计算尾概率；不是limit的三累积量HBE，速度比较不能证明极小p值校准。',
               '输入为联合重采样。正式全扫描内存还需计入完整区域特征和预处理临时数组；不能将单批峰值视为完整任务上限。']
        (root/'method_comparison.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(json.dumps(result,indent=2))

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode',choices=['run','monitor','summarize'])
    parser.add_argument('--data_dir')
    parser.add_argument('--output_dir',required=True)
    parser.add_argument('--processes',type=int,default=16)
    parser.add_argument('--expected_n',type=int,default=10000)
    parser.add_argument('--n_perm',type=int,default=199)
    parser.add_argument('--limit_output',default='bench_n10000_p16_20261009')
    parser.add_argument('--prior_output',default='bench_n10000_gamma199_p16_20261009')
    args=parser.parse_args()
    if not 1<=args.processes<=24: parser.error('processes must be 1..24')
    if args.n_perm<2: parser.error('n_perm must be at least 2')
    if args.mode=='run': run(args)
    elif args.mode=='summarize': summarize(args.output_dir)
    else:
        bm.OUT=pathlib.Path(args.output_dir).resolve()
        bm.monitor(runner_script=pathlib.Path(__file__).resolve(),runner_args=sys.argv[2:],
                   metadata_override={'command':' '.join(sys.argv),'processes':args.processes,
                                      'method':'gamma','n_perm':args.n_perm,'shared_permutations':True},
                   sample_interval_s=5,min_available_gib=48,max_pss_gib=220)
        capture_cgroup(bm.OUT)
        resource=json.loads((bm.OUT/'resource_summary.json').read_text())
        if resource['exit_code']==0:
            summarize(bm.OUT)
            if args.expected_n==10000 and args.n_perm==999:
                from compare_gamma_stability import compare
                compare(bm.OUT,args.prior_output,args.limit_output)
        else: sys.exit(1)

if __name__=='__main__': main()
