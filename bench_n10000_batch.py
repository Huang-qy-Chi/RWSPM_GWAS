"""One parallel batch: exactly one SNP/region limit task per worker, P<=24."""
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[key]='1'
import argparse, json, multiprocessing as mp, pathlib, sys, time
import benchmark_adni44 as bm

PREP=None
BARRIER=None
OUT=None


def worker_init(adni, features, seed, barrier):
    global BARRIER
    import GWAS_left as g
    BARRIER=barrier
    g.init_worker(adni,features,1,'limit',seed,backend='c',cache=False,
                  cache_max=0,distance_backend='cdist')
    (OUT/f'worker_initialized_{os.getpid()}.json').write_text(json.dumps({'pid':os.getpid(),'time':time.time()}))


def region_task(r):
    import numpy as np
    import bcov as b
    import GWAS_left as g
    (OUT/f'task_started_{r}.json').write_text(json.dumps({'pid':os.getpid(),'region':r,'time':time.time()}))
    BARRIER.wait(timeout=300)
    start=time.perf_counter()
    x=g._snp_vector(0)
    ry=g.get_region_rank(r)
    t1=time.perf_counter()
    dy=g._region_distance(r)
    t2=time.perf_counter()
    stat=b.bcov_snp_stat(x,ry,backend='c')
    t3=time.perf_counter()
    original_bdd=b._bdd_matrix_bias_c
    original_eigh=np.linalg.eigh
    elapsed={'bdd':0.0,'eigh':0.0}
    def bdd(*args,**kwargs):
        tick=time.perf_counter()
        value=original_bdd(*args,**kwargs)
        elapsed['bdd']+=time.perf_counter()-tick
        return value
    def eigh(*args,**kwargs):
        tick=time.perf_counter()
        value=original_eigh(*args,**kwargs)
        elapsed['eigh']+=time.perf_counter()-tick
        return value
    b._bdd_matrix_bias_c=bdd
    np.linalg.eigh=eigh
    try:
        _,p=b.bcov_snp_limit(x,DY=dy,RY=ry,stat=stat,prep=PREP)
    finally:
        b._bdd_matrix_bias_c=original_bdd
        np.linalg.eigh=original_eigh
    end=time.perf_counter()
    assert np.isfinite(stat) and np.isfinite(p) and 0<=p<=1
    row={'region':r,'pid':os.getpid(),'rank_s':t1-start,'distance_s':t2-t1,
         'stat_c_s':t3-t2,'limit_s':end-t3,'image_bdd_s':elapsed['bdd'],
         'eigh_s':elapsed['eigh'],'total_s':end-start,'stat':float(stat),'p':float(p)}
    tmp=OUT/f'task_done_{r}.json.tmp'
    tmp.write_text(json.dumps(row,indent=2))
    tmp.replace(OUT/f'task_done_{r}.json')
    return row


def summarize(output):
    import numpy as np
    root=pathlib.Path(output).resolve()
    resource=json.loads((root/'resource_summary.json').read_text())
    config=json.loads((root/'configuration.json').read_text())
    rows=json.loads((root/'batch_results.json').read_text())
    p=config['processes']
    assert resource['exit_code']==0 and len(rows)==p
    assert sorted(x['region'] for x in rows)==list(range(p))
    assert len({x['pid'] for x in rows})==p
    assert all(np.isfinite(x['p']) and 0<=x['p']<=1 for x in rows)
    stages={s['phase']:s for s in resource['stages']}
    batch=config['batch_wall_s']
    prep=stages['snp_kernel_prep']['wall_s']
    lqd=stages['lqd_parallel']['wall_s']
    cov=stages['covariate_adjustment']['wall_s']
    # 1488 / 16 = 93 full batches. Actual measured parallel throughput is used.
    region_scan=batch*config['m_all']/p
    estimate=prep+cov+lqd*config['m_all']/p+region_scan
    result={'n':config['n'],'processes':p,'tested_regions':len(rows),
            'm_all':config['m_all'],'batch_wall_s':batch,'snp_prep_s':prep,
            'lqd_batch_s':lqd,'covariate_adjustment_s':cov,
            'peak_pss_gib':resource['peak_pss_gib'],
            'estimated_region_scan_s':region_scan,
            'estimated_single_snp_total_s':estimate,
            'estimated_single_snp_days':estimate/86400,
            'memory_note':'Measured task-tree PSS, no region cache; full 1488-region G_tilde is 2.328 GiB, exceeding the sampled batch by about 2.30 GiB. Full-stage preprocessing peaks were not measured.',
            'time_note':'One batch, first P regions. Extrapolation assumes stable throughput; includes extrapolated LQD, once SNP prep and measured small-batch covariate adjustment. Full covariate adjustment and aggregation remain unmeasured.'}
    (root/'single_snp_estimate.json').write_text(json.dumps(result,indent=2))
    lines=['# n=10000 单批并行实测与单 SNP 外推','',
           f'样本量10000，原窗宽11、步长3，总1488区；测试首{p}个区域，{p}个不同BCov worker各计算一次。SNP第0列、C统计量、cdist、limit/HBE、无RY/DY缓存、BLAS单线程。',
           '输入为ADNI联合重采样数据；重复行会影响时间与谱结构，不能代表独立真实受试者。','',
           '| 指标 | 数值 |','|---|---:|',
           f'| 并行BCov批次秒数 | {batch:.2f} |',
           f'| SNP预处理秒数（一次） | {prep:.2f} |',
           f'| 实测进程树峰值PSS/GiB | {resource["peak_pss_gib"]:.3f} |',
           f'| 1488区BCov扫描外推/天 | {region_scan/86400:.2f} |',
           f'| 单SNP总时间近似/天 | {estimate/86400:.2f} |','',
           '外推采用实测批次墙钟 × 1488/P，不再用单核时间除以进程数。一次小批不能验证全扫描的长期吞吐量。',
           '全1488区 G_tilde 为2.328 GiB，比本批多约2.30 GiB共享数组；协变量校正还可能同时保留未调整特征与堆叠临时数组。因此实测批峰值不是完整预处理峰值保证。',
           '正式全扫描必须保持缓存关闭或受限。cgroup硬上限240 GiB、MemoryHigh200 GiB、禁止本任务swap；还有可用内存48 GiB与PSS220 GiB软保护。',
           '原始证据：resource_summary.json、memory.jsonl、task_done_*.json、status.json、run.log、configuration.json、source_sha256.txt。']
    (root/'single_snp_estimate.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(json.dumps(result,indent=2))


def run(args):
    global PREP,OUT
    import numpy as np
    import bcov as b
    import GWAS_left as g
    from generate_adni_data import load_generated_adni
    from concurrent.futures import ProcessPoolExecutor
    OUT=pathlib.Path(args.output_dir).resolve()
    event=bm.event
    assert b.snp_backend_available() and b._load_cball()
    event('load_data')
    adni,cov,hipp=load_generated_adni(args.data_dir)
    if adni.shape[0]!=args.expected_n or adni.shape[1]!=1:
        raise ValueError('Unexpected dataset size/SNP count')
    event('partition')
    part=g.slide_width_step(3,11,g.M1,g.M2)
    assert len(part['idx'])==1488
    selected=part['idx'][:args.processes]
    config={'n':adni.shape[0],'processes':args.processes,'tested_regions':args.processes,
            'm_all':1488,'window':11,'step':3,'method':'limit','cache':False,
            'source_data':str(pathlib.Path(args.data_dir).resolve())}
    (OUT/'configuration.json').write_text(json.dumps(config,indent=2))
    ctx=mp.get_context('fork')
    event('lqd_parallel')
    with ProcessPoolExecutor(max_workers=args.processes,mp_context=ctx,
                             initializer=g._lqd_init,
                             initargs=(hipp,selected,np.linspace(0,1,21))) as pool:
        features=list(pool.map(g._lqd_one_region,range(args.processes)))
    del hipp
    event('covariate_adjustment')
    n,q=cov.shape
    P_cov=np.eye(n)-cov@np.linalg.inv(cov.T@cov+1e-8*np.eye(q))@cov.T
    adjusted=np.stack([P_cov@f for f in features])
    del cov,features
    event('checkpoint_features')
    np.save(OUT/'adjusted_features.npy',adjusted,allow_pickle=False)
    event('snp_kernel_prep')
    PREP=b.bcov_snp_limit_prep(np.asarray(adni[:,0],dtype=float))
    event('checkpoint_snp_kernel')
    np.save(OUT/'DX.npy',PREP['DX'],allow_pickle=False)
    np.save(OUT/'Kx_c.npy',PREP['Kx_c'],allow_pickle=False)
    PREP=None
    PREP={'DX':np.load(OUT/'DX.npy',mmap_mode='r'),'Kx_c':np.load(OUT/'Kx_c.npy',mmap_mode='r')}
    event('bcov_parallel')
    tick=time.perf_counter()
    barrier=ctx.Barrier(args.processes)
    with ctx.Pool(args.processes,initializer=worker_init,
                  initargs=(adni,adjusted,1145,barrier)) as pool:
        future=pool.map_async(region_task,range(args.processes),chunksize=1)
        rows=future.get(timeout=40*3600)
    config['batch_wall_s']=time.perf_counter()-tick
    assert len({r['pid'] for r in rows})==args.processes
    (OUT/'configuration.json').write_text(json.dumps(config,indent=2))
    (OUT/'batch_results.json').write_text(json.dumps(rows,indent=2))
    event('finished')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode',choices=['run','monitor','summarize'])
    parser.add_argument('--data_dir')
    parser.add_argument('--output_dir',required=True)
    parser.add_argument('--processes',type=int,default=16)
    parser.add_argument('--expected_n',type=int,default=10000)
    args=parser.parse_args()
    if not 1<=args.processes<=24:
        parser.error('processes must be between 1 and 24')
    if args.mode=='summarize':
        summarize(args.output_dir)
    elif args.mode=='run':
        run(args)
    else:
        bm.OUT=pathlib.Path(args.output_dir).resolve()
        bm.monitor(runner_script=pathlib.Path(__file__).resolve(),runner_args=sys.argv[2:],
                   metadata_override={'command':' '.join(sys.argv),'processes':args.processes,
                                      'notes':'Single batch, one region per worker; cgroup capped by systemd service.'},
                   sample_interval_s=5,min_available_gib=48,max_pss_gib=220)
        report=json.loads((bm.OUT/'resource_summary.json').read_text())
        if report['exit_code']==0:
            summarize(bm.OUT)
        else:
            sys.exit(1)


if __name__=='__main__':
    main()
