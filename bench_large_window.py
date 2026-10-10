"""Single-process ADNI large-window memory/time probe using existing kernels."""
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[key]='1'
import argparse, json, pathlib, sys, time
import benchmark_adni44 as monitor


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode',choices=['monitor','run'])
    parser.add_argument('--data_dir',required=True)
    parser.add_argument('--output_dir',required=True)
    parser.add_argument('--window',type=int,default=100)
    parser.add_argument('--step',type=int,default=50)
    parser.add_argument('--max_regions',type=int,default=1)
    args=parser.parse_args()
    out=pathlib.Path(args.output_dir).resolve()
    monitor.OUT=out
    if args.mode=='monitor':
        monitor.monitor(runner_script=pathlib.Path(__file__).resolve(),runner_args=sys.argv[2:],
                        metadata_override={'command':' '.join(sys.argv),
                        'notes':'One sequential process; original LQD, projection, SNP C stat, cdist, Ball BDD and eigh/HBE. Separate timing phases, no full GWAS aggregation.'})
        return
    import numpy as np
    import bcov as b
    import GWAS_left as g
    from generate_adni_data import load_generated_adni
    event=monitor.event
    assert b.snp_backend_available() and b._load_cball()
    if not (1<=args.window<=min(g.M1,g.M2)) or args.step<1 or args.max_regions<1:
        raise ValueError('Invalid window, step or max_regions')
    event('load_data')
    adni,cov,hipp=load_generated_adni(args.data_dir)
    n=adni.shape[0]
    x=np.asarray(adni[:,0],dtype=float)
    assert np.unique(x).size>1
    event('partition')
    partition=g.slide_width_step(args.step,args.window,g.M1,g.M2)
    indices=partition['idx']
    m=min(args.max_regions,len(indices))
    if len(indices)>10:
        raise ValueError('Requested partition exceeds the 10-region test limit')
    config={'n':n,'window':args.window,'step':args.step,'m_all':len(indices),
            'm_tested':m,'pixels_per_region':indices.shape[1],
            'grid':[partition['width_x'],partition['width_y']],
            'processes':1,'distance_backend':'cdist','method':'limit','snp_column':0}
    (out/'configuration.json').write_text(json.dumps(config,indent=2))
    print(json.dumps(config),flush=True)
    event('lqd')
    start=time.perf_counter()
    features=g.LQD(hipp,indices[:m],t=np.linspace(0,1,g.N_QUANTILE))
    lqd_s=time.perf_counter()-start
    del hipp
    event('covariate_adjustment')
    p=cov.shape[1]
    P_cov=np.eye(n)-cov@np.linalg.inv(cov.T@cov+1e-8*np.eye(p))@cov.T
    adjusted=np.stack([P_cov@region for region in features])
    del features,cov
    event('snp_distance')
    DX=np.abs(x[:,None]-x[None,:])
    event('snp_bdd_and_center')
    Kx=b._center_bdd_matrix(b._bdd_matrix_bias_c(DX))
    results=[]
    for r in range(m):
        row={'region':r}
        def phase(name):
            event(f'region_{r}_{name}')
            return time.perf_counter()
        start=phase('rank_c')
        RY=b.build_rank_matrix(adjusted[r],backend='c')
        row['rank_s']=time.perf_counter()-start
        start=phase('distance_cdist')
        DY=g.pairwise_region_distance(adjusted[r],backend='cdist')
        row['distance_s']=time.perf_counter()-start
        start=phase('stat_c')
        stat=b.bcov_snp_stat(x,RY,backend='c')
        row['stat_c_s']=time.perf_counter()-start
        start=phase('image_bdd_and_center')
        Ky=b._center_bdd_matrix(b._bdd_matrix_bias_c(DY))
        row['image_bdd_s']=time.perf_counter()-start
        start=phase('joint_kernel')
        joint=Kx*Ky
        row['joint_s']=time.perf_counter()-start
        start=phase('eigh')
        eigenvalues=np.linalg.eigh(joint)[0]
        eigenvalues=eigenvalues[eigenvalues>0]/n
        row['eigh_s']=time.perf_counter()-start
        start=phase('hbe')
        pvalue=float(b._hbe(eigenvalues,n*stat))
        row['hbe_s']=time.perf_counter()-start
        assert np.isfinite(stat) and np.isfinite(pvalue) and 0<=pvalue<=1
        row.update(stat=float(stat),p=pvalue)
        results.append(row)
        del RY,DY,Ky,joint,eigenvalues
        (out/'region_results.json').write_text(json.dumps(results,indent=2))
    (out/'lqd_timing.json').write_text(json.dumps({'lqd_s':lqd_s,'regions':m}))
    event('finished')


if __name__=='__main__':
    main()
