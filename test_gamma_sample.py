"""Sampling, rank semantics, scaled moments, C/NumPy and multi-SNP regressions."""
import json,pathlib,tempfile
import numpy as np
from scipy import stats
import bcov as b
from gammafit import gamma_moment_match

def main():
    assert b.snp_backend_available(),'Compile the SNP library before running C checks'
    rng=np.random.default_rng(44)
    for counts in ((98,1,1),(80,20,0),(0,80,20),(40,0,60),(99,0,1),(0,100,0)):
        x=np.repeat(np.arange(3),counts).astype(float)
        for ratio in (0.001,0.1,0.49,1):
            plan=b.make_gamma_sample_plan(x,ratio,39,seed=1145)
            other=b.make_gamma_sample_plan(x,ratio,39,seed=1145)
            np.testing.assert_array_equal(plan['indices'],other['indices'])
            assert len(set(plan['indices']))==plan['m']
            assert set(x[plan['indices']])==set(x)
            assert sum(plan['sample_counts'])==plan['m']
            assert all(q>=1 and q<=n for q,n in zip(plan['sample_counts'],plan['full_counts']))
            assert all(np.array_equal(np.sort(p),np.arange(plan['m'])) for p in plan['perms'])
    for ratio in (0,-1,1.1,np.nan,np.inf):
        try:b.make_gamma_sample_plan(np.array([0,1,2]),ratio,9)
        except ValueError:pass
        else:raise AssertionError('Invalid ratio accepted')
    for x in (np.array([0,np.nan,2]),np.array([0,-9,1]),np.array([0,0.5,1])):
        try:b.make_gamma_sample_plan(x,0.1,9)
        except ValueError:pass
        else:raise AssertionError('Invalid dosages accepted')
    x=np.repeat([0.,1.,2.],[20,9,1]);feat=np.round(rng.normal(size=(30,4)),1)
    ry=b.build_rank_matrix(feat,backend='c')
    plan=b.make_gamma_sample_plan(x,0.4,79,seed=9);idx=plan['indices'];m=plan['m']
    reranked=b._rank_rows_leq(ry[np.ix_(idx,idx)]).astype(np.int64)
    direct=b.build_rank_matrix(feat[idx],backend='c')
    np.testing.assert_array_equal(reranked,direct)
    stat,p,detail=b.bcov_snp_gamma_sample(x,RY=ry,n_perm=79,ratio=0.4,plan=plan,backend='c',return_details=True)
    raw=b.bcov_snp_null(x[idx],direct,plan['perms'],backend='c')
    fit=gamma_moment_match(m*raw)
    np.testing.assert_allclose(p,stats.gamma.sf(len(x)*stat,a=fit['shape'],scale=fit['scale']),rtol=1e-12)
    nstat,npval=b.bcov_snp_gamma_sample(x,RY=ry,n_perm=79,ratio=0.4,plan=plan,backend='numpy')
    np.testing.assert_allclose([stat,p],[nstat,npval],rtol=1e-10,atol=1e-14)
    full=b.make_gamma_sample_plan(x,1,79,seed=9)
    fullstat,fp=b.bcov_snp(x,None,n_perm=79,method='gamma_sample',RY=ry,ratio=1,sample_plan=full,backend='c')
    null=b.bcov_snp_null(x,ry,full['perms'],backend='c');ff=gamma_moment_match(null)
    np.testing.assert_allclose(fp,stats.gamma.sf(fullstat,a=ff['shape'],scale=ff['scale']),rtol=1e-12)
    # Two-point sampled null is invariant; fallback must be explicit and finite.
    _,fp,fd=b.bcov_snp_gamma_sample(np.array([0.,1.,0.,1.]),
        RY=b.build_rank_matrix(rng.normal(size=(4,2)),backend='c'),n_perm=39,ratio=0.1,backend='c',return_details=True)
    assert fd['fallback_reason'] is not None and np.isfinite(fp)
    # Real spawn workers, three SNPs including a constant SNP. No input files.
    import GWAS_left as g
    from smoke_test_GWAS import make_synthetic_data
    adni,cov,hipp=make_synthetic_data(30,30,20,7)
    adni=np.column_stack([adni[:,0],rng.binomial(2,0.1,30),np.zeros(30)])
    with tempfile.TemporaryDirectory(dir=pathlib.Path(__file__).parent) as out:
        g.SIDE='left';g.M1,g.M2=30,20;g.OUT_DIR=out
        g.load_data=lambda:(adni,cov,hipp)
        result=g.run_gwas(n_snp=3,n_perm=39,method='gamma_sample',ratio=0.1,
                          n_process=2,max_regions=4,backend='c',cache_rank=False)
        assert len(result)==3 and result[2]['p_agg']==1
        for j in range(3):
            planmeta=json.loads((pathlib.Path(out)/f'gamma_sample_plan_snp_{j:05d}.json').read_text())
            diag=json.loads((pathlib.Path(out)/f'gamma_sample_diagnostics_snp_{j:05d}.json').read_text())
            assert len(diag['regions'])==4
            assert all(r['snp_sha256']==planmeta['snp_sha256'] for r in diag['regions'])
            assert all(np.isfinite(r['p']) and 0<=r['p']<=1 for r in diag['regions'])
    print('PASS: retained genotypes, quotas, seeds, subset reranking, scaling, C/NumPy, ratio=1, fallback, Windows multi-SNP workers')
if __name__=='__main__':main()
