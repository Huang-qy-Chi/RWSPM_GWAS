"""Distance-only feasibility probe; does not modify the GWAS pipeline."""
import os
for key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '1'
import json, resource, time
import numpy as np
from scipy.spatial.distance import cdist
import bcov
from GWAS_left import pairwise_region_distance

rng = np.random.default_rng(1145)
checks = []
for name, feat in [('continuous', rng.normal(size=(60,21))),
                   ('ties', rng.integers(-2,3,size=(60,21)).astype(float))]:
    feat[1] = feat[0]
    ref = np.linalg.norm(feat[:,None,:]-feat[None,:,:],axis=-1)
    got = pairwise_region_distance(feat, backend='cdist')
    blocked = pairwise_region_distance(feat, backend='numpy_blocked', block_size=13)
    x = rng.integers(0,3,size=60).astype(float)
    sr, pr = bcov.bcov_snp_limit(x,DY=ref)
    sc, pc = bcov.bcov_snp_limit(x,DY=got)
    sb, pb = bcov.bcov_snp_limit(x,DY=blocked)
    np.testing.assert_allclose(got,ref,rtol=3e-15,atol=1e-14)
    assert np.array_equal(blocked,ref)
    np.testing.assert_allclose([sc,pc],[sr,pr],rtol=2e-12,atol=1e-14)
    assert (sb,pb)==(sr,pr)
    checks.append({'case':name,'max_distance_difference':float(np.max(np.abs(got-ref))),
                   'cdist_bitwise_equal':bool(np.array_equal(ref,got)),
                   'blocked_bitwise_equal':True,'stat_difference':float(sc-sr),
                   'p_difference':float(pc-pr)})
print(json.dumps({'small_checks':checks}),flush=True)

feat = rng.normal(size=(10000,21))
baseline_kib = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
start = time.perf_counter()
dy = pairwise_region_distance(feat, backend='cdist')
elapsed = time.perf_counter()-start
assert dy.shape == (10000,10000) and np.all(np.diag(dy)==0)
for i in (0,1270,5000,9999):
    np.testing.assert_array_equal(dy[i],dy[:,i])
    assert np.isfinite(dy[i]).all()
print(json.dumps({'n':10000,'features':21,'synthetic':True,'method':'cdist/euclidean',
                  'distance_wall_s':elapsed,'output_gib':dy.nbytes/2**30,
                  'baseline_peak_rss_gib':baseline_kib/2**20,
                  'process_peak_rss_gib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/2**20,
                  'scope':'one process, distance only; no BDD/eigh/full GWAS'}),flush=True)
