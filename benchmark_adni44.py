"""Remote-only, additive instrumentation of GWAS_left.run_gwas."""
import os
for key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[key] = '1'
import sys, time, json, pathlib, subprocess, threading, argparse

ROOT = pathlib.Path(__file__).resolve().parent
OUT = ROOT / os.environ.get('ADNI_BENCHMARK_TAG', 'benchmark_adni44_20261008')
PREP = None

def event(phase):
    print('@@PHASE ' + json.dumps({'phase': phase, 'time': time.time()}), flush=True)

def preflight():
    import numpy as np
    import bcov as b
    from scipy.spatial.distance import cdist
    assert b.snp_backend_available()
    b.diagnose_cball()
    rng = np.random.default_rng(1145)
    x = rng.integers(0, 3, 60).astype(float)
    feat = rng.normal(size=(60, 21))
    dy = cdist(feat, feat)
    ry = b.build_rank_matrix(feat, backend='c')
    dx = np.abs(x[:, None] - x[None, :])
    stat_ref = b._bcov_from_dm(dx, dy)
    stat = b.bcov_snp_stat(x, ry, backend='c')
    prep = b.bcov_snp_limit_prep(x)
    p_ref = b.bcov_snp_limit(x, DY=dy)[1]
    p = b.bcov_snp_limit(x, DY=dy, RY=ry, stat=stat, prep=prep)[1]
    np.testing.assert_allclose(stat, stat_ref, rtol=2e-14, atol=1e-16)
    np.testing.assert_allclose(p, p_ref, rtol=2e-12, atol=1e-14)
    print(json.dumps({'stat_reference': stat_ref, 'stat_c': stat,
                      'p_reference': p_ref, 'p_optimized': p}), flush=True)
    import GWAS_left as g
    snp = np.load(ROOT / g.SNP_FILE, mmap_mode='r')
    assert snp.shape[0] == 1270
    x0 = snp[:, 0]
    assert np.isfinite(x0).all() and np.unique(x0).size > 1
    print(json.dumps({'snp_file': g.SNP_FILE, 'shape': snp.shape,
                      'first_snp_values': np.unique(x0, return_counts=True)[0].tolist(),
                      'first_snp_counts': np.unique(x0, return_counts=True)[1].tolist()}))

def compute_region(args):
    import numpy as np
    import GWAS_left as g
    import bcov as b
    j, r = args
    x = g._snp_vector(j)
    assert j == 0
    started = time.perf_counter()
    ry = g.get_region_rank(r)
    t1 = time.perf_counter()
    dy = g._region_distance(r)
    t2 = time.perf_counter()
    stat = b.bcov_snp_stat(x, ry, backend='c')
    t3 = time.perf_counter()
    before = _EIGH_SECONDS
    _, p = b.bcov_snp_limit(x, DY=dy, RY=ry, stat=stat, prep=PREP)
    t4 = time.perf_counter()
    assert np.isfinite(p) and 0 <= p <= 1
    row = {'region': r, 'pid': os.getpid(), 'rank_s': t1-started,
           'distance_s': t2-t1, 'stat_c_s': t3-t2, 'limit_s': t4-t3,
           'eigh_s': _EIGH_SECONDS-before, 'total_s': t4-started, 'stat': stat, 'p': p}
    with open(OUT / ('worker_%s.jsonl' % os.getpid()), 'a') as f:
        f.write(json.dumps(row) + '\n')
    return float(p)

_EIGH_SECONDS = 0.0

def run():
    global PREP
    import builtins, multiprocessing as mp
    import numpy as np
    import bcov as b
    import GWAS_left as g
    mp.set_start_method('fork', force=True)
    OUT.mkdir(exist_ok=True)
    g.OUT_DIR = str(OUT / 'results')
    orig_print = builtins.print
    def traced_print(*args, **kwargs):
        message = ' '.join(map(str, args))
        if 'Loading data' in message: event('load_data')
        elif 'LQD transform on' in message: event('lqd_parallel')
        elif 'Covariate adjustment (' in message: event('covariate_adjustment')
        elif 'Adjusted LQD features:' in message: event('bcov_configuration')
        elif 'SUMMARY (' in message: event('summary_and_save')
        orig_print(*args, **kwargs, flush=True)
    g.print = traced_print
    orig_load = g.load_data
    def load():
        arrays = orig_load()
        assert arrays[0].shape[0] == arrays[1].shape[0] == arrays[2].shape[0] == 1270
        event('partition')
        return arrays
    g.load_data = load
    orig_pool = mp.Pool
    def pool(*args, **kwargs):
        global PREP
        event('snp_kernel_prep')
        x = kwargs['initargs'][0][:, 0]
        PREP = b.bcov_snp_limit_prep(x)
        event('bcov_parallel')
        return orig_pool(*args, **kwargs)
    mp.Pool = pool
    orig_eigh = np.linalg.eigh
    def eigh(*args, **kwargs):
        global _EIGH_SECONDS
        started = time.perf_counter()
        value = orig_eigh(*args, **kwargs)
        _EIGH_SECONDS += time.perf_counter() - started
        return value
    np.linalg.eigh = eigh
    g.compute_one_region = compute_region
    orig_aggregate = g.rw_mtcct_report
    def aggregate(*args, **kwargs):
        event('rw_mtcct_aggregation')
        result = orig_aggregate(*args, **kwargs)
        event('snp_result_save_and_pool_shutdown')
        return result
    g.rw_mtcct_report = aggregate
    event('pipeline_start')
    g.main(['--snp_start','0','--n_snp','1','--n_process','44',
            '--backend','c','--method','limit','--n_perm','1',
            '--no_cache_rank','--no_share_perms'])
    event('finished')

def proc_sample(root_pid):
    parents = {}
    for path in pathlib.Path('/proc').iterdir():
        if not path.name.isdigit(): continue
        try:
            raw = (path / 'stat').read_text()
            parents[int(path.name)] = int(raw[raw.rfind(')')+2:].split()[1])
        except (OSError, ValueError, IndexError): pass
    pids = {root_pid}
    while True:
        extra = {pid for pid, parent in parents.items() if parent in pids}
        if extra <= pids: break
        pids |= extra
    totals = dict(pss_kib=0, rss_sum_kib=0, private_kib=0, swap_kib=0, processes=0)
    for pid in pids:
        try:
            fields = {}
            for line in pathlib.Path('/proc/%s/smaps_rollup' % pid).read_text().splitlines()[1:]:
                key, rest = line.split(':', 1)
                fields[key] = int(rest.strip().split()[0])
            totals['pss_kib'] += fields.get('Pss',0)
            totals['rss_sum_kib'] += fields.get('Rss',0)
            totals['private_kib'] += fields.get('Private_Clean',0)+fields.get('Private_Dirty',0)
            totals['swap_kib'] += fields.get('Swap',0)
            totals['processes'] += 1
        except (OSError, ValueError): pass
    mem = {}
    for line in pathlib.Path('/proc/meminfo').read_text().splitlines():
        key, rest = line.split(':',1)
        mem[key] = int(rest.strip().split()[0])
    totals['system_available_kib'] = mem['MemAvailable']
    return totals

def monitor(runner_script=None, runner_args=(), metadata_override=None,
            sample_interval_s=0.5, min_available_gib=None, max_pss_gib=None):
    OUT.mkdir(exist_ok=False)
    metadata = {
        'command': 'GWAS_left.main --snp_start 0 --n_snp 1 --n_process 44 --backend c --method limit --n_perm 1 --no_cache_rank --no_share_perms',
        'python':sys.executable, 'sample_interval_s':sample_interval_s,
        'notes':'Additive harness; C stat + shared parent SNP prep. Original LQD preserved; region distance uses GWAS_left defaults. PSS accounts for shared fork pages.'}
    metadata.update(metadata_override or {})
    (OUT / 'metadata.json').write_text(json.dumps(metadata, indent=2))
    start = time.time()
    child = subprocess.Popen([sys.executable,'-u',str(runner_script or pathlib.Path(__file__).resolve()),'run',*runner_args],
                             stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1,
                             start_new_session=True)
    (OUT / 'monitor.pid').write_text(str(os.getpid()))
    (OUT / 'pipeline.pid').write_text(str(child.pid))
    phases = [{'phase':'imports', 'time':start}]
    phase = ['imports']
    def reader():
        with open(OUT / 'run.log','w',buffering=1) as f, open(OUT / 'phases.jsonl','w',buffering=1) as e:
            for line in child.stdout:
                f.write(line)
                if line.startswith('@@PHASE '):
                    item = json.loads(line[len('@@PHASE '):])
                    phases.append(item)
                    phase[0] = item['phase']
                    e.write(json.dumps(item)+'\n')
    thread = threading.Thread(target=reader)
    thread.start()
    samples=[]
    safety_failure=None
    def save_status(value):
        tmp=OUT/'status.json.tmp'
        tmp.write_text(json.dumps(value,indent=2))
        tmp.replace(OUT/'status.json')
    with open(OUT / 'memory.jsonl','w',buffering=1) as f:
        while child.poll() is None:
            s = {'elapsed_s':time.time()-start, 'time':time.time(), 'phase':phase[0], **proc_sample(child.pid)}
            samples.append(s)
            f.write(json.dumps(s)+'\n')
            peak=max(x['pss_kib'] for x in samples)/2**20
            save_status({'state':'running','updated_at':time.time(),'pid':child.pid,
                         'phase':phase[0],'elapsed_s':time.time()-start,
                         'peak_pss_gib':peak,'current_pss_gib':s['pss_kib']/2**20,
                         'available_gib':s['system_available_kib']/2**20,
                         'processes':s['processes']})
            if min_available_gib is not None and s['system_available_kib']/2**20 < min_available_gib:
                safety_failure='system available memory below safety reserve'
            if max_pss_gib is not None and s['pss_kib']/2**20 > max_pss_gib:
                safety_failure='task PSS above safety budget'
            if safety_failure:
                import signal
                (OUT/'safety_failure.json').write_text(json.dumps({'reason':safety_failure,'sample':s}))
                try:
                    os.killpg(child.pid,signal.SIGTERM)
                    child.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    os.killpg(child.pid,signal.SIGKILL)
                    child.wait()
                except ProcessLookupError:
                    child.wait()
                break
            time.sleep(sample_interval_s)
    thread.join()
    end=time.time()
    stages=[]
    for a,c in zip(phases,phases[1:]+[{'time':end}]):
        ss=[s for s in samples if a['time'] <= s['time'] < c['time']]
        stages.append({'phase':a['phase'],'wall_s':c['time']-a['time'],
                       'peak_pss_gib':max((s['pss_kib'] for s in ss),default=0)/2**20,
                       'peak_private_gib':max((s['private_kib'] for s in ss),default=0)/2**20,
                       'peak_rss_sum_gib':max((s['rss_sum_kib'] for s in ss),default=0)/2**20,
                       'samples':len(ss)})
    report={'exit_code':child.returncode,'total_wall_s':end-start,'stages':stages,
            'peak_pss_gib':max((s['pss_kib'] for s in samples),default=0)/2**20,
            'min_available_gib':min((s['system_available_kib'] for s in samples),default=0)/2**20}
    (OUT / 'resource_summary.json').write_text(json.dumps(report,indent=2))
    save_status({'state':'completed' if child.returncode==0 else 'failed',
                 'updated_at':end,'exit_code':child.returncode,'phase':phase[0],
                 'elapsed_s':end-start,'peak_pss_gib':report['peak_pss_gib'],
                 'safety_failure':safety_failure})
    print(json.dumps(report,indent=2),flush=True)

if __name__ == '__main__':
    {'preflight':preflight, 'run':run, 'monitor':monitor}[sys.argv[1]]()
