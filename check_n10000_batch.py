"""Read-only compact status for the protected ADNI batch."""
import argparse,json,pathlib,subprocess,time
p=argparse.ArgumentParser()
p.add_argument('--output_dir',default='bench_n10000_p16_20261009')
p.add_argument('--unit',default='adni-n10000-p16-20261009.service')
a=p.parse_args()
root=pathlib.Path(a.output_dir)
result={'checked_at':time.time(),'output_dir':str(root.resolve())}
for name in ('status','configuration','resource_summary','safety_failure','single_snp_estimate'):
    file=root/(name+'.json')
    if file.exists():
        try: result[name]=json.loads(file.read_text())
        except Exception as e: result[name+'_read_error']=str(e)
done=[]
for file in sorted(root.glob('task_done_*.json')):
    try: done.append(json.loads(file.read_text()))
    except Exception: pass
result['completed_regions']=len(done)
result['completed_worker_pids']=len({x['pid'] for x in done})
command=['systemctl','--user','show',a.unit]
for key in ('ActiveState','SubState','Result','MemoryCurrent','MemoryPeak','MemoryMax','MemoryHigh','MemorySwapMax','ExecMainStatus','ControlGroup'):
    command.append('--property='+key)
proc=subprocess.run(command,capture_output=True,text=True,timeout=15)
result['service']={line.split('=',1)[0]:line.split('=',1)[1] for line in proc.stdout.splitlines() if '=' in line}
result['service_query_exit']=proc.returncode
cg=result['service'].get('ControlGroup','')
if cg:
    for name in ('memory.events','memory.current','memory.peak','memory.max','memory.high','memory.swap.max'):
        file=pathlib.Path('/sys/fs/cgroup'+cg)/name
        if file.exists(): result['cgroup_'+name]=file.read_text().strip()
status=result.get('status',{})
if 'updated_at' in status: result['status_age_s']=time.time()-status['updated_at']
print(json.dumps(result,indent=2))
