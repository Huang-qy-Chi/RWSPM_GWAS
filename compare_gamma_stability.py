"""Nested 199/999 Gamma fit stability and exploratory held-out null diagnostics."""
import argparse,csv,json,pathlib
def compare(output,prior='bench_n10000_gamma199_p16_20261009',limit='bench_n10000_p16_20261009'):
    import numpy as np
    from scipy import stats
    from gammafit import gamma_moment_match
    root=pathlib.Path(output)
    oldroot=pathlib.Path(prior)
    rows=json.loads((root/'batch_results.json').read_text())
    old={x['region']:x for x in json.loads((oldroot/'batch_results.json').read_text())}
    limits={x['region']:x for x in json.loads((pathlib.Path(limit)/'batch_results.json').read_text())}
    comparisons=[]
    for r in rows:
        i=r['region']
        null=np.load(root/f'null_stats_{i}.npy')
        first=np.load(oldroot/f'null_stats_{i}.npy')
        assert len(null)==999 and np.array_equal(null[:199],first)
        np.testing.assert_allclose(r['stat'],old[i]['stat'],rtol=1e-10,atol=1e-14)
        record={'region':i,'stat':r['stat'],'limit_p':limits[i]['p'],
                'gamma199_p':old[i]['p'],'gamma999_p':r['p'],
                'shape199':old[i]['shape'],'shape999':r['shape'],
                'scale199':old[i]['scale'],'scale999':r['scale'],
                'mean_relative_change':r['mean']/old[i]['mean']-1,
                'variance_relative_change':r['variance']/old[i]['variance']-1,
                'delta_log10_p_999_minus_199':float(stats.gamma.logsf(r['stat'],a=r['shape'],scale=r['scale'])/np.log(10)-stats.gamma.logsf(old[i]['stat'],a=old[i]['shape'],scale=old[i]['scale'])/np.log(10)),
                'observed_exceedances_999':int(np.sum(null>=r['stat'])),
                'empirical_permutation_p_999':float((1+np.sum(null>=r['stat']))/1000)}
        for count in (199,499,999):
            fit=gamma_moment_match(null[:count])
            record[f'log10_p_fit_{count}']=float(stats.gamma.logsf(r['stat'],a=fit['shape'],scale=fit['scale'])/np.log(10))
        fit=gamma_moment_match(null[:499])
        held=stats.gamma.sf(null[499:],a=fit['shape'],scale=fit['scale'])
        for alpha in (0.05,0.01):
            record[f'heldout_500_rejection_rate_{alpha}']=float(np.mean(held<alpha))
        comparisons.append(record)
    with open(root/'pvalue_stability.csv','w',newline='',encoding='utf-8') as file:
        writer=csv.DictWriter(file,fieldnames=list(comparisons[0]))
        writer.writeheader();writer.writerows(comparisons)
    result={'regions':comparisons,'prefix_199_null_statistics_verified':True,
            'median_abs_delta_log10_p':float(np.median([abs(x['delta_log10_p_999_minus_199']) for x in comparisons])),
            'max_abs_delta_log10_p':float(max(abs(x['delta_log10_p_999_minus_199']) for x in comparisons)),
            'gamma999_smaller_than_limit_count':sum(x['gamma999_p']<x['limit_p'] for x in comparisons),
            'note':'Observed signal p-values cannot establish null calibration. Held-out permutations diagnose Gamma fit at 0.05/0.01 conditionally on this SNP and regions; shared indices create regional dependence. 999 permutations cannot validate extreme GWAS tails or establish whether limit is conservative.'}
    (root/'pvalue_stability.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    lines=['# Gamma 199/999次置换稳定性与limit比较','',
           '前199组置换索引和每区前199个零统计量与上次完全一致；同一数据、SNP、16区域和16进程。','',
           '| 区域(0起) | limit p | Gamma199 p | Gamma999 p | log10(p999/p199) |','|---|---:|---:|---:|---:|']
    for x in comparisons:
        lines.append(f'| {x["region"]} | {x["limit_p"]:.3e} | {x["gamma199_p"]:.3e} | {x["gamma999_p"]:.3e} | {x["delta_log10_p_999_minus_199"]:.3f} |')
    lines+=['',f'log10 p变化绝对值：中位数{result["median_abs_delta_log10_p"]:.3f}，最大{result["max_abs_delta_log10_p"]:.3f}。',
            '参数稳定性、199/499/999嵌套拟合、观测统计量的置换超过次数与499训练/500留出诊断见CSV。',
            result['note']]
    (root/'pvalue_stability.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(json.dumps({k:v for k,v in result.items() if k!='regions'},indent=2))
if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output_dir',required=True)
    parser.add_argument('--prior',default='bench_n10000_gamma199_p16_20261009')
    parser.add_argument('--limit',default='bench_n10000_p16_20261009')
    args=parser.parse_args();compare(args.output_dir,args.prior,args.limit)
