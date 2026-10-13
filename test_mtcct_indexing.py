"""Regression tests using an independent four-neighbour grid traversal."""
import itertools
import numpy as np
import GWAS_left as g
from rwspm import RW_MTCCT,CCT_chisq,slide_width_step

def reference(p,nrows,ncols,alpha):
    sig=set(np.flatnonzero(p<alpha).tolist())
    if not sig:return 1.0,0,0
    start=min(sig,key=lambda i:(p[i],i))
    selected={start};stack=[start]
    while stack:
        i=stack.pop();r,c=divmod(i,ncols)
        for rr,cc in ((r-1,c),(r+1,c),(r,c-1),(r,c+1)):
            if 0<=rr<nrows and 0<=cc<ncols:
                j=rr*ncols+cc
                if j in sig and j not in selected:
                    selected.add(j);stack.append(j)
    weights=np.zeros(len(p));weights[list(selected)]=1/(len(p)-len(sig)+len(selected))
    return CCT_chisq(p,weights),len(selected),len(sig)

def check(p,nrows,ncols,alpha=0.01):
    expected=reference(p,nrows,ncols,alpha)
    actual=g.rw_mtcct_report(p,nrows,ncols,alpha0=alpha)
    assert actual[1:]==expected[1:],(nrows,ncols,actual,expected)
    np.testing.assert_allclose(actual[0],expected[0],rtol=1e-12,atol=1e-15)
    np.testing.assert_allclose(RW_MTCCT(p,nrows,ncols,threshold=alpha),expected[0],rtol=1e-12,atol=1e-15)

def main():
    part=slide_width_step(3,11,150,100)
    assert (part['width_x'],part['width_y'])==(48,31)
    p=np.ones(1488);p[30]=1e-7;p[31]=1e-20;p[32]=2e-7
    check(p,48,31,0.001)
    assert g.rw_mtcct_report(p,48,31)[1:]==(2,3)
    cases=1
    # Exhaustive subsets cover row-wrap, vertical links, isolated regions,
    # no signals, all signals, ties and square/single-axis grids.
    for nr,nc in ((2,3),(3,2),(2,2),(1,4),(4,1)):
        for bits in itertools.product((False,True),repeat=nr*nc):
            p=np.where(bits,0.001,0.5)
            check(p,nr,nc);cases+=1
    rng=np.random.default_rng(1145)
    for nr,nc in ((3,4),(4,3),(3,3),(48,31)):
        for _ in range(12):
            p=rng.uniform(0.001,0.1,nr*nc)
            check(p,nr,nc,0.025);cases+=1
    print(f'PASS: {cases} grid cases; report and RW_MTCCT agree with independent traversal')
if __name__=='__main__':main()
