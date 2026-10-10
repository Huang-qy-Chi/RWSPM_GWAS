"""Verify every generated row against its recorded real ADNI source row."""
import json
import sys
from pathlib import Path

import numpy as np
import GWAS_left as g
from generate_adni_data import load_generated_adni

root = Path(sys.argv[1]).resolve()
adni, cov, hipp = load_generated_adni(root)
indices = np.load(root/'subject_indices.npy', allow_pickle=False)
columns = np.load(root/'snp_indices.npy', allow_pickle=False)
manifest = json.loads((root/'manifest.json').read_text())
source_adni = np.load(Path(g.HERE)/g.SNP_FILE, mmap_mode='r', allow_pickle=False)
source_cov = g.load_rdata_matrix(str(Path(g.HERE)/g.COV_FILE),g.COV_KEY)
source_hipp = g.load_rdata_matrix(str(Path(g.HERE)/g.HIPP_FILE),g.HIPP_KEY)
assert indices.shape == (adni.shape[0],)
assert columns.shape == (adni.shape[1],)
assert indices.min() >= 0 and indices.max() < source_adni.shape[0]
for start in range(0,len(indices),256):
    stop = min(start+256,len(indices))
    batch = indices[start:stop]
    np.testing.assert_array_equal(adni[start:stop],source_adni[np.ix_(batch,columns)])
    np.testing.assert_array_equal(cov[start:stop],source_cov[batch])
    np.testing.assert_array_equal(hipp[start:stop],source_hipp[batch])
unique,counts = np.unique(indices,return_counts=True)
assert len(unique) == manifest['n_unique_source_subjects']
summary = {'all_rows_match_source':True, 'adni_shape':list(adni.shape),
           'cov_shape':list(cov.shape), 'hipp_shape':list(hipp.shape),
           'unique_source_subjects':len(unique),'largest_source_multiplicity':int(counts.max()),
           'snp_source_columns':columns.tolist(),
           'matrix_gib':sum(x.nbytes for x in (adni,cov,hipp))/2**30}
(root/'validation.json').write_text(json.dumps(summary,indent=2)+'\n')
print(json.dumps(summary,indent=2))
