"""Focused checks for row alignment, reproducibility, and disk output."""
import tempfile
from pathlib import Path

import numpy as np
from generate_adni_data import generate_adni_resample, load_generated_adni


def main():
    adni = np.arange(24).reshape(6,4)
    cov = np.column_stack((np.ones(6), np.arange(6)))
    hipp = np.arange(42).reshape(6,7)
    first = generate_adni_resample(adni,cov,hipp,100,seed=1145,snp_indices=[3,0],batch_size=7)
    repeat = generate_adni_resample(adni,cov,hipp,100,seed=1145,snp_indices=[3,0],batch_size=31)
    different = generate_adni_resample(adni,cov,hipp,100,seed=1146,snp_indices=[3,0])
    indices = first['subject_indices']
    assert np.unique(indices).size < len(indices)
    assert not np.array_equal(indices,different['subject_indices'])
    for key in ('adni','cov','hipp','subject_indices'):
        np.testing.assert_array_equal(first[key],repeat[key])
    np.testing.assert_array_equal(first['adni'],adni[np.ix_(indices,[3,0])])
    np.testing.assert_array_equal(first['cov'],cov[indices])
    np.testing.assert_array_equal(first['hipp'],hipp[indices])
    with tempfile.TemporaryDirectory(prefix='adni_generator_test_') as temporary:
        directory = Path(temporary)/'generated'
        disk = generate_adni_resample(adni,cov,hipp,100,seed=1145,
                                     snp_indices=[3,0],output_dir=directory,batch_size=9)
        for key,array in zip(('adni','cov','hipp'),load_generated_adni(directory)):
            assert isinstance(array,np.memmap) and not array.flags.writeable
            np.testing.assert_array_equal(array,first[key])
        np.testing.assert_array_equal(np.load(directory/'subject_indices.npy'),indices)
        try:
            generate_adni_resample(adni,cov,hipp,10,output_dir=directory)
        except FileExistsError:
            pass
        else:
            raise AssertionError('Existing dataset was overwritten')
        # Release mapped files before TemporaryDirectory cleanup (also Windows).
        disk.clear()
        del array
    for bad in (0,-1,1.5,True):
        try:
            generate_adni_resample(adni,cov,hipp,bad)
        except ValueError:
            pass
        else:
            raise AssertionError('Invalid n_samples accepted')
    try:
        generate_adni_resample(adni,cov[:-1],hipp,10)
    except ValueError:
        pass
    else:
        raise AssertionError('Misaligned inputs accepted')
    print('PASS: aligned rows, selected SNP mapping, seed/batch reproducibility, read-only mmap, no overwrite, input validation')


if __name__=='__main__':
    main()
