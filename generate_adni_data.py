"""Aligned ADNI bootstrap data for performance tests, not inference.

Keep SNP, covariate and image rows together. No jitter, fitted model, or new
independent biological subjects are introduced. Disk outputs use float64 NPY
files written in row batches; manifest.json is written only after completion.
"""
import argparse
import json
import operator
from pathlib import Path

import numpy as np


def _positive_integer(value, name):
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be a positive integer")
    try:
        result = operator.index(value)
    except TypeError as exc:
        raise ValueError(f"{name} must be a positive integer") from exc
    if result < 1:
        raise ValueError(f"{name} must be a positive integer")
    return result


def generate_adni_resample(adni, cov, hipp, n_samples, *, seed=1145,
                           snp_indices=None, output_dir=None, batch_size=256,
                           provenance=None):
    """Resample aligned subject rows with replacement to any positive n.

    Parameters
    ----------
    adni, cov, hipp : 2D numeric arrays
        Original aligned SNP (n,p), covariate (n,q), image (n,v) matrices.
    n_samples : positive int
        Number of output subjects. This does not create independent subjects.
    seed : int
        NumPy default_rng seed, recorded in the manifest.
    snp_indices : sequence of int or None
        Original SNP columns in output order. None retains all columns.
    output_dir : path or None
        New directory for NPY outputs; existing directories are refused.
        None returns RAM arrays. A directory without manifest.json is incomplete.
    batch_size : positive int
        Rows per copy batch; caps temporary copying for disk output.
    provenance : JSON-compatible dict or None
        Optional source filenames/hemisphere information.

    Returns
    -------
    dict
        adni, cov, hipp, subject_indices, snp_indices, manifest. Disk outputs
        are returned as read-only memory maps.
    """
    n_samples = _positive_integer(n_samples, 'n_samples')
    batch_size = _positive_integer(batch_size, 'batch_size')
    source = tuple(np.asarray(x) for x in (adni, cov, hipp))
    if any(x.ndim != 2 or x.shape[0] == 0 or x.shape[1] == 0 for x in source):
        raise ValueError('Inputs must be nonempty two-dimensional matrices')
    n_source = source[0].shape[0]
    if any(x.shape[0] != n_source for x in source):
        raise ValueError('SNP, covariate and image rows must be aligned')
    for x in source:
        if not np.issubdtype(x.dtype, np.number) or np.iscomplexobj(x):
            raise ValueError('Inputs must be real numeric matrices')
        for start in range(0, n_source, batch_size):
            if not np.isfinite(x[start:start+batch_size]).all():
                raise ValueError('Inputs contain non-finite values')
    if snp_indices is None:
        columns = np.arange(source[0].shape[1], dtype=np.int64)
    else:
        raw = np.asarray(snp_indices)
        if raw.ndim != 1 or raw.size == 0 or not np.issubdtype(raw.dtype, np.integer):
            raise ValueError('snp_indices must be a nonempty integer sequence')
        if np.any(raw < 0) or np.any(raw >= source[0].shape[1]):
            raise ValueError('SNP column index out of range')
        columns = raw.astype(np.int64)
        if np.unique(columns).size != columns.size:
            raise ValueError('SNP column indices must be distinct')
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, n_source, size=n_samples, dtype=np.int64)
    shapes = ((n_samples, columns.size), (n_samples, source[1].shape[1]),
              (n_samples, source[2].shape[1]))
    keys = ('adni', 'cov', 'hipp')
    manifest = {
        'format_version': 1, 'mode': 'aligned_bootstrap',
        'purpose': 'performance_testing_only', 'replacement': True,
        'seed': int(seed), 'bit_generator': type(rng.bit_generator).__name__,
        'numpy_version': np.__version__, 'n_source': n_source,
        'n_samples': n_samples, 'n_unique_source_subjects': int(np.unique(indices).size),
        'source_shapes': {k: list(x.shape) for k,x in zip(keys,source)},
        'output_shapes': {k: list(shape) for k,shape in zip(keys,shapes)},
        'dtype': 'float64', 'batch_size': batch_size,
        'snp_column_mapping': 'snp_indices.npy',
        'subject_row_mapping': 'subject_indices.npy',
        'files': {k: k+'.npy' for k in keys},
        'matrix_bytes': int(sum(np.prod(shape, dtype=np.int64)*8 for shape in shapes)),
        'provenance': provenance or {},
        'limitations': 'Duplicate subjects are present; not an independent n-subject sample. No null/alternative model is imposed.'}
    # Fail on non-serializable metadata before creating any output files.
    manifest_text = json.dumps(manifest, indent=2, ensure_ascii=False)
    root = None if output_dir is None else Path(output_dir).resolve()
    if root is None:
        result = {k: np.empty(shape, dtype=np.float64) for k,shape in zip(keys,shapes)}
    else:
        root.mkdir(parents=True, exist_ok=False)
        result = {k: np.lib.format.open_memmap(root/(k+'.npy'), mode='w+',
                                               dtype=np.float64, shape=shape)
                  for k,shape in zip(keys,shapes)}
    for start in range(0, n_samples, batch_size):
        stop = min(start+batch_size, n_samples)
        batch = indices[start:stop]
        result['adni'][start:stop] = source[0][np.ix_(batch, columns)]
        result['cov'][start:stop] = source[1][batch]
        result['hipp'][start:stop] = source[2][batch]
    if root is not None:
        for array in result.values():
            array.flush()
        result.clear()
        del array
        np.save(root/'subject_indices.npy', indices, allow_pickle=False)
        np.save(root/'snp_indices.npy', columns, allow_pickle=False)
        temporary = root/'manifest.json.tmp'
        temporary.write_text(manifest_text+'\n', encoding='utf-8')
        temporary.replace(root/'manifest.json')
        result = {k: np.load(root/(k+'.npy'), mmap_mode='r', allow_pickle=False) for k in keys}
    result.update(subject_indices=indices, snp_indices=columns, manifest=manifest)
    return result


def load_generated_adni(directory):
    """Load a completed generated dataset as read-only memory maps."""
    root = Path(directory).resolve()
    manifest = json.loads((root/'manifest.json').read_text(encoding='utf-8'))
    if manifest['format_version'] != 1 or manifest['mode'] != 'aligned_bootstrap':
        raise ValueError('Unsupported generated ADNI format')
    arrays = tuple(np.load(root/(k+'.npy'), mmap_mode='r', allow_pickle=False)
                   for k in ('adni','cov','hipp'))
    for key, array in zip(('adni','cov','hipp'), arrays):
        if list(array.shape) != manifest['output_shapes'][key] or array.dtype != np.float64:
            raise ValueError(f'Unexpected shape/dtype for {key}')
    return arrays


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--n_samples', type=int, required=True)
    parser.add_argument('--output_dir', required=True)
    parser.add_argument('--seed', type=int, default=1145)
    parser.add_argument('--n_snp', type=int, default=1,
                        help='number of original SNP columns from snp_start; default 1')
    parser.add_argument('--snp_start', type=int, default=0)
    parser.add_argument('--batch_size', type=int, default=256)
    args = parser.parse_args()
    import GWAS_left as g
    if args.n_snp < 1 or args.snp_start < 0:
        parser.error('n_snp must be positive and snp_start must be nonnegative')
    adni = np.load(Path(g.HERE)/g.SNP_FILE, mmap_mode='r', allow_pickle=False)
    cov = g.load_rdata_matrix(str(Path(g.HERE)/g.COV_FILE), g.COV_KEY)
    hipp = g.load_rdata_matrix(str(Path(g.HERE)/g.HIPP_FILE), g.HIPP_KEY)
    generated = generate_adni_resample(
        adni, cov, hipp, args.n_samples, seed=args.seed,
        snp_indices=np.arange(args.snp_start,args.snp_start+args.n_snp),
        output_dir=args.output_dir, batch_size=args.batch_size,
        provenance={'side':g.SIDE,'snp_file':g.SNP_FILE,'cov_file':g.COV_FILE,
                    'hipp_file':g.HIPP_FILE,'image_shape':[g.M1,g.M2]})
    print(json.dumps(generated['manifest'], indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
