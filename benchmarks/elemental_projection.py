"""Compare CPU meson55 projection with an explicitly archived generator.py."""
import argparse
from contextlib import nullcontext
import hashlib
import importlib.util
import json
from pathlib import Path
import time
from unittest.mock import patch

import numpy as np

from lqcdweave.operators.single_hadron.api import meson55_preset, momDict_meson55, mom_dict_to_list
from lqcdweave.production.elemental.api import ElementalKernel
from lqcdweave.production.elemental import projection


class Handle:
    def __init__(self, array):
        self.array = array

    def load(self, key):
        return self.array


def benchmark(reference_generator=None, *, reference_projection=None, weight_bytes=None,
              size=6, ne=24, repeats=3):
    if weight_bytes is not None and weight_bytes <= 0:
        raise ValueError('weight_bytes must be positive')
    if weight_bytes is not None and reference_projection is None:
        raise ValueError('weight_bytes requires reference_projection so both versions use the same limit')
    if repeats < 1:
        raise ValueError('repeats must be positive')
    reference_path = reference_projection or reference_generator
    name = 'lqcdweave.production.elemental._benchmark_reference'
    spec = importlib.util.spec_from_file_location(name, reference_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    rng = np.random.default_rng(20261015)
    raw = rng.normal(size=(1,size,size,size,4,3,3)) + 1j*rng.normal(size=(1,size,size,size,4,3,3))
    gauge = np.linalg.qr(raw)[0]
    vectors = (rng.normal(size=(1,ne,size,size,size,3))
               + 1j*rng.normal(size=(1,ne,size,size,size,3))) / np.sqrt(3*size**3)
    table = meson55_preset()
    rows = [dict(id=i,terms=table[row]['terms']) for i,row in enumerate((16,22,23,24))]
    paths = list(dict.fromkeys(tuple(term['path']) for row in rows for term in row['terms']))
    args = dict(paths=paths,path_version='v2',momentum_list=mom_dict_to_list(momDict_meson55))
    reference_cls = ElementalKernel if reference_projection else module._ElementalEngine
    engines = {name: cls((size,size,size,1),Handle(gauge),Handle(vectors),**args).load('benchmark')
               for name,cls in [('reference',reference_cls),('candidate',ElementalKernel)]}
    outputs, timings = {}, {name: [] for name in engines}
    def contexts(name):
        # Same warmup and timed boundary: projection only, no input or file I/O.
        selected = module if name == 'reference' and reference_projection else projection
        implementation = (patch.object(projection, 'project_rows', module.project_rows)
                          if name == 'reference' and reference_projection else nullcontext())
        capacity = patch.object(selected, '_WEIGHT_BYTES', weight_bytes) if weight_bytes is not None else nullcontext()
        return implementation, capacity
    for name, engine in engines.items():
        implementation, capacity = contexts(name)
        with implementation, capacity:
            outputs[name] = engine.calc_path_blocks(0,rows=rows)['v2v'].copy()
    for repeat in range(repeats):
        # Alternate AB/BA to reduce ordering bias from host load and temperature.
        for name in (tuple(engines) if repeat % 2 == 0 else tuple(reversed(engines))):
            implementation, capacity = contexts(name)
            with implementation, capacity:
                start = time.perf_counter()
                engines[name].calc_path_blocks(0,rows=rows)
                timings[name].append(time.perf_counter()-start)
    np.testing.assert_allclose(outputs['candidate'],outputs['reference'],atol=2e-11,rtol=2e-11)
    return dict(size=size,ne=ne,momenta=33,rows=[16,22,23,24],paths=len(paths),
        numpy_version=np.__version__,repeats=repeats,seconds=timings,
        speedup=float(np.median(timings['reference'])/np.median(timings['candidate'])),
        max_absolute_error=float(np.max(abs(outputs['candidate']-outputs['reference']))),
        reference_source_sha256=hashlib.sha256(Path(reference_path).read_bytes()).hexdigest(),
        candidate_projection_sha256=hashlib.sha256(Path(projection.__file__).read_bytes()).hexdigest(),
        weight_bytes=weight_bytes or projection._WEIGHT_BYTES,
        sampling_order='alternating AB/BA after one warmup per version',
        timing_boundary='projection only; same input, precision, process, and warmup')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    reference = parser.add_mutually_exclusive_group(required=True)
    reference.add_argument('--reference-generator',type=Path)
    reference.add_argument('--reference-projection',type=Path)
    parser.add_argument('--weight-mib',type=float)
    parser.add_argument('--size',type=int,default=6)
    parser.add_argument('--ne',type=int,default=24)
    parser.add_argument('--repeats',type=int,default=3)
    parser.add_argument('--output',type=Path)
    args = parser.parse_args()
    result = benchmark(args.reference_generator,reference_projection=args.reference_projection,
        weight_bytes=None if args.weight_mib is None else int(args.weight_mib*1024**2),
        size=args.size,ne=args.ne,repeats=args.repeats)
    content = json.dumps(result,indent=2)+'\n'
    if args.output:
        args.output.write_text(content,encoding='utf-8')
    print(content,end='')
