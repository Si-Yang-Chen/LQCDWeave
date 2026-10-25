"""Reduce an exact sparse whole baryon against its one closed merged leg.

Only address-free shape/support programs and compiled code are cached.
Preparation owns the current inputs and device descriptors until close().
"""
from dataclasses import dataclass
from functools import lru_cache

import numpy as np


CPU_MIN_MODES = 8


@dataclass(frozen=True)
class ScalarSumRecipe:
    permutation: tuple
    modes: tuple
    width: int
    output_shape: tuple
    leg_lane: bool
    spatial_lanes: tuple
    support: tuple


@lru_cache(maxsize=256)
def _recipe(equation, shapes, supports):
    if len(equation.inputs) != 2 or len(equation.output) > 1:
        return None
    core, leg = equation.inputs
    lane = equation.output[0] if equation.output else None
    physical = core[-6:]
    if (len(core) not in (6, 7) or len(set(physical)) != 6
            or (len(core) == 7 and (lane is None or core[0] != lane))
            or lane in physical):
        return None
    leg_shape = shapes[-1]
    leg_lane = lane is not None and len(leg) == 7 and leg[0] == lane
    if len(leg) != 6 + int(leg_lane) or len(leg_shape) != len(leg):
        return None
    body = leg[1:] if leg_lane else leg
    if len(set(body)) != 6 or set(body) != set(physical):
        return None
    canonical = tuple(axis for pair in zip(physical[:3], physical[3:]) for axis in pair)
    permutation = tuple(leg.index(axis) for axis in canonical)
    ordered_shape = tuple(leg_shape[index] for index in permutation)
    if ordered_shape[::2] != (4, 4, 4) or any(size <= 0 for size in ordered_shape):
        return None
    modes = ordered_shape[1::2]
    widths = [leg_shape[0]] if leg_lane else []
    spatial_lanes = []
    for spin_shape, spatial_shape in zip(shapes[:-1:2], shapes[1:-1:2]):
        if spin_shape != (4, 4, 4) or len(spatial_shape) not in (3, 4):
            return None
        dynamic = len(spatial_shape) == 4
        if spatial_shape[-3:] != modes or (dynamic and (lane is None or len(core) != 7)):
            return None
        if dynamic:
            widths.append(spatial_shape[0])
        spatial_lanes.append(dynamic)
    if lane is None:
        if widths:
            return None
        width, output_shape = 1, ()
    else:
        if not widths or widths[0] <= 0 or len(set(widths)) != 1:
            return None
        width, output_shape = widths[0], (widths[0],)
    support = tuple(sorted(set().union(*map(set, supports))))
    if (not support or any(len(entry) != 3 or any(index not in range(4) for index in entry)
                           for entry in support)):
        return None
    return ScalarSumRecipe(permutation, modes, width, output_shape, leg_lane,
                           tuple(spatial_lanes), support)


_CUDA = r'''
#include <cupy/complex.cuh>
typedef complex<REAL> C;
extern "C" __global__ void partial(const long long* m,const C* coefficients,
 const long long na,const long long nb,const long long nc,
 const int o0,const int o1,const int o2,const int groups,C* out) {
 const int lane=blockIdx.y;
 const long long* ss=m+2*PARTS+1;
 const long long* ps=m+5*PARTS+1;
 const long long* ls=m+9*PARTS+1;
 const long long* spins=m+9*PARTS+8;
 const C* leg=reinterpret_cast<const C*>((unsigned long long)m[2*PARTS]);
 C total(0,0);
 for(long long q=(long long)blockIdx.x*blockDim.x+threadIdx.x;
     q<na*nb*nc;q+=(long long)blockDim.x*gridDim.x) {
  long long n1=o1==0?na:(o1==1?nb:nc),n2=o2==0?na:(o2==1?nb:nc);
  long long d0=q/(n1*n2),d1=(q/n2)%n1,d2=q%n2;
  long long a=o0==0?d0:(o1==0?d1:d2);
  long long b=o0==1?d0:(o1==1?d1:d2);
  long long c=o0==2?d0:(o1==2?d1:d2);
  C phis[PARTS];
  #pragma unroll
  for(int p=0;p<PARTS;++p) {
   const C* phi=reinterpret_cast<const C*>((unsigned long long)m[2*p+1]);
   phis[p]=phi[lane*ps[4*p]+a*ps[4*p+1]+b*ps[4*p+2]+c*ps[4*p+3]];
  }
  long long base=lane*ls[0]+a*ls[2]+b*ls[4]+c*ls[6];
  for(int g=0;g<groups;++g) {
   long long i=spins[3*g],j=spins[3*g+1],k=spins[3*g+2];
   C phi(0,0);
   #pragma unroll
   for(int p=0;p<PARTS;++p) {
    const C* spin=reinterpret_cast<const C*>((unsigned long long)m[2*p]);
    phi+=coefficients[p]*spin[i*ss[3*p]+j*ss[3*p+1]+k*ss[3*p+2]]*phis[p];
   }
   total+=phi*leg[base+i*ls[1]+j*ls[3]+k*ls[5]];
  }
 }
 __shared__ REAL sr[256],si[256];
 sr[threadIdx.x]=total.real();si[threadIdx.x]=total.imag();__syncthreads();
 for(int step=128;step;step/=2) {
  if(threadIdx.x<step){sr[threadIdx.x]+=sr[threadIdx.x+step];si[threadIdx.x]+=si[threadIdx.x+step];}
  __syncthreads();
 }
 if(threadIdx.x==0)out[lane*gridDim.x+blockIdx.x]=C(sr[0],si[0]);
}
extern "C" __global__ void finish(const C* partials,const int count,C* out) {
 const int lane=blockIdx.x;
 C total(0,0);for(int i=threadIdx.x;i<count;i+=256)total+=partials[lane*count+i];
 __shared__ REAL sr[256],si[256];
 sr[threadIdx.x]=total.real();si[threadIdx.x]=total.imag();__syncthreads();
 for(int step=128;step;step/=2) {
  if(threadIdx.x<step){sr[threadIdx.x]+=sr[threadIdx.x+step];si[threadIdx.x]+=si[threadIdx.x+step];}
  __syncthreads();
 }
 if(threadIdx.x==0)out[lane]=C(sr[0],si[0]);
}
'''


@lru_cache(maxsize=32)
def _gpu_program(dtype, parts, device):
    del device  # Device identity separates compiled modules, never input owners.
    import cupy
    code = _CUDA.replace('REAL', 'float' if np.dtype(dtype) == np.dtype('complex64') else 'double')
    code = code.replace('PARTS', str(parts))
    module = cupy.RawModule(code=code, options=('--std=c++11', '--fmad=false'))
    return module.get_function('partial'), module.get_function('finish')


class PreparedScalarSum:
    """Current operation only; compute returns an independent result owner."""
    def __init__(self, recipe, components, leg, backend, dtype):
        self.recipe, self.components, self.leg = recipe, components, leg
        self.backend, self.dtype, self.closed = backend, dtype, False
        self.metadata = self.coefficients = self.kernels = None
        if backend.__name__ == 'numpy':
            return
        unit = dtype.itemsize
        pointers, spins, spatials = [], [], []
        for (spin, spatial, _), dynamic in zip(components, recipe.spatial_lanes):
            pointers.extend((spin.data.ptr, spatial.data.ptr))
            spins.extend(stride // unit for stride in spin.strides)
            spatials.extend(((spatial.strides[0] // unit) if dynamic else 0,
                             *(stride // unit for stride in spatial.strides[-3:])))
        pointers.append(leg.data.ptr)
        legs = [(leg.strides[0] // unit) if recipe.leg_lane else 0]
        legs.extend(leg.strides[index] // unit for index in recipe.permutation)
        descriptor = (*pointers, *spins, *spatials, *legs,
                      *(index for row in recipe.support for index in row))
        self.metadata = backend.asarray(descriptor, dtype='int64')
        self.coefficients = backend.asarray([part[2] for part in components], dtype=dtype)
        self.order = tuple(sorted(range(3), key=lambda index: abs(legs[2*index+2]), reverse=True))
        self.blocks = min(4096, max(1, (int(np.prod(recipe.modes)) + 255) // 256))
        self.kernels = _gpu_program(dtype.str, len(components), int(backend.cuda.runtime.getDevice()))

    def compute(self):
        if self.closed:
            raise RuntimeError('scalar baryon preparation has already been released')
        recipe, backend = self.recipe, self.backend
        result = backend.empty(recipe.output_shape, dtype=self.dtype)
        if backend.__name__ == 'numpy':
            permutation = ((0,) if recipe.leg_lane else ()) + recipe.permutation
            leg = self.leg.transpose(permutation)
            for lane in range(recipe.width):
                current_leg = leg[lane] if recipe.leg_lane else leg
                total = np.zeros((), dtype=self.dtype)
                for i, j, k in recipe.support:
                    spatial_sum = np.zeros(recipe.modes, dtype=self.dtype)
                    for (spin, spatial, coefficient), dynamic in zip(self.components, recipe.spatial_lanes):
                        phi = spatial[lane] if dynamic else spatial
                        spatial_sum += (coefficient * spin[i, j, k]) * phi
                    total += np.sum(spatial_sum * current_leg[i, :, j, :, k, :], dtype=self.dtype)
                if recipe.output_shape:
                    result[lane] = total
                else:
                    result[...] = total
            return result
        partial = backend.empty((recipe.width, self.blocks), dtype=self.dtype)
        first, final = self.kernels
        first((self.blocks, recipe.width), (256,), (self.metadata, self.coefficients,
            *(np.int64(n) for n in recipe.modes), *(np.int32(n) for n in self.order),
            np.int32(len(recipe.support)), partial))
        final((recipe.width,), (256,), (partial, np.int32(self.blocks), result))
        return result

    def close(self):
        self.components = ()
        self.recipe = self.leg = self.backend = None
        self.metadata = self.coefficients = self.kernels = None
        self.closed = True


def prepare_scalar_sum(action, arrays, backend, dtype):
    """Prepare exact sparse scalar/lane ports, otherwise retain the old route.

    The caller selects this only for the normal no_search=False whole-core
    route. Sparse metadata declares exact support; current spin values are
    read numerically, including values changed within that declared support.
    """
    core = getattr(action, 'core', None)
    description = getattr(action, 'kernel_description', None)
    if (core is None or core.kind != 'baryon' or description is None
            or description.implementation != 'ordered-join-v1'
            or not 1 <= len(core.components) <= 3
            or backend.__name__ not in ('numpy', 'cupy')):
        return None
    legs = action.operands[len(core.inputs):]
    if len(legs) != 1 or any(part.sparse_spin is None for part in core.components):
        return None
    if backend.__name__ == 'numpy':
        # Prepare-inclusive measurements retain the old small CPU route.
        # Read actual leg dimensions, including permuted and lane layouts.
        dimensions = dict(zip(action.equation.inputs[1], arrays[legs[0]].shape))
        if any(dimensions.get(axis, 0) < CPU_MIN_MODES for axis in action.equation.inputs[0][-3:]):
            return None
    dtype = np.dtype(dtype)
    if dtype not in (np.dtype('complex64'), np.dtype('complex128')):
        return None
    values = tuple(arrays[key] for part in core.components for key in part.inputs) + (arrays[legs[0]],)
    if (any(np.dtype(value.dtype) != dtype for value in values)
            or any(not isinstance(value, backend.ndarray) for value in values)):
        return None
    supports = tuple(tuple((i, j, k) for i, j, k, _ in part.sparse_spin) for part in core.components)
    recipe = _recipe(action.equation, tuple(tuple(value.shape) for value in values), supports)
    if recipe is None:
        return None
    components = tuple((arrays[part.inputs[0]], arrays[part.inputs[1]], dtype.type(part.coefficient))
                       for part in core.components)
    return PreparedScalarSum(recipe, components, values[-1], backend, dtype)


__all__ = ['prepare_scalar_sum', 'PreparedScalarSum']
