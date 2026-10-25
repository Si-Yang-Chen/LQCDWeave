"""Bound CuPy preparation kernels; no numerical input owners are cached.

Raw ElementwiseKernel parameters carry their own shape/stride metadata by value.
A lane computes one spatial value, then writes its 16 spin entries. This avoids
repeating the profile reduction for every spin or allocating projected tensors.
"""
from collections import OrderedDict
from dataclasses import dataclass, replace
from functools import lru_cache
from math import prod

import numpy as np

from .diagnostics import debug_enabled
from .raw_arguments import bind_pointer_arguments


# Keep source bounded independently for the two kernel ABIs. RawKernel receives
# one pointer per array and three fixed arguments; ElementwiseKernel retains its
# CArray-heavy conservative envelope.
_MAX_RAW_LANES = 32
_MAX_LEGACY_LANES = 8
_MAX_ARGUMENT_BYTES = 3072
_MAX_VARIANTS = 8
_DTYPES = {np.dtype(t) for t in (np.float32, np.float64, np.complex64, np.complex128)}
_CALL_DTYPES = {dtype: dtype for dtype in _DTYPES}
_CTYPE = {np.dtype('float32'): 'float', np.dtype('float64'): 'double',
          np.dtype('complex64'): 'complex<float>', np.dtype('complex128'): 'complex<double>'}

# Materialized ufunc intermediates round before the next multiply/add. Volatile
# component stores preserve those boundaries without disabling the backend's
# usual contraction *within* an individual complex multiply. In particular, a
# small product cannot be contracted into the complex128 profile accumulation.
_ROUND_PREAMBLE = r"""
template <typename T> __device__ T ed_round(T value) {
    volatile T rounded = value;
    return rounded;
}
template <typename T> __device__ complex<T> ed_round(complex<T> value) {
    volatile T re = value.real(), im = value.imag();
    return complex<T>(T(re), T(im));
}
"""


def supports_fused_preparation(backend):
    return (getattr(backend, '__name__', '') == 'cupy'
            and callable(getattr(backend, 'ElementwiseKernel', None)))


def _cast(expression, source, target):
    source, target = np.dtype(source), np.dtype(target)
    if source.kind == 'c' and target.kind != 'c':
        expression = f'({expression}).real()'
    return f'{_CTYPE[target]}({expression})'


def _multiply(left, left_dtype, right, right_dtype):
    # CuPy ndarray operands, including a 0-D coefficient view, have strong
    # dtype promotion. Unlike Python scalars, their values never affect it.
    dtype = np.result_type(left_dtype, right_dtype)
    return (f'ed_round({_cast(left, left_dtype, dtype)} * '
            f'{_cast(right, right_dtype, dtype)})', dtype)


@dataclass(frozen=True)
class _SpinLeaf:
    recipe: object
    constants: tuple
    spec: object
    casts: tuple = ()


def _raw_preparation_layout(leaves, start, end, input_aliases=None):
    """Compact identical constants and current-input positions within a chunk."""
    template, positions, aliases, constants, inputs = [], [], [], {}, {}
    for index in range(start, end):
        canonical = index if input_aliases is None else input_aliases[index]
        position = inputs.get(canonical)
        if position is None:
            position = len(template)
            inputs[canonical] = position
            positions.append((position, canonical))
            template.append(None)
        aliases.append(position)
        for value in leaves[index].constants:
            position = constants.get(id(value))
            if position is None:
                position = len(template)
                constants[id(value)] = position
                template.append(value)
            aliases.append(position)
    return tuple(aliases), tuple(template), tuple(positions)


def _bind_raw_preparation_call(kernel, start, end, spatial_size, leaves, *, layout=None,
                               backend=None):
    """Bind constant arguments and geometry; keep input/output slots empty."""
    _, template, positions = (_raw_preparation_layout(leaves, start, end)
                              if layout is None else layout)
    template = list(template)
    template.extend((spatial_size, np.int64(start), None))
    template = tuple(template)
    grid = (((end-start)*int(spatial_size) + 255)//256,)
    block = (256,)
    native_arguments = bind_pointer_arguments(backend, kernel, grid, block, template, positions)

    def launch(inputs, result):
        if native_arguments is not None and not debug_enabled():
            native_arguments(inputs, result)
            return
        arguments = list(template)
        for position, index in positions:
            arguments[position] = inputs[index]
        arguments[-1] = result
        kernel(grid, block, tuple(arguments))
    launch.native_arguments = native_arguments
    return launch


class FusedSpinPreparation:
    """A loader/backend-local descriptor and bounded metadata-only kernel cache."""

    def __init__(self, backend, leaves, spec, *, stacked=False, prefer_raw=False):
        self.backend, self.leaves, self.spec = backend, tuple(leaves), spec
        self.stacked = stacked
        self._lane_views = any(leaf.recipe.raw_index is not None for leaf in self.leaves)
        self.prefer_raw = bool(prefer_raw)
        self._variants = OrderedDict()
        self._calls = OrderedDict()

    @classmethod
    def spin(cls, backend, recipe, constants, spec, *, prefer_raw=False):
        if ((recipe.raw_index is not None and not recipe.spatial_support)
                or recipe.spatial_structure != "general"
                or not supports_fused_preparation(backend) or spec is None
                or spec.dtype not in _DTYPES
                or any(value.dtype not in _DTYPES for value in constants)):
            return None
        return cls(backend, (_SpinLeaf(recipe, constants, spec),), spec,
                   prefer_raw=prefer_raw)

    @classmethod
    def stack(cls, backend, members, spec, *, prefer_raw=False):
        if not members or any(member is None for member in members) or spec.dtype not in _DTYPES:
            return None
        shape = members[0].leaves[0].spec.shape
        if any(member.spec.shape != spec.shape[1:]
               or any(leaf.spec.shape != shape or (bool(leaf.recipe.spin_columns), leaf.recipe.spatial_only) != (bool(members[0].leaves[0].recipe.spin_columns), members[0].leaves[0].recipe.spatial_only) for leaf in member.leaves)
               for member in members):
            return None
        leaves = tuple(replace(leaf, casts=(*leaf.casts, spec.dtype))
                       for member in members for leaf in member.leaves)
        return cls(backend, leaves, spec, stacked=True,
                   prefer_raw=prefer_raw and all(member.prefer_raw for member in members))

    def materialize(self, blocks, destination=None, *, inputs=None):
        inputs = (tuple(blocks[leaf.recipe.inputs[0]] for leaf in self.leaves)
                  if inputs is None else inputs)
        raw_inputs = inputs
        if self._lane_views:
            # A lane is a current-input view. Select it before deriving the
            # bounded geometry key; never retain its address or parent array.
            inputs = tuple(value if leaf.recipe.raw_index is None else value[leaf.recipe.raw_index]
                           for leaf, value in zip(self.leaves, inputs))
        # Canonical dtypes avoid strings and cannot retain dtype.metadata owners.
        # Unsupported inputs have no cached call; the cold path rejects them.
        input_aliases = None
        if (len(inputs) >= 8 and not debug_enabled()
                and (inputs[0] is inputs[1] or inputs[-1] is inputs[-2]
                     or inputs[0] is inputs[-1])):
            # The current sequence keeps every input alive through this call,
            # so IDs cannot be recycled while forming the ordered layout key.
            # Rebuild on the next invocation: callers can change array strides.
            layouts, input_signature, input_aliases = {}, [], []
            for index, value in enumerate(inputs):
                identity = id(value)
                entry = layouts.get(identity)
                if entry is None:
                    layout = (_CALL_DTYPES.get(value.dtype), value.shape, value.strides)
                    entry = layouts[identity] = (layout, index)
                layout, canonical = entry
                input_signature.append(layout)
                input_aliases.append(canonical)
            input_signature = tuple(input_signature)
            input_aliases = tuple(input_aliases)
        else:
            input_signature = tuple((_CALL_DTYPES.get(v.dtype), v.shape, v.strides) for v in inputs)
        target = None if destination is None else (destination.dtype.str, destination.shape, destination.strides)
        raw_factory = getattr(self.backend, 'RawKernel', None)
        use_raw = (self.prefer_raw and callable(raw_factory)
                   and (destination is None or destination.flags.c_contiguous))
        call_key = (input_signature, target, use_raw, input_aliases if use_raw else None)
        cached = self._calls.get(call_key)
        if cached is not None and not debug_enabled():
            shape, dtype, calls = cached
            result = self.backend.empty(shape, dtype=dtype) if destination is None else destination
            if use_raw and not result.flags.c_contiguous:
                return self.materialize(blocks, result, inputs=raw_inputs)
            if use_raw:
                for call in calls:
                    call(inputs, result)
                self._calls.move_to_end(call_key)
                return result
            # CArray kernels still consume logical views of current inputs.
            values = tuple(((v if not leaf.recipe.permutation
                             else v.transpose(leaf.recipe.permutation)),
                            *leaf.constants) for leaf, v in zip(self.leaves, inputs))
            for kernel, start, end, spatial_size in calls:
                arguments = tuple(v for arrays in values[start:end] for v in arrays)
                kernel(*arguments, spatial_size, np.int64(start), result,
                       size=(end-start)*int(spatial_size))
            self._calls.move_to_end(call_key)
            return result
        metadata, values = [], []
        for leaf, value in zip(self.leaves, inputs):
            recipe, constants = leaf.recipe, leaf.constants
            if value.dtype not in _DTYPES:
                return None
            profile = len(constants) == 4
            if profile and (value.ndim != 2 or constants[1].shape[1] != value.shape[0]
                            or constants[2].shape[1] != value.shape[1]):
                # Keep the original general broadcasting/error semantics.
                return None
            if recipe.permutation:
                value = value.transpose(recipe.permutation)
            dtype = (np.dtype(recipe.compute_dtype) if recipe.compute_dtype is not None else
                     np.result_type(constants[0].dtype, np.complex128 if profile else value.dtype))
            shape = (value.shape if recipe.spatial_only else
                     (4, *value.shape) if recipe.spin_columns else (4, 4, *value.shape))
            if shape != leaf.spec.shape or (debug_enabled() and dtype != leaf.spec.dtype):
                if debug_enabled():
                    raise ValueError('materialized component differs from its physical metadata')
                return None
            if any(dtype not in _DTYPES for dtype in leaf.casts):
                return None
            arrays = (value, *constants)
            # CuPy's internal specialization includes rank, contiguity and index
            # width. Bound them in our key too, so its inner cache cannot grow
            # independently as arbitrary input views are supplied over time.
            array_signature = tuple((v.dtype.str, v.ndim, bool(v.flags.c_contiguous),
                               tuple(v.shape), tuple(v.strides)) for v in arrays)
            entry = (array_signature, tuple(np.dtype(d).str for d in leaf.casts),
                     recipe.conjugate, recipe.permutation, dtype.str)
            if recipe.spatial_support:
                entry = (*entry, recipe.compute_dtype, recipe.spin_columns,
                         recipe.spatial_only, recipe.spatial_support)
            elif recipe.spatial_only or recipe.spin_columns:
                entry = (*entry, recipe.compute_dtype, recipe.spin_columns, recipe.spatial_only)
            elif recipe.compute_dtype is not None:
                entry = (*entry, recipe.compute_dtype)
            metadata.append(entry)
            values.append(arrays)
        spatial_start = (0 if self.leaves[0].recipe.spatial_only else
                         1 if self.leaves[0].recipe.spin_columns else 2)
        spatial_size = prod(self.leaves[0].spec.shape[spatial_start:])
        dtype = self.spec.dtype if self.stacked else np.dtype(metadata[0][4])
        shape = self.spec.shape
        if destination is not None and (destination.shape != shape or destination.dtype not in _DTYPES):
            return None
        # ElementwiseKernel carries CArray metadata by value. RawKernel fixes
        # shapes and strides in source and only passes array pointers plus the
        # two int64 scalars and output pointer (24 bytes total).
        output_rank = len(shape) if destination is None else destination.ndim
        def legacy_argument_bytes(arrays):
            return 64 + 16 * output_rank + sum(
                16 + 16 * value.ndim for value in arrays)

        if not use_raw and any(legacy_argument_bytes(arrays) > _MAX_ARGUMENT_BYTES
                               for arrays in values):
            return None
        result = (self.backend.empty(shape, dtype=dtype) if destination is None else destination)
        if spatial_size == 0:
            return result
        if use_raw and not result.flags.c_contiguous:
            use_raw = False
            # An unusual backend may return a non-C output even for empty().
            # Recheck only because the pre-allocation path selected RawKernel.
            if any(legacy_argument_bytes(arrays) > _MAX_ARGUMENT_BYTES for arrays in values):
                return None
        call_key = (input_signature, target, use_raw, input_aliases if use_raw else None)
        # Nested stack casts are emitted even when the final destination has a
        # wider dtype, preserving each original stack's intermediate rounding.
        offset = 0
        calls = []
        while offset < len(values):
            end = offset
            argument_bytes = 24 if use_raw else 64 + 16 * result.ndim
            lane_limit = _MAX_RAW_LANES if use_raw else _MAX_LEGACY_LANES
            while end < len(values) and end - offset < lane_limit:
                size = (8 * len(values[end]) if use_raw else
                        sum(16 + 16 * value.ndim for value in values[end]))
                if end > offset and argument_bytes + size > _MAX_ARGUMENT_BYTES:
                    break
                argument_bytes += size
                end += 1
            chunk = tuple(metadata[offset:end])
            layout = (_raw_preparation_layout(self.leaves, offset, end, input_aliases)
                      if use_raw else None)
            aliases = layout[0] if use_raw else None
            key = (use_raw, chunk, aliases, result.dtype.str, result.ndim,
                   bool(result.flags.c_contiguous), tuple(result.shape), tuple(result.strides))
            kernel = self._variants.get(key)
            if kernel is None:
                kernel = (_make_spin_raw_kernel(raw_factory, chunk, result.dtype,
                                                result.shape, result.strides, spatial_size,
                                                parameter_aliases=aliases)
                          if use_raw else
                          _make_spin_kernel(self.backend.ElementwiseKernel, chunk, result.dtype))
                self._variants[key] = kernel
                if len(self._variants) > _MAX_VARIANTS:
                    self._variants.popitem(last=False)
                    self._calls.clear()
            else:
                self._variants.move_to_end(key)
            if use_raw:
                call = _bind_raw_preparation_call(kernel, offset, end,
                    np.int64(spatial_size), self.leaves, layout=layout, backend=self.backend)
                call(inputs, result)
                calls.append((kernel, offset, end, np.int64(spatial_size), call))
            else:
                arguments = tuple(value for arrays in values[offset:end] for value in arrays)
                kernel(*arguments, np.int64(spatial_size), np.int64(offset), result,
                       size=(end - offset) * spatial_size)
                calls.append((kernel, offset, end, np.int64(spatial_size)))
            offset = end
        if all(any(kernel is variant for variant in self._variants.values())
               for kernel, *_ in calls):
            bound_calls = (tuple(call[-1] for call in calls)
                           if use_raw else tuple(calls))
            self._calls[call_key] = (shape, dtype, bound_calls)
        if len(self._calls) > _MAX_VARIANTS:
            self._calls.popitem(last=False)
        return result


def _spin_kernel_parts(metadata, output_dtype):
    params, branches = [], []
    arrays = []
    for lane, entry in enumerate(metadata):
        signature, casts, conjugate, permutation, result_dtype = entry[:5]
        compute_dtype = np.dtype(entry[5]) if len(entry) > 5 and entry[5] is not None else None
        spin_columns = entry[6] if len(entry) > 6 else ()
        spatial_only = entry[7] if len(entry) > 7 else False
        support = entry[8] if len(entry) > 8 else ()
        spin_size = 1 if spatial_only else 4 if spin_columns else 16
        dtypes = [np.dtype(item[0]) for item in signature]
        names = [f'x{lane}', f's{lane}', f'l{lane}', f'r{lane}', f'w{lane}'][:len(signature)]
        params.extend(f'raw {dtype.name} {name}' for dtype, name in zip(dtypes, names))
        arrays.extend((name, dtype, item[3], item[4])
                      for name, dtype, item in zip(names, dtypes, signature))
        raw, spin = names[:2]
        raw_dtype, spin_dtype = dtypes[:2]
        raw_value = f'{raw}[q]'
        if compute_dtype is not None:
            raw_value = _cast(raw_value, raw_dtype, compute_dtype)
            raw_dtype = compute_dtype
        lines = [f'if (lane == {lane}) {{']
        if support:
            from .structured_preparation import spatial_support_condition
            condition = spatial_support_condition(support, signature[0][3])
            lines += [f'if (!({condition})) {{', '#pragma unroll',
                      f'for (int s = 0; s < {spin_size}; ++s) {{',
                      f'output[((offset + lane) * {spin_size} + s) * spatial_size + q] = {_CTYPE[np.dtype(output_dtype)]}(0);',
                      '}', '} else {']
        if len(names) == 5:
            left, right, weights = names[2:]
            left_dtype, right_dtype, weight_dtype = dtypes[2:]
            left_shape, right_shape = signature[2][3], signature[3][3]
            cols = signature[0][3][1]
            row, col = (f'(q % {cols})', f'(q / {cols})') if permutation == (1, 0) else (
                f'(q / {cols})', f'(q % {cols})')
            left_value = f'{left}[p * {left_shape[1]} + {row}]'
            if left_dtype.kind == 'c':
                left_value = f'conj({left_value})'
            product1, dtype1 = _multiply(f'{weights}[p]', weight_dtype, left_value, left_dtype)
            product2, dtype2 = _multiply('first', dtype1, raw_value, raw_dtype)
            product3, dtype3 = _multiply('second', dtype2,
                                         f'{right}[p * {right_shape[1]} + {col}]', right_dtype)
            accumulator = np.dtype(np.complex128) if compute_dtype is None else compute_dtype
            initial = ('complex<double> value(0.0, 0.0);' if compute_dtype is None else
                       f'{_CTYPE[accumulator]} value = {_cast("0.0", np.dtype("float64"), accumulator)};')
            lines += [initial,
                      '#pragma unroll 1', f'for (long long p = 0; p < {left_shape[0]}; ++p) {{',
                      f'const {_CTYPE[dtype1]} first = {product1};',
                      f'const {_CTYPE[dtype2]} second = {product2};',
                      f'const {_CTYPE[dtype3]} third = {product3};',
                      ('value = ed_round(value + complex<double>(third));' if compute_dtype is None else
                       f'value = ed_round(value + {_cast("third", dtype3, accumulator)});'), '}']
            value_dtype = accumulator
        else:
            lines += [f'{_CTYPE[raw_dtype]} value = {raw_value};']
            value_dtype = raw_dtype
        if conjugate and value_dtype.kind == 'c':
            lines += ['value = conj(value);']
        if spin_columns:
            table = ', '.join(str(4*row + column) for row, column in enumerate(spin_columns))
            lines.append(f'const int ed_spin_index[4] = {{{table}}};')
        spin_index = 'ed_spin_index[s]' if spin_columns else 's'
        if spatial_only:
            dtype = np.result_type(spin_dtype, value_dtype)
            expression = _cast('value', value_dtype, dtype)
        else:
            expression, dtype = _multiply(f'{spin}[{spin_index}]', spin_dtype, 'value', value_dtype)
        lines += ['#pragma unroll', f'for (int s = 0; s < {spin_size}; ++s) {{',
                  f'{_CTYPE[dtype]} prepared = {expression};']
        expression = 'prepared'
        for index, target in enumerate((*casts, output_dtype)):
            target = np.dtype(target)
            lines.append(f'const {_CTYPE[target]} cast{index} = '
                         f'ed_round({_cast(expression, dtype, target)});')
            expression, dtype = f'cast{index}', target
        lines += [f'output[((offset + lane) * {spin_size} + s) * spatial_size + q] = {expression};', '}']
        if support:
            lines.append('}')
        lines.append('}')
        branches.append('\n'.join(lines))
    params += ['int64 spatial_size', 'int64 offset']
    operation = ('const long long lane = i / spatial_size, q = i % spatial_size;\n'
                 + '\nelse '.join(branches))
    return params, arrays, operation


def _make_spin_kernel(kernel_type, metadata, output_dtype):
    params, _, operation = _spin_kernel_parts(metadata, output_dtype)
    return kernel_type(', '.join(params), f'raw {np.dtype(output_dtype).name} output',
                       operation, 'ed_prepare_spin', preamble=_ROUND_PREAMBLE)


@lru_cache(maxsize=64)
def _raw_spin_kernel(factory, source):
    """Cache generated code only; live arrays and launch values stay per call."""
    return factory(source, 'ed_prepare_spin_raw', options=('--std=c++17',))


def _make_spin_raw_kernel(factory, metadata, output_dtype, output_shape, output_strides,
                          spatial_size, *, parameter_aliases=None):
    _, arrays, operation = _spin_kernel_parts(metadata, output_dtype)
    pointer_parameters, proxy_declarations, proxy_initializers = [], [], []
    aliases = range(len(arrays)) if parameter_aliases is None else parameter_aliases
    parameter_names, view_types = {}, {}
    for alias, (name, dtype, shape, strides) in zip(aliases, arrays, strict=True):
        ctype = _CTYPE[np.dtype(dtype)]
        parameter_name = parameter_names.get(alias)
        if parameter_name is None:
            parameter_name = parameter_names[alias] = f'ed_arg_{name}'
            pointer_parameters.append(f'const {ctype}* {parameter_name}')
        declaration, initializer = _raw_view_parts(
            name, dtype, shape, strides, parameter_name=parameter_name)
        view_key = (dtype, shape, strides)
        existing_view = view_types.get(view_key)
        if existing_view is None:
            view_types[view_key] = name
        else:
            declaration = f'using ed_view_{name} = ed_view_{existing_view};'
        proxy_declarations.append(declaration)
        proxy_initializers.append(initializer)
    operation = _shared_raw_spin_operation(metadata, output_dtype, operation)
    ctype = _CTYPE[np.dtype(output_dtype)]
    output_parameter = f'{ctype}* ed_arg_output'
    declaration, initializer = _raw_view_parts(
        'output', output_dtype, output_shape, output_strides, writable=True)
    proxy_declarations.append(declaration)
    proxy_initializers.append(initializer)
    # The operation body is shared verbatim with ElementwiseKernel. Each proxy
    # implements its original logical row-major indexing over fixed byte strides.
    size = len(metadata) * spatial_size
    source = ('#include <cupy/complex.cuh>\n'
        + _ROUND_PREAMBLE + '\n'
        + '\n'.join(proxy_declarations)
        + '\nextern "C" __global__ void ed_prepare_spin_raw('
        + ', '.join((*pointer_parameters, 'long long spatial_size', 'long long offset',
                     output_parameter))
        + ') {\n'
        + 'const long long i = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;\n'
        + f'if (i >= {size}LL) return;\n'
        + '\n'.join(proxy_initializers) + '\n'
        + operation + '\n}\n')
    return _raw_spin_kernel(factory, source)


def _shared_raw_spin_operation(metadata, output_dtype, fallback):
    """Select current lane views before one copy of each identical leaf program.

    The grouping key includes all indexing, casts and arithmetic metadata. Only
    input pointers vary: no values, owners or addresses enter the generated code.
    Keeping the arithmetic in the kernel avoids device calls and their spills.
    """
    groups = {}
    for lane, entry in enumerate(metadata):
        groups.setdefault(entry, []).append(lane)
    if len(groups) == len(metadata):
        return fallback
    branches = []
    for entry, lanes in groups.items():
        _, arrays, operation = _spin_kernel_parts((entry,), output_dtype)
        # The singleton generator supplies the exact original arithmetic. Drop
        # its lane==0 wrapper, but retain the actual launch lane for output.
        body = operation.split('\n', 1)[1].splitlines()[1:-1]
        selected = []
        for name, *_ in arrays:
            stem = name[:-1]  # singleton names are x0/s0/l0/r0/w0
            pointers = [f'{stem}{lane}.pointer' for lane in lanes]
            pointer = pointers[-1]
            for lane, candidate in reversed(tuple(zip(lanes[:-1], pointers[:-1]))):
                pointer = f'(lane == {lane} ? {candidate} : {pointer})'
            selected_name = f'ed_selected_{name}'
            selected.append(f'ed_view_{stem}{lanes[0]} {selected_name}{{{pointer}}};')
            body = [line.replace(f'{name}[', f'{selected_name}[') for line in body]
        condition = ' || '.join(f'lane == {lane}' for lane in lanes)
        branches.append('if (' + condition + ') {\n'
                        + '\n'.join((*selected, *body)) + '\n}')
    return ('const long long lane = i / spatial_size, q = i % spatial_size;\n'
            + '\nelse '.join(branches))


def _raw_view_parts(name, dtype, shape, strides, *, writable=False, parameter_name=None):
    ctype = _CTYPE[np.dtype(dtype)]
    cv = '' if writable else 'const '
    reference = f'{ctype}&' if writable else f'const {ctype}&'
    if any(size == 0 for size in shape):
        address = '0LL'
    else:
        address = ' + '.join(
            f'((index / {prod(shape[axis+1:])}LL) % {size}LL) * {stride}LL'
            for axis, (size, stride) in enumerate(zip(shape, strides))) or '0LL'
    view = f'ed_view_{name}'
    declaration = (f'struct {view} {{\n'
        f'    {cv}char* pointer;\n'
        f'    __device__ {reference} operator[](long long index) const {{\n'
        f'        const long long byte_offset = {address};\n'
        f'        return *reinterpret_cast<{cv}{ctype}*>(pointer + byte_offset);\n'
        f'    }}\n'
        f'}};')
    parameter_name = f'ed_arg_{name}' if parameter_name is None else parameter_name
    initializer = f'{view} {name}{{reinterpret_cast<{cv}char*>({parameter_name})}};'
    return declaration, initializer
