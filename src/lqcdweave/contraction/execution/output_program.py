"""Prebound, fused device execution of R's fixed-point output recipes.

R defines C recipe semantics; W supplies frozen fixed-point records. X lowers small
fields to one CuPy kernel each; shape operations become index arithmetic,
including scatter and disconnected outer products. Large expressions use
bounded ordered stages with lifetime-reused, explicitly budgeted scratch.
No host term value is produced; literal arrays upload once during binding.
"""
from dataclasses import dataclass, field
from functools import lru_cache
from math import isfinite, prod
import numbers
from weakref import ref

import numpy as np

from lqcdweave.contracts.correlator.api import OutputRecipe, CoordinateSpec
from .diagnostics import debug_enabled
from .operations.output import (
    BoundOutputLayout, _BoundLayoutPackedOutputs, _OutputOwnerSnapshot,
)


_DTYPES = {np.dtype(name) for name in (
    'bool', 'int8', 'uint8', 'int16', 'uint16', 'int32', 'uint32',
    'int64', 'uint64', 'float32', 'float64', 'complex64', 'complex128')}
_CTYPE = dict(zip((np.dtype(name) for name in (
    'bool', 'int8', 'uint8', 'int16', 'uint16', 'int32', 'uint32',
    'int64', 'uint64', 'float32', 'float64', 'complex64', 'complex128')),
    ('bool', 'signed char', 'unsigned char', 'short', 'unsigned short',
     'int', 'unsigned int', 'long long', 'unsigned long long',
     'float', 'double', 'complex<float>', 'complex<double>')))
_NATIVE_PROBE_TYPES = frozenset((bool, int, float, complex, np.ndarray,
                                 *(dtype.type for dtype in _DTYPES)))
_MAX_ARGUMENT_BYTES = 3072
# Bound each generated CUDA stage; larger fields retain their ordered
# arithmetic through explicitly budgeted temporary numeric inputs.
_MAX_STAGE_NODES = 8192
_MAX_LITERAL_TEXT_ENTRIES = 128
_OUTPUT_KERNEL_VERSION = 3
_MAX_OUTPUT_LAYOUT_SCHEMAS = 16
# NumPy's scalar promotion policy is fixed for the loaded runtime. Parsing its
# version is not part of evaluating each binary expression's dtype.
_LEGACY_SCALAR_PROMOTION = np.lib.NumpyVersion(np.__version__) < '2.0.0'
_PREAMBLE = r"""
template <typename T> __device__ T ed_output_round(T value) {
    volatile T rounded = value;
    return rounded;
}
template <typename T> __device__ complex<T> ed_output_round(complex<T> value) {
    volatile T re = value.real(), im = value.imag();
    return complex<T>(T(re), T(im));
}
"""


class _ExpressionWeakRefs:
    # Inherited weakref support also works on the Python 3.10 minimum.
    __slots__ = ('__weakref__',)


@dataclass(eq=False, slots=True)
class _Expression(_ExpressionWeakRefs):
    kind: str
    shape: tuple
    probe: object
    children: tuple = ()
    value: object = None
    work: int = field(init=False, repr=False)
    dtype: np.dtype = field(init=False, repr=False)

    def __post_init__(self):
        children = self.children
        count = len(children) if type(children) is tuple else -1
        self.work = (1 if count == 0 else
                     1 + children[0].work if count == 1 and type(children[0]) is _Expression else
                     1 + children[0].work + children[1].work
                     if count == 2 and type(children[0]) is _Expression and type(children[1]) is _Expression else
                     1 + sum(child.work for child in children))
        self.dtype = np.asarray(self.probe).dtype


@dataclass(frozen=True)
class _Stage:
    field: int
    target: tuple
    expression: _Expression


def _check_dtype(dtype):
    dtype = np.dtype(dtype)
    if dtype not in _DTYPES or not dtype.isnative:
        raise TypeError(f'device output assembly does not support dtype {dtype}')
    return dtype


def _number(value):
    # Same coefficient normalization as R, including SymPy numeric literals.
    if hasattr(value, 'item'):
        return value.item()
    if getattr(value, 'is_Integer', False):
        return int(value)
    if getattr(value, 'is_real', None) is True:
        return float(value)
    if getattr(value, 'is_number', False):
        return complex(value)
    if isinstance(value, numbers.Integral):
        return int(value)
    if isinstance(value, numbers.Rational):
        return float(value)
    if isinstance(value, numbers.Real):
        return float(value)
    if isinstance(value, numbers.Complex):
        return complex(value)
    return value


def _array_probe(dtype, shape):
    # One element is enough for dtype/type promotion, irrespective of real size.
    return np.ones((1,) if shape else (), dtype=dtype)


def _constant(value):
    if type(value) not in _NATIVE_PROBE_TYPES:
        _check_dtype(np.asarray(value).dtype)
        return _Expression('constant', (), value, value=value)
    expression = _Expression('constant', (), value, value=value)
    _check_dtype(expression.dtype)
    return expression


def _binary(kind, left, right):
    # Two scalar operands already have the scalar output shape.
    shape = (() if not left.shape and not right.shape else
             np.broadcast_shapes(left.shape, right.shape))
    # NumPy <2 has data-dependent weak scalar promotion. A scalar term's
    # runtime magnitude is unavailable at bind time, so reject that ambiguity.
    if _LEGACY_SCALAR_PROMOTION:
        for scalar, array in ((left, right), (right, left)):
            if (isinstance(array.probe, np.ndarray) and array.shape and not scalar.shape
                    and scalar.kind not in {'constant', 'literal_array', 'zeros'}):
                # Include selected NumPy scalars and reshaped 0-D arrays. A
                # narrow scalar whose full dtype already gives the same result
                # is safe; a Python int can additionally overflow that dtype.
                stable = (type(scalar.probe) is not int and
                          np.result_type(array.dtype, scalar.dtype) ==
                          np.result_type(array.probe, scalar.probe))
                if not stable:
                    raise TypeError('device output requires NumPy >=2 for value-dependent scalar/array promotion')
    probe = (left.probe + right.probe if kind == 'add'
             else left.probe * right.probe)
    if type(probe) not in _NATIVE_PROBE_TYPES:
        _check_dtype(np.asarray(probe).dtype)
        return _Expression(kind, shape, probe, (left, right))
    expression = _Expression(kind, shape, probe, (left, right))
    _check_dtype(expression.dtype)
    return expression


class _Description:
    """Array-free compiled metadata; tiny probes preserve NumPy scalar types."""
    def __init__(self, recipe, problem):
        # W supplies the frozen C record. X checks only the fixed kernel
        # ABI here and does not import or invoke R's assembly implementation.
        if not isinstance(recipe, OutputRecipe) or not isinstance(recipe.coordinates, CoordinateSpec):
            raise TypeError('device output requires an OutputRecipe with CoordinateSpec')
        if recipe.coordinates.scan_shape or len(recipe.coordinates.points) != 1:
            raise ValueError('device output requires fixed single-point coordinates')
        if (not isinstance(recipe.term_ids, tuple) or
                len(set(recipe.term_ids)) != len(recipe.term_ids) or
                any(not isinstance(key, str) or not key for key in recipe.term_ids)):
            raise ValueError('device output requires unique nonempty term IDs')
        terms = tuple(problem.terms)
        computation_dtype = recipe.intermediate_dtype or recipe.dtype
        self.compute_dtype = None if computation_dtype is None else _check_dtype(computation_dtype)
        self.boundary_dtype = _check_dtype(recipe.dtype) if recipe.intermediate_dtype else None
        if len(terms) != len(recipe.term_ids):
            raise ValueError('output recipe term IDs do not match the contraction problem')
        self.term_specs, self.constants, self.fields = [], [], []
        for term in terms:
            dimensions, dtypes = {}, []
            if len(term.inputs) != len(term.subscripts.inputs):
                raise ValueError('term input labels do not match its operands')
            for key, labels in zip(term.inputs, term.subscripts.inputs, strict=True):
                spec = problem.specs[key]
                if len(labels) != len(spec.shape):
                    raise ValueError('term labels do not match its TensorSpec')
                for label, dimension in zip(labels, spec.shape, strict=True):
                    if label in dimensions and dimensions[label] != dimension:
                        raise ValueError('term dimensions disagree for a shared axis')
                    dimensions[label] = dimension
                dtypes.append(spec.dtype)
            shape = tuple(dimensions[label] for label in term.subscripts.output)
            dtype = _check_dtype(np.result_type(*dtypes))
            if not shape and dtype.kind in 'biu':
                raise TypeError('device output assembly requires floating/complex scalar terms; Python integer results can overflow their bound dtype')
            self.term_specs.append((shape, dtype))
        self.term_ids = dict(zip(recipe.term_ids, range(len(terms)), strict=True))
        self._nodes = {}
        try:
            with np.errstate(all='ignore'):
                self.structure = self._structure(recipe.root, ())
        finally:
            # Identity lookup is needed only while compiling the output tree.
            del self._nodes
        self.stages, self.temp_specs = [], []
        for field, (_, expression) in enumerate(self.fields):
            lowered = self._segment(expression, field)
            self.stages.append(_Stage(field, ('field', field), lowered))
        self.indirect_stages, self.table_layout, seen = [], [], set()
        self.indirect_fields = [False] * len(self.fields)
        last_use = {}
        offset = 0
        for index, stage in enumerate(self.stages):
            expression = stage.expression
            arguments = self._arguments(expression)
            # The same ordered dependency walk defines scratch lifetimes and
            # kernel arguments. Keep only temporary last uses, not a second
            # retained copy of every stage's argument list.
            for (kind, temporary), _, _ in arguments:
                if kind == 'temporary':
                    last_use[temporary] = index
            argument_bytes = (64 + 16 * len(expression.shape)
                              + sum(16 + 16 * len(shape) for _, shape, _ in arguments))
            indirect = argument_bytes > _MAX_ARGUMENT_BYTES
            self.indirect_stages.append(indirect)
            self.indirect_fields[stage.field] |= indirect
            if indirect:
                for key, shape, dtype in arguments:
                    if key not in seen:
                        seen.add(key)
                        self.table_layout.append((key, shape, dtype, offset))
                        offset += 1 + len(shape)
        self._plan_scratch(last_use)
        self.table_offsets = {key: offset for key, _, _, offset in self.table_layout}
        self.table_nbytes = offset * np.dtype('uint64').itemsize

    def _materialize(self, expression, field):
        index = len(self.temp_specs)
        self.temp_specs.append((expression.shape, expression.dtype))
        self.stages.append(_Stage(field, ('temporary', index), expression))
        # Keep the original NumPy/Python scalar-kind probe. Materialization is
        # storage only, not a new semantic array conversion or promotion step.
        return _Expression('temporary', expression.shape, expression.probe, value=index)

    def _segment(self, expression, field):
        if expression.work <= _MAX_STAGE_NODES:
            return expression
        pending, lowered = [(expression, False)], {}
        while pending:
            node, ready = pending.pop()
            if id(node) in lowered:
                continue
            if node.work <= _MAX_STAGE_NODES:
                lowered[id(node)] = node
                continue
            if not ready:
                pending.append((node, True))
                pending.extend((child, False) for child in reversed(node.children)
                               if id(child) not in lowered)
                continue
            children = [lowered[id(child)] for child in node.children]
            current = _Expression(node.kind, node.shape, node.probe, tuple(children), node.value)
            if current.work > _MAX_STAGE_NODES and node.kind == 'array':
                # Batch wide uniform arrays into contiguous groups rather than
                # launching a kernel for each scalar child. All group stores
                # use the original stack dtype, preserving its single cast.
                chunks, group, work = [], [], 1
                for child in children:
                    if child.work + 1 > _MAX_STAGE_NODES:
                        child = self._materialize(child, field)
                    if group and work + child.work > _MAX_STAGE_NODES:
                        chunk = _Expression('array', (len(group), *group[0].shape),
                            _array_probe(node.dtype, (len(group), *group[0].shape)), tuple(group))
                        chunks.append(self._materialize(chunk, field))
                        group, work = [], 1
                    group.append(child)
                    work += child.work
                if group:
                    chunk = _Expression('array', (len(group), *group[0].shape),
                        _array_probe(node.dtype, (len(group), *group[0].shape)), tuple(group))
                    chunks.append(self._materialize(chunk, field))
                if len(chunks) + 1 > _MAX_STAGE_NODES:
                    raise ValueError('device output uniform array has too many fused chunks')
                current = _Expression('chunks', node.shape, node.probe, tuple(chunks))
            else:
                while current.work > _MAX_STAGE_NODES:
                    if not children:
                        raise ValueError('device output node cannot be segmented safely')
                    index = max(range(len(children)), key=lambda i: children[i].work)
                    if children[index].work <= 1:
                        raise ValueError('device output node fanout cannot be segmented safely')
                    children[index] = self._materialize(children[index], field)
                    current = _Expression(node.kind, node.shape, node.probe, tuple(children), node.value)
            lowered[id(node)] = current
        return lowered[id(expression)]

    def _plan_scratch(self, last_use):
        slots, self.temp_slots = [], {}
        for index, stage in enumerate(self.stages):
            kind, temporary = stage.target
            if kind != 'temporary':
                continue
            shape, dtype = self.temp_specs[temporary]
            size = prod(shape)
            candidates = [slot for slot, (capacity, dt, end) in enumerate(slots)
                          if dt == dtype and end < index]
            if candidates:
                selected = min(candidates, key=lambda slot: (max(size - slots[slot][0], 0), slots[slot][0]))
                slots[selected] = (max(slots[selected][0], size), dtype, last_use[temporary])
            else:
                selected = len(slots)
                slots.append((size, dtype, last_use[temporary]))
            self.temp_slots[temporary] = selected
        self.scratch_slots = tuple((size, dtype) for size, dtype, _ in slots)
        self.scratch_nbytes = sum(size * dtype.itemsize for size, dtype in self.scratch_slots)

    def _arguments(self, expression):
        result, visited, keys, pending = [], set(), set(), [expression]
        while pending:
            node = pending.pop()
            children = node.children
            if children:
                identity = id(node)
                if identity in visited:
                    continue
                visited.add(identity)
            if node.kind in {'input', 'literal_array', 'temporary'}:
                key = node.kind, node.value
                if key not in keys:
                    keys.add(key)
                    if node.kind in {'input', 'temporary'}:
                        specs = self.term_specs if node.kind == 'input' else self.temp_specs
                        shape, dtype = specs[node.value]
                    else:
                        array = self.constants[node.value]
                        shape, dtype = array.shape, array.dtype
                    result.append((key, shape, dtype))
            if children:
                pending.extend(reversed(children))
        return tuple(result)

    def _structure(self, node, path):
        if node.kind in {'dict', 'list', 'tuple'}:
            keys = node.keys if node.kind == 'dict' else tuple(range(len(node.children)))
            return (node.kind, tuple(keys), tuple(self._structure(child, (*path, key))
                for key, child in zip(keys, node.children, strict=True)))
        expression = self._expression(node)
        if self.boundary_dtype is not None:
            expression = _Expression('cast', expression.shape,
                _array_probe(self.boundary_dtype, expression.shape), (expression,))
        field = len(self.fields)
        self.fields.append((path, expression))
        return ('field', field)

    def _expression(self, node):
        identity = id(node)
        if identity not in self._nodes:
            self._nodes[identity] = self._compile(node)
        return self._nodes[identity]

    def _term(self, term_id, coefficient):
        index = self.term_ids[term_id]
        shape, dtype = self.term_specs[index]
        probe = _array_probe(dtype, shape)
        if not shape and self.compute_dtype is None:
            probe = probe.item()  # R scalarizes only original 0-D term arrays.
        elif not shape:
            probe = probe[()]
        raw = _Expression('input', shape, probe, value=index)
        if self.boundary_dtype is not None:
            raw = _Expression('cast', shape, _array_probe(self.compute_dtype, shape), (raw,))
        return _binary('multiply', self._constant(_number(coefficient)), raw)

    def _constant(self, value):
        return _constant(value if self.compute_dtype is None else self.compute_dtype.type(value))

    def _compile(self, node):
        kind = node.kind
        if kind == 'term':
            return self._term(node.value.term_id, node.value.coefficient)
        if kind == 'diagram_sum':
            result = self._constant(0)
            for term_id, coefficient in node.value:
                result = _binary('add', result, self._term(term_id, coefficient))
            return result
        if kind == 'constant':
            return self._constant(node.value)
        if kind == 'constant_array':
            from lqcdweave.contracts.correlator.api import ArrayConstant
            value = node.value.array() if isinstance(node.value, ArrayConstant) else np.asarray(node.value)
            _check_dtype(value.dtype)
            index = len(self.constants)
            value = np.asarray(value, dtype=self.compute_dtype)
            self.constants.append(np.array(value, copy=True))
            return _Expression('literal_array', value.shape,
                _array_probe(value.dtype, value.shape) if value.shape else value.copy(), value=index)
        if kind == 'zeros':
            dtype = self.compute_dtype or _check_dtype(node.value)
            return _Expression('zeros', node.shape, np.zeros((1,) if node.shape else (), dtype=dtype))
        if kind in {'add', 'multiply'}:
            result = self._constant(0 if kind == 'add' else 1)
            for child in node.children:
                result = _binary(kind, result, self._expression(child))
            return result
        if kind == 'power':
            child = self._expression(node.children[0])
            probe = child.probe ** node.value
            if self.compute_dtype is not None:
                probe = self.compute_dtype.type(probe)
            _check_dtype(np.asarray(probe).dtype)
            return _Expression(kind, child.shape, probe, (child,), node.value)
        if kind == 'array':
            if not node.children or any(child.kind in {'array', 'constant_array', 'dict', 'list', 'tuple'}
                                        for child in node.children):
                raise TypeError('device output assembly does not support object or nested object arrays')
            children = tuple(self._expression(child) for child in node.children)
            arrays = tuple(isinstance(child.probe, np.ndarray) for child in children)
            if (len(set(arrays)) != 1 or len({child.shape for child in children}) != 1):
                raise TypeError('device output assembly does not support ragged or mixed scalar/array outputs')
            probe = np.stack([child.probe for child in children])
            shape = (*node.shape, *children[0].shape)
            return _Expression(kind, shape, _array_probe(probe.dtype, shape), children, node.shape)
        if kind in {'reshape', 'transpose', 'select', 'scatter'}:
            child = self._expression(node.children[0])
            shape = tuple(node.shape)
            if kind == 'reshape':
                if prod(shape) != prod(child.shape):
                    raise ValueError('reshape changes the number of result elements')
                probe = _array_probe(child.dtype, shape)
            elif kind == 'transpose':
                if len(node.value) != len(child.shape) or tuple(child.shape[i] for i in node.value) != shape:
                    raise ValueError('transpose child shape differs from the output layout')
                probe = _array_probe(child.dtype, shape)
            elif kind == 'scatter':
                expected = (*(len(axis) for axis in node.value), *shape[len(node.value):])
                if child.shape != expected:
                    raise ValueError('scatter block shape differs from its target positions')
                probe = _array_probe(child.dtype, shape)
            else:
                if (len(node.value) > len(child.shape)
                        or any(index >= size for index, size in zip(node.value, child.shape))):
                    raise ValueError('select indexes outside the leading result axes')
                shape = child.shape[len(node.value):]
                probe = _array_probe(child.dtype, shape)
                if not shape:
                    probe = probe[()]  # Fully indexed arrays become NumPy scalars.
            return _Expression(kind, shape, probe, (child,), node.value)
        raise TypeError(f'device output assembly does not support numeric node {kind!r}')


def describe_device_output_fields(recipe, problem):
    """Return exact (path, shape, dtype.str) device field metadata, without CUDA."""
    return tuple((path, expression.shape, expression.dtype.str)
                 for path, expression in _Description(recipe, problem).fields)


def estimate_device_output_metadata_bytes(recipe, problem):
    """Bytes in each host/device pointer-table payload (zero for direct fields)."""
    return _Description(recipe, problem).table_nbytes


def estimate_device_output_bytes(recipe, problem):
    """Reserve fields, literals, metadata and reusable scratch; exclude input terms."""
    return _device_output_bytes(_Description(recipe, problem))


def _device_output_bytes(description):
    return (sum(prod(expression.shape) * expression.dtype.itemsize
                for _, expression in description.fields)
            + sum(value.nbytes for value in description.constants) + description.table_nbytes
            + description.scratch_nbytes)


def _device_output_owner_bytes(description):
    # These are separate empty/asarray requests in _BoundDeviceOutput. Scratch
    # views share their slot owner; stages do not allocate one owner each.
    return (*tuple(prod(expression.shape) * expression.dtype.itemsize
                   for _, expression in description.fields),
            *tuple(value.nbytes for value in description.constants),
            *tuple(size * dtype.itemsize for size, dtype in description.scratch_slots),
            *((description.table_nbytes,) if description.table_nbytes else ()))


def describe_device_output_reservation(recipe, problem):
    """Describe fields and both reservation payloads in one offline traversal."""
    description = _Description(recipe, problem)
    return dict(fields=tuple((path, expression.shape, expression.dtype.str)
                             for path, expression in description.fields),
                device_bytes=_device_output_bytes(description),
                device_owner_bytes=_device_output_owner_bytes(description),
                metadata_bytes=description.table_nbytes)


def _coordinates(index, shape):
    return tuple(f'(({index}) / {prod(shape[axis + 1:]) or 1}LL % {size or 1}LL)'
                 for axis, size in enumerate(shape))


def _linear(coordinates, shape):
    return ' + '.join(f'({coordinate}) * {prod(shape[axis + 1:])}LL'
                      for axis, coordinate in enumerate(coordinates)) or '0LL'


def _broadcast(index, target, source):
    if target == source:
        return index
    coordinates = _coordinates(index, target)[len(target) - len(source):]
    return _linear(tuple('0LL' if size == 1 else coordinate
                         for coordinate, size in zip(coordinates, source, strict=True)), source)


def _literal(value, dtype):
    dtype = np.dtype(dtype)
    if dtype.kind == 'c':
        real_dtype = np.dtype('float32' if dtype.itemsize == 8 else 'float64')
        return f'{_CTYPE[dtype]}({_literal(value.real, real_dtype)}, {_literal(value.imag, real_dtype)})'
    if dtype.kind == 'f':
        value = float(value)
        if isfinite(value):
            return f'{_CTYPE[dtype]}({value.hex()})'
        if np.isnan(value):
            return f'{_CTYPE[dtype]}(NAN)'
        return f'{_CTYPE[dtype]}({"-" if value < 0 else ""}INFINITY)'
    if dtype.kind == 'b':
        return 'true' if value else 'false'
    value = int(value)
    if value == -(2 ** 63):
        return '(-9223372036854775807LL - 1LL)'
    return f'{value}{"ULL" if dtype.kind == "u" else "LL"}'


_INTEGER_LITERALS = {dtype: (_literal(0, dtype), _literal(1, dtype)) for dtype in _DTYPES}


def _cast(value, dtype):
    return f'({_CTYPE[np.dtype(dtype)]})({value})'


class _Emitter:
    def __init__(self, description, *, indirect=False):
        self.description, self.lines, self.arguments = description, [], {}
        self.indirect = indirect
        self.literal_text = {}
        self.counter = 0

    def line(self, text):
        self.lines.append(text)

    def variable(self, dtype, value):
        name = f'v{self.counter}'
        self.counter += 1
        self.lines.append(f'const {_CTYPE[dtype]} {name} = {value};')
        return name

    def emit(self, node, index):
        kind, dtype = node.kind, node.dtype
        if kind == 'constant':
            value = node.value
            value_type = type(value)
            # Native integer zero/one have no signed-zero or conversion-hook ambiguity.
            if value_type is int and value in (0, 1):
                literal = _INTEGER_LITERALS[dtype][value]
            else:
                if value_type is float:
                    key = dtype, value_type, value.hex()
                elif value_type is complex:
                    key = dtype, value_type, value.real.hex(), value.imag.hex()
                elif value_type is int or value_type is bool:
                    key = dtype, value_type, value
                else:
                    key = None
                if key is None:
                    literal = _literal(value, dtype)
                else:
                    literal = self.literal_text.get(key)
                    if literal is None:
                        literal = _literal(value, dtype)
                        if len(self.literal_text) < _MAX_LITERAL_TEXT_ENTRIES:
                            self.literal_text[key] = literal
            name = f'v{self.counter}'
            self.counter += 1
            self.lines.append(f'const {_CTYPE[dtype]} {name} = {literal};')
            return name
        if kind in {'input', 'literal_array', 'temporary'}:
            key = (kind, node.value)
            if key not in self.arguments:
                if kind in {'input', 'temporary'}:
                    specs = self.description.term_specs if kind == 'input' else self.description.temp_specs
                    shape, arg_dtype = specs[node.value]
                else:
                    array = self.description.constants[node.value]
                    shape, arg_dtype = array.shape, array.dtype
                self.arguments[key] = (f'x{len(self.arguments)}', shape, arg_dtype)
            name, shape, arg_dtype = self.arguments[key]
            if self.indirect:
                offset = self.description.table_offsets[key]
                if shape:
                    coordinates = _coordinates(index, shape)
                    position = ' + '.join(f'({coordinate}) * ((long long)addresses[{offset + axis + 1}])'
                        for axis, coordinate in enumerate(coordinates))
                else:
                    position = '0LL'
                value = (f'*reinterpret_cast<const {_CTYPE[arg_dtype]}*>('
                         f'reinterpret_cast<const char*>(addresses[{offset}]) + ({position}))')
            else:
                value = f'{name}[{index}]'
            ctype = _CTYPE[dtype]
            name = f'v{self.counter}'
            self.counter += 1
            self.lines.append(f'const {ctype} {name} = ({ctype})({value});')
            return name
        if kind == 'zeros':
            return self.variable(dtype, _cast('0', dtype))
        if kind in {'add', 'multiply'}:
            # Flat ordered sums/products lower to a left-deep arithmetic tree.
            # Explicit postorder avoids Python recursion limits without
            # reassociation or changing any intermediate rounding boundary.
            pending, values = [(node, index, False)], []
            while pending:
                item, position, ready = pending.pop()
                if ready:
                    # Only binary nodes place a ready frame on this stack.
                    right, left = values.pop(), values.pop()
                    operation = '+' if item.kind == 'add' else '*'
                    item_dtype = item.dtype
                    ctype = _CTYPE[item_dtype]
                    name = f'v{self.counter}'
                    self.counter += 1
                    self.lines.append(
                        f'const {ctype} {name} = ed_output_round(({ctype})({left}) {operation} ({ctype})({right}));')
                    values.append(name)
                elif item.kind in {'add', 'multiply'}:
                    pending.append((item, position, True))
                    left, right = item.children
                    shape = item.shape
                    pending.append((right, position if shape == right.shape else
                                    _broadcast(position, shape, right.shape), False))
                    pending.append((left, position if shape == left.shape else
                                    _broadcast(position, shape, left.shape), False))
                else:
                    values.append(self.emit(item, position))
            return values[0]
        child = node.children[0]
        if kind == 'cast':
            return self.variable(dtype, f'ed_output_round({_cast(self.emit(child, index), dtype)})')
        if kind == 'power':
            value = self.emit(child, index)
            exponent = node.value
            if exponent == 0:
                expression = _cast('1', dtype)
            elif exponent == 1:
                expression = _cast(value, dtype)
            elif exponent == 2:
                expression = f'{_cast(value, dtype)} * {_cast(value, dtype)}'
            elif exponent == -1:
                expression = f'{_cast("1", dtype)} / {_cast(value, dtype)}'
            elif dtype.kind in 'biu' or (dtype.kind == 'c' and abs(exponent) <= 100):
                # NumPy/CPython complex integer powers use repeated squaring
                # for small exponents; this also preserves zero**positive-int.
                base = self.variable(dtype, _cast(value, dtype))
                power = self.variable(dtype, _cast('1', dtype))
                remaining = abs(exponent)
                while remaining:
                    if remaining & 1:
                        power = self.variable(dtype, _cast(f'{power} * {base}', dtype))
                    remaining >>= 1
                    if remaining:
                        base = self.variable(dtype, _cast(f'{base} * {base}', dtype))
                expression = (f'{_cast("1", dtype)} / {power}' if exponent < 0 else power)
            else:
                expression = f'pow({_cast(value, dtype)}, {_cast(str(exponent), dtype)})'
            return self.variable(dtype, f'ed_output_round({expression})')
        if kind == 'reshape':
            return self.emit(child, index)
        if kind == 'transpose':
            coordinates = _coordinates(index, node.shape)
            old_coordinates = tuple(coordinates[node.value.index(axis)] for axis in range(len(child.shape)))
            return self.emit(child, _linear(old_coordinates, child.shape))
        if kind == 'select':
            offset = sum(value * prod(child.shape[axis + 1:]) for axis, value in enumerate(node.value))
            return self.emit(child, f'({index}) + {offset}LL')
        result = f'v{self.counter}'
        self.counter += 1
        self.line(f'{_CTYPE[dtype]} {result};')
        if kind == 'scatter':
            coordinates = list(_coordinates(index, node.shape))
            condition = []
            for axis, positions in enumerate(node.value):
                inverse = '-1LL'
                for source, target in reversed(tuple(enumerate(positions))):
                    inverse = f'({coordinates[axis]} == {target}LL ? {source}LL : {inverse})'
                name = f's{self.counter}'
                self.counter += 1
                self.line(f'const long long {name} = {inverse};')
                coordinates[axis] = name
                condition.append(f'{name} >= 0')
            self.line(f'if ({" && ".join(condition)}) {{')
            value = self.emit(child, _linear(coordinates, child.shape))
            self.line(f'{result} = {value};')
            self.line('} else {')
            self.line(f'{result} = {_cast("0", dtype)};')
            self.line('}')
        elif kind == 'chunks':
            start = 0
            for lane, item in enumerate(node.children):
                end = start + prod(item.shape)
                self.line(f'{"if" if lane == 0 else "else if"} (({index}) < {end}LL) {{')
                value = self.emit(item, f'({index}) - {start}LL')
                self.line(f'{result} = {_cast(value, dtype)};')
                self.line('}')
                start = end
            self.line(f'else {{ {result} = {_cast("0", dtype)}; }}')
        elif kind == 'array':
            size = prod(child.shape)
            for lane, item in enumerate(node.children):
                self.line(f'{"if" if lane == 0 else "else if"} (({index}) / {size or 1}LL == {lane}LL) {{')
                value = self.emit(item, f'({index}) % {size or 1}LL')
                self.line(f'{result} = {_cast(value, dtype)};')
                self.line('}')
            self.line(f'else {{ {result} = {_cast("0", dtype)}; }}')
        else:
            raise AssertionError(kind)
        return result


def describe_device_output_program(recipe, problem):
    """Describe support, bounded stages and peak reserved bytes without CUDA.

    The recipe must already have passed R validation at the workflow boundary.
    Unsupported shapes/dtypes raise TypeError/ValueError before any GPU work.
    ``metadata_bytes`` is the size of each host/device pointer-table payload;
    ``peak_scratch_bytes`` includes all reusable device scratch slot owners.
    Source sizes describe generated kernel bodies, not GPU compilation time.
    """
    description = _Description(recipe, problem)
    source_bytes, field_stages = [], [0] * len(description.fields)
    for index, stage in enumerate(description.stages):
        emitter = _Emitter(description, indirect=description.indirect_stages[index])
        value = emitter.emit(stage.expression, 'i')
        emitter.line(f'output[i] = {value};')
        source_bytes.append(len('\n'.join(emitter.lines).encode()))
        field_stages[stage.field] += 1
    output_bytes = sum(prod(expression.shape) * expression.dtype.itemsize
                       for _, expression in description.fields)
    constant_bytes = sum(value.nbytes for value in description.constants)
    return dict(supported=True, field_count=len(description.fields), stage_count=len(description.stages),
        fields=[dict(path=path, shape=expression.shape, dtype=expression.dtype.str,
                     node_count=expression.work, stage_count=field_stages[index])
                for index, (path, expression) in enumerate(description.fields)],
        field_node_counts=[expression.work for _, expression in description.fields],
        field_stage_counts=field_stages,
        stage_node_counts=[stage.expression.work for stage in description.stages],
        stage_source_bytes=source_bytes, source_bytes=sum(source_bytes),
        stage_indirect=list(description.indirect_stages),
        peak_scratch_bytes=description.scratch_nbytes, scratch_slot_count=len(description.scratch_slots),
        metadata_bytes=description.table_nbytes, output_bytes=output_bytes, constant_bytes=constant_bytes,
        device_bytes=output_bytes + constant_bytes + description.scratch_nbytes + description.table_nbytes)


def _restore(structure, fields):
    kind = structure[0]
    if kind == 'field':
        return fields[structure[1]]
    _, keys, children = structure
    values = [_restore(child, fields) for child in children]
    if kind == 'dict':
        return dict(zip(keys, values, strict=True))
    return values if kind == 'list' else tuple(values)


def _output_kernel_options(backend):
    # Compiler dialect and rounding are implementation requirements, not task
    # parameters. Reading is_hip does not initialize a device. Never send NVRTC
    # flags to HIPRTC; its clang frontend controls contraction differently.
    runtime = getattr(getattr(backend, 'cuda', None), 'runtime', None)
    contraction = '-ffp-contract=off' if getattr(runtime, 'is_hip', False) else '--fmad=false'
    return ('--std=c++17', contraction)


@lru_cache(maxsize=16)
def _output_kernel(factory, params, output, operation, options, version):
    # Cache code only: no recipe, constants, term values, pointers or cfg IDs.
    # CuPy owns the per-device/signature binary cache and disk-cache validity.
    # Naming by code also reuses identical stages at different field positions.
    from hashlib import sha256
    native_v2 = (type(version) is int and version == 2
                 and type(params) is str and type(output) is str and type(operation) is str
                 and type(_PREAMBLE) is str and type(options) is tuple
                 and all(type(option) is str for option in options))
    if native_v2:
        # Versioned, length-framed raw source avoids its large escaped repr.
        identity = sha256(b'ED.output-kernel\x00v2\x00')
        for value in (params, output, operation, _PREAMBLE):
            payload = value.encode('utf-8', 'surrogatepass')
            identity.update(len(payload).to_bytes(8, 'little'))
            identity.update(payload)
        payload = repr((options, version)).encode()
        identity.update(len(payload).to_bytes(8, 'little'))
        identity.update(payload)
    else:
        # Preserve legacy versions and extension metadata conversion behavior.
        identity = sha256(repr((params, output, operation, _PREAMBLE, options, version)).encode())
    name = f'ed_output_v{version}_{identity.hexdigest()[:24]}'
    return factory(params, output, operation, name, preamble=_PREAMBLE, options=options)


def _output_compile_failure(error, backend, options):
    # Failure-only diagnostics. Keep normal execution free of file IO, device
    # queries, test launches and synchronization. A failed log write must not
    # hide the original compiler failure.
    import tempfile
    try:
        with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8',
                prefix='lqcdweave-output-compile-', suffix='.log', delete=False) as log:
            log.write(f'output implementation: {_OUTPUT_KERNEL_VERSION}\n')
            log.write(f'backend: {getattr(backend, "__version__", "unknown")}\noptions: {options!r}\n')
            runtime = getattr(getattr(backend, 'cuda', None), 'runtime', None)
            for name in ('is_hip', 'runtimeGetVersion', 'driverGetVersion', 'getDevice'):
                try:
                    value = getattr(runtime, name)
                    log.write(f'{name}: {value() if callable(value) else value}\n')
                except Exception:
                    log.write(f'{name}: unavailable\n')
            log.write(f'{type(error).__name__}: {error}\n')
            log.write(str(getattr(error, 'code', '')))
        detail = f'diagnostic log: {log.name}'
    except OSError:
        detail = 'diagnostic log unavailable; see the chained compiler error'
    return RuntimeError(f'GPU output kernel compilation failed; {detail}')


class _BoundDeviceOutput:
    def __init__(self, description, backend, debug):
        self.description, self.backend, self.debug = description, backend, debug
        self.compiler_options = _output_kernel_options(backend)
        compiler = getattr(getattr(backend, 'cuda', None), 'compiler', None)
        self.compile_error = getattr(compiler, 'CompileException', ())
        self.programs = []
        for number, stage in enumerate(description.stages):
            expression = stage.expression
            indirect = description.indirect_stages[number]
            emitter = _Emitter(description, indirect=indirect)
            value = emitter.emit(expression, 'i')
            emitter.line(f'output[i] = {value};')
            arguments = tuple(emitter.arguments.items())
            params = ('raw uint64 addresses' if indirect else
                      ', '.join(f'raw {spec[2].name} {spec[0]}' for _, spec in arguments))
            kernel = _output_kernel(backend.ElementwiseKernel, params,
                f'raw {expression.dtype.name} output', '\n'.join(emitter.lines),
                self.compiler_options, _OUTPUT_KERNEL_VERSION)
            self.programs.append((kernel, tuple(key for key, _ in arguments), expression, indirect, stage.target))
        self.constants = tuple(backend.asarray(value) for value in description.constants)
        self.call_owners = self.constants
        self.metadata_bytes = description.table_nbytes
        self.scratch_bytes = description.scratch_nbytes
        self.stage_count = len(description.stages)
        self.last_stage_launches = 0
        self.fields = tuple((path, expression.shape, expression.dtype.str)
                            for path, expression in description.fields)
        self.nbytes = (sum(prod(expression.shape) * expression.dtype.itemsize for _, expression in description.fields)
                       + sum(value.nbytes for value in description.constants) + description.table_nbytes
                       + description.scratch_nbytes)
        self._owner_layout_schemas = {}

    def _validate_term_values(self, values):
        for index, value in enumerate(values):
            if not isinstance(value, self.backend.ndarray):
                raise TypeError('device output assembly requires CuPy term arrays; host fallback is disabled')
            if (tuple(value.shape), np.dtype(value.dtype)) != self.description.term_specs[index]:
                raise ValueError('term value differs from its bound device output shape/dtype')

    @staticmethod
    def _owner_schema(layout):
        return tuple((len(keys), tuple(shape), np.dtype(dtype))
                     for keys, shape, dtype in layout.groups)

    def _validate_snapshot_owners(self, values, schema):
        owners = values.owners
        if len(owners) != len(schema):
            raise ValueError('device output snapshot owner count mismatch')
        for owner, (lanes, shape, dtype) in zip(owners, schema, strict=True):
            if not isinstance(owner, self.backend.ndarray):
                raise TypeError('device output snapshot requires backend owner arrays')
            if (tuple(owner.shape) != (lanes, *shape)
                    or np.dtype(owner.dtype) != dtype):
                raise ValueError('snapshot owner differs from its bound output layout')

    def _remember_owner_schema(self, layout, schema):
        self._owner_layout_schemas[id(layout)] = (ref(layout), schema)
        if len(self._owner_layout_schemas) > _MAX_OUTPUT_LAYOUT_SCHEMAS:
            del self._owner_layout_schemas[next(iter(self._owner_layout_schemas))]

    def _validate_layout_terms(self, layout, schema):
        for (group, lane), spec in zip(layout.positions, self.description.term_specs, strict=True):
            if (group < 0 or group >= len(schema) or lane < 0 or lane >= schema[group][0]):
                raise ValueError('device output snapshot lane is outside its bound layout')
            if schema[group][1:] != spec:
                raise ValueError('term value differs from its bound device output shape/dtype')

    def __call__(self, values):
        self.last_stage_launches = 0
        expected = len(self.description.term_specs)
        try:
            count = len(values)
        except TypeError as exc:
            raise TypeError('ordered term values must be a sized sequence') from exc
        if count != expected:
            raise ValueError(f'ordered term value count mismatch; expected={expected}, actual={count}')
        debug = self.debug or debug_enabled()
        compact = type(values) is _OutputOwnerSnapshot
        layout = (values.provenance if compact or type(values) is _BoundLayoutPackedOutputs else None)
        native_layout = type(layout) is BoundOutputLayout
        cache_key = id(layout) if native_layout else None
        cached = self._owner_layout_schemas.get(cache_key) if native_layout else None
        identity_hit = cached is not None and cached[0]() is layout
        if cached is not None and not identity_hit:
            del self._owner_layout_schemas[cache_key]
        if identity_hit and not debug:
            self._validate_snapshot_owners(values, cached[1])
        elif compact and native_layout and not debug:
            # The snapshot was copied by this frozen X layout. Prove its
            # ordered ABI from metadata once, without constructing Term views.
            schema = self._owner_schema(layout)
            self._validate_layout_terms(layout, schema)
            self._validate_snapshot_owners(values, schema)
            self._remember_owner_schema(layout, schema)
        else:
            # First handoff, external values, and every debug call keep the
            # complete term-by-term kernel ABI validation.
            self._validate_term_values(values)
            if native_layout and not debug:
                schema = self._owner_schema(layout)
                self._validate_snapshot_owners(values, schema)
                self._remember_owner_schema(layout, schema)
        scratch = tuple(self.backend.empty((size,), dtype=dtype)
                        for size, dtype in self.description.scratch_slots)
        temporaries = tuple(scratch[self.description.temp_slots[index]][:prod(shape)].reshape(shape)
                            for index, (shape, _) in enumerate(self.description.temp_specs))
        def argument(kind, index):
            return values[index] if kind == 'input' else (temporaries[index]
                if kind == 'temporary' else self.constants[index])
        self.call_owners = (*self.constants, *scratch)
        table = None
        if self.description.table_nbytes:
            pointers = np.empty(self.description.table_nbytes // 8, dtype='uint64')
            strides = pointers.view('int64')
            owner_metadata = (tuple((owner.data.ptr, owner.strides[0], owner.strides[1:])
                                    for owner in values.owners)
                              if compact and not debug else ())
            positions = values.positions if compact and not debug else ()
            for (kind, index), shape, dtype, offset in self.description.table_layout:
                if compact and kind == 'input' and not debug:
                    group, lane = positions[index]
                    pointer, lane_stride, value_strides = owner_metadata[group]
                    pointers[offset] = pointer + lane * lane_stride
                else:
                    value = argument(kind, index)
                    pointers[offset] = value.data.ptr
                    value_strides = value.strides
                if shape:
                    for axis in range(len(shape)):
                        strides[offset + axis + 1] = value_strides[axis]
            table = self.backend.asarray(pointers)
            self.call_owners = (*self.constants, *scratch, pointers, table)
        results = [None] * len(self.description.fields)
        for kernel, arguments, expression, indirect, (kind, target) in self.programs:
            output = (temporaries[target] if kind == 'temporary' else
                      self.backend.empty(expression.shape, dtype=expression.dtype))
            if prod(expression.shape):
                inputs = ((table,) if indirect else tuple(
                    argument(kind, index) for kind, index in arguments))
                try:
                    kernel(*inputs, output, size=prod(expression.shape))
                except self.compile_error as error:
                    raise _output_compile_failure(error, self.backend, self.compiler_options) from error
                self.last_stage_launches += 1
            if kind == 'field':
                results[target] = output
        return _restore(self.description.structure, results)


def bind_device_output_assembly(recipe, problem, backend, *, debug=False):
    """Bind fixed-point numeric output fusion on the caller's current stream.

    Every final numeric field is an independently owned CuPy array, including
    scalars (0-D). Object/ragged arrays and extended precision fail explicitly;
    this opt-in boundary never falls back to CPU. Oversized signatures use one
    compact pointer/stride-table upload per call, shared by all affected fields.
    Larger fields are split into ordered stages of at most 8192 expression
    nodes. Explicitly budgeted scratch is reused after each temporary's last
    consumer; scalar-kind and arithmetic rounding remain unchanged.
    The callable retains immutable recipe constants, kernel metadata and the
    most recent scratch and metadata owners in ``call_owners``. An asynchronous consumer
    must capture that tuple per call until its completion event. No runtime
    term values are cached, results transferred, or stream synchronization done.
    Compiler settings are private backend policy. CuPy compiles each actual
    array signature on first use and reuses its native device/disk cache;
    binding alone does not claim that a device binary has been compiled.
    """
    if type(debug) is not bool:
        raise TypeError('debug must be bool')
    if (getattr(backend, '__name__', '') != 'cupy'
            or not callable(getattr(backend, 'ElementwiseKernel', None))):
        raise TypeError('device output assembly requires the CuPy backend')
    return _BoundDeviceOutput(_Description(recipe, problem), backend, debug)


__all__ = ['bind_device_output_assembly', 'describe_device_output_fields',
           'estimate_device_output_bytes', 'estimate_device_output_metadata_bytes',
           'describe_device_output_program']
