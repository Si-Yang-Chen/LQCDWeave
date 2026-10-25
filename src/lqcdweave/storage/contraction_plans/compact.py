"""Positional contract records and per-value string/description dictionaries.

This is an archive representation, not an execution IR. The existing whitelist,
constructors and semantic checks remain authoritative. Negative scalar integers
reference strings; negative application integers have an explicit scalar tag.
"""
from collections.abc import Mapping
from dataclasses import MISSING, fields
import math

import numpy as np

from . import api


TAG = 'archive.compact-v2'
_INT, _TUPLE, _LIST, _MAP, _SET, _FROZEN = -1, -2, -3, -4, -5, -6
_DTYPE, _COMPLEX, _NUMPY, _REF, _SHARED = -7, -8, -9, -10, -11
_DIRECT_SCALAR_TYPES = (bool, int)
_CONTAINERS = {_TUPLE: tuple, _LIST: list, _SET: set, _FROZEN: frozenset}


def _literal_depth(value):
    """Depth of an immutable default literal, or None for unsupported values."""
    cls=type(value)
    if value is None or cls in (str,bool,int):
        return 0
    if cls is float:
        return 0 if math.isfinite(value) else None
    if cls is tuple:
        depths=[_literal_depth(item) for item in value]
        return None if None in depths else 1+max(depths) if depths else 0
    return None


def _same_literal(value,default):
    if value is default:
        return True
    if type(value) is not type(default):
        return False
    if type(default) is float:
        return value.hex()==default.hex()
    if type(default) is tuple:
        return len(value)==len(default) and all(_same_literal(a,b) for a,b in zip(value,default))
    return value==default


def _encoder(*, preserve_identity=False, omit_defaults=True):
    by_type, _, _ = api._plan_type_maps()
    schemas, schema_ids, strings, string_ids = [], {}, [], {}
    seen, descriptions = {}, {}
    schema_defaults=[]
    # Large occurrence records are positional, not individually dictionary-owned.
    inline = (api.contract_api.Action, api.contract_api.MemoryPoint,
              api.contract_api.ContractionTerm, api.contract_api.PendingTerm)

    def text(value):
        index = string_ids.get(value)
        if index is None:
            index = len(strings)
            string_ids[value] = index
            strings.append(value)
        return -index-1

    def visit(value, depth=0):
        if depth > 128:
            raise api.PlanArchiveError('plan value nesting exceeds the supported limit')
        cls = type(value)
        if cls is str:
            return text(value)
        if value is None or cls is bool:
            return value
        if cls is int:
            return value if value >= 0 else [_INT, value]
        if cls is float:
            if not math.isfinite(value):
                raise api.PlanArchiveError('plan values cannot contain non-finite floats')
            return value
        if cls in by_type:
            previous = seen.get(id(value))
            if previous is not None:
                _, identity, complete = previous
                if not complete:
                    raise api.PlanArchiveError('cyclic shared plan record')
                return [_REF, identity]
            schema = schema_ids.get(cls)
            if schema is None:
                schema = len(schemas)
                names = api._plan_record_schema(cls)[0]
                schema_ids[cls] = schema
                schemas.append([by_type[cls], list(names)])
                defaults={}
                encoded_defaults=[]
                if omit_defaults:
                    for index,item in enumerate(fields(cls)):
                        default=item.default
                        literal_depth=None if default is MISSING else _literal_depth(default)
                        if literal_depth is not None:
                            defaults[index]=(default,literal_depth)
                            encoded_defaults.append([index,visit(default,0)])
                    if encoded_defaults:
                        schemas[schema].append(encoded_defaults)
                schema_defaults.append(defaults)
            names = schemas[schema][1]
            shared = (preserve_identity or cls not in inline) and cls.__dataclass_params__.frozen
            key = None
            if not preserve_identity and cls is api.contract_api.AxisPattern:
                key = (cls, value.inputs, value.output)
            elif not preserve_identity and cls is api.contract_api.TensorSpec and value.description is None:
                key = (cls, value.shape, value.dtype.str)
            elif not preserve_identity and cls is api.contract_api.Factor:
                key = (cls, value.tensor, value.labels)
            if shared:
                previous = descriptions.get(key) if key is not None else None
                if previous is not None:
                    identity, complete = previous
                    if not complete:
                        raise api.PlanArchiveError('cyclic shared plan record')
                    return [_REF, identity]
                identity = len(seen)
                # Every definition consumes exactly one sequential ID, also
                # when equal descriptions from distinct objects are merged.
                # Keep the actual object alive for the encoder's lifetime.
                # Stream callers may release a chunk before the next one;
                # an id without its object can then refer to unrelated data.
                seen[id(value)] = (value, identity, False)
                if key is not None:
                    descriptions[key] = (identity, False)
            defaults=schema_defaults[schema]
            values=[]
            omitted_depth=-1
            if defaults:
                mask=0
                for index,name in enumerate(names):
                    item=getattr(value,name)
                    default=defaults.get(index)
                    if default is not None and _same_literal(item,default[0]):
                        omitted_depth=max(omitted_depth,default[1])
                    else:
                        mask|=1<<index
                        values.append(item if depth < 128 and (item is None or
                            (type(item) in _DIRECT_SCALAR_TYPES and item >= 0)) else
                            text(item) if depth < 128 and type(item) is str else visit(item,depth+1))
                values.insert(0,mask)
            else:
                items=(getattr(value,name) for name in names)
                values=[item if depth < 128 and (item is None or
                    (type(item) in _DIRECT_SCALAR_TYPES and item >= 0)) else
                    text(item) if depth < 128 and type(item) is str else visit(item,depth+1)
                    for item in items]
            if omitted_depth>=0 and depth+1+omitted_depth>128:
                raise api.PlanArchiveError('plan value nesting exceeds the supported limit')
            if shared:
                seen[id(value)] = (value, identity, True)
                if key is not None:
                    descriptions[key] = (identity, True)
                return [_SHARED, identity, schema, *values]
            return [schema, *values]
        if isinstance(value, np.dtype):
            return [_DTYPE, text(value.str)]
        if isinstance(value, np.generic):
            if value.dtype.kind not in 'biufc':
                raise api.PlanArchiveError(f'unsupported NumPy scalar dtype {value.dtype}')
            return [_NUMPY, text(value.dtype.str), text(value.tobytes().hex())]
        if isinstance(value, complex):
            if not math.isfinite(value.real) or not math.isfinite(value.imag):
                raise api.PlanArchiveError('plan values cannot contain non-finite complex numbers')
            return [_COMPLEX, value.real, value.imag]
        if cls is dict or isinstance(value, Mapping):
            result = [_MAP]
            for key, item in value.items():
                result.extend((visit(key, depth+1), visit(item, depth+1)))
            return result
        if cls in (tuple, list):
            return [_TUPLE if cls is tuple else _LIST,
                    *(item if depth < 128 and (item is None or
                       (type(item) in _DIRECT_SCALAR_TYPES and item >= 0)) else
                       text(item) if depth < 128 and type(item) is str else visit(item,depth+1)
                      for item in value)]
        if cls in (set, frozenset):
            # Canonicalize before assigning string/shared IDs, independently of
            # hash iteration order and without keeping a second encoded tree.
            items = sorted(value, key=lambda item: api._json_bytes(api._encode_plan_value(item)))
            return [_SET if cls is set else _FROZEN,
                    *(visit(item, depth+1) for item in items)]
        raise api.PlanArchiveError(f'unsupported plan value type {cls.__module__}.{cls.__qualname__}')

    return schemas, strings, visit


def encode(value, *, preserve_identity=False):
    """Encode one root with local dictionaries; retain no cross-save state."""
    schemas, strings, visit = _encoder(preserve_identity=preserve_identity)
    root = visit(value)
    return {'type': TAG, 'value': {'schemas': schemas, 'strings': strings, 'root': root}}


def _decoder(*, allow_defaults=True, _consume=False):
    _, by_name, _ = api._plan_type_maps()
    schemas, strings, schema_types = [], [], set()
    shared = []

    def default_literal(value,level=0):
        if level>128:
            raise api.PlanArchiveError('plan value nesting exceeds the supported limit')
        cls=type(value)
        if value is None or cls is bool or (cls is int and value>=0):
            return value
        if cls is int:
            return string(value)
        if cls is float and math.isfinite(value):
            return value
        if cls is list and value and type(value[0]) is int:
            if value[0]==_TUPLE:
                return tuple(default_literal(item,level+1) for item in value[1:])
            if value[0]==_INT and len(value)==2 and type(value[1]) is int and value[1]<0:
                return value[1]
        raise api.PlanArchiveError('compact defaults require immutable literal values')

    def extend(raw_schemas, new_strings):
        if (type(new_strings) is not list or any(type(s) is not str for s in new_strings)
                or type(raw_schemas) is not list):
            raise api.PlanArchiveError('invalid compact plan dictionaries')
        strings.extend(new_strings)
        for row in raw_schemas:
            if (type(row) is not list or len(row) not in ((2,3) if allow_defaults else (2,)) or type(row[0]) is not str
                    or row[0] not in by_name or type(row[1]) is not list
                    or any(type(name) is not str for name in row[1])):
                raise api.PlanArchiveError('unknown compact plan record schema')
            cls, names = by_name[row[0]], tuple(row[1])
            supported = api._plan_record_schema(cls)
            actual = frozenset(names)
            compatible = (actual <= supported[1] and supported[1] - actual <= supported[3]
                          and names == tuple(name for name in supported[0] if name in actual))
            if cls in schema_types or (names != supported[0] and not compatible):
                raise api.PlanArchiveError('compact plan fields do not match the supported schema')
            schema_types.add(cls)
            defaults={}
            if len(row)==3:
                if type(row[2]) is not list:
                    raise api.PlanArchiveError('invalid compact default fields')
                current_fields={item.name:item for item in fields(cls)}
                eligible={index for index,name in enumerate(names)
                          if current_fields[name].default is not MISSING
                          and _literal_depth(current_fields[name].default) is not None}
                previous=-1
                for item in row[2]:
                    if (type(item) is not list or len(item)!=2 or type(item[0]) is not int
                            or item[0] not in eligible or item[0]<=previous):
                        raise api.PlanArchiveError('invalid compact default field index')
                    default=default_literal(item[1])
                    defaults[item[0]]=(default,_literal_depth(default))
                    previous=item[0]
            constructor = api._bind_positional_record_constructor(cls, names, row[0], supported[4])
            base=([defaults.get(index,(None,))[0] for index in range(len(names))]
                  if constructor is not None else {names[index]:value for index,(value,_) in defaults.items()})
            required=sum(1<<index for index in range(len(names)) if index not in defaults)
            schemas.append((cls,names,row[0],supported[4],constructor,
                            (defaults,base,required,{}) if len(row)==3 else None))

    def string(index):
        if type(index) is not int or index >= 0 or -index > len(strings):
            raise api.PlanArchiveError('invalid compact string reference')
        return strings[-index-1]

    def record(schema, fields, level):
        if type(schema) is not int or not 0 <= schema < len(schemas):
            raise api.PlanArchiveError('unknown compact plan record')
        cls,names,kind,integer_fields,constructor,snapshot=schemas[schema]
        layout=None
        if snapshot is not None:
            defaults,base,required,layouts=snapshot
            if not fields or type(fields[0]) is not int:
                raise api.PlanArchiveError('invalid compact record field mask')
            mask=fields[0]
            layout=layouts.get(mask)
            if layout is None:
                if mask<0 or mask>>len(names) or mask&required!=required:
                    raise api.PlanArchiveError('invalid compact record field mask')
                count=mask.bit_count()
                if len(fields)-1!=count:
                    raise api.PlanArchiveError('compact plan record has the wrong field count')
                indices=tuple(index for index in range(len(names)) if mask&(1<<index))
                depth=max((item[1] for index,item in defaults.items() if not mask&(1<<index)),default=-1)
                layout=(indices,tuple(names[index] for index in indices),depth,count)
                if len(layouts)<64:
                    layouts[mask]=layout
            elif len(fields)-1!=layout[3]:
                raise api.PlanArchiveError('compact plan record has the wrong field count')
            if layout[2]>=0 and level+1+layout[2]>128:
                raise api.PlanArchiveError('plan value nesting exceeds the supported limit')
            fields=fields[1:]
        elif len(fields)!=len(names):
            raise api.PlanArchiveError('compact plan record has the wrong field count')
        if constructor is not None:
            if layout is not None:
                values=base.copy()
                if level < 128:
                    for index,item in zip(layout[0],fields):
                        values[index]=(item if item is None or (type(item) in _DIRECT_SCALAR_TYPES and item >= 0)
                                       else strings[-item-1] if type(item) is int and -item <= len(strings)
                                       else visit(item, level+1))
                else:
                    for index,item in zip(layout[0],fields):
                        values[index]=visit(item, level+1)
            elif level < 128:
                values = [item if item is None or (type(item) in _DIRECT_SCALAR_TYPES and item >= 0)
                          else strings[-item-1] if type(item) is int and -item <= len(strings)
                          else visit(item, level+1) for item in fields]
            else:
                values = [visit(item, level+1) for item in fields]
            return constructor(values)
        selected_names=names if layout is None else layout[1]
        if level < 128:
            values = {name: item if item is None or (type(item) in _DIRECT_SCALAR_TYPES and item >= 0)
                      else strings[-item-1] if type(item) is int and -item <= len(strings)
                      else visit(item, level+1) for name, item in zip(selected_names, fields)}
        else:
            values = {name: visit(item, level+1) for name, item in zip(selected_names, fields)}
        if layout is not None:
            complete=base.copy()
            complete.update(values)
            values=complete
        return api._construct_plan_record(cls, values, kind, integer_fields)

    def visit(value, level):
        if level > 128:
            raise api.PlanArchiveError('plan value nesting exceeds the supported limit')
        cls = type(value)
        if value is None or cls is bool:
            return value
        if cls is int:
            if value >= 0:
                return value
            # Type and sign are already known here; tagged dtype/NumPy fields
            # still use string() to check their own untrusted references.
            if -value > len(strings):
                raise api.PlanArchiveError('invalid compact string reference')
            return strings[-value-1]
        if cls is float:
            if not math.isfinite(value):
                raise api.PlanArchiveError('plan archive contains a non-finite float')
            return value
        if cls is not list or not value or type(value[0]) is not int:
            raise api.PlanArchiveError('invalid compact plan node')
        tag = value[0]
        # File-owned record/container subtrees can be discarded after use.
        # Their short reference/scalar nodes disappear with the parent.
        if tag >= 0:
            result = record(tag, value[1:], level)
            if _consume:
                value.clear()
            return result
        if tag == _REF:
            if (len(value) != 2 or type(value[1]) is not int
                    or not 0 <= value[1] < len(shared) or shared[value[1]] is None):
                raise api.PlanArchiveError('invalid or forward compact plan reference')
            return shared[value[1]]
        if tag == _SHARED:
            if (len(value) < 3 or type(value[1]) is not int or value[1] != len(shared)
                    or type(value[2]) is not int or not 0 <= value[2] < len(schemas)
                    or not schemas[value[2]][0].__dataclass_params__.frozen):
                raise api.PlanArchiveError('invalid compact shared plan definition')
            identity = value[1]
            shared.append(None)
            shared[identity] = record(value[2], value[3:], level)
            if _consume:
                value.clear()
            return shared[identity]
        if tag == _INT and len(value) == 2 and type(value[1]) is int and value[1] < 0:
            return value[1]
        container = _CONTAINERS.get(tag)
        if container is not None:
            if level < 128:
                items = [item if item is None or (type(item) in _DIRECT_SCALAR_TYPES and item >= 0)
                         else strings[-item-1] if type(item) is int and -item <= len(strings)
                         else visit(item, level+1) for item in value[1:]]
            else:
                items = [visit(item, level+1) for item in value[1:]]
            result = container(items)
            if _consume:
                value.clear()
            return result
        if tag == _MAP and len(value) % 2 == 1:
            result = {}
            for i in range(1, len(value), 2):
                key, item = value[i], value[i+1]
                if level < 128:
                    if key is not None and not (type(key) in _DIRECT_SCALAR_TYPES and key >= 0):
                        key = (strings[-key-1] if type(key) is int and -key <= len(strings)
                               else visit(key, level+1))
                    if item is not None and not (type(item) in _DIRECT_SCALAR_TYPES and item >= 0):
                        item = (strings[-item-1] if type(item) is int and -item <= len(strings)
                                else visit(item, level+1))
                else:
                    key, item = visit(key, level+1), visit(item, level+1)
                if key in result:
                    raise api.PlanArchiveError('compact mapping contains duplicate keys')
                result[key] = item
            if _consume:
                value.clear()
            return result
        if tag == _DTYPE and len(value) == 2:
            return np.dtype(string(value[1]))
        if tag == _COMPLEX and len(value) == 3:
            if any(type(item) not in (int, float) or not math.isfinite(item) for item in value[1:]):
                raise api.PlanArchiveError('invalid compact complex value')
            return complex(value[1], value[2])
        if tag == _NUMPY and len(value) == 3:
            dtype, raw = np.dtype(string(value[1])), bytes.fromhex(string(value[2]))
            if dtype.kind not in 'biufc' or len(raw) != dtype.itemsize:
                raise api.PlanArchiveError('invalid compact NumPy scalar')
            return np.frombuffer(raw, dtype=dtype, count=1)[0]
        raise api.PlanArchiveError('unknown or malformed compact plan node')

    return extend, visit


def decode(payload, *, depth=0, allow_defaults=True, _consume=False):
    """Decode checked schemas once and construct the original contract types."""
    if type(payload) is not dict or set(payload) != {'schemas', 'strings', 'root'}:
        raise api.PlanArchiveError('invalid compact plan fields')
    extend, visit = _decoder(allow_defaults=allow_defaults, _consume=_consume)
    extend(payload['schemas'], payload['strings'])
    try:
        return visit(payload['root'], depth)
    except api.PlanArchiveError:
        raise
    except (TypeError, ValueError, KeyError, OverflowError) as exc:
        raise api.PlanArchiveError(f'invalid compact plan value: {exc}') from exc


class StreamEncoder:
    """One file's dictionaries; chunks retain the original identity partition."""

    def __init__(self, *, omit_defaults=True):
        self.schemas, self.strings, self.visit = _encoder(preserve_identity=True,omit_defaults=omit_defaults)

    def chunk(self, values):
        schema_offset, string_offset = len(self.schemas), len(self.strings)
        roots = [self.visit(value) for value in values]
        return self.schemas[schema_offset:], self.strings[string_offset:], roots


class StreamDecoder:
    """Resolve dictionaries and immutable record references across file chunks."""

    def __init__(self, *, allow_defaults=True):
        self.extend, self.visit = _decoder(allow_defaults=allow_defaults)

    def chunk(self, schemas, strings, roots):
        self.extend(schemas, strings)
        try:
            return [self.visit(root, 0) for root in roots]
        except api.PlanArchiveError:
            raise
        except (TypeError, ValueError, KeyError, OverflowError) as exc:
            raise api.PlanArchiveError(f'invalid compact plan value: {exc}') from exc
