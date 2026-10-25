"""Complete cfg delivery around existing SceneStoragePlan templates."""
from dataclasses import InitVar, dataclass, is_dataclass
from math import prod
import numpy as np
from typing import Mapping
from types import MappingProxyType
from .preparation import ArrayConstant
from .storage import SceneStoragePlan, SourcePreparation, StorageEvent
from .ir import TensorSpec
from .outputs import OutputRecipe
from .dense_batch import DenseKernelBatch
from .gemm_group import DenseGemmGroup, dense_batch_leaves


@dataclass(frozen=True)
class _CfgBindingValidation:
    """Identity proof for one already accepted immutable binding graph."""
    templates: tuple
    instances: tuple


def _freeze_definition(value):
    if isinstance(value,Mapping):
        return MappingProxyType({key:_freeze_definition(item) for key,item in value.items()})
    if isinstance(value,(list,tuple)):
        return tuple(_freeze_definition(item) for item in value)
    return value


@dataclass(frozen=True)
class CfgOutputField:
    """Lossless byte packing of one numeric leaf; offsets are int64 words."""
    path: tuple
    shape: tuple[int,...]
    dtype: str
    offset: int
    words: int
    dt_offset: int = -1
    time_role: str | None = None

    def __post_init__(self):
        dtype=np.dtype(self.dtype)
        if (dtype.kind not in 'biufc' or any(type(n) is not int or n<0 for n in self.shape)
                or type(self.offset) is not int or self.offset<0
                or type(self.words) is not int or self.words!=(prod(self.shape)*dtype.itemsize+7)//8
                or type(self.dt_offset) is not int or self.dt_offset<-1):
            raise ValueError('invalid numeric output field layout')
        if self.dt_offset>=0 and dtype.kind not in 'fc':
            raise ValueError('dt means require floating or complex numerical fields')


@dataclass(frozen=True)
class CfgInstance:
    template: int
    coordinate: tuple[tuple[str, int], ...]
    node_ids: ArrayConstant
    reused: ArrayConstant
    events: tuple[StorageEvent, ...]
    releases: tuple[int, ...]
    action_offset: int
    kernel_batches: tuple = ()
    tensor_releases: ArrayConstant | None = None
    # Empty retains the original one-instance/one-coordinate convention.
    point_indices: tuple[int, ...] = ()

    def __post_init__(self):
        object.__setattr__(self, 'kernel_batches', tuple(self.kernel_batches))
        table = self.tensor_releases
        if table is not None:
            if (type(table) is not ArrayConstant or table.dtype != '<i8'
                    or len(table.shape) != 2 or table.shape[1] != 2):
                raise ValueError('cfg tensor releases require an int64 (action, global ID) table')


@dataclass(frozen=True)
class CfgTemplate:
    plan: SceneStoragePlan
    keys: tuple[str, ...]
    output: OutputRecipe
    preparation_closure: Mapping[str, SourcePreparation] | None = None
    preparation_specs: Mapping[str, TensorSpec] | None = None
    output_axes: tuple[int | None, ...] = ()

    def __post_init__(self):
        object.__setattr__(self, 'keys', tuple(self.keys))
        closure, specs = self.preparation_closure, self.preparation_specs
        if closure is None and specs is None:
            return
        if closure is None or specs is None:
            raise ValueError('cfg preparation closure and specs must be supplied together')
        if not isinstance(closure, Mapping) or not isinstance(specs, Mapping):
            raise TypeError('cfg preparation closure and specs must be mappings')
        closure = dict(closure)
        specs = dict(specs)
        if (any(not isinstance(key, str) or not key for key in closure)
                or any(not isinstance(prep, SourcePreparation) or prep.recipe is None
                       for prep in closure.values())):
            raise TypeError('cfg preparation closure requires described sources with string keys')
        if (any(not isinstance(key, str) or not key for key in specs)
                or any(not isinstance(spec, TensorSpec) for spec in specs.values())):
            raise TypeError('cfg preparation specs require TensorSpec values with string keys')
        if not set(closure) <= set(self.keys) or not set(specs) <= set(closure):
            raise ValueError('cfg preparation closure keys differ from template roles')
        if any(prep.recipe.kind != 'identity' and key not in specs
               for key, prep in closure.items()):
            raise ValueError('cfg non-identity preparations require saved TensorSpecs')
        if any(prep.recipe.kind in {'stack', 'linear_combination'}
               and any(member not in closure for member in prep.recipe.inputs)
               for prep in closure.values()):
            raise ValueError('cfg preparation closure omits a stack member')
        object.__setattr__(self, 'preparation_closure', MappingProxyType(closure))
        object.__setattr__(self, 'preparation_specs', MappingProxyType(specs))


@dataclass(frozen=True)
class CfgPlan:
    """Read-only parameterized arrangement; handles and arrays stay runtime-only.

    Global IDs identify owned input/preparation/compute values within ONE cfg.
    Instances bind template local IDs using packed integer/bool tables. Their
    storage events and releases were selected before delivery, not at replay.
    """
    templates: tuple[CfgTemplate, ...]
    instances: tuple[CfgInstance, ...]
    requests: Mapping
    blocks: Mapping
    preparations: Mapping
    specs: Mapping
    peak_host_bytes: int
    prefetch_host_bytes: int
    output_mode: str
    output_host_bytes: int
    statistics: Mapping
    coordinate_shape: tuple[int, ...] = ()
    reader_aliases: tuple[tuple[str, ...], ...] = ()
    output_fields: tuple[CfgOutputField,...] = ()
    output_words: int = 0
    dt_words: int = 0
    physical_lt: int | None = None
    output_storage_shape: tuple[int,int] = (0,0)
    dt_storage_shape: tuple[int,int] = (0,0)
    time_roles: tuple[str,str] = ('0','1')
    output_definition: Mapping = None
    output_layout: str = 'pairs_and_dt'
    dt_coordinates: tuple[int,...] = ()
    dt_packed_storage_shape: tuple[int,int] = (0,0)
    one_point_storage_shape: tuple[int,int] = (0,0)
    one_point_slots: int = 0
    backend: str = 'numpy'
    peak_device_bytes: int = 0
    prefetch_device_bytes: int = 0
    execution_target: object = None
    target_summary: Mapping | None = None
    implementation_version: str = 'unified-device-v1'
    # Batched instances still publish the original ordered physical points.
    output_coordinates: tuple[tuple[tuple[str, int], ...], ...] = ()
    # InitVar keeps this process-local proof out of equality and archives.
    # dataclasses.replace carries it through getattr; changed binding objects
    # never match the proof and are checked just like external constructors.
    _binding_validation: InitVar[object] = None

    def __post_init__(self, _binding_validation):
        object.__setattr__(self,'templates',tuple(self.templates))
        object.__setattr__(self,'instances',tuple(self.instances))
        object.__setattr__(self,'output_definition',_freeze_definition(self.output_definition or {}))
        if self.target_summary is not None:
            if not isinstance(self.target_summary, Mapping):
                raise TypeError('target_summary must be a mapping or None')
            object.__setattr__(self,'target_summary',_freeze_definition(self.target_summary))
        for name in ('requests','blocks','preparations','specs','statistics'):
            object.__setattr__(self,name,MappingProxyType(dict(getattr(self,name))))
        if self.backend not in {'numpy','cupy'}:
            raise ValueError('cfg backend must be numpy or cupy')
        if (not isinstance(self.implementation_version,str) or not self.implementation_version
                or self.implementation_version.strip()!=self.implementation_version):
            raise ValueError('implementation_version must be a nonempty string')
        target_backend=getattr(self.execution_target,'backend',None)
        if target_backend is not None and target_backend!=self.backend:
            raise ValueError('execution target backend differs from cfg backend')
        if self.output_mode not in {'memory','dataset'}:
            raise ValueError('cfg output mode must be memory or dataset')
        if self.output_layout not in {'pairs_and_dt','dt_packed'}:
            raise ValueError('unknown cfg output layout')
        if self.output_layout=='dt_packed':
            if self.output_mode!='dataset' or not self.physical_lt or not self.dt_coordinates or not self.dt_words:
                raise ValueError('packed dt requires dataset reconstruction fields and coordinates')
            if (len(set(self.dt_coordinates))!=len(self.dt_coordinates)
                    or any(type(dt) is not int or not 1<=dt<self.physical_lt for dt in self.dt_coordinates)):
                raise ValueError('invalid packed physical dt coordinates')
            auxiliary=[f for f in self.output_fields if f.dt_offset<0]
            if any(f.time_role not in self.time_roles for f in auxiliary):
                raise ValueError('one-point field requires an explicit saved time role')
            if type(self.one_point_slots) is not int or self.one_point_slots<0:
                raise ValueError('invalid one-point slot count')
            if auxiliary and self.one_point_slots<=max(int(value) for point in self.coordinates for role,value in point if role in self.time_roles):
                raise ValueError('one-point layout omits requested time slots')
            sizes=(self.dt_words*len(self.dt_coordinates),sum(f.words for f in auxiliary)*self.one_point_slots)
            for shape,size in zip((self.dt_packed_storage_shape,self.one_point_storage_shape),sizes):
                if len(shape)!=2 or any(type(n) is not int or n<0 for n in shape) or prod(shape)<size:
                    raise ValueError('packed cfg output storage shape is too small')
        if not self.templates or not self.instances:
            raise ValueError('cfg plan needs templates and instances')
        if self.output_coordinates:
            count=len(self.output_coordinates)
            if (count != (prod(self.coordinate_shape) if self.coordinate_shape else count)
                    or sorted(position for instance in self.instances for position in instance.point_indices)
                    != list(range(count))):
                raise ValueError('cfg batches must cover every declared coordinate exactly once')
            if any(len(template.output_axes) != len(template.plan.state.results) for template in self.templates):
                raise ValueError('cfg batches require one lane axis per output term')
        for name in ('peak_host_bytes','prefetch_host_bytes','output_host_bytes',
                     'peak_device_bytes','prefetch_device_bytes'):
            if type(getattr(self,name)) is not int or getattr(self,name)<0:
                raise ValueError('cfg resource bounds must be nonnegative integer bytes')
        if self.backend=='numpy' and (self.peak_device_bytes or self.prefetch_device_bytes):
            raise ValueError('NumPy cfg plans cannot retain device resources')
        if self.output_fields:
            words=dt_words=0;paths=set()
            for f in self.output_fields:
                if f.path in paths or f.offset!=words:
                    raise ValueError('output packing must have unique fields and contiguous offsets')
                paths.add(f.path);words+=f.words
                if f.dt_offset>=0:
                    if f.dt_offset!=dt_words: raise ValueError('dt packing has invalid offsets')
                    dt_words+=f.words
            if words!=self.output_words or dt_words!=self.dt_words:
                raise ValueError('output packing size differs from field layouts')
            for shape,size in ((self.output_storage_shape,words),(self.dt_storage_shape,dt_words)):
                if (len(shape)!=2 or any(type(n) is not int or n<0 for n in shape)
                        or prod(shape)<size or (size and 0 in shape)):
                    raise ValueError('invalid saved output storage shape')
            if self.physical_lt is not None and (type(self.physical_lt) is not int or self.physical_lt<1):
                raise ValueError('invalid saved physical Lt')
            if dt_words and self.physical_lt is None: raise ValueError('dt packing requires physical Lt')
        if (type(_binding_validation) is _CfgBindingValidation
                and _binding_validation.templates is self.templates
                and _binding_validation.instances is self.instances):
            object.__setattr__(self, '_binding_validation', _binding_validation)
        else:
            immutable = self._validate_bindings()
            object.__setattr__(self, '_binding_validation',
                _CfgBindingValidation(self.templates, self.instances) if immutable else None)

    def _validate_bindings(self):
        """Check external bindings once; mutable duck types are never cached."""
        # Batch membership is a boundary check, but repeatedly scanning all
        # template batches makes it quadratic in the branch count. Templates
        # share immutable batch objects; value-equal decoded copies use the
        # same saved Action indices to locate their comparison candidates.
        batch_indices = []
        immutable = True
        for template in self.templates:
            roots = tuple(getattr(template.plan, 'kernel_batches', ()))
            selected = roots + tuple(b for b in dense_batch_leaves(roots) if isinstance(b, DenseKernelBatch))
            state = template.plan.state
            immutable = immutable and (
                type(template) is CfgTemplate and type(template.keys) is tuple
                and type(template.plan) is SceneStoragePlan
                and is_dataclass(state) and type(state).__dataclass_params__.frozen
                and type(state.actions) is tuple
                and type(template.plan.kernel_batches) is tuple
                and all(type(batch) in (DenseKernelBatch, DenseGemmGroup) for batch in selected))
            by_actions = {}
            for batch in selected:
                by_actions.setdefault(batch.action_indices, []).append(batch)
            batch_indices.append(({id(batch) for batch in selected}, by_actions))
        for instance in self.instances:
            immutable = immutable and (
                type(instance) is CfgInstance
                and type(instance.node_ids) is ArrayConstant
                and type(instance.reused) is ArrayConstant
                and all(type(batch) in (DenseKernelBatch, DenseGemmGroup) for batch in instance.kernel_batches))
            if not 0 <= instance.template < len(self.templates):
                raise ValueError('cfg instance names an unknown template')
            template=self.templates[instance.template]
            if (instance.node_ids.shape != (len(template.keys),)
                    or instance.node_ids.dtype != '<i8'
                    or instance.reused.shape != (len(template.plan.state.actions),)
                    or instance.reused.dtype != '|b1'):
                raise ValueError('cfg instance binding table differs from template')
            if instance.tensor_releases is not None:
                rows = instance.tensor_releases.array()
                if (np.any(rows < 0) or np.any(rows[:, 0] >= len(template.plan.state.actions))
                        or np.any(rows[1:, 0] < rows[:-1, 0])):
                    raise ValueError('cfg tensor release has an invalid Action index or global ID')
            selected_ids, selected_batches = batch_indices[instance.template]
            if not instance.kernel_batches:
                continue
            reused=instance.reused.array()
            for batch in instance.kernel_batches:
                if (id(batch) not in selected_ids
                        and batch not in selected_batches.get(batch.action_indices, ())):
                    raise ValueError('cfg instance names a batch absent from its template')
                if any(reused[i] for i in batch.action_indices):
                    raise ValueError('cfg instance batch contains a reused Action')
        return immutable

    @property
    def coordinates(self):
        return self.output_coordinates or tuple(instance.coordinate for instance in self.instances)


__all__=['CfgInstance','CfgTemplate','CfgPlan','CfgOutputField']
