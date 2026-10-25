"""Native HDF5 fields for complete cfg results; temporal arithmetic stays in TS."""
from pathlib import Path
from math import prod
from collections.abc import Mapping
import numpy as np
from lqcdweave.contracts.dataset import DatasetSpec, canonical_json
from lqcdweave.contracts.time_selection import TimeSelectionRecord
from lqcdweave.sampling.temporal.api import TimePairAccumulator
from .dataset import DatasetWriter, DatasetReader

def _at(value,path):
    for key in path: value=value[key]
    return value


def _validate_packed(value,shape):
    if not isinstance(value,np.ndarray) or value.shape!=shape or value.dtype!=np.dtype('int64'):
        raise ValueError('prepared payload differs from the saved host byte layout')


def _document_value(value):
    """Detach frozen metadata for DatasetSpec's JSON-native document."""
    if isinstance(value,Mapping): return {key:_document_value(item) for key,item in value.items()}
    if isinstance(value,(list,tuple)): return [_document_value(item) for item in value]
    return value


def _output_coordinates(plan, coordinates):
    explicit = coordinates is not None
    if coordinates is None:
        points=getattr(plan,'coordinates',None)
        coordinates = (tuple(dict(point) for point in points) if points is not None
                       else tuple(dict(instance.coordinate) for instance in plan.instances))
    else:
        if isinstance(coordinates, (str, bytes, Mapping)):
            raise TypeError('coordinates must be an ordered sequence of coordinate mappings')
        coordinates = tuple(dict(point) for point in coordinates)
    if not coordinates:
        raise ValueError('output coordinates must be nonempty')
    if explicit:
        shape = getattr(plan, 'coordinate_shape', ())
        if len(coordinates) != (prod(shape) if shape else 1):
            raise ValueError('output coordinates differ from the declared coordinate shape')
    return coordinates


def validate_output_binding(plan,binding,*,coordinates=None):
    if binding.physical_lt!=plan.physical_lt or not binding.physical_times:
        raise ValueError('saved reconstruction requires explicit physical time binding')
    coordinates = _output_coordinates(plan, coordinates)
    times=binding.physical_times
    if plan.output_layout=='dt_packed' and plan.one_point_slots and len(times)!=plan.one_point_slots:
        raise ValueError('one-point slots differ from saved physical time binding')
    if (len(set(times))!=len(times) or any(type(t) is not int or not 0<=t<plan.physical_lt for t in times)):
        raise ValueError('invalid actual physical time coordinates')
    pairs=[]
    for point in coordinates:
        if any(role not in point or not 0<=point[role]<len(times) for role in plan.time_roles):
            raise ValueError('time coordinate outside actual physical time binding')
        pairs.append(tuple(point[role] for role in plan.time_roles))
    if len(set(pairs))!=len(pairs):
        raise ValueError('dt reconstruction requires unique selected time pairs')



def _spec(plan, cfg, coordinates, binding):
    fields = [dict(path=list(f.path), shape=list(f.shape), dtype=f.dtype,
                   time_role=f.time_role, averaged_dt=f.dt_offset >= 0) for f in plan.output_fields]
    return DatasetSpec(dict(schema=1, kind='correlator',
        configuration=dict(ensemble=binding.ensemble, configuration=None if cfg is None else str(cfg),
            lattice=list(binding.lattice or (1,1,1,binding.physical_lt or plan.physical_lt or 1))),
        axes=[], dtype=None, bases={},
        definition=dict(encoding='native-hdf5-fields-v1', fields=fields,
            task=_document_value(plan.output_definition), coordinate_shape=list(plan.coordinate_shape),
            coordinates=_document_value(coordinates), time_roles=list(plan.time_roles),
            physical_times=list(binding.physical_times), physical_lt=plan.physical_lt,
            physical_dt=list(plan.dt_coordinates) if plan.output_layout=='dt_packed' else list(range(plan.physical_lt or 0)),
            output_layout=plan.output_layout, measured_pair_counts=[],
            missing_dt='nan', dt_aggregation='mean over measured pairs'),
        conventions=dict(numeric_dtypes='native per field', time_axis='trailing',
            lattice_declared=binding.lattice is not None), parents=list(binding.parents)))


def _numeric_field(field, value, debug):
    value = np.asarray(value)
    if value.shape != field.shape or value.dtype.str != field.dtype:
        raise ValueError('output differs from saved field shape/dtype')
    if debug and value.dtype.kind in 'fc' and not np.isfinite(value).all():
        raise ValueError('nonfinite correlator output')
    return value


def _unpack(packed, field, *, offset=None, shape=None, words=None):
    shape = field.shape if shape is None else shape
    offset = field.offset if offset is None else offset
    words = field.words if words is None else words
    payload = packed.reshape(-1)[offset:offset+words].view(np.uint8)
    return payload[:prod(shape)*np.dtype(field.dtype).itemsize].view(np.dtype(field.dtype)).reshape(shape)


def _coordinate_chunks(field_shape, scan_shape, dtype):
    """Tile one numerical field at a time, with singleton trailing scan axes."""
    shape = (*field_shape, *scan_shape)
    if not shape or any(size == 0 for size in shape):
        return None
    chunks = [*field_shape, *([1] * len(scan_shape))]
    while prod(chunks) * np.dtype(dtype).itemsize > 1024 * 1024:
        axis = max(range(len(field_shape)), key=chunks.__getitem__)
        chunks[axis] = (chunks[axis] + 1) // 2
    return tuple(chunks)


class CfgOutputWriter:
    def __init__(self,plan,cfg,directory,binding,*,debug=False,coordinates=None,prepacked=False):
        self.plan,self.cfg,self.binding=plan,cfg,binding
        self.debug,self.prepacked=debug,prepacked
        self.coordinates=_output_coordinates(plan,coordinates)
        if plan.physical_lt is not None:
            validate_output_binding(plan,binding,coordinates=coordinates)
        self.directory=Path(directory);self._seen=set();self.sums={};self.counts=None
        if plan.physical_lt is not None and not prepacked:
            self.accumulator=TimePairAccumulator(
                {f.path:(f.shape,f.dtype) for f in plan.output_fields if f.dt_offset>=0},Lt=plan.physical_lt)
            self.counts,self.sums=self.accumulator.counts,self.accumulator.sums
        self.writer=None

    def _ensure_writer(self):
        if self.writer is None:
            self.writer=DatasetWriter(self.directory,_spec(self.plan,self.cfg,self.coordinates,self.binding),
                provenance=dict(self.binding.provenance),format='hdf5',structured=True)

    def _store(self,index,values):
        if index in self._seen or not 0<=index<len(self.coordinates):
            raise ValueError('duplicate or invalid coordinate result')
        self._ensure_writer()
        scan_shape=tuple(self.plan.coordinate_shape)
        scan_index=tuple(np.unravel_index(index,scan_shape)) if scan_shape else ()
        for field,value in values:
            value=_numeric_field(field,value,self.debug)
            self.writer.write_field(('pairs',*field.path),value,
                shape=(*field.shape,*scan_shape),selection=(*[slice(None)]*len(field.shape),*scan_index),
                chunks=_coordinate_chunks(field.shape,scan_shape,field.dtype))
        self._seen.add(index)

    def consume(self,index,value):
        self._store(index,((f,_at(value,f.path)) for f in self.plan.output_fields))
        if self.counts is not None:
            point=self.coordinates[index];sink,source=self.plan.time_roles;times=self.binding.physical_times
            self.accumulator.add(times[point[source]],times[point[sink]],
                ((path,_at(value,path)) for path in self.sums))

    def consume_packed(self,index,packed):
        if not self.prepacked:
            raise ValueError('prepacked input requires prepacked writer mode')
        _validate_packed(packed,self.plan.output_storage_shape)
        self._store(index,((f,_unpack(packed,f)) for f in self.plan.output_fields))

    def _store_dt(self,counts,records=None):
        self._ensure_writer()
        self.counts=counts
        covered=np.flatnonzero(counts)
        batches=((dt,None) for dt in covered) if records is None else zip(covered,records,strict=True)
        for dt,packed in batches:
            if packed is not None:
                _validate_packed(packed,self.plan.dt_storage_shape)
            for field in self.plan.output_fields:
                if field.dt_offset<0:continue
                value=(self.accumulator.mean(field.path,int(dt),inplace=True) if packed is None
                    else _unpack(packed,field,offset=field.dt_offset))
                if value.dtype.kind in 'fc' and not np.isfinite(value).all():
                    raise ValueError('nonfinite reconstructed correlator output')
                self.writer.write_field(('dt',*field.path),value,
                    shape=(*field.shape,self.plan.physical_lt),
                    selection=(*[slice(None)]*len(field.shape),int(dt)),
                    chunks=_coordinate_chunks(field.shape,(self.plan.physical_lt,),field.dtype),
                    fillvalue=np.asarray(np.nan,dtype=value.dtype)[()])

    def _publish(self):
        if len(self._seen)!=len(self.coordinates):
            raise ValueError('incomplete coordinate coverage')
        self._ensure_writer()
        self.writer.update_definition(measured_pair_counts=[] if self.counts is None else self.counts.tolist())
        try:return self.writer.publish()
        finally:self.writer.close();self.sums.clear()

    def publish(self):
        if self.counts is not None and self.plan.dt_words:
            self._store_dt(self.counts)
        return self._publish()

    def publish_prepacked(self,counts,records):
        if not self.prepacked:
            raise ValueError('prepacked publication requires prepacked writer mode')
        try:
            if self.plan.physical_lt is not None and self.plan.dt_words:
                self._store_dt(counts,iter(records))
            return self._publish()
        except BaseException as error:
            try:self.abort(error)
            except BaseException as cleanup:
                if callable(getattr(error,'add_note',None)):error.add_note(str(cleanup))
            raise

    def abort(self,error):
        if self.writer is not None:self.writer.close()
        self.sums.clear()


class PackedDtOutputWriter(CfgOutputWriter):
    def __init__(self,plan,cfg,directory,binding,*,debug=False,coordinates=None,prepacked=False):
        super().__init__(plan,cfg,directory,binding,debug=debug,coordinates=coordinates,prepacked=prepacked)
        if plan.one_point_slots and len(binding.physical_times)!=plan.one_point_slots:
            self.abort(ValueError());raise ValueError('one-point slots differ from physical time binding')
        self.loops={f.path:np.zeros((plan.one_point_slots,*f.shape),dtype=f.dtype)
            for f in plan.output_fields if f.dt_offset<0} if not prepacked else {}
        self.seen={path:np.zeros(plan.one_point_slots,dtype=bool) for path in self.loops}

    def consume(self,index,value):
        if index in self._seen or not 0<=index<len(self.coordinates):
            raise ValueError('duplicate or invalid coordinate result')
        point=self.coordinates[index];times=self.binding.physical_times;sink,source=self.plan.time_roles
        def contributions():
            for field in self.plan.output_fields:
                array=_numeric_field(field,_at(value,field.path),self.debug)
                if field.dt_offset>=0:
                    yield field.path,array
                else:
                    slot=point[field.time_role]
                    if not self.seen[field.path][slot]:
                        self.loops[field.path][slot]=array;self.seen[field.path][slot]=True
        self.accumulator.add(times[point[source]],times[point[sink]],contributions(),reject_equal=True)
        self._seen.add(index)

    def _write_final(self,packed=None,one_point=None):
        self._ensure_writer()
        offsets={True:0,False:0}
        for field in self.plan.output_fields:
            dt=field.dt_offset>=0;n=len(self.plan.dt_coordinates) if dt else self.plan.one_point_slots
            shape=(*field.shape,n);words=field.words*n
            if packed is None:
                if dt:
                    array=np.empty(shape,dtype=field.dtype)
                    self.accumulator.write_means(field.path,self.plan.dt_coordinates,array,missing='nan')
                else:
                    if not self.seen[field.path].all():raise ValueError('incomplete one-point time coverage')
                    array=np.moveaxis(self.loops[field.path],0,-1)
            else:
                array=_unpack(packed if dt else one_point,field,offset=offsets[dt],shape=shape,words=words)
            if self.debug and dt:
                for column,coordinate in enumerate(self.plan.dt_coordinates):
                    if self.counts[coordinate] and not np.isfinite(array[...,column]).all():
                        raise ValueError('nonfinite reconstructed output')
            self.writer.write_field(('dt' if dt else 'one_point',*field.path),array,
                chunks=_coordinate_chunks(field.shape,(n,),field.dtype))
            offsets[dt]+=words
        self.loops.clear();self.seen.clear()

    def publish(self):
        self._write_final()
        return self._publish()

    def publish_prepacked(self,counts,packed,one_point=None):
        if not self.prepacked:raise ValueError('prepacked publication requires prepacked writer mode')
        if counts.shape!=(self.plan.physical_lt,):raise ValueError('counts differ from physical Lt')
        _validate_packed(packed,self.plan.dt_packed_storage_shape)
        if self.plan.one_point_storage_shape!=(0,0):
            _validate_packed(one_point,self.plan.one_point_storage_shape)
        self.counts=counts;self._seen=set(range(len(self.coordinates)))
        self._write_final(packed,one_point)
        return self._publish()

    def abort(self,error):
        super().abort(error);self.loops.clear();self.seen.clear()


def load_correlator_output(directory,path=(),*,dt=False,configuration=None):
    with DatasetReader(directory,cfg=configuration) as reader:
        definition=reader.spec.document['definition']
        if definition.get('encoding')!='native-hdf5-fields-v1':
            raise ValueError('unknown native output encoding')
        field=next((field for field in definition['fields'] if canonical_json(field['path'])==canonical_json(list(path))),None)
        if field is None:raise KeyError(path)
        if definition['output_layout']=='dt_packed':
            role='dt' if field['averaged_dt'] else 'one_point'
        else:role='dt' if dt else 'pairs'
        value=reader.read_field((role,*path))
        expected=tuple(field['shape'])
        if role=='dt':expected+=tuple([len(definition['physical_dt'])])
        elif role=='one_point':expected+=tuple([len(definition['physical_times'])])
        else:expected+=tuple(definition['coordinate_shape'])
        if value.shape!=expected or value.dtype.str!=field['dtype']:
            raise ValueError('native output field shape/dtype differs')
        return value
