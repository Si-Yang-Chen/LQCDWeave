"""Physical slab descriptions for native baryon preparation leaves."""
from lqcdweave.contracts.correlator.api import BatchTensorBinding, PhysicalSlabBinding


def register_baryon_slab(registry, key, reader, family, arguments, selection, materialize, workspace):
    if not callable(getattr(reader, 'describe_slab', None)) or not callable(getattr(reader, 'read_slab', None)):
        return
    selection = dict(selection)
    if family == 'propagator':
        time, source, sector, *context = arguments
        fixed = (source, sector, *context)
        def describe(times):
            return reader.describe_slab(times, *fixed, **selection)
        def read(times):
            return reader.read_slab(times, *fixed, **selection)
        def background(times):
            return reader.read_slab_in_background(times, *fixed, **selection)
    else:
        component, momentum, time, sector = arguments
        fixed = (component, momentum, sector)
        def describe(times):
            return reader.describe_slab(component, momentum, times, sector, **selection)
        def read(times):
            return reader.read_slab(component, momentum, times, sector, **selection)
        def background(times):
            return reader.read_slab_in_background(component, momentum, times, sector, **selection)
    identity = ('baryon-slab', id(reader), family, repr(fixed),
                tuple(sorted((name, repr(value)) for name, value in selection.items())))
    slab = PhysicalSlabBinding(identity, describe, read,
        (lambda metadata: reader.read_slab_workspace_bytes(metadata, **selection))
        if callable(getattr(reader, 'read_slab_workspace_bytes', None)) else None,
        background if getattr(reader, 'background_read_safe', False) is True
        and callable(getattr(reader, 'read_slab_in_background', None)) else None)
    registry.batch_bindings[key] = BatchTensorBinding(key, (slab,), (time,), (materialize,),
        workspace_host_bytes=workspace)
