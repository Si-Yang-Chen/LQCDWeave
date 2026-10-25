"""Place a described task on a target without importing its array runtime."""
from dataclasses import replace


def retarget_expanded(expanded, backend):
    """Change allocation placement, preserving scientific identities/readers."""
    if backend not in ('numpy', 'cupy'):
        raise ValueError('unsupported target array backend')
    if expanded.inputs is None:
        return expanded
    cache = {}

    def place(inputs):
        if id(inputs) in cache:
            return cache[id(inputs)]
        gpu = backend == 'cupy'
        blocks = {key: replace(block,
            host_bytes=max(block.host_bytes, block.read_bytes),
            device_bytes=(max(block.device_bytes, block.host_bytes, block.read_bytes)
                          if gpu and block.read_kind == 'dense' else 0),
            transfer_bytes=max(block.transfer_bytes, block.read_bytes) if gpu else 0,
            staging_host_bytes=(block.staging_host_bytes + max(block.host_bytes, block.read_bytes)
                                if gpu and block.read_kind == 'dense' else block.staging_host_bytes),
            staging_device_bytes=block.staging_device_bytes if gpu else 0)
            for key, block in inputs.blocks.items()}
        preparations = {}
        for key, prep in inputs.preparations.items():
            workspace = prep.workspace_host_bytes + prep.workspace_device_bytes
            device_prepare = gpu and prep.recipe is not None
            preparations[key] = replace(prep,
                workspace_host_bytes=0 if device_prepare else workspace,
                workspace_device_bytes=workspace if device_prepare else 0)
        problem = replace(inputs.problem, backend=backend) if inputs.problem is not None else None
        value = replace(inputs, problem=problem, blocks=blocks, preparations=preparations,
            assembly_host_bytes=inputs.assembly_host_bytes + inputs.assembly_device_bytes,
            assembly_device_bytes=0,
            task_batches=tuple(place(item) for item in inputs.task_batches),
            batch_candidates={width: tuple(place(item) for item in items)
                              for width, items in inputs.batch_candidates.items()})
        cache[id(inputs)] = value
        return value

    return replace(expanded, inputs=place(expanded.inputs))


__all__ = ['retarget_expanded']
