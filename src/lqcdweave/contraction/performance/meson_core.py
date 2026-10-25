"""Constant-size coarse pricing for a closed whole-meson join."""
from math import prod


def estimate_meson_whole(kernel, specs, output_shape):
    """Do not inspect values or enumerate a private contraction path."""
    layouts = tuple(specs[key] for key in kernel.inputs)
    size = sum(prod(spec.shape) * spec.dtype.itemsize for spec in layouts)
    outputs = prod(output_shape)
    propagators = layouts[:2 if kernel.kind == 'connected' else 1]
    extent = max(max(spec.shape[-2:]) for spec in propagators)
    spaces = layouts[len(propagators):2*len(propagators)]
    components = sum(spec.shape[-3] if kernel.representation == 'unit-spin'
                     else spec.shape[-5] for spec in spaces)
    flops = 32 * max(1, components) * extent**3 + 256 * outputs * extent**2
    # Deliberately coarse. The local implementation bounds its own wide tiles;
    # the ordinary execution allocator and existing global reserve still apply.
    vertices = sum(prod(specs[key].shape) * specs[key].dtype.itemsize for key in kernel.cores)
    mode_products = sum(prod(spec.shape) * spec.dtype.itemsize for spec in spaces)
    if kernel.representation == 'unit-spin':
        mode_products *= 16
    workspace = 4 * (size + vertices + mode_products)
    if layouts[0].dtype.name == 'complex64' and kernel.representation == 'unit-spin':
        dimensions = {a: n for axes, spec in zip(kernel.private_equation.inputs, layouts)
                      for a, n in zip(axes, spec.shape)}
        lanes = 1 if kernel.lane_axis is None else dimensions[kernel.lane_axis]
        gram = (256 * prod(spec.shape[-3] for spec in spaces) if kernel.kind == 'connected'
                else 16 * spaces[0].shape[-3])
        workspace += lanes * min(4096, gram) * min(64, max(1, (extent**2 + 1023)//1024)) * 16
    return flops, 3 * size, workspace
