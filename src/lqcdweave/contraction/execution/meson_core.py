"""Fixed local whole-meson algorithms; no path search or numerical cache."""
from math import prod
import numpy as np

from lqcdweave.contracts.correlator.api import (
    Action, AxisPattern, FactorizedWideReduction, TensorSpec,
)


def bind_meson_whole_action(action, catalog, backend):
    """Bind geometry once, resolving every numerical operand at compute time.

    Unit spin keeps Phi compact. Connected kernels form two mode products,
    then the original terminal FWR for complex64. Version two complex128
    closes modes after the first phase map, applying the other phase only to
    the small result. Version three retains the original small spin-Gram
    algorithm when both spatial contribution axes are singleton. Saved older
    versions keep their own arithmetic.
    A loop's two endpoints may belong to the same operand.
    The dense fallback performs the existing K/H algebra inside this kernel.
    No operation graph is searched, copied or stored in the archive.
    """
    from .operations.contract import bind_dense_action
    from .factorized_reduction import bind_factorized_reduction
    kernel = action.core
    equation = kernel.private_equation
    specs = tuple(catalog[key] for key in action.operands)
    dtype = specs[0].dtype
    dimensions = {label: size for labels, spec in zip(equation.inputs, specs, strict=True)
                  for label, size in zip(labels, spec.shape, strict=True)}
    lane = kernel.lane_axis

    def bare(labels):
        return tuple(a for a in labels if a != lane)

    def with_lane(labels, inputs):
        labels = bare(labels)
        return ((lane, *labels) if lane is not None and any(lane in item for item in inputs)
                else tuple(labels))

    def pair(axes, output, layouts):
        local = AxisPattern(tuple(axes), tuple(output))
        result = TensorSpec(tuple(dimensions[a] for a in output), layouts[0].dtype)
        keys = ('left', 'right')
        local_action = Action('contract', 'result', keys, local)
        function = bind_dense_action(local_action, dict(zip(keys, layouts)) | {'result': result}, backend)
        return function, result

    if kernel.representation == 'unit-spin' and dtype == np.dtype('complex64') and kernel.kind == 'loop':
        core = FactorizedWideReduction(equation, tuple(spec.shape for spec in specs), (0, 1),
            tile_elements=4096, output_dtype='complex128')
        terminal = Action('contract', 'result', tuple(map(str, range(3))), equation,
            core=core, wide_accumulation=True)
        local_catalog = dict(zip(terminal.operands, specs)) | {'result': TensorSpec(core.output_shape, core.result_dtype)}
        return bind_factorized_reduction(terminal, local_catalog, backend)

    count = 2 if kernel.kind == 'connected' else 1
    product_axes, product_specs, products = [], [], []
    for index in range(count):
        prop = bare(equation.inputs[index])
        spatial = bare(equation.inputs[count + index])
        if kernel.representation == 'unit-spin':
            # P[B,A,O,N] Phi[c,N,Q] -> K[c,Q,B,A,O].
            output = (spatial[0], spatial[-1], *prop[:3])
            if kernel.kind == 'loop':
                # The trace retains spin until its small phase factor closes.
                output = (spatial[0], *prop[:2])
        else:
            # Full Gamma*Phi[c,A,D,N,Q] -> K[c,B,D,O,Q].
            output = (spatial[0], prop[0], spatial[2], prop[2], spatial[-1])
            if kernel.kind == 'loop':
                output = (spatial[0],)
        axes = (equation.inputs[index], equation.inputs[count + index])
        output = with_lane(output, axes)
        function, spec = pair(axes, output, (specs[index], specs[count + index]))
        product_axes.append(output)
        product_specs.append(spec)
        products.append(function)

    if kernel.kind == 'connected' and kernel.representation == 'unit-spin' and dtype == np.dtype('complex64'):
        # Preserve contribution multiplicity and terminal cast: never first
        # multiply phase into a complex64 K and lose the wide sum boundary.
        axes = (*equation.inputs[4:], *product_axes)
        layouts = (*specs[4:], *product_specs)
        local = AxisPattern(axes, equation.output)
        core = FactorizedWideReduction(local, tuple(spec.shape for spec in layouts), (2, 3),
            tile_elements=4096)
        terminal = Action('contract', 'result', tuple(map(str, range(4))), local,
            core=core, wide_accumulation=True)
        local_catalog = dict(zip(terminal.operands, layouts)) | {'result': TensorSpec(core.output_shape, core.result_dtype)}
        finish = bind_factorized_reduction(terminal, local_catalog, backend)

        def connected_wide(values):
            left, _ = products[0]((values[0], values[2]))
            right, _ = products[1]((values[1], values[3]))
            return finish((values[4], values[5], left, right))
        return connected_wide

    implementation = (None if kernel.kernel_description is None
                      else kernel.kernel_description.implementation)
    version_two = implementation in ('whole-matrix-v2', 'whole-matrix-v3')
    if (version_two and kernel.kind == 'connected'
            and kernel.representation == 'unit-spin' and dtype == np.dtype('complex128')):
        if implementation == 'whole-matrix-v3' and all(spec.shape[-3] == 1 for spec in specs[2:4]):
            # With one spatial block on each side, the old effective kernel
            # closes modes into a small spin Gram before either phase map.
            # Singleton is a structural identity, not a fitted size threshold.
            prop0, prop1 = tuple(map(bare, equation.inputs[:2]))
            space0, space1 = tuple(map(bare, equation.inputs[2:4]))
            phase0 = bare(equation.inputs[4])
            axes = tuple(product_axes)
            gram = with_lane((space1[0], prop1[0], prop1[1],
                              space0[0], prop0[0], prop0[1]), axes)
            close_modes, gram_spec = pair(axes, gram, tuple(product_specs))
            axes = (equation.inputs[4], gram)
            small = with_lane((phase0[0], space1[0], prop1[1], prop0[0]), axes)
            first, small_spec = pair(axes, small, (specs[4], gram_spec))
            finish, _ = pair((equation.inputs[5], small), equation.output, (specs[5], small_spec))

            def connected_gram(values):
                left, _ = products[0]((values[0], values[2]))
                right, _ = products[1]((values[1], values[3]))
                current, _ = close_modes((left, right))
                current, _ = first((values[4], current))
                return finish((values[5], current))
            return connected_gram

        # Preserve the effective fixed chain of the old ordinary matrix path:
        # phase0*K0 -> H0; K1*H0 -> small; phase1*small -> output. Mapping both
        # large K values first needlessly repeats a mode-sized phase product.
        # This is a local algorithm, not a path search or an archived subgraph.
        prop0, prop1 = tuple(map(bare, equation.inputs[:2]))
        phase0 = bare(equation.inputs[4])
        axes = (equation.inputs[4], product_axes[0])
        mapped = with_lane((phase0[0], phase0[3], bare(product_axes[0])[1],
                            prop0[0], prop0[2]), axes)
        first, first_spec = pair(axes, mapped, (specs[4], product_specs[0]))
        axes = (product_axes[1], mapped)
        small = with_lane((phase0[0], prop0[0], bare(product_axes[1])[0], prop1[1]), axes)
        close_modes, small_spec = pair(axes, small, (product_specs[1], first_spec))
        finish, _ = pair((equation.inputs[5], small), equation.output, (specs[5], small_spec))

        def connected_chain(values):
            left, _ = products[0]((values[0], values[2]))
            right, _ = products[1]((values[1], values[3]))
            left, _ = first((values[4], left))
            current, _ = close_modes((right, left))
            return finish((values[5], current))
        return connected_chain

    mapped_axes, mapped_specs, maps = [], [], []
    for index in range(count):
        phase = bare(equation.inputs[2 * count + index])
        product = bare(product_axes[index])
        if kernel.kind == 'loop':
            output = equation.output
        elif kernel.representation == 'unit-spin':
            prop = bare(equation.inputs[index])
            output = (phase[0], prop[0], phase[3], prop[2], product[1])
        else:
            output = (phase[0], *product[1:])
        axes = (equation.inputs[2 * count + index], product_axes[index])
        output = with_lane(output, axes)
        function, spec = pair(axes, output, (specs[2 * count + index], product_specs[index]))
        mapped_axes.append(output)
        mapped_specs.append(spec)
        maps.append(function)

    if kernel.kind == 'loop':
        def loop(values):
            current, _ = products[0]((values[0], values[1]))
            return maps[0]((values[2], current))
        return loop

    finish, _ = pair(tuple(mapped_axes), equation.output, tuple(mapped_specs))

    def connected(values):
        left, _ = products[0]((values[0], values[2]))
        right, _ = products[1]((values[1], values[3]))
        left, _ = maps[0]((values[4], left))
        right, _ = maps[1]((values[5], right))
        return finish((left, right))
    return connected


__all__ = ['bind_meson_whole_action']
