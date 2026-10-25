"""Select the shared catalog construction for fixed native matrix requests.

Scanned requests retain the existing coordinate batching implementation until
the catalog binder supports that lifecycle. No numeric IO occurs here.
"""
from dataclasses import replace
from lqcdweave.contracts.correlator.api import describe_coordinates
from .matrix_catalog import describe_meson_matrix
from .matrix_inputs import bind_meson_matrix_catalog
from .meson_matrix import expand_meson_matrix as expand_matrix_rows
from .native import NativeDiagramBinding
from .structured import contains_structured


def expand_meson_matrix(correlator, *, adapter, bindings, time_axes=None,
                        time_pairs=None, batch='auto'):
    if not contains_structured(correlator) or not isinstance(adapter, NativeDiagramBinding):
        raise NotImplementedError('meson_matrix requires structured=True and native Meson bindings')
    coordinates, batching = describe_coordinates(time_axes=time_axes, time_pairs=time_pairs, batch=batch)
    if coordinates.scan_shape:
        return expand_matrix_rows(correlator, adapter=adapter, bindings=bindings,
                                 time_axes=time_axes, time_pairs=time_pairs, batch=batch)
    times = dict(coordinates.points[0].values)
    described = describe_meson_matrix(correlator, bindings=bindings, representative_times=times)
    expanded = bind_meson_matrix_catalog(described, bindings=bindings,
                                        coordinates=coordinates, row_block=16,
                                        _factorize_spin=True, _whole_vertices=True)
    # Retain the user's validated batch declaration even for a fixed coordinate.
    inputs = None if expanded.inputs is None else replace(expanded.inputs, batch=batching)
    return replace(expanded, correlator=correlator, inputs=inputs, batch=batching)


def try_expand_search_matrices(correlator, *, adapter, bindings, time_axes=None,
                              time_pairs=None, batch='auto', _factorize_spin=True):
    """Bind fixed dense matrices as closed whole-vertex local operations.

    Component coefficients stay array factors. Mode partitions become physical
    rectangular inputs; radial output projects their small matrices, while
    memory-only partitions sum ordered blocks into the original output shape.
    Private component/spin/phase factors remain data dependencies of the local
    kernel. P prices whole connected/trace operations without searching them.
    Unsupported requests retain the existing general expansion route.
    """
    if not contains_structured(correlator) or not isinstance(adapter, NativeDiagramBinding):
        return None
    coordinates, batching = describe_coordinates(time_axes=time_axes, time_pairs=time_pairs, batch=batch)
    if coordinates.scan_shape:
        return None
    times = dict(coordinates.points[0].values)
    try:
        described = describe_meson_matrix(correlator, bindings=bindings, representative_times=times)
        expanded = bind_meson_matrix_catalog(described, bindings=bindings, coordinates=coordinates,
                                            _factorize_spin=_factorize_spin, _whole_vertices=True)
    except NotImplementedError:
        return None
    inputs = None if expanded.inputs is None else replace(expanded.inputs, batch=batching, matrix_kernels=())
    return replace(expanded, correlator=correlator, inputs=inputs, batch=batching)


__all__ = ['expand_meson_matrix', 'try_expand_search_matrices']
