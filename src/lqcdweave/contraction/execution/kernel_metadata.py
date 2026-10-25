"""X's data-free implementation declarations for contraction orchestration."""
from lqcdweave.contracts.correlator.api import VertexKernelDescription


def describe_vertex_kernel(kind):
    if kind not in ('baryon', 'meson'):
        raise ValueError('unknown local join kind')
    return VertexKernelDescription(kind, 'ordered-join-v1' if kind == 'baryon' else 'whole-matrix-v3')
