"""Neutral whole-vertex implementation declaration, with no numerical owner."""
from dataclasses import dataclass


@dataclass(frozen=True)
class VertexKernelDescription:
    kind: str
    implementation: str

    def __post_init__(self):
        if (self.kind, self.implementation) not in (
                ('baryon', 'ordered-join-v1'), ('meson', 'whole-matrix-v1'),
                ('meson', 'whole-matrix-v2'), ('meson', 'whole-matrix-v3')):
            raise ValueError('unknown whole-vertex implementation')
