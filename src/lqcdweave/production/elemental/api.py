"""Public API for meson, baryon, path and raw current elementals."""

from .frontend import BaryonElementalGenerator
from .frontend import ElementalGenerator
from .kernels import generate_baryon_elemental, re_combine
from .workspace import ElementalWorkspace

__all__ = [
    "ElementalGenerator",
    "BaryonElementalGenerator",
    "ElementalWorkspace",
    "generate_baryon_elemental",
    "re_combine",
]

# Stable numerical boundary for existing dataset workflows. These engines
# retain endpoint views and streaming callbacks; user tasks use the table frontend.
from .generator import _ElementalEngine as ElementalKernel
from .baryon import _BaryonEngine as BaryonElementalKernel
from .structure import elemental_structures as describe_elemental_structures
