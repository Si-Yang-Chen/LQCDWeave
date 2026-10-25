"""External formats, numerical backends, and cluster adapters."""

from .dense_dirac import DenseDiracBackend
from .quda import QudaBackend
from .laplace_quda import QudaLaplaceBackend
from .propagator import ArrayPropagator, PropagatorPairReader, select_propagator_times
from .glueball import ArrayGlueball
from .baryon_on_demand import OnDemandBaryonElemental
from .legacy_fields import DenseFileHandle, IldgGauge, QDPTimesliceField
from .scene_basis_region import MaxBasisRegionLoader, RegionReadRequest
from .scene_storage import NpyBlockBinding, NpySceneBlockLoader, NpyPrefixRequest, merge_npy_prefixes
from .scene_device_budget import scene_device_budget

__all__ = ["ArrayGlueball", "DenseDiracBackend", "QudaBackend", "QudaLaplaceBackend",
           "ArrayPropagator", "PropagatorPairReader", "select_propagator_times", "OnDemandBaryonElemental", "DenseFileHandle",
           "IldgGauge", "QDPTimesliceField", "MaxBasisRegionLoader", "RegionReadRequest",
           "NpyBlockBinding", "NpySceneBlockLoader", "NpyPrefixRequest", "merge_npy_prefixes", "scene_device_budget"]
