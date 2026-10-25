"""One implementation for each scheduled contraction operation."""

from .baryon_join import contract_baryon_join
from .contract import contract_dense
from .load import load_retained_source, load_source, read_block
from .materialize import materialize_from_blocks, materialize_source
from .output import copy_output, select_output_lane
from .release import release_tensor
from .sparse_bridge import contract_sparse_bridge, sparse_pairs
from .transfer import copy_to_backend

__all__ = [
    "contract_baryon_join",
    "contract_dense",
    "contract_sparse_bridge",
    "copy_output",
    "copy_to_backend",
    "load_retained_source",
    "load_source",
    "materialize_from_blocks",
    "materialize_source",
    "read_block",
    "release_tensor",
    "select_output_lane",
    "sparse_pairs",
]
