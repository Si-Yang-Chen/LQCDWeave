"""Materialize one logical source through an injected storage executor."""

from functools import partial


def materialize_source(source_execution, tensor, source, *, timed, device=False):
    """Call the source executor's explicit materialization operation."""
    return timed(
        "materialize",
        partial(source_execution.materialize, tensor, source),
        device=device,
    )


def materialize_from_blocks(loader, tensor, source, blocks, *, timed, device=False):
    """Materialize one logical tensor from an explicit borrowed-block map."""
    return timed(
        "materialize",
        partial(loader.materialize, tensor, source, blocks),
        device=device,
    )


__all__ = ["materialize_from_blocks", "materialize_source"]
