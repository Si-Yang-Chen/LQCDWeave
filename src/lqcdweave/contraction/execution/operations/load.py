"""Read an explicitly supplied logical source through a loader interface."""

from functools import partial


def load_source(loader, tensor, source, regions, *, timed):
    """Load or view a source while retaining shared-region reuse semantics.

    This operation only calls the caller-provided loader. It does not discover
    filenames or interpret external formats.
    """
    if hasattr(loader, "load_region") and hasattr(loader, "view"):
        if source.region not in regions:
            regions[source.region] = timed(
                "region_open_or_read",
                partial(loader.load_region, source.region, source),
                nbytes=source.region_bytes,
            )
        return timed(
            "view_or_read",
            partial(loader.view, regions[source.region], tensor, source),
            nbytes=source.snapshot_bytes,
        )
    if source.region in regions:
        raise TypeError("shared regions require a load_region/view loader")
    value = timed("source_get", partial(loader, tensor, source), nbytes=source.region_bytes)
    regions[source.region] = value
    return value


def read_block(loader, block_id, block, *, timed):
    """Read one declared block without resolving its storage location."""
    return timed(
        "read_block",
        partial(loader.read_block, block_id, block),
        nbytes=block.read_bytes,
        device=bool(block.device_bytes),
    )


def load_retained_source(source_execution, tensor, arrays, *, timed):
    """Restore a retained source into the active replay array table."""
    arrays[tensor] = timed(
        "retained_reuse",
        partial(source_execution.retained_value, tensor),
    )


__all__ = ["load_retained_source", "load_source", "read_block"]
