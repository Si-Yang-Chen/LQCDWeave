"""Release one logical tensor and its no-longer-needed source region view."""


def release_tensor(action, arrays, regions, sources, *, timed):
    """Apply one planned release without owning raw block lifecycle."""
    def release():
        del arrays[action.tensor]
        source = sources.get(action.tensor)
        if source is not None and not any(
            key in arrays and item.region == source.region for key, item in sources.items()
        ):
            regions.pop(source.region, None)

    timed("release", release)


__all__ = ["release_tensor"]
