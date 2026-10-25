"""Symbolic legality and precision of a declared coupled sum boundary."""


def closed_sum_mode(before, after, barriers):
    """Return narrow/wide, or None when a closed trigger leaves a dependency.

    Callers activate boundaries only for a single-precision computation. These
    set relations do not price, schedule or enumerate physical operations.
    """
    closed = tuple(dependent for trigger, dependent in barriers
                   if before.intersection(trigger) and not after.intersection(trigger))
    if any(after.intersection(dependent) for dependent in closed):
        return None
    return bool(closed)
