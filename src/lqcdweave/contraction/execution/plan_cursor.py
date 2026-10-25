"""Run-local cursors over an existing plan or its already-bound call view.

No steps, dependencies, resource assignments or numerical owners are copied.
Readiness and submission belong to the caller's existing execution services.
"""


WAITING = object()


def plan_operations(operations, *, belongs=None, ready=None):
    """Read saved operations in order, yielding WAITING at a blocked head.

    ``belongs`` filters by an already-assigned resource, never chooses one.
    ``ready`` observes current dependencies and resource admission without
    blocking. It must not reserve resources on unsuccessful probes. The caller
    submits an offered operation before advancing this cursor again. Submission
    is not completion; events and owner lifetimes stay with the execution service.

    The ordinary fixed-order replay has no extra readiness protocol: its saved
    calls already enforce waits. Keep its original iterator and per-call cost.
    Cursors are run-local, disposable, and must never be archived or reused as a
    retry after partial execution. The underlying saved view remains immutable.
    """
    if belongs is None and ready is None:
        return iter(operations)
    return _ready_operations(operations, belongs, ready)


def _ready_operations(operations, belongs, ready):
    for operation in operations:
        if belongs is not None and not belongs(operation):
            continue
        while ready is not None and not ready(operation):
            yield WAITING
        yield operation


def drive_plan_cursors(cursors, *, submit, wait):
    """Submit available original operations; wait only when all heads stall.

    ``cursors`` maps fixed resource names to run-local iterators. ``submit``
    atomically admits/enqueues (or synchronously executes) the offered operation;
    ``wait`` waits for existing completion/admission events, or reports an
    impossible blocked state. Neither callback performs planning. Only O(number
    of resources) references are retained here, not an additional operation DAG.

    Return means all operations were submitted, not all devices finished. The
    existing owner scope must finish output waits and drain in-flight work even
    on failure. Close cursors on exit; do not cancel someone else's executors.
    """
    active = dict(cursors)
    try:
        while active:
            progress = False
            exhausted = []
            for resource, cursor in active.items():
                try:
                    operation = next(cursor)
                except StopIteration:
                    exhausted.append(resource)
                    continue
                if operation is not WAITING:
                    submit(resource, operation)
                    progress = True
            for resource in exhausted:
                del active[resource]
            if active and not progress:
                wait()
    finally:
        for cursor in active.values():
            close = getattr(cursor, 'close', None)
            if close is not None:
                close()


__all__ = ['WAITING', 'plan_operations', 'drive_plan_cursors']
