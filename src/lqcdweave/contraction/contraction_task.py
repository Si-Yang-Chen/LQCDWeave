"""Explicit W lifecycle around the existing planning and replay services."""
from collections.abc import Iterator
from dataclasses import replace
from threading import Lock
from types import MappingProxyType


class ContractionTask:
    """Declare once, explicitly plan, then replay without changing the task.

    Arguments are the declarations accepted by ``calculate``, except execution
    and archive controls. Reader handles remain live; no numerical data is
    copied or read by construction. Each run returns a fresh ``Calculation``.
    """

    def __init__(self, correlator, **options):
        unsupported = options.keys() & {'execute', 'plan_archive', 'measure'}
        if unsupported:
            raise TypeError('task construction does not accept ' + ', '.join(sorted(unsupported)))
        self._correlator = correlator
        self._options = dict(options)
        if 'dtype' in self._options:
            from lqcdweave.contracts.correlator.precision import declared_contraction_dtype
            dtype = declared_contraction_dtype(self._options['dtype'])
            self._options['dtype'] = None if dtype is None else dtype.name
        self._selected = None
        self._lock = Lock()
        self._closed = False
        self._stream = None

    def _acquire(self):
        if not self._lock.acquire(blocking=False):
            raise RuntimeError('task is in use; finish or close its result stream first')
        if self._closed:
            self._lock.release()
            raise RuntimeError('task is closed')

    def _require_plan(self):
        if self._selected is None:
            raise RuntimeError('call task.plan() before saving or running the task')
        return self._selected

    def plan(self):
        """Install selected work atomically; repeated calls return this task."""
        self._acquire()
        try:
            if self._selected is None:
                from .correlator_task_plan import plan_task
                self._selected = plan_task(self._correlator, self._options)
            return self
        finally:
            self._lock.release()

    @property
    def plan_report(self):
        """Read-only summary of the installed plan, without executing it."""
        selected = self._require_plan()
        return MappingProxyType(dict(selected.report, prediction=selected.prediction,
                                     planning_seconds=selected.planning_seconds))

    def save_plan(self, path, **options):
        """Save installed work using the existing archive format."""
        self._acquire()
        try:
            from .correlator import save_calculation_plan
            return save_calculation_plan(path, self._require_plan(), **options)
        finally:
            self._lock.release()

    @classmethod
    def load_plan(cls, path, **bindings):
        """Restore saved work and bind current readers, without planning."""
        from .correlator import load_calculation_plan
        task = cls(None)
        task._selected = load_calculation_plan(path, **bindings)
        return task

    def run(self, *, backend=None, measure=False, debug=None, device_memory_limit=None):
        """Execute installed work; coordinate and definition changes need a new task."""
        self._acquire()
        try:
            selected = self._require_plan()
            result = selected.run(backend=backend, measure=measure, debug=debug,
                                  device_memory_limit=device_memory_limit)
            if isinstance(result.value, Iterator):
                stream = _TaskValueStream(result.value, self)
                result = replace(result, value=stream)
                self._stream = stream
                return result
        except BaseException:
            try:
                if self._selected is not None:
                    self._selected.close()
            finally:
                self._lock.release()
            raise
        self._lock.release()
        return result

    def _finish_stream(self, *, failed=False):
        try:
            if failed:
                self._selected.close()
        finally:
            self._stream = None
            self._lock.release()

    def close(self):
        """Close an outstanding stream and release resources; delivered arrays survive."""
        if self._closed:
            return
        if self._stream is not None:
            self._stream.close()
        self._acquire()
        try:
            if self._selected is not None:
                self._selected.close()
            self._closed = True
        finally:
            self._lock.release()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()


class _TaskValueStream:
    """Keep the task occupied even when a cfg iterator has not started yet."""

    def __init__(self, values, task):
        self._values, self._task = values, task
        self._finished = False
        self._cursor_lock = Lock()

    def __iter__(self):
        return self

    def __next__(self):
        if not self._cursor_lock.acquire(blocking=False):
            raise RuntimeError('result stream is in use')
        try:
            return self._next()
        finally:
            self._cursor_lock.release()

    def _next(self):
        if self._finished:
            raise StopIteration
        try:
            return next(self._values)
        except StopIteration:
            self._finish()
            raise
        except BaseException:
            try:
                self._values.close()
            finally:
                self._finish(failed=True)
            raise

    def _finish(self, *, failed=False):
        if not self._finished:
            self._finished = True
            self._task._finish_stream(failed=failed)

    def close(self):
        if not self._cursor_lock.acquire(blocking=False):
            raise RuntimeError('result stream is in use')
        try:
            if not self._finished:
                try:
                    self._values.close()
                finally:
                    self._finish()
        finally:
            self._cursor_lock.release()
