"""Bounded CPU planning producer for X's already-planned window interface."""
from concurrent.futures import ThreadPoolExecutor
from contextlib import suppress
import os
import subprocess
import sys

from lqcdweave.contraction._cfg_plan_worker import send, WindowReceiver


class PlanningWorkerError(RuntimeError):
    """The private CPU planner failed; the cfg must not be published."""


class _PlanningWindows:
    def __init__(self, args, options):
        self.closed = False
        self.process = self.pool = self.future = None
        self.receiver = WindowReceiver()
        env = dict(os.environ, OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1')
        try:
            self.process = subprocess.Popen(
                [sys.executable, '-m', 'lqcdweave.contraction._cfg_plan_worker'],
                stdin=subprocess.PIPE, stdout=subprocess.PIPE, env=env,
                creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
            send(self.process.stdin, (args, options))
            self.pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix='cfg-plan-receiver')
            self.future = self.pool.submit(self._receive)
        except BaseException:
            self.close()
            raise

    def _receive(self):
        self.process.stdin.write(b'N')
        self.process.stdin.flush()
        return self.receiver.receive(self.process.stdout)

    def __iter__(self):
        return self

    def __next__(self):
        if self.closed:
            raise StopIteration
        try:
            kind, value = self.future.result()
            if kind == 'end':
                self.process.wait(timeout=5)
                self.close()
                raise StopIteration
            if kind != 'window':
                raise PlanningWorkerError(value)
            self.future = self.pool.submit(self._receive)
            return value
        except StopIteration:
            raise
        except BaseException as error:
            self.close()
            if isinstance(error, PlanningWorkerError):
                raise
            raise PlanningWorkerError(f'CPU planning worker failed: {error}') from error

    def next_ready(self):
        """Return None while planning; otherwise consume one window or raise EOF."""
        if self.closed:
            raise StopIteration
        return next(self) if self.future.done() else None

    @property
    def statistics(self):
        """Per-window host delivery times; they overlap the executor's work."""
        return dict(planning_received_windows=self.receiver.received_windows,
                    planning_received_bytes=self.receiver.received_bytes,
                    planning_receive_seconds=self.receiver.receive_seconds,
                    planning_restore_seconds=self.receiver.restore_seconds)

    def close(self):
        if self.closed:
            return
        self.closed = True
        # Terminate only this private CPU worker. This also unblocks a receiver
        # whose computation, send, or exception response has not finished.
        process = self.process
        if process is not None:
            if process.poll() is None:
                process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=5)
        if self.pool is not None:
            self.pool.shutdown(wait=True, cancel_futures=True)
        self.future = None
        self.receiver.clear()
        if process is not None:
            for handle in (process.stdin, process.stdout):
                with suppress(OSError):
                    handle.close()

    def __enter__(self):
        return self

    def __exit__(self, *unused):
        self.close()


def plan_cfg_windows(plan, output, points, requests, preparations, specs, **options):
    """Start one CPU planner with at most one requested future window.

    Arguments match P.iter_cfg_plan_windows. Coordinates and descriptors are
    pure metadata; readers, arrays and backend handles stay in the executor.
    Use as a context manager so cancellation always reaps the private worker.
    The worker starts via its own module, avoiding fork of a CUDA context and
    avoiding re-execution of the caller's frontend script on Windows.
    """
    return _PlanningWindows((plan, output, tuple(points), requests, preparations, specs), options)


__all__ = ['plan_cfg_windows', 'PlanningWorkerError']
