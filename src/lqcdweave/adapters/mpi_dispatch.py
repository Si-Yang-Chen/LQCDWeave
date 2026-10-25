"""Bind the shared file queue to mpi4py's current communicator."""

from lqcdweave.runtime.dispatch import Dispatch as _Dispatch, combine


class Dispatch(_Dispatch):
    """Legacy call shape: Dispatch(filename, suffix=None, cfg_list=None)."""

    def __init__(self, filename, suffix=None, cfg_list=None):
        try:
            from mpi4py import MPI
        except ImportError as exc:
            raise ImportError("Dispatch requires mpi4py; install lqcdweave[mpi]") from exc
        super().__init__(filename, suffix, cfg_list, comm=MPI.COMM_WORLD)


__all__ = ["Dispatch", "combine"]
