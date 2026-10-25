"""Shared MPI file queue copied from legacy lattice/dispatch.py.

It assigns configuration names to workers; output completion and provenance
remain the caller's responsibility. Use a unique suffix for each run.
"""

from io import FileIO
import os
import random
from time import sleep
from typing import Iterable, Iterator

if os.name == "nt":
    import msvcrt

    def lock(f: FileIO):
        while True:
            try:
                msvcrt.locking(f.fileno(), msvcrt.LK_NBRLCK, 1)
                break
            except OSError:
                sleep(0.1)

    def unlock(f: FileIO):
        msvcrt.locking(f.fileno(), msvcrt.LK_UNLCK, 1)

elif os.name == "posix":
    import fcntl

    def lock(f: FileIO):
        if f.writable():
            fcntl.lockf(f, fcntl.LOCK_EX)

    def unlock(f: FileIO):
        if f.writable():
            fcntl.lockf(f, fcntl.LOCK_UN)


class AtomicOpen:
    def __init__(self, path, *args, **kwargs):
        self.file = open(path, *args, **kwargs)
        try:
            self.begin = self.file.tell()
            self.lock_offset = 0
            self.file.seek(self.lock_offset)
            lock(self.file)
            self.file.seek(self.begin)
        except BaseException:
            try:
                self.file.close()
            except BaseException:
                pass
            raise

    def __enter__(self, *args, **kwargs):
        return self.file

    def __exit__(self, exc_type=None, exc_value=None, traceback=None):
        cleanup_error = None
        cleanup_traceback = None
        try:
            self.file.flush()
            os.fsync(self.file.fileno())
        except BaseException as error:
            cleanup_error = error
            cleanup_traceback = error.__traceback__
        try:
            self.file.seek(self.lock_offset)
        except BaseException as error:
            if cleanup_error is None:
                cleanup_error = error
                cleanup_traceback = error.__traceback__
        try:
            unlock(self.file)
        except BaseException as error:
            if cleanup_error is None:
                cleanup_error = error
                cleanup_traceback = error.__traceback__
        try:
            self.file.close()
        except BaseException as error:
            if cleanup_error is None:
                cleanup_error = error
                cleanup_traceback = error.__traceback__

        # Keep an exception from the with-body as the primary failure, while
        # still attempting every cleanup step above.
        if exc_type is not None:
            return False
        if cleanup_error is not None:
            raise cleanup_error.with_traceback(cleanup_traceback)
        return False


def rand(suffix: str = None) -> str:
    ret = str(suffix)
    if suffix is None:
        ret = hex(random.randint(0, 0xFFFFFFFF))[2:].upper()
    return ret


class Dispatch:
    def __init__(self, filename: str, suffix: str = None, cfg_list: Iterable = None, *, comm=None) -> None:
        # if cfg_list is not None:
        if comm is None:
            raise ValueError("pass an MPI communicator or use lqcdweave.adapters.mpi_dispatch.Dispatch")
        self.comm = comm
        self.rank = self.comm.Get_rank()
        suffix = self.comm.bcast(rand(suffix) if self.rank == 0 else None, root=0)
        self.tmp = f"{filename}.{suffix}.tmp"
        error = None
        if self.rank == 0:
            try:
                # Lock a separate initialization file before creating the queue;
                # independent worker groups must not observe a half-written queue.
                with AtomicOpen(self.tmp + ".lock", "a+"):
                    if not os.path.exists(self.tmp):
                        with open(filename, encoding="utf-8") as source:
                            lines = [line.strip() for line in source if line.strip() and not line.lstrip().startswith("#")]
                        with open(self.tmp, "x", encoding="utf-8") as output:
                            output.write("".join(line + "\n" for line in lines))
            except Exception as exc:
                error = str(exc)
        error = self.comm.bcast(error, root=0)
        if error:
            raise RuntimeError(f"Dispatch initialization failed: {error}")

    def __iter__(self) -> Iterator[str]:
        comm = self.comm
        rank = self.rank
        while True:
            if rank == 0:
                with AtomicOpen(self.tmp, "r+") as f:
                    lines = f.readlines()
                    f.seek(0)
                    if lines == []:
                        line = None
                    else:
                        line = lines.pop(0).strip()
                        f.writelines(lines)
                        f.truncate()
            else:
                line = "dummy"
            line = comm.bcast(line, root=0)
            if line == "dummy":
                raise ValueError(f"MPI broadcast error at rank {rank}!")
            if line is None:
                print("Dispatch: finish all!")
                break
            elif line == "" or line.startswith("#"):
                continue
            else:
                yield line

    def clear(self):
        self.comm.Barrier()
        if self.rank == 0:
            os.remove(self.tmp)
            if os.path.exists(self.tmp + ".lock"):
                os.remove(self.tmp + ".lock")


def combine(output: str, line: str):
    with AtomicOpen(output, "a") as f:
        f.write(f"{line}\n")
