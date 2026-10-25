"""Atomic I/O helpers for contraction-plan archives."""

from contextlib import contextmanager, suppress
import os
from pathlib import Path
import tempfile


@contextmanager
def _atomic_archive_file(path):
    """Publish only a completely written archive; preserve an existing target on error."""
    destination = Path(path)
    parent = destination.parent
    if not parent.exists() or not parent.is_dir():
        raise FileNotFoundError(f"plan archive parent directory does not exist: {parent}")
    temporary_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            prefix=f".{destination.name}.",
            suffix=".tmp",
            dir=parent,
            delete=False,
        ) as handle:
            temporary_name = handle.name
            yield handle
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, destination)
        temporary_name = None
        try:
            directory_fd = os.open(parent, os.O_RDONLY)
        except OSError:
            directory_fd = None
        if directory_fd is not None:
            try:
                os.fsync(directory_fd)
            except OSError:
                pass
            finally:
                os.close(directory_fd)
    finally:
        if temporary_name is not None:
            with suppress(FileNotFoundError):
                os.unlink(temporary_name)
