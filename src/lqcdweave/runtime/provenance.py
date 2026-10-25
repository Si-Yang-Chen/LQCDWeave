"""Small reproducible runtime snapshot, separate from scientific definitions."""

import hashlib
from pathlib import Path
import platform

import numpy as np


def local_provenance():
    """Hash current package Python sources, including uncommitted changes."""
    root = Path(__file__).resolve().parents[1]
    digest = hashlib.sha256()
    for path in sorted(root.rglob("*.py")):
        name = path.relative_to(root).as_posix().encode("utf-8")
        data = path.read_bytes()
        digest.update(len(name).to_bytes(8, "big") + name)
        digest.update(len(data).to_bytes(8, "big") + data)
    return dict(code_sha256=digest.hexdigest(), python=platform.python_version(),
                numpy=np.__version__, backend="numpy-cpu")

