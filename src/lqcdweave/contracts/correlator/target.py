"""Declared execution targets for correlator planning.

This module contains only value records. It does not inspect the current host,
select a backend, or initialize a numerical runtime.
"""

from collections.abc import Mapping
from dataclasses import asdict, dataclass

from ..memory import TargetHardwareMemory


@dataclass(frozen=True, slots=True)
class ExecutionTarget:
    """A declared backend, runtime, memory budget, and optional cost profile."""

    name: str
    backend: str
    runtime: str
    memory: TargetHardwareMemory
    profile: str | None = None
    hardware_fingerprint: str = "unmeasured"
    operations: tuple[str, ...] = ("contract", "baryon_join", "sparse_bridge")
    software: tuple[tuple[str, str], ...] = ()

    def __post_init__(self):
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("execution target name must be a nonempty string")
        compatible = {
            ("numpy", "cpu"),
            ("cupy", "cuda"),
            ("cupy", "hip"),
        }
        if (self.backend, self.runtime) not in compatible:
            raise ValueError(
                "backend/runtime must be numpy/cpu, cupy/cuda, or cupy/hip"
            )
        if not isinstance(self.memory, TargetHardwareMemory):
            raise TypeError("memory must be TargetHardwareMemory")
        if self.profile is not None and (
            not isinstance(self.profile, str) or not self.profile.strip()
        ):
            raise ValueError("profile must be a nonempty string or None")
        if (not isinstance(self.hardware_fingerprint, str)
                or not self.hardware_fingerprint.strip()):
            raise ValueError("hardware_fingerprint must be a nonempty string")

        operations = _normalize_sequence(self.operations, "operations")
        if any(not isinstance(operation, str) or not operation.strip()
               for operation in operations):
            raise ValueError("operations must contain nonempty strings")
        object.__setattr__(self, "operations", operations)

        software = _normalize_sequence(self.software, "software")
        normalized_software = []
        for entry in software:
            if isinstance(entry, (str, bytes)):
                raise TypeError("software entries must be (name, version) pairs")
            try:
                pair = tuple(entry)
            except TypeError as exc:
                raise TypeError(
                    "software entries must be (name, version) pairs"
                ) from exc
            if (len(pair) != 2 or not isinstance(pair[0], str)
                    or not pair[0].strip() or not isinstance(pair[1], str)):
                raise ValueError(
                    "software entries must contain a nonempty name and string version"
                )
            normalized_software.append(pair)
        object.__setattr__(self, "software", tuple(normalized_software))

    def to_dict(self) -> dict:
        """Return a JSON-compatible mapping for this target."""
        return {
            "name": self.name,
            "backend": self.backend,
            "runtime": self.runtime,
            "memory": asdict(self.memory),
            "profile": self.profile,
            "hardware_fingerprint": self.hardware_fingerprint,
            "operations": list(self.operations),
            "software": [list(entry) for entry in self.software],
        }

    @classmethod
    def from_dict(cls, document: Mapping) -> "ExecutionTarget":
        """Build a validated target from a JSON-decoded mapping."""
        if not isinstance(document, Mapping):
            raise TypeError("execution target document must be a mapping")
        allowed = {
            "name", "backend", "runtime", "memory", "profile",
            "hardware_fingerprint", "operations", "software",
        }
        unknown = set(document) - allowed
        if unknown:
            raise ValueError(f"unknown execution target fields: {sorted(unknown)}")
        missing = {"name", "backend", "runtime", "memory"} - set(document)
        if missing:
            raise ValueError(f"missing execution target fields: {sorted(missing)}")

        values = dict(document)
        memory = values["memory"]
        if isinstance(memory, Mapping):
            values["memory"] = TargetHardwareMemory(**memory)
        elif not isinstance(memory, TargetHardwareMemory):
            raise TypeError("memory must be a mapping or TargetHardwareMemory")
        return cls(**values)


def _normalize_sequence(value, field):
    if isinstance(value, (str, bytes, Mapping)):
        raise TypeError(f"{field} must be a sequence")
    try:
        return tuple(value)
    except TypeError as exc:
        raise TypeError(f"{field} must be a sequence") from exc


__all__ = ["ExecutionTarget"]
