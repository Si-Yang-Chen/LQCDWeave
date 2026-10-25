#!/usr/bin/env python3
"""Prepare and produce the C24P29 partner point-source campaign.

This is an explicit campaign driver, not a general production frontend. It
keeps the full physical-time PSV coverage and the Gamma5-reduced PSP triangle
in the dataset contracts, while each invocation may calculate only a bounded
set of still-missing source times.
"""
from __future__ import annotations

import argparse
import ctypes
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile
import time
from typing import Any

import numpy as np

from lqcdweave.adapters.spatial_points import SpatialPointHandle
from lqcdweave.contracts.propagator import PropagatorEndpoint, PropagatorBlock
from lqcdweave.contracts.spatial_sampling import SpatialSamplingDesign
from lqcdweave.contracts.time_selection import TimeSelectionRecord
from lqcdweave.planning.spatial_points import sampled_spatial_points_spec
from lqcdweave.production.perambulator.api import (
    PerambulatorGenerator, PointBasis, VectorBasis, PropagatorReader, PropagatorWriter,
    propagator_pair_spec, reverse_propagator,
)
from lqcdweave.sampling.spatial.api import sample_spatial_points
from lqcdweave.storage.dataset import (
    DatasetReader, DatasetWriter, inspect_dataset, read_dataset_metadata,
)
from lqcdweave.storage.metadata import write_json_document
from lqcdweave.contracts.dataset import canonical_json, stable_digest


CAMPAIGN_ID = "c24p29-partner-point-p64-18cfg-v1"
SPIN_BASIS = "QUDA_DEGRAND_ROSSI_GAMMA_BASIS"
GAMMA5 = np.diag([1.0, 1.0, -1.0, -1.0]).astype(np.complex128)
MIN_FREE_GPU_BYTES = int(1.5 * 1024**3)


def _cfg_text(value: Any) -> str:
    if isinstance(value, bool) or not isinstance(value, (str, int, np.integer)):
        raise ValueError("configuration IDs must be strings or integers")
    text = str(value)
    if not text or any(char in text for char in "\r\n/\\"):
        raise ValueError("invalid configuration ID")
    return text


def validate_manifest(document: dict, *, campaign_profile: bool = True) -> dict:
    """Validate the explicit inventory manifest without probing large inputs."""
    if not isinstance(document, dict) or document.get("schema") != 1:
        raise ValueError("expected a schema-1 campaign manifest")
    required = {
        "ensemble", "lattice", "cfgs", "root", "gauge_pattern",
        "legacy_vsv_root", "basis_entries", "basis_ne", "sampling", "dirac", "storage",
    }
    missing = required - document.keys()
    if missing:
        raise ValueError(f"campaign manifest is missing fields: {sorted(missing)}")
    if not isinstance(document["ensemble"], str) or not document["ensemble"]:
        raise ValueError("manifest ensemble is required")
    lattice = document["lattice"]
    if (not isinstance(lattice, list) or len(lattice) != 4
            or any(type(value) is not int or value < 1 for value in lattice)):
        raise ValueError("manifest lattice must be [Lx,Ly,Lz,Lt]")
    cfgs = tuple(_cfg_text(value) for value in document["cfgs"])
    if not cfgs or len(set(cfgs)) != len(cfgs):
        raise ValueError("manifest cfgs must be a nonempty ordered unique list")
    for name in ("root", "gauge_pattern", "legacy_vsv_root"):
        if not isinstance(document[name], str) or not document[name]:
            raise ValueError(f"manifest {name} must be a nonempty path/string")
    if isinstance(document["basis_ne"], bool) or type(document["basis_ne"]) is not int or document["basis_ne"] < 1:
        raise ValueError("basis_ne must be a positive integer")
    sampling = document["sampling"]
    dirac = document["dirac"]
    storage = document["storage"]
    for name, value in (("sampling", sampling), ("dirac", dirac), ("storage", storage),
                        ("basis_entries", document["basis_entries"])):
        if not isinstance(value, dict):
            raise ValueError(f"manifest {name} must be an object")
    for name in ("num_points", "temporal_policy", "seed_base"):
        if name not in sampling:
            raise ValueError(f"manifest sampling is missing {name}")
    for name in ("mass", "tol", "maxiter", "xi_0", "nu", "clover_coeff_t",
                 "clover_coeff_r", "t_boundary", "multigrid", "stout"):
        if name not in dirac:
            raise ValueError(f"manifest dirac is missing {name}")
    if not isinstance(dirac["stout"], dict):
        raise ValueError("manifest dirac.stout must be an object")
    if storage.get("PSP_time_triangle") != "sink>=source" or storage.get("dtype") != "complex128":
        raise ValueError("campaign requires complex128 PSP coverage sink>=source")
    if campaign_profile:
        expected = {
            "ensemble": "C24P29", "lattice": [24, 24, 24, 72], "basis_ne": 100,
        }
        for key, value in expected.items():
            if document[key] != value:
                raise ValueError(f"campaign {key} must be {value!r}")
        if len(cfgs) != 18:
            raise ValueError("the production manifest must contain all 18 configurations")
        expected_sampling = {"num_points": 64, "temporal_policy": 4, "seed_base": 20261011}
        if any(sampling.get(key) != value for key, value in expected_sampling.items()):
            raise ValueError("campaign sampling must be P64, temporal policy 4, seed base 20261011")
        expected_dirac = {
            "mass": -0.277, "tol": 1e-7, "maxiter": 5000, "xi_0": 1.0,
            "nu": 1.0, "clover_coeff_t": 1.160920226,
            "clover_coeff_r": 1.160920226, "t_boundary": -1, "multigrid": None,
        }
        for key, value in expected_dirac.items():
            if dirac.get(key) != value:
                raise ValueError(f"campaign dirac.{key} must be {value!r}")
        stout = dirac["stout"]
        if stout != {"nstep": 1, "rho": 0.125, "dir_ignore": 4}:
            raise ValueError("campaign gauge transform must be stout(1,0.125,4)")
        if set(map(str, document["basis_entries"])) != set(cfgs):
            raise ValueError("basis_entries must cover exactly the 18 campaign configurations")
        for cfg in cfgs:
            entries = document["basis_entries"].get(cfg)
            if entries is None:
                entries = document["basis_entries"].get(next(k for k in document["basis_entries"] if str(k) == cfg))
            if not isinstance(entries, list) or len(entries) != 72:
                raise ValueError(f"basis_entries[{cfg}] must contain all 72 physical times")
            times = [entry.get("time") for entry in entries if isinstance(entry, dict)]
            if sorted(times) != list(range(72)):
                raise ValueError(f"basis_entries[{cfg}] must contain each time 0..71 once")
    return document


def load_manifest(path: str | os.PathLike, *, campaign_profile: bool = True) -> dict:
    with Path(path).open("r", encoding="utf-8") as stream:
        document = json.load(stream)
    validate_manifest(document, campaign_profile=campaign_profile)
    return document


def _cfg_seed(base: int, cfg: str) -> int:
    try:
        return int(base) + int(cfg)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"configuration {cfg!r} must be numeric to derive its sampling seed") from exc


def _manifest_digest(manifest: dict) -> str:
    return stable_digest(manifest)


def _manifest_matches_except_solver_tolerance(existing: dict, requested: dict) -> bool:
    old = json.loads(canonical_json(existing))
    new = json.loads(canonical_json(requested))
    try:
        old["dirac"]["tol"] = new["dirac"]["tol"]
    except (KeyError, TypeError):
        return False
    return canonical_json(old) == canonical_json(new)


def _point_dir(manifest: dict) -> Path:
    return Path(manifest["root"]) / "inputs" / "points"


def _point_provenance(manifest: dict, cfg: str, seed: int) -> dict:
    return {
        "campaign": CAMPAIGN_ID,
        "producer": "lqcdweave.sampling.spatial.points.sample_spatial_points",
        "manifest_id": _manifest_digest(manifest),
        "configuration": cfg,
        "seed": seed,
        "temporal_policy": manifest["sampling"]["temporal_policy"],
        "algorithm": "numpy.PCG64/partial-fisher-yates-v1",
    }


def _point_provenance_matches_sampling(existing: dict, requested: dict) -> bool:
    if not isinstance(existing, dict) or not isinstance(requested, dict):
        return False
    if set(existing) != set(requested) or "manifest_id" not in existing:
        return False
    old = {key: value for key, value in existing.items() if key != "manifest_id"}
    new = {key: value for key, value in requested.items() if key != "manifest_id"}
    return canonical_json(old) == canonical_json(new)


def _publish_points(manifest: dict, cfg: str, *, campaign_profile: bool = True) -> dict:
    lattice = tuple(manifest["lattice"])
    design = SpatialSamplingDesign(
        lattice, manifest["sampling"]["num_points"],
        manifest["sampling"]["temporal_policy"],
    )
    seed = _cfg_seed(manifest["sampling"]["seed_base"], cfg)
    point_dir = _point_dir(manifest)
    configuration = {"ensemble": manifest["ensemble"], "configuration": cfg,
                     "lattice": list(lattice)}
    spec = sampled_spatial_points_spec(design, configuration=configuration, seed=seed)
    provenance = _point_provenance(manifest, cfg, seed)
    state = inspect_dataset(point_dir, cfg=cfg)
    if state["state"] in ("partial", "complete"):
        if state["spec"].document != spec.document:
            raise ValueError(f"existing point dataset for cfg={cfg} differs from the manifest")
        if state["state"] == "complete":
            point_handle = SpatialPointHandle.from_dataset(point_dir, cfg=cfg)
            if canonical_json(state["provenance"]) != canonical_json(provenance):
                if not _point_provenance_matches_sampling(state["provenance"], provenance):
                    raise ValueError(f"existing point dataset for cfg={cfg} differs from the manifest")
                expected_points = sample_spatial_points(design, seed=seed)
                if not np.array_equal(point_handle.points.coordinates,
                                      expected_points.coordinates):
                    raise ValueError(
                        f"completed point dataset for cfg={cfg} does not match the requested seed")
            return _write_point_audit(manifest, cfg, point_handle, point_dir)
        if canonical_json(state["provenance"]) != canonical_json(provenance):
            raise ValueError(f"partial point dataset for cfg={cfg} differs from the manifest")
    points = sample_spatial_points(design, seed=seed)
    with DatasetWriter(point_dir, spec, provenance=provenance,
                       resume=(state["state"] == "partial")) as writer:
        for index in writer.missing:
            writer.write(index, points.coordinates[index])
        if not writer.missing:
            writer.publish()
    point_handle = SpatialPointHandle.from_dataset(point_dir, cfg=cfg)
    return _write_point_audit(manifest, cfg, point_handle, point_dir)


def _write_point_audit(manifest: dict, cfg: str, points: SpatialPointHandle, point_dir: Path) -> dict:
    coordinates = np.asarray(points.points.coordinates, dtype="<i8")
    audit = {
        "schema": 1,
        "campaign": CAMPAIGN_ID,
        "ensemble": manifest["ensemble"],
        "configuration": cfg,
        "data_id": points.data_id,
        "path": str(point_dir / f"{cfg}.npy"),
        "coordinate_sha256": hashlib.sha256(coordinates.tobytes(order="C")).hexdigest(),
        "lattice": list(points.points.lattice_size),
        "times": list(points.points.times),
        "point_count": points.Np,
        "temporal_policy": points.points.design.temporal_policy,
        "seed": points.points.seed,
        "algorithm": points.points.algorithm,
        "pool_ids": list(points.points.pool_ids or ()),
    }
    write_json_document(point_dir / f"{cfg}.json", audit)
    return audit


def prepare_campaign(manifest: dict, *, campaign_profile: bool = True) -> list[dict]:
    """Register deterministic full-time point datasets and campaign inputs."""
    validate_manifest(manifest, campaign_profile=campaign_profile)
    root = Path(manifest["root"])
    root.mkdir(parents=True, exist_ok=True)
    inputs = root / "inputs"
    inputs.mkdir(parents=True, exist_ok=True)
    manifest_path = root / "manifest.json"
    if manifest_path.exists():
        existing = load_manifest(manifest_path, campaign_profile=False)
        if canonical_json(existing) != canonical_json(manifest):
            if not _manifest_matches_except_solver_tolerance(existing, manifest):
                raise ValueError("campaign root already contains a different manifest")
            write_json_document(manifest_path, manifest)
    else:
        write_json_document(manifest_path, manifest)
    cfg_list_path = root / "cfgs.txt"
    cfg_text = "".join(f"{_cfg_text(cfg)}\n" for cfg in manifest["cfgs"])
    if cfg_list_path.exists():
        if cfg_list_path.read_text(encoding="utf-8") != cfg_text:
            raise ValueError("campaign cfgs.txt differs from the manifest order")
    else:
        cfg_list_path.write_text(cfg_text, encoding="utf-8", newline="\n")
    return [_publish_points(manifest, _cfg_text(cfg), campaign_profile=campaign_profile)
            for cfg in manifest["cfgs"]]


def _basis_entries(manifest: dict, cfg: str) -> list[dict]:
    for key, entries in manifest["basis_entries"].items():
        if str(key) == cfg:
            return sorted(entries, key=lambda item: item["time"])
    raise KeyError(f"no manifest basis_entries for cfg={cfg}")


def gauge_identity(manifest: dict, cfg: str) -> str:
    pattern = manifest["gauge_pattern"].format(cfg=cfg, configuration=cfg)
    if os.path.isabs(pattern):
        path = pattern
    else:
        path = os.path.join(manifest.get("rootremote") or "", pattern)
    return "qio:" + stable_digest({
        "ensemble": manifest["ensemble"], "configuration": cfg,
        "path": os.path.normpath(path), "source": "manifest gauge_pattern",
    })


def basis_identity(manifest: dict, cfg: str) -> str:
    return "legacy-vsv:" + stable_digest({
        "campaign": CAMPAIGN_ID,
        "ensemble": manifest["ensemble"],
        "configuration": cfg,
        "basis_ne": manifest["basis_ne"],
        "entries": _basis_entries(manifest, cfg),
        "payload_order": "mode,z,y,x,color:C-order",
    })


class LegacyBasisHandle:
    """Read legacy per-time pickle payloads as raw memmaps, never unpickling."""
    def __init__(self, manifest: dict, cfg: str):
        self.cfg = cfg
        self.entries = {int(item["time"]): dict(item) for item in _basis_entries(manifest, cfg)}
        self.lattice = tuple(manifest["lattice"])
        self.Ne = int(manifest["basis_ne"])
        self.time_selection = TimeSelectionRecord(self.lattice[3], tuple(range(self.lattice[3])))
        self.basis_id = basis_identity(manifest, cfg)
        self.endpoint = PropagatorEndpoint("V", self.basis_id, range(self.Ne))
        self._mapped_time = None
        self._mapped = None

    def load(self, key):
        if _cfg_text(key) != self.cfg:
            raise ValueError("legacy basis configuration differs from the requested key")
        raise NotImplementedError("legacy basis is streamed by physical time")

    def _close_map(self):
        if self._mapped is not None:
            self._mapped._mmap.close()
            self._mapped = None
            self._mapped_time = None

    def read_modes(self, key, time, start, stop):
        if _cfg_text(key) != self.cfg:
            raise ValueError("legacy basis configuration differs from the requested key")
        if (type(time) is not int or time not in self.entries
                or type(start) is not int or type(stop) is not int
                or not 0 <= start < stop <= self.Ne):
            raise ValueError("invalid legacy basis time or mode block")
        entry = self.entries[time]
        expected_shape = [self.Ne, self.lattice[2], self.lattice[1], self.lattice[0], 3]
        if entry.get("shape") != expected_shape or np.dtype(entry.get("dtype")) != np.dtype("<c16"):
            raise ValueError(f"legacy basis cfg={self.cfg} time={time} has an unexpected payload layout")
        path = Path(entry["path"])
        stat = path.stat()
        if entry.get("file_bytes") is not None and stat.st_size != entry["file_bytes"]:
            raise ValueError(f"legacy basis file size changed after inventory: {path}")
        if entry.get("mtime_ns") is not None and stat.st_mtime_ns != entry["mtime_ns"]:
            raise ValueError(f"legacy basis mtime changed after inventory: {path}")
        offset = entry.get("offset")
        if type(offset) is not int or offset < 0:
            raise ValueError("legacy basis inventory must declare a nonnegative raw payload offset")
        payload_bytes = int(np.prod(expected_shape)) * np.dtype("<c16").itemsize
        if offset + payload_bytes > stat.st_size:
            raise ValueError(f"legacy basis payload is truncated: {path}")
        if self._mapped_time != time:
            self._close_map()
            self._mapped = np.memmap(path, dtype="<c16", mode="r", offset=offset,
                                     shape=tuple(expected_shape), order="C")
            self._mapped_time = time
        return self._mapped[start:stop]

    def close(self):
        self._close_map()


class _GaugeHandle:
    def __init__(self, manifest: dict):
        self.manifest = manifest
        self.gauge_id = None

    def load(self, key):
        cfg = _cfg_text(key)
        pattern = self.manifest["gauge_pattern"].format(cfg=cfg, configuration=cfg)
        path = (pattern if os.path.isabs(pattern) else
                os.path.join(self.manifest.get("rootremote") or "", pattern))
        self.gauge_id = gauge_identity(self.manifest, cfg)
        from pyquda_utils import io

        return io.readQIOGauge(path)


def _dirac_definition(manifest: dict) -> dict:
    dirac = manifest["dirac"]
    stout = dict(dirac["stout"])
    return {
        "action": "anisotropic_clover",
        "mass": dirac["mass"],
        "xi_0": dirac["xi_0"],
        "nu": dirac["nu"],
        "clover_coeff_t": dirac["clover_coeff_t"],
        "clover_coeff_r": dirac["clover_coeff_r"],
        "t_boundary": dirac["t_boundary"],
        "gauge_transform": {"kind": "stout", **stout},
        "gamma_basis": SPIN_BASIS,
        "gamma5_hermitian": True,
    }


def make_pair_specs(manifest: dict, cfg: str, point_handle: SpatialPointHandle,
                    basis_handle: LegacyBasisHandle, *, point_count: int | None = None,
                    psv_pairs: list[tuple[int, int]] | None = None,
                    psp_pairs: list[tuple[int, int]] | None = None):
    lt = manifest["lattice"][3]
    npoints = point_handle.Np if point_count is None else int(point_count)
    point_endpoint = PropagatorEndpoint("P", point_handle.data_id, range(npoints))
    vector_endpoint = PropagatorEndpoint("V", basis_handle.basis_id,
                                         range(basis_handle.Ne))
    if psv_pairs is None:
        psv_pairs = [(point_time, mode_time)
                     for point_time in range(lt) for mode_time in range(lt)]
    if psp_pairs is None:
        psp_pairs = [(sink_time, source_time)
                     for source_time in range(lt) for sink_time in range(source_time, lt)]
    configuration = {"ensemble": manifest["ensemble"], "configuration": cfg,
                     "lattice": list(manifest["lattice"])}
    dirac = _dirac_definition(manifest)
    psv = propagator_pair_spec(
        "PSV", configuration=configuration, sink=point_endpoint,
        source=vector_endpoint, time_pairs=psv_pairs,
        gauge_id=gauge_identity(manifest, cfg),
        dirac=dirac, gamma5=GAMMA5, spin_basis=SPIN_BASIS,
        gamma5_hermitian=True, dtype="complex128",
    )
    psp = propagator_pair_spec(
        "PSP", configuration=configuration, sink=point_endpoint,
        source=point_endpoint, time_pairs=psp_pairs,
        gauge_id=gauge_identity(manifest, cfg),
        dirac=dirac, gamma5=GAMMA5, spin_basis=SPIN_BASIS,
        gamma5_hermitian=True, dtype="complex128",
    )
    return point_endpoint, vector_endpoint, psv, psp


def _output_provenance(manifest: dict, cfg: str, point_handle: SpatialPointHandle,
                       basis_handle: LegacyBasisHandle, product: str) -> dict:
    gauge_id = gauge_identity(manifest, cfg)
    return {
        "campaign": CAMPAIGN_ID,
        "product": product,
        "gauge_id": gauge_id,
        "basis_id": basis_handle.basis_id,
        "point_data_id": point_handle.data_id,
        "dirac": _dirac_definition(manifest),
        "solver": {
            "tol": manifest["dirac"]["tol"],
            "maxiter": manifest["dirac"]["maxiter"],
            "multigrid": manifest["dirac"]["multigrid"],
        },
        "sampling": {
            "num_points": manifest["sampling"]["num_points"],
            "temporal_policy": manifest["sampling"]["temporal_policy"],
            "seed": _cfg_seed(manifest["sampling"]["seed_base"], cfg),
        },
        "storage": {
            "dtype": "complex128",
            "coverage": "full-physical-time",
            "PSP_time_triangle": "sink>=source" if product == "PSP" else None,
        },
    }


class _PairOutput:
    def __init__(self, directory: Path, spec, provenance: dict, *, cfg: str,
                 array_module=np):
        self.directory, self.spec, self.provenance, self.cfg = directory, spec, provenance, cfg
        state = inspect_dataset(directory, cfg=cfg)
        self.initial_state = state["state"]
        if state["state"] in ("partial", "complete"):
            if state["spec"].document != spec.document:
                raise ValueError(f"existing {spec.document['definition']['product']} dataset spec differs")
            if canonical_json(state["provenance"]) != canonical_json(provenance):
                raise ValueError(f"existing {spec.document['definition']['product']} provenance differs")
            completed = set(state["completed"])
        else:
            completed = set()
        pairs = tuple(tuple(pair) for pair in spec.coordinates("time_pair"))
        self.missing_pairs = {pair for index, pair in enumerate(pairs) if index not in completed}
        self._resume = state["state"] == "partial"
        self.writer = None
        self.facade = None
        self.array_module = array_module

    def ensure_writer(self, *, force: bool = False):
        if self.writer is None and (self.missing_pairs or (force and self._resume)):
            self.writer = DatasetWriter(
                self.directory, self.spec, provenance=self.provenance,
                resume=self._resume, format="npy",
            )
            self.facade = PropagatorWriter(self.writer, array_module=self.array_module)
        return self.facade

    def write(self, block, canonical_pair):
        if canonical_pair not in self.missing_pairs:
            return False
        facade = self.ensure_writer()
        if facade is None:
            raise RuntimeError("missing pair has no writable dataset")
        facade.write(block)
        self.missing_pairs.remove(canonical_pair)
        return True

    def finalize(self):
        if self.writer is None and self.initial_state == "partial" and not self.missing_pairs:
            self.ensure_writer(force=True)
        if self.writer is None:
            return self.initial_state == "complete"
        try:
            if self.missing_pairs:
                return False
            self.writer.publish()
            self.writer.close()
            self.initial_state = "complete"
            return True
        finally:
            if self.writer is not None and not self.writer._closed:
                self.writer.close()

    def close(self):
        if self.writer is not None and not self.writer._closed:
            self.writer.close()

    def source_missing(self, source_time: int, product: str, lt: int) -> bool:
        if product == "PSV":
            return any((source_time, sink_time) in self.missing_pairs
                       for sink_time in range(lt))
        return any((sink_time, source_time) in self.missing_pairs
                   for sink_time in range(source_time, lt))


def _make_generator(manifest: dict, cfg: str, backend, basis_handle,
                    gauge_handle, source_times, sink_times,
                    *, sink_cache_limit_bytes: int | None = None):
    dirac = manifest["dirac"]
    lattice = tuple(manifest["lattice"])
    source_selection = TimeSelectionRecord(lattice[3], tuple(source_times))
    sink_selection = TimeSelectionRecord(lattice[3], tuple(sink_times))
    if sink_cache_limit_bytes is None:
        sink_cache_limit_bytes = (lattice[3] * basis_handle.Ne * lattice[0] * lattice[1]
                                  * lattice[2] * 3 * np.dtype("complex128").itemsize)
    generator = PerambulatorGenerator(
        lattice, gauge_handle,
        mass=dirac["mass"], tol=dirac["tol"], maxiter=dirac["maxiter"],
        xi_0=dirac["xi_0"], nu=dirac["nu"],
        clover_coeff_t=dirac["clover_coeff_t"], clover_coeff_r=dirac["clover_coeff_r"],
        t_boundary=dirac["t_boundary"], multigrid=dirac["multigrid"],
        contract_prec="complex128",
        MRHS=False, backend=backend, stream_eigenvectors=True,
        sink_ne_block_size=32, sink_time_block_size=1,
        sink_cache_limit_bytes=sink_cache_limit_bytes,
        source_selection=source_selection, sink_selection=sink_selection,
        rhs_batch_size=1,
    )
    generator.load(cfg)
    stout = dirac["stout"]
    generator.stout_smear(stout["nstep"], stout["rho"], stout["dir_ignore"])
    return generator


def _to_numpy(array, array_module):
    if array_module is np:
        return np.asarray(array)
    return array_module.asnumpy(array)


def _psp_gamma5_reverse(block: np.ndarray, gamma5: np.ndarray = GAMMA5) -> np.ndarray:
    values = np.asarray(block)
    if values.ndim != 6 or values.shape[:2] != (4, 4) or values.shape[2] != values.shape[4] or values.shape[3] != 3 or values.shape[5] != 3:
        raise ValueError("PSP block must have axes spin_sink,spin_source,P,3,P,3")
    point_color_count = values.shape[2] * 3
    swapped = values.transpose(1, 0, 4, 5, 2, 3).conj().reshape(4, 4, point_color_count, point_color_count)
    reversed_values = np.einsum("ai,ijxy,jb->abxy", gamma5, swapped, gamma5, optimize=True)
    return reversed_values.reshape(values.shape)


def _required_rows(pair_output: _PairOutput, source_time: int, product: str,
                   lt: int) -> bool:
    return pair_output.source_missing(source_time, product, lt)


def _append_json_line(path: Path, record: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as stream:
        stream.write(canonical_json(record) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def produce_cfg(manifest: dict, cfg: str, *, backend,
                gauge_handle=None, basis_handle=None, point_handle=None,
                source_times=None, max_source_times: int | None = None,
                sink_times=None, point_count: int | None = None,
                output_root: str | os.PathLike | None = None,
                memory_guard: bool = False, write_solve_log: bool = False,
                campaign_profile: bool = True) -> dict:
    """Calculate only requested missing sources; publish only complete specs.

    Injected handles/backend keep this routine CPU-testable. The production
    CLI supplies the manifest-backed read-only gauge/basis and QudaBackend.
    """
    total_start = time.perf_counter()
    validate_manifest(manifest, campaign_profile=campaign_profile)
    cfg = _cfg_text(cfg)
    cfgs = {_cfg_text(value) for value in manifest["cfgs"]}
    if cfg not in cfgs:
        raise ValueError(f"cfg={cfg} is absent from the manifest")
    lt = int(manifest["lattice"][3])
    root = Path(manifest["root"] if output_root is None else output_root)
    if point_handle is None:
        point_handle = SpatialPointHandle.from_dataset(root / "inputs" / "points", cfg=cfg)
    if basis_handle is None:
        basis_handle = LegacyBasisHandle(manifest, cfg)
    if gauge_handle is None:
        gauge_handle = _GaugeHandle(manifest)
    point_count = point_handle.Np if point_count is None else int(point_count)
    if not 1 <= point_count <= point_handle.Np:
        raise ValueError("point_count must select a nonempty prefix of the prepared points")
    psv_endpoint, vector_endpoint, psv_spec, psp_spec = make_pair_specs(
        manifest, cfg, point_handle, basis_handle, point_count=point_count,
    )
    psv_provenance = _output_provenance(manifest, cfg, point_handle, basis_handle, "PSV")
    psp_provenance = _output_provenance(manifest, cfg, point_handle, basis_handle, "PSP")
    psv_output = _PairOutput(root / "data" / "PSV", psv_spec,
                             psv_provenance, cfg=cfg, array_module=backend.array_module)
    psp_output = _PairOutput(root / "data" / "PSP", psp_spec,
                             psp_provenance, cfg=cfg, array_module=backend.array_module)
    requested = tuple(range(lt)) if source_times is None else tuple(int(t) for t in source_times)
    if (not requested or len(set(requested)) != len(requested)
            or any(not 0 <= time < lt for time in requested)):
        raise ValueError("source_times must be nonempty unique physical times in [0,Lt)")
    active = [time for time in requested if (
        _required_rows(psv_output, time, "PSV", lt)
        or _required_rows(psp_output, time, "PSP", lt))]
    if max_source_times is not None:
        if isinstance(max_source_times, bool) or not isinstance(max_source_times, int) or max_source_times < 1:
            raise ValueError("max_source_times must be a positive integer")
        active = active[:max_source_times]
    sink_times = tuple(range(lt)) if sink_times is None else tuple(int(t) for t in sink_times)
    if (not sink_times or len(set(sink_times)) != len(sink_times)
            or any(not 0 <= time < lt for time in sink_times)):
        raise ValueError("sink_times must be nonempty unique physical times in [0,Lt)")
    if not active:
        psv_complete = psv_output.finalize()
        psp_complete = psp_output.finalize()
        complete = psv_complete and psp_complete
        return {"cfg": cfg, "state": "complete" if complete else "partial",
                "source_times": [], "remaining_psv_pairs": len(psv_output.missing_pairs),
                "remaining_psp_pairs": len(psp_output.missing_pairs), "solves": 0,
                "total_runtime_seconds": time.perf_counter() - total_start}
    generator = None
    solve_start = len(getattr(backend, "solve_records", ()))
    processed = []
    memory_record = None
    try:
        generator = _make_generator(
            manifest, cfg, backend, basis_handle, gauge_handle,
            active, sink_times,
        )
        point_basis = PointBasis(point_handle, count=point_count)
        sink_bases = [VectorBasis(basis_handle, count=basis_handle.Ne), point_basis]
        for index, source_time in enumerate(active):
            source_start = time.perf_counter()
            record_start = len(getattr(backend, "solve_records", ()))
            vsp, psp = generator.calc(source_time, src=point_basis, sink=sink_bases)
            if memory_guard and memory_record is None and hasattr(backend, "memory_snapshot"):
                memory_record = backend.memory_snapshot()
                expected_cache = (lt * basis_handle.Ne * np.prod(manifest["lattice"][:3])
                                  * 3 * np.dtype("complex128").itemsize)
                if generator.sink_cache_used_bytes != expected_cache:
                    raise RuntimeError(
                        f"full-time sink eigenvector cache incomplete: "
                        f"{generator.sink_cache_used_bytes} != {expected_cache} bytes")
                if memory_record["free_bytes"] < MIN_FREE_GPU_BYTES:
                    raise RuntimeError(
                        f"only {memory_record['free_bytes']} free GPU bytes remain; "
                        f"at least {MIN_FREE_GPU_BYTES} are required")
            for sink_time in vsp.sink_times:
                block = vsp.at(sink_time)
                psv_output.write(block, (source_time, block.sink_time))
            for sink_time in psp.sink_times:
                block = psp.at(sink_time)
                canonical_pair = (block.sink_time, source_time)
                if canonical_pair in psp_output.missing_pairs:
                    psp_output.write(block, canonical_pair)
            del vsp, psp, block
            processed.append(source_time)
            records = list(getattr(backend, "solve_records", ()))[record_start:]
            if write_solve_log:
                _append_json_line(root / "logs" / f"solve-summary-{cfg}.jsonl", {
                    "campaign": CAMPAIGN_ID, "cfg": cfg, "source_time": source_time,
                    "rhs_count": len(records),
                    "max_true_residual": max((item["true_residual"] for item in records), default=None),
                    "max_iterations": max((item["iterations"] for item in records), default=None),
                    "sink_cache_used_bytes": generator.sink_cache_used_bytes,
                    "gpu_memory": memory_record,
                    "elapsed_seconds": time.perf_counter() - source_start,
                })
        psv_complete = psv_output.finalize()
        psp_complete = psp_output.finalize()
        complete = psv_complete and psp_complete
        return {
            "cfg": cfg,
            "state": "complete" if complete else "partial",
            "source_times": processed,
            "remaining_psv_pairs": len(psv_output.missing_pairs),
            "remaining_psp_pairs": len(psp_output.missing_pairs),
            "solves": len(getattr(backend, "solve_records", ())) - solve_start,
            "sink_cache_used_bytes": generator.sink_cache_used_bytes,
            "gpu_memory": memory_record,
            "total_runtime_seconds": time.perf_counter() - total_start,
        }
    finally:
        psv_output.close()
        psp_output.close()
        if hasattr(basis_handle, "close"):
            basis_handle.close()


def select_job_openmpi():
    """Select the cluster's HPC-X OpenMPI extension before QUDA/MPI imports."""
    import mpi4py

    native_path = "/opt/hpc/software/mpi/hpcx/v2.11.0/gcc-7.3.1/lib/libmpi.so.40"
    native = ctypes.CDLL(native_path, mode=ctypes.RTLD_GLOBAL)
    candidates = list(Path(mpi4py.__file__).parent.glob("MPI.openmpi*.so"))
    if len(candidates) != 1 or "mpi4py.MPI" in sys.modules:
        raise RuntimeError("cannot select an unambiguous, previously unloaded OpenMPI extension")
    spec = importlib.util.spec_from_file_location("mpi4py.MPI", candidates[0])
    module = importlib.util.module_from_spec(spec)
    sys.modules["mpi4py.MPI"] = module
    spec.loader.exec_module(module)
    mpi4py.MPI = module
    if not module.Is_initialized():
        module.Init()
    initialized = ctypes.c_int(0)
    native.MPI_Initialized.argtypes = [ctypes.POINTER(ctypes.c_int)]
    native.MPI_Initialized.restype = ctypes.c_int
    status = native.MPI_Initialized(ctypes.byref(initialized))
    if status != 0 or initialized.value != 1:
        raise RuntimeError("QUDA-linked OpenMPI remains uninitialized")
    return module, native_path, native


class CheckedQudaBackend:
    """QudaBackend wrapper that verifies the loaded gamma basis and each solve."""
    def __init__(self, *, tolerance: float):
        from lqcdweave.adapters.quda import QudaBackend

        if not np.isfinite(tolerance) or tolerance <= 0:
            raise ValueError("QUDA solve tolerance must be finite and positive")
        max_true_residual = 10.0 * float(tolerance)

        class _Checked(QudaBackend):
            def __init__(self):
                super().__init__()
                self.solve_records = []
                self.gamma_basis_verified = False

            def load(self, gauge, parameters):
                super().load(gauge, parameters)
                from pyquda.enum_quda import QudaGammaBasis
                expected = QudaGammaBasis.QUDA_DEGRAND_ROSSI_GAMMA_BASIS
                actual = self.dirac.invert_param.gamma_basis
                if actual != expected:
                    self.destroy()
                    raise RuntimeError(
                        f"QUDA gamma basis is {actual!r}, expected DeGrand-Rossi {expected!r}")
                self.gamma_basis_verified = True

            def solve_into(self, rhs, solution, *, tol, maxiter):
                if len(rhs) != 1:
                    raise RuntimeError("campaign requires one QUDA RHS per solve")
                super().solve_into(rhs, solution, tol=tol, maxiter=maxiter)
                params = self.dirac.invert_param
                true_residual = _first_active_true_residual(
                    params.true_res, maximum=max_true_residual)
                iterations = int(params.iter)
                if iterations < 0 or iterations >= maxiter:
                    raise RuntimeError(f"QUDA did not converge within maxiter={maxiter}: {iterations}")
                self.solve_records.append({
                    "true_residual": true_residual, "iterations": iterations,
                    "maximum_accepted_true_residual": max_true_residual,
                })

            def memory_snapshot(self):
                cp = self.array_module
                free_bytes, total_bytes = cp.cuda.runtime.memGetInfo()
                pool = cp.get_default_memory_pool()
                return {
                    "free_bytes": int(free_bytes), "total_bytes": int(total_bytes),
                    "pool_used_bytes": int(pool.used_bytes()),
                    "pool_total_bytes": int(pool.total_bytes()),
                }

        self._backend = _Checked()

    def __getattr__(self, name):
        return getattr(self._backend, name)


def _first_active_true_residual(values, *, maximum: float) -> float:
    """Validate the active single-RHS residual, ignoring QUDA padding slots."""
    try:
        residuals = np.asarray(values, dtype=np.float64).reshape(-1)
    except (TypeError, ValueError) as exc:
        raise RuntimeError("QUDA true residual is not a numeric array") from exc
    if residuals.size == 0:
        raise RuntimeError("QUDA returned an empty true residual array")
    active = float(residuals[0])
    if not np.isfinite(active) or active > maximum:
        raise RuntimeError(
            f"QUDA active true residual failed validation: {active!r} > {maximum!r}")
    return active


def _initialize_quda(manifest: dict, resource_path: Path):
    mpi, native_path, native = select_job_openmpi()
    if mpi.COMM_WORLD.Get_size() != 1:
        raise RuntimeError("this campaign is configured for exactly one QUDA MPI rank")
    import pyquda
    from pyquda_utils import core
    import cupy as cp

    resource_path.mkdir(parents=True, exist_ok=True)
    core.init([1, 1, 1, 1], list(manifest["lattice"]), resource_path=str(resource_path))
    if pyquda.getMPISize() != 1:
        raise RuntimeError("PyQUDA reports more than one MPI process")
    runtime = {
        "pyquda_file": str(Path(pyquda.__file__).resolve()),
        "core_file": str(Path(core.__file__).resolve()),
        "cupy_version": cp.__version__,
        "device_count": int(cp.cuda.runtime.getDeviceCount()),
        "mpi_library": mpi.Get_library_version().strip(),
        "mpi_module": str(Path(mpi.__file__).resolve()),
        "quda_linked_mpi": native_path,
    }
    return mpi, native, runtime


def _run_worker(manifest: dict, *, cfg: str | None, queue: str | None,
                suffix: str | None, source_times, max_source_times: int | None,
                max_cfgs: int = 1):
    root = Path(manifest["root"])
    if isinstance(max_cfgs, bool) or not isinstance(max_cfgs, int) or max_cfgs < 1:
        raise ValueError("max_cfgs must be a positive integer")
    import signal

    def _terminate(signum, _frame):
        raise SystemExit(f"received signal {signum}; closing partial dataset writers")

    signal.signal(signal.SIGTERM, _terminate)
    mpi, native, runtime = _initialize_quda(manifest, root / "runtime" / "quda")
    backend = CheckedQudaBackend(tolerance=manifest["dirac"]["tol"])
    completed = []
    try:
        if cfg is not None:
            result = produce_cfg(manifest, cfg, backend=backend,
                                 source_times=source_times,
                                 max_source_times=max_source_times,
                                 memory_guard=True, write_solve_log=True)
            completed.append(result)
        else:
            from lqcdweave.adapters.mpi_dispatch import Dispatch
            from lqcdweave.runtime.dispatch import combine

            dispatch = Dispatch(queue, suffix=suffix)
            for current_cfg in dispatch:
                result = produce_cfg(manifest, current_cfg, backend=backend,
                                     source_times=source_times,
                                     max_source_times=max_source_times,
                                     memory_guard=True, write_solve_log=True)
                completed.append(result)
                if result["state"] == "complete":
                    combine(str(root / f"completed-{suffix}.txt"), current_cfg)
                if len(completed) >= max_cfgs:
                    break
        print(json.dumps({"runtime": runtime, "results": completed}, sort_keys=True))
    finally:
        backend.destroy()
        # mpi4py finalizes at interpreter exit, after PyQUDA's later-registered
        # endQuda hook has completed its MPI collectives.


def _run_smoke(manifest: dict, cfg: str) -> dict:
    smoke_start = time.perf_counter()
    cfg = _cfg_text(cfg)
    cfgs = {_cfg_text(value) for value in manifest["cfgs"]}
    if cfg not in cfgs:
        raise ValueError(f"cfg={cfg} is absent from the manifest")
    point_handle = SpatialPointHandle.from_dataset(_point_dir(manifest), cfg=cfg)
    basis_handle = LegacyBasisHandle(manifest, cfg)
    gauge_handle = _GaugeHandle(manifest)
    lt = manifest["lattice"][3]
    source_times = (0, 4)
    sink_times = tuple(range(lt))
    point_endpoint, vector_endpoint, psv_spec, _ = make_pair_specs(
        manifest, cfg, point_handle, basis_handle, point_count=1,
        psv_pairs=[(source_time, sink_time)
                    for source_time in source_times for sink_time in sink_times],
    )
    psv_provenance = _output_provenance(manifest, cfg, point_handle, basis_handle,
                                        "PSV-smoke")
    temporary_output = tempfile.TemporaryDirectory(prefix="partner-point-smoke-")
    psv_dir = Path(temporary_output.name) / "PSV"
    mpi, native, runtime = _initialize_quda(manifest, Path(tempfile.gettempdir()) / "partner-point-quda-smoke")
    backend = CheckedQudaBackend(tolerance=manifest["dirac"]["tol"])
    cache_bytes = (lt * basis_handle.Ne * np.prod(manifest["lattice"][:3]) * 3
                   * np.dtype("complex128").itemsize)
    generator = None
    cross_04 = cross_40 = None
    expected_readback = {}
    sample_pairs = {(0, 0), (4, lt - 1)}
    before = backend.memory_snapshot()
    source_time_runtimes = {}
    try:
        generator = _make_generator(
            manifest, cfg, backend, basis_handle, gauge_handle,
            source_times, sink_times,
            sink_cache_limit_bytes=int(cache_bytes),
        )
        point_basis = PointBasis(point_handle, count=1)
        sink_bases = [VectorBasis(basis_handle, count=basis_handle.Ne), point_basis]
        with DatasetWriter(psv_dir, psv_spec, provenance=psv_provenance) as writer:
            psv_writer = PropagatorWriter(writer, array_module=backend.array_module)
            for source_time in source_times:
                source_start = time.perf_counter()
                vsp, psp = generator.calc(source_time, src=point_basis, sink=sink_bases)
                if source_time == 0:
                    cross_04 = _to_numpy(
                        psp.values[generator.sink_selection.slot(4)], backend.array_module).copy()
                else:
                    cross_40 = _to_numpy(
                        psp.values[generator.sink_selection.slot(0)], backend.array_module).copy()
                for sink_time in vsp.sink_times:
                    block = vsp.at(sink_time)
                    canonical_pair = (source_time, block.sink_time)
                    if canonical_pair in sample_pairs:
                        expected = reverse_propagator(block, GAMMA5,
                                                      gamma5_hermitian=True,
                                                      array_module=backend.array_module)
                        expected_readback[canonical_pair] = _to_numpy(
                            expected.values, backend.array_module).copy()
                    psv_writer.write(block)
                del vsp, psp, block
                source_time_runtimes[str(source_time)] = time.perf_counter() - source_start
            if generator.sink_cache_used_bytes != cache_bytes:
                raise RuntimeError(
                    f"smoke filled {generator.sink_cache_used_bytes} of {cache_bytes} cache bytes")
            if not backend.gamma_basis_verified:
                raise RuntimeError("QUDA DeGrand-Rossi gamma basis was not verified")
            if len(backend.solve_records) != 24:
                raise RuntimeError(f"smoke expected exactly 24 RHS solves, got {len(backend.solve_records)}")
            if cross_04 is None or cross_40 is None:
                raise RuntimeError("smoke did not retain the two physical cross-time PSP blocks")
            reversed_04 = _psp_gamma5_reverse(cross_04)
            denominator = np.linalg.norm(cross_40)
            gamma5_relative_error = float(np.linalg.norm(cross_40 - reversed_04)
                                           / (denominator if denominator else 1.0))
            solver_tolerance = float(manifest["dirac"]["tol"])
            gamma5_rtol = max(1e-8, 100.0 * solver_tolerance)
            gamma5_atol = max(1e-10, solver_tolerance)
            if not np.allclose(cross_40, reversed_04,
                               rtol=gamma5_rtol, atol=gamma5_atol):
                raise RuntimeError(f"PSP Gamma5 time-reversal check failed: {gamma5_relative_error}")
            writer.publish()
        with DatasetReader(psv_dir, cfg=cfg) as reader:
            propagator_reader = PropagatorReader([reader])
            readback_errors = {}
            for pair, expected in expected_readback.items():
                block = propagator_reader.read("PSV", pair[0], pair[1],
                                               sink=point_endpoint, source=vector_endpoint)
                actual = np.asarray(block.values)
                readback_errors[f"{pair[0]},{pair[1]}"] = float(
                    np.linalg.norm(actual - expected) / max(np.linalg.norm(expected), 1.0))
                if not np.array_equal(actual, expected):
                    raise RuntimeError(f"VSP-to-PSV dataset readback changed pair {pair}")
        after = backend.memory_snapshot()
        residuals = [item["true_residual"] for item in backend.solve_records]
        report = {
            "campaign": CAMPAIGN_ID, "cfg": cfg, "status": "passed",
            "source_times": list(source_times), "sink_times": ["all", lt],
            "point_prefix": 1, "basis_modes": basis_handle.Ne,
            "rhs_solves": len(backend.solve_records),
            "gamma_basis_verified": backend.gamma_basis_verified,
            "gamma5_psp_relative_error": gamma5_relative_error,
            "gamma5_psp_rtol": gamma5_rtol,
            "gamma5_psp_atol": gamma5_atol,
            "psv_readback_relative_errors": readback_errors,
            "true_residual_max": max(residuals),
            "iterations_max": max(item["iterations"] for item in backend.solve_records),
            "sink_cache_bytes": generator.sink_cache_used_bytes,
            "gpu_memory_before": before, "gpu_memory_after": after,
            "runtime": runtime,
            "source_time_runtime_seconds": source_time_runtimes,
            "smoke_runtime_seconds": time.perf_counter() - smoke_start,
        }
        return report
    finally:
        basis_handle.close()
        backend.destroy()
        # Leave MPI finalization to mpi4py so PyQUDA's endQuda atexit hook can
        # finish its collectives first.
        temporary_output.cleanup()


def _parse_times(value: str | None):
    if value is None:
        return None
    try:
        result = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    except ValueError as exc:
        raise argparse.ArgumentTypeError("times must be comma-separated integers") from exc
    if not result:
        raise argparse.ArgumentTypeError("times must not be empty")
    return result


def _build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test", action="store_true",
                        help="run the bounded pure-CPU numerical and resume checks")
    sub = parser.add_subparsers(dest="command")
    prepare = sub.add_parser("prepare", help="sample/register all campaign point sets")
    prepare.add_argument("manifest")
    worker = sub.add_parser("worker", help="produce complete or resumable per-cfg PSV/PSP data")
    worker.add_argument("manifest")
    worker.add_argument("--queue")
    worker.add_argument("--suffix")
    worker.add_argument("--cfg")
    worker.add_argument("--source-times", type=_parse_times)
    worker.add_argument("--max-source-times", type=int)
    worker.add_argument("--max-cfgs", type=int, default=1,
                        help="maximum queue configs consumed by this worker (default: one)")
    smoke = sub.add_parser("smoke", help="run the bounded real QUDA 24-RHS smoke check")
    smoke.add_argument("manifest")
    smoke.add_argument("--cfg", required=True)
    return parser


def main(argv=None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    if args.self_test:
        from tests.campaigns.test_partner_point_production import run_cpu_self_test
        run_cpu_self_test()
        print(json.dumps({"self_test": "passed"}, sort_keys=True))
        return 0
    if args.command is None:
        parser.error("choose prepare, worker, smoke, or --self-test")
    manifest = load_manifest(args.manifest)
    if args.command == "prepare":
        records = prepare_campaign(manifest)
        print(json.dumps({"prepared": len(records), "root": manifest["root"]}, sort_keys=True))
        return 0
    if args.command == "worker":
        if (args.cfg is None) == (args.queue is None):
            parser.error("worker requires exactly one of --cfg or --queue")
        if args.queue is not None and not args.suffix:
            parser.error("worker --queue requires --suffix")
        if args.cfg is not None and args.suffix is not None:
            parser.error("--suffix is only used with --queue")
        _run_worker(manifest, cfg=args.cfg, queue=args.queue, suffix=args.suffix,
                    source_times=args.source_times,
                    max_source_times=args.max_source_times,
                    max_cfgs=args.max_cfgs)
        return 0
    if args.command == "smoke":
        report = _run_smoke(manifest, args.cfg)
        print(json.dumps(report, sort_keys=True))
        return 0
    parser.error("unknown command")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
