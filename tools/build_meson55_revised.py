"""Rebuild the corrected ordinary G rows and their rest-frame couplings.

Offline dependencies: NumPy and the bundled 本软件 symmetry module. The runtime reads only the
resulting JSON. Historical resources are inputs and are never overwritten.
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
from pathlib import Path

import numpy as np
from lqcdweave.symmetry import irrep_matrices

ROOT = Path(__file__).resolve().parents[1]
PRESETS = ROOT / "tools/presets_raw"
CHANGED = (16, 22, 23, 24)
DIMS = {"A1": 1, "A2": 1, "E": 2, "T1": 3, "T2": 3}


def orbit(vector):
    return sorted({tuple(sign * value for sign, value in zip(signs, perm))
                   for perm in itertools.permutations(vector)
                   for signs in itertools.product((-1, 1), repeat=3)})


def shortest_paths(endpoint):
    steps = tuple(axis + (3 if value < 0 else 0)
                  for axis, value in enumerate(endpoint) for _ in range(abs(value)))
    return sorted(set(itertools.permutations(steps)))


def revised_row(row_id, row_index, endpoints, weights, family, irrep, copy):
    terms = []
    for endpoint, weight in zip(endpoints, weights):
        paths = shortest_paths(endpoint)
        terms.extend({"path": list(path), "coefficient": [float(weight / len(paths)), 0.0]}
                     for path in paths)
    return dict(id=row_id, kind="ordinary", target_l=4, leading_l=4,
                family=family, irrep=irrep, copy=copy, row=row_index,
                endpoint_weights=[dict(endpoint=list(p), weight=float(w))
                                  for p, w in zip(endpoints, weights)], terms=terms)


def revised_catalog(legacy):
    catalog = json.loads(json.dumps(legacy))
    axes, diagonal = orbit((3, 0, 0)), orbit((2, 2, 1))
    endpoints = sorted(axes + diagonal)
    weights = [(4 if p in axes else -1) / np.sqrt(120) for p in endpoints]
    catalog["rows"][16] = revised_row(16, 0, endpoints, weights, "r9_300_221", "A1g+", 0)
    for row_index in range(3):
        weights = []
        for endpoint in diagonal:
            x, y, z = np.roll(endpoint, -row_index)
            weights.append(y * z * (x*x - 2) / np.sqrt(384))
        catalog["rows"][22 + row_index] = revised_row(
            22 + row_index, row_index, diagonal, weights, "n5_221", "T2g+", 1)
    paths = {tuple(t["path"]) for r in catalog["rows"] for t in r["terms"]}
    catalog.update(version="v2", revised_row_ids=list(CHANGED), max_links=5,
                   used_path_count=len(paths),
                   convention="ordinary rows have zero endpoint angular projection below target L on each radial shell; magnetic and electric rows are unchanged and not certified by this scalar endpoint audit")
    return catalog


def group_rows():
    seen, group = set(), []
    for name, matrix in irrep_matrices((0, 0, 0), "T1").items():
        rotation = np.rint(np.asarray(matrix, complex).real).astype(int)
        key = tuple(rotation.flat)
        if round(np.linalg.det(rotation)) == 1 and key not in seen:
            group.append((name, rotation))
            seen.add(key)
    if len(group) != 24:
        raise ValueError("expected 24 proper cubic rotations")
    targets = {rep: tuple(np.asarray(irrep_matrices((0, 0, 0), rep)[name], complex)
                          for name, _ in group) for rep in DIMS}
    return group, targets


def rotate(path, rotation):
    result = []
    for code in path:
        vector = rotation[:, code % 3] * (1 if code < 3 else -1)
        axis = int(np.flatnonzero(vector)[0])
        result.append(axis + (3 if vector[axis] < 0 else 0))
    return tuple(result)


def kernel_representation(rows, group):
    paths = sorted({tuple(t["path"]) for r in rows for t in r["terms"]})
    lookup = {p: i for i, p in enumerate(paths)}
    basis = np.zeros((len(paths), len(rows)), complex)
    for j, row in enumerate(rows):
        for term in row["terms"]:
            basis[lookup[tuple(term["path"])], j] += complex(*term["coefficient"])
    gram = basis.conj().T @ basis
    if not np.allclose(gram, np.eye(len(rows)) * np.trace(gram)/len(rows), atol=1e-12):
        raise ValueError("unequal or nonorthogonal path norms")
    for transform in (lambda p: tuple((c+3) % 6 for c in p),
                      lambda p: tuple((c+3) % 6 for c in reversed(p))):
        action = np.zeros_like(basis)
        for path, i in lookup.items():
            action[lookup[transform(path)]] += basis[i]
        if not np.allclose(action, basis, atol=1e-12):
            raise ValueError("revised ordinary G rows must have positive P and C")
    result = []
    for _, rotation in group:
        action = np.zeros_like(basis)
        for path, i in lookup.items():
            action[lookup[rotate(path, rotation)]] += basis[i]
        matrix = np.linalg.solve(gram, basis.conj().T @ action)
        if not np.allclose(basis @ matrix, action, atol=1e-12):
            raise ValueError("revised rows do not close under cubic rotations")
        result.append(matrix)
    return result


def intertwiner(source, target):
    n, d = source[0].shape[0], target[0].shape[0]
    for i in range(n):
        for j in range(d):
            seed = np.zeros((n, d), complex)
            seed[i, j] = 1
            embedding = sum(a @ seed @ b.conj().T for a, b in zip(source, target))/24
            norm = np.linalg.norm(embedding)
            if norm < 1e-10:
                continue
            embedding *= np.sqrt(d)/norm
            first = embedding.flat[np.flatnonzero(abs(embedding) > 1e-10)[0]]
            embedding *= first.conjugate()/abs(first)
            if not np.allclose(embedding.conj().T @ embedding, np.eye(d), atol=1e-10):
                raise ValueError("non-unitary coupling")
            if any(not np.allclose(a @ embedding, embedding @ b, atol=1e-10)
                   for a, b in zip(source, target)):
                raise ValueError("coupling transformation law failed")
            return embedding.T
    raise ValueError("irrep absent from tensor product")


def revised_bindings(catalog, legacy):
    result = json.loads(json.dumps(legacy))
    group, targets = group_rows()
    kernels = {ids: kernel_representation([catalog["rows"][i] for i in ids], group)
               for ids in ((16,), (22, 23, 24))}
    for recipe in result["multiplets"]:
        ids = tuple(recipe["kernel_ids"])
        if ids not in kernels:
            continue
        spin = targets["A1" if len(recipe["masks"]) == 1 else "T1"]
        source = tuple(np.kron(a, b) for a, b in zip(spin, kernels[ids]))
        coefficients = intertwiner(source, targets[recipe["irrep"]]).reshape(
            DIMS[recipe["irrep"]], len(recipe["masks"]), len(ids))
        row = catalog["rows"][ids[0]]
        recipe["kernel_origin"] = [row[k] for k in ("kind", "target_l", "family", "irrep", "copy")]
        recipe["coefficients_real"] = coefficients.real.tolist()
        recipe["coefficients_imag"] = coefficients.imag.tolist()
    result["version"] = "v2"
    result["reference_scope"] = "legacy reference specifies Reynolds coupling convention; revised rows are constructed by tools/build_meson55_revised.py"
    return result


def build(output_dir):
    old_catalog_bytes = (PRESETS / "meson55.json").read_bytes()
    old_binding_bytes = (PRESETS / "meson55_bindings.json").read_bytes()
    old_catalog_sha = hashlib.sha256(old_catalog_bytes).hexdigest()
    catalog = revised_catalog(json.loads(old_catalog_bytes))
    catalog["legacy_catalog_sha256"] = old_catalog_sha
    catalog_bytes = (json.dumps(catalog, indent=2) + "\n").encode()
    bindings = revised_bindings(catalog, json.loads(old_binding_bytes))
    bindings.update(catalog_sha256=hashlib.sha256(catalog_bytes).hexdigest(),
                    legacy_catalog_sha256=old_catalog_sha,
                    legacy_bindings_sha256=hashlib.sha256(old_binding_bytes).hexdigest())
    binding_bytes = (json.dumps(bindings, indent=2) + "\n").encode()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "meson55_v2.json").write_bytes(catalog_bytes)
    (output_dir / "meson55_bindings_v2.json").write_bytes(binding_bytes)
    return {"catalog_sha256": hashlib.sha256(catalog_bytes).hexdigest(),
            "bindings_sha256": hashlib.sha256(binding_bytes).hexdigest(),
            "used_path_count": catalog["used_path_count"]}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    print(json.dumps(build(parser.parse_args().output_dir), indent=2))
