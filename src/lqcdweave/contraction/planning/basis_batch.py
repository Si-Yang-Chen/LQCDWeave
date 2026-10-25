"""Combine mode/point-prefix scene problems into one shared-I/O problem."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
import json
from types import MappingProxyType

from lqcdweave.contracts.correlator.api import ContractionTerm, TensorSpec
from .state import ContractionProblem, SourceSpec, _nbytes


@dataclass(frozen=True)
class SceneBasisBatch:
    """One global problem and the origin of each namespaced source request."""

    problem: ContractionProblem
    used_bases: tuple[tuple[int, int], ...]
    source_origins: Mapping[str, tuple[int, str]]
    term_ranges: tuple[tuple[int, int], ...]
    representative_by_region: Mapping[str, str]
    maxima_by_region: Mapping[str, tuple[int, int]]
    spec_by_region: Mapping[str, TensorSpec | None]

    def __post_init__(self):
        object.__setattr__(self, "source_origins", MappingProxyType(dict(self.source_origins)))
        object.__setattr__(self, "representative_by_region", MappingProxyType(dict(self.representative_by_region)))
        object.__setattr__(self, "maxima_by_region", MappingProxyType(dict(self.maxima_by_region)))
        object.__setattr__(self, "spec_by_region", MappingProxyType(dict(self.spec_by_region)))


def combine_basis_problems(
    variants: Sequence[tuple[int, int, ContractionProblem]], *, backing_identity: str
) -> SceneBasisBatch:
    """Combine all Terms while reading each declared backing region at most once.

    Equal original ``SourceSpec.region`` values are shared only under the
    caller's explicit ``backing_identity``. That identity must bind the same
    configuration, source data, *ordered* eigenmode basis and point basis.
    Each smaller Ne/Np must be an exact prefix of the corresponding largest
    basis. No identity is inferred from shape, counts or a filename.
    """
    if not isinstance(backing_identity, str) or not backing_identity:
        raise ValueError("backing_identity must identify verified ordered mode and point bases")
    if not variants:
        raise ValueError("at least one basis problem is required")

    specs = {}
    sources: dict[str, SourceSpec] = {}
    terms = []
    bridges = []
    joins = []
    origins = {}
    term_ranges = []
    used_bases = []
    backend = None
    factorized = False

    for problem_index, (used_ne, used_np, problem) in enumerate(variants):
        if any(isinstance(value, bool) or not isinstance(value, int) or value < 1 for value in (used_ne, used_np)):
            raise ValueError("used_ne and used_np must be positive integers")
        if not isinstance(problem, ContractionProblem):
            raise TypeError("each variant needs a ContractionProblem")
        if backend is None:
            backend = problem.backend
        if problem.backend != backend:
            raise ValueError("all basis problems must use the same backend")
        used_bases.append((used_ne, used_np))
        factorized = factorized or problem.factorized_vertices
        key_map = {key: f"basisbatch:{problem_index}:{key}" for key in problem.sources}
        offset = len(terms)
        for key, source in problem.sources.items():
            if isinstance(source.metadata, Mapping) and "used_np" in source.metadata:
                if source.metadata["used_np"] != used_np:
                    raise ValueError(f"source {key!r} declares a different used_np")
            if isinstance(source.metadata, Mapping) and "used_ne" in source.metadata:
                if source.metadata["used_ne"] != used_ne:
                    raise ValueError(f"source {key!r} declares a different used_ne")
            combined_key = key_map[key]
            region = json.dumps((backing_identity, source.region), separators=(",", ":"))
            sources[combined_key] = replace(source, region=region)
            origins[combined_key] = (problem_index, key)
        specs.update({key_map[key]: spec for key, spec in problem.specs.items()})
        terms.extend(
            replace(term, inputs=tuple(key_map[key] for key in term.inputs))
            for term in problem.terms
        )
        bridges.extend(
            replace(
                bridge,
                left=key_map[bridge.left],
                right=key_map[bridge.right],
                record=key_map[bridge.record],
                scene=None if bridge.scene is None else offset + bridge.scene,
            )
            for bridge in problem.sparse_bridges
        )
        joins.extend(
            replace(
                join,
                scene=offset + join.scene,
                spin=key_map[join.spin],
                elemental=key_map[join.elemental],
            )
            for join in problem.baryon_joins
        )
        term_ranges.append((offset, len(terms)))

    by_region: dict[str, list[str]] = defaultdict(list)
    for key, source in sources.items():
        by_region[source.region].append(key)
    representative_by_region = {}
    maxima_by_region = {}
    spec_by_region = {}
    for region, keys in by_region.items():
        maxima = (
            max(used_bases[origins[key][0]][0] for key in keys),
            max(used_bases[origins[key][0]][1] for key in keys),
        )
        maxima_by_region[region] = maxima
        representative = max(
            keys,
            key=lambda key: (
                sources[key].region_bytes,
                key,
            ),
        )
        representative_by_region[region] = representative
        if representative in specs:
            reference_spec = specs[representative]
            for key in keys:
                if key not in specs:
                    raise ValueError("one backing region cannot mix dense and sparse requests")
                spec = specs[key]
                if (
                    spec.dtype != reference_spec.dtype
                    or len(spec.shape) != len(reference_spec.shape)
                ):
                    raise ValueError("one backing region needs matching dense dtype and axis order")
            maximum_shape = tuple(max(specs[key].shape[axis] for key in keys) for axis in range(len(reference_spec.shape)))
            spec_by_region[region] = TensorSpec(maximum_shape, reference_spec.dtype)
        elif any(key in specs for key in keys):
            raise ValueError("one backing region cannot mix dense and sparse requests")
        else:
            spec_by_region[region] = None

        maximum_dense_bytes = 0 if spec_by_region[region] is None else _nbytes(spec_by_region[region])
        region_bytes = max(maximum_dense_bytes, *(sources[key].region_bytes for key in keys))
        host_bytes = max(sources[key].host_region_bytes for key in keys)
        if host_bytes:
            host_bytes = max(host_bytes, maximum_dense_bytes)
        staging_bytes = max(sources[key].staging_bytes for key in keys)
        transfer_bytes = max(sources[key].transfer_bytes for key in keys)
        if transfer_bytes:
            transfer_bytes = max(transfer_bytes, maximum_dense_bytes)
        read_bytes = max(
            region_bytes,
            max(
                sources[key].region_bytes if sources[key].read_bytes is None else sources[key].read_bytes
                for key in keys
            ),
        )
        for key in keys:
            sources[key] = replace(
                sources[key],
                region_bytes=region_bytes,
                host_region_bytes=host_bytes,
                staging_bytes=staging_bytes,
                transfer_bytes=transfer_bytes,
                read_bytes=read_bytes,
            )

    combined = ContractionProblem(
        specs,
        tuple(terms),
        sources,
        sparse_bridges=tuple(bridges),
        baryon_joins=tuple(joins),
        backend=backend,
        single_read_regions=True,
        factorized_vertices=factorized,
    )
    return SceneBasisBatch(
        combined,
        tuple(used_bases),
        origins,
        tuple(term_ranges),
        representative_by_region,
        maxima_by_region,
        spec_by_region,
    )


__all__ = ["SceneBasisBatch", "combine_basis_problems"]
