# SS: spatial sampling

## Ownership and public boundary

The unique implementation is `src/lqcdweave/sampling/spatial/`.
`api.py` is its public symbol list; the package `__init__.py` re-exports that
list so the former import `lqcdweave.sampling.spatial` remains valid.
The sampler depends only on NumPy and the single shared data contract in
`contracts.spatial_sampling`. It does not import or call EV, EL, or PR, does
not read or write files, and never chooses a temporal subset automatically. An explicit selection retains only its ordered rows.

`SpatialSamplingDesign`, `SpatialPointSet`, and `SamplingPoolSpec` are imported
from `contracts.spatial_sampling`; SS does not define copies. The
`adapters.spatial_points` module provides `SpatialPointHandle` for production
and existing file-reader adaptation. Spatial-point arrays use the same
canonical `(time,point,xyz)` layout throughout the package.

`sampling.mode_weights.dilution_pair_weights` is a separately importable modal
dilution helper, not a spatial-coordinate probability routine. Elemental
production uses the formula through its local
`production.elemental.mode_weights.dilution_pair_weights` implementation to
keep the production module boundary independent of sampling. SS neither
imports the helper nor applies its weights.

## Exact API and defaults

The package exports these symbols from `sampling.spatial.api`:

```text
SpatialSamplingDesign(lattice_size, num_points, temporal_policy=1)
SpatialPointSet(lattice_size, times, coordinates, design=None, seed=None, algorithm=None)
SamplingPoolSpec(
    group_id, spatial_sites, sampled_points,
    sampling_design="simple_random_without_replacement",
    compensation="expected", observed_counts=None,
)
sample_spatial_points(design, *, seed=None, selection=None)
```

`SpatialSamplingDesign` is the immutable rule: extents use `(Lx,Ly,Lz,Lt)`,
`num_points` is `Np`, and `temporal_policy` defaults to the positive integer 1
and accepts an integer segment length `b` from 1 through `Lt`. It divides
physical times from zero into segments
`[k*b, min((k+1)*b, Lt))`; the final segment may be shorter. All times in one
segment reuse the same ordered point set, and distinct segments are sampled
independently without replacement. Thus `b=1` means independent samples at
each time and `b=Lt` means one shared set for all times. The legacy strings
`"independent"` and `"same"` remain accepted as aliases for 1 and `Lt`,
respectively; when supplied, the original string is retained in design
metadata. Its checks also require positive integer extents and `1 <= Np <= V`,
where `V=Lx*Ly*Lz`.

`SpatialPointSet` is the immutable realization returned by the only sampling
entrypoint, `sample_spatial_points(SpatialSamplingDesign(...), seed=...,
selection=...)`. `coordinates` is a read-only `int64` array with shape
`(len(times),Np,3)` and integer `(x,y,z)` components. Array row `i` belongs to
the physical time `times[i]`; point axis position is the point-basis order and
is preserved verbatim. `num_points` is a read-only property returning the
point-axis extent. Generated sets default to `times == tuple(range(Lt))`; explicit selection preserves its ordered times. Imported
sets may declare a unique ordered subset. `design`, `seed`, and `algorithm`
are provenance fields and may be `None` for imported points. The generated
algorithm string is
`"numpy.PCG64/partial-fisher-yates-v1"`.

`SamplingPoolSpec` is the shared explicit pool metadata contract, not a second
coordinate container. `group_id` is caller supplied; SS does not derive it
from a seed, `Np`, shape, coordinates, time, or offset. The current contract
records `spatial_sites`, `sampled_points`, `sampling_design`, `compensation`,
and optional `observed_counts`. It does not store the coordinate array or a
coordinate dataset URI. Consumers may treat two records as one point pool
only after their data binding has established that they reference the same
ordered coordinates. No inferred equality is provided by SS.

`sample_spatial_points` defaults to all `Lt` times; explicit selection returns its `Nt` ordered rows. The
keyword-only `seed` accepts a nonnegative integer or `None`; `None` requests
entropy from NumPy. Existing coordinates are represented by
`SpatialPointSet(lattice_size=lattice, times=tuple(range(Lt)), coordinates=coords)`;
the constructor preserves the declared time/point order and stores an immutable
`int64` copy. There is no point-layout conversion API. Selection rows follow
`selection.times` order, while segment membership is determined by each
physical time and the `[k*b,min((k+1)*b,Lt))` boundaries; selection row
positions do not redefine the segments. With a fixed seed, each selected row
equals that physical-time row in the full result.

## Sampling algorithm and probabilities

Coordinates flatten with `x` fastest:
`flat = x + Lx*(y + Ly*z)`. For each physical segment, the sampler runs one
partial Fisher–Yates procedure over the implicit range `[0,V)`, then reuses
that ordered sample for every time in the segment. At
step `p`, it draws `j` uniformly from `[0,V-p)`, resolves the sparse swap
table, emits that site, and moves the remaining endpoint into the selected
slot. The procedure samples without replacement, preserves the draw order in
the point axis, and never allocates a permutation of all `V` sites.

For a design with `Np` of `V` sites, every single site in one sampled slice has
inclusion probability `Np/V`. For `k` specified distinct sites in the same
slice, their joint inclusion probability is
`(Np)_k/(V)_k` when `k <= Np`, and zero otherwise. Here `(a)_k` is the falling
factorial. Every time in one segment copies the same ordered realization, so
its cross-time inclusion events are perfectly correlated and the ordering is
identical. Different segments use separate successive RNG draws, making
their inclusion events independent. For events requiring `m_s` distinct
sites from segment `s`, the joint probability across segments is the product
of `(Np)_(m_s)/(V)_(m_s)` over segments, with a factor zero when `m_s > Np`.
An observed coincidence between independently sampled segments does not
establish a shared pool.

SS returns coordinates and provenance; it applies no estimator compensation.
For a single linear spatial sum the first-order probability gives the usual
`V/Np` factor, but products across points or times require the corresponding
joint inclusion probability. No universal multi-point weight is applied.

## Random state, identity, and ownership

Every call creates a local `numpy.random.Generator(numpy.random.PCG64(seed))`.
It does not read or mutate NumPy's legacy global random state. Given the same
geometry, policy, seed, algorithm version, and compatible NumPy implementation,
the current draw sequence is deterministic. `seed=None` is not replayable from
the returned metadata. Even with a seed, seed and shape alone do not prove two
realizations equal. Persist or compare the actual coordinates in their actual
time and point order; retain the design, seed, and algorithm string as
additional provenance. Imported points carry no invented generation history.

The dataclasses are frozen. `SpatialPointSet` copies coordinate values into an
immutable bytes-backed `int64` array; later edits to the source array cannot
change the realization, and callers cannot make its data writable. The pool
contract freezes its metadata and takes a shallow immutable copy of
`observed_counts`.

## Memory and failure behavior

The sampler's sparse swap table has `O(Np)` auxiliary entries per sampled
temporal segment and is discarded before the next segment. It has no `O(V)` point list
or rejection loop. The returned coordinates need `O(Nt*Np)` storage. For
segment length `b`, it creates `ceil(Lt/b)` ordered
samples, including a possibly shorter final segment. Explicit selection still
consumes all physical segments' RNG sequence to preserve each selected row
of a fixed-seed full reference; it does not change the `ceil(Lt/b)` sample
count. It reduces retained output rows, not the number of random draws. Building
the immutable `SpatialPointSet` temporarily retains the mutable output while
copying its bytes, so peak coordinate storage during construction is also
`O(Nt*Np)` with a constant-factor duplicate. Larger segments save RNG work and
reuse each sparse sample, but the API materializes every selected time row.
There is no streaming or chunked output API.

Invalid dimensions, noninteger/bool values, an out-of-range `Np`, a nonpositive
or greater-than-`Lt` integer segment length, unsupported temporal policy, invalid
seed, malformed time mapping, noninteger/out-of-range point
coordinates, duplicate points within a time slice, or a realization that
conflicts with its design (including different ordered coordinates inside a
shared segment) raise `ValueError`. A non-mapping
`SamplingPoolSpec.observed_counts` raises `TypeError`. These paths fail
explicitly; they do not sort, deduplicate, fill missing times, or claim a
sampling identity.

## Origin, platform, and verification

The sampling kernel was copied from the prior
`src/lqcdweave/sampling/spatial.py` implementation at baseline commit
`cc0d354`. Its PCG64 construction, random draw order, partial swap-table
updates, flattening order, and algorithm identifier were preserved. The
coordinate arrays use the canonical time-first layout throughout; SS exposes
no layout conversion API. No EV, EL, PR, correlator, or estimator-weight code
was copied into SS.

The module is CPU NumPy code, available under the project's Python `>=3.10`
and NumPy `>=1.23` requirements. It has no GPU kernel or GPU validation; its
output is a host NumPy coordinate array. It does not invoke a cluster or
production orchestration.

Module-owned tests are in `tests/sampling/spatial/`; package `__init__.py`
files make standard unittest discovery recurse into the directory. Run them
from the repository root with:

```sh
python -m unittest discover -s tests/sampling/spatial -t tests -v
```

The independent ordered-without-replacement reference, same/independent
behavior, full-volume coverage, local RNG isolation, contract identity,
immutable ownership, canonical coordinate layout, and invalid inputs are covered there.
The segment-policy tests compare lengths 1, 2, 3, and 7 with independent
ordered-list references, including a short final segment; check legacy-string
equivalence, shuffled selections against physical segment boundaries, and
invalid segment lengths or conflicting point order within a segment; and
verify full and selected dataset save/restore plus local-source validation.
Production point consumption and weak-field integration tests remain in their
original cross-module `tests/` locations.

On 2026-09-29, all 8 SS-owned CPU tests passed. The targeted existing point-only
Elemental production test also passed. No GPU run was made because this module
has no GPU implementation. The broader architecture-boundary test was run but
currently fails on the concurrent TS migration's import from
`contraction/two_point.py` to `sampling.temporal.reconstruction`; that path is
outside SS. Full CPU integration remains with the root agent.

## 来源与用户脚本

`plan_indexed_spatial_points` 是只读预览：它描述默认完整或显式选择的物理时间范围和预期来源，不抽样、读取坐标或写文件。设计和 seed 仅说明采样意图，不证明实际坐标或已知来源。`sample_spatial_points` 默认生成完整 `Lt` 的坐标，显式 selection 仅保留 Nt 行；调用方脚本负责按任务选择输入、计算、保存和读取方式，具体文件格式与存储后端不由 SS 固定。需要持久化时保存实际有序坐标及来源记录；fresh `require_known` 只接受已实际保存/登记并验证的来源事件，不能因 seed 或设计预告而视为已知。已有坐标没有来源证据时仍是 unknown。

空间点集由脚本显式调用 `sample_spatial_points` 后使用所选保存接口写入；runtime 不内置点集计算与保存的组合生产器。

## Production handle boundary

`adapters.spatial_points.SpatialPointHandle(points, *, cfg=None)` owns an immutable
`int64` `(Nt,Np,3)` snapshot and exposes `Np`, `points` and `load(key)`. Explicit cfg
is checked; the default is deliberately unbound. The point object supplies the ordered physical time mapping, exposed as time_selection. No basis identity is invented.
`from_npy(path, *, lattice_size, cfg=None, time_selection=None)` reuses preset.load; geometry
is explicit, disk axes must be canonical `(time,point,xyz)`, and generation history is unknown. Compact arrays require time_selection; None means canonical full Lt. Mappings close after copying.
`from_dataset(directory, *, cfg=None)` reuses DatasetReader, requires a complete spatial_points
dataset registration with time/point/xyz axes and canonical point/xyz labels, binds its selected cfg
(cfg may be omitted when the registration contains only one cfg),
and retains provenance/source_origin/data_id. Explicit local-sampling
realization fields restore design/seed/algorithm; other provenance stays opaque.
No resampling or source-status upgrade occurs. Both readers produce owned
`int64` `(Nt,Np,3)` snapshots usable after the file reader closes. Handle storage is O(Nt*Np);
this is not a streaming or out-of-core interface. See the [complete usage](spatial_sampling.md).

Explicit selected registrations retain physical lattice Lt in cfg metadata and ordered TimeSelectionRecord metadata; stored time axes are canonical slots. SpatialPointHandle.from_dataset reconstructs points.times without sorting, and read_points(key, physical_time, start=0, stop=None) maps the physical endpoint to a stored slot. Missing physical times raise KeyError. No resampling, new storage, or new recovery layer is introduced. Selected CPU tests compare full-seed reference rows including wrapped order; production/memory/GPU claims remain limited to measured paths. The default whole-array NPY/HDF5 format is described in [data storage](../storage/data_storage.md).
