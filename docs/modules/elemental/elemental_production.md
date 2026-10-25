# EL: elemental production (L2)

## Managed spatial production

Ordinary measurements bind `contraction.correlator.Elemental` and call calculate.
frontend owns metadata admission, coverage reuse, shared stout products and Dataset
publication; EL retains the numerical kernels. No user preparation/producer/write
callback is required. `api.describe_elemental_structures` supplies metadata-only
support proofs. The internal orchestration calls `ElementalGenerator.weight_spatial_block`
only for an unweighted spatial block: the method reuses the existing D/P weighting
services with actual physical-time pools even when its numerical slice has Lt=1.
It rejects already weighted producers and temporal paths. Native binding propagates
calculate's single elemental_weighted option without modifying the source template.
See [scope and CPU evidence](../correlator/managed_elemental.md).

This document describes the implementation in `src/lqcdweave/production/elemental/`.
The coarse module owns meson, baryon, path, spatial point-sector, derivative,
displacement, and raw directed-current spatial tensors. `api.py` is the public
entry point; `__init__.py` only re-exports it. Existing imports from
`lqcdweave.production.elemental` continue to work.

## Confirmed frontend contract

The confirmed public frontend is specified in [elemental_frontend.md](../operators/elemental_frontend.md).
The table frontend and callers are implemented; the signatures below describe retained
numerical engines used by the public calculators and caller-owned scripts. These are explicitly available
from api.py as ElementalKernel/BaryonElementalKernel, while the package facade exposes
the two table generators. The public entry points preserve endpoint selection and streaming. Public initialization takes
fields, eigenvectors, momenta and one coefficient table. Both accept the shared
`momentum_dict=...` with `operator_table=...`; legacy positional/keyword `momenta`
sequences remain supported. Supply exactly one momentum input. Mapping values,
integer XYZ triples or legacy three-integer strings, define storage slots in
insertion order regardless of keys. `contracts.elemental.normalize_momenta` is
the common boundary used by both engines and correlator insertion selection;
it freezes nonempty, distinct integer XYZ triples. The frontend exposes these
as `generator.momenta` for persistence. No phase or numerical kernel changes.
All optional selections are
fixed there; calc_path_blocks(t) and meson calc_derivative(t) take time only. Baryon
uses one shared basis by default. There is no separate derivative_table input or
independent right point selection in the confirmed frontend.

## Table frontend implementation

`frontend.py` normalizes flat component/path/coefficient records or id/terms catalog
rows with `contracts.elemental.elemental_operator_rows`. Arrays load immediately (meson eigenvectors remain borrowed; gauge and points are copied);
lazy handles require lattice_size and load(key). Row labels retain first appearance
order. Spatial V/V path tables use the existing fused projection kernel without a
complete raw-path output stack. Other path sectors use transport followed by weighted
row accumulation; sparse P/P combines the union of COO pairs. Ordered derivatives
reuse the established two-sided recipe and then combine its explicit rows.

The meson constructor write callback consumes final matrices after allocation; it
does not provide output-memory admission. Baryon delegates correlated triples and
its write callback to the existing tiled projection kernel. Shared array bases are
adapted once so all three legs share one owned input snapshot. No format detection,
file saving or job launch occurs in the frontend. Numerical tests cover complex sums,
all meson sectors, ordered derivatives, shared/independent baryon bases and Ne prefixes.
CPU synthetic verification does not establish GPU or physical production acceptance.

### Shared spatial projection

`projection.py` combines the color transports of paths with the same unwrapped
XYZ displacement within each row before applying them to the mode vectors.
Only their color matrices are combined: distinct path coefficients and midpoint
phases retain their original meaning, including paths that wind around the lattice.
The grouped color cache is limited to 512 MiB; wider tables stream groups instead.
Momentum/row batches retain the 256 MiB weighted-field target and existing full
matrix callback order. A field-sized scratch replaces the momentum-by-field
temporary, and the conjugated left basis is prepared once per call.
These are working buffers, not a bound on input/output or total process memory.
No numerical transport cache survives a calculation, load, time change or smear.
NumPy and CuPy both call the same `project_rows` implementation. Color transport,
phase stacks, weighted fields, contraction, output and callbacks use the selected
array module without implicit host transfers. Only shape estimates and the small
momentum/displacement constants use NumPy; scalar coefficients remain on the host
to avoid iterating device scalars and introducing synchronizations. There is no
separate per-path GPU projection implementation or new frontend option.
Tests compare complex grouped paths, winding phases,
cancelled rows, weights, streaming fallback and changed gauge fields against the
independent site/link oracle. `benchmarks/elemental_projection.py` compares a
frozen original generator with the current kernel on identical synthetic inputs.
The shared-backend change has CPU and NumPy protocol coverage only. Its CuPy
test is provided for later device validation; GPU numerical results, allocation
behavior and performance have not been validated for this implementation.

### Explicit V/D bases and nine meson sectors

`ElementalGenerator(..., vector_bases={"V": v_handle, "D": d_handle})` binds
actual inputs independently. The positional eigenvectors argument remains the
V compatibility input. Ne and eigenvector_selection may be kind mappings;
each input retains its own ordered mode labels and physical-time map. The
vector numerical kernels are shared for V and D. Nine bilinear products are
explicitly named v2v/v2d/v2p/d2v/d2d/d2p/p2v/p2d/p2p; only p2p has the existing
sparse color-block representation. Defaults remain v2v. Derivatives and the
three-leg baryon frontend retain their previous scope.

An actual D handle carries contracts.dilution.DilutionSource under
dilution_source. The public generator exposes basis_endpoints and
dilution_endpoints keyed by (sector, bar|field), suitable for ArrayElemental.
Scalar axis labels refer to actual flattened D positions; the source preserves
the corresponding scientific (block,in_block) labels and parent grouping.
Dataset conventions.dilution_endpoints stores referenced D documents, while
definition.logical_sector preserves D even when the storage kind uses an
existing generic elemental schema. Persistent vector/vector planning accepts
product, right_basis, right_basis_id and right_modes; point planning accepts
d2p/p2d. Existing DatasetWriter/Reader owns publication and coverage.

With elemental_weighted=True a single QR D endpoint receives H1=n/m. In a
same-pool, same-group D/D matrix actual equal mixed-column labels receive H1;
different labels receive H2=n(n-1)/(m(m-1)). Different independent groups/pools
factor their first moments. Proven diagonal support uses only H1 and does not
need an unavailable H2. Exact/full-column groups need no correction; legacy
U(1) supports one endpoint only under this estimator. P local compensation
uses the same Boolean, so D/P carries both supported factors. Graph residuals
remove each local H_local_distinct according to actual equality partitions,
without changing physical graph connections or creating another frontend flag.

Persistent D products record the internal dilution_weight_scheme value
component_inclusion-v1. Weighted Dataset products without the marker or with
an obsolete scheme are rejected and require reweighting from raw data. Bare
ArrayElemental weighted declarations refer to the current component semantics.
Old estimator archives are not compatible, even for raw data: replan with actual
D provenance. Missing structural labels still mean general. See [the V/D
reference](../../.agents/skills/lqcdweave-elemental/references/dilution.md).

### Explicit backend routing and host handoff

The table frontend already passes `array_module` through to its numerical engine;
NumPy is the default and explicit CuPy keeps transport, phase, projection and
component arithmetic on device. Registered spatial V/V preparation now exposes
the same `prepare_elemental_stage(..., array_module=np)` option. The helper records
the actual module name in production provenance instead of hardcoding NumPy. It
does not probe or select a device, change the scientific dataset definition, or
introduce another production runner. Callers using the host-only DatasetWriter
must explicitly transfer the final borrowed row, for example with `cp.asnumpy`,
before synchronously writing it. Point/current scripts likewise select their numerical backend explicitly
and transfer device results before host storage.

Consumption of already-produced elementals is a separate X execution concern:
DeviceExecution rebinds PreparedSourceLoader to the selected CuPy scope, uploads
raw blocks, and evaluates profile/dagger/spin/stack preparations there. Constants
are rebuilt for that scope and shared across equal immutable recipes. Existing
fused preparation kernels and CuPy fallbacks are retained; this change does not
replace them or claim a new fused implementation.

`test_elemental_backend.py` checks selected-module calls, complex
table projection, reordered physical output times and explicit dataset handoff
against the independent full-site reference. Its optional CuPy class verifies
actual device output and host serialization. X's
`correlator/execution/test_elemental_backend_routing.py` starts with a populated
NumPy preparation cache, rebinds it to a device scope, and checks independent
complex profile/dagger/spin/stack scalar oracles and constant reuse. Its real
CuPy class is collected separately from the protocol test. No CPU protocol result
establishes CUDA/HIP compilation, transfer performance or production-scale memory.

## Explicit sparse physical-time production

Both public table generators add `selection=None` and `eigenvector_selection=None`;
mesons additionally add `point_selection=None`. Defaults remain full physical time.
`selection` names ordered output anchor times. Raw compact arrays default to that
input mapping; an explicit input selection overrides it for extra endpoint coverage
or a different saved order. Baryon input selections may be one record or a triple.
Domain SpatialPointSet inputs retain their actual `times`. Handles use explicit
`time_selection`; absent metadata means full canonical input, never shape-based
inference. An explicit declaration cannot override contradictory handle metadata.

Producers expose `Lt` (physical), `Nt` (anchor slots), `time_selection` and `times`.
The public `iter_slices()` yields `(physical_t, borrowed_result)` in anchor order
by calling `calc_path_blocks(t)` for each anchor; it streams path results only.
`calc_path_blocks(t)` and `calc_derivative(t)` both take physical time, and
derivatives require explicit `calc_derivative(t)` calls at each requested time.
Streaming `read_modes(key, physical_t, start, stop)` remains a physical-time protocol;
its adapter performs the storage-slot lookup. Eager kernels validate input_Nt against
the input record and map physical time independently per basis/point input.

Cross-time mesons compute physical endpoints from gauge-path displacement before
resolving each input's slot. All path endpoints required by the requested sectors
are checked before numerical output delivery. Missing coverage raises KeyError;
wrong lattice/shape or contradictory mappings raise ValueError. No missing slice
is synthesized, padded, or replaced by an adjacent slot. The existing geometrical
open-boundary zero prescription is unchanged; geometrically excluded paths consume no endpoint fields. Valid paths still
require their actual endpoint coverage. Baryon loading checks every leg covers every selected anchor.

Gauge ownership and physical Lt are unchanged. Sparse inputs remain compact; eager
baryon snapshots retain each unique input_Nt and requested mode prefix, while streaming
staging remains one mode tile per leg. This does not limit total gauge, cache, output,
or solver memory. Numerical transport/bilinear/trilinear kernels are unchanged.
Independent CPU tests in `tests/production/elemental/test_sparse_time.py` cover complete-
time references selected afterward, wrap order, asymmetric input maps, temporal +/-
paths, missing endpoints before callbacks, and physical-time streaming. GPU/MPI and
large production memory remain separate unverified scopes.

## Persistent sparse selections through existing helpers

`prepare_elemental_stage(..., selection=None)`,
`plan_indexed_point_elementals(..., selection=None)`,
`plan_indexed_directed_current(..., selection=None)` and
`plan_indexed_baryon_elementals(..., selection=None)` preserve full-time defaults.
An explicit anchor record produces canonical output coordinates `0..Nt-1` with
`conventions.time_coordinates` carrying the original physical Lt and actual order.
The gauge contract remains full physical Lt. Parent basis/point views retain their
own independent mappings, IDs and ordered mode/point labels.

Independent scripts map the output slot to the saved physical time, call the
public numerical kernel, and pass the resulting array to `DatasetWriter.write`.
Point and basis handles retain each input's own time selection. Publication and
registration remain storage operations. There is no combined basic-product
compute-and-save entry point. Baryon uses the same explicit script composition.

`tests/test_sparse_elemental_dataset.py` publishes actual mapped input datasets,
uses the planners and test-owned production scripts, resumes an interrupted point product, reloads every
slot and compares with independent complete-physical-time site/determinant oracles.
It covers V/P, P/V, sparse P/P, eight-direction current and independently ordered
three-leg baryon bases, and rejects missing physical endpoints. Cross-time products
can be produced/persisted with extra coverage; their use in a pure-slot contraction
requires a separately supported endpoint binding and must not silently use slot +/-1.

## Retained numerical engines


```text
_ElementalEngine(
    latt_size, gauge_field, eigenvector=None, num_nabla=0,
    momentum_list=((0, 0, 0),), dilution=None, is_blending=False,
    usedNe=None, usedNp=None, calc_mode="calc_disp", debug=False,
    paths=None, path_version="legacy", path_boundary="periodic", point=None,
    *, point_right=None, usedNp_right=None, array_module=np,
    ne_block_size=32, phase_convention="midpoint", selection=None,
)
```

`latt_size` is `(Lx,Ly,Lz,Lt)`. `gauge_field.load(key)` supplies gauge links
as `(Lt,Lz,Ly,Lx,4,3,3)`. An optional `eigenvector.load(key)` supplies
`(input_Nt,Ne,Lz,Ly,Lx,3)` with explicit time metadata when sparse. Point inputs are either the neutral
`contracts.spatial_sampling.SpatialPointSet` (actual ordered coordinates
`(input_Nt,Np,3)`, actual times, matching physical geometry) or a legacy handle whose loaded
coordinates are `(Np,input_Nt,3)`. The handle API is intentionally just explicit
input loading; it does not infer configuration, basis, or gauge identity.

Important constructor behavior:

- `eigenvector=None` is valid for `p2p` only. `usedNe` needs eigenvectors;
  `usedNp` needs a left point input. `point_right` and `usedNp_right` select
  the independent right endpoint; when omitted, the legacy `point` input and
  `usedNp` prefix serve both endpoints. Handles carry numeric arrays only;
  actual dataset identity and ordered label selection belong to planning and
  the orchestration.
- `paths=None` expands `num_nabla` through `paths_up_to`. `legacy` enumerates
  spatial directions only; `v2` enumerates eight spatial/temporal directions.
  Explicit paths require `path_version="v2"`; they are ordered `GaugePath`s
  with no duplicate path. An empty path is the local component.
- `momentum_list` is a nonempty ordered sequence of integer `(px,py,pz)` tuples.
  `phase_convention` is `midpoint`, `anchor`, or `none`.
- `path_boundary` is `periodic` or `open`. Open paths that leave the temporal
  range have zero dense blocks or empty sparse COO blocks; spatial coordinates
  remain periodic.
- `calc_mode` selects the default path or derivative recipe when recipe is omitted. `debug` is a
  retained compatibility argument and currently has no effect.
- Dilution is opt-in: `is_blending=True` requires
  `dilution=(total_counts, used_counts)`. Passing `dilution` without that flag
  is an error. The resulting mode-pair estimator is applied once to V/V values.
- `ne_block_size` bounds V-mode tiles for path blocks. It does not bound the
  resident input eigenvector handle or the complete requested output.

Public methods and defaults:

```text
_ElementalEngine.load(self, key)
_ElementalEngine.calc_path_blocks(
    self, t, blocks=("v2v",), *, recipe=None, rows=None, write=None,
)
_ElementalEngine.stout_smear(self, nstep, rho, *, matrix_batch_size=4096)
_ElementalEngine.project_SU3(self, *, matrix_batch_size=4096)

```

`calc_path_blocks` returns a sector dictionary containing borrowed complex arrays,
or None when a synchronous write callback is supplied. For the meson generator the usual
V/V path shape is `(Npath,Nmomentum,Ne_left,Ne_right)`; the derivative recipe has
`(Nderivative,Nmomentum,Ne,Ne)`. `calc_path_blocks` returns selected sectors:

| Sector | Dense block axes | Result without `write` |
|---|---|---|
| `v2v` | `(path,momentum,left_mode,right_mode)` | workspace buffer |
| `v2p` | `(path,momentum,mode,point,color)` | workspace buffer |
| `p2v` | `(path,momentum,point,color,mode)` | workspace buffer |
| `p2p` | `(path,momentum)` | tuple-of-tuples of `SparsePointBlock` COO values |

The `blocks` argument must contain distinct known sectors. A vector sector
requires eigenvectors; every point sector requires an actual point set/handle.
Each `SparsePointBlock` has `rows`, `columns`, `(nnz,3,3)` `values`, and
`point_shape=(Nleft,Nright)`. `to_dense()` is an explicit allocation of
`(Nleft,3,Nright,3)`; no implicit dense point-pair matrix is created.

`calc_path_blocks(..., write=callback)` calls
`callback(path_index, momentum_index, sector, slices, value)` synchronously.
For `p2p`, `slices=()` and `value` is a `SparsePointBlock`; other sectors use
mode/point slices. The callback must consume or copy each borrowed tile before
returning. It returns `None` and does not publish completion metadata.

Meson `calc_path_blocks(t, rows=rows, write=None)` accepts consecutive rows of
`{"id": int, "terms": [{"path": steps, "coefficient": [real, imag]}]}`.
Only spatial paths are accepted. Duplicate paths in a row are merged before
projection. Without a callback it returns
`(Nrow,Nmomentum,Ne,Ne)`; the callback is
`write(row_index, momentum_index, "v2v", (left_slice,right_slice), value)`.
Projection sums path contributions before mode-pair contraction.
Projected callbacks are emitted in row-major, then momentum order. Phase arrays
are requested for the current momentum batch; a later phase-construction error
can therefore follow already emitted callbacks. The producer does not roll
back that synchronously consumed prefix; runtime/storage owns durable recovery.

Eight-direction current values use the same path entry: construct with
`paths=[(d,) for d in range(8)]`, `path_version="v2"`, midpoint phase and the
required path boundary. One borrowed V/V row has shape `(8,Nmomentum,Ne,Ne)`.
Temporal paths consume the actual neighboring basis; open-edge directions
are zero. The caller loops over physical times and copies borrowed rows if
assembling the legacy all-time layout. `contracts.current` builds and validates
the raw layout contract independently of numerical production. Gauge preparation
still holds all times; a one-time output is not a process memory cap.

The meson recipe keyword accepts `paths` or `derivatives`. If omitted, the
constructor's calc_mode selects the default. Projected rows select projection
and cannot accompany an explicit recipe or a non-V/V sector. Derivative output
uses the existing full-stack kernel, including when a write callback is supplied.
Callbacks consume complete derivative matrices synchronously; this is not a
streaming derivative working-set guarantee. Invalid selectors fail explicitly.

The three-leg API is:

```text
_BaryonEngine(
    latt_size, gauge_field, eigenvector,
    momentum_list=((0, 0, 0),), usedNe=None, paths=None, *,
    flavors=("q", "q", "q"), array_module=np, ne_block_size=16,
    spatial_block_size=256, phase_convention="anchor",
    path_boundary="periodic", cache_bytes=64*1024**2,
    stream_eigenvectors=False, vector_scratch_limit_bytes=None, selection=None,
)
_BaryonEngine.load(self, key)
_BaryonEngine.calc_path_blocks(
    self, t, blocks=("vvv",), *, rows=None, distances=None, write=None,
)
_BaryonEngine.distance_components(distances, *, flavors=("q","q","q"))
```

`eigenvector` is one reusable handle or three handles. Non-stream mode snapshots
the selected data on `load`; stream mode requires three `read_modes(key,time,
start,stop)` handles, each with integer `Ne`, and stages only one tile per leg.
`usedNe` and `ne_block_size` accept one positive integer or a triple. Baryon
`paths` are unique triples of ordered paths, defaulting to three local paths.
`flavors` labels are stored in `BaryonComponent` metadata only; the raw numeric
kernel does not apply flavor or spin tensors. The color epsilon contraction is
unconjugated and has no hidden volume normalization.

`calc_path_blocks` returns
`(component,momentum,N0,N1,N2)` or `None` for streaming. `blocks` must be
exactly `("vvv",)`. Its synchronous callback is
`write(component_index, momentum_index, "vvv", (slice0,slice1,slice2), value)`.
`rows` projection rows have consecutive integer `id`s and terms containing
`paths=[P0,P1,P2]` and `coefficient=[real,imag]`; each correlated path triple is
contracted before coefficients are summed. `distances` takes a sequence of
three-distance tuples; each nonzero leg distance independently averages six
straight spatial paths.

The low-level workspace and standalone numerical helpers keep these exact
signatures and are re-exported for compatibility:

```text
ElementalWorkspace(latt_size, gauge_field, *, array_module=np, phase_convention="midpoint")
ElementalWorkspace.invalidate(self)
ElementalWorkspace.load(self, key)
ElementalWorkspace.links(self)  # property
ElementalWorkspace.time(self, t)
ElementalWorkspace.buffer(self, name, shape)
ElementalWorkspace.phase(self, momentum, displacement=(0, 0, 0), convention=None)
ElementalWorkspace.stout_smear(self, nstep, rho, *, matrix_batch_size=4096)
ElementalWorkspace.project_SU3(self, *, matrix_batch_size=4096)

generate_baryon_elemental(vectors, momentum, lattice_shape, *, phase_sign=1,
                          block_size=256, backend=np, out=None)
re_combine(array, insertion_list, axis, out=None, *, array_module=np)
```

`generate_baryon_elemental` takes three `(Ne,Lz,Ly,Lx,3)` arrays (mode counts
may differ) and returns their epsilon-contracted momentum tensor. `phase_sign`
is `+1` or `-1`; it changes the phase conjugation only. `re_combine` contracts
an existing component axis with finite `(coefficient,index)` rows; a supplied
`out` must have compatible shape/dtype and may not alias `array`.

Internal helper signatures (not exported from `api.py`; callers should use
the generator methods above) are:

```text
paths_up_to(order, version="legacy")
derivatives_up_to(order)
derivative_assignments(directions)
covariant_derivative(vectors, direction, links, t, xp)
differentiate(vectors, directions, links, t, xp, step_transports=None)
displace(vectors, distance, links, t, xp)

path_transport(links, path, t, boundary="periodic", *, array_module=np)
apply_transport(transport, vectors, displacement, *, array_module=np)
momentum_phase(latt_size, momentum, displacement=(0, 0, 0),
               convention="midpoint", *, array_module=np)

bilinear(left, right, phase=None, *, out=None, array_module=np)
point_pairs(left, right, displacement, spatial_size)
sparse_point_block(transport, phase, points_left, points_right, displacement,
                   spatial_size, *, array_module=np, pair_indices=None)
trilinear(vectors, phase, *, block_size=256, out=None, array_module=np)
dilution_pair_weights(total_counts, used_counts)
```

Private implementation methods retain these signatures in the current source:
`_ElementalEngine._load_inputs(self,key)`, `_time(self,t)`,
`_buffer(self,name,shape)`, `_v(self,t,s)`,
`_directed_current_inputs(self,boundary)`,
`_phase(self,momentum,displacement=(0,0,0),convention=None)`,
`_calc_recipe(self,t,recipes,*,kind)`, `_BaryonEngine._load_inputs(self,key)`,
`_time(self,t)`, `_calculate(self,t,rows,name,write)`, and
`_triple_counts(value,name)`. They are not cross-module APIs.

## State, algorithms, errors, and ownership

`load(key)` invalidates old inputs first. A successful load owns a C-contiguous
direction-major gauge copy `(4,Lt,Lz,Ly,Lx,3,3)` and stores the selected input
references/snapshots described below. Any gauge, vector, point, or mode-block
load error clears the usable state; calculation and gauge transforms then fail
until another successful load. Failed/partial stout or SU(3) transforms also
invalidate the workspace. A callback exception propagates; callback output is
not transactional and completed slices are not tracked here.

Shape, range, duplicate-path/point, invalid boundary/phase/mode, unsupported
sector, or incomplete basis-count errors raise `ValueError`. Calculation
without a usable load raises `ValueError`. Input handle exceptions propagate
after invalidating the producer. Streamed baryon reads must return finite
floating/complex `(mode,z,y,x,color)` blocks; a bad read invalidates state.
`SparsePointBlock.to_dense()` rejects duplicate COO pairs because no duplicate
accumulation policy is defined. Open-boundary path excursions are a valid
zero/empty result, not an error. Baryon rejects a non-callable callback with
`ValueError`; meson calls the supplied callback when it emits a tile, so a
non-callable value raises Python's call `TypeError`. Exceptions raised by a
valid callback propagate to the caller.

Gauge links are loaded from the supplied handle through the existing
`production.gauge.project_su3` and `production.gauge.stout_smear` service.
These transforms act on the producer-owned spatial links; temporal links stay
unchanged. They do not alter the external handle. `ElementalWorkspace.links`
is a borrowed internal array and callers must treat it as read-only except for
the workspace's own transformations; Python does not enforce a read-only flag.
Phases are cached independently of gauge state (workspace LRU cap: 64 spatial
phase arrays). The projection branch requests phases per momentum batch instead of
holding a second dictionary for all momenta. Its batch stack and phase-building
temporaries are separate owners, so the LRU count is not a whole-call peak
memory bound. With more momenta than the LRU retains, a later row batch can
recreate evicted phases; this trades work for bounded cache residency and does
not promise a runtime improvement. Named workspace buffers are mutable scratch
owned by the workspace.

Meson eigenvectors are consumed as a borrowed `load(key)` result; the generator
does not snapshot the complete vector field. CPU mode slices used by bilinear
kernels are copied. A backend-native array with matching dtype can be read as
a view; dtype conversion may allocate. The handle must keep returned data valid
through the calculation. Point-set coordinates are copied into generator-owned
host storage. The neutral `SpatialPointSet` is consumed directly; no spatial
sampling generator is imported or run.

Baryon non-stream mode snapshots the selected vectors immediately. Shared
handles are copied once at the maximum requested leg count and reused as
read-only inputs. Stream mode calls `read_modes` per tile, validates the
`(mode,z,y,x,color)` block, copies it to producer scratch, and invalidates all
inputs if a read/validation fails. The three stream buffers are overwritten on
the next tile. No EV or PR generator/private state is accessed.

Dense return arrays are workspace-owned borrowed results. A subsequent call
using the same output name and shape overwrites them. Copy before retaining an
earlier result. Callback values must be copied, written, or fully consumed
synchronously before callback return; this is essential for baryon tiles,
which explicitly reuse two mode-tile buffers. Sparse `p2p` rows, columns and
matrix values are allocated per emitted block; keep them or call
`SparsePointBlock.copy()` to make an independent copy. Sparse COO stores only
matching ordered point pairs; `to_dense()` deliberately allocates its dense
result and rejects duplicate pairs.

Numerical conventions retained by this coarse module:

- Meson V/V is `left† phase transport(right)`; no spatial-volume factor is
  applied. V/P and P/V keep the color axis explicit; P/P is a sparse color
  matrix. Midpoint phase includes half the spatial path displacement; anchor
  phase is anchored at the left endpoint. `none` disables phase.
- `GaugePath` uses ordered steps `0..5=(+x,+y,+z,-x,-y,-z)` and `6,7=(+t,-t)`.
  Periodic temporal paths wrap. Open temporal paths check the full path's
  intermediate time bounds, not only its final endpoint.
- Derivatives retain ordered left/right action and the `(-1)**right_order`
  sign. Spatial covariant difference is `U(x)psi(x+mu)-U†(x-mu)psi(x-mu)`.
- Baryon tensors retain color epsilon ordering and no conjugation of the three
  vector fields; phase, paths, and the flavor labels keep their established
  axes. Spin and external source/sink terms are assembled by callers.
- Dilution weights are computed by the copied local helper
  `elemental.mode_weights.dilution_pair_weights`; their numeric formula is
  unchanged from `sampling.mode_weights`. The old helper has no remaining
  production-code consumer after this move; `production/elemental/test_elemental.py` and
  `test_elemental_cuda.py` still import it as an independent expectation.

The raw producer does not create `SourceOrigin`, dataset identity, or parent
records. `gauge_field` and vector handles currently expose `load(key)` (and the
baryon streaming protocol where requested), not stable source IDs. Thus actual
gauge variant, configuration and ordered basis identity must be bound by the
calling runtime/storage contract; shape, `usedNe`, and the `key` alone are
not accepted as identity. This module does not generate missing eigenvectors,
point sets, perambulators, or spin/flavor tensors. Explicit smearing and
projection mutate the producer's gauge copy but do not record a derived
gauge-version ID; the module also does not attach `SourceOrigin` to each tensor.

## Raw spatial V/V memory estimate

`planning.production.estimate_elemental_memory(input_specs, output_spec, *, gauge_id, basis_id, ne_block_size=32, target="cpu", workspace_reserve_bytes=0, host_staging_bytes=0)` is a pure logical-allocation model for explicitly mapped or default full-time, complex128, equal-time spatial `elemental_vv` output with the same explicit stored basis prefix at both ends. A script may call it to inform its own resource choices. It does not open dataset payloads, create a producer or writer, apply runtime admission, publish, or recover results.

The model accounts for both registered parent payloads; the full gauge reader stack overlapping the workspace's owned direction-major complex128 links; and the larger calculation/write phase, including one complete `(component,momentum,sink_mode,source_mode)` row, mode/path-link arrays, the bounded phase cache, writer validation mask, digest allowance, and caller-declared workspace/staging reserves.

For phase construction, `none` still allocates `xp.indices` before taking its no-phase branch: three int64 coordinate planes overlap complex128 `ones`, so a cache miss costs 40 bytes per spatial site. Anchor/midpoint construction is modeled at 64 bytes per site. The model compares phase-specific peaks rather than summing nonoverlapping phases. Python object overhead, allocator/pool retention, filesystem page residency, and opaque native/BLAS scratch remain outside the estimate; declared reserves are not a hard RSS or native-memory bound. The CPU-only model does not imply GPU support.

`tests/test_elemental_production_budget.py` retains independent checks of the pure allocation arithmetic and phase-cache lifetime. No automatic memory admission or persistent production transaction is attached to this estimator.

## Memory scope

All byte estimates below assume complex128, int64 point coordinates, and
exclude Python containers, allocator pools, external loader caches, backend
workspaces, and output retained by a callback:

- Owned gauge input is `4*Lt*Lz*Ly*Lx*9*16` bytes. This is a full resident copy;
  loading is not record-streamed.
- Meson input vectors remain resident in the caller's load result, typically
  `input_Nt*Ne*Lz*Ly*Lx*3*16` bytes, without an EL copy. Point coordinates are copied
  as `input_Nt*Np*3*8` bytes. The workspace LRU retains up to 64 phase arrays, or
  `64*Lz*Ly*Lx*16` bytes; projected batching also builds a current-batch stack
  and construction temporaries, so this LRU cap is not a whole-call phase
  memory bound.
- Meson path `ne_block_size` bounds mode staging/contraction tiles, not the
  output. Dense output sizes are `Npath*Nmomentum*Ne^2*16` for V/V,
  `Npath*Nmomentum*Ne*Np*3*16` for V/P, and the same with `Ne` last for P/V.
  P/P stores `nnz*(2*8+9*16)` bytes per momentum/path, plus COO/container
  overhead. Full volume path transport uses `Lz*Ly*Lx*9*16` bytes at a time.
- Derivative calculation uses its complete output plus a cached derivative
  field bank while `field_bytes*Nderivative <= 1 GiB`; above that threshold it
  streams derivative recipes by mode tile. The output is still resident.
  Projected meson output is resident unless a callback is used. Path transport
  cache is capped by its 512 MiB decision; projected row/momentum/field working
  tiles are chosen against a nominal 256 MiB field budget, not a total-process
  memory cap. Projected phases are gathered per momentum batch, with no extra
  all-momentum phase dictionary or unused spatial coordinate grids.
- Baryon eager input snapshots use
  `sum(input_Nt*unique_handle_max_modes)*Lz*Ly*Lx*3*16` bytes. Stream scratch is
  `sum(min(usedNe_i,block_i))*Lz*Ly*Lx*3*16`, reported as
  `vector_scratch_bytes` and checked against the optional limit. Its output is
  `Ncomponent*Nmomentum*N0*N1*N2*16` bytes unless a callback is used. The
  per-call transformed-field/path cache is capped by `cache_bytes`; each
  three-leg mode tile and two reusable epsilon-result tiles scale as
  `prod(min(Ni,block_i))*16` bytes each. `spatial_block_size` bounds the
  contraction's volume slice, not cached inputs or output.
- Callback output is user-owned memory. The generator cannot constrain how
  much the callback retains. Consequently neither a mode/spatial tile nor
  `stream_eigenvectors=True` alone proves total working-set or output fits in
  memory.

These are code-derived bounds; no process-wide peak-RSS study or GPU allocator
measurement was performed for this change.

## Source mapping and verification status

The numeric starting points are the pre-existing canonical EL implementations
in `production/elemental/generator.py`, `baryon.py`, `workspace.py`,
`structures.py`, `transport.py`, `kernels.py`, plus the pre-existing
`production/gauge.py` and `stout_smear.cu`. The compatibility mapping is
`lattice/generator/elemental.py` → meson/path/derivative and raw-current paths,
`lattice/generator/baryon_elemental.py` → three-leg production, and
`lattice/generator/stout_smear.cu` → the GPU stout kernel. The local dilution
helper is copied from `sampling/mode_weights.py`. No numeric kernel was
rewritten for API unification. The eight-link current recipe
is now shared with the one-time entry; existing direction, phase, endpoint
and contract formulas are preserved.

The strict legacy migration registry remains authoritative for old-symbol migration;
this API facade does not close unrelated migration entries. GPU/CUDA compilation,
QUDA conversion, MPI and production-scale memory require separate evidence. The
generator itself does not publish datasets or resume work; callers compose the
existing storage/runtime services.

EL-owned numerical tests live in `tests/production/elemental/`; cross-module,
weak-field and publication tests remain in `tests/`. `test_sparse_time.py` checks
compact meson/baryon data, physical endpoints and streamed input order against
independent site matrices and determinants. `tests/test_sparse_elemental_dataset.py`
checks actual publication/reload, resumed missing slots, and independent left/right
point-time and point-label mappings. Existing full-time and weak-field invariants
remain covered. Historical test counts are not a claim about the
current checkout.
These bounded synthetic CPU checks do not certify GPU/MPI, physical ensembles or
production-size memory/performance.

## Registered raw spatial projection derivation

`contracts.elemental.spatial_projection_rows(rows)` owns the existing finite spatial path/consecutive-row description, shared by the projection branch. The fused production algorithm still merges repeated paths and uses its existing transport and bilinear kernels. No persistent identity or I/O enters the numerical kernel.

`planning.production.elemental_projection_documents` validates a complete registered raw dataset with its declared time mapping, selects required paths and optional ordered mode prefixes, and describes the projected result. `production.elemental.planning.plan_indexed_elemental_projection(index, raw_id, rows, *, usedNe=None, data_ids=None, source_policy="allow_unknown")` is a read-only planner. The script reads the requested raw components, calls public `re_combine`, and passes the result to `DatasetWriter.write`; it owns publication and the reader/writer lifetimes. Preserve the actual raw parent identity and time mapping.

`planning.production.estimate_elemental_projection_memory(raw_spec, projected_spec, *, raw_id, target="cpu", workspace_reserve_bytes=0, host_staging_bytes=0, host_writer_hash_peak_bytes=0)` is an independent pure model and returns `(host_estimate, device_estimate)` using `contracts.memory.MemoryEstimate`. It requires the exact projected spec generated from the actual raw parent/view and finite recipe. It accounts for full raw parent bytes B, one-time selected input X, coefficient matrix C, complex128 output row Y, finite mask F=Y/16 and the caller-declared writer digest reserve H. It compares phases B+H, B+X+C+2Y, and B+Y+max(F,H), then adds workspace and staging reserves. Duplicate recipe terms remain ordered in the scientific definition and accumulate into the coefficient matrix. The script releases the input stack after recombination and the projected row after synchronous write, avoiding cross-time overlap. This logical CPU estimate is not an RSS/native hard bound; opaque NumPy/BLAS/reader staging remains a caller reserve.

The result binds actual raw/basis IDs, requested input coordinates and the full complex recipe. Raw and projected kinds are distinct; projected cannot attest missing raw paths. Existing anchor/midpoint/none phases are inherited. This spatial V/V derivation does not implement current/time-shifted projection. Interface scripts and layouts are in [production 流程](elemental_persistence.md#只读规划).

## Registered eight-link current contract

`planning.production.directed_current_dataset_document(gauge, basis, *, gauge_id, basis_id, momentum_list=((0,0,0),), usedNe=None, boundary="periodic", selection=None)` returns `(requested_basis_spec, document)`, with full physical time by default or explicitly mapped anchor slots, all eight components, complex128, midpoint phase, actual ordered modes and input views. IDs may be logical references until a caller binds the actual parents. It validates known compatible canonical fields, boundary, unique integer XYZ momenta and a nonempty covered mode prefix without I/O or producer creation.

`production.elemental.planning.plan_indexed_directed_current(index, gauge_id, basis_id, *, momentum_list=((0,0,0),), usedNe=None, boundary="periodic", data_ids=None, source_policy="allow_unknown", selection=None)` plans the registered inputs. The script calls `ElementalKernel.calc_path_blocks(t, ("v2v",))` for the eight declared raw directions and synchronously saves the existing arrays through `DatasetWriter.write`. The spec binds the actual parents, boundary, phase, momenta and physical-time mapping. See the [production skill](../../../.agents/skills/lqcdweave-elemental/references/production.md).

### Pure directed-current CPU memory estimate

`planning.production.estimate_directed_current_memory(input_specs, output_spec, *, gauge_id, basis_id, target="cpu", workspace_reserve_bytes=0, host_staging_bytes=0)` is a separate pure logical-array model, not runtime admission. It counts both complete parent payloads by registered dtype, the full gauge read overlapping owned complex128 links, a conservative selected-time vector reserve, path transport construction and bilinear temporaries. Selected Ne bounds the default mode tiles. The phase model includes the 64-entry midpoint LRU and miss construction; the output model includes one reused complete row, validation mask and caller-declared digest allowance. Synchronous writing consumes the borrowed row before the next call reuses its buffer; there is no full-time output array in this calculation.

Caller workspace/staging reserves cover opaque NumPy/BLAS and reader/adapter workspace. This is a logical CPU model, not an RSS or hard native-memory bound. It provides no GPU, MPI, shared-filesystem, or production-scale capacity result. The legacy raw-current adapter remains a full-physical-time consumer path; mapped DatasetElemental binding and V/P endpoint consumption are separate interfaces with explicit endpoint checks. See [the 流程 usage](elemental_persistence.md#只读规划). The numerical current API does not certify a Ward identity: transport and Dirac gauge IDs remain separately explicit, and strict Wilson conservation requires measurement-level validation of actual links, action/r, boundary and normalization.

## Registered point-ended spatial contract

The metadata planner `planning.production.point_elemental_dataset_document(gauge, basis, points, *, product, gauge_id, basis_id, points_id, components, momenta, modes=(), point_labels=(), points_right=None, points_right_id=None, point_right_labels=None, phase_convention="anchor", current_schema=None, temporal_path_schema=None, path_boundary="periodic", selection=None)` defines point-ended products from registered actual IDs. `production.elemental.planning.plan_indexed_point_elementals(index, gauge_id, basis_id, points_id, *, paths=((),), momentum_list=((0,0,0),), sectors=("v2p", "p2v", "p2p"), usedNe=None, usedNp=None, points_right_id=None, usedNp_right=None, point_right_labels=None, phase_convention="anchor", current_schema=None, temporal_path_schema=None, path_boundary="periodic", ne_block_size=32, data_ids=None, source_policy="allow_unknown", selection=None)` previews those requests. The script calls the public elemental kernel and separately saves its result. V/P and P/V are dense arrays; P/P carries sparse rows, columns, values and point_shape. Preserve sparse support and the actual left/right point labels when choosing the stored representation. No library helper computes and writes these products.

`basis_id` is `None` for a pure `("p2p",)` request; that request rejects `usedNe`. Every product binds actual gauge and point IDs; V/P products additionally bind the actual ordered basis ID and mode labels. Products default to physical times `0..Lt-1`; an explicit selection uses mapped `0..Nt-1` storage slots. They use complex128 and unscaled normalization, and preserve caller component and momentum order. By default, components are spatial path coordinates `{"path": [step, ...]}` with ordered steps 0..5, periodic spatial paths, and anchor, midpoint or no Fourier phase. Temporal steps require the explicit `temporal_path_schema` described below; the one-link current schema remains separate and mutually exclusive.

The fixed axis layouts are:

| Product | Kind | Axes after time | Bound bases |
|---|---|---|---|
| V/P | `elemental_v2p` | `component,momentum,mode_left,point_right,color_right` | `mode_left` to actual basis; `point_right` to actual point set |
| P/V | `elemental_p2v` | `component,momentum,point_left,color_left,mode_right` | `point_left` to actual point set; `mode_right` to actual basis |
| P/P | `elemental_p2p_row_sparse` | `component,momentum,point_left,color_left,color_right` | `point_left` to actual point set |

P/P uses the existing sparse pair kernel but stores a fixed left-point row table, which can be rectangular. Each row slot contains the 3×3 value for its unique translated point pair; slots with no right match contain zero. Its `point_pair_layout` convention is `left-point-row-slots-v1`. The legacy same-pool definition binds both endpoints to the same actual point dataset and ordered prefix. A distinct right dataset is recorded as `definition.point_right_id`, with `point_right_selection="explicit_ordered_labels"` and the exact ordered `point_right_labels`; the right actual ID is also an actual parent. The explicit component path, both registered coordinate tables and their ordered labels determine sparse column support. Readers must reconstruct that support from the registered point datasets and path; numerical zeros never encode the sparsity pattern.

The planner accepts optional `points_right_id`, `usedNp_right`, or `point_right_labels`. `points_id`/`usedNp` name the left endpoint and its stored prefix. When no right-specific option is supplied, the right endpoint keeps same-pool prefix semantics. For an independent right pool, omit `usedNp_right` to select all registered right labels, give `usedNp_right` for its stored prefix, or give `point_right_labels` for an explicit ordered selection; the count and label forms cannot be combined. P/V consumes only the left pool. A pure right-only V/P request may pass `points_id=None` with an explicit `points_right_id`; pure P/P still needs both actual endpoint identities and `basis_id=None`.

The planner validates registered configuration, axes, selected point labels, phase, unique paths/momenta and complete physical time. Row writing uses actual coordinates and rejects duplicates or out-of-range positions before storing rows. A script that needs publication or recovery composes the generic storage/runtime services explicitly.

`planning.production.estimate_point_elemental_memory(input_specs, output_specs, *, ne_block_size=32, target="cpu", workspace_reserve_bytes=0, host_staging_bytes=0)` is a separate pure calculator for selected V/P, P/V and rectangular P/P products. It models distinct parent payloads, full gauge conversion, ordered point tables, basis rows and mode tiles, phase/transport arrays, output rows, and P/P sparse blocks retained through dense-row assembly. Caller reserves cover opaque Python/NumPy/BLAS/reader work; the estimate is not RSS/native admission and does not execute or constrain production.

### Optional per-elemental point compensation

`ElementalGenerator(..., elemental_weighted=False)` keeps the existing raw-value behavior by default. With `elemental_weighted=True`, the public generator multiplies each current's V/P block by its local inverse-inclusion factor. The same Boolean is used by the correlator's `calculate(..., elemental_weighted=True)` when consuming those values; the correlator validates the data marker and computes any graph-level residual coefficient.

When `blocks` includes a point-ended sector, the weighted input must be a `SpatialPointSet` with an explicit `SpatialSamplingDesign` and one explicit `pool_id` for every retained physical time, or a `SpatialPointHandle` whose owned `.points` satisfies the same contract. For a handle with a bound configuration, call `generator.load(cfg)` before calculating. Imported bare coordinates cannot be weighted. `sample_spatial_points` issues a UUID-backed realization ID and a distinct ID for each actual draw segment; times sampled by one segment share its ID. A new call gets a new realization identity even when the seed and coordinates match. Save/read and reuse paths must preserve actual pool identity: caller-supplied point sets retain their `pool_ids`, while registered local-sampling data retain the realization through their actual `data_id` and design segments. Never reconstruct a shared pool from a seed or coordinates. The generator exposes `elemental_weighted`, `sampling_pool_ids`, and `sampling_groups` for explicit downstream binding. A plain coordinate array remains usable with the default `False` option.

For spatial volume `M=prod(lattice_size[:3])` and the original sampled count `n`, the factors applied to generated blocks are:

| Sector | Local factor |
|---|---:|
| V/V | `1` |
| V/P or P/V | `M/n` |
| P/P, same explicit pool and same ordered point label | `M/n` |
| P/P, same explicit pool and different ordered point labels (`n>=2`) | `M(M-1)/(n(n-1))` |
| P/P, different explicitly independent pools | `(M_left/n_left) * (M_right/n_right)` |

Elemental structure metadata has exactly three values: `general`, `diagonal`, and `point_diagonal` (user-facing names: 一般、完全对角、P 对角). `diagonal` means equality of the complete left/right basis index: mode index for V/D and point plus color for P. `point_diagonal` applies to P and guarantees equal point indices while retaining the full color matrix. Spin and Gamma factors are independent of these basis indices. Unit and zero matrices are diagonal numerical cases, not extra structure types; the producer never scans numeric zeros to infer a declaration.

The producer derives a sparse `ElementalGenerator.elemental_structures` mapping keyed by `(sector, component, momentum_slot)` and stores only non-general proofs. P/P with an empty spatial path is fully diagonal, including nonzero momentum slots; a closed Wilson path proves only P diagonality. V/V and D/D are fully diagonal only for an empty path at zero momentum and an explicitly orthogonal actual input basis. A registered `DatasetFieldHandle` supplies this proof from the actual basis spec's `conventions.normalization="orthonormal"` or `conventions.basis_orthogonal=True`; a custom advanced handle may declare `basis_orthogonal=True`, which its caller is responsible for making truthful. A bare array is not an orthogonality proof. Noise-vector orthogonality propagates only from a proven parent; legacy one-vector groups remain diagonal even when their norm is not one. QR proofs apply only within the same realization and time. The producer does not infer overlaps between distinct actual V/D bases.

For P/P point-equality support, every recipe term must have zero spatial endpoint displacement modulo each spatial extent. Temporal displacement is accepted only for the same actual time or an explicitly shared ordered point basis across times. Registered P/P metadata requires the same actual point dataset and identical ordered labels; the numeric producer may prove additional shared-time cases from the actual point set. The planner has no frontend switch and never inspects values.

Advanced imports to `ArrayElemental(...)`, `from_npy(...)`, and `from_binary(...)` accept `elemental_structures={(sector, component, momentum_slot): structure}`, where `component` is the actual component identity. The old `point_diagonal_components` input remains a compatibility alias for P diagonality. A Dataset stores its own sector's declarations in `conventions.elemental_structures` as entries `{component: row_slot, momentum: momentum_slot, structure: ...}` and restores row slots against its component table. Legacy `conventions.point_diagonal_components` rows read back as `point_diagonal`; missing structure metadata is `general`. If selected left/right basis or point coordinates have different ordered labels, that selected view drops the proof to `general`. Sparse P/P records use the same structure declaration while retaining the existing COO representation. Numeric zero values never establish support.

For dense fully diagonal V/V and P/P factors, the correlator prepares only the diagonal coefficients multiplied by Gamma, then merges the corresponding mode or point-plus-color index without dropping those coefficients. P-diagonal factors still contract the color matrix. Sparse P/P continues through the existing COO bridge; this declaration does not add GPU sparse contraction support. When P/P structure proves equal point labels and the actual sampling pool and ordered selection also match, the sampling expansion can quotient those labels before Bell-partition enumeration, as described in the [user guide](../operators/elemental_frontend.md#elemental-结构).

`ArrayElemental(...)`, `ArrayElemental.from_npy(...)`, and `ArrayElemental.from_binary(...)` accept `elemental_weighted=True` to declare that their values already contain these factors; adapters never multiply them again. `DatasetElemental` reads `conventions.elemental_weighted` and requires all loaded sectors to agree. `SparseElemental` inherits the dense reader's marker and treats COO values as already weighted; a weighted identity record is not accepted when a non-unit factor is required because it has no explicit values to modify. In every adapter, point-pool IDs and `SamplingPoolSpec` metadata remain explicit bindings.

The spatial-point part supports the existing fixed-size uniform design, including a vector endpoint of kind D. Observed-count, nonuniform or correlated point sampling remains unsupported. QR D uses explicit group/pool provenance and raw covariance weights under the same elemental_weighted option; it does not use the older uniform mode-selection dilution contract. See [V/D weighting](../../../.agents/skills/lqcdweave-elemental/references/dilution.md). See the [user guide](../operators/elemental_frontend.md#空间点采样计权), correlator's [sampling-current reference](../../../.agents/skills/lqcdweave-correlator/references/sampling-current.md#elemental-weighted), and the [sampling-weight CPU regression](../../../tests/production/elemental/test_sampling_weights.py). CPU coverage is listed in `tests/production/elemental/test_sampling_weights.py`; it does not establish GPU/MPI or production-scale behavior.

### Registered temporal point paths

The explicit `temporal_path_schema="lattice.elemental.temporal-point-path/v1"` supports V/P and P/V products for ordered `GaugePath` steps 0–7 that include at least one temporal direction. It is separate from the one-link Current schema and does not create a `CurrentTerm`. The numerical producer supports V/P, P/V, and sparse P/P products under this schema. The caller selects path, momentum, actual gauge/basis/point identities, ordered labels, phase convention, and periodic or open boundary. All physical anchor times are retained; periodic execution wraps according to the path, while an open anchor whose full path excursion leaves the temporal domain contributes zero. Sparse P/P records that anchor as an empty raw row. The numerical producer does not infer temporal identity from shape or reader.

The registered temporal point-path Diagram chain uses this raw schema through `DatasetElemental`, B's explicit sector capability, and E's existing Diagram expansion. A normal Meson remains V/V by default. To request point-ended contributions, the caller explicitly declares logical `requested_sectors`; source dagger maps a selected logical sector to its corresponding raw sector without changing the request. Sparse P/P keeps its declared left-row/right-column point identities. The default Diagram path is scalar; explicit sink_batch uses the restricted NumPy sparse P/P batch path described in the measurement orchestration, subject to its topology and memory checks. Any persistent output and row recovery are selected by the calling script through generic storage/runtime. Its CPU-only scope and measured limits are documented in [persistent Diagram measurement](../measurements/diagram_measurement.md#registered-temporal-vp-point-path-diagrams).

## Registered baryon VVV planning and numerical calculation

Pure record planning is in `planning.baryon`; `production.elemental.baryon_planning.plan_indexed_baryon_elementals(index, gauge_id, basis_ids, *, product="raw_paths", components=None, momentum_list=((0,0,0),), usedNe=None, phase_convention="anchor", path_boundary="periodic", data_ids=None, source_policy="allow_unknown", selection=None)` validates the registered fields/recipe and returns a wrapper containing `requests`, `graph`, `dataset_spec`, `times` and `product`. The product string names the output stage and dataset key. The public numerical calculation uses `BaryonElementalGenerator`; scripts choose output format and may compose a generic writer/runtime. There is no fixed baryon execution runner.

The three recipes call the public `calc_path_blocks` entry with paths/rows/distances. Spatial definitions use complex128, explicit mapped or default full-time coverage, tagged components, epsilon/unconjugated/unscaled conventions and independent actual mode0/1/2 bindings. The generator may stream eigenvectors when the caller requests it; a synchronous borrowed tile must be consumed or copied before the next calculation reuses its buffers. No full production memory bound or GPU/MPI validation follows from callback/streaming.

`adapters.OnDemandBaryonElemental` reuses this public generator to compute one selected runtime VVV block for an existing `Baryon` binding. It accepts only spatial path triples and does not write or register the result. Its input providers, selection order, cache scope and memory limits are documented in [the user guide](baryon_on_demand.md).

`planning.baryon.estimate_baryon_vvv_memory` is an independent pure estimate of visible input payloads, gauge load/copy, role-specific eager snapshots or streaming buffers, elemental caches and tiles, one output row, coverage bookkeeping and finite checks. Caller reserves remain necessary for unmodeled Python, allocator, BLAS and native workspaces. It does not inspect runtime capacity, admit a production job, write, publish or recover datasets; its result is not an RSS or device-capacity guarantee.

## Script boundary after fixed-runner removal

Spatial V/V projected rows internally tile the right-mode axis for dense delivery when that allows
more momenta to share a transported field under the existing 256 MiB weighted-field target.
Left modes, unwrapped displacements, midpoint phases, coefficients and mode-pair weights are unchanged.
The 512 MiB cache allowance is shared by color sums, reusable dense phases and transported tiles;
the latter two are only reused inside the current call. Callback delivery retains complete mode-pair
blocks and its ordered streaming phase lifetime. Neither target includes all inputs, outputs or native
workspaces, and a single minimal tile may exceed the weighted-field target. CPU tests compare independent
site/link matrices, remainder tiles and allocation/transport counts. GPU and production-scale performance
require separate validation; no new frontend parameter is introduced.

The elemental and baryon modules provide numerical calculators and input preparation/read-only planning helpers. Calling scripts choose the measurement recipe, actual times, output format/path and reader/writer lifetime. A planner does not execute production; the calling script computes a result and separately passes it to its writer. Generic storage/runtime can be composed when publication, registration, receipts or recovery are needed. The pure memory models are explicit calculators, not runtime admission around a fixed runner. For joint time sparsification, preserve the per-configuration `selection`, calculate `selection.times` into `Nt` logical slots, and keep gauge fields and Dirac solves on physical `Lt`; see [the shared temporal production contract](../temporal_sampling/temporal_production.md).
