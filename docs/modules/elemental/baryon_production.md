# Baryon elemental calculations and dataset planning

使用压缩时间输入前，先核对[共同时间交接约定与当前接线限制](../temporal_sampling/temporal_production.md)；不从 Nt×Nt 的 shape 推断输入或重建布局。

`BaryonElementalGenerator` is the public numerical entry for three-leg spatial baryon elementals. It calculates one actual physical time at a time; the calling script chooses the recipe and how to save, register and publish results. `plan_indexed_baryon_elementals` remains a read-only planner and does not execute a persistent production run.

## Numerical calculator

```python
from lqcdweave.production.elemental.api import BaryonElementalGenerator

calculator = BaryonElementalGenerator(
    gauge_field, (basis0, basis1, basis2), momenta, operator_table,
    Ne=(Ne0, Ne1, Ne2), ne_block_size=16,
    phase_convention="anchor", path_boundary="periodic",
)
for time in physical_times:
    row = calculator.calc_path_blocks(time)["vvv"]
    save_or_consume(time, row)  # consume synchronously or copy before the next calculation
```

One basis can be supplied for all three legs, or an explicit ordered basis triple can be used. `Ne=None` takes each complete basis; an integer is shared across legs, and a triple selects separate ordered prefixes. Stored mode labels and leg order are preserved. `operator_table` holds the component, three ordered spatial paths and complex coefficient. Distance recipes are expanded into correlated three-leg path triples before calculation. The result has `(component,momentum,N0,N1,N2)` axes, is unscaled, and adds no volume factor.

Arrays are loaded at initialization. Lazy reader handles use `load(configuration)`. With `stream_eigenvectors=True`, each basis handle must provide a positive integer `Ne` and `read_modes(key,t,start,stop)`; `t` is an actual physical time, and the returned block has shape `(stop-start,Lz,Ly,Lx,3)`. `calc_path_blocks(t)` supports VVV. Its output may borrow calculator workspace, so save or copy before the next calculation. `write` can synchronously receive borrowed three-leg tiles; it handles calculation delivery only and does not manage storage lifecycle. The generator does not add spin, flavor or volume factors. See [the elemental skill's baryon reference](../../../.agents/skills/lqcdweave-elemental/references/baryon.md) for current options.

## Indexed planning

```python
from lqcdweave.production.elemental.baryon_planning import plan_indexed_baryon_elementals

plan = plan_indexed_baryon_elementals(
    index, gauge_id, (basis0_id, basis1_id, basis2_id),
    product="raw_paths", components=[((0,), (), (3,))],
    momentum_list=((0, 0, 0),), usedNe=(2, 3, 2),
)
dataset_spec = plan["dataset_spec"]
stored_times = plan["times"]  # physical coordinates by default; slots with explicit selection
```

The planner validates actual IDs and compatible registered input specs, constructs product requests, and returns `requests`, `graph`, `dataset_spec`, `times`, and `product`. Source-policy and candidate-product decisions are visible in `graph`. It does not instantiate the numerical generator, create an output, or publish data. A caller may use its resolved dataset spec when composing a chosen writer/runtime path; that script owns writing and lifecycle.

`basis_ids` identifies three ordered actual basis roles and may repeat an ID. `usedNe=None` selects every mode; a positive integer applies the same prefix to each leg, while a triple specifies a separate prefix. Each leg view records the actual basis ID and labels.

`product` and `components` select one of the following forms:

| Product | Component input | Contract component |
|---|---|---|
| `raw_paths` | Ordered sequence of spatial path triples; default is three local paths | `BaryonComponent.paths(paths)` |
| `projected` | Rows with `id` and `terms`; each term has three paths and a finite `[real, imag]` coefficient | `BaryonComponent(("projected", row_id))` |
| `distance` | Nonnegative distance triples | `BaryonComponent(("distance", (d0,d1,d2)))` |

Spatial path steps are 0–5. Projected row IDs are consecutive from zero; repeated terms are allowed, including exact coefficient cancellation. Distance products independently average the six straight spatial directions on each leg before forming the correlated triple. Flavor belongs to propagator/Wick binding, not the spatial product identity. Phase is `anchor` or `none`; path boundary is `periodic` or `open`.

The dataset kind is `elemental_baryon_vvv` with axes `time,component,momentum,mode0,mode1,mode2` and complex128 values. All three mode axes bind actual basis IDs and exact ordered labels. Gauge/basis roles, requested views and parent IDs remain in the DatasetSpec. Default planning uses all physical times. `plan_indexed_baryon_elementals(..., selection=record)` instead produces canonical `0..Nt-1` output slots with the record stored in the existing time-coordinate contract; `plan["times"]` reports that output axis. The numerical generator accepts the same anchor selection and still takes physical t. Three input bases may keep independent saved order and extra coverage through their own time selections; every leg must cover each anchor. `DatasetBaryonElemental` reads the compact dataset in slot coordinates and retains `time_selection` for the physical mapping. Actual per-leg IDs/labels remain required; Nt or equal shape cannot establish compatibility. Caller-owned kernel/DatasetWriter publication and reload with independent leg mappings are checked by the bounded synthetic CPU regression `tests/test_sparse_elemental_dataset.py`.

## Reading a published baryon elemental

```python
from lqcdweave.adapters.baryon_dataset import DatasetBaryonElemental
from lqcdweave.contraction.binding.api import Baryon
from lqcdweave.contracts.elemental import BaryonComponent

with index.open(published_id) as dataset:
    elemental = DatasetBaryonElemental(dataset)
    component = BaryonComponent.paths(((0,), (), (3,)))
    header = elemental.describe_block(component, 0, 0, "vvv")
    vertex = Baryon(elemental, baryon_operator, source=True)
    # Keep the dataset reader alive while consuming the vertex.
```

`DatasetBaryonElemental` borrows its opened `DatasetReader`/view and does not close it. `describe_block` reads metadata; block reads select component, momentum, stored time coordinate and modes. For an explicitly mapped compact dataset, time is a slot in `0..Nt-1`; `elemental.time_selection` retains the actual physical order, `elemental.Lt`/`Nt` are the execution slot count, and `physical_Lt` is the original lattice extent. Compact baryon bindings support the three equal-time legs; nonzero temporal anchor/endpoint offsets are explicitly unsupported. The existing `Baryon` binding owns source-leg reversal, spatial conjugation and spin-basis conversion. The operator component must match the product's explicit component labels.

Persistent Diagram measurement remains a separate consumer ([see the measurement guide](../measurements/diagram_measurement.md)); it does not generate missing elemental inputs. `DatasetPropagator`/VSV remains the propagator input. The contraction checks each source leg's actual basis ID and coordinate order against the field ordering and graph edges; equal dimensions do not make different basis identities compatible. Existing in-memory bindings continue to calculate without indexed publication, but cannot be used to publish an indexed measurement. Shared physical blocks and logical tensors must preserve axis binding so reuse cannot conceal a mismatched edge.

The current baryon Diagram route is dense-V, NumPy, scalar batching, and scans the declared source/sink time slots (all physical times by default). Existing mixed Meson/Baryon graphs remain supported where the Diagram expansion declares them. A `sink_batch` request containing a Baryon is rejected before output creation; pure Meson sink batching keeps its established behavior. If a mixed graph explicitly chooses a three-item `usedNe`, the three mode counts must be equal; a pure baryon graph can retain independent per-leg mode selections.

## Ownership, source and memory scope

The generator's streaming protocol can stage one mode tile per leg. Its borrowed tile must be copied or synchronously consumed before the next tile reuses the buffers; this is still in addition to the full output time row and the numerical kernel workspace. `cache_bytes` limits only the transport cache. Callback use or streaming alone does not establish a process-wide memory bound.

`planning.baryon.estimate_baryon_vvv_memory` is an explicit pure logical-allocation model for visible input payloads, gauge loading/copy, role-specific eager snapshots or streaming buffers, elemental caches and tiles, one output row, coverage bookkeeping and finite checks. Caller workspace/staging reserves cover unmodeled Python, allocator, BLAS and native workspace. It does not inspect runtime capacity, admit or constrain an execution, create outputs, publish or recover datasets; it is neither RSS nor device-capacity proof.

For joint time sparsification, calculate `selection.times` into `Nt` logical slots, preserve the same `selection` with the data, and continue using physical `Lt` for gauge fields and Dirac solves. See the [shared temporal production contract](../temporal_sampling/temporal_production.md). The script chooses output format, path, source/provenance and lifecycle. No fixed baryon persistent runner owns these decisions.
