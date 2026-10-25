# PR：perambulator 生产（L2）

`PerambulatorGenerator` 从显式规范场句柄、实际源基底、按序汇基底列表和 Dirac 后端生产传播子。构造器只配置格点、作用量、资源与时间约定；每次 `calc` 提供实际源及汇。PR 不识别文件格式、不生成 D、不自动发布或恢复数据。V/P/D 产品、Gamma5 物理时间交换、六类 pair 保存和九类读取视图见本页后半部分。

默认使用完整物理时间 `Lt`。显式 `source_selection` / `sink_selection` 接受独立有序物理时间记录；求解仍在完整 `Lt` 上进行，只在所选汇时间投影，返回第一轴按 `sink_selection.times` 排列的 `Nt_sink` 槽。时间稀疏化、存储及生产脚本边界见[时间工作流](../temporal_sampling/temporal_production.md)和[通用生产说明](../../architecture/generic_production.md#perambulator-入口)。

## 模块与依赖

实现位于 `src/lqcdweave/production/perambulator/`，精确入口为 `lqcdweave.production.perambulator.api`；`PerambulatorGenerator` 也可从 `lqcdweave.production` 导入。`VectorBasis`、`PointBasis`、`PropagatorResult` 与 `PropagatorSolution` 从 perambulator `api` 导入。

PR 使用 `contracts.propagator` 的 `PropagatorBackend`、向量 `ModeBlockHandle`、`PropagatorEndpoint` 与 `PropagatorBlock` 契约。输入的路径、文件格式、QIO、设备初始化及原生布局转换由调用方 adapters 负责。D 是实际输入；PR 不生成混合系数或估计量权重。

后端至少提供：

```text
array_module
load(gauge, parameters) -> None
solve_into(rhs, solution, *, tol, maxiter) -> None
empty(shape, dtype)
zeros(shape, dtype)
```

`stout_smear(gauge, *, nstep, rho, dir_ignore)` 是可选的规范场变换能力。`QudaBackend` 在 adapter 边界处理 PyQUDA payload；PR core 不访问 QUDA 原生对象。变换只作用于 Dirac operator 所用规范场，不改写 Laplace 基底来源。

## 公开 API

### 构造器

```text
PerambulatorGenerator(
    latt_size, gauge_field, mass, tol, maxiter,
    xi_0=1.0, nu=1.0, clover_coeff_t=0.0, clover_coeff_r=1.0,
    t_boundary=1, multigrid=None, contract_prec="<c16", *,
    backend,
    MRHS=False, rhs_batch_size=None, stream_eigenvectors=False,
    sink_ne_block_size=32, sink_time_block_size=1,
    sink_cache_limit_bytes=0, vector_scratch_limit_bytes=None,
    source_selection=None, sink_selection=None, measure=False,
    gamma5=None, gamma5_hermitian=None,
)
```

| 参数 | 含义 |
|---|---|
| `latt_size` | `(Lx,Ly,Lz,Lt)` 四个正整数；自旋 4、颜色 3。 |
| `gauge_field` | 提供 `load(key)` 的规范场句柄。`generator.load(key)` 只装载规范场；源汇句柄在每次 `calc` 按实际需要读取。 |
| `mass`, `tol`, `maxiter` | 作用量质量与求解参数。`tol` 要在 `(0,1)`，`maxiter` 为正整数。 |
| `xi_0`, `nu`, `clover_coeff_t/r`, `t_boundary`, `multigrid` | 显式 Dirac 作用量参数；`t_boundary` 为 `+1` 或 `-1`。具体后端支持情况由后端决定。 |
| `contract_prec` | 向量输入副本、投影输出和块值的复数精度，仅支持 complex64/complex128。Dirac RHS 与 solution 使用 complex128。 |
| `backend` | 必须显式提供 `PropagatorBackend`；不会自动选择 QUDA 或初始化设备。 |
| `MRHS`, `rhs_batch_size` | 控制一次求解提交的 RHS 数；向量源最多每批 4 个自旋源，点源最多每批 12 个自旋颜色源。 |
| `stream_eigenvectors` | 为真时按物理时间和模态块调用 `read_modes(key,time,start,stop)`，不常驻全部向量。 |
| `sink_ne_block_size` | 向量汇投影的模态块宽度，流式和 eager 路径都适用，默认 32。 |
| `sink_time_block_size` | 流式路径中每组遍历的汇物理时间数上限，默认 1，不得大于 `Lt`。内部仍逐物理时间投影，不会同时 stack 多个时间，也不生成 sparse full-solution gather。大于 1 时需启用流式模式。 |
| `sink_cache_limit_bytes` | 可选的流式汇向量共轭块缓存预算，默认关闭。 |
| `vector_scratch_limit_bytes` | 只约束生成器自己的流式向量 scratch，不覆盖 reader/adapter 暂存、solver、规范场或结果。 |
| `source_selection`, `sink_selection` | 独立有序 `TimeSelection` / `TimeSelectionRecord`；省略时分别使用完整物理 `Lt`。二者可不同，但 `Lt` 必须一致。 |
| `measure` | 控制是否在生成器阶段同步计时。异步后端的普通提交耗时不是设备完成时间。 |
| `gamma5`, `gamma5_hermitian` | Gamma5 矩阵和作用量 Gamma5 Hermiticity 声明。省略时从后端读取；普通自定义后端需显式传入两项才可使用内置反向。 |

构造器只配置格点、作用量、后端、资源和时间约定；实际源汇端点在每次计算时绑定，产品由端点组合决定。

### 基底与计算

```python
VectorBasis(handle, count=None, endpoint=None)
PointBasis(handle, count=None, highmode_basis=None, endpoint=None)
```

```text
PerambulatorGenerator.load(key) -> None
PerambulatorGenerator.stout_smear(nstep, rho, dir_ignore=3) -> None
PerambulatorGenerator.endpoint(basis, *, source=False) -> PropagatorEndpoint
PerambulatorGenerator.calc(t_src, *, src, sink, write=None) -> list[PropagatorResult | PropagatorSolution]
PropagatorSolution.project(*, sink, write=None) -> list[PropagatorResult]
PropagatorResult.at(sink_time) -> PropagatorBlock
PropagatorResult.reverse(sink_time=None) -> PropagatorBlock | list[PropagatorBlock]
PerambulatorGenerator.describe() -> dict
```

`VectorBasis` 表示实际有序 V 或 D 向量集合；优先读显式 `basis.endpoint.kind` 或 `handle.endpoint.kind`，没有 endpoint 时再读 `handle.dilution_source` 或 `handle.vector_kind`，普通 Laplace reader 默认为 V。已登记为 D 但缺少 dilution 来源的句柄仍按 D 处理，来源保持 unknown，不会补造。自定义 D 句柄须提供真实 D endpoint，可以用 `VectorBasis(handle, endpoint=d_endpoint)` 绑定。`count` 选择输入顺序中的前缀，省略时使用句柄提供的全部向量数。D 句柄 `.Ne` 表示实际 D 向量数量 `Nd`。

`PointBasis` 表示实际 P 点集；高模点通过该对象的 `highmode_basis` 指定。端点 kind 为 P。源和汇各自提供实际句柄、截断和身份；shape、数量或标签相同不会合并两个独立端点。

`generator.endpoint(basis, source=False)` 在已成功 `load(key)` 后读取并返回该 basis 按本次源/汇时间选择形成的实际 endpoint，不执行 Dirac 求解；pair spec 等存储定义需要 endpoint 时可显式调用。`source=True` 按 source selection 解析，默认按 sink selection 解析。`describe()` 只返回 operator、组态、时间选择和变换说明，端点身份保存在每个计算结果中。

`sink` 必须是非空、有序列表，每项为 `VectorBasis`、`PointBasis` 或 `None`。同次计算中的所有汇共用一个源及其 Dirac 求解，返回结果列表与汇列表逐项对应。`sink=None` 是 `[None]` 简写。带点端的同一源可同时求多个 V/D/P 汇，不需重复求解。

| 调用形态 | 返回项 |
|---|---|
| `sink=[VectorBasis(...)]` | `PropagatorResult`，投影到该 V/D 汇。 |
| `sink=[PointBasis(...)]` | `PropagatorResult`，采集该点汇。 |
| `sink=[basis_a, basis_b, ...]` | 每个汇一个结果，顺序不变。 |
| `sink=None` 或 `sink=[None]` | `PropagatorSolution`，保留完整未缩并解场。 |

`PropagatorResult.values` 拥有该汇的完整投影数组，`.source` / `.sink` 是实际端点，`.product` 按 `A_sink S B_source` 返回产品名。`.sink_times` 保存其第一轴对应的有序物理汇时间。`.at(t_sink)` 返回该物理时间对的 `PropagatorBlock` 视图。

### 结果、原始解场和反向

投影结果数组第一轴为汇时间槽，第二、三轴依次是 sink spin、source spin。V/D 端各有一个无颜色轴的向量索引；P 端有 point 与 color 轴，颜色数为 3。完整九类轴见[V/P/D 产品](#vpd-target)。结果数组由调用方拥有，可跨后续 `calc` 保留。

未缩并结果 `raw.values` 的布局为：

```text
V/D source: (source_index, source_spin, Lt, z, y, x, sink_spin, sink_color)
P source:   (source_index, source_spin, source_color, Lt, z, y, x, sink_spin, sink_color)
```

`raw.project(sink=[...], write=None)` 将已求解解场投影到新的实际汇列表，不重复反演；`write` 可选地同步接收投影块。须继续使用同一生成器、同一已装载组态，并且中间未执行 stout smearing；重新 load 或变换后不能用该生成器复投影旧解。初次 `calc` 请求完整未缩并解场时不能同时传 `write`。

若后端提供 Gamma5 矩阵和声明 `gamma5_hermitian=True`，`PropagatorResult.reverse(t_sink)` 内置 Gamma5 反向并返回交换时间与 endpoint 的 `PropagatorBlock`。`QudaBackend` 与 `DenseWilsonBackend` 内置此声明；普通自定义后端可在构造器显式传入 `gamma5`、`gamma5_hermitian=True`。不传时间时，它按原结果 `sink_times` 顺序返回各反向 block 列表；这些 block 有各自不同的源时间，不能作为单个反向源时间行解释。反向会交换两个 spin 轴及完整源汇端点轴、复共轭并作用 Gamma5；它不是整行转置。

### 同步块写出

给 `calc` 传 `write(index, selection, value)` 可逐块消费投影，不分配完整投影结果数组。`index` 是汇列表下标，`selection` 是该 dense 结果轴形状上的切片元组，`value` 是后端数组，使用 `contract_prec`。回调返回前必须同步消费或复制该借用块。返回列表仍保留每项的 source/sink/product/time 元数据，但 `PropagatorResult.values` 为 `None`；不能调用 `.at()` 或 `.reverse()` 读取未保留的数值。

写块模式只接受有实际基底的汇，不能与完整未缩并 `None` 汇混用。回调失败会中断本次 calc；即使回调已写出部分数据，调用方也不得登记为完整或发布。仅当 `calc` 成功返回后才能记完成。生成器不管理 writer、事务或发布；调用方应让 writer 将未完成片段保持为未发布状态。

## 轴、公式和科学约定

Dirac 内部场布局为 `(rhs,Lt,Lz,Ly,Lx,spin=4,color=3)`。向量输入按 `[time_slot,mode,z,y,x,color]`；点坐标按 `[time_slot,point,(x,y,z)]`。源向量不额外规范化或重排；点源在实际坐标写入颜色单位源。backend 求解 `D x = rhs`，不得修改 RHS。

产品按 `A_sink† S B_source` 命名，其中 S 是同一物理作用量的 Dirac 逆。数组的 spin 轴始终为 `(sink_spin,source_spin)`；P 端各自带 point/color 索引，V/D 端为单一向量轴：

| 产品 | 数组轴 |
|---|---|
| VSV / VSD / DSV / DSD | `(Nt_sink,4,4,N_vector_sink,N_vector_source)` |
| PSV / PSD | `(Nt_sink,4,4,N_point_sink,3,N_vector_source)` |
| VSP / DSP | `(Nt_sink,4,4,N_vector_sink,N_point_source,3)` |
| PSP | `(Nt_sink,4,4,N_point_sink,3,N_point_source,3)` |

D 向量标签由实际 endpoint 的 `vector_labels` 绑定到 `(dilution_block, vector_in_block)`；组号不能当成向量索引。结果不含时间差平移、空间采样权重、随机权重或关联函数归一化。源汇时间、mode/point 顺序始终沿实际输入记录。

点句柄可声明 compact `time_selection`。输入时间记录必须覆盖所需物理时间，PR 不从数组 shape 猜映射。流式 `read_modes` 参数仍是物理时间。源时间参数 `t_src` 是物理坐标且必须属于 `source_selection`；Dirac 格点、规范场及完整解场仍为 `Lt`。

## 状态、变换与异常

`load(key)` 清除先前状态并装载 gauge handle；后端算符在首次 calc 时加载。每次 calc 读取本次源和汇需要的基底/点集。`stout_smear(nstep, rho, dir_ignore=3)` 要求已成功 load，`nstep` 为非负整数、`dir_ignore` 为 0..4、`rho` 有限，且后端支持变换；成功后后端需重新装载 operator。变换的是 Dirac operator 规范场，不推断或改写 Laplace 基底使用的规范场版本。

加载或 smearing 失败会使生成器不能继续沿用旧载荷，必须重新 load。求解、输入读取或投影失败时，本次结果不能当成完整数据使用；原始异常传播。没有 silent retry 或部分结果自动补齐。一个 generator/backend 实例不支持并发或重入调用。

## 内存与输入策略

完整结果在每次 `calc` 中独立分配并由结果对象持有；内存约随请求 sink 数及其实际轴尺寸增加。`stream_eigenvectors`、汇模态/时间分块与 sink cache 只限制生成器管理的输入暂存或投影工作区；`write` 可避免完整投影输出驻留。它们不限制完整 RHS、解场、规范场、reader/adapter 暂存、solver/native 工作区或设备内存池。HighmodeBasis 使用的完整父空间物理时间片另计。`planning.production.estimate_perambulator_memory` 是调用方显式使用的逻辑估算器，不执行资源准入或推断 RSS；它按当前生命周期估计一源一汇调用，不覆盖多汇列表的合计输出或保留 raw 解场的需求，需依实际任务另行联合估算。

`measure=False` 保持后端的 stream 提交路径；提交耗时不代表设备完成时间。需要阶段诊断时用 `measure=True`。内存估算、后端范围及平台证据见[通用生产说明](../../architecture/generic_production.md)和[运行时内存说明](../runtime_memory/runtime_memory.md)。

<a id="vpd-target"></a>
## V/P/D 产品、Gamma5 与 pair IO

V 是实际 Laplace 向量，D 是已有 dilution 混合向量，P 是实际点坐标处的颜色单位源。V/D 使用同一向量源、求解与投影路线；P 使用点源和点采集路线。D 不替代完整 low/medium 父空间，不做额外归一化或估计量加权。

| 汇端 / 源端 | V | P | D |
|---|---|---|---|
| V | VSV | VSP | VSD |
| P | PSV | PSP | PSD |
| D | DSV | DSP | DSD |

`ALL_PRODUCTS` 是九类直接生产产品；`SAVED_PRODUCTS` 为六种规范保存产品 `VSV, PSV, PSP, VSD, PSD, DSD`。`VectorBasis` / `PointBasis` 的实际类型独立决定每个结果的产品，不在生成器构造时挑产品。不同源/汇 V/D 组合可在同一次 source solve 的汇列表中列出。

### Gamma5 关系

底层 `gamma5_reverse(values, product, gamma5, *, array_module=np, leading_axes=0)` 是纯数组变换：交换自旋和完整 sink/source 端点轴，不解释物理时间。`reverse_propagator(block, gamma5, *, gamma5_hermitian, array_module=np)` 面向单个 `PropagatorBlock`，同时交换物理时间与端点记录，并要求明确声明 Gamma5 Hermiticity。通常用户从已绑定后端约定的 `PropagatorResult.reverse` 使用时间感知反向。

兼容数组入口 `psv_to_vsp_dagger(psv_block, gamma5, *, backend=np)` 继续保留，但它保留前导时间轴，不交换物理时间或实际端点；需要时间感知结果时使用 `PropagatorResult.reverse`。旧导入路径 `contraction.propagator_views.psv_to_vsp_dagger` 转发到同一实现。

Gamma5 的产品关系为 VSP↔PSV、DSV↔VSD、DSP↔PSD；VSV、DSD、PSP 与自身配对。必须交换 `(sink_time,source_time)` 以及真实 sink/source endpoint；V/D 标签、point/color 轴和 high-mode projection metadata 也属于 endpoint。直接计算产品不需要依赖此对称性。

### 六类保存和九类读取

pair 生产通过 `calc` 显式处理实际源与完整汇列表，随后用 `PropagatorWriter.write` 保存各汇结果；不根据 writer 缺失覆盖自动求解。源时间行布局由脚本对每次 `calc` 的结果调用 `DatasetWriter.write(slot, result.values)` 保存。

```text
propagator_pair_spec(
    product, *, configuration, sink, source, time_pairs, gauge_id,
    dirac, gamma5, spin_basis, gamma5_hermitian, dtype="complex128",
)
```

该 builder 只接受六类保存产品，返回 `DatasetSpec(kind="perambulator_pairs")`，编码 `pr-six-pairs-v1`。首轴为 `time_pair`，每个元素是物理 `(sink_time, source_time)`，不使用旧 `source_time × sink_time` 网格。存储 normalization `unscaled` 表示 PR 没有额外缩放；配方与 D endpoint 仍保留实际 raw 幅度及实际父分组、向量标签、realization/pool 身份。

`PropagatorWriter(writer, *, array_module=np).write(block)` 将一个 `PropagatorBlock` 写入调用方注入的 writer。六类规范代表按声明坐标直接保存；VSP、DSV、DSP 会通过 Gamma5 转换到相反的规范保存代表，并交换 endpoint 和时间对。此 facade 不改写底层 dataset、dtype 或已有覆盖。

`PropagatorReader(readers, *, array_module=np).read(product, sink_time, source_time, *, sink, source)` 返回九类产品的直接或 Gamma5 派生 `PropagatorBlock`。派生读取核对完整配置、Dirac/Gamma5 约定与 endpoint 身份；D 标签、realization 和 pool 记录须精确匹配。缺少端点或时间对抛 `KeyError`，不会启动 solver。直接读值可借用 reader，Gamma5 派生值有独立数组。

`coverage(product, *, sink, source)` 返回实际端点已覆盖的物理时间对；`missing(product, time_pairs, *, sink, source)` 按输入顺序返回缺失时间对。它们只读 metadata，不读数值。新增缺失覆盖用独立 dataset；已发布数据保持不变，resume 由通用 writer 管理。重复覆盖或 context 不一致会被拒绝。

历史 source-time 行由 adapters 的 `SourceTimePropagatorReader` 显式接入，见[已有传播子的增量复用](perambulator_reuse.md)。PR 不识别 pickle、组合向量轴、路径或 D 分组。

调用方负责 DatasetWriter / DatasetReader 生命周期、publish、续算与 resume。`production.perambulator.pair_planning.plan_indexed_perambulator_pairs(index, spec)` 只通过 metadata 核对登记的实际父项、兼容分片和 Gamma5 覆盖，查询不触发求解。调用方显式确定计算范围，通过 `calc` 与 `PropagatorWriter.write` 完成生产和保存。旧 `planning.production` 与 `production.perambulator.planning.plan_indexed_perambulators` 保留四类 V/P 产品的 `source_time` 行布局，不规划 D，也不理解 `pr-six-pairs-v1`；两种布局不可互换。

### 高模端点与估计量交接

`ModePartition(parent_id, low_modes, medium_groups)` 与 `HighmodeBasis(parent_handle, partition)` 定义完整 low+medium 补空间。`PointBasis(..., highmode_basis=...)` 可在点源求解前及点汇采样前应用该投影，覆盖 PSV/VSP/PSD/DSP/PSP；源汇父空间不同时分别在两端 PointBasis 指定。父句柄必须声明 `basis_orthonormal=True`，D 输入须携带完整实际 `DilutionSource`。高模 P 端点的 `high_mode_projection` metadata 会随 pair 保存和 Gamma5 反向保留。

若已有完整父传播子，`project_highmode_block(..., partition=..., overlaps=..., auxiliary=...)` 可派生五类含 P 的 block；两侧 PSP 保留正交交叉项。D 不是完整 low/medium 投影空间的替代品。投影点在物理上记作 H，但仍使用 P endpoint 和产品编码；HSH 与 HSV/HSD 有不同覆盖需求。elemental 使用原始点坐标与既有权重约定；PR 不施加估计量权重。

普通 Meson Diagram 经 `adapters.propagator.PropagatorPairReader` 消费 pair 产品，elemental 的 D 来源与传播子实际 endpoint 必须一致。局部及高阶计权由共同 `elemental_weighted` 管理，不能从 VSD/PSD/DSD 数组本身推断无偏估计量。完整工作流见[三分区交接](three_sector_production.md)。

## 测试与支持范围

数值参考、独立多汇、V/D、稀疏时间、原始解场、块写出和高模的测试归属在 [`tests/production/perambulator/`](../../../tests/production/perambulator)；源时间行脚本与 pair 保存适配器的覆盖在 [`tests/`](../../../tests) 与 [`tests/adapters/test_perambulator_rows.py`](../../../tests/adapters/test_perambulator_rows.py)。CPU 独立数值测试不证明真实 GPU/QUDA 或 MPI 生产链已验收。H 生产的支持边界见[H 契约](point_highmode.md)。
