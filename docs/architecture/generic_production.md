# 通用生产计划、写出与脚本边界

EV 与 PR 的数值 API 由用户脚本按任务组合加载、计算和保存；新数据默认每 cfg 一个完整 NPY，多字段数据使用 HDF5，并由共享根 `dataset.json` 登记。planning 提供纯描述/只读预览，普通 storage/runtime 提供可组合服务，不代表固定产品生产入口。模块交接见 [production L0/L1](production-l0-l1.md)，数值细节见 [EV L2](../modules/eigenvector/eigenvector.md) 和 [PR L2](../modules/perambulator/perambulator.md)；具体格式见[数据保存说明](../modules/storage/data_storage.md)。

## 输入与调用顺序

EV 脚本默认遍历所有物理时间 `range(Lt)`；明确稀疏化时从同 cfg 的选择记录读取实际有序时间，并按记录槽序保存 Nt 个逻辑槽。规范场与求解仍使用物理 `Lt`。完整脚本见[EV 生产脚本指南](../../.agents/skills/lqcdweave-eigenvector/references/production-script.md)（指南示例使用 NPY）。

PR 脚本读取实际规范场、基底/点集和 Dirac 后端，配置 `PerambulatorGenerator` 后按所需物理源时间调用 `calc`，显式传入源与有序汇列表。完整结果由 `PropagatorResult` 持有；同步 `write` 回调收到的块需在回调内消费或复制。物理格点保持 `Lt`；显式 `source_selection/sink_selection` 控制所选源求解与汇投影，返回紧凑汇槽；带映射的输入可仅保留所需物理片。具体科学参数、输入轴、产品和缓冲生命周期见 [PR L2](../modules/perambulator/perambulator.md)。

## Laplace 入口

用户 EV 脚本通过 `LaplaceEigenvectorGenerator.load(key)` 加载 cfg，再对全时间默认范围 `range(Lt)` 调用 `calc(t)`，将每个时间片的 basis 和 eigenvalues 按既有文件格式保存（指南示例使用 NPY）。明确要求稀疏化时，先从同 cfg 选择文件读取 `TimeSelectionRecord`，按其中 `times` 对应的物理时间计算，并按原槽顺序保存 Nt 行；规范场和 Laplace 求解仍使用物理 `Lt`。完整脚本见[EV 生产脚本指南](../../.agents/skills/lqcdweave-eigenvector/references/production-script.md)。生成器不自动发布产物、恢复缺行或做资源准入。

`lqcdweave.production.eigenvector.planning.plan_indexed_laplace_basis(index, gauge_id, *, Ne=2, tolerance=1e-11, data_ids=None, source_policy="allow_unknown", selection=None)` 保留为只读预览，不执行求解或写出。基底和本征值来自同一次 `calc`，由脚本分别交给对应 `DatasetWriter.write`。

### Laplace 生产内存模型

`planning.laplace.estimate_laplace_memory` 保留为独立的纯逻辑估算器，只读取已登记 gauge metadata 与 Ne，不读取 payload，也不触发 流程、脚本预算准入或自动恢复。它分别估算 host/device 逻辑峰值，覆盖完整规范场、loader 原始数组和独立持有的 complex128 链接、每个物理时间的本征向量/本征值、已知矩阵自由 Laplacian 临时量、所选 Nt 本征值缓存（默认 Nt=Lt）、host 输出行和有限值 mask；完整 Lt DatasetSpec 不等于整个输出数据集同时驻留 RAM，EV basis row 为 `(Ne,Lz,Ly,Lx,3)`，eigenvalue row 为 `(Ne,)`。估算只供显式调用方参考，不约束 EV load/calc 脚本或 solver 原生工作区，也不是 RSS/allocator 的硬上限。

## Perambulator 入口

`PerambulatorGenerator` 是数值计算入口，不探测路径、QIO 或原生 QUDA 对象。脚本提供输入读取器和显式 `PropagatorBackend`，每次调用以 `VectorBasis` / `PointBasis` 显式指定源和有序汇；产品由实际端点组合确定。同一源的多汇共享求解。返回数组归 `PropagatorResult` 持有，可由调用方保存；若通过 `write(index, selection, value)` 同步流式写出，回调内的块在返回后失效。

完整时间生产以物理 `range(Lt)` 为默认源时间范围。明确稀疏化时，先由 `storage.time_selection.load_time_selection(path, Lt=Lt)` 读取同 cfg 的记录，按 `selection.times` 所列物理时间调用生成器，并按同一槽序保存结果。不得把逻辑 `Nt` 当成格点 `Lt`；显式 `sink_selection` 使生成器只投影/提取所选汇时间并按记录顺序输出，不需事后裁剪。选择记录的生产、读取与身份见[时间选择文档](../modules/temporal_sampling/select_t.md)。

`plan_indexed_perambulators` 保留为只读输入兼容性与覆盖预览，不调用求解器或写数据。`planning.production.estimate_perambulator_memory` 是独立逻辑估算器，当前按一源一汇调用估计；多汇列表和保留 raw 解场应按实际任务联合估算。显式调用结果只供用户规划，不自动准入或约束脚本资源。通用 adapters、DatasetWriter、storage/runtime 仍可按调用方需要组合；它们不会自动生成 PR 输入、发布数值结果或恢复缺失计算。

## 脚本组合计算和保存

独立基础产物不提供计算并写出的库入口。源时间行布局的传播子脚本如下；
`output_spec` 由 `planning.production.propagator_dataset_document` 绑定实际作用量、父项、
源/汇有序时间及 dtype；`generator` 已配置同一实际输入与时间映射，`src` 和 `sinks`
分别为实际源基底和有序汇基底列表：

```python
from lqcdweave.storage.dataset import DatasetWriter

generator.load(configuration)
with DatasetWriter(output_directory, output_spec, provenance=provenance) as writer:
    for slot, t_src in enumerate(source_times):
        [result] = generator.calc(t_src, src=src, sink=[sink_basis])
        writer.write(slot, result.values)
        del result
    data_id = writer.publish()
```

多汇共享求解只调用一次 `calc(t_src, src=src, sink=sinks)`，每项交给对应 writer。
六类 pair 布局使用 `PropagatorWriter.write(result.at(t_sink))`；它只保存已有结果，
不会安排求解。两种布局不能互换。完整 pair 示例见[传播子 skill](../../.agents/skills/lqcdweave-perambulator/references/dilution.md)。

Reader 生命周期覆盖计算期间；handle 不拥有 reader。PR 完整结果由结果对象持有，
数值 `write` 回调的块则借用生成器缓冲，须在返回前同步消费或复制。GPU 结果由脚本
显式转为主机数组再交给 host writer；保存不会自动选择后端或隐式求解。
EV 的两份输出保留同次求解的配对，不重解一侧来拼接另一侧。
生成器与保存服务的组合参考见[Wilson CPU 回归](../../tests/test_wilson_pipeline.py)及其测试夹具；这些是测试接线，不是安装包的生产 runner。

## 保存、恢复与验收范围

结果保存和读取由用户脚本按明确的科学字段组合通用存储服务；新数据遵循本页开头的默认格式。PerambulatorGenerator 本身不创建、发布或恢复 dataset。`plan_indexed_perambulators` 和内存估算器只提供只读决定，不执行数值生产。高级恢复行为按当前 storage/runtime 实现和验收核对。

EV 测试覆盖只读预览、Wilson 参考脚本和逻辑内存估算。PR 数值、适配器、planning 与脚本交接测试各自验证对应层，不将先前独立持久 runner 的历史测试作为当前接口验收；CPU/interface 验收不证明 EV 脚本或 PR 数值 API 的 GPU/QUDA/MPI、真实系综、共享集群存储、整体超内存或硬件吞吐。

## 进阶：显式槽契约

初学用法保持全时间；显式稀疏时，EV/elemental/point builder 接受 selection，PR builder
接受 source_selection/sink_selection。输出物理格点不变，时间轴为 range(Nt)，
已有 conventions 中保存每端实际有序 `{Lt,times}`。EV/PR 脚本用物理时间调用数值
生产器，以对应槽号写入同一 cfg 文件；输入覆盖和原始选择必须与 writer 一致。简单数组默认使用整块 NPY，多字段结果用 HDF5 并由共享根 `dataset.json` 登记，见[数据保存说明](../modules/storage/data_storage.md)。

`DatasetFieldHandle`、`DatasetPointHandle` 通过 time_selection 和物理 read_modes/read_points
将稀疏生产输入映射到槽；同一 reader 的数据数组保持 compact，不能把物理坐标直接当数组下标。
详见[从生产到结果保存](../modules/temporal_sampling/temporal_production.md)，其中分别说明 NPY 导入、绝对源汇槽、
循环差槽兼容、跨时间额外端点、保存时可选平均和资源验收边界。

<a id="unregistered-target"></a>
## 条件参考：设备目标尚未登记

仅在目标尚无可解析的硬件档案时读取此节；普通生产调用不额外插入一个内存预检步骤。
这里“登记”指现有 M.api 的 `ExecutionTarget` 档案，而不是基底/规范场的 Dataset 登记。
已提供且可解析的 `target_hardware` 名称可直接沿用；存在 CostProfile 文件本身不能证明目标预算已登记。

现有 `load_execution_target(name, directories=None)` 只读档案，不探测硬件；默认查找
`$LQCDWEAVE_PROFILE_DIR/targets`（未设置时为 `~/.cache/lqcdweave/profiles/targets`）及包内 targets。
仅明确的 not-found 表示没有匹配目标；格式损坏、别名歧义和不相容模型不能当作未登记后覆盖。
`register_execution_target(target, directory=None)` 保存或显式更新实际 `ExecutionTarget`，
不自动发现设备、不启动计算。预算、backend/runtime、软件与 fingerprint 应按实际目标填写，不能拿规划机的空闲容量冒充远端容量。
完整字段及模型覆盖见[性能 L2](../modules/performance/performance.md)。登记与校准是两件事，未校准回退模型不能说成实测。

确需为这个未登记目标准备生产规模时，选择实际生产器已有的纯估算器：

- Laplace：`planning.laplace.estimate_laplace_memory(gauge_spec, ne, *, target, ...)`
- Perambulator：`planning.production.estimate_perambulator_memory(...)`
- 介子/投影/有向流/点端：同模块的 `estimate_elemental_memory`、`estimate_elemental_projection_memory`、`estimate_directed_current_memory`、`estimate_point_elemental_memory`
- 重子 VVV：`planning.baryon.estimate_baryon_vvv_memory(...)`

传入实际 DatasetSpec、父 ID、所选视图及配置，各函数的输入和支持域不同；精确签名见
[EV](../modules/eigenvector/eigenvector.md)、[PR](../modules/perambulator/perambulator.md)、[EL](../modules/elemental/elemental_production.md)，不引入统一 shape-only 估算器。
例如已有真实 gauge_spec、Ne 与已知额外 reserve 时：

```python
from lqcdweave.planning.laplace import estimate_laplace_memory
host, device = estimate_laplace_memory(
    gauge_spec, Ne, target="cpu",
    workspace_reserve_bytes=solver_workspace_bytes,
    host_staging_bytes=reader_staging_bytes,
)
```

返回 `(host_estimate, device_estimate)` 是逻辑数组模型，不是 RSS、allocator、QUDA/BLAS/求解器隐藏工作区的硬上限。
reserve 默认 0 只表示调用方未提供额外预算，不表示已验证零开销。分开报告常驻输入、复制/共轭、计算块、输出、缓存及未知工作区。
默认全时间，不自动减少 Lt、Ne 或精度。目标已登记也不能跳过现有[运行期资源检查](../modules/runtime_memory/runtime_memory.md)；
这里的条件仅控制文档读取及额外人工准备，不改变执行安全条件。
随机模态权重与设备登记无关，只在明确采用该估计量时读[模态稀释](../../.agents/skills/lqcdweave-elemental/references/meson.md#mode-sampling)。
