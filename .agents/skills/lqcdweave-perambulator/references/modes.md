# 模态输入、流式投影与内存

本页只在任务需要显式资源控制、流式输入、原始解场或同步写出时读取。基础 VSV 脚本直接使用 [主 skill](../SKILL.md) 的常规入口。

## 独立源汇与多个汇

源和每个汇都在 `calc` 中给出。`VectorBasis(handle, count=...)` 选择实际有序向量前缀；省略 count 使用句柄提供的全部数量。源汇可以独立截断，也可以在同一 sink 列表中放入多个不同身份或表示的汇。一次 Dirac 求解供所有汇使用，结果按列表顺序返回：

```python
from lqcdweave.production.perambulator.api import VectorBasis, PointBasis

results = generator.calc(
    source_time,
    src=VectorBasis(source_handle, count=N_source),
    sink=[
        VectorBasis(vector_sink_a, count=N_a),
        VectorBasis(vector_sink_b, count=N_b),
        PointBasis(point_sink, count=N_point),
    ],
)
```

不同句柄即使 shape 或数量相同也保留各自 endpoint。D 端应使用携带真实 D endpoint 的句柄；普通 Laplace 句柄按 V 识别。各结果的 `.source` / `.sink` 保存实际端点，`.product` 给出 `A_sink S B_source` 产品名。

## 流式向量读取

`stream_eigenvectors=True` 让生成器按实际物理时间和向量块调用 `read_modes(key, time, start, stop)`，不要求句柄一次返回完整向量数组。时间与点集 compact 映射仍由句柄的 `time_selection` 声明。

- `sink_ne_block_size` 控制向量汇投影块宽度，在 eager 与流式路径均适用。
- `sink_time_block_size` 是流式路径中每组遍历的汇物理时间数上限，内部仍逐物理时间投影；不会同时 stack 多个时间，也不生成 sparse full-solution gather。大于 1 时需启用流式模式。
- `sink_cache_limit_bytes` 限制可保留的汇向量共轭块缓存。
- `vector_scratch_limit_bytes` 只限制生成器自己的向量 scratch。
- `MRHS` / `rhs_batch_size` 控制后端同时求解的右端项数。

这些参数只覆盖生成器明示拥有的输入暂存或缓存，不限制完整 RHS/解场、求解器工作区、规范场、读取器暂存、输出或设备内存池。`planning.production.estimate_perambulator_memory` 当前只估算一源一汇调用；多汇列表或保留 raw 解场需要按整个任务联合估算。完整资源模型与独立估算器见 [PR L2](../../../../docs/modules/perambulator/perambulator.md#内存与输入策略)。

## 保留未缩并解场

把 sink 设为 `None` 可保留该源全部 spin/color 解场，返回 `PropagatorSolution`：

```python
from lqcdweave.production.perambulator.api import PointBasis, VectorBasis

source = VectorBasis(source_handle)
[raw] = generator.calc(source_time, src=source, sink=None)
projected = raw.project(sink=[
    VectorBasis(vector_sink_handle), PointBasis(point_sink_handle),
])
```

`raw.values` 的轴为 `(source_index, source_spin[, source_color], Lt, z, y, x, sink_spin, sink_color)`；点源包含 `source_color`。`project` 复用已有解，不再反演，并保持汇列表顺序。只能在原生成器仍装载同一组态且未执行 smearing 时调用；重新 load 或变换后不能复投影旧结果。

## 同步块写出

当完整投影数组不适合驻留内存时，给 `calc` 传 `write(index, selection, value)`。它在计算期间同步接收借用的汇块；`index` 是 sink 列表下标，`selection` 是该结果 dense 形状上的切片，`value` 位于后端并使用 `contract_prec`。回调返回前必须消费或复制块。传入 `write` 时返回结果的 `values` 为 `None`；此调用不能同时请求 `None` 汇/raw 解场。

只有 `calc` 成功返回后，调用方才能登记整行完成或发布结果。回调抛错时停止本次计算，调用方不得发布部分写出的数据；回调需要自行处理已写片段。生成器不自动管理 writer、事务或发布。具体轴、块范围和失败约定见 [PR L2](../../../../docs/modules/perambulator/perambulator.md#同步块写出)。
