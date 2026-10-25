# Diagram 计算与已登记数据规划

`contraction.diagram_measurement` 目前只提供 `plan_indexed_diagram_measurement` 元数据规划。规划用真实 DatasetIndex ID 核对给定 `Diagram` / `DiagramBindings` 的输入视图、端点和默认全物理时间或显式共同槽覆盖；不会调用数值核、创建输出或运行持久化任务。

## 规划已有输入

```python
from sympy import Symbol
from lqcdweave.contraction.diagram_measurement import plan_indexed_diagram_measurement

source_t, sink_t = Symbol("source_t"), Symbol("sink_t")
plan = plan_indexed_diagram_measurement(
    index, diagram, bindings=bindings,
    source_time_symbol=source_t, sink_time_symbol=sink_t,
    source_policy="require_known",
)
```

`bindings` 是 `DiagramBindings`，传播子映射绑定 `DatasetPropagator`，顶点绑定 `Meson` 或 `Baryon` 及其对应的 elemental reader。规划结果包括 `requests`、`graph`、`measurement_spec`、`source_times`、`sink_times` 、`input_roles` 和 `time_selection`（全时间未标注时为 None）。它核验已登记父 ID、轴坐标、真实传播子端点和顶点端点；稀疏 P/P 结构可读取登记的点坐标以确认布局。不会读取传播子或 elemental 数值块。

输入缺失时，由调用方决定如何生产所需数据。实际收缩可用 `contraction.correlator.calculate`，或按需要组合公开的 E/P/X/R API 与 `calc_diagram_eval`。这些路线保留符号时间到实际端点的映射；不能以顶点锚点替代每条夸克线的真实端点。持久输出需要的轴、通道、recipe 和覆盖由调用者依据任务选择定义，并通过通用 `DatasetSpec`、`DatasetWriter` 和 storage/runtime 服务保存；本模块不提供固定测量 writer、receipt 编排或 resume 流程。

## 科学与数据约束

结果应保留 diagram 邻接关系、顶点和传播子顺序、真实时间符号、每个算符项的路径和动量、复系数、current 配方、源端 dagger、味道/端点标签、实际 sector IDs、基底视图、Dirac 定义、边界、相位和归一化。重子 source 的原始腿轴按 E 的 `field_order` 映射。缺失元数据不能通过 shape 或文件名推断。

sampling pool 需要显式声明总体、样本大小、组间独立性和适用的补偿方式。不同点池分别绑定真实 point dataset；P/P sparse support 保留左端行槽、右端 ID 和有序标签。普通 full-time 输入协议不因空间点采样而变成时间稀疏测量。

时间范围由当前坐标需求决定，默认全物理源/汇时间。open 路径按逐项端点判定是否越界；不得用零值伪造未定义路径，也不得把周期路径的取模规则套到费米子时间边界。精度、存储顺序与实际坐标均由输入 Spec 明确声明。

## Registered temporal VP point-path diagrams

Temporal VP point-path elemental 可用 `plan_indexed_point_elementals` 的 temporal path schema 规划；生产计算使用对应数值 kernel，并由脚本调用通用 `DatasetWriter.write` 保存已有结果。Diagram 端仍由 `DiagramBindings` 的真实传播子/顶点视图和 `calc_diagram_eval` 或 `calculate` 消费。时间路径的每个跃迁必须保留方向、起点、终点和 open/periodic 规则；`[6]` 与 `[7]` 是方向相反的实际分量，不是可互换标签。

独立点图 oracle 和 endpoint 测试分别检查完整时间格点、实际 PSP 端点、current crossing 与 source dagger。空间点有限样本估计另由纯公式和 in-memory sampling 测试覆盖。上述 CPU 合成数据仅验证所列路线，不声称任意估计量、生产规模或其他后端已验收。

P/P 时间路径的数值参考必须将原始矩阵行放在路径锚点 `t`，列放在实际终点
`t + dt`，并使用这两个时间片各自的实际点坐标与有序标签。普通 V/V 时间链也遵守
这一规则；source dagger 对整个矩阵取伴随，连同时间端点和左右点集一起交换。
例如两个原始路径矩阵 `K0`、`K1` 的闭合参考为 `Tr(K0 S K1 S)`，其中
`S = D_W^-1`，各 `K` 包含所选基底投影、自旋和规范输运；第二个顶点为源端时，
将 `K1` 换为其自旋伴随并共轭算符系数。不能只给传播子另一端加 `dt`，也不能因
P/P 路径净时间位移为零而省去另一条 V/V 时间链的位移。
`test_temporal_pp_path_diagram.py` 同时以投影块收缩和完整时空矩阵迹核对正向、
反向与跨边界路径、open 净零内部跃迁、双 P/P 源端伴随及全部源汇时间对。

## 可选：同一实际选择的 compact 输入

所有已消费 propagator 源/汇轴和 elemental/point 时间轴须显式绑定同一 TimeSelectionRecord，包括物理 Lt 与有序 times。只看 Nt 相同不能证明兼容。此时 source_times/sink_times 返回 0..Nt-1 槽，measurement_spec 的 source_time/sink_time 也是槽，并在既有 conventions.time_coordinates 内保存完整物理映射；configuration.lattice[3] 保持 Lt，time_coverage 为 selected_physical。

该两轴助手仍只读规划，不新增 writer 或 runner。按 plan.source_times 遍历源槽，用原生 calculate 计算汇槽行，同步交给已有 DatasetWriter；writer.missing 表示当前 cfg 文件中待写的首轴槽。默认持久格式和 `.state` 边界见[数据保存说明](../storage/data_storage.md)，高级恢复保证按当前实现与验收核对。共同映射不一致、缺少声明或两端非对称时明确拒绝，使用支持显式端点绑定的原生 calculate 处理相应任务。E 的实际 mode/point 父身份与端点审计仍执行，不以时间映射代替。

数值证据 tests/test_sparse_persisted_pipeline.py 覆盖实际 compact Laplace/elemental/VSV 产物，经此助手规划、原生逐行收缩和 generic writer 后，再读取四个槽值对照完整物理空间 Wilson/Laplace 独立参考。旧中断恢复用例只验证当时的存储实现，不作为当前 NPY/HDF5 高级恢复验收。普通结果保存/物理 dt 重建另经 save_correlator/load_correlator 核验。当前为小型 CPU 合成数据；一般非对称持久助手、生产规模、GPU/MPI 不在此验收内。
