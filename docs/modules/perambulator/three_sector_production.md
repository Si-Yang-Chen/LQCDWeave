# V/D/H 三分区闭环

本页连接普通介子 Diagram 的生产、六类保存、九类读回与统一计权。low 使用完整确定性 V，
medium 使用实际 QR dilution D，high 使用随机空间点及完整 low+medium 父空间的补投影。
下文将投影后的点向量称为 H；API 和存储仍沿用 P 编码，以投影元数据区分 raw P。
定义、产品对照及生产路线选择见[H 契约](point_highmode.md#h-contract)。
九扇区生产与登记读回的 CPU 参考见[elemental dilution 回归](../../../tests/production/elemental/test_dilution_sectors.py)，
整图计权参考见[关联函数 dilution 回归](../../../tests/test_dilution_correlator.py)。

## 一份科学分区

low 标签及 medium 各组的父标签属于同一个实际 Laplace 基底，必须互不重复。
medium 组有 n 个父模式、m 个实际 D 向量；high 减除 n 而不是 m。

\[
U=(V_{\rm low},V_{\rm medium,parent}),\quad
Q=I-UU^\dagger,\quad N_{\rm cut}=N_{\rm low}+\sum_g n_g.
\]

U 必须正交归一。例如 low 四模、medium 五取四，则减除九维，保留的四个 D 向量不能
替代完整五维 medium 投影。输入位置和科学标签分别保存：NV 的 mode_offset/mode_groups
选择存储位置，describe.mode_groups 保留实际 parent handle.mode_labels。

```python
from lqcdweave.production.perambulator.api import ModePartition, HighmodeBasis

partition = ModePartition(parent_handle.basis_id, low_modes, medium_parent_groups)
high = HighmodeBasis(parent_handle, partition)
```

parent_handle 是已有完整父基底的句柄，不重新生产或求解传播子。已登记的
DatasetFieldHandle 从 `normalization="orthonormal"` 提供 `basis_orthonormal=True`；高级自定义
句柄须真实声明该属性、basis_id 和有序 mode_labels。只有互相正交或相同 shape 不足以定义 Q。
HighmodeBasis 选择 partition.projection_modes 的全部父向量，保留真实时间映射；两端父空间
不同时，分别将 source_high 与 sink_high 绑定到源和汇的 `PointBasis(..., highmode_basis=...)`。

## 传播子生产

在实际源汇对象的 `PointBasis(..., highmode_basis=high)` 中指定 high。V/D 的求解和投影不变；
点源在求解前构造 Q eta，点汇输出前取 eta† Q solution。只加载完整父向量用于投影，
不增加 medium 父模式的 Dirac RHS。`stream_eigenvectors=True` 且父句柄支持 `read_modes`
时，高模父向量按物理时间读取，每个点端最多保留一个完整投影时间片；其它父句柄仍 eager 加载。
此完整单时间片不计入普通 V/D sink cache 限额，规划时须另留其内存。

| 产品 | 实际数值 |
|---|---|
| VSV / VSD / DSV / DSD | 原来的 V/D 投影 |
| PSV / PSD | eta† Q_sink S V/D |
| VSP / DSP | V/D† S Q_source eta |
| PSP | eta† Q_sink S Q_source eta |

V 端点须完整对应 partition.low_modes；D handle 的 dilution_source 须对应全部 medium
父分组及实际 D 轴。DatasetFieldHandle 的 V 标签和实际 D 来源可自动形成生产端点，
无需重新手写随机来源记录；自定义已有 endpoint 仍保留显式来源。两端各绑定一种 V 或 D，
以下列出生产六个保存代表时使用的源/汇组合。每个 `calc` 将所需汇放进同一个有序
`sink` 列表，使它们共享向量源求解；PSP 使用点源与点汇。

| 源向量 / 汇向量 | 请求产品 |
|---|---|
| V / V | VSV、PSV、PSP |
| D / V | VSD、PSD |
| D / D | DSD |

PSP 使用 `src=PointBasis(...)` 与 `sink=[PointBasis(...)]`；其余行使用对应的
`VectorBasis` 源以及 V/D/点汇列表。实际作用量、边界、精度、点集和源汇时间由调用方选择。
每项 `PropagatorResult` 拥有自己的 `.values`，并通过 `.source` / `.sink` 暴露实际端点；
`.at(t_sink)` 可取得一个物理时间对的 `PropagatorBlock`。
固定点池且只需 H 交叉块时，也可直接解 H 源，共用解场投影到 V/D，再按 Gamma5 保存
HSV/HSD；无需为了此目的先补齐完整父传播子。见[H 生产路线](point_highmode.md)。

已有完整父传播子时也可调用 PR 的 `project_highmode_block`：overlaps 的 sink/source 是
U(point)，布局 `(parent_mode,point,color)`；auxiliary 的 sink 是 U† S B、source 是 A† S U，
双点端再提供 cross=U† S U。辅助块必须绑定完整实际 U 和同一物理时间对。PSP 的结果
保留正的双侧交叉项。此派生路线需要已有完整父传播子；原生 high 点源路线不需要这些额外解。

## 六类保存、九类读取与补覆盖

使用 `PropagatorResult.source` / `.sink` 的实际端点建立 `propagator_pair_spec`，再组合原
DatasetWriter、PropagatorWriter。P metadata.high_mode_projection 保存完整 partition；
父基底实际 ID 同时进入 spec.parents。不能用 raw P 的端点记录保存已经投影的值。
原始点和 high 点在读回、Gamma5 反向转换、缓存和计划身份中保持不同。

保存仍为 VSV、PSV、PSP、VSD、PSD、DSD。PropagatorReader 从六类代表恢复九类视图，
反向请求交换物理时间、端点和完整投影来源，不能补猜未保存的时间对。
物理上对应 VSV、HSV、HSH、VSD、HSD、DSD。六类是产品代表数，不等于每类时间对
都可减半。相同两端定义的 VSV/DSD/HSH 才能按 Gamma5 保存时间三角；HSV/HSD/VSD
保留所需完整有向覆盖。只完成 HSV/HSD 的任务不构成包含 HSH 的九类完整输入。

需要核对已有覆盖时，可只读调用 `plan_indexed_perambulator_pairs(index, pair_spec)`。
它检查实际父项和 metadata，合并兼容分片及合法 Gamma5 覆盖，不读数值或启动求解。
计算范围由脚本显式给定；同源多汇一次计算，然后分别写入各自的 pair dataset：

```python
from lqcdweave.production.perambulator.api import PropagatorWriter

# dataset_writers 与 sink_bases 一一对应，spec 已声明实际端点和物理时间对。
outputs = [PropagatorWriter(writer, array_module=generator.xp)
           for writer in dataset_writers]
generator.load(cfg)
for source_time in source_times:
    results = generator.calc(source_time, src=source_basis, sink=sink_bases)
    for output, result in zip(outputs, results):
        for sink_time in sink_times:
            output.write(result.at(sink_time))
for writer in dataset_writers:
    writer.publish()
```

写入只消费已有结果，不依据缺失覆盖自动安排求解。spec 中的端点、高模投影、规范场、
精度及 Dirac 定义须与实际计算一致；作用量、质量、时间边界及求解参数由脚本如实记录。
调用方管理 reader/writer 生命周期和发布，通用存储恢复能力不等于自动补算。
旧 plan_indexed_perambulators 保留历史四类 V/P source_time 布局；
pair 存储使用六类物理 time_pair 布局，两种结果不能直接互换。

## Elemental 与 correlator

ElementalGenerator 导入实际 V/D 及原始点坐标，`blocks` 显式包含全部九扇区。
Elemental 的 P 腿仍用原始 eta 空间核；Q 已在传播子两端施加，不对 elemental 再施加 Q。
这样各项展开连接完整 low、medium 与 high 空间，不重复计入 low/medium。

```python
from lqcdweave.contracts.elemental_sectors import ELEMENTAL_SECTORS
from lqcdweave.production.elemental.api import ElementalGenerator
from lqcdweave.contraction.binding.api import Meson
from lqcdweave.contraction.correlator import calculate

generator = ElementalGenerator(
    gauge, None, momentum_dict=momentum_dict, operator_table=operator_table,
    vector_bases={"V": v_handle, "D": d_handle}, points=point_handle,
    blocks=ELEMENTAL_SECTORS, elemental_weighted=elemental_weighted,
    lattice_size=lattice,
).load(cfg)
# 保存当前块和实际来源后，读回 elemental_reader。
vertex = Meson(elemental_reader, operator, requested_sectors=ELEMENTAL_SECTORS)
result = calculate(
    diagram, vertex_map=vertices, propagator_map=propagators,
    elemental_weighted=elemental_weighted, groups_independent=True,
)
```

propagators 使用 PropagatorPairReader 包装实际六类 reader。elemental 数据须保存并传递
实际 V/D/P 标签、D 来源、点池与结构。DatasetElemental 的 sampling_groups 和
sampling_pool_ids 从保存的实际点池声明提供，不能从 seed 推断；完整例子及
[点采样参考](../../../.agents/skills/lqcdweave-correlator/references/sampling-current.md)说明接线。
只有实际 D 与随机 P 的抽样相互独立时才能声明 groups_independent=True；共享池保留共享身份。

两端流程仍只有同一个 elemental_weighted 差别：False 将统计补偿放在整图，True 在
elemental 放置局部权重、整图补余权。D 同池按实际相等/不同混合列用 H_r，P 同池按实际点
重合关系计权，不能把每个端点的一阶系数直接相乘。多 current 和完全对角证明沿用
[D 计权契约](../../../.agents/skills/lqcdweave-elemental/references/dilution.md)。

## 验收与范围

[完整参考测试](../../../tests/test_three_sector_closed_loop.py)使用 SU(3) 规范场、实际 Wilson 求解、
九个 elemental Dataset 与六个 pair Dataset。一般路径/非零动量的三取二平均、随机 high
二取一平均，以及二者同时抽样的 36 个联合方案平均，分别对照独立完整空间 Wilson 迹；
raw/preweighted 和显式 pair 写出一致，indexed 查询核对保存前后的覆盖。原 weakfield 三取二、五取四和同时间双流验收继续保留。
教学示例验证单份随机实现的接线和独立矩阵迹，不把一份随机实现称为完整空间结果。

本闭环范围为 CPU 普通 Meson Diagram。自动优化重要性分组、D 重子/专用矩阵路线、GPU/QUDA/MPI
和生产规模仍各有独立验收范围，不由本页的闭环测试推导。
设备小样本验证不能代替完整九扇区及生产规模验收。
