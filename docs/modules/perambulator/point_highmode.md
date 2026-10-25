# 实际点集、Overlap 与高模传播子

PR 已支持 V/P/D 九类直接生产、Gamma5 反向转换及六类物理时间对保存/九类读取，见[PR V/P/D 统一模块](perambulator.md#vpd-target)。
PR 通过 `PointBasis(..., highmode_basis=...)` 统一生产 V/D/P 的 high 点端块；`project_highmode_block` 可从完整父传播子派生 PSD/DSP 等块。本页以下 Overlap planner 和数组接口仍描述 V/P 路线。普通 V/D/P Meson Diagram 的随机估计量按[实际 dilution 来源](../../../.agents/skills/lqcdweave-elemental/references/dilution.md)接通，high 减除维度是完整 low+medium 父空间。

完整生产、保存、indexed pair 补覆盖及消费步骤见[V/D/H 三分区闭环](three_sector_production.md)。

<a id="h-contract"></a>
## H 的定义与生产路线

H 是投影后的实际点向量：`H(t)=Q(t) eta(t)`，其中 `Q=I-U U†`，U 包含完整 low 和
medium 父空间的正交归一向量。H 的点标签、顺序和颜色来自实际 P；H 一般既非单位范数，
也非彼此正交。不能重新归一化 H，或用抽取的 D 列替代完整父空间来定义 Q。
改变每组 D 的选取列数不改变 Q；改变父模式集合则改变 H 的科学身份。

H 是物理名称，API 仍使用 `PointBasis` 和 P endpoint / 产品名；不新增 `"H"` endpoint kind、
`"HSV"` 传播子产品或 `h2v` elemental block。投影由 P 端点的
`metadata.high_mode_projection` 保存，目录名不能替代该记录。

| 物理产品 | API/schema 产品 | 实际数值 |
|---|---|---|
| HSV / HSD | PSV / PSD | eta† Q_sink S V/D |
| VSH / DSH | VSP / DSP | V/D† S Q_source eta |
| HSH | PSP，两端均有投影记录 | eta† Q_sink S Q_source eta |

规划时根据已有输入、点池是否固定、所需产品和未来复用选择路线：

| 路线 | 必要输入与成本 | 适用情形 |
|---|---|---|
| 直接 H 源 | 完整 U 向量和实际 P；求解 S(Q eta)，全颜色/自旋每源时间为 12 Np 个 RHS | 固定点池，多个向量汇共用同一解 |
| V/D 源加 H 汇 | 已有或新求解的完整解场，输出前取 eta† Q solution | 正在生产 V/D 源时增加 H 汇投影，共用原 RHS |
| 紧凑块事后派生 | raw 点块、U(point) 和对应完整父传播子辅助块 | 父传播子已存在，或多种后续研究值得保留它 |

只有 U 向量就足以在求解前构造 Q eta，或对尚在内存中的完整解场投影。若已只保存
紧凑点块，计算 `S Q eta = S eta - S U (U† eta)` 还需要相应 S U 信息；不能从
V/D 的少量投影恢复它。缺少辅助块时，应比较直接 H 源成本，不能默认补产完整父传播子。
源/汇投影、临时数组、IO 与 Dirac 求解分别计时；多个产品共用的 RHS 只计一次。

通用生成器每个向量端绑定一种 V 或 D。共享解场到多类汇时，保存结果仍须保留真实 V/D 端点及其来源，不能把临时联合汇当作新的科学基底。

Elemental 仍使用原始点核和 P 扇区，Q 在传播子端施加一次；点池与 D 池的统计计权仍由
同一个 `elemental_weighted` 管理。H 的父空间维数不替代点抽样的包含概率。
保存、Gamma5 恢复与半三角条件见[存储约定](../storage/data_storage.md#h-storage)。

从已有完整父传播子派生 high 点端时，`project_highmode_block` 使用批量矩阵乘法完成
单侧减除；PSP 的交叉项分成两次矩阵乘法，并按实际源汇点数、父模态数选择乘法顺序。
这避免对每对点颜色重复遍历两个父模态指标，支持两端不同父空间和点数，
保持完整 low+medium 减除、原有精度及输入只读语义。NumPy/CuPy 共用此路径，无需新增选项。

空间采样、Overlap 和高模投影提供数值接口与只读规划/验证服务。用户脚本根据实际输入组合加载、计算和保存；文件格式、后端与是否登记产物由调用方选择。默认科学时间域仍为完整物理 `Lt`，时间稀疏化必须读取同 cfg 的实际选择记录，并另行保存槽与物理时间的映射。

## 空间点集与来源

空间点集唯一通过 `sample_spatial_points(SpatialSamplingDesign(...), seed=..., selection=...)` 生成，返回 `SpatialPointSet`；所有坐标均采用 `(time,point,xyz)`，`coordinates` 是只读 `int64`。已有坐标可构造 `SpatialPointSet(lattice_size=lattice, times=tuple(range(Lt)), coordinates=coords)`，再通过 `SpatialPointHandle(points)` 或 `SpatialPointHandle.from_npy(path, lattice_size=lattice, cfg=cfg)` 交给 PR，见[用法](../spatial_sampling/spatial_sampling.md)。

`sample_spatial_points(design, *, seed=None, selection=None)` 返回实际坐标和顺序，不读写文件。sampler 允许 `seed=None`；seed、设计或点数只描述抽样意图，不能代替保存的坐标，也不能证明产物来源已知。需要查看产物定义与已有覆盖时可调用只读的 `plan_indexed_spatial_points(index, design, *, configuration, seed, data_ids=None, source_policy="allow_unknown")`；planner 不抽样、不保存。`source_policy="require_known"` 检查已登记实际来源；fresh 或已有 unknown 来源都不能靠 design/seed 变成 known。

空间点集由脚本显式调用 `sample_spatial_points` 后使用所选保存接口写入；runtime 不内置点集计算与保存的组合生产器。

已登记输入的来源副本由 DatasetIndex 按实际 ID 核对；来源声明是登记信息，不是真实性认证，也不等于科学验证通过。通用索引及来源策略见[产物生命周期](../storage/dataset_lifecycle.md)。

## Overlap

`production.overlap_planning.plan_indexed_overlap(index, basis_id, points_id, *, usedNe=None, usedNp=None, data_ids=None, source_policy="allow_unknown")` 核验已登记的 basis/points、实际坐标与请求前缀，并预览兼容性和覆盖；检查会读取请求涉及的数据行，但不计算 overlap 或写文件。

数值内核 `production.eigenvector.api.calc_overlap_row(vectors, coordinates)` 计算单个物理时间行。对 `explicit_basis` 或 `laplace_basis` 与 `spatial_points`，输出布局为 `(mode, point, color)`，绑定两端实际有序标签；`M[i,p,c] = xi_i(x_p,c)`，表示 `eta_dagger_xi`，不额外共轭、不乘体积或采样补偿。dtype 保持基底的 complex64/complex128。`get_overlap_matrix` 接受 `(Nt,Np,3)` 的空间点坐标，是内存中的全时间便捷计算，不提供存储或恢复。

脚本读取实际 basis 和点集，按各自的物理时间坐标配对输入，调用行内核，再由脚本选择写出方式。纯 DatasetSpec builder 位于 `planning.overlap`；它不会自行发现输入或生成数据。

## 直接 VSP

`calc(t_src, src=PointBasis(...), sink=[VectorBasis(...)])` 直接生产 VSP 或 DSP，取决于实际汇向量 endpoint；同一汇列表还可加入点汇，同时生产 PSP，共享点源求解。VSP 数组轴为 `(Nt_sink,4,4,Nv_sink,Np_source,3)`，需要源点集和汇 V 基底，不需要源 V 基底或汇点集；DSP 将汇向量轴换为 dilution-vector 轴。每项 `PropagatorResult` 保留实际源汇 endpoint 和完整拥有的 `.values`；需要长期保存可由调用方按结果轴写出。

数值生成器默认输出全部物理汇时间；显式 source_selection/sink_selection 时，仅在所选物理源构造 RHS，并只投影/提取所选物理汇片，输出已按汇记录顺序紧凑为 Nt_sink。规范场、Dirac 格点和单次求解工作区仍为物理 Lt。压缩输入显式携带 time_selection；eager/stream 与借用限制见[PR L2](perambulator.md)。

直接 VSP/DSP 由解场投影到汇向量基底，不假设 Gamma5 对称性。后端若声明 Gamma5 与 `gamma5_hermitian=True`，可用 `result.reverse(t_sink)` 反向该物理时间对；省略时间会返回按原汇时间顺序排列的一组反向 block。变换会交换真实端点及源汇物理时间，并作用于自旋和完整端点轴。由反向 PSV 派生 VSP 时仍须核验作用量、质量、边界、自旋基、实际端点和来源约定，不能为缺数据盲目共轭。D 端必须精确匹配有序向量标签与随机实现身份。

## 高模派生

`production.perambulator.highmode_planning.plan_indexed_highmode_propagators(index, *, products=("PSV",), vsv_id=..., psv_id=None, vsp_id=None, psp_id=None, overlap_sink_id=None, overlap_source_id=None, data_ids=None, source_policy="allow_unknown")` 是对已有实际父项的只读兼容性预览。`validate_registered_highmode(index, spec)` 核验已登记高模定义及真实父项；二者都不计算、保存或恢复缺失产物。

数值投影函数位于 `contraction.propagator_views`：

| 产品 | 必要输入 | 单个物理时间对的计算 |
|---|---|---|
| PSV 高模 | VSV、PSV、汇 Overlap | `PSV - M_sink VSV` |
| VSP 高模 | VSV、VSP、源 Overlap | `VSP - VSV conj(M_source)` |
| PSP 高模 | 四类传播子、两端 Overlap | `PSP - M_sink VSP_high - PSV conj(M_source)` |

PSP 公式保留双侧投影的正交交叉项。源汇基底及点数可以不同，Overlap 模式集合定义各端的 `Q = I - VV†`。必须声明 orthonormal 基底；没有 Gram 矩阵修正路径。原传播子可按实际相同基底 ID 选择有序标签，但不能改变 Overlap 的投影模式集合；同一实际 ID 的两个角色仍保留独立视图。

输入必须是相互兼容的实际传播子和 Overlap。投影结果保留 psv/vsp/psp 数组布局及实际端点身份；定义记录 `high_mode_projection`、`input_roles`、精确 `input_views` 和实际父 ID。计算 dtype 按所有数学输入的 `numpy.result_type` 提升，不改变估计量或归一化。reader 返回值可能借用，下一次读取前须复制需要跨调用保留的数据。脚本负责时间对循环、输出分块与所选存储格式。

消费适配器会重建嵌入视图并校验定义；Diagram 预检还核验实际登记父项。消费视图可按需裁剪端点模态或点标签（包括重排），但不能改变 Overlap 投影集合、实际父身份或科学约定。时间、自旋、颜色轴不因此放宽。原始独立 VSP 与受限 Gamma5 VSP 分别验证；缺失父项、错误端点、标签顺序、作用量、求解容差或边界约定都会拒绝。

通用 `DatasetWriter`、登记、读取与恢复服务仍可供脚本按需组合；本页不定义空间点集、Overlap 或高模的固定持久生产入口。模块与估计量的 CPU 验收范围见仓库测试，不由只读规划或数值内核测试推导 GPU/MPI、真实系综或生产规模结论。

## 可选稀疏时间：逐层启用

1. 默认不传 selection，保留原 full-time 数值与来源契约
2. 空间采样与 point spec/planner 接受 selection；Overlap builder/planner 接受同一显式 selection，
   按输入各自映射读取物理时间，输出保持 Nt 槽
3. 高模输入的各 source/sink 记录须相互一致；overlap 记录绑定对应端点。实际基底、点池、
   投影模式与顺序仍严格核验，不能因 shape 相同或相同 seed 推断一致
4. 原始与派生输出通过相同 DatasetWriter、实际父身份和完整覆盖发布；重启复用原记录

详细导入、slot/physical 域与保存/重建见[共同时间交接](../temporal_sampling/temporal_production.md)。
额外的物理端点不能由两点重建补齐，缺失显式报错。

<a id="highmode-task"></a>
## 任务入口：已有传播子派生

选择所需产品，核对上方必要父项与实际物理端点；已登记输入先用
`plan_indexed_highmode_propagators` 做只读兼容性/覆盖预览，缺项不自动求解。
按相同物理时间对读取并复制需要跨读取保留的借用数组后，调用对应内核：

```python
from lqcdweave.contraction.propagator_views import (
    project_psv_highmode, project_vsp_highmode, project_psp_highmode,
)
psv_high = project_psv_highmode(vsv, psv, sink_overlap)
vsp_high = project_vsp_highmode(vsv, vsp, source_overlap)
psp_high = project_psp_highmode(vsv, psv, vsp, psp, sink_overlap, source_overlap)
```

这里全部传播子是单时间对：VSV `(4,4,Nv_sink,Nv_source)`，PSV `(4,4,Np_sink,3,Nv_source)`，
VSP `(4,4,Nv_sink,Np_source,3)`，PSP `(4,4,Np_sink,3,Np_source,3)`；每端 overlap 为 `(Nv,Np,3)`。
默认 `array_module=np`，返回 owned 数组及输入 result_type；PSP 保留双侧交叉项，不能只减两项。
这些数值函数检查轴/dtype/有限值，不认证物理身份。正交基底、作用量、边界、自旋基、实际端点及父项视图仍须核验，无 Gram 修正或额外采样权重。
按选择的文件格式同步保存结果和父项/约定；通用 DatasetWriter 的恢复与发布仍按原契约，未增加高模 runner。

### 有条件的 PSV dagger 转换

```python
from lqcdweave.production.perambulator.api import psv_to_vsp_dagger
vsp_time_block = psv_to_vsp_dagger(psv_time_block, gamma5)
```

这是保留的数组兼容入口，与 `contraction.propagator_views.psv_to_vsp_dagger` 指向同一 PR
实现。输入必须带前导时间轴 `(time,4,4,point,3,mode)`，输出 `(time,4,4,mode,point,3)`；
不是单个时间对 block API。默认 `backend=np`，返回拥有新数值存储的数组。调用方核实
Gamma5 Hermiticity 所需作用量、质量、边界及自旋基，同时明确所取反向物理源汇时间；这个
数组助手保留前导时间轴，不自动交换时间或添边界因子。需要交换真实时间与实际端点记录时，
使用 `PropagatorResult.reverse(t_sink)`，由已绑定后端或生成器构造参数提供 Gamma5 声明；低层 `reverse_propagator` 辅助函数仍要求显式传入 Gamma5 信息。
不把 dagger 当成缺 VSP 的默认补救。数值测试见 `tests/test_scene_storage_prefix.py` 和 `tests/test_highmode_projection.py`、`tests/test_highmode_dataset.py`；
GPU/生产验收不由数组转换测试推出。
