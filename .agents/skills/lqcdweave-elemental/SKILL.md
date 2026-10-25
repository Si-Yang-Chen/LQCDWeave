---
name: lqcdweave-elemental
description: "独立生产 LQCDweave 介子、流及重子 elemental，维护空间配方、端点、结构与计权契约；作为 correlator 自动准备的条件参考，关联函数测量仍由 correlator 统一入口负责。"
---

# LQCDweave · Elemental 生产

消费端整体顶点准备保留原 spin×spatial 配对、复系数、源伴随及颜色 epsilon 契约；
不要求改变 elemental 文件布局或预先合并不同算符，见[介子内核准备](../../../docs/modules/execution/meson-whole-vertex.md)。
同一算符可同时引用 diagonal、point_diagonal 与 dense 分量；消费端按实际端点标签嵌入支持后求和。
声明外存储条目不参与物理缩并，不能靠关闭 compact 标记读取它们；仍须保存实际点池和有序基底身份。

共同任务路由、前端边界、保存/身份及执行范围统一见[技能共同约定](../README.md)。只读取当前任务的条件参考。

## 任务归属与自动准备

目标是关联函数时以 [correlator skill](../lqcdweave-correlator/SKILL.md) 为主，本 skill 只补充空间配方及输入契约；目标是独立 elemental 产物时使用下述生成器与保存交接。

关联函数测量中的 elemental 自动准备采用固定测量入口，用户只绑定实际输入、科学参数及已有少量选项。不要求用户组合 calculator/writer/runtime，不提供 prepare、producer、loader 或保存回调作为测量补产步骤。

独立生产 elemental 文件时，代理编写脚本组合 `ElementalGenerator`（介子/流）或 `BaryonElementalGenerator`（重子）与既有公共存储服务，组织计算、保存及完成核验；具体交接见[保存与持久交接](references/production.md)。这类任务不要求构造关联函数，也不通过 `calculate` 发起生产。

普通纯空间介子/流从 `contraction.correlator` 导入 `Elemental, calculate`，声明实际登记输入、绑定 Meson 后直接 calculate。后端按真实测量请求确定 V/D/P 扇区、分量、动量、物理时间，检查登记覆盖、复用兼容结果、补产缺项并保存。计权只在 `calculate(..., elemental_weighted=True)` 设置，`execute=False` 不生产。示例和输入契约见[自动准备](../../../docs/modules/correlator/managed_elemental.md)。

当前支持全时间登记输入、九类空间介子/流扇区和版本化目录的重子 VVV。重子同样使用 `Elemental(gauge, basis, baryon_preset().select(...), momentum_dict, directory=...)`；三腿独立基底可传有序三元组。普通重子收缩目前使用固定时间坐标。跨时间路径与 D/P 重子尚未支持。发现缺口应补齐后端，不能让用户自行写补产流程。已有完整 elemental reader 直接消费。

多 cfg 的任务分配、循环前共同准备及完成核验统一按 [Dispatch skill](../lqcdweave-dispatch/SKILL.md)；循环内加载当前 cfg 的实际规范场、基底、点集与时间记录，本 skill 负责 elemental 配方及保存约定。

按实际规范场、基底和测量配方生产介子、重子或流的空间数据。已有数据直接消费，不隐式重算基底。只读当前任务需要的参考，不要求逐页读取全部参考。

## 独立生产入口：配套预制表

以下输入 `gauge` 为 `(Lt,Lz,Ly,Lx,4,3,3)`，`vectors` 为 `(Lt,Ne,Lz,Ly,Lx,3)`，必须来自已核验的同 cfg 和实际有序基底。示例明确选择 mom3 和长度 1，不替其它任务选配方：

```python
from lqcdweave.operators.single_hadron.api import (
    momDict_mom3, gauge_link_preset, gauge_link_tables,
)
from lqcdweave.production.elemental.api import ElementalGenerator
from lqcdweave.storage.dataset import DatasetWriter

momentum_dict = momDict_mom3
selected = gauge_link_preset(max_length=1)
operator_table, insertion_dict = gauge_link_tables(selected)
generator = ElementalGenerator(
    gauge, vectors, momentum_dict=momentum_dict, operator_table=operator_table,
)
momenta = generator.momenta  # 保存时记录生成器实际使用的有序动量轴
with DatasetWriter(output_directory, output_spec, provenance=provenance) as writer:
    for slot, (t, blocks) in enumerate(generator.iter_slices()):
        writer.write(slot, blocks["v2v"])  # 同步消费，再计算下一片
    data_id = writer.publish()
```

`output_spec` 明确本例的直接投影配方、实际规范场与基底、时间顺序和输出轴；
构造要求见[保存与持久交接](references/production.md)。独立基础生产只组合计算和保存接口，
库不再提供代为计算并写出的基础产物入口。关联函数测量中的自动准备仍由 `calculate` 管理。

本例使用 `gauge_link_tables` 返回的配套表，因此生产与收缩共用它的 `operator_table`、`insertion_dict` 和同一份 `momentum_dict`。动量字典值支持三整数字符串或整数 XYZ 三元组，按插入顺序对应槽，key 不作为槽号也不排序。`generator.momenta` 是实际使用的有序 XYZ tuple，保存时记录它。`mom_dict_to_list(momentum_dict)` 可转换自定义映射，原有 key=0/1/2/3/9 预制表编号也保留；key=2 是 `p²<=2` 的 19 项表，顺序与 `momDict_mom3` 前 19 项一致。例中 `momDict_mom3` 有 27 项，不是 p² 上限。旧的 `momenta` 有序列表参数继续可用；它与 `momentum_dict` 二选一。路径长度 0/1/2/3 的行数为 1/7/37/187，和动量表无关。自定义路径表仍用同一入口；meson55 的后续绑定没有 `insertion_dict`。

预制 `gauge_link` 表和自定义路径表都通过 `calc_path_blocks(t)` 计算。`iter_slices()` 按初始化的物理时间惰性调用 `calc_path_blocks(t)`，只迭代路径结果；需要导数时，应对每个所需物理时间显式调用 `calc_derivative(t)`。

## 必填、关键默认与物理约定

- 必填 `gauge_field, eigenvectors, operator_table` 和动量表；动量表使用 `momentum_dict=`，或用旧的第三位置/`momenta=` 有序列表，二者只能选一个。数组自动确定格点。惰性句柄另给 `lattice_size` 并 `load(cfg)`
- 默认 NumPy、`Ne=None` 使用实际有序全基底、`blocks=("v2v",)`、全物理时间；不自动稀疏化或缩小 Ne
- 默认介子路径相位 midpoint，重子 anchor；导数为 anchor（显式 none 除外）。路径 `0..7=+x,+y,+z,-x,-y,-z,+t,-t`；导数只用 0/1/2，保留左右作用与 forward−backward，不自动除以 2a
- 预制 component 使用目录标识、版本和目录内编号；路径、复系数及腿顺序由目录规定，存储槽与目录编号分开。空间组合系数只在生产施加一次，Flavor/Gamma/源端 dagger 在消费层按约定处理一次
- 默认不施加随机模态或空间抽样权重。相同 shape、Ne、seed 或文件名不证明实际基底/点池/规范场一致；Laplace、Dirac、elemental 的规范场版本分别保存
- 本征矢输入借用，生成器使用期间保持有效且不修改；返回/回调可能借用输出，下一次计算前同步保存或复制。保存具名轴、实际时间、动量顺序、相位、表格、基底与规范場来源及归一化

## 限制与按需参考

- 用户选择 meson55 时：`meson55_preset()` 默认给出 v2 的 55 行空间表，`momentum_recipe("mom4").momenta` 的 33 个动量顺序保持不变。历史目录 的 G 行可用 `version="csns"` 显式选择；v2 修正的四行需生产对应 elemental，其余 51 行路径系数相同，但目录引用仍按版本匹配，不自动跨版本读取。component 身份、行保存及版本边界见[meson55](../../../docs/modules/operators/meson55.md)。此入口不自动选择涂抹、Ne 或时间。
- 介子路径、复系数或导数：[介子前端](references/meson.md)
- 重子三腿、独立基底与截断：[重子前端](references/baryon.md)。当前只有 V/V/V；不以最小 Ne 静默合并三腿
- 局域/有向流与真实跨时间端点：[流与端点](references/current.md)
- 确实需要 V/P、P/V、P/P：[点端数据](references/point-flow.md)。确定点集直接消费，随机空间取样才用对应 skill
- 独立生产预权 V/P 数据时，使用 `ElementalGenerator(..., elemental_weighted=True)` 和带显式池身份的 `SpatialPointSet`（也可用其 `.points` 满足契约的 `SpatialPointHandle`，并调用 `generator.load(cfg)`）；受管测量只在 calculate 设置。公式与限制见[计权参考](references/current.md#空间点采样计权)
- 用户明确时间稀疏化：[共同时间交接](../../../docs/modules/temporal_sampling/temporal_production.md)。复用同 cfg 已保存 selection；`calc_path_blocks(t)` 与 `calc_derivative(t)` 的 t 始终是物理时间。`iter_slices()` 仅迭代路径结果；Nt 是槽数，物理 Lt 不变，跨时间路径不能用槽邻接代替真实端点
- 保存、普通续算：[保存与持久交接](references/production.md)。新数组按 cfg 保存完整 NPY，多字段结果用 HDF5，并由共享 `dataset.json` 登记；只复用已验证完成的 cfg 覆盖。格式细节见[数据保存说明](../../../docs/modules/storage/data_storage.md)
- 仅设备目标尚未登记时：[目标档案与生产内存估算](../../../docs/architecture/generic_production.md#unregistered-target)。不在默认流程添加独立预检调用；已有运行期必要内存检查照常保留
- 仅明确使用随机模态稀释估计量：[模态抽样权重](references/meson.md#mode-sampling)。与设备是否登记、空间抽样和确定性截断无关
- medium QR 混合向量由独立 [Noisevector skill](../lqcdweave-noisevector/SKILL.md)生产。导入已有 V/D 时用 `vector_bases={"V": v_handle, "D": d_handle}` 和明确的九扇区 `blocks`；D handle 携带 `dilution_source`。D 与点采样共用 `elemental_weighted`，详细来源、整组选择和同池高阶范围见[V/D 接线](references/dilution.md)。保持旧 `dilution=None`、`is_blending=False`；旧选模入口针对 Laplace 父模态，当前 H_r 包含系数针对实际 QR 混合列；两者不互换来源或入口。同组 D/D 对角乘 H1、非对角乘 H2。

介子 write 在完整组合结果之后交付，重子 write 可交付三腿计算块；计算、读取和写入粒度不定义持久文件布局，也不等于整体内存上限。GPU/MPI、真实系综和生产规模须分别验收。正式提交前按[参数核对](../README.md#execution-scope)核对实际版本、范围和已有授权；脚本准备及小规模验证不增加重复审批。

V/D/H 的[完整交接流程](../../../docs/modules/perambulator/three_sector_production.md)及示例使用九个 V/D/P 扇区；H 是传播子端的投影点名称，不新增 H elemental 扇区。P elemental 保留原始点空间核，完整 low+medium 补投影只在传播子端施加。两条流程共用 `elemental_weighted`，不增加第二个权重开关。实际点身份、结构和计权边界见[H 点端交接](references/point-flow.md#h-elemental)。

介子和重子预制目录统一使用[版本化目录](../../../docs/modules/operators/elemental_catalogs.md)。重子 `baryon_preset(version="v1")` 收录局域/单链三腿、距离平均、S3 和全部 S2 槽位投影；用返回表的 `.select` 按名称或目录编号选择，保留目录元数据。空间交换标签包含颜色 epsilon，自旋及味道仍需单独组合，不能自动声明完整系数对称性。
