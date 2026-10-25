# 介子关联函数的本征模分块

`split_eigenmodes` 将已有 correlator 中的介子 Diagram 转为模式分块数组。
elemental 和 perambulator 都按实际端点选择读取；优化器接收真实的小矩形尺寸，
没有将块外元素置零后仍执行全尺寸缩并。

## 只为降低计算存储而分块

### 对角空间核的零块

Elemental reader 的 `structure="diagonal"` 声明沿 Meson 绑定传递。对于不相交的实际
V/V 模态选择，`Meson.is_structurally_zero(t)` 可在无数值 IO 的情况下证明顶点为零。
结构化与紧凑矩阵展开在规划前跳过这些分量，不读取或收缩相应块，保留原输出轴中的零项。
部分重叠、重排、缺少有效对角声明的情况不会因此删块；不能按动量或目录名称猜测结构。
算符轴本身需要缩短时，构造物理算符基之前使用同一证明筛选，之后仍联合规划。
这只清理严格零项，不是径向低秩近似，也不证明任何给定运行时间收益。

历史 V/V Dataset 缺少结构声明时，只有在其科学定义已确认的前提下，才可显式传给
`DatasetElemental(..., elemental_structures={("v2v", component, momentum_slot): "diagonal"})`。
该高级声明只支持 V/V，检查实际端点基底和维度，与已存声明冲突时报错；选中结构进入
绑定身份，原始数据及 DatasetSpec 不变。它不是按数据大小自动识别零块的开关。

无需保留各分区的物理结果时，使用 `partition_mode="memory"`：

```python
from lqcdweave.contraction.correlator import calculate

result = calculate(
    correlator, vertex_map=vertices, propagator_map=propagators,
    configuration=cfg, usedNe=70, partitions=[0, 16, 32, 48, 70],
    partition_mode="memory", optimize="anytime", storage="bounded",
)
```

`partition_mode` 默认是 `"radial"`，保留下文的 PC 径向结果；`"memory"` 用单位
权重求和全部有序物理块，返回原来的关联函数。矩阵仍是 `(Nsink,Nsource)`，
扫描时只追加原时间轴，不增加 `10×10` 分区轴。源端 dagger、复系数及断开图
保持原约定；两个独立圈分别求和后相乘，不进行真空减除。错误模式、重复分块及
末边界与显式 `usedNe` 不一致均在读取前拒绝。`"memory"` 必须同时给出 partitions；
已经分块的请求交给 `calculate` 时无需再传这两个参数。

固定边界的多个 cfg 可预先构造请求：

```python
from functools import partial
from lqcdweave.operators.correlator.api import split_meson_mode_pairs

blocked = split_meson_mode_pairs(
    correlator, partitions=[0, 16, 32, 48, 70], partition_mode="memory",
)
compute = partial(calculate, blocked, vertex_map=vertices, propagator_map=propagators,
                  usedNe=70, optimize="anytime", storage="bounded")
```

两种模式都读取实际矩形切片，并使用同一规划、资源审计与档案执行机制；普通
Diagram、结构化请求、固定 `meson_matrix` 及普通 anytime 均可使用。完整 cfg
档案保存求和配方，加载时无需再指定 partition_mode，也不允许覆盖该选项。

该选项减少相对径向分区结果的输出与组装预留，但完整 TermValues、常驻输入、
原始读取块、准备工作区和缓存仍按原机制计费。每个连通网络仍有 q⁴ 个物理块，
每个圈有 q² 个；首次规划开销和块结果保留尚未消除。因此不保证任意任务的总
峰值低于不分块版本，也不构成进程 RSS 或 GPU 峰值上限。小规模合成复数 CPU
参考覆盖数值、真实块宽、扫描、混合输出与档案恢复；真实 GPU 和生产规模待验收。

## 固定分块的 cfg 批处理

先在 cfg 循环外调用 `split_meson_mode_pairs`，再将所有通道、表示行及径向块组成完整请求，
交给一次 `calculate(..., execute=False)`。后续使用已选计划的 Dataset `.run(...)`，
或保存完整 cfg 档案后以 cfg iterator 加载回放；具体绑定见[一次规划与回放](../../../.agents/skills/lqcdweave-correlator/references/plans.md#联合任务与一次规划反复回放)。
`partial(calculate, radial, ...)` 只固定函数参数，每次调用仍会规划，不能作为固定路径生产模板。

符号分块不读取输入数据。每个 cfg 按其实际有序基底、模式选择及当前 reader 做兼容检查，
不要求不同 cfg 的本征基底身份相同。reader 保持有效到执行结束；物理时间覆盖与实际槽映射显式交付。
模态分块不会自动启用时间稀疏化。一次性任务仍可直接使用 `calculate(..., partitions=...)`。

## 保留全部有序块

需要分别保留每个有序块时，仍可使用 `split_eigenmodes`，然后把整个块 correlator 交给一次 `calculate`；数值结果也可再由 `project_meson_mode_pairs` 组合为 PC 径向输出：

```python
from functools import partial
from lqcdweave.operators.correlator.api import (
    split_eigenmodes, project_meson_mode_pairs,
)
from lqcdweave.contraction.correlator import calculate

blocked = split_eigenmodes(correlator, partitions=[0, 30, 60, 90, 120])
compute_ordered = partial(
    calculate, blocked, vertex_map=vertices, propagator_map=propagators,
    usedNe=120, time_axes={0: range(Lt), 1: range(Lt)},
    optimize="none", storage="bounded", execute=False,
)
for cfg in configurations:
    planned = compute_ordered(configuration=cfg)
    print(planned.prediction)
    values = planned.run().value
```

四段示例采用已验证的确定性路径；两段联合 `anytime` 已通过数值测试，四段搜索
效率仍待完善。整份嵌套 correlator 一次调用 `calculate`。原字典、列表和元组保留；原算符数组的
各元素若有相同分块形状，新轴追加在原数组轴之后。时间扫描轴在结果的末尾。
每个不同的 `(symbolic time, vertex)` 对应两条相邻的模式轴；顺序按时间的字符串
和顶点的表示排序，顶点内是源端 dagger **之后**的空间第 0、1 槽。
两个顶点的标量两点函数产生 `(4,4,4,4)`；原形状 `(Nsink,Nsource)` 的矩阵产生
`(Nsink,Nsource,4,4,4,4)`，再追加用户明确请求的时间扫描轴。

边界是实际有序基底中的位置，从零开始，左闭右开；末边界须等于显式 `usedNe`。
允许不等长分段。这个函数只构造符号，不读取本征值、elemental 或传播子。
每条传播子的源汇模式选择由相连顶点传递，源端 dagger 同步交换原始槽位。
双方声明实际基底身份时，沿用原生绑定的 ID 和有序坐标核对。

## 保留 PC 的径向算符

直接需要 PC 径向输出时，`split_meson_mode_pairs` 可在一次符号调用中完成分块与投影：

```python
from lqcdweave.operators.correlator.api import split_meson_mode_pairs

radial = split_meson_mode_pairs(correlator, partitions=[0, 16, 32, 48, 70])
```

每个顶点追加一条径向 pair 轴；四段两点函数为 `(10,10)`，原算符矩阵则为
`(Nsink,Nsource,10,10)`。容器、原矩阵轴、顶点排序及源端 dagger 与有序分块
一致。该入口先复用独立 Diagram 的投影，再组装线性表达式和断开圈乘积；
全部有序基础块仍参与计算，省去整份任务未投影的中间数组。纯常量保持原值，
矩阵中的零元素按其它元素的共同分块形状补齐。

单个有序块不单独具有原来的 C。对已有确定 PC 的等时中性介子算符，
在同一模式划分下使用对角块与两腿交换后的对称组合：

$$
O^{(aa)}=O^{aa},\qquad
O^{(ab)}=(O^{ab}+O^{ba})/\sqrt2\quad(a<b).
$$

上述对称组合保持裸算符 C=c0。若需要完整有序径向空间，还需反对称组合
`(O_ab-O_ba)/sqrt(2)`，它属于 C=−c0；对角项没有对应反对称基。
一般目标 C 区取 `(O_ab+(C/c0)O_ba)/sqrt(2)`。不同裸 C 标签因此可以在相同物理 C 区中共存。
带独立标签的物理算符线性组合可经 `Hadron → gen_correlator` 构造；绑定两种实际模式选择时，
紧凑矩阵后端按物理端点拆分并加和，不展开全部标量矩阵元。

`project_meson_mode_pairs` 将每一对分块轴变成一条这样的组合轴。
四段的顺序是 `(0,0),(0,1),(0,2),(0,3),(1,1),…,(3,3)`，共十项。
它可以作用在符号数组上，也可以作用在数值结果上；后者不重复执行缩并。
投影直接取对角块，或将两个交换块相加后除以 `sqrt(2)`，不对符号数组执行含零
权重的稠密矩阵乘法。一次 `split_eigenmodes` 内，相同 Diagram 的有序分块符号及
相同标量表达式的替换结果复用；返回的数组各自拥有元素槽位。分块仍完整生成，
这项优化不减少所需基础块数量，也不改变源端共轭、轴顺序或输出归一化。

```python
# 标量两点函数：四条分块轴位于开头，后面可以有时间轴。
radial_matrix = project_meson_mode_pairs(values, mode_axis=0, vertices=2)
# 原算符矩阵 (Nsink,Nsource,4,4,4,4,...) 则使用 mode_axis=2。
```

取对角项权重 1、非对角组合权重 `sqrt(2)`，在两个组合轴上加权求和即恢复原函数。
直接求和全部有序分块轴也恢复原函数。I=0 的断开图仍作为闭合圈的乘积，原味道
系数和真空项保留；本接口不进行逐组态真空减除。

## 优化、读取和限制

- 所有块进入同一原生展开和路径规划。相同顶点选择复用绑定对象，相同传播子块和
  elemental 块通过数据身份与选择身份共享；公共中间结果由现有优化器处理。
- `anytime` 搜索预算与数值执行时间不同。四段两点函数有 256 个基础块，搜索可能
  比整块显著更长；预算内没有完整可行路径会明确报错，不自动换算法。
  `optimize="none"` 使用现有确定性规划路径，适合先核对结果和执行成本。
- 分块读取依赖 reader 的公开 `mode_indices` 接口；不支持该选择的自定义 reader
  必须补齐接口。登记 dataset 的打开校验仍可能扫描完整文件，不据此宣称磁盘 IO
  或常驻输入也缩小到一个模式块。
- 本接口用于 `calculate`；调用脚本保存结果时须显式记录分块约束与模式选择，不能将分块结果登记成原整块关联函数。登记 Diagram 的只读预检 schema 尚未表达这些分块约束。
- 当前支持等时 dense V/V 介子图，包括连通图、独立闭合圈及其和/乘积、时间扫描。
  先做量子数投影和源端共轭，再做模式分块。不同时间位移的项、点端扇区、重子和
  与 SpectralProfile 联用未验收；选中顶点不接受已有 spectral profile。
- 每个加法项必须恰好包含同一组具名顶点；重复完全相同的时间和顶点标签必须
  先通过不同标签区分。再次分块、Diagram 的非线性幂和不一致的顶点集合明确拒绝。
- PC 保持需要同位旋伙伴使用同一实际模式选择，模式区间定义随对称变换一致；
  若切在严格简并本征子空间中间，应调整边界以保留完整简并空间。

## 验证范围

正式 CPU 测试 `tests/test_mode_block_correlator.py` 使用小规模合成复数数据，
以完整尺寸零掩码和独立直接求和参考核对逐块结果、源端共轭、不等长分段、
连通/断开合并、PC 配对、嵌套输出和时间扫描。数据不对应物理作用量或真实系综；
四段的 256 个连通图共享 64 个物理输入块；小规模测试核对两类 reader 各读取
32 次，并确认每次读取的是对应子区间。两段的联合 anytime 路径也有数值对照。
`tests/test_mode_projection.py` 另核对合并入口的全部基础项、反向顶点顺序、
独立数值配对参考、断开圈因式分解、复系数、时间扫描及公共数组槽位隔离。
不作为 GPU 或生产规模验收。全部块总耗时与整块耗时应在相同设备、同一计时范围
内分别测量；小块调用开销可能显著，正确性不意味着任意尺寸都会更快。
