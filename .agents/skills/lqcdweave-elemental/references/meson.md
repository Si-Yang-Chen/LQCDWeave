# 介子 elemental 的表格前端

公开入口：`lqcdweave.production.elemental.api.ElementalGenerator`。

本页仅用于独立生产或后端开发。普通测量使用 `Elemental` 声明和 `calculate`，
见[自动准备](../../../../docs/modules/correlator/managed_elemental.md)。

## meson55 生产表与后续消费

meson55 默认使用 v2：`meson55_preset()` 与 `momentum_recipe("mom4").momenta` 给出 55 行路径、
复系数、归一化与 component 身份都由预制表给出。表不替用户选择规范场涂抹、Ne、时间选择或数组后端，
这些参数须按实际数据明确提供。动量轴顺序和可复用保存定义见[meson55 使用页](../../../../docs/modules/operators/meson55.md)。

需要消费历史目录 的四个 G 行时，生产表与绑定均显式设 `version="csns"`；v2 改变这些行的配方，使用修正核必须重新生产或补充对应 elemental。其余 51 行配方相同，但目录引用也按版本匹配，不自动跨版本读取。

高分波目录的逐壳判据、历史配方差异和核验范围统一见[低分波投影检查](../../../../docs/modules/operators/meson55.md#低分波投影检查)，按实际目录版本生产，不从标签推断投影已通过。

将登记产物接入 correlator 时，使用 correlator skill 中的 `RecipeVertex` 预制绑定按冻结目录名称选择算符。
它同时应用 Dirac 结构与空间耦合；常规 meson55 测量不需要手写 `InsertionTerm` / `Insertion.from_rows`。
只有用户要构造目录之外的配方时，才展开这些低层接口。

## 默认与可选时间选择

默认完整时间，不需要时间参数。明确稀疏化时传 `selection=load_time_selection(..., Lt=physical_Lt)`，直接数组按该记录紧凑存储；`iter_slices()` 返回真实 t 与借用路径结果，并逐片调用 `calc_path_blocks(t)`。它不迭代导数；需要导数时，对每个所需物理时间显式调用 `calc_derivative(t)`。`Lt` 不变，`Nt` 为输出槽数。

额外输入覆盖或不同槽序使用 `eigenvector_selection`（重子可为三腿记录）；介子点数组可用 `point_selection`，SpatialPointSet 保持自身 times。句柄使用 time_selection，缺省为完整按序输入。跨时间路径先求真实 t±offset 再查槽，缺片报错；不会重算输入、按槽邻接或补零。

## 初始化参数

- 必填：`gauge_field, eigenvectors, operator_table` 和动量表；使用 `momentum_dict=` 或旧的 `momenta` 有序列表，二者只能选一个。纯 P/P 可传 eigenvectors=None。
- `momentum_dict` 可直接与 correlator 的 `Insertion` / `InsertionGaugeLink` 共用；字典值的插入顺序决定动量槽，key 不作为槽号。`generator.momenta` 保存实际使用的有序 XYZ tuple。
- `Ne=None`：继承实际有序基底模态数；指定整数时取该有序前缀，左右共用。
- `blocks=("v2v",)`：V/P 基础路线选择四扇区；实际 D 通过 vector_bases 扩展到九扇区，见[统一接线](dilution.md)。
- `points=None`：单一点集，两端按真实时间使用；数组 `(time,point,xyz)` 或 SpatialPointSet。不接收 points_right。
- `array_module=np`：默认 NumPy；已有 CuPy 环境中显式传 `array_module=cupy`，空间运输、相位、投影和组合结果使用该后端，返回设备数组。实际设备覆盖按本次实现与输入核验。
- `ne_block_size=32`：模态计算块，不代表全部输入或输出内存上限。
- 数值块、运输缓存和输出各有容量；这些参数不限制整体峰值。内部投影实现与验收范围见[EL L2](../../../../docs/modules/elemental/elemental_production.md#shared-spatial-projection)。
- `phase_convention="midpoint"`：路径可选 anchor/midpoint/none；导数使用 anchor，none 则取消相位。
- `path_boundary="periodic"`：可选 open；空间周期，时间越界路径为零。
- `dilution=None, is_blending=False`：明确指定的模态对权重；启用 blending 时需配方，应用一次。
- `write=None`：同步消费最终组合的回调，不负责文件保存。介子先构造完整结果，不能据此宣称降低峰值内存。
- `lattice_size=None`：数组从规范场推断；惰性 load(key) 句柄须提供 `(Lx,Ly,Lz,Lt)`。

## 一个表格

每条记录包含 component、path、coefficient。同 component 的记录线性相加，首次出现顺序为结果行顺序；系数接受复数或 `[实部,虚部]`。也接受已有目录的 `id/terms` 嵌套格式。路径与系数必须一起导入，不另设 paths、rows 或 derivative_table。

GaugePath 方向 `0..7 = +x,+y,+z,-x,-y,-z,+t,-t`，也接受这些方向字符串。空路径为局域项，不自动枚举路径。调用方负责读取表格文件，生成器只消费记录。

## 计算与输出

```python
generator = ElementalGenerator(gauge, vectors,
    momentum_dict=momentum_dict, operator_table=operator_table,
    Ne=Ne, blocks=("v2v",), array_module=np)
for t in times:
    result = generator.calc_path_blocks(t)
    save(result)  # 调用方选择保存方式；借用结果须在下次计算前保存或复制
```

- `calc_path_blocks(t)`：计算表中每个路径的运输并按系数组合，返回 sector 字典。
- `calc_derivative(t)`：同格式表的 path 此时是有序导数方向序列，仅接受 0/1/2，空序列为零阶；仅 V/V。保留左右作用符号及方向顺序，forward−backward 不自动除以 2a。
- 两个计算方法都只接收物理时间 `t`。`operator_table` 与 `blocks` 扇区在初始化时固定；调用 `calc_path_blocks` 时，表中 path 按规范场路径解释，调用 `calc_derivative` 时则按有序导数方向解释。需要多个时间的导数结果时，逐个物理时间显式调用 `calc_derivative(t)`；`iter_slices()` 始终只调用 `calc_path_blocks(t)`。
- 数组输入立即加载：规范场与点坐标形成快照，本征矢保留借用引用，调用方在生成器使用期间保持数据有效且不修改；句柄使用 load(configuration)。load 异常后不能继续消费旧数据。
- `project_SU3(matrix_batch_size=4096)`、`stout_smear(nstep,rho,matrix_batch_size=4096)` 的选择、顺序和规范场版本单独记录。
- V/V 为 `(component,momentum,Ne,Ne)`。点端布局与稀疏所有权见[点端数据](point-flow.md)。
- write 回调签名 `(row,momentum,sector,slices,value)`，同步消费，启用时返回 None。

空间组合系数属于表；Flavor、Gamma 和最终自旋耦合由算符层处理，不能重复乘系数。保存方式按[保存与持久交接](production.md)选择。公开签名与前端行为以 `production/elemental/api.py`、`frontend.py` 及 `tests/production/elemental/test_table_frontend.py` 为核对入口。

## GPU 数值与主机保存的边界

生产时在已有设备环境中选择同一个前端，不新增另一套生成器：

```python
import cupy as cp

generator = ElementalGenerator(gauge, vectors, momenta, operator_table,
    array_module=cp)
for t, result in generator.iter_slices():
    save_slice(t, cp.asnumpy(result["v2v"]))  # 主机文件写出显式转移一次
```

输入 reader、文件 IO 和配方/时间坐标元数据仍可在主机；不要在每个数值步骤把设备矩阵转回 NumPy。
已有登记场的 V/V 辅助入口 `prepare_elemental_stage(..., array_module=cp)` 也把同一后端传给 kernel，
并记录真实后端；它不写文件、不自动选择设备，也不替调用方完成主机转移。

消费已有 elemental 进行关联函数收缩时，由 `calculate` / 已选计划的执行 `backend` 和
`target_hardware` 选择设备。选择 CuPy 后，原始数值块上传后在同一设备 scope 执行
profile、源端 dagger、自旋展开和 stack；系数常量在 scope 内缓存，支持的配方走已有 fused kernel，
其余支持的布局保持 CuPy 数组运算。无需重新生产 elemental，也不把普通传播子的反向边改成额外 gamma5 派生。
具体执行与预算沿用[收缩执行说明](../../lqcdweave-correlator/references/execution.md)。
CPU 协议替身验证路由；真实 CuPy 数值、CUDA/HIP 编译和速度须分别在实际设备验收。

<a id="mode-sampling"></a>
## 仅明确均匀无放回选模时：包含概率权重

确定性 Ne 前缀截断、普通空间点抽样和设备登记不触发此步骤。

这段描述抽取实际父模态，不描述将多个父模态混成一个 D 的 QR dilution。
QR 向量按 [Noisevector skill](../../lqcdweave-noisevector/SKILL.md)生成，
其原始二阶矩为 `(m/n) V V†`；当前估计量在实际 QR 混合列轴上区分相等和不同标签，按 H_r=(n)_r/(m)_r 使用包含系数及自由和/等标签场景。这个来源与直接抽取原 Laplace 父模态不同，见[统一 V/D 计权](dilution.md)。
使用实际 D 生产原始 elemental 时保持 `dilution=None,is_blending=False`，
不自动施加下面的 dilution_pair_weights，也不据此宣称完整估计量无偏。

```python
from lqcdweave.sampling.mode_weights import dilution_pair_weights
weights = dilution_pair_weights([4], [2])
# [[2., 6.], [6., 2.]]，不是两个单模权重向量的外积
```

输入每块总模数/抽取模数，假定块间独立、块内均匀无放回；输出为所选模总数的方阵，顺序与块及实际选模一致。同模对角权重与不同模对包含概率不同；非平凡块只抽一个模无法估计不同模对，非法计数报错。ElementalGenerator 的 `dilution=(total_counts, used_counts), is_blending=True` 已施加一次，不再乘本函数结果；默认 `dilution=None,is_blending=False`。保存实际抽样模态标签、抽样设计及估计量，不能用 seed/shape 代替实际基底身份。
