# Elemental 与单强子算符

裸二进制及稀疏 P/P 的 Dataset 读取入口见[数据读取说明](../data_readers/data_readers.md)。
配套 H 传播子时仍生产原始点核与 V/D/P 扇区，完整父空间的 Q 投影只在传播子端施加。
H 不是新的 elemental 正交基底；点池、结构与统一计权见[H 点端交接](../../../.agents/skills/lqcdweave-elemental/references/point-flow.md#h-elemental)。

通常先读取已经计算好的 elemental，再指定各分量的系数来构造算符。这个流程不需要规范场、本征矢或生产器。
生产数值使用公开 elemental calculator；持久数组默认每 cfg 保存整块 NPY，并由共享根 `dataset.json` 登记。产物保存与登记由调用脚本按需组合通用 storage/runtime，见[生产脚本说明](elemental_persistence.md)和[默认存储格式](../storage/data_storage.md)。

## 使用已有数据

```python
from lqcdweave.adapters.elemental import ArrayElemental
from lqcdweave.contracts.elemental import ElementalComponent
from lqcdweave.operators.single_hadron import Operator, OperatorTerm
from lqcdweave.contraction.binding.api import Meson

components = [ElementalComponent("row0"), ElementalComponent("row1")]
operator = Operator.from_terms("example", [
    OperatorTerm(15, components[0], momentum=0, coefficient=1),
    OperatorTerm(15, components[1], momentum=0, coefficient=0.2j),
])
with ArrayElemental.from_npy(
    {"v2v": "elemental.npy"}, components,
    axis_order=("component", "momentum", "time"),
    phase_convention="midpoint", cfg="cfg001",
) as data:
    meson = Meson(data, operator).load("cfg001")
    factors = list(meson.factors(t=0))
    vertex = meson.get(0)  # (spin_left,spin_right,mode_left,mode_right)
```

`OperatorTerm.gamma` 使用 0..15 的 Clifford bitmask：按顺序相乘的四个 Euclidean Gamma 对应位 1、2、4、8。`gamma(15)` 为 Gamma5。系数支持复数。

`ArrayElemental` 接收数组或只读 NPY memory map，内部轴为 `(component,momentum,time,...)`。`from_npy` 的 `axis_order` 显式说明文件的前三轴。末尾轴分别为：

| sector | 末尾轴 | 定义 |
|---|---|---|
| v2v | mode_left, mode_right | V† K V |
| p2v | point_left, color_left, mode_right | P† K V |
| v2p | mode_left, point_right, color_right | V† K P |
| p2p | point_left, color_left, point_right, color_right | P† K P |

生产器 P/P 输出保持稀疏，不必转换为该稠密数组格式。可实现 `ElementalReader.block(component,momentum,t,sector)`，直接返回 `SparsePointBlock`；`Meson.factors` 会保留它的稀疏表示。reader 还须声明 `Lt`、`phase_convention`；可声明 `cfg`。

分量可以是投影行或路径。路径用 `ElementalComponent.path(GaugePath(...))` 定义。只有已保存的分量可以被引用；读取缺项会报错，不触发自动生产。调用者须确保各输入属于同一组态、同一本征基底及顺序、对应的点集、规范场版本和动量表；当前数组适配器不替代产物台账。

`Meson(..., source=True)` 对系数取共轭、自旋应用 `gamma4 @ Gamma† @ gamma4`，并交换、共轭空间两端。混合块从相反 sector 读取，不根据路径名称猜测可重建关系。`load(..., usedNe=..., usedNp=...)` 显式截取两端的前若干模式或点，超出覆盖会报错。

`factors` 每项提供 `coefficient`、`spin`、`spatial`、`endpoint_times`。因子拥有空间块的副本，可在 reader 的下一次读操作后保留。`get` 显式物化顶点，仅允许所选项具有相同的端点时间；跨时间算符请按因子给出的两个时间选择传播子再收缩。关闭 NPY reader 后，不得继续使用直接从其 `block` 获取的借用视图。

## Insertion 前端

```python
from lqcdweave.operators.single_hadron import (
    Insertion, Operator, GammaName, DerivativeName, ProjectionName,
)
momenta = {0: "0 0 0", 1: "1 0 0"}
insertion = Insertion(GammaName.RHO, DerivativeName.NABLA,
                      ProjectionName.A1, momenta)
operator = Operator("rho_nabla", [insertion[0](0, 0, 0)], [1])
```

`Insertion` 生成表示耦合后的系数，不计算 elemental。`InsertionGaugeLink` 接收用户提供的空间系数字典，并使用相同的耦合实现。动量索引按 `momentum_dict` 的值的插入顺序确定。`Operator` 的 insertion 输入使用 cyclic Gamma 命名，索引 5/13 的命名符号在构造时转换一次；`OperatorTerm` 和 `OperatorPath` 直接使用 bitmask，不进行该转换。

`little_group_projection` 使用包内置的格点群表示实现；自动表示投影属于 Correlator
功能，要求 Python 3.11+ 与 `[correlator]` extra。基本算符、静止系耦合和生产只需要 NumPy。

`LocalVectorCurrent`、`LocalAxialCurrent`、`PseudoScalarDensity` 和 `ConservedVectorCurrent(wilson_r=...)` 都只生成系数配方。守恒流的专属 `ConservedVectorInsertion(direction=3, momentum_index=0, wilson_r=1.0)` 绑定一个方向和一个已存动量索引，保留正反链的两项端点；其 `.operator()` 可生成交给 `Meson.factors` 的路径算符。`direction=0..3` 分别是 x/y/z/t。不会自动施加重整化系数或晶格间距因子；它们应由测量定义明确给出。

### 跨时间有向电流 V2V 与缩并

八方向有向流复用 `calc_path_blocks`：构造生成器时指定八条单链路径、v2 编号、中点相位和任务的时间边界。全时间数组由调用方逐片复制拼接，raw 布局契约由 contracts.current 生成。下例中的 `preloaded_vsv` 是调用者已经加载且核对身份的 VSV 读取器：

```python
from lqcdweave.contracts.current import validate_current_raw_contract
from lqcdweave.contraction.directed_current import (
    contract_directed_current_pair_v2v,
)
from lqcdweave.operators.single_hadron.current import ConservedVectorInsertion

from lqcdweave.contracts.current import build_current_raw_contract
# generator 已配置 paths=[(d,) for d in range(8)]、v2、midpoint 和 periodic。
values = np.stack([generator.calc_path_blocks(t)["v2v"].copy()
                   for t in range(generator.latt_size[3])], axis=1)
generated = {"v2v": values, "contract": build_current_raw_contract(
    {"v2v": values}, boundary="periodic", available_ne=available_ne,
    used_ne=generator.usedNe, momentum_count=generator.num_momentum)}
raw = {"v2v": generated["v2v"]}
contract = validate_current_raw_contract(raw, generated["contract"], require_temporal=True)
insertion = ConservedVectorInsertion(direction=3, momentum_index=0, wilson_r=1.0)
Ne = generator.usedNe
result = contract_directed_current_pair_v2v(
    insertion, raw, contract, insertion, raw, contract, preloaded_vsv,
    first_anchor_time=0, second_anchor_time=1,
    first_field_ne=Ne, first_bar_ne=Ne,
    second_field_ne=Ne, second_bar_ne=Ne,
)
```

`v2v` 的轴为 `(direction,time,momentum,bar_mode,field_mode)`。方向 0..7 为 `+x,+y,+z,-x,-y,-z,+t,-t`；第 6 项使用 `U_t(t)`，第 7 项使用 `U_t(t-1)†`。相位固定为旧版 midpoint 约定，含空间半格点位移；此 raw 入口拒绝 blending 权重。`boundary="open"` 时越界时向基块为零，缩并遇到越界端点会报错。`preloaded_vsv.get(source_time,sink_time)` 必须返回 `(sink_spin,source_spin,sink_mode,source_mode)` 复数块；缩并不会加载或生产传播子。

`contract_directed_current_v2v` 逐项计算 `S(field→sink) J(bar←field) S(source→bar)`，保留外部自旋和模态轴；`contract_directed_current_pair_v2v` 逐项配对计算无味道、无符号的闭环迹。两者可直接消费专属 insertion；显式给出的动量索引若与 insertion 绑定值不同会报错。它们在每项收缩后才乘 `coefficient * normalization`。闭环入口不自动施加 Wick 负号、味道因子、体积归一化、共轭、取实部或源时间平均。raw 契约的 `cache_identity` 只标识布局与约定，不标识数组内容或实际本征基底；消费前仍须核对组态、基底顺序、规范场版本、动量表及 VSV 来源。此内存入口目前经过 NumPy CPU 小格点验证，完整输出常驻内存。`calc_path_blocks` 返回单个物理时间的借用缓冲；八方向数值准备与行写入辅助见[Elemental 流程 模块](elemental_production.md#registered-eight-link-current-contract)，产物发布与恢复由调用脚本按需组合通用 storage/runtime。登记数据的双电流闭环迹可通过[持久测量入口](../measurements/directed_current_measurement.md)输出并恢复；单链持久派生、通用多点/点端测量、专用旧格式导入和 GPU/MPI 验收范围仍见各自接口文档。

#### 格式适配与身份校验

专属 insertion 只保存算符定义。调用 `.operator()` 后，`Meson.factors` 使用现有 `ElementalReader.block(component,momentum,time,"v2v")` 协议；对于时向分量，需要显式登记 `GaugePath((6,))` 和 `GaugePath((7,))` 两个分量，以及 midpoint 相位。数组、NPY 和裸二进制读取器都可实现这个协议。普通数组 reader 能检查分量、时间、模态尺寸与相位，不能从数组形状证明组态、实际基底和规范场身份。

沿用旧公式的 `contract_directed_current_*_v2v` 则要求 `raw={"v2v": values}`、旧版八方向契约以及 `(direction,time,momentum,sink_ne,source_ne)` 复数数组。已有**登记产物**可通过 `load_registered_directed_current` 适配：

```python
from lqcdweave.adapters.directed_current_dataset import load_registered_directed_current
from lqcdweave.storage.dataset import DatasetReader

# cfg is the configuration being measured.
with DatasetReader(elemental_dir, cfg=cfg) as elemental, DatasetReader(vsv_dir, cfg=cfg) as vsv:
    adapted = load_registered_directed_current(elemental, vsv)
    result = contract_directed_current_pair_v2v(
        insertion, adapted["raw"], adapted["contract"],
        insertion, adapted["raw"], adapted["contract"], adapted["vsv"][0],
        first_anchor_time=0, second_anchor_time=1,
        first_field_ne=Ne, first_bar_ne=Ne,
        second_field_ne=Ne, second_bar_ne=Ne,
    )
```

登记的 `elemental_vv` 须具名提供 `component,time,momentum,sink_mode,source_mode` 轴，其中前三轴可换序；分量须包含八个单链路径，额外路径不会进入此缩并。时间须覆盖全部物理时间，动量为明确的 XYZ 表。数据集登记还须声明 `directed_basis="lattice.current.directed-one-link-basis/v1"`、`path_boundary`、`gauge_id`、midpoint 相位、未缩放归一化、实际有序基底及父产物。适配器按具名坐标重排并复制完整 raw；VSV 登记须与其组态、有序基底、自旋基、时间边界及未缩放归一化兼容。`DatasetReader` 检查登记、文件布局和数组契约；适配结果返回 elemental 与 VSV 的 `data_id`，保存缩并结果时应作为父数据引用。未登记数组只能得到布局校验；登记中的科学身份仍取决于登记时提供的真实元数据，软件不会从文件名或相同数组形状推断身份。这个适配入口目前会物化完整八方向数组，不提供按需块读取。

这里检查的是显式链式算符的数据兼容性，分别保留 elemental 与 Dirac 的规范场 ID，
不要求所有输运算符与 Dirac 使用相同规范场。`ConservedVectorInsertion` 是 Wilson 系数与
端点配方；名字和代数迹正确不证明该输入上的守恒性。若测量要宣称严格 Wilson 守恒流，
还须核对实际电流链接与对应 Dirac 链接、Wilson 作用量/参数 `r`、边界及归一化等定义匹配，
并完成相应物理验证。当前 adapter 不接收测量配方，不执行这些额外条件，也未验证 Ward identity。


`SpectralProfile(left, right, coefficients, cfg=...)` 保存 `(profile_basis,time,mode)` 权重，消费 V/V 时计算 `sum(c * left.conj() * E * right)`；两端使用各自的时间。`SpectralProfile.gaussian(eigenvalues,sigma,cfg=...)` 构造 Gaussian 权重，不重新计算本征矢。

## 生产 elemental

已确认的数据与表格前端见[Elemental 前端](../operators/elemental_frontend.md)。初始化提供规范场、本征矢、动量、整合路径与系数的 operator_table，以及需要的点集/扇区；计算只调用 `calc_path_blocks(t)` 或 `calc_derivative(t)`。Ne 可省略并继承输入。后端及调用方正在适配，目标接口与已验收的旧数值核范围分别记录。

不把路径和系数拆成两个输入，也不自动按长度枚举路径。八方向流作为明确单链表格、中点相位和真实时间端点的配方，不另设专用数值计算方法。输出及保存约定、稀疏点块和可选执行参数均以该页为准；实现细节见[EL L2](elemental_production.md)。

## 运行范围

数值接口已在 NumPy CPU 与单卡 CuPy/V100 上验证。MPI 域分解和分布式归约仍待接入。单强子接口当前提供系数与顶点，不替代完整 Wick 图收缩和产物台账。小格点生产与独立参考见[elemental CPU 回归](../../../tests/production/elemental/test_elemental.py)，生成/保存接线见[测试夹具](../../../tests/fixtures/elemental_script.py)；在已分配 GPU 的节点上运行 `python tests/run_elemental_cuda.py --weak-field-dir DIR --report FILE` 可复跑数值回归与分块性能测试。

## 共享工作区与失败恢复

介子与重子生产器各自持有 `production.elemental.ElementalWorkspace`，负责规范场加载、空间 stout、SU(3) 投影、相位缓存及输出缓冲复用。现有构造与计算入口保持不变。

工作区 `load(key)` 从句柄复制方向优先的连续规范场；`links` 返回 `(4,Lt,Lz,Ly,Lx,3,3)` 借用数组。调用方不得修改链接或 `phase(...)` 返回的相位。`buffer(name, shape)` 返回可写 complex128 缓冲，同名同形状调用会复用内存；保留结果时需复制。将借用值的有效期限制在所属生产器下次加载或计算之前，不在并发生产器之间共享工作区。

相位只依赖几何与相位约定，重新加载或变换规范场后可继续缓存；输运场仍由每次计算重新构造。加载或变换失败会使规范场失效，必须重新 `load(key)`。空间变换保留时间链接，沿用已有数值核。本次提取已通过 CPU 回归，真实 GPU 运行尚未复验。


`Insertion.little_group_projection` 的非零动量分支需要 `correlator` 可选依赖；
缺少 SymPy 或 lazy-object-proxy 时提供[符号依赖安装指引](../correlator/correlator_frontend.md#符号构造)。
零动量沿用原来的原对象返回，generic rows 仍按原约定先拒绝不适用的投影。
异常提示不改变实际表示、系数、行序或已验收范围。
