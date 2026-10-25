# 使用已有数据

历史projected Dataset缺少目录标签、但保存了完整路径系数时，可显式绑定
`DatasetElemental(reader, catalog=meson55_preset(version="v2"))`。
后端逐行比较所有有序路径系数后才交付目录分量，保留原data_id及父项，不根据文件名推断。

介子、空间流或重子 VVV 缺少 elemental 时，普通用法是绑定受管 `Elemental` 输入声明并直接 calculate，
由后端复用或补产，不要求用户写生成器或准备回调。范围与示例见[自动准备](../../../../docs/modules/correlator/managed_elemental.md)。

## 从 meson55 目录绑定预制算符

已登记的 elemental 和 perambulator reader 传给 `RecipeVertex` 与 `calculate`。按冻结目录名称选择符号，
绑定器负责生成耦合后的 `Meson` 并传递源端 dagger：

```python
from lqcdweave.contraction.binding.api import RecipeVertex
from lqcdweave.operators.correlator.api import Flavor, Hadron, Tag, gen_correlator
from lqcdweave.contraction.correlator import calculate

mesons = RecipeVertex(elemental)
pion = Hadron(mesons.symbol("op_00_gamma5_A1", tag=Tag(0, 0))[0], Flavor("ud"))
correlator = gen_correlator([[pion], [pion]], [0, 1], [False, True], structured=True)
result = calculate(
    correlator, vertex_map=mesons,
    propagator_map={"S^q": perambulator, r"S^q_\mathrm{local}": perambulator},
    configuration=cfg, time_axes={0: sink_coordinates, 1: source_coordinates}, usedNe=Ne,
)
```

这里的 `elemental`、`perambulator` 和扫描坐标已按数据声明准备好。输入加载、完整时间/紧凑槽坐标及其
`selection.times` / `physicalLt` metadata 见[meson55 使用页](../../../../docs/modules/operators/meson55.md)和[时间稀疏化](../../../../docs/modules/temporal_sampling/temporal_production.md)。
绑定默认使用 v2；要保留历史目录 的四个 G 行时，显式使用 `RecipeVertex(elemental, version="csns")`。
修正 G 行需提供对应版本的 elemental；其余 51 行配方相同，但目录引用也按版本匹配，不自动跨版本读取。默认动量表是有序
`momentum_recipe("mom4").momenta`；consumer 顺序必须与 elemental 的实际动量轴相同。预制绑定采用 midpoint
相位和默认周期时间端点约定。空间路径按周期边界生产；绑定消费已保存的路径定义，
不更改生产边界。绑定器从符号行的 `dagger` 字段生成 `Meson(source=True)`。
目录选择、moving irrep 和耦合核验细节见[meson55 使用页](../../../../docs/modules/operators/meson55.md)。
同一 `RecipeVertex` 自动复用不可变算符行定义；重复调用仍返回独立绑定视图，
不缓存数值数据或模式选择，无需在用户脚本中另建算符缓存。

具名原生映射可直接写为 `vertex_map={"pion_sink": sink_vertex, "pion_source": source_vertex}`，键对应 `HadronIrrep.hadron_name`，并必须等于各顶点明确设置的 `Operator.name`。缺失、重复或不一致的名称在加载前报错；源端共轭仍在 `Meson/Baryon/Current(..., source=True)` 显式声明。名字只用于查找，实际物理兼容性仍由元数据核对。绑定仅消费 `gen_correlator` 产生的关联函数；不通过手写内部拓扑另建任务。

`propagator_map` 把图中的传播子名称绑定到已有 reader 或支持的 preset 声明；`vertex_map` 把抽象粒子绑定到 Meson/Baryon/Current 对象或构造函数。未加载的 preset 顶点由原生 `calculate(configuration=cfg)` 建立独立视图，原模板不变。支持范围、循环外定义及 reader 生命周期以[前端说明](../../../../docs/modules/correlator/correlator_frontend.md#把共同定义放在-cfg-循环外)为准；其它格式显式注入 reader/adapter，不把任意文件路径或任意 `load` 对象当作已支持的数据源。

使用现有登记 reader/adapter 并保持其生命周期，实际路径与 cfg 来自本任务输入；
不要求用户实现 load、getter 或 shape 回调。

具体顶点统一绑定Meson/Baryon/Current。Current按Meson的方式使用；守恒流必须有全部相邻端点和路径的数据，不能用只有锚点的普通局域elemental替代。

非简并设置degenerate=False后提供味道对应的传播子，不能仍把所有标签绑定同一轻夸克数据。比较实际组态、基底及顺序、覆盖、相位和边界；相同Ne或shape不证明数据可共用。未知来源不能自动认定一致。

需要新增格式或描述能力时定位已有 adapter/reader 的公开边界；在任务授权范围内完成适配和验证，使用脚本组合这些服务，不复制另一套格式管理。自定义数据交接见[公开模块](interfaces.md)。缺失生产数据转到相应生产 skill，不在算符构造中补算。

## 按任务延伸

<a id="h-binding"></a>
### 投影点 H

H=Q eta 仍由 P schema 的 reader 提供，HSV/HSD/HSH 对应 PSV/PSD/PSP；身份来自
实际点池、完整父空间及 `high_mode_projection`，不能由目录名、Np 或相同 shape 判断。
已投影的 H 块直接绑定到现有 P 扇区，不再次调用 highmode 减除；配套 elemental 使用
同一原始点核，Q 只在传播子端。raw/preweighted 两路沿用同一个 `elemental_weighted`。

比较不同 P/D 数量时，核对 D 的实际混合列选择和包含系数、P 的原标签/共享池，以及各
方案实际 Q。固定父空间内从每组多列改取少列不改变 Q；不同 Q 的块不互相冒充。
完整覆盖按真实物理时间对检查；HSV/HSD 可提供其 Gamma5 反向，但不能推导 HSH。
单流完整另一腿的算符作用源也不是普通 H 交叉传播子的同义产物。
缺基础输入时交给 PR 选择[直接生产或派生路线](../../../../docs/modules/perambulator/point_highmode.md#h-contract)，
不能仅为使用 H 就要求全父传播子，也不能把专项生产脚本视作 calculate 的自动补产能力。

### 其它条件入口

已有完整传播子、而 elemental 使用实际有序紧凑时间槽时，使用原生惰性时间视图：

```python
from lqcdweave.adapters.propagator import select_propagator_times

perambulator = select_propagator_times(original_reader, selection)
```

`selection` 是该 cfg 实际的 `TimeSelectionRecord`，与连接的 elemental 端点一致。
原 reader 保持其声明的完整 absolute/relative 布局及物理 Lt，并保持打开；视图只在
普通 `calculate` 请求时映射实际端点，不预读、重排或写出整个紧凑时间方阵。
例如原始 relative128 数据按实际物理端点模128取差，不按40个入选槽的编号差取模。
原 data_id、cfg、轴、dtype 和端点基底继续由父 reader 提供；完整计划保存实际选择和
父坐标解释，加载仍核验当前 reader。无需脚本 loader、回调、缓存或新的 calculate 参数。
已有紧凑父 reader 也按其明确记录映射，不把数组 shape 当成物理坐标。详见
[惰性时间视图](../../../../docs/modules/data_readers/data_readers.md#传播子的惰性时间视图)。

需要由已有传播子派生高模/验证 PSV dagger 才读[高模任务](../../../../docs/modules/perambulator/point_highmode.md#highmode-task)；不自动生产缺失父项。
需要合并同身份 NPY 前缀读取来减少 IO 才读[共享读取完整模板](../../../../docs/modules/data_readers/data_readers.md#shared-npy)；普通 preset 不要求切换接口。
内存仅准备部分传播子时间对时，`ArrayPropagator` 的 sector 可传
`{(sink_slot, source_slot): block}`，使用 absolute 约定和实际 source/sink 时间选择。
缺项在普通 metadata 规划阶段报错，不能填零伪造完整覆盖；详细示例见
[时间对读取](../../../../docs/modules/data_readers/data_readers.md#只持有实际需要的时间对)。无需新增 calculate 选项。

### 普通反向传播子

普通反向传播子仍是同一普通 reader 的反向端点读取：forward 取
`block(t_sink, t_src)`，backward 取 `block(t_src, t_sink)`。不要自动构造第二份
存储或加 gamma5/dagger；`source=True` 只处理源顶点的约定。PSV→VSP 等明确派生
是另一项布局/物理契约，不能与普通反向边混用。底层 absolute/relative 保存约定仍由
reader 按实际物理端点解释，不能仅交换压缩槽后猜测文件里的时间差。

普通 Diagram 的 D 选择仅支持完整组保留/重排；组内减少混合列需要显式联合列选择估计量，不能直接裁数组后沿用原 m 和来源。专用扫描实现的支持不等于普通 reader 支持，见[D 来源契约](../../lqcdweave-elemental/references/dilution.md#同池混合列包含系数与范围)。

## 胶球已有数据

用 `Hadron(existing_irrep_row, Flavor(""))` 声明胶球，继续通过 `gen_correlator` 构造。
`ArrayGlueball(data_or_npy_path, components=("G",), momentum_list=((0,0,0),),
phase_convention="site", cfg=cfg)` 读取已完成空间求和及规范迹的数据。
默认逻辑轴为 `(component,momentum,time)`，旧文件可显式传 `axis_order` 作视图重排。
分量标签必须对应实际已投影的物理行；相位约定和真实有序动量表由数据来源确定。

从 `lqcdweave.adapters.glueball` 导入 `ArrayGlueball`，从
`lqcdweave.contraction.binding.api` 导入 `Glueball`。
原 `vertex_map` callable 中按符号行返回
`Glueball(reader, "G", momentum=0, source=row.dagger)`；其它行照旧返回 Meson/Baryon。
纯胶球可以省略 `propagator_map`。需要具名映射时显式给 `Glueball(..., name="G")`。
胶球不带 OperatorTerm、mode/spin/color 轴或 V/D/P 扇区。
普通胶球与多强子乘积沿用普通入口；structured 仍只支持线性单强子插入。

胶球值属于当前 cfg 的输入，计划不存数值。`.run()`、具名时间扫描和完整 cfg
计划回放沿用原入口；同 cfg 内乘入其它 term 后再跨 cfg 统计，不先平均输入。
真空期望值扣除由后处理显式完成。完整职责见[架构](../../../../docs/architecture/glueball_architecture.md)。

已登记的 `kind="glueball"` 数据可把原 `Dataset(directory)` 放进 Glueball，
`calculate(configuration=cfg)` 自动绑定。其 definition 必须记录实际回路、投影和归一化，
有序 component/momentum/time 轴、相位和真实父项保存在原 DatasetSpec。
完整档案回放沿原约定注入当前已加载 reader；多 cfg 使用原 cfg source factory。
Dataset 读取不另开格式，HDF5/NPY 均由原存储服务负责。


## 依赖主源行的已有传播子

若已有数据的反向块按当前主源行重建，不能把不同源行伪装成同一个全局 Hermitian
传播子。使用声明了 `read_context_axes=("source",)` 的任务 reader，令其
`describe_block(sink,source,sector,primary_source,...)` 与 `block` 消费显式上下文，
并完整声明实际读取身份。所有源/汇仍由同一公共联合任务一次规划并存档，随后沿保存
的读取表回放；不在逐源循环重新 E/P，也不在运行中切换 reader 全局状态。
普通登记 reader 无须增加此声明。接口和身份要求见
[E 读取上下文](../../../../docs/modules/expansion/expansion.md#显式传播子读取上下文)。
