# 关联函数符号前端

实现与接线详见[L0/L1](../../architecture/correlator-l0-l1.md)及[各模块 L2](../README.md)；
本文侧重实际使用方式，设计中的待实现接口不能直接作为调用示例。

算符确定后的常规流程使用固定入口和已有少量选项，不要求用户编写 prepare、producer、loader
或保存回调。空间介子/流需要缺项自动生产时，使用
`from lqcdweave.contraction.correlator import Elemental, calculate`，声明登记的规范场、
V/D 基底和可选点集，之后直接 calculate；后端检查覆盖、复用、补产和保存。
计权只设置 calculate 的 `elemental_weighted`，只规划不生产。见[完整用法与范围](managed_elemental.md)。

重子的具体算符也使用 `Insertion` / `Operator`；“自旋表示 × 空间规范核表示 → 目标表示”
的自动构造、重复表示份数及非零动量小群投影见[重子表示耦合](../operators/baryon_irrep_coupling.md)。

## 日常计算、复用计划与保存结果

普通、完整 cfg、窗口和兼容 session 数值计算均返回 `Calculation`。统一用 `.run()` 执行、
`.value` 取原数值/流/发布记录、`.report` 取映射报告；只规划也有状态报告。
完整说明及旧字段兼容见[共同返回约定](calculation.md)。

需要清楚区分声明、规划和执行时，用 `ContractionTask` 保存整个任务：

```python
from lqcdweave.contraction.correlator import ContractionTask

with ContractionTask(correlator, vertex_map=vertices, propagator_map=propagators,
                     configuration=cfg, time_axes=times) as task:
    task.plan()
    task.save_plan("plans/task.json.gz")
    result = task.run()
```

构造不计算；`plan()` 完成全部请求坐标，`run()` 读取当前数据并执行。
换进程时使用 `ContractionTask.load_plan` 并提供当前 reader；无需重建 correlator，
也不会重新规划。内存与 Dataset 输出共用完整计划，批次范围及生命周期见
[任务用法](calculation.md)。下文保留原 `Calculation` 的固定模板与兼容调用。

固定生产任务先把全部兼容通道、表示行、方向与径向需求组成一个嵌套请求，
在代表 cfg 上 `calculate(..., execute=False)` 一次选路，随后固定回放。
`partial(calculate, ...)` 只共享参数，每次调用仍可能重做展开和规划，不能替代计划复用。
`.run()` 保持原固定绑定；`.run(output=..., configuration=cfg, propagator_map=...,
vertex_map=..., time_axes/time_pairs=...)` 通过原生有界窗口服务绑定当前 cfg 并保存全坐标，
不接受 optimize/search/profile/新算符等规划选项，不重新展开或搜索。
普通 Calculation 与完整 cfg handle 重复 `.run()` 时自动复用适用的执行表；固定单坐标
输出还复用装配绑定。每次仍读取当前输入并交付独立结果，无需新增开关。
换计划/后端、debug 或可变扩展描述按规则重绑；缓存不进入档案。
`vertex_map` 在此使用完整档案的 reader-role 约定：单一 elemental 可直接给当前 reader。
有 compact 时间时显式给 `physical_times/physical_lt/time_roles`，不假定不同 cfg 的实际时间相同。
示例见[meson55 绑定](../operators/meson55.md)，边界见[固定回放 skill](../../../.agents/skills/lqcdweave-correlator/references/plans.md#联合任务与一次规划反复回放)。
它不会自动成为跨 cfg 的完整档案，也不提供跨 cfg 连续窗口流；完整档案 iterator 入口继续沿用。

时间标签可直接用用户选定的字符串，例如 `sink`、`source`、`current`；同一个 key
表示同一个时间变量。符号声明与 `time_axes` 使用同一 key。名字没有内置物理含义；
输出扫描轴严格按 `time_axes` 声明顺序追加，整数 key 仍是有效的显式标签。

### 配对的计划保存与导入

```python
from lqcdweave.contraction.correlator import (
    calculate, complete_calculation, save_calculation_plan, load_calculation_plan,
)

# 立即规划 / 在当前绑定上执行
planned = calculate(correlator, **common, configuration=cfg, execute=False)
result = planned.run()

# 固定绑定档案：载入时显式提供当前已绑定、结构兼容的 expected
save_calculation_plan("fixed.json.gz", planned)
restored = load_calculation_plan("fixed.json.gz", expected=planned)
result = restored.run()

# 跨 cfg 完整档案：先完成全部坐标、输入准备和资源安排，再保存
# selected 是单坐标的已选计划；coordinates 由 describe_coordinates 构造
complete = complete_calculation(selected, coordinates=coordinates,
                                readers=current_readers)
save_calculation_plan("complete.json.gz", complete)
restored = load_calculation_plan("complete.json.gz", configuration=cfg,
                                readers=current_readers)
result = restored.run()
```

`expected` 模式读取普通固定绑定档案，未给 `expected` 时读取 complete-cfg 档案，
两种模式不会暗中转换。普通模式也可传当前 `ExpandedCorrelator`，此时须提供 `hardware`。
完整模式可用 `source_factory(cfg)` 提供每个 cfg 的 reader-role mapping，或给显式
`propagator_map`/`vertex_map`；三种绑定方式不能混用。`calculate(plan_archive=...)`
是完整模式加载后执行的便利入口。所有载入都不搜索、不重新规划；改变坐标或输出布局应重新完成计划。
旧档案不能冒充新计划；高级资源、档案迁移及平台限制见[计划档案](../plan_archive/plan_archive.md)。

仅完整 cfg 的 Dataset 输出后处理成为瓶颈时，可在补全计划或显式窗口入口选择
`postprocess_backend="gpu"`；默认仍为 `"cpu"`，GPU 选择随完整计划保存。
它要求 GPU 缩并目标与独立的输出缓冲预算，不改变 dt/逐对输出含义；
调用、示例与同设备验收见[可选 GPU 输出后处理](../results/gpu_output_postprocessing.md)。

### 保留结构的数值结果保存

```python
from lqcdweave.contraction.correlator import save_correlator, load_correlator

save_correlator("results/cfg-100", result)
loaded = load_correlator("results/cfg-100")
values = loaded.value
# loaded.axes / coordinates / field_axes / field_coordinates 保留输出轴与实际坐标
```

在 `calculate(..., dtype="complex64")` 或 `ContractionTask(..., dtype="complex64")`
中固定整个任务的数据类型，传播子、elemental、自旋、谱权重与数值系数按声明类型进入处理。
读入的数据块先转换，再共轭、计权和缩并；规划使用转换后的类型估算成本及内存。
支持 float32/64 和 complex64/128，实数声明不能丢弃复数输入的虚部。
省略声明时兼容原来的最低浮点分量精度选择，例如 complex64 与 complex128 统一为 complex64。
新计划的 float32/complex64 输出表达式内部自动使用 float64/complex128，保留小乘积与相消
求和的精度，最后才转换输出字段；输入和大矩阵乘法精度保持原约定。该选择及临时内存额度
写入计划，CPU/GPU 回放一致；旧档案仍执行原来记录的逐节点精度。
转换按所需数据块执行，原始文件不改写；已有混合精度计划需重新准备。

默认保存完整数值结果，保留 dict/list/tuple/ndarray 容器、算符维度、计算 dtype、时间轴顺序与实际坐标，
不依赖 pickle，也不重新生成符号算符。`Calculation` 自动提供时间扫描元数据；原始数组/嵌套值
须显式传 `axes` 和 `coordinates`。结构化声明的算符轴自动保存表达式/味标签；没有显式声明的算符维度使用稳定位置名，可用 `field_axes`
及 `field_coordinates` 为每个字段路径声明实际算符标签。对于 cfg iterator，应逐个保存其
`(cfg, value)`，并传该任务的坐标元数据；不要把整个迭代器当数组保存。

源时间平均只在保存时显式启用：

```python
save_correlator("results/cfg-100-averaged", result,
               average_tsrc=True, source_axis="source", sink_axis="sink",
               physical_lt=Lt, boundary="periodic", coordinate_domain="physical")
```

上例要求结果的扫描轴就是所声明的两个绝对物理时间轴。`coordinate_domain="physical"`
必须显式声明；紧凑槽数据改用 `coordinate_domain="slot", selection=record`，物理 Lt 来自记录。未知域拒绝平均，不能把 range(Nt) 当作真实时间。先按
`dt=(sink-source)%Lt` 对齐，再按每个 dt 实际参与的源汇对数平均；不是对固定 sink
简单求均值。算符轴保留。输出完整 `dt=0..Lt-1`，未覆盖的 dt 默认报错；显式 `missing="nan"` 才用 NaN 表示且计数为零。
约定、原始坐标、计数和来源随结果保存。重新导入保持已平均结果，不能还原被舍弃的源时间。
已有相对时间布局须显式给 `relative_axis` 和 `source_axis`，其坐标必须是各源共用的真实物理 dt，只对源轴平均。
源槽×循环差槽的第二轴不是物理 dt；保存时显式 `coordinate_domain="slot", selection=record, time_layout="cyclic_difference"` 和 `relative_axis`，再按真实差值配对。
额外时间变量、时间对列表或非周期语义不自动推断，必须先明确转换/估计量。
详细原始值 API 与支持限制见[数值结果存储](../storage/correlator_results.md)。

计划与结果读回的 CPU 接线见[前端往返回归](../../../tests/test_correlator_frontend_roundtrip.py)，
dt 归约与独立参考见[结果存储说明](../storage/correlator_results.md)。合成数据不作为生产系综或 GPU 验收。

## Diagram 构造之后：统一计算入口

选择 meson55 时使用下面的预制绑定入口；其它已有顶点沿用通用映射或绑定函数。
`time_axes` 的值必须符合 reader/adapter 的坐标契约；输出按声明顺序追加独立扫描轴，不会自动转为循环差槽。

### 从 meson55 目录选择算符

已登记的 meson55 elemental 使用 `RecipeVertex` 预制绑定。用户从目录选择符号算符，
再用普通 `Hadron` / `gen_correlator` 组织味道与 Wick 图；计算时统一传入 `vertex_map=mesons`。
绑定器自动生成 Dirac×空间耦合顶点并处理源端 dagger，不需要逐算符手填绑定字典：

绑定默认选 v2。消费历史 CSNS 的四个 G 行时显式使用 `RecipeVertex(elemental, version="csns")`；使用修正 G 核则按 v2 生产或补充对应 elemental。其余 51 行保持相同完整空间身份，可跨版本使用；绑定按身份匹配，详见[版本边界](../operators/meson55.md#低分波投影检查)。

```python
from lqcdweave.contraction.binding.api import RecipeVertex
from lqcdweave.operators.correlator.api import Flavor, Hadron, Tag

mesons = RecipeVertex(elemental)
pion = Hadron(mesons.symbol("op_00_gamma5_A1", tag=Tag(0, 0))[0], Flavor("ud"))
```

输入加载、完整计算调用、用户选择项与预制职责统一见
[meson55 使用流程](../operators/meson55.md#从登记-elemental-构造-correlator)；目录筛选和移动动量见
[算符选择](../operators/meson55.md#名称目录查询与移动动量)。当前目录绑定片段见
[数据绑定参考](../../../.agents/skills/lqcdweave-correlator/references/binding.md#从-meson55-目录绑定预制算符)。

### 已有顶点与 reader

使用压缩时间输入前，先读[共同时间交接约定](../temporal_sampling/temporal_production.md)。把 selection 显式绑定到 Array/preset reader，或从 Dataset 共享目录登记读取，再向 `calculate` 传槽坐标 `range(Nt)`。新持久数组按 cfg 保存整块 NPY/HDF5，并由共享根 `dataset.json` 登记；详见[数据保存格式](../storage/data_storage.md)。源×绝对汇槽输出可直接交给 selection 两点重建，默认布局为 absolute；旧循环差槽必须显式声明。域来自 reader/登记记录，不从 shape 或扫描长度猜测。源汇可有独立有序记录，绑定按每条传播子的实际端点角色核对。

```python
from lqcdweave.contraction.correlator import calculate

result = calculate(
    correlator,
    propagator_map=propagators, vertex_map=bind_vertex,
    configuration=configuration, usedNe=Ne,
    time_axes={0: range(Lt), 1: range(Lt)},
)
values = result.value
```

`propagator_map` 将图中的传播子标签映射到已有 reader 或支持的 preset 数据源；`vertex_map` 是顶点到 `Meson/Baryon/Current` 的映射或构造该对象的函数。用已加载 reader 构造顶点时无需再次调用 `load`；未加载的 preset 声明由 `configuration=cfg` 自动解析，顶点定义保持不变。时间扫描按声明的轴顺序追加到输出；例如 2×2 correlator 返回 `(2, 2, Lt, Lt)`，无需手动 reshape。dict、list、tuple 及嵌套结构保持原类型。这个示例采用入口默认执行和规划设置；若任务要求指定分批、存储策略、内存预算、校准 `profile` 或计时报告，仍须明确传入这些必需选项。示例省略它们不表示可以删除任务约束。

```python
# 任务需要这些控制项时，再按实际输入和资源设置覆盖默认值。
result = calculate(
    correlator, propagator_map=propagators, vertex_map=bind_vertex,
    configuration=configuration, usedNe=Ne,
    time_axes={0: range(Lt), 1: range(Lt)},
    batch={"axis": 1, "size": 2}, optimize="anytime", storage="bounded",
    memory_limit=14 * 2**30, host_memory_limit=12 * 2**30,
    profile=cost_profile, measure=True,
)
```

`kernel_batching` 默认为 `False`，跳过 CPU/GPU 普通缩并的 Cartesian 合批，并在 GPU 上跳过独立 GEMM 分组；联合规划、共享中间结果与资源核算仍会执行。它与时间坐标的 `batch` 参数独立，只影响新规划；档案回放沿用保存时的选择。沿用现有调用参数映射时，可这样开启合批：

```python
planned = calculate(correlator, kernel_batching=True, **bindings)
```

这里的 `bindings` 是原来传给 `calculate` 的关键字参数映射；需要先返回计划时，在其中保留 `execute=False`。

### 把共同定义放在 cfg 循环外

多 cfg 的任务领取、完成核验与恢复统一见 [Dispatch](../dispatch/dispatch.md)。共同算符、完整通道容器、
数据源声明及径向定义放在领取循环外；可在代表 cfg 或首次实际领取后惰性选定一次模板。
随后只使用已选模板的 `.run(output=...)`，或下一节完整档案的 iterator 入口。
当前 cfg 的实际基底、时间选择、池关系及来源仍逐 cfg 绑定核验。

原生 preset 声明支持的类型见[preset](../storage/preset.md)。普通 `calculate(configuration=cfg)`
每次创建独立绑定，不会修改旧 Calculation；这是一种单次绑定能力，不等于跨 cfg 复用规划。
`partial(calculate, ...)` 和 prepare_* helper 只共享参数/定义，不能作为固定生产的复用方案。
多通道、轻重不同模数及当前 reader 生命周期的约定见[meson55 绑定](../operators/meson55.md)。
未知数据源继续使用已支持 adapter；不要通过全局缓存或替换文件绕过当前 cfg 的身份检查。

### 保存完整计划，重复 cfg 直接执行

需要复用路径、跨时间对/cfg 连续预读或流式产物输出时，使用完整 cfg 档案。
先为代表 cfg 完成参数化坐标、资源与输出安排并保存；之后可省略 correlator：
`calculate(plan_archive=archive_path, configuration=Dispatch(cfg_file, run_suffix), ...)`。
保留这个 iterator 入口供跨 cfg 连续预读，任务分配规则及接线见 [Dispatch](../dispatch/dispatch.md#跨-cfg-连续预读)；只有主线程逐项领取，后台读取不领取 cfg。其它 cfg iterator 的支持保持不变。
当前数据源通过原 reader 映射或 `diagram_adapter=source_factory` 绑定，执行不展开或规划。
输出模式和缓冲在保存前固定，输出根目录可在加载时更换。具体调用、返回值及 CPU/GPU
边界见[矩阵直接执行](../execution/meson_matrix_execution.md)与[生产脚本与保存](correlator.md)。

多 cfg memory 的 `result.value` 仍逐 cfg 交付完整 `(cfg, values)`。GPU 内部在保存额度允许时
自动用双槽重叠准备/计算、下载和单点组装；不用增加前端开关。提前停止迭代时用
`contextlib.closing(result.value)` 或显式 `result.value.close()`，使在途工作和 reader 完成清理；
已经返回的数组可保留。计划及结果持久交接见[保存/回放回归](../../../tests/test_correlator_frontend_roundtrip.py)。

显式适配器通过 `diagram_adapter=` 传入，`data=` 保留为兼容别名。其 `describe(correlator, bindings=..., coordinates=..., batch=...)` 只根据元数据生成全部 Term、物理块目录和延迟 loader。数据位置、物理约定与图到数据的映射属于适配器，不是优化参数。

原生读取器也可直接通过 `propagator_map`、`vertex_map`、`configuration` 和 `usedNe` 传给 `calculate`；前端会注入相应适配器，不需要单独调用 prepare/load。底层 `expand_correlator` 保持纯 contraction 层接口，调用时需显式传入 `DiagramData` 适配器。

| 参数 | 含义 |
|---|---|
| `optimize` | `anytime`（默认：按当前操作共享分类，先取得各类首解，再按基础名额及边界同分规则继续 DFS；按计算成本选路，随后重排固定 DAG 并审计物理存储）；`exact`、`dynamic_programming` 在既有原子操作域求完整总时间最优；`none` 使用固定 direct 规则；`meson_matrix` 是 `anytime` 的兼容别名 |
| `storage="bounded"` | 搜索接受此历史参数并统一解析为 lifetime；anytime 对选定路径做物理审计，exact/DP 在搜索中纳入物理生命周期；direct 模式使用固定计划审计 |
| `storage="lifetime"` | 搜索路线的生命周期策略；最终计划检查原始块、逻辑张量、工作区和输出峰值，搜索过程中的约束范围随 optimize 策略确定 |
| `execute=False` | 只使用 shape 等元数据规划和预测，不调用数值 getter |
| `search={...}` | 统一预算：`deadline_seconds`、`max_states`、`max_transitions`、`search_memory_limit`；anytime 默认 `root_width=4` 严格限制根候选数，`branch_width=4` 为后续层基础名额，默认 `include_ties=True` 仅在后续层保留边界全部同主分候选；同分内优先最近改写项重叠较多的操作。根分支默认自动按本机 CPU 配额并行；高级 `component_workers=1` 可选择串行参考，正整数限制总 worker 数。详见[规划模块](../planning/planning.md#公开模式和参数)。`widen` 已移除，显式传入会报错 |
| `measure=True` | 普通执行的校准计时，返回 `result.report`；完整 cfg 不接受该开关，直接报告实际阶段耗时 |

预算单位是字节；CPU 的 `memory_limit` 是主机预算，CuPy 是设备预算。显式进程预留可通过 `reserved_host_bytes`、`reserved_device_bytes` 给出，适配器另声明输出组装空间。未提供校准 `profile` 时使用默认模型，预测不代表实际硬件耗时。anytime 等启发式没有全局最优保证。

搜索路线兼容传入历史的 `storage="bounded"`，当前统一解析为 lifetime；默认参数为 `storage=None`。
`anytime` 在逻辑搜索完成后重排固定计算 DAG，再审计物理生命周期；保持完整模型时间，审计超限不会自动重新选路；
`exact` 和 `dynamic_programming` 在搜索中纳入完整物理生命周期成本与约束。
这些路线不会以全部原始块常驻预留代替生命周期审计。
其中 `exact` 和 `dynamic_programming` 仅支持一个完整 Problem，task/auto batches 会在读取前报 `NotImplementedError`，
避免把逐批最优宣称为全任务最优。预算耗尽时返回此前已审计的最好完整方案；没有完整方案则报
`PathSearchExhausted`，不表示已证明无解。精确搜索穷尽操作域并证明无解时才报 `NoLegalContractionPath`。
统一控制项见[规划模块](../planning/planning.md#统一-anytime-优化器的模块设计)。

```python
planned = calculate(diagrams, data=data, optimize="anytime",
                    storage="bounded", memory_limit=2**30, execute=False)
print(planned.planning_seconds, planned.prediction)
print(planned.plan.peak_host_bytes, planned.plan.peak_device_bytes)
result = planned.run(measure=True)  # 使用同一计划，不再次搜索
```

普通 `Calculation.run` 重用当前绑定的数据及计划；换组态或端点应重新绑定，不能只替换文件。
原生普通扫描会合并任务的物理 slab 需求，每种兼容批形状搜索一次，再重绑定后续批次，
尾批保持真实长度。多次独立调用不共享这一生命周期。完整 cfg 的 `plan_archive` 路线另支持
参数化重绑、连续预读及 `output` 流式写出；不要把普通局部路径与完整档案混用。
计划档案使用 `storage.contraction_plans.api`。

### rho–ππ 矩阵示例

此测量使用已登记的 VSV 和 raw-path elemental，组成零动量 rho 与一动量 ππ 的 2×2 矩阵。
elemental 覆盖 `(0,0,0)`、`(0,0,1)`、`(0,0,-1)` 动量和空规范路径；实际有序动量表、
基底、组态、相位与时间覆盖在绑定时核对。符号声明沿用 `Hadron/Flavor → gen_correlator`，
完整矩阵交给上文的 `calculate`；不依赖已退出仓库的 `prepare_matrix` 项目 helper。

输入 readers 在规划和回放期间保持有效，缺失数据不自动生产。历史小格点 rho–ππ 对照不代表
当前代码的真实 GPU 或生产规模性能；结构化矩阵的维护回归见[CPU 接线测试](../../../tests/test_structured_correlator.py)。

### 重子守恒流三点函数

用 `Hadron/Flavor` 定义外态与守恒流，保留实际 `(sink,current,source)` 物理时间，
由 `gen_correlator` 生成关联函数，再绑定当前 `Baryon/Current` 和传播子 reader，调用同一 `calculate`。
源端共轭及三腿对应关系见[重子说明](../operators/baryon.md)；固定时间三点与多点的定义见[多点测量](multipoint_template.md)。
普通调用不提供历史 `prepare_three_point` 脚本包装。生产循环按[完整计划复用](../../../.agents/skills/lqcdweave-correlator/references/plans.md)
读取共同档案，不逐 cfg 构造新计划。

### 已提供的符号 Diagram 数据适配器

`adapters.vsv_diagram.VSVData` 接通等时介子 V/V 图的 CPU 参考路线：配置 `time_map`、`vertex_spec`、`propagator_spec`、两个 getter 和 `dataset_identity`。spec 回调与 getter 的参数相同，但只返回 `TensorSpec`；数值读取时校验 shape/dtype。调用方负责核对实际基底、覆盖与物理约定，getter 返回独立所有权的数据。同名图只生成一个 Term，表达式系数、乘积及矩阵轴在最后恢复。

适配器契约见[展开模块](../expansion/expansion.md)。每个请求视为一个物理块，不识别文件 slab 或 Ne/Np 前缀。生产数据、重子、点端和跨时间流需要对应的 `DiagramData` 适配器返回 `contracts.correlator.api.DiagramInputs`；旧 `contraction.diagram_inputs` 转发路径已删除。统一入口支持已有 scene 内核，但此参考适配器不扩展这些物理能力。现有外层 campaign 尚未改接这个入口，运行中的快照保持原样。

## 符号构造

旧工作流的纯符号调用方式保留，包名前缀改为 `lqcdweave`。此可选依赖组支持 Python 3.11 或更新版本。格点群表示和投影由正式模块 `lqcdweave.symmetry` 提供，接口与依赖边界见[格点对称性](../symmetry/README.md)。在仓库根目录安装：

```sh
python -m pip install -e ".[correlator]"
```

该 extra 会安装 SymPy、SciPy、lazy-object-proxy 和 HDF5 支持；最小安装层仍只含 NumPy 与 opt_einsum。

Windows 会随此依赖组安装 psutil，以查询默认规划预算所需的实际可用主机内存；无法查询时须显式提供内存预算，不能猜测容量。

```python
from lqcdweave.operators.correlator.api import (
    Tag, HadronFlavorStructure, operator_transform, Hadron, gen_correlator, HadronIrrep,
)

row = HadronIrrep("pi", [0, 0, 0], "A_1", -1, Tag(0, 0))[0]
hadron = Hadron(row, HadronFlavorStructure("ud"))
correlator = gen_correlator([[hadron], [hadron]], [0, 1], [False, True])
rotated = operator_transform(correlator, "iden")
assert rotated.shape == (1, 1)

```

`0`、`1` 是源、汇的符号时间；下方有界 CPU 数值参考对每个实际 `(source_time,sink_time)` 传一次整数映射。`Hadron.set_time`、`gen_correlator` 的时间列表和 dagger 列表沿用旧语义。`gen_correlator` 与旧版一样会把传入的 `halves` 列表元素替换为带时间和 dagger 的实例；同名符号沿用旧版 SymPy 的合并规则。需要区分不同顶点时，算符名称、动量或 dagger 必须提供足够身份。`Diagram` 保存符号时间、顶点和传播子标签；`quark_contract`、`diagram_simplify` 和 `remove_disconneted_diagram` 是数据无关的图操作。

单强子插入的旧式调用顺序也可用新包名写出：

```python
from lqcdweave.operators.single_hadron import (
    DerivativeName, GammaName, Insertion, Operator, ProjectionName,
)
from lqcdweave.operators.single_hadron.mom_dict import momDict_mom1

ins = Insertion(GammaName.PI, DerivativeName.IDEN, ProjectionName.A1, momDict_mom1)
operator = Operator("pi", [ins[0](0, 0, 0)], [1.0])
```

此处的动量字典也可直接传给 `ElementalGenerator` / `BaryonElementalGenerator`
的 `momentum_dict=`，生产与算符选择共用字典值的插入顺序，key 不作为存储槽号。
值支持整数 XYZ 三元组或三整数字符串；保存时使用生成器的有序 `momenta`。
自定义表及完整配套示例见[动量与规范链接表](../elemental/elemental_presets.md)。

符号构造入口本身不读取文件或生产缺失数据。把图绑定到已有 elemental、传播子时，须由 `planning` 和读取适配器核对实际动量顺序、路径、基底、时间覆盖与约定。当前已登记数据的完整时间片连通介子两点数值入口仍见[介子工作流](../measurements/meson_measurement.md)。

对于已显式绑定的**等时介子 V/V 图**，可用有界 CPU 参考入口按物理时间求值：

```python
from lqcdweave.contraction.vsv_graph import evaluate_vsv_matrix

values = evaluate_vsv_matrix(
    correlator, vertex_getter, propagator_getter, {0: source_time, 1: sink_time},
)
```

`vertex_getter(row, time)` 返回 `(left_spin,right_spin,left_mode,right_mode)` 顶点，`propagator_getter(tag, sink_time, source_time)` 返回 `(sink_spin,source_spin,sink_mode,source_mode)` VSV 块；两者须来自已核对身份、相位、模态顺序和覆盖的数据。该入口只支持每个介子顶点一条入线和一条出线，不支持点端、重子或跨时间 current，也不承担持久输出和恢复。rho–ππ CPU 参考测试 使用 3³×2、Ne=2、全部时间片，从合成 SU(3) 规范场生产基底、VSV 和 elemental，构造 `ρ0/ππ(p_z=±1)` 符号矩阵并收缩；需安装 `.[laplace,correlator]`。对于一个显式绑定的固定 scalar Diagram，现可用持久工作流消费已登记的 `DatasetPropagator` 与 `Meson(DatasetElemental)`，覆盖所有物理源汇时间、逐源行恢复并生成 receipt；身份、点端边界与 CPU 限制见[持久测量说明](../measurements/diagram_measurement.md)。该有限路线不生产缺失输入，也不代表任意图结构和测量都已接通。

符号表示、味结构与实际单强子配方的物理对应由调用方明确选择；组表示或 C 宇称不能从数组形状推断。完整旧功能迁移状态仍以项目外严格登记表为准。

## 统一前端测量参考

以下维护文档说明相同前端下的不同测量定义，数值绑定消费当前已打开的 reader：

- rho–ππ：见上文的 2×2 单粒子/双粒子矩阵说明。
- [重子及守恒流](../operators/baryon.md)：保留外态、流与实际端点的映射。
- [弱三点与双守恒流四点](multipoint_template.md)：明确动量转移、各流分量与有序插入。

统一用 `HadronIrrep`、`Flavor` 描述抽象粒子，`hadron_little_group_projection`
构造耦合算符，`gen_correlator(..., degenerate=True)` 生成关联函数。
`time_slice_list` 中显式的 `None` 保留该组合内每个粒子的时间；原来的整数项仍统一改写时间。
`degenerate=False` 保留味道传播子标签，计算时应提供相应 `propagator_map`。

具体插入统一交给 `Operator` 构造，再绑定 `Meson`、`Baryon` 或 `Current(Meson)`。
重子 `BaryonComponent` 的路径键描述空间结构；兼容用的 `flavors` 元数据不定义物理味道，
味道在符号 Wick 项和传播子绑定中表达。
重子三腿使用普通 `Insertion` 和 `Operator`：静止系局域行可用 `Insertion.local_spin`，
一般自旋表示与空间表示的耦合使用 `Insertion.from_irreps`；显式投影行仍可用
`Insertion.from_rows` 表达。
守恒流使用 `ConservedVectorOperator(Operator)` 预制组合，保留逐项真实端点；
手写相同的 `Operator.from_terms` 项可表达等价计算。
`calculate` 自动选择已有原生绑定，不要求用户指定重子适配器。
在此范围内，统一入口把重子自旋核心与三模态空间核心分别供给现有 `baryon_join`，不再预先展开六阶顶点。自旋基变换与源端共轭仍由绑定完成；可用的非零自旋结构随计划传入执行层，模态求和保持稠密。`none` 与搜索规划均支持这一交接，未匹配稀疏实现的结构保留稠密计算。
直接用已绑定 `Baryon` 读取运行时生成的单个 VVV elemental 块，见[按需重子 elemental reader](../elemental/baryon_on_demand.md)；该入口不发布持久产物。
实现目录、接口交接与模块测试见 [L0/L1](../../architecture/correlator-l0-l1.md) 和[各模块 L2](../README.md)。
半轻衰变例子输出裸Euclidean强子矩阵元，不包含轻子、弱耦合系数或衰变率分析。


移动中子弱三点的固定例子取 p_i=(0,0,1)、p_f=(0,0,0)、q_insert=p_i-p_f，输入必须覆盖移动基底与流动量。双守恒流四点例子是一项有序 c/u 交叉贡献，以两个 `ConservedVectorOperator` 绑定到同一 `calculate`，不自动求和电磁电荷因子。


## 基础导入与缺少可选依赖

基础包、单强子基本结构和 `contraction.results.api` 可在未安装 SymPy、SciPy
或 lazy-object-proxy 时导入。R 的 `assemble_correlator` 消费已构造的中立
OutputRecipe/TermValue，`assemble_workload_outputs` 消费已保存的 workload
recipe；二者不构造符号 Diagram。它们保留原坐标、系数和数组所有权语义。
`collect_diagrams` 与 `build_output_recipe` 在调用时才加载唯一 S Diagram 和
SymPy；符号功能始终需要上述可选依赖组。

`operators.correlator`、其 `api`、固定 `contraction.diagram_measurement` 与
`contraction.vsv_graph` 是符号功能入口。缺少 SymPy 或 lazy-object-proxy 时，
ModuleNotFoundError 指明缺失依赖并提供 `.[correlator]` 安装步骤，保留原异常
作为 cause。缺少 SciPy 的 CPU Laplace 求解应安装 `.[laplace]`；完整 CPU extra
同时包含 SciPy、SymPy 和 lazy-object-proxy。其它缺失的传递依赖、坏安装中的
ImportError 或运行错误继续原样传播。
单强子非零动量小群操作在调用点提供同样指引；零动量仍立即返回原对象。
不会生成假的 Diagram、替代投影结果或伪完成状态。

含暂存/恢复动作的方案当前仅支持元数据规划和审计，执行及保存会在 IO 前拒绝。无暂存计划仍可执行和保存。跨 task batch 暂存衔接、真实临时文件和设备迁移待后续补充。
# 介子本征模分块

可在关联函数生成后调用 `split_eigenmodes`，再将全部小块一次交给 `calculate`。
两条模式腿的区间同时传给 elemental 与相连传播子；PC 保持和原函数恢复通过
`project_meson_mode_pairs` 完成。接口、轴顺序与当前限制见
[介子模式分块](../operators/meson_mode_blocks.md)。

仅为降低计算存储而分块、无需保留径向结果时，使用
`calculate(..., partitions=[0,16,32,48,70], partition_mode="memory")`。
返回值与未分区时具有相同的算符和时间轴；默认 `partition_mode="radial"` 仍保留
径向分区输出。固定 cfg 批处理也可预先调用
`split_meson_mode_pairs(..., partition_mode="memory")`，之后无需重复传分区选项。
实际物理块、预算和完整档案保持现有机制；总峰值仍取决于输入、块结果和缓存。

## 显式介子矩阵分组

用户可将各量子数的 `gen_correlator(..., structured=True)` 请求放入嵌套 dict，
调用 `calculate(..., optimize="meson_matrix", partitions=[0,16,32,48,70])`。
该固定流程直接生成整行共享计算，不搜索缩并顺序；档案可选。已有分区请求
无需再传 partitions。当前支持 NumPy dense V/V 介子矩阵和独立单圈，完整
系数、模态投影及扫描输出保持；详见[接口和完整流程](../execution/meson_matrix_execution.md)。
