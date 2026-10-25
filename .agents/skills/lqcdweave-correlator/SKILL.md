---
name: lqcdweave-correlator
description: "为 LQCDweave 构造和计算强子关联函数，包括谱学、介子/重子多点与流插入；从物理算符经 gen_correlator 和 calculate 联合规划、固定回放与验证；测量中的 elemental 自动准备由同一入口管理。"
---

# LQCDweave · 强子关联函数

内核开发/验收见[介子整体顶点](../../../docs/modules/execution/meson-whole-vertex.md)和
[重子准备/计算](../../../docs/modules/execution/baryon-join-preparation.md)。内部方案不新增 calculate 参数；
专门页说明当前支持边界；验证结果按共同约定保存在项目目录，不能把小核 GPU 通过泛化为完整任务通过。
单合并腿的整体重子内部方案见[标量归约](../../../docs/modules/execution/baryon-scalar-sum.md)：
沿用已选计划和批宽，准备读取当前数组，返回独立结果；不增加前端设置。
普通 Diagram 整体源路线将同一个介子 Operator 的 dense/diagonal/point_diagonal 分量保留为一个实际端点绑定的整体顶点，
支持嵌入由后端按声明和实际有序标签完成；无需预先合并文件或新增 calculate 参数。
全 diagonal、纯 dense 和单分量沿原快路；COO/D 和专门矩阵仍按各自能力，见上述介子页。
默认 anytime 矩阵分支和 meson_matrix 已将分量/FWR 留在固定局部核，
P 只消费整体 connected/trace 与粗成本；完整档案和尾批沿用同一契约。
算法及限制见介子页，用户调用不需增加参数。
原生 dense-v 重子可通过中立常量/identity 准备从 CPU reader 进入 GPU；
源伴随、精度、owner 与池内存边界见上述重子页，无需新增前端开关。
重子的多分量合并腿可在核内先累加小中间量，再执行一次公共尾缩并；外部计划和批宽不变，
浮点加权位置会变化。算法与支持范围见重子页的“合并腿的分量中间和”。
含重子的请求（也含介子/流、胶球、交换多重态、模态分块与一行多个 Φ）自动把重子顶点降为行堆叠自旋常量（能精确低秩时拆成双夸克矩阵与行因子）与 Φ 的普通因子并保留行轴，部分行组合拆成若干子满积，P 以逐场景 DP 树引导首路线；
默认 `"auto"` 分别规划按族规则与统一的 `pair`、`nonzero`，保留预测耗时最小者，无需参数；高级 `baryon_vertices` 可强制
`"pair"`、`"nonzero"`、`"dense"` 或不打开内核 `"joined"`（仅显式声明时，供对照），
各结构的选择规则见[重子行打包因子网络](../../../docs/modules/expansion/baryon-factor-network.md)。

从物理量组织符号定义、数据绑定、规划、执行和保存；常规统一用 `contraction.correlator.calculate`。只有算符构造允许自由组合；后续流程使用固定入口和少量选项，不向用户提供自定义 prepare、producer、loader、缓存或保存流程。公开低层接口用于后端开发，不能绕过默认工作流。只读当前任务需要的参考。

共同任务路由、前端边界、保存/身份及执行范围统一见[技能共同约定](../README.md)。只读取当前任务的条件参考。

关联函数只能从 `Hadron/Flavor` 等物理算符声明调用 `gen_correlator` 产生，不导入、直接构造或手写内部图对象及夸克连线。后续只对返回的关联函数使用公开的投影、径向分块、贡献选择和容器组合接口；高级接线也不得绕过这一符号入口。

纯空间介子/流或目录重子 VVV 缺少 elemental 时，从同一 `contraction.correlator` 入口导入 `Elemental, calculate`，绑定实际登记输入，由后端在执行前检查覆盖、复用、补产并保存；用户只调用 calculate，计权只设置其 `elemental_weighted`，只规划不生产。用法及跨时间/重子限制见[自动准备](../../../docs/modules/correlator/managed_elemental.md)。其它基础输入不会因此自动求解；缺少支持应报告并修复后端，不能要求用户手写补产回调。

## 生产任务必须联合规划并固定回放

- **接到计算任务，先考察完整计算结构和复用范围，再组织执行。** 不同 configuration/cfg、不同 tsrc 通常只改变数据和物理坐标；核对算符/缩并结构、端点维数、有序轴、稀疏/抽样关系、边界及目标资源兼容后，归入同一个结构模板。不能因为 cfg、tsrc、文件名、进程或核编号不同就分别规划；相同 shape 也不能代替科学兼容检查。
- **同一结构模板一次规划并保存，之后串行和并行都固定读取这份计划。** 保存范围包括缩并顺序、时间批次/窗口及资源/存储排程；执行只绑定当前 cfg、tsrc 和实际 readers。单核循环不得逐 cfg/tsrc 重建排程，多核/多进程工作者也不得各自搜索或规划。档案不包含历史数值，当前输入仍须核验；核数或任务分配变化本身不构成重规划理由。
- 已明确的多通道、表示行、方向、径向块和参数点，先组成完整嵌套 correlator，再交给一次 `calculate(..., execute=False)`。不要在通道、行、源/汇时间或 cfg 循环中反复调用 `calculate`，也不要把 `partial(calculate, ...)`、只移动符号构造或单行计划复用称为整套任务复用。
- 同一兼容模板后续只执行已选计划：固定绑定用 `.run()`；普通 dense 单坐标模板可用 `.run(time_axes=..., batch="auto")` 首次补全内存内扫描，Dataset 的 `.run(output=..., time_axes/time_pairs=..., ...)` 提供有界窗口能力。会补全安排的调用不放入生产重复任务循环；先保存完整安排，之后读档案执行，完整 cfg 档案沿用 `calculate(plan_archive=..., configuration=tasks, ...)`。不要在脚本中循环 `complete_calculation`、手动试批宽、捕获内存错误拆窗口或保存后立刻读回。详细接线见[一次规划与回放](references/plans.md#联合任务与一次规划反复回放)。
- 仅在实际结构、端点维数/稀疏模式、抽样池等价关系、包含概率、目标后端/预算等不兼容时分模板或重规划；记录具体不兼容依据，不按“通道名不同”或“换了 cfg”直接分开，也不通过 shape 相同跳过科学兼容检查。后端资源不足时修复后端或明确报告限制，不能退回前端逐通道/时间计算并宣称完成优化。
- 若现有入口仍在兼容的 cfg/tsrc 循环中补全坐标、重新合批或规划存储，应补齐后端的计划保存与参数化绑定；不能把“零路径搜索”称为“固定读取完整计划”。使用已支持的存档/完整坐标入口，不向用户提供手工修改内部图或复制排程的替代流程。见[结构检查与串并行复用](references/plans.md#结构检查与串并行共用档案)。
- CPU/GPU 共用上述组织方式；无卡时只报告 CPU/元数据验证。多次执行测试须封住 E 展开、路径搜索和坐标/资源排程，核对串行与不同工作者读取同一档案、不同 cfg/tsrc 数值及各自来源、所有坐标覆盖；一次性准备、档案加载、当前输入绑定和数值执行分别计时。历史试算协议只能用于明确的诊断对照，不作生产模板。

## 最小完整调用

需要先声明、显式规划再执行时，可从同一 `contraction.correlator` 导入 `ContractionTask`，
用原 calculate 的声明参数构造，依次调用 `task.plan()`、`task.run()`；构造不展开、不读数组，
重复 plan 不重新搜索。run 返回原 `Calculation`，不得在 run 更改时间或算符；保存/加载、
惰性流所有权与关闭约定见[任务生命周期](../../../docs/modules/correlator/calculation.md)。常规 calculate 兼容不变。
`ContractionTask.plan()` 同时固定完整坐标安排、批宽及尾批，内存和 Dataset 输出共用完整计划；
跨进程用 `save_plan` / `ContractionTask.load_plan`，加载只绑定当前输入，不补做坐标规划。
专用内核未支持坐标合批时，`batch="auto"` 保留该内核的固定坐标实例；显式要求不支持的批宽会报错。

所有关联函数数值工作流返回 `Calculation`：`.run()` 返回新的计算结果，`.value` 保持原数值、
惰性 cfg 流或发布记录，`.report` 始终为映射。旧返回名及字段仅作兼容，新脚本统一取 value/report；
流的 pending/running/completed/closed/failed 状态、窗口重跑与存档边界见[共同返回约定](../../../docs/modules/correlator/calculation.md)。

下面从 meson55 目录声明带味道的介子算符，再由 `gen_correlator` 产生两点关联函数。
`elemental` 和 `propagator` 是实际基底、模数、相位及时间覆盖已匹配的 reader；不同味道和不同截断按数据绑定参考处理。
下面代码放在脚本主函数内，由 `if __name__ == "__main__":` 调用，以支持规划子进程。

```python
from lqcdweave.contraction.binding.api import RecipeVertex
from lqcdweave.operators.correlator.api import Flavor, Hadron, Tag, gen_correlator
from lqcdweave.contraction.correlator import calculate

mesons = RecipeVertex(elemental)
pion = Hadron(mesons.symbol("op_00_gamma5_A1", tag=Tag(0, "sink"))[0], Flavor("ud"))
correlator = gen_correlator([[pion], [pion]], ["sink", "source"], [False, True],
                            structured=True)
result = calculate(
    correlator, vertex_map=mesons, configuration=elemental.cfg,
    dtype="complex64",
    propagator_map={"S^q": propagator, r"S^q_\mathrm{local}": propagator},
    time_axes={"sink": range(Lt), "source": range(Lt)},
)
values = result.value
```

`gen_correlator` 处理味道组合、源端共轭及 Wick 收缩；`RecipeVertex` 按符号行绑定实际顶点，
无需手写连线或逐算符映射。
见[meson55 绑定参考](references/binding.md#从-meson55-目录绑定预制算符)和[meson55 使用页](../../../docs/modules/operators/meson55.md)。目录绑定按实际登记的名称、版本和分量编号匹配，不自动跨版本读取。

介子两点的调用接线见[meson55 绑定](references/binding.md#从-meson55-目录绑定预制算符)；真实输入的加载与科学声明见[数据绑定](references/binding.md)。多 cfg 的任务分配、循环前共同准备、完成核验与恢复统一按 [Dispatch skill](../lqcdweave-dispatch/SKILL.md)；原生 preset 的逐 cfg 绑定见[前端](../../../docs/modules/correlator/correlator_frontend.md#把共同定义放在-cfg-循环外)。需要跨 cfg 连续预读时，保留[完整计划的 iterator 入口](references/plans.md#完整-cfg-计划输出与-cfg-队列)。

## 必填、默认与不能隐藏的约定

- 明确粒子/流、实际动量、物理时间、估计量和输入；未知科学选择不能猜。手工具名映射的键对应 HadronIrrep 名和 Operator.name；`RecipeVertex` 的 `RecipeVertex` callable 按符号行自动解析，无需逐算符手工字典。非简并味道用各自传播子
- 常规 `execute=True`、`optimize="anytime"`、`debug=False`；只需规划显式 `execute=False`。默认全物理时间，不因优化启用稀疏化、源平均或随机采样
- 在首次 `calculate` 或 `ContractionTask` 声明任务所需 `dtype`，例如 `"complex64"`；传播子、elemental、自旋、权重和数值系数统一转换后接入处理。按需读块时先转换，不预读整套数据；计划和回放固定该类型。已有高精度临时累加仍保留，输出为声明类型。省略 dtype 的旧调用沿用最低输入精度策略，详见[精度约定](../../../docs/modules/correlator/correlator_frontend.md)。
- 可选内核合批默认 `kernel_batching=False`；任务需要时在首次完整联合规划中显式设为 `True`，后续 `.run()` 和档案回放沿用该选择。开启前读取[内核合批](references/performance.md#kernel_batching)：按 CPU/GPU 适用范围判断复用与代价，可能增加首次编译、打包复制与共享缓冲存活期，浮点求和顺序也可能变化；以同任务的数值、首次与稳定期端到端耗时及内存峰值判断收益，不以 GPU 利用率升高代替加速，也不因合批改变既定数值验收标准。关闭合批仍保留公共执行表与适用缓存的复用；复用执行结构不复用历史数值输入或地址，当前输入检查与资源准入仍保留。它与时间坐标 `batch` 独立，不改变算符、坐标覆盖或估计量。
- 重复执行较大 CUDA thin-wide 任务时，预算与可选实现见[Gram 数值策略](references/performance.md#factorized_gram_strategy)，并分别核对首次编译、预算与暖机总收益。
- 核对 cfg、实际有序基底/点集、具名轴、覆盖、源汇 dagger、相位、自旋基、边界、复系数和归一化。shape、Ne、seed、文件名不能证明兼容；复系数和抽样权重按约定只施加一次
- 各粒子和流保留真实端点，不能把相邻槽当成物理相邻时间；预制 elemental 共用[动量与规范链接配套表](../../../docs/modules/elemental/elemental_presets.md)，裸 NPY 显式给有序 momentum_list 和完整 components
- reader/借用输入保持有效，跨下一次读取保留的值及时复制；结果保留原嵌套结构、算符轴、真实扫描坐标和 output recipe。普通调用不自动发布可恢复产物
- `target_hardware` 是已登记目标名或 ExecutionTarget，`backend` 只选执行实现；设备档案不免除必要运行期资源检查。缺数据、不兼容或预算不足明确报告，不静默换 CPU、基底、时间或估计量

## 按任务读取

- 物理目标尚未确定、需展示投影候选：[候选选择](references/measurement-choice.md)；目标已明确不重复选择固定例子
- 新通道、多点/流、Wick 与端点：[测量契约](references/measurement.md)、[符号](references/symbolic.md)、[算符](references/operators.md)
- 重子 gamma/自旋从 `baryon_spin_operators(version="v1")` 的 24 个多重态选择，完整覆盖 64 行；已按双立方群、宇称及 S3 联合投影。MS/MA 用 `partner` 配对，自旋标签不自动变成完整重子交换标签。与原始收缩程序同型的 `(q0ᵀCΓq1)q2` 双夸克算符（Γ 取全部 16 个 Clifford 元，G1/H × g/u 共 24 个配方）用 `baryon_diquark_operators(version="v1")`，接口相同且为低秩，`diquark_Cg5_G1g` 即原程序的 G13。用法见[自旋目录](../../../docs/modules/operators/baryon_spin_catalog.md)。
- 重子先组合 spin×spatial 并作立方投影，再用 `insertion.project_exchange(...)` 投影其组合交换表示，flavor 后组合；S/A、MS/MA、重复份数、S2 槽位及配套生产表见[组合交换投影](../../../docs/modules/operators/baryon_exchange_projection.md)。
- 已知重子三夸克交换特征时，在 `Hadron(..., quark_symmetry=...)` 声明完整系数的标签以合并等价 Wick 图；包含颜色与原费米子符号的约定、部分交换和分块限制见[符号](references/symbolic.md)
- 已有数据或非简并传播子：[绑定](references/binding.md)；明确需要高模派生/PSV dagger 才读[高模任务](../../../docs/modules/perambulator/point_highmode.md#highmode-task)
- 绑定投影点 H 时，按[H 数据交接](references/binding.md#h-binding)核对完整父空间和实际 P 身份；已投影块直接消费，elemental 仍用原始点核，避免二次减除。H 交叉块不自动补齐 HSH 或完整算符作用源。
- 时间扫描：[坐标](references/coordinates.md)；明确时间稀疏化才读[共同时间交接](../../../docs/modules/temporal_sampling/temporal_production.md)，复用同 cfg selection，保留 Nt 槽与物理 Lt
- 明确随机空间取样：[取样计权](references/sampling-current.md)。已有确定点集不重抽；模态随机稀释另见 elemental skill 的条件入口
- 消费由 `ElementalGenerator(..., elemental_weighted=True)` 预先计权的 D 或点端 elemental 时，在 `calculate(..., elemental_weighted=True)` 使用同一选项；V/D 扇区、实际 dilution 来源和同池高阶见[QR dilution](../lqcdweave-elemental/references/dilution.md)；多 current 图的剩余系数、独立池声明与支持范围见[取样计权参考](references/sampling-current.md#elemental-weighted)
- 输出、保存/读回及显式 dt 对齐平均：[结果](references/results.md)。缺失 dt 默认报错，不补零；仅完整 cfg Dataset 后处理成为瓶颈时，读[可选 GPU 后处理](../../../docs/modules/results/gpu_output_postprocessing.md)，默认保持 CPU
- 普通持久续算：同定义、实际父项及稳定来源下补齐缺失坐标，只复用登记完成覆盖；跨阶段时读[项目恢复](../../../docs/modules/dataset_planning/dataset_planning.md#project-resume)。默认格式为每 cfg 整块 NPY/HDF5 与共享根 `dataset.json`，见[数据保存说明](../../../docs/modules/storage/data_storage.md)
- 改变搜索、预算或复用策略：[规划](references/planning.md)；执行/后端与持久交接：[执行](references/execution.md)
- 需要减少共享 NPY 重读：[共享读取](../../../docs/modules/data_readers/data_readers.md#shared-npy)；需要测量、预测、校准或 CLI：[性能](references/performance.md)
- 多 cfg/坐标重用已选路线：[计划档案](references/plans.md)。加载恢复已保存表，不搜索、重排或重算 prediction；当前 reader/source factory 及 Dataset 输出根目录仍显式提供
- 专家已有数组要算单时间对连通矩阵：[三层矩阵入口](../../../docs/modules/measurements/meson_profile_matrix.md#entry-routing)；已有低层回放需显存观察/临时磁盘存储：[专家资源](references/execution.md#expert-resources)
- 有限点/向量传播子的生产方法验收，两条相邻current腿都必须限制到实际入选标签，不能因为存在完整解场、父辅助块或算符作用源缓存，就换成完整另一腿估计量。同池按实际相同/不同标签计权，缺少异标签项的包含支持则明确报错。常规物理测量仍用本skill默认入口。
- 用户要求重复独立点抽样时，每份各自缩并和计权，再在cfg内平均，最后跨cfg统计；不先平均传播子，不把replica当作额外cfg。保留每份实际点/池身份及总预算；重新排列旧点不等于新的全空间抽样。
- 改变D/H分界时，重要性轮廓与分量预算仅定义父区间，不额外乘进估计量；全颜色点按三个分量计预算，包含概率仍按空间点标签计算。复用已保存中间量须匹配实际输入及物理定义，不重复投影或准备elemental；缺实际父基底或某组n<m的设计明确保留缺口。
- 只有用户明确要求完整另一腿、单次空间单位阵插入的专家研究时，才用[专家单流数值入口](../../../docs/modules/measurements/external_current.md)。支持流端算符作用源表示、原始缓存读回、显式允许时的缺失求解、逐腿贡献保存/重组和统计；完整解场为另一个显式专家入口。此条件入口不能用于有限生产输入的方法验收，也不是普通多流Diagram工作流；不能从紧凑点块猜全场，不能套用双插入二阶计权。物理流源时间与RHS时间分别报告；不把共享求解成本当成每个参数点的独立成本。
- 仅设备目标尚未登记时：[目标档案说明](../../../docs/architecture/generic_production.md#unregistered-target)。不把独立资源预检加到默认任务
- 逐腿选择、模式分块：[基底与截断](references/vertex-legs.md)；固定分区保留 PC 时优先在 cfg 循环外 `split_meson_mode_pairs`，顺序与归一化见[模式分块](../../../docs/modules/operators/meson_mode_blocks.md)
- 扩展接口/分阶段组合：[公开模块](references/interfaces.md)。不复制搜索、数值、缓存或存储系统
- 谱学任务的通用组织：[谱学](references/spectroscopy.md)；带流多点的通用组织：[多点](references/multipoint.md)

## 限制与交付

胶球通过 `Flavor("")` 保留零夸克插入，`Glueball` 绑定已有数据；普通和 structured
单强子入口复用原结果配方。数组/NPY 输入用 `ArrayGlueball`，固定块为标量，源端只共轭一次；
不生产规范回路，也不在前端手工回填。见[胶球绑定](references/binding.md#胶球已有数据)。

V/D/H 的[完整闭环](../../../docs/modules/perambulator/three_sector_production.md)已验证 CPU 普通介子关联函数：六类传播子恢复九类，九扇区 elemental 与实际 D/点池共同计权；原始与预权路径共用 `elemental_weighted`。独立 Wilson 参考覆盖 medium、高点及二者联合的有限方案平均；单份随机实现不等于完整空间结果。

支持范围以实际 API、L2 和测试为准；coordinate-batched sparse GPU、缺少可序列化 preparation recipe 的 opaque 输入归档仍不支持。一次调用不保证零重复 IO，拆分调用不自动补齐功能。CPU 小样本、模拟设备、真实 GPU/MPI 与生产规模分别报告。

执行范围遵守共同约定；交付实际入口、数据缺口、计划或结果及本次验证范围。缺基础输入时才转相应生产 skill。
