# 保存与复用计算路径

新任务可使用统一的 `ContractionTask`：将完整 correlator 容器、绑定及 time_axes/time_pairs
一次交给构造函数，调用 `.plan()` 后 `.save_plan(path)`。构造不读数组，保存不补规划。
内存与 Dataset 共用完整 cfg 档案；后者仅在构造时增加 `output`，加载时指定当前输出目录。
`ContractionTask.load_plan(path, configuration=..., propagator_map=..., vertex_map=...)`
不要求原 correlator/expected，也不展开、搜索或选择批宽。普通 dense 保存实际批次及尾批，
运行按原坐标装配；不支持合批的结构保持单点实例，不能把实例复用称为已向量化。
绑定支持与生命周期见[任务约定](../../../../docs/modules/correlator/calculation.md)。下文旧函数接口继续兼容。

生产计算先按下节核对结构并保存共同计划；同一绑定的临时小样本诊断可直接复用 `calculate(execute=False)` 返回的对象，无需为一次性试算写盘。存档时区分收缩编排的 Calculation 与 A 接收的纯计划。

## 结构检查与串并行共用档案

收到任务时先列出全部算符、通道、配置和时间扫描，以及实际输入的维数/有序轴、稀疏结构、抽样池等价关系、包含概率和物理边界。按这些会改变计算结构或执行兼容性的条件分模板，不按 cfg、tsrc 或工作者编号分模板。通常不同 cfg 和 tsrc 的数据值、文件位置及时间读入坐标不同，而缩并图、批宽和存储寿命安排可以相同；应检查并利用这一点，不能仅凭 shape 推断兼容。

每个兼容模板在生产循环外完成一次符号展开、选路、坐标批次/窗口及资源排程，并保存成可读取的共同档案。单核串行循环和多核/多进程并行工作者都读取同一份档案，只替换当前配置、参数化时间及 readers；不在循环或工作者内调用规划入口，不复制出每个 tsrc 的同结构计划。并行调度只决定谁执行哪个任务，不能成为重复规划的理由。读回后不要紧接着保存、比较完整对象或重算 prediction。

区分一次性的完整计划准备与每次的输入绑定、资源准入和执行。多 cfg 不共享历史数值；与 tsrc 无关、但依赖 cfg 的公共量在每个 cfg 内准备一次，源相关部分沿共同模板循环。时间参数化须保留真实物理坐标、相位/边界和输入别名；周期时间平移只有在实际契约支持时才可采用，不直接改内部图或把固定坐标冒充参数。

`run(time_axes=...)` 或公共窗口入口仍可能重建执行描述；只复用缩并路线并不等于满足上述完整计划复用。生产任务应先选择已支持的完整档案入口；若后端缺少所需的参数化存档/绑定能力，补齐后端并验证，不能默默退回逐 cfg/tsrc 重排程。仅在有具体不兼容证据时另建模板，例如算符/端点结构、稀疏/抽样关系、物理边界或目标后端/预算实质改变。当前文档中的一般坐标补全示例说明能力边界，不构成逐源循环补全计划的推荐用法。

验证时同时禁止符号展开、路径搜索和坐标/资源排程，确认串行及并行工作者只读取保存的安排；用不同 cfg 和不同 tsrc 的实际数据检查数值与来源，并核对当前输入/输出坐标。计划准备、档案加载、运行绑定和数值执行分别计时，不把一次性排程费用乘上 cfg 或 tsrc 数量。

```python
from lqcdweave.contraction.correlator import save_calculation_plan
save_calculation_plan(plan_file, planned)
result = planned.run()
```

临时诊断可直接执行已有对象而省去保存；生产任务仍先保存共同安排。另一次调用复用普通固定绑定档案时，使用
`load_calculation_plan(plan_file, expected=current)`，其中 `current` 提供该次输入绑定。
保存后立即读回和比较完整对象属于验收，不加入生产循环。

若调用方持有的是 `PlannedContraction`，则使用 `storage.contraction_plans.api.save_contraction_plan(plan_file, path, ...)`；不能将 Calculation 直接交给该 A 接口。

收缩编排的 `load_calculation_plan` 与保存入口配对：普通 Calculation 档案给当前 `expected`；
完整 cfg Calculation 档案不给 `expected`，而给当前 readers/source_factory 与 configuration。
不混用两种编码，不在加载时补规划。详见[前端说明](../../../../docs/modules/correlator/correlator_frontend.md)。

## 联合任务与一次规划反复回放

内存内坐标扫描由后端逐个已选联合批次调用共同结果装配，只保留最终字段；不能在脚本
按通道拆算或去掉预算检查。内存模型包含全部最终字段、最终堆叠重叠、单点装配工作区
以及当前 GPU 批次下载，调用方 reserve 只计一次。`measure=True` 保留原始逐批报告，
需要额外主存并在读取前准入；生产计时可沿外部墙钟/RSS监控，不能为省内存静默丢弃诊断值。
固定选择和存档读回都沿用同一装配方式，返回结构与坐标顺序保持原定义。

先声明完整科学任务。例如把所有通道/表示行放进同一字典；分块输出不等于分块调用规划。

```python
selected = calculate(channels, **bindings, time_axes={"sink": 1, "source": 0},
                     execute=False)
# 通道、算符、径向定义、优化器和目标预算已固定。当前 reader 始终在有效期内。
result = selected.run(output=output_root, configuration=cfg,
    propagator_map=current_propagators, vertex_map=current_elemental,
    time_axes={"sink": sink_slots, "source": source_slots},
    physical_times=actual_ordered_times, physical_lt=Lt, time_roles=("sink", "source"))
```

Dataset 回放的 `vertex_map` 与完整档案加载相同：只有一个 elemental 角色时可直接给当前 reader，
否则传已保存角色到 reader 的映射；不是重新构造 Meson 算符的 callable。
`Calculation.run` 不接受 `correlator/optimize/search/profile/partitions` 等规划输入，
只替换当前运行绑定；收缩编排固定原选路，通过既有窗口规划生产者、执行器与 Dataset writer
完成全部坐标后发布。它不会调用 E 或路径搜索，也不在用户脚本中二分汇批宽。
输入结构或抽样池兼容性检查仍执行；失败直接报告，不偷偷搜索新的模板。
大范围生产使用上述带 `output` 的公共窗口入口。固定缩并顺序不等于坐标准备免费：
内存内时间批扫描仍可能预先构造大量执行描述。每个窗口保留完整联合任务，窗口只控制
驻留和交付；不要在任务脚本中按通道重建图、搜索路径或复制全部坐标的DAG。
代表 reader 的不可变描述可能仍用于窗口构造，应保持有效直到该模板最后一次回放结束；
其它 cfg reader 每 cfg 绑定、执行并释放，不把所有 cfg 数值输入常驻。

无参数 `.run()` 保持已选绑定和坐标。普通 dense 单坐标模板可用
`.run(time_axes=..., batch="auto")` 补全有界的内存内坐标批次，沿用同一缩并顺序；
详见[坐标回放](coordinates.md)。不能用修改 `options` 字典来改变物理任务。
多 cfg 完整持久档案使用下文 iterator 入口，保留跨 cfg 连续预读。
内存中 selected 的 Dataset 回放当前逐 cfg 调用，不声称支持跨 cfg 连续窗口流。

程序无法知道未来调用者还会声明哪些通道，因此不能从一次独立 `calculate` 判断任务是否拆散，
也不建立进程全局缓存来猜测任务。AI 必须在首次规划前核对完整任务范围。
可兼容联合的任务不得因为脚本循环方便而拆开；单通道诊断、明确选定的子任务及实际不兼容的模板除外。
重建窗口的资源排程与重新搜索路径分开统计，避免把“零路径搜索”误报为“零规划开销”。

## 固定路径加载

`storage.contraction_plans.api.load_contraction_plan` 用于普通固定路径档案，要求兼容的当前 `expected`
ExpandedCorrelator 与目标规格，返回绑定当前输入的 PlannedContraction，不搜索或执行。加载直接恢复档案
中的 prediction 与 storage trajectory；不重算成本、不重排动作或重新规划资源。显式预算覆盖只与保存峰值
比较，足够时可直接复用。坐标、CostProfile、reserve 或输出布局改变时需重新规划并保存新档案。
收缩编排的 `calculate(plan_archive=...)` 是另一条完整 cfg 档案入口：它先加载 complete-cfg 计划，可省略 correlator，但仍要提供 configuration
和当前数据 readers/source factory。

档案回放沿用保存时的合批选择，不受新规划默认关闭合批的影响。`kernel_batching` 是新规划选项；切换它需要重新规划并保存新档案。不要在 `calculate` 的 `plan_archive` 加载调用中传入 `kernel_batching=True`：加载调用会把它视为规划覆盖并拒绝，和 `search` 等已有规划参数规则相同。

档案保存纯描述与兼容元数据；后端以 `numpy`/`cupy` 等标识保存，不序列化 Python backend
module/object。reader、loader、回调、结果数组及运行期暂存目录不保存。跨组态/坐标核对结构、
物理语义、轴、稀疏结构、profile 和目标硬件；相同 shape 不证明科学身份兼容。

显式覆盖的新预算不会重算计划轨迹，只核对是否覆盖已保存的峰值；不足时拒绝加载。其它计划相关输入不兼容时
报告原因，须明确新建计划，不能用完整重搜假称固定路径复用。

## 完整 cfg 计划、输出与 cfg 队列

对固定的多坐标任务，可用代表 cfg 规划并由收缩编排完成完整 cfg 计划，再以
`save_calculation_plan(path, complete)` 保存。执行时调用
`calculate(plan_archive=path, configuration=cfg_iterator, ...)`；完整 cfg 档案在 E 展开前加载，
故 correlator 可省略。当前 readers/source factory 必须按保存的角色重新绑定。完整 cfg 档案固定
所选路径、坐标、输出模式、输出布局及缓冲安排；加载只恢复保存的执行表，不搜索、重排或重算 prediction。
完整补全会自动在预算内跨内部窗口保留实际读取身份和完整准备配方相同的终端 dense owner，
让固定时间端的 elemental 等输入继续复用；不按 source 字段名或 shape 判断，也不改变窗口或计算顺序。
延寿、跳过的准备/读取事件和收紧后的预读额度均保存到档案，执行沿普通 owner 表回放；无需前端缓存或开关。
原输入请求、别名、实际有序时间/基底/点集及 profile 仍完整保留，换 cfg 时照常核验。
Dataset 模式的 output 根目录不保存，加载时重新传入。`dt_packed` 是 Dataset 专用布局，加载时不转换。
显式 GPU 后处理的选择保存在 `plan.statistics["postprocess_backend"]`，连同
`postprocess_device_bytes/postprocess_host_bytes` 输出保留量重放；缺省仍为 CPU。
它不改变科学输出定义或 Dataset schema。改变后处理选择须从已选模板分别补全计划，
不在加载时替换预算或退回 CPU。条件入口见[GPU 后处理](../../../../docs/modules/results/gpu_output_postprocessing.md)。
旧 complete-cfg 编码不自动迁移，须按当前任务重新规划并保存。

任务领取、共同准备与完成核验统一按 [Dispatch skill](../../lqcdweave-dispatch/SKILL.md)。
保留 `calculate(plan_archive=path, configuration=tasks, ...)` 的完整 cfg iterator 入口，
其中 `tasks` 可直接是 Dispatch；不要在外层拆成逐 cfg 的独立计算调用。
完整 cfg 执行维护单个有界连续 ReadAhead 队列，可跨配置、扫描坐标与时间对。
只有 owner thread 调用 iterator 的 `next()`；后台线程只读取已领取 cfg 的输入。
接线见 [Dispatch 的预读入口](../../../../docs/modules/dispatch/dispatch.md#跨-cfg-连续预读)，
执行支持范围和 CuPy 限制见 [生产脚本与保存](../../../../docs/modules/correlator/correlator.md)。

## 档案版本与离线迁移

当前新写计划档案使用 v8 envelope。普通 `plan-bundle` reader 直接接受 v5/v6/v7/v8；v4 generic bundle
用 `migrate_plan_bundle(source, destination)` 显式离线迁移。`storage-schedule`、`correlator-plan` 和
`complete-cfg-plan` 的直接 reader 仅接受 v8；`migrate_plan_archive(source, destination)` 显式转换
v4/v5 `correlator-plan` 与 `storage-schedule`，以及 v5 `plan-bundle`。常规加载不会迁移或规划、重算档案。
旧 complete-cfg/matrix 格式没有迁移器；如需复用，须按当前任务重建完整 cfg 计划并另行保存。

## 多批 schedule 与会话

`storage.contraction_plans.api` 的三个 schedule 接口分别要求：

| 接口 | 必填参数与行为 |
|---|---|
| `save_storage_schedule` | `path, plans, shapes, hardware`；`shapes` 精确覆盖全部 tensor 输入 |
| `load_storage_schedule` | `path, hardware, expected`；当前 `blocks/preparations` 可由 ExpandedCorrelator 提供，或成对传入 |
| `rebind_storage_schedule` | `plans, blocks, preparations`；替换兼容的实际来源/选择身份，保留固定排程 |

其中 `shapes`、`hardware`、`expected`、`blocks`、`preparations` 均为关键字参数。
可选参数和类型见[计划档案 L2](../../../../docs/modules/plan_archive/plan_archive.md)。

上述 schedule 与既有 Session 仅用于底层已有对象的兼容接线，普通任务不另选这些入口。
档案结构、版本及编码实现统一见[计划档案 L2](../../../../docs/modules/plan_archive/plan_archive.md)。

## 大 cfg 分块档案与显式窗口执行

完整计划补全按已选驻留窗口释放构造临时表，再合并全部已排定实例；原有
`calculate/complete_calculation/save_calculation_plan` 自动采用此处理，无需脚本拆分任务。
原生宽缩并/FWR 跨科学 term 复用时按实际输入和完整数值配方绑定同一 owner，
保留宽精度、输入角色、输出布局和固定 tile；term 的证明位置不另生成数值 owner。
新规划会自动保留终端三输入 tiled FWR 的内部 complex128 叶子，最终发布字段仍转换为
任务 dtype，不需要新增前端参数。旧档案的已保存 cast 语义保持不变；需要新的内部叶子
精度时由普通入口重新规划，不能仅更换执行器后沿用旧数值档案。
最终计划的全部坐标、读取、准备、释放与资源上界仍一次保存；读回不重新补全或排程。
完整 cfg 的大请求/绑定表保存时可自动采用有界分块 stream 编码（小计划仍保留 JSON），
小 JSON 计划内部也自动采用紧凑记录，保留原共享对象身份、完整字段与构造校验，减少后续
固定加载的解析和恢复成本。旧档案直接读取；需要新表示时由同一保存入口另存，无需重规划。
大 stream 的 gzip 保存默认使用级别3以减少准备阶段耗时；接口、完整计划内容和加载
规则不变，旧 gzip 档案仍直接读取，无需重新规划或迁移。
现有 save/load 接口不变；两者都使用 v8 envelope 与同一白名单 CfgPlan 契约，
保留实际时间坐标、输出布局和保存预算，加载不搜索或重排。详见[计划档案 L2](../../../../docs/modules/plan_archive/plan_archive.md)。
完整 Dataset 写出优先 `Calculation.run(output=...)`；显式窗口生产者/执行器仅属内部开发，
不作为第二个用户流程。内部交接见[生产脚本与保存](../../../../docs/modules/correlator/correlator.md#有界-cpu-规划生产者)。

## 抽样输入的模板兼容性

完整计划补全与回放核验实际抽样池的等价关系及包含概率规则，不兼容坐标分别选模板，不能按 shape 相同强行复用。带抽样的时间位移 elemental 须按当前实现核对完整回放支持；空间等时路径和一般无采样路径沿用对应入口的范围。
