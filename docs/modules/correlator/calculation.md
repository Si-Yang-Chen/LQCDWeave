# Calculation：共同执行与返回约定

需要显式区分声明、规划和执行时，使用同一模块的 `ContractionTask`：

```python
from lqcdweave.contraction.correlator import ContractionTask

with ContractionTask(correlator, vertex_map=vertices, propagator_map=propagators,
                     configuration=cfg, time_axes=times, dtype="complex64") as task:
    task.plan()
    result = task.run()
    values, report = result.value, result.report
```

构造只保存声明和 reader 引用；`plan()` 才调用现有规划服务，重复调用不重规划。
`dtype` 在声明时固定，传播子、elemental、自旋与数值系数统一按该类型进入准备和计算。
数值输入按需分块读取，每块先转换，再进行共轭、计权等处理；匹配类型无需额外转换，
也不改写原始文件。计划保存该类型，回放不能覆盖它。`calculate(..., dtype="complex64")`
使用相同约定；旧调用省略 `dtype` 时保留原有的最低输入精度选择。
已有的高精度临时累加策略仍保留，最终输出使用声明类型。
`run()` 必须在规划后调用，每次仍返回 `Calculation`，不接受时间范围或算符定义的更改。
`plan_report` 提供只读摘要；`save_plan(path)` 和 `ContractionTask.load_plan(path, ...)`
复用现有档案服务，不做隐式规划。新任务统一完成 cfg 计划，内存与 Dataset 输出均保存
全部坐标、所选批宽和尾批；加载只需当前数据绑定，不需要原 correlator 或 `expected`。
旧 `save_calculation_plan` 保存的固定模板档案仍沿用原 `expected` 加载合同。

```python
task = ContractionTask(correlator, vertex_map=vertices, propagator_map=propagators,
                       configuration=cfg, time_axes=times, batch=2)
task.plan()
task.save_plan('plans/task.json.gz')
task.close()
with ContractionTask.load_plan('plans/task.json.gz', configuration=cfg,
        vertex_map=current_elemental, propagator_map=current_propagators) as replay:
    result = replay.run()
```

单个 elemental 可直接传当前 reader，也支持 `RecipeVertex`；使用名称键的顶点映射会保存
对应 reader 键，加载时可继续使用同名映射。多个自定义 callable 来源仍使用明确 reader
角色映射。Dataset 在构造时传 `output=...`，加载时提供当前输出目录；不能读档后切换输出模式。
专门的 sparse/重子结构在尚未支持坐标提升时使用已保存的单点实例，显式多 lane 请求会明确拒绝。

同一 task 不允许并发执行。惰性 cfg 流在耗尽或关闭前一直占用 task；`task.close()`
也会关闭未耗尽的流。执行失败清理资源后可重试，单次 cfg iterator 不会自动倒带。
任务关闭后不能重新运行；已交付数组仍有效。这不改变下文旧 `Calculation.close()` 的兼容语义。

完整批次打包 dt 输出时，单点字段按全部保存坐标确定时间槽，包含批内后续位置才出现的时间。

随时间变化的 `SpectralProfile` 在完整计划中按每个实际输入时间选择权重行，源端仍沿原
伴随处理；固定实例与坐标批次使用同一规则。完整档案只保存这些已经选好的行，回放不做
权重选择或重新规划。旧实现曾在坐标提升时沿用代表时间的行；因此此前产生的、带时间变化
谱权重的完整扫描档案需要重新规划，加载不会自动修正历史计划。普通固定时间计算不受此问题影响。

`contraction.correlator.calculate`、`load_calculation_plan`、完整 cfg 补全、窗口执行及旧
`ContractionSession` 的数值方法均返回同一个 `Calculation`。常规用户只需 `calculate`：

```python
selected = calculate(correlator, vertex_map=vertices, propagator_map=propagators,
                     configuration=cfg, time_axes=times, execute=False)
result = selected.run()
values = result.value
report = result.report
```

`.run()` 返回新的 `Calculation`，原对象的 `.value` 和 `.report` 不变。选定的路径和适用的
结构执行表可以复用，每次重新计算数值；不在复用表中保存输入数组、reader 或输出。
固定计算与完整 cfg 回放不重新展开或搜索。窗口回放仍需生成有界的坐标窗口安排。

普通 dense 单坐标选择可在 `.run(time_axes=..., batch="auto")` 中补全内存内坐标扫描。
它只提升同一已选路径的时间批轴，按预算分批，不重新搜索；返回的 Calculation 之后
直接 `.run()` 复用这些批次。来源检查、预算和限制见[时间坐标](../../../.agents/skills/lqcdweave-correlator/references/coordinates.md)。
这与 `kernel_batching` 独立；Dataset 的 `output=...` 回放继续使用原有窗口安排。

同一句柄及其结果副本可自动复用一个有预算约束的空闲 GPU 内存池。每次仍处理
当前输入，清理票据并同步交付结果；动态主机/设备额度检查继续执行。CPU 不建 GPU 池。
无需增加运行参数。结束回放后可用 `selected.close()` 提前释放空闲池，或将句柄用于
`with selected:`；关闭不改变已有 `.value`，之后仍可再次 `.run()`。
debug 使用原有临时池；惰性 cfg 流继续通过自己的 `value.close()` 管理正在执行的资源。
所有权、并发与失效边界见[执行模块](../execution/execution.md)。

| 模式 | `.value` | `.report` / 重跑 |
| --- | --- | --- |
| 只规划 | `None` | `status="planned"` |
| 固定输入、单 cfg 内存计算 | 原来的 ndarray、dict、list、tuple 或嵌套容器 | 同步完成后 `status="completed"` |
| cfg iterator 的内存计算 | 惰性迭代器，逐 cfg 产出 `(cfg, value)`，有 `.close()` | `.run()` 不领取 cfg；报告随迭代更新 |
| 完整 cfg 或窗口 Dataset 输出 | `{cfg: {directory, configuration, data_id}}` 发布记录 | 保留全部执行与发布统计 |
| 旧 session 数值计算 | 原来的 term tuple 或装配后的数组 | 保留原统计；`.run()` 重用所选 IR 或 PreparedBundle |

`.report` 始终是映射，公共键为 `status`、`execution_mode`、`output_mode`。只规划也有报告。
其余键保留相应执行服务的统计，不为统一格式额外读取输入、查询设备或逐操作计时。
普通计算的 `measure=True` 将原始测量报告保存在 `report["execution"]`，包括 `timings`、
`execution_profile` 等；原有不冲突的属性读取仍兼容，例如 `report.execution_profile`。
完整 cfg 和窗口的原有平铺字典字段保持可用。具体计时边界须按对应字段解释。

惰性报告的 `status` 从 `pending` 进入 `running`，耗尽后为 `completed`；提前 `.close()`
为 `closed`，异常为 `failed`。关闭未开始的流不领取 cfg，关闭已开始的流释放所持 reader。
要确认全部 cfg 完成必须耗尽迭代器；最后一个值刚产出时仍可能尚未结束清理。
单次 iterator 保持单次消费语义，不会为再次 `.run()` 自动物化或复制它。

普通 `.run(output=..., configuration=..., propagator_map=..., vertex_map=...,
time_axes/time_pairs=...)` 沿原生窗口服务执行。返回的 Calculation 保存完整坐标和运行绑定，
可用 `result.run(output=other_directory)` 重跑；读者仍须有效。进度回调、输出布局、dt、
GPU 后处理及原来的资源检查继续由原服务处理。已发布目录仍服从既有存储的重复写入规则。

窗口结果带的是已选模板和完整运行请求，不能作为完整 cfg 档案保存；请保存原选择或用
`complete_calculation` 完成后保存。`save_calculation_plan` 按实际 `PlannedContraction` /
`CfgPlan` 路由，保持原档案格式；加载后获得新的 Calculation 和空执行缓存。
Dataset 发布记录不能再次当成内存数值交给 `save_correlator`。

## 旧接口兼容

- `CompleteCalculation` 是 `Calculation` 的兼容名称；旧的 `CompleteCalculation(plan, ...)`
  和 `plan=...` 构造仍可用。不能再用两种类名区分固定与完整任务，按 `execution_mode` 或计划种类判断。
- `CfgWindowResult(value, report)`、`CfgExecutionResult(values, stats)` 兼容构造返回 Calculation。
  手工构造的旧结果没有所选任务，`.run()` 会明确报错；实际执行获得的结果可以重跑。
- `.values` 是 `.value` 的别名，旧 session 的 `.stats` 保留原统计对象。旧数组返回调用仍可索引、
  查看 shape/dtype、参与 NumPy 运算；`return_stats=True` 仍可解包为 `(array, stats)`。
  新代码统一显式取 `.value` 和 `.report`。session 的 `.run()` 使用该 handle 绑定的输入；
  需要重新调用 getter 绑定新时间/cfg 时仍通过原 session 方法构造新的计算。
  此类结果缺少前端坐标声明，保存时沿用原始数值的显式 axes/coordinates 等元数据。

这里统一的是收缩编排的计算与返回对象。E/P/X/R 的纯描述、X 内部的 `ExecutionResult`、
存储读回的 `CorrelatorResult`、reader、plan、session 准备对象保留各自职责和类型。
`measure=True`、内存覆盖及 runtime 绑定仍按原执行模式支持，不因返回类型统一扩大支持范围。
