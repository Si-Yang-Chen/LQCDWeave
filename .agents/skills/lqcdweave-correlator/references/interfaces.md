# LQCDweave 当前公开接口与交接

群表示、旋转与投影使用正式模块 `lqcdweave.symmetry`，接口与约定见[格点对称性](../../../../docs/modules/symmetry/README.md)。

架构以[总体架构](../../../../ARCHITECTURE.md)和
[Correlator L0/L1](../../../../docs/architecture/correlator-l0-l1.md)为准；精确选项和实际支持范围
核对[模块 L2](../../../../docs/modules/README.md)与源码。下表中的源码目录相对仓库根目录，
接口目录不表示所有多点组合都已验收。

## 按职责选择

| 职责 | 公开入口 | 对应 L2 |
|---|---|---|
| S 符号、投影与 Wick | `operators.correlator.api`：HadronIrrep、Flavor、Hadron、hadron_little_group_projection、gen_correlator | [symbolic](../../../../docs/modules/symbolic/symbolic.md) |
| O 两腿/三腿插入与流配方 | `operators.single_hadron.api`：Insertion、Operator、ConservedVectorOperator、meson55_operators | [operators](../../../../docs/modules/operators/operators.md) |
| B 已有数据顶点与描述 | `contraction.binding.api`：Meson、RecipeVertex、Baryon、Current、绑定/描述服务 | [binding](../../../../docs/modules/binding/binding.md) |
| E 网络、坐标和采样展开 | `contraction.expansion.api`：expand_correlator、NativeDiagramBinding | [expansion](../../../../docs/modules/expansion/expansion.md) |
| P 路径与物理存储规划 | `contraction.planning.api`：find_contraction_path、solve_storage、rebind_scene_path、plan_scene_storage | [planning](../../../../docs/modules/planning/planning.md) |
| X 固定计划执行 | `contraction.execution.api`：execute_contraction_path、BoundContractionExecution、replay_scene_storage、replay_storage_batches | [execution](../../../../docs/modules/execution/execution.md) |
| R 结果映射与组装 | `contraction.results.api`：build_output_recipe、assemble_correlator | [results](../../../../docs/modules/results/results.md) |
| M 模型与实测 | `contraction.performance.api`：目标登记/解析、profile、校准缓存、拟合/融合及报告接口 | [performance](../../../../docs/modules/performance/performance.md) |
| A 计划存取、完整 cfg 档案与重绑 | `storage.contraction_plans.api`：save/load_contraction_plan、save/load_storage_schedule、save/load_cfg_plan、migrate_plan_archive、migrate_plan_bundle | [plan_archive](../../../../docs/modules/plan_archive/plan_archive.md) |
| C 中立描述 | `contracts.correlator.api`：ExecutionTarget、ExpandedCorrelator、PlannedContraction 等 | [contracts](../../../../docs/modules/contracts/contracts.md) |
| 收缩编排 | `contraction.correlator`：calculate、Calculation、save_calculation_plan、load_calculation_plan；`contraction.contraction_session`：ContractionSession | [correlator](../../../../docs/modules/correlator/correlator.md) |

导入时使用完整包前缀 `lqcdweave.`。跨模块通过上述公开 API 或中立协议，
不恢复已删除的 `contraction.replay`、scene_state/scene_paths/scene_storage 等转发路径。
X 的存储回放消费已由 P 联合完成的计划，不在回放阶段补做规划或稀疏工作区分析。

## 后端开发与兼容接口

普通用户的唯一入口与返回约定见[correlator skill](../SKILL.md)；本页只定位后端模块。
明确开发某个阶段时才组合 E/P/X/R，不将它们列为普通任务的可选流程。

分阶段开发通过 E.api `expand_correlator`、P.api `find_contraction_path`、
X.api `execute_contraction_path`、R.api `assemble_correlator` 组合。
预算与模型在调用 P 前解析；P 返回 `path.expanded` 时，后续执行使用该交付描述，
不混用批宽选择前的输入。执行复用既有 runtime 服务和输入协议，不复制数值核或搜索逻辑。

自定义组合不会自动补齐能力。逐腿选择、多点或零连接顶点是否可用，仍需核对 B/E
的描述、真实读取、轴以及 P/X 的预算交接；不能把能构造图等同于已经能执行。
[胶球接入](../../../../docs/architecture/glueball_architecture.md)已提供 `binding.api.Glueball` 和
`adapters.glueball.ArrayGlueball`：原 HadronIrrepRow/Flavor("") 构造、标量输入 term、
Dataset 和计划回放，复用原 `calculate`。不增加胶球专用算符或执行入口。

## 档案、持久输入与恢复

用户持有 Calculation 时使用收缩编排的保存/加载族，具体分支见[计划复用](plans.md)。
A 的纯 PlannedContraction、schedule 和既有 Session 用于相应低层对象，
不能因为都叫计划就互换编码。读取/存储、发布和恢复分别由 adapters、storage、
runtime 负责，共同约定见[技能路由](../../README.md)。

新增格式或功能时扩展所属模块的公开边界，不复制第二套搜索、数值、缓存或存储系统。
符号输入仍从 gen_correlator 产生，底层接口不授权手写内部拓扑。
