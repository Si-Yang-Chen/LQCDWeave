# 技能任务路由与共同约定

本页是仓库八个 LQCDweave skill 的共同规则和入口索引。先按用户要交付的产物选择
一个主 skill；其它 skill 仅处理该任务确实缺少的依赖，不作为同一工作的平行入口。
任务材料与长期文档的边界遵循 [AGENTS 第 6、7 条](../../AGENTS.md)；示例参数不成为默认值。

## 计算项目脚本的位置

为具体计算问题编写的临时脚本、配置、日志和报告统一放在 `scripts/<计算项目>/`，
整个 `scripts/` 不由 Git 跟踪。同一项目的验收、性能诊断、测量和交付可设子目录，
不按每次试算另建一个顶层项目，也不把脚本散放在 `scripts/` 根目录。项目脚本的
测试按实际用途从简；复用的科学与数值约定仍按相应 skill 执行。

可复用的库实现放在 `src/lqcdweave/`，维护工具放在 `tools/`，稳定用法和回归
分别放在 `docs/`、`tests/` 并随实现提交。长期文档及 skill 应引用仓库中的稳定入口，
不能依赖仅存在于某台机器的项目脚本。

## 唯一推荐入口

| 用户任务 | 主 skill | 推荐入口与边界 |
|---|---|---|
| 计算强子关联函数，包括谱学、多点、流和关联矩阵 | [correlator](lqcdweave-correlator/SKILL.md) | `Hadron/Flavor → gen_correlator → contraction.correlator.calculate`；同任务联合规划 |
| 测量需要 elemental，且有已登记基础输入 | correlator | 同模块的 `Elemental` 声明交给 `calculate`；后端管理补产，不另开手工生产流程 |
| 独立生产 elemental 文件或开发空间核 | [elemental](lqcdweave-elemental/SKILL.md) | 介子/流用 `ElementalGenerator`；三腿重子用 `BaryonElementalGenerator`，均由代理接线既有保存服务 |
| 生产或扩展 Laplace 基底 | [eigenvector](lqcdweave-eigenvector/SKILL.md) | `LaplaceEigenvectorGenerator`；必填 `backend`，字符串选择已有后端，非字符串传入用户后端对象 |
| 按已定父分组生成实际 QR dilution 向量 | [noisevector](lqcdweave-noisevector/SKILL.md) | `NoisevectorGenerator`；不自动选择重要性分组 |
| 生产 V/D/P/H 传播子 | [perambulator](lqcdweave-perambulator/SKILL.md) | `PerambulatorGenerator`；每次计算显式给源和有序汇列表，产品由实际端点决定，H 沿用 P 编码 |
| 生成随机空间点集 | [spatial-sampling](lqcdweave-spatial-sampling/SKILL.md) | `sample_spatial_points(SpatialSamplingDesign(...), seed=..., selection=...)` 返回 `SpatialPointSet`，其坐标采用 `(time,point,xyz)` 只读 `int64` |
| 新建稀疏时间选择 | [temporal-sampling](lqcdweave-temporal-sampling/SKILL.md) | `select_t`；已有记录统一 `load_time_selection`，不重新选择 |
| 按已有时间选择重建两点函数 | temporal-sampling | `reconstruct_two_point_from_coordinates`；旧无记录数组才用兼容入口 |
| 分配多份独立计算任务 | [dispatch](lqcdweave-dispatch/SKILL.md) | 多消费者共享队列用 `Dispatch`；数值工作仍由对应主 skill 执行 |

已有产物直接绑定，不因经过某个 skill 而重算。多个阶段按实际依赖依次路由；集群提交由
目标环境的提交 skill 负责，不由数值 skill 隐式提交。单 cfg worker 不必建 Dispatch 队列；
已明确每数组项对应一个 cfg 的任务保留静态映射，与共享队列消费者模式互斥。

<a id="frontend-boundary"></a>
## 前端与内部接口

关联函数使用 `contraction.correlator.calculate`，保留复杂调度和
elemental 自动准备。其它基础产物的库内计算并保存入口已删除，不保留兼容导出；
范例必须显式展示生成器或 sampler 的计算调用，以及已有保存接口的独立调用。
不在库中重新包装等价的基础产物 runner。

常规测量固定使用 `calculate`。受管 `Elemental` 是它的输入声明，不是第二个测量入口。
仅在[支持范围](../../docs/modules/correlator/managed_elemental.md)内自动补产；其它基础数据不会自动求解。

EV/NV/PR/SS/TS 和独立 EL 目前提供数值类、适配器及存储服务，不能声称都已有一个
统一持久 runner。代理负责按其公开接口完成可计算脚本、输入读取和保存；用户只选择
科学输入、既有后端、范围和输出，不填写内部回调或另造缓存框架。缺交接时定位并修复
后端，不把未实现能力写成已支持。

E/P/X/R、row writer、临时窗口、低层 profile 矩阵和计划存储模块用于后端开发或明确的
专家输入场景。不要为普通任务提供这些模块与 `calculate` 的多个任选入口。
同一功能的兼容别名保留读取/导入兼容，新脚本采用表中规范入口；不同输入或物理语义的
接口不以名称相似强行合并。支持范围见[受管 elemental](../../docs/modules/correlator/managed_elemental.md)，GPU 输出的条件入口见[输出后处理](../../docs/modules/results/gpu_output_postprocessing.md)。

## 共享科学与数据约定

- 实际 cfg、基底/点标签、顺序、物理时间、规范场版本及作用量定义决定身份；shape、seed、
  目录名不证明兼容。未知来源保持未知。Laplace、Dirac、elemental 的规范处理分别记录。
- 时间选择与 `Lt/Nt` 的唯一交接规则见[时间工作流](../../docs/modules/temporal_sampling/temporal_production.md)。空间点集含
  实际坐标、池和顺序；实际 D 含父分组、列标签和池。保持借用数据在消费期间有效。
- [数据格式](../../docs/modules/storage/data_storage.md)统一保存与发布；[项目恢复](../../docs/modules/dataset_planning/dataset_planning.md#project-resume)
  管理已有依赖。计算块大小、文件长度、预分配及已完成覆盖是不同概念，不以文件存在判完成。
- 测量内 `elemental_weighted` 只在 `calculate` 选择；独立生产预权数据时由生成器使用
  同名选项并保存标记，消费与之匹配。H 契约见[投影点](../../docs/modules/perambulator/point_highmode.md#h-contract)，
  D 的统一包含系数见[计权定义](lqcdweave-elemental/references/dilution.md)。
- 数学参考、CPU、模拟设备、真实 GPU/DCU、MPI 和完整生产分别报告证据。校验按本次
  改动及实际风险选择，不把历史作业成绩或实现日志变成每个任务必做的清单。

<a id="execution-scope"></a>
## 参数核对与执行范围

先从用户消息、现有配置及实际输入解决已知项，完成可审阅脚本与适用验证。科学定义或
执行范围确实缺失时只询问该缺项；已有“执行/开跑/提交”授权在同一任务范围内有效，
无需为代码版本号或例行性能修复再次索要确认。仅要求拟定脚本不等于要求启动正式计算。
改变未授权的物理任务、输入范围或破坏性操作时说明差异后再处理；沉默和超时不构成授权。

共同核对 cfg/格点、实际输入及身份、输出与恢复范围、后端/资源、精度和适用验证。
用户只需理解实际方案；不强制填表、给固定口令或重复提供已知参数。各任务再核对以下差异：

| 任务 | 特有科学参数 |
|---|---|
| EV | Laplace 规范处理与边界、Ne、谱顺序、容差、旧前缀策略、本征值与向量配对 |
| NV | 实际父基底、父分组、每组 n/m、列顺序、随机实现及跨时间池关系 |
| EL | 路径/导数/三腿配方、component 和动量顺序、相位、V/D/P 扇区、计权及结构声明 |
| PR | 质量/作用量/边界/自旋基、真实源汇、tol/maxiter、产品与物理时间覆盖；H 另核对完整父投影和保存方向 |
| SS | Np、均匀无放回设计、时间段共享、实际点池；确定点集只核对坐标和来源 |
| TS | Lt/Nt/loop、所需 dt、评分与覆盖、实际 selection 顺序、重建布局及配对数 |
| 关联函数 | 完整算符/味道/动量/表示及行列基、Wick 贡献、源端 dagger、时间和归一化 |

谱学另核对净味道扇区的 C 闭合及移动系标签；多流另核对每条流的味道、方向、转移动量、
锚点/真实端点、重合及接触项。仅实际随机测量才核对共享池和联合包含系数，不给纯模态
任务强加点集信息。逐腿差异保留实际映射，不能用共同默认值覆盖。
