---
name: lqcdweave-dispatch
description: "使用 LQCDweave Dispatch 分配多 cfg 或其它独立任务，组织共同准备、逐任务计算与完成核验；需要跨 cfg 预读时保留完整计划的 iterator 入口。单 cfg 数值计算使用对应计算 skill。"
---

# LQCDweave · Dispatch 任务分配

共同任务路由、前端边界、保存/身份及执行范围统一见[技能共同约定](../README.md)。只读取当前任务的条件参考。

多任务脚本统一用 `adapters.mpi_dispatch.Dispatch` 领取任务；计算、计划、输入和保存仍调用已有模块。任务通常是一份 cfg，也可是一行能由脚本明确解析的独立任务标识。单 cfg 调用不必建立队列。

## 一般原则

- 先明确任务单位、清单、共享路径、同一次运行的 suffix，以及一个消费者使用多少 MPI rank。不同任务的输出位置应独立；同一输出有依赖的阶段按已有阶段服务排序，不把依赖任务直接当成可任意领取的独立任务。
- 算符、图、数据源声明、计算函数、可兼容复用的计划和输出规则放在 cfg 循环前。循环内只绑定当前 cfg 的实际输入/选择、计算、保存并核验完成；依赖 cfg 的实际基底、点集和时间记录不能提前固定为代表 cfg 的值。
- Dispatch 分配任务，消费者调用现有计算和 storage/runtime 服务。固定关联函数生产必须在循环前联合全部兼容通道/表示行/径向需求，在代表 cfg 上选定一次模板（允许首次领取后惰性初始化一次），此后循环只兼容绑定和回放；`partial(calculate, ...)` 或 Dispatch 不会自动复用计划。禁止把逐通道/时间/cfg 重复 calculate 当成生产模板。模板不兼容时记录原因，按 [correlator 的固定回放规则](../lqcdweave-correlator/SKILL.md#生产任务必须联合规划并固定回放)处理；完整 cfg iterator 入口仍按下文保留。
- 每个独立 MPI 作业是一组消费者，rank 0 领取后向本作业全部 rank 广播同一任务。多 rank 如何协作由计算模块决定，产物和完成标记只写一次。共享队列的单 GPU 任务通常每消费者一个 rank，此时 Slurm array 索引标识消费者。若任务已明确每数组项对应一个 cfg，则采用静态映射，worker 直接计算该 cfg，不再同时领取 Dispatch 队列。
- “已领取”与“已完成”分开。已有数据按实际格式、身份和覆盖核验；只有保存和核验成功后才记完成。`combine` 是追加标记，不是产物验证或自动恢复。默认数据布局见[数据保存说明](../../../docs/modules/storage/data_storage.md)，跨阶段恢复能力按当前实现与验收确认。
- 不同 cfg 分别写各自数据文件；共享目录的 `dataset.json` 由 storage 在短登记锁内增量更新，不在脚本中重建全量登记或给整个计算循环加锁。格式与调用方式不变；指定 cfg 的读取只恢复它的元数据。元数据性能核验见[登记基准](../../../benchmarks/README.md#dataset-registration)。
- 同一运行的全部消费者显式共用 suffix；新运行用新 suffix。同名 `.tmp` 继续剩余队列，不能通过重建 Dispatch 或修改原清单让它自动补回任务。失败/提前停止可能留下已领取未完成项；复跑前核对原清单、剩余队列、有效产物及仍运行的消费者，明确生成待补任务范围。

## 默认：逐任务循环

安装 MPI 可选依赖及队列锁、共享文件系统要求见 [Dispatch 用法](../../../docs/modules/dispatch/dispatch.md)。下面的 `compute_one` 和 `save_and_verify` 是按任务使用已有接口准备好的调用函数；示例按单 rank 消费者组织，多 rank 时还须协调全部 rank 的成功状态与单次输出。

```python
from lqcdweave.adapters.mpi_dispatch import Dispatch, combine

# 共同定义、计算函数和可复用计划已在循环前准备。
tasks = Dispatch(cfg_file, run_suffix)
for cfg in tasks:
    result = compute_one(cfg)
    save_and_verify(cfg, result)
    if tasks.rank == 0:
        combine(completed_file, cfg)
```

循环前准备已有计划不等于所有 cfg 的数值输入常驻；reader/借用数组的有效期仍遵守计算接口。队列耗尽只表示分配结束，任务整体完成要核对原清单要求的有效输出。

## 保留入口：跨 cfg 连续预读

需要完整关联函数计划的跨 cfg 预读时，直接把 Dispatch 作为 `configuration` iterator 交给 `calculate(plan_archive=..., ...)`，不在外层拆成逐 cfg 的独立计算调用，也不先转成 `list(tasks)` 领取全部任务。现有执行服务维护连续有界 ReadAhead，只有 owner thread 推进任务 iterator；后台线程只读取已领取 cfg。

memory 模式惰性交付 `(cfg, values)`，提前退出用 `contextlib.closing(result.value)` 或 `close()`；Dataset 模式在共享目录逐 cfg 发布完整产物，由 `dataset.json` 登记。简单数组为每 cfg 完整 NPY，多字段结果为 HDF5；格式细节见[数据保存说明](../../../docs/modules/storage/data_storage.md)。保存计划、绑定当前 reader/source factory、输出模式和 CPU/GPU 限制见 [完整 cfg 计划](../lqcdweave-correlator/references/plans.md)；接线示例见 [Dispatch 的预读入口](../../../docs/modules/dispatch/dispatch.md#跨-cfg-连续预读)。计划兼容性、结果组装及资源预算仍由原模块处理。

## 恢复与按需交接

当前队列领取会移除任务，没有自动放回、租约或重试。完成标记不能替代产物验证；恢复沿用 [既有产物与项目恢复](../../../docs/modules/dataset_planning/dataset_planning.md#project-resume)，不另建检查点或来源系统。共享队列不安排跨阶段依赖、不切分 cfg 内坐标、不改变科学参数或抽样规则。

清理队列前确认同 suffix 的全部消费者结束并核对完成范围；`clear()` 的 barrier 只覆盖当前 communicator，不能代替其它独立作业的结束确认。保留已有运行授权和范围，任务分配本身不提交作业。

计算需求按产品使用 [基底](../lqcdweave-eigenvector/SKILL.md)、[elemental](../lqcdweave-elemental/SKILL.md)、[传播子](../lqcdweave-perambulator/SKILL.md)、[空间取样](../lqcdweave-spatial-sampling/SKILL.md)、[时间取样](../lqcdweave-temporal-sampling/SKILL.md) 或 [关联函数](../lqcdweave-correlator/SKILL.md) skill；只读取当前任务涉及的说明。
