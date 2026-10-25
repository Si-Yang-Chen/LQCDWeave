# Dispatch 任务分配

多 cfg 或其它独立任务统一由 Dispatch 提供任务 iterator；共同定义、计算函数、可兼容复用的计划与输出规则放在循环前，循环内绑定当前任务的输入、计算、保存和核验。完整的脚本组织原则见 [任务分配 skill](../../../.agents/skills/lqcdweave-dispatch/SKILL.md)；单 cfg 计算继续使用对应模块入口。

Dispatch 只领取任务标识，不解释文件格式或科学参数。任务通常是一份 cfg；其它任务可以用一行独立标识，由脚本明确解析。共享输出有依赖的阶段仍按已有阶段服务排序。普通 `partial(calculate, ...)` 仍可能每 cfg 规划；复用路径需先准备兼容计划，再按当前 cfg 绑定输入执行。

## 逐任务循环

安装 MPI 可选依赖后，使用新包内的旧式入口：

```sh
python -m pip install -e ".[mpi]"
```

```python
from lqcdweave.adapters.mpi_dispatch import Dispatch, combine

# compute_one、save_and_verify 按对应模块接口在循环前准备。
tasks = Dispatch("/shared/cfgs.txt", "run-20261009")
for cfg in tasks:
    result = compute_one(cfg)
    save_and_verify(cfg, result)
    if tasks.rank == 0:
        combine("/shared/completed.txt", cfg)
```

示例按单 rank 消费者组织。多 rank 的计算、保存和成功确认须按计算模块协调，产物和完成标记只写一次。reader 和借用结果的有效期遵守原计算接口，不因循环前准备共同定义而跨 cfg 保留已失效数据。

## 共享队列与 MPI

`Dispatch` 从旧 `lattice/dispatch.py` 复制队列与锁语义。初次使用时创建 `cfgs.txt.<suffix>.tmp`；同一次运行的全部消费者须显式共用 suffix，新运行用新 suffix。省略 suffix 会为每个独立 MPI 作业生成随机值，不能组成跨作业共享队列。已有同名队列会继续使用，绝不能把它当作全新任务清单。各独立作业的 rank 0 在文件锁下取走一行，再向本 MPI 作业中的其它 rank 广播。空行及 `#` 注释不成为任务。Slurm 数组下，**数组索引选择一个消费者，不是组态号**。

默认适配器使用 `mpi4py.MPI.COMM_WORLD`。只用一个 GPU 处理一个组态时，每个数组元素须为一个 MPI rank；若同一个作业启用多个 rank，全部 rank 会收到同一个组态，调用方必须自行协调并行计算与单次输出。`runtime.dispatch.Dispatch(..., comm=...)` 是显式通信对象入口，可用于其它通信实现或无 MPI 的测试。

## 跨 cfg 连续预读

已有完整 cfg memory 计划时，直接把 Dispatch 交给现有执行入口；当前 reader 由 `source_factory(cfg)` 上下文管理器提供，按保存的角色绑定：

```python
from contextlib import closing
from lqcdweave.adapters.mpi_dispatch import Dispatch, combine
from lqcdweave.contraction.correlator import calculate

# 完整计划、source_factory 和保存函数已在领取 cfg 前准备。
tasks = Dispatch(cfg_file, run_suffix)
result = calculate(plan_archive=archive_file, configuration=tasks,
                   diagram_adapter=source_factory)
with closing(result.value) as cfg_results:
    for cfg, values in cfg_results:
        save_and_verify(cfg, values)
        if tasks.rank == 0:
            combine(completed_file, cfg)
```

只有 owner thread 调用任务 iterator 的 `next()`，后台读取线程只读取已领取 cfg。连续有界 ReadAhead 可跨 cfg、坐标和时间对；不要用 `list(tasks)` 提前领取全部任务，也不要在外层逐 cfg 调用 `calculate` 来替代这个入口。提前停止时关闭结果 iterator，清理在途工作；已领取但未完成的 cfg 仍需恢复核对。普通循环也可复用完整计划，但每次独立调用不共享跨 cfg 预读队列。

Dataset 模式使用规划时已固定为 dataset 输出的完整档案，并在上述 `calculate` 调用中传入 `output=output_root`。多个 cfg 共用 `output_root`，每个 cfg 的完整结果作为 NPY 或 HDF5 文件保存，由根 `dataset.json` 登记；成功返回的 `result.value` 是 cfg 到输出位置及数据 ID 的映射。核验后由 rank 0 追加完成标记。若途中异常，按已登记的 cfg 完成状态核验，不能因没有返回整个映射便认定全部失败。输出模式在档案中固定，不在加载时切换 memory/dataset。格式说明见[数据保存说明](../storage/data_storage.md)。

完整计划的准备、绑定和输出接口见 [关联函数前端](../correlator/correlator_frontend.md#保存完整计划重复-cfg-直接执行) 与 [生产脚本与保存](../correlator/correlator.md)。Dispatch 的引入不更改计算、结果组装、资源预算或 reader 生命周期。

## 完成、恢复与清理

队列只表示待分配任务：取走任务后进程失败不会自动放回，预读和提前停止也可能留下已领取未完成项。队列为空只表示没有待领取项。复跑前核对原清单、当前 `.tmp`、有效输出和仍在运行的消费者；已领取但没有有效完整产物的任务也要纳入明确的待补清单。修改原清单不会重建已有同 suffix 的队列。

`combine` 只是加锁追加完成标记，不替代数据集登记校验。已有数据沿用实际格式、身份与覆盖检查；跨阶段缺失数据的恢复入口和验收范围按当前项目回执/依赖图实现核对，队列标记本身不完成恢复。`clear()` 会删除队列及初始化锁文件；其 barrier 只覆盖当前 MPI 作业，须先确认同 suffix 的全部独立作业已结束并核对完成范围，再由操作者明确清理。

队列锁固定在文件第 0 字节，因此 append 文件增长后各写入者仍争用同一把锁；进入上下文时会恢复调用方原有的读写位置。同步或解锁失败时仍会尝试释放锁并关闭文件，首个同步/清理错误会传播给调用方。若任务行已从队列取出，发生此类写入错误时仍需根据队列和完成台账人工核对，不会自动回滚或重试。这里验证的是本机文件锁行为，不代表共享集群文件系统已经验收。

cfglist、队列、完成标记和输出须位于所有作业可见的共享文件系统。库不选择 Slurm partition、设备数或重试策略，也不从文件名推断规范场或本征基底身份。
