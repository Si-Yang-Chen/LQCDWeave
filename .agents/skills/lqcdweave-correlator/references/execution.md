# 只规划、执行与持久输出

普通任务的唯一数值入口是 `calculate`；`execute=False` 只规划，已选对象用 `.run()` 执行。
reader 在执行完成前保持有效。规划、执行、保存各阶段按[任务路由](../../README.md)区分。

后端开发中，CPU 稀疏缩并可保留方程明确要求的独立端点坐标；这不等于同点约束。
双流环在后续因子仍需要该坐标时必须保留，并交由共同规划与 bounded COO 回放处理。
该底层能力不提供点分辨测量的新用户入口，也不扩大坐标合批 sparse 的 GPU 支持范围；
普通测量仍从 `gen_correlator` 接入 `calculate`。轴和预算契约见
[执行说明](../../../../docs/modules/execution/execution.md#coordinate-batched-sparse-bridge-workspace)。

```python
planned = calculate(correlator, propagator_map=propagators,
                    vertex_map=vertices, backend=backend, execute=False)
result = planned.run(backend=backend, measure=True)
values = result.value
```

同一对象反复 run 复用路径与适用的执行描述，每次重新读取当前数值并交付独立结果；
不同 calculate 调用不自动共享计划。`debug=None` 继承原设置，debug 只加深校验，不改变
科学定义、必要同步或生命周期。dtype 规则和计时口径统一见[性能](performance.md)。

固定回放内部通过操作游标读取原计划的绑定视图，不新增前端开关或另一份计划。
device预读在原额度下分开IO与CPU打包/上传提交，计算游标等待相应ticket并复用设备event。
小型resident inline、CPU普通任务及测量模式保留原短路径；真实GPU验收与线程协议验证分别报告。
重子join在原入口内拆分当前准备与计算，沿用既有数值选择和存储档案；无需用户编写prepare。
实际范围见[重子内核准备](../../../../docs/modules/execution/baryon-join-preparation.md)，不据此宣称跨操作准备已并行。
游标本身不执行计算、不重新选设备，也不意味着所有CPU准备已异步化；完成凭据和
IO/设备缓冲继续由现有执行服务管理。内部接入范围见[执行L2](../../../../docs/modules/execution/execution.md)。

CUDA 的有界 thin-wide reduction 自动复用完整/尾 tile 的主机参数绑定，无需新增选项；
它只减少参数交付，不改变科学算符、跨 tile 的宽精度累加、工作区预算或计划档案。
首次编译和稳定期运行分别计时，是否改善完整任务仍须同设备、同输入测量。

同一模板中完全相同的 thin-wide core 自动共享不含数值 owner 的执行工厂，并在绑定时
解析固定几何；每个 Action 和每次当前输入的契约检查继续保留。用户仍只复用
`calculate` 的已选对象或完整档案，不需要手工拆任务、缓存数组或设置新的优化开关。

显式开启 `calculate(..., factorized_gram_strategy='tiled')` 后，较大 tiled thin-wide 的矩阵型第一阶段复用相邻输出点的输入读取；每点保持原
宽精度累计次序和已选 partial/finish 接口。不适用结构沿通用核，用户入口与预算不变。
tiled 是当前调用的执行选择，普通 `.run()`、完整 cfg 与 iterator 回放保持该选择，
不写入科学档案身份。读档时可显式选择。首次编译成本与完整任务的暖机收益仍需分别测量，
内部条件见[执行 L2](../../../../docs/modules/execution/execution.md)。

策略 `'gemm'` 则在新离线准备时让 P 为适用几何保存宽输入包、完整 Gram 及其预算，
X 按该数值实现回放。用 `calculate(..., factorized_gram_strategy='gemm', execute=False)`
准备后保存完整档案，后续 `calculate(plan_archive=..., configuration=tasks, ...)` 跟随档案；
不能在旧 reduction 档案上临时指定 gemm。默认 None 对新 CUDA 任务尝试保守 GEMM 选择，对其它新任务选择 reduction；对档案遵循
已保存实现；旧 bool 参数仅兼容 reduction/tiled，冲突设置报错。适用性与冷暖取舍见
[Gram 数值策略](performance.md#factorized_gram_strategy)。
gemm 未选中的归约使用 tiled 输入复用，不兼容结构继续用通用归约核。

普通与完整 cfg 的 NumPy/CuPy CUDA/HIP 路线共用执行服务；backend 必须匹配已选目标。
coordinate-batched sparse GPU、无可序列化 preparation recipe 的 opaque 归档仍不支持。
范围及内部 Action/队列/所有权实现只在[执行 L2](../../../../docs/modules/execution/execution.md)维护。

## 结果写出与恢复

当前绑定的内存结果用 `save_correlator` 保存；多坐标 Dataset 写出用
`selected.run(output=..., time_axes=...或time_pairs=..., ...)`，后端管理有界窗口与发布。
完整档案、多 cfg 和跨 cfg 预读统一见[计划复用](plans.md)，不在脚本里循环低层窗口执行器。
数值结果的坐标与平均见[结果](results.md)，身份、发布和恢复见[数据格式](../../../../docs/modules/storage/data_storage.md)。

run 返回的 Calculation 本身不等于发布数据集。惰性结果提前停止须 close，或用
`contextlib.closing`；已发布覆盖按实际定义核对。借用数组在失效前消费或复制。
`calculation.close()` 释放句柄的闲置设备资源，已有数值结果仍有效。

## 显式 GPU 输出后处理

默认 CPU 后处理。若明确要求完整 cfg Dataset 的 GPU 后处理，采用唯一条件路线：
先对已选单坐标计划调用 `complete_calculation(..., output_mode="dataset", postprocess_backend="gpu")`，
再 `.run()`；需反复使用则保存完整档案。不要同时推荐显式窗口 runner 作为另一个普通入口。
普通固定绑定 `.run(output=...)` 目前不接受此参数，不能伪造转交；memory 输出不支持 GPU 后处理。
后端、输出布局及资源预算在补全时固定，加载时不改写。具体选项与实例见
[GPU 后处理](../../../../docs/modules/results/gpu_output_postprocessing.md)。

<a id="expert-resources"></a>
## 专家资源：观察和临时存储是两项独立选择

以下仅用于已有低层回放或后端开发，不是普通 calculate 的另一条流程。

- 设备观测：`SceneDeviceAdmission` 把动态观察交给已有 batch 服务；它不设置 pool cap。
  真正 pool 限制使用 `scene_device_budget`，需要默认池独占，也不承诺整个设备硬上限。
- 磁盘暂存：已有可行单路径 NumPy 计划可使用 `TemporaryNpyArrayStore` 和正的
  `disk_limit_bytes`；只支持 owned contiguous host tensor 的 spill/restore。
  临时文件不构成科学产物、跨任务缓存或恢复覆盖。GPU moves、跨批暂存及不支持布局拒绝。

精确接线、所有权及异常清理见[执行 L2](../../../../docs/modules/execution/execution.md#共享块回放与存储策略)。
模拟设备协议通过不代替真实设备验收。
