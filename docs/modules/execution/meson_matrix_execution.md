# 介子矩阵优化别名与低层核

`calculate(..., optimize="meson_matrix")` 是统一 `anytime` 搜索入口的兼容别名。它使用相同候选搜索、搜索预算、固定 DAG 存储排程和独立资源审计；不锁定 K/H 逐行次序，也不跳过搜索。新代码可继续使用默认 `optimize="anytime"`。低层 K/H 数值核和 `MatrixKernelSpec` 记录保留给兼容 Action 使用，不构成独立 planner 或设备执行路线。

完整 cfg 计划保存已选模板、坐标实例、输入/准备描述、资源事件、host/device 峰值与预读额度及输出布局。当前 complete-cfg payload 使用 `unified-device-v2`；常规加载只解码保存的安排，不搜索、审计、重演路径、重新合批或预测成本。旧 complete-cfg 编码不迁移；须按当前任务重新规划并保存。`SceneStoragePlan.kernel_batches` 保留已选低层批注，加载和执行不会重新合批。详见[规划 L2](../planning/planning.md)与[档案 L2](../plan_archive/plan_archive.md)。

C 定义纯目标记录 `ExecutionTarget`；M.api 提供 `register_execution_target`、`load_execution_target`、profile 解析和本地校准缓存。收缩编排的 `target_hardware` 可传已登记目标名称或记录；名称本身不加载或初始化运行库。实际执行使用调用方指定的 backend 和相应 reader/runtime。详见[性能 L2](../performance/performance.md)。

X 的 `device_execution` 统一处理 NumPy 与 CuPy；NumPy 使用空设备 scope，CuPy CUDA/HIP 使用同一资源所有权接口。`debug=False` 是临时执行选项；`run(debug=None)` 沿用本次 `Calculation` 的选项，显式 debug 只控制重复深度审计，不写入档案，也不改变测量、资源依赖、cleanup 或同步。详见[执行 L2](execution.md)。

全部 GPU 核、DCU/HIP 与生产规模仍待验收。10 秒/64 MiB 自动短测只校准合成 dense contraction 与 snapshot；不覆盖 source preparation、文件读取、预读/传输、稀疏/重子核、批核、输出组装或资源移动。既有 V100 密集批核、旧矩阵固定路线与 fake-CUDA 数值仍是历史诊断，不能代替当前任务验证或外推到全 GPU 路线。

当前矩阵及完整计划契约见[矩阵架构](../../architecture/meson_matrix_architecture.md)。
