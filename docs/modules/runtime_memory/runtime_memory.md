# Runtime 内存观察服务

职责依据 [架构](../../../ARCHITECTURE.md) 与 [Correlator L0/L1](../../architecture/correlator-l0-l1.md)。
`runtime.memory` 从旧 root budget 迁入已有观察与执行 guard，不新建设备管理器。
共享记录和异常唯一定义在 `contracts.memory`，纯字节估算/上限合成归 P.api。

公开接口为 `host_available_memory()`、`device_memory_info(backend)`、
`resolve_memory_limits(budget, backend=None, *, host_available=None, device_info=None)`、
`actual_available_bytes(backend=None)` 与
`ensure_within_budget(estimate, budget, *, backend=None, limits=None, label="execution batch")`。
逐批执行的 pool-aware 上限由 `fresh_execution_limits(budget, backend=None)` 提供；
driver free 与本进程可复用 pool free 分字段保留，并仅合计一次用于增量上限。
X 消费返回的 C MemoryLimits，不直接探测 driver 或默认内存池的全局可用容量。
`RunResources.reusable_bytes` 只在自身内存池空闲、设备上下文匹配且成功取得所有权锁时，
读取该池可复用字节及已有常量字节；通过 runtime 的作用域 provider 参与容量合成。
忙碌或并发执行不得借用该额度。此所有权记账不授权其它 X 模块探测显存容量。
每次 fresh 调用只查询一次 driver 和一次 host；pool 查询失败按零可复用字节处理。
driver 查询失败沿用 `device_memory_info` 的 BudgetError，预算不足才报
BudgetExceededError。旧 X 实现冗余第二次 driver 查询曾另包为 BudgetExceededError；
该第二次采样与不一致包装随职责收口移除。

显式后端可以是 NumPy/CuPy 模块或忽略大小写的字符串；默认 NumPy。
非法后端报 ValueError。host 观察优先 psutil，POSIX 可用 sysconf fallback；
未知保持 None。GPU memGetInfo 失败报 BudgetError，CPU 不查询设备。
未注入的观察每次重新查询，注入的值不触发对应探测；None 表示未注入。
解析只向 P.api 传 target 与 host/device 数值，不向 P 传设备对象。

已保存 workload 政策保留 70% host available allowance；显式 cap 与它取较小值。
未知 host 时显式 cap 仍可用，否则 BudgetError。GPU free 先扣除
`max(min_reserve_bytes, int(total_bytes * reserve_fraction))`，再与显式 device cap 取较小值，
结果下限为零。沿用旧 floor 与 zero-total 范围；C ExecutionBudget 的 ceil 方法
有不同规则，不在此次职责迁移中替换。逻辑峰值超出有效上限报 BudgetExceededError。
启动可行不保证执行时资源仍可用；X 保留逐批 fresh guard 和真实后端错误传播。

`adapters.hardware_budget` 使用同一公开 host/device 观察，但其场景规划政策仍为
80%、cgroup/RLIMIT 限制及离线声明，不与保存 workload 的 70%/reserve 政策合并。
该 adapter 不获得数值数组所有权，runtime 观察不证明物理分配峰值。

专项覆盖未知/注入/fresh 观察、默认与显式 cap、错误分类、P 委托及 adapter
限制策略；CuPy 模拟观察仅验证协议。真实 GPU/DCU、MPI 与生产峰值仍待验收。

## 与 X 局部执行资源的分界

runtime 负责容量观察、显式目标解析和进程级资源生命周期。X 的 DeviceExecution 负责单次 contraction
的有界池、上传 stream/event、借用输入与数组释放栅栏，支持嵌套作用域并在异常退出时清理。
该局部所有权不允许 X 自主选设备、memGetInfo 容量探测或全局设备注册，也不允许 runtime 反向导入 X。
相关所有权与清理回归见 execution/test_device.py、test_loader_cleanup.py；CPU/fake backend 不能代替真实 GPU 验收。
