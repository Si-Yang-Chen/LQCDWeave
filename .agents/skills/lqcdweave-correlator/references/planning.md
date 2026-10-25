# 优化、联合存储与资源规划

普通调用在 `calculate(..., execute=False)` 规划；元数据阶段不读取数值。一次提交完整兼容
任务，不将通道或坐标拆成独立搜索。选路后的生命周期统一见[计划复用](plans.md)。

后端目标设计见[分层链接与共享操作更新](../../../../docs/modules/planning/planning-shared-links.md)：
保留总体 DFS/并行流程，只改内部数据表示、共享分类、会话内一次估价和局部更新；
已通过 CPU/并行回归，桥接/屏障适配及跨进程缓存范围见该页实现记录，不新增前端选项。
选定路线后的普通稠密、整体介子/重子及 factorized 存储核算共用增量生命周期表，
不重新推导 term 或估价；完整符号回放留在 debug。点稀疏桥仍保留原资源描述回放，
实现范围及固定 DAG 排序器尚未简化的部分见该页第 12 节。

thin-wide core 的派生整数几何在构造/读档时自动绑定，并对最多2048种完整有序规格复用；
不需要前端开关。对象身份、档案字段、旧精度语义和正常域校验保留，不缓存数值或配置输入。

新 CUDA 计划还会在既有选中路线内按预算和成本选择 thin-wide tile，随后保存完整安排，
无需手工设置宽度或拆任务。只有需要保持原规划宽度策略作诊断时，才用高级
`calculate(..., factorized_retile=False)`；默认 None 对新计划启用自动择宽。
完整 `plan_archive` 入口的 None 仅沿用已保存宽度，显式 True/False 会被拒绝，
不能把关闭选项误当作执行时重选旧档案。冷编译、暖执行和新增工作区分别核验，
默认策略不保证每个规模都获得相同收益。

## 当前关联函数路径选项

| 选项 | 推荐用法与边界 |
|---|---|
| `optimize="anytime"` | 默认有预算搜索，返回本次最好可行路径 |
| `optimize="exact"` | 明确要求单个完整 Problem 的精确操作域时使用；不承诺分批全局最优 |
| `optimize="none"` | 明确要求固定直接路径时使用，仍检查预算，storage 仅 None/bounded，不传 search |
| `kernel_batching=False` | 默认跳过普通 Cartesian 合批及 CUDA 独立 GEMM/归约分组；公共执行结构复用仍保留，与时间坐标 batch 独立，见[性能参考](performance.md#kernel_batching) |
| `memory_limit / host_memory_limit` | 字节；CPU 前者为主存预算，CuPy 前者为设备、后者为主存 |
| `reserved_host_bytes / reserved_device_bytes` | 任务常驻等外部预留，与实际工作区共同计费 |

`dynamic_programming` 是 exact 的兼容写法，`meson_matrix` 是 anytime 的兼容写法；新脚本
分别使用 exact/anytime，不把别名当作另一条科学或设备路线。搜索中的 storage
None/bounded/lifetime 解析为同一 lifetime 策略。旧 greedy/lookahead/storage 优化器名
及 candidate_width/lookahead_depth/widen 参数不再属于此接口。

`target_hardware` 指已登记名或 ExecutionTarget，决定规划规格；backend 只选执行实现。
离线规划不初始化目标设备，也不把本机空闲资源当成远端规格。profile 与实测见[性能](performance.md)。

只在任务确需调整搜索时使用 `search`：deadline_seconds、max_states、max_transitions、
search_memory_limit、root_width、branch_width、include_ties、component_workers 等。
根宽度严格限制根名额，并列开关仅扩展后续层同主分候选；exact 不接受 anytime 专属选项。
使用图搜索的路线，根分支默认按 affinity/配额自动并行，脚本应从 `if __name__ == "__main__"` 启动。
不为普通任务另建并行搜索队列。
含 baryon join 的常规非 debug 串行路线先在完整任务上找廉价完整路径，再在剩余预算
内作有限局部改进；只在固定路径交付和存储排期通过后记录可行方案。它保留相同
前端、完整输出、精度和档案，不要求用户选择新的算法名。dense、bridge-only、显式
并行及 verify 路线仍沿既有实现。局部回归已通过，完整真实任务的性能按独立验收记录报告。
初始共享分类是启发式安排，当前 dense 快路还未统一宽边界和终端宽核的身份；
候选合法性仍由整数搜索和最终审计负责。worker 路径的 owner 独立匹配按各 Term
核验宽/窄精度及耦合求和边界，不因其它共享操作传递连组而混用配方。

可选 `search={"scorer": ("sharing_count", "element_reduction")}` 按主、次评分依次比较；
单个名称保持原行为。评分只控制候选顺序，不替代真实执行成本；完整并列策略在两个评分均相同时适用。

计算搜索和取得首个存储可行方案后的优化各有默认3600秒预算；显式 deadline_seconds
只覆盖计算搜索。这不是整套规划的硬超时，首个存储可行方案、展开和交付另计。
有限搜索未找到解不等于证明无解。详细默认、操作域、并行预算与统计的唯一说明在
[规划 L2](../../../../docs/modules/planning/planning.md)。

## 联合复用与读取

共享依据实际物理块、有序标签、准备配方及科学身份；同序嵌套前缀可合并读取，
不同基底或非嵌套选择保留映射。计划复用不保证零重复 IO，以真实读量核对。
同一任务优先固定回放，输入结构/抽样池/包含概率或科学任务不兼容时才重新规划。
`optimize="none"` 关闭缩并搜索，仍保留已选整体内核的读取、缩并和释放安排；
执行直接回放该安排，不要求用户另选执行器或自行预读全部输入。

## 资源与可行性

预算包含输入、准备、共轭/打包、缓存、中间量、工作区、输出、传输及后端预留。
借用视图和自有快照分别计费，规划模型峰值与实测峰值分开报告。
资源不足时报告具体缺口，不能静默减少物理范围、改估计量或切换后端。
低层 `solve_storage` 的最终物理审计会拒绝超额候选并继续搜索；若没有已接受路径，
返回 `UNSOLVED`、`plan=None`，自然结束原因可为 `physical_admission`，不作为不可行证明。
公共选路入口将无可交付路径报告为搜索失败；固定计划超预算仍在 IO 前拒绝，执行不重规划。
临时存储和设备观测只在明确的[专家资源](execution.md#expert-resources)场景使用。
构造、优化和固定 DAG 的内部实现细节统一归 L2，不在 skill 复述逐次优化日志。
