# 时间扫描和批量

`Tag.time`、`Flavor(time)`、`gen_correlator` 和 `time_axes` 可统一使用用户自定的非空字符串键，例如 `source/sink/current`；这些词无保留语义。相同键表示同一变量，不同键独立；整数标签仍支持。`time_axes` 的值遵守 reader 的明确坐标域：默认全时间为物理坐标，紧凑 reader 为非负槽索引；输出扫描轴按声明顺序排列。

常规扫描直接指定 reader/adapter 契约中的时间坐标。以下示例针对全时间物理 reader；本例明确声明符号0是汇、1是源：

```python
result = calculate(
    correlator, propagator_map=propagators, vertex_map=vertices,
    time_axes={0: range(Lt), 1: range(Lt)},
    batch={"axis": 0, "size": 4}, execute=False,
)
```

4只是批宽例值；batch="auto"选择可行批宽。存储宽裕时可以多个汇时间一起导入和向量化，不要求逐源汇对独立计算。尾批保持实际长度。其它任务先核对自己的符号时间标签，不能照搬0/1含义。

已选的普通 dense 单坐标路径也可直接扩展到内存内时间扫描，CPU/GPU 共用 `.run()`：

```python
result = selected.run(time_axes={"source": source_slot, "sink": range(40)},
                      batch={"axis": "sink", "size": "auto"})
repeated = result.run()  # 复用已经补全的批次执行表
```

这里沿用最初联合任务的算符与缩并顺序，不再搜索路径；按当前可用 host/device 预算，
从完整汇宽和较小的 2 的幂候选中选择最大可行批宽。首次补全会重新绑定真实坐标和审计
存储，费用记录在 `prediction["coordinate_batching"]`；数值执行走同一个公共执行器。
与汇无关的输入/子图在批内只算一次，动态输入增加共同保留的时间轴，尾批不填充。
显式 `time_pairs` 可用 `batch="auto"` 或 `"full"`；后端统一其 pair 批轴，
父任务与实际计算批保持同一批次声明。
旧 `kernel_batching` 选择继续沿用；只有仍满足其代数和资源条件的 Cartesian 内核组可保留，
双动态输入用真正的 batched matmul，不能把不同时间交叉相乘。

固定坐标GPU回放与 `output=...` 的公共窗口回放共用pool内的allocator经验余量政策，批宽和预读均不得
占用它；28 GiB预算默认留1 GiB，小预算沿既有20%默认留白上限。CPU为零，未知设备
明确标记未校准回退，结果统计保留来源；这不是任意碎片的数学保证。无需新增运行开关，
从已有坐标模板切换入口时替换旧自动余量，不叠加计费；调用方reserve继续保留。
政策及观测依据见[设备分配与失败证据](../../../../docs/modules/execution/allocation-failures.md#固定坐标回放的池余量)。

此回放要求完整可描述的 dense preparations，保留实际输入精度和科学兼容检查；
opaque、sparse join 和矩阵专用 Action 当前明确拒绝；已有中立准备的 dense-V baryon join
可沿同一缩并顺序提升坐标轴，源端静态输入共享，汇端随坐标合批。输出及尚未消费的时间点
结果都须在预算内；不足时在读取前报错。`output=...` 的既有 Dataset 窗口回放保持原约定，
不接受这个显式 `batch` 覆盖。不要在脚本中捕获 OOM 后缩批，或用逐时间 `calculate` 代替。

重子完整任务使用 `ContractionTask(..., time_axes=..., batch=...).plan()` 固定全部坐标、
批宽和尾批，再 `run()`；该入口的内存和 Dataset 输出均已验证保存/读回，无需前端逐点循环。
当前验收覆盖现有 CPU reader 的 complex128 dense-V 两点、多分量源汇及真实 GPU；
低精度原生输入、受管 VVV 自动补产与完整多重子生产算例分别验收，不能从小核结果推定。

固定重子时间逐角色绑定到 `time_axes` 的实际标量，不能把 dt 列表当作源/流/汇时间。跨时间守恒流保留每项实际端点，并核对周期或开放边界。

普通时间 GaugePath 也须保留路径矩阵行的锚点时间和列的实际终点时间；P/P 两端分别使用该时间片的实际点集，V/V 的模态轴同样按各自时间投影。source dagger 连同矩阵左右轴一起交换时间端点；净零 P/P 路径不消除同图中其它时间链的位移。独立数值参考及矩阵迹公式见[时间路径测量约定](../../../../docs/modules/measurements/diagram_measurement.md#registered-temporal-vp-point-path-diagrams)。

时间轴按用户声明顺序输出；range(Lt)表示完整物理时间扫描。用户明确要求稀疏时间时，先读[共同时间约定与当前接线限制](../../../../docs/modules/temporal_sampling/temporal_production.md)。`selection.times` 是物理坐标，压缩 reader 使用 Nt 个逻辑槽；两者不能互换。显式绑定 selection 的 reader 以 `range(Nt)` 扫描，输出源×绝对汇槽矩阵，selection 重建默认直接接受；旧循环差槽才需 `time_layout="cyclic_difference"`。不得仅按 shape 推断域，源汇不同选择按实际连接角色核对。

普通重子流的固定时间 dense-V 支持范围须按当前绑定和执行核对；普通时间扫描的可用性不代表任意重子采样流都已接通。

常规符号任务由 `gen_correlator` 产生，时间键使用其支持的非空字符串或非负整数；不要为了接受其它对象而另造内部拓扑或绕过符号入口。计划与数值文件继续遵守公开编码支持的类型。
