# 固定路径的时间坐标合批

收缩编排的普通 `Calculation.run(time_axes/time_pairs=..., batch=...)` 将已有 dense
单坐标选择补全为内存内扫描。输入坐标使用既有编码，不从槽差推断物理 dt。
没有新算符、E 展开或路径搜索；无参数再次 `.run()` 使用补全后的同一描述。

P 的 `plan_coordinate_batches` 先调用[坐标输入绑定](coordinate-inputs.md)，再对每个
实际组调用 `lift_coordinate_path` 与既有 `plan_scene_storage`。auto 尝试完整宽度与
较小的 2 的幂，从最大候选开始，找到首个完整可行安排即退出；显式宽度不降级。
每批动态输入新增共同保留的 lane 轴，静态值及子图在该批共享。尾批为实际长度。
`kernel_batching=True` 时重新描述并核算可行的旧 Cartesian 内核组；双动态保留轴
不满足旧组契约时，使用 X 已有的 batched matmul/宽精度 reduction。

所有批次通过既有 `join_storage_batches` 合并 raw 生命周期。顶层准备项按批唯一命名，
标量成员按真实输入/配方身份去重，完整闭包可由同一 `restore_input_loader` 恢复。
收缩编排仅组合已有 `DiagramInputs`、`PlannedContraction.batch_paths` 和 OutputRecipe；
X 共用 task-batch replay，R 共用原嵌套输出恢复。没有第二个数值执行器。

预算包含已保留输出、当前批下载的 host owner、坐标快照和可选 measured report 的
原始数值快照；GPU 端的 output-copy owner 仍由原存储轨迹核算。每个实际批及跨批
raw 保留量都必须可行，才允许读取；执行期仍核对当前资源，不捕获 OOM 后试批。
P 记录候选、选择宽度、实际批长、峰值、内核组数、绑定/排期时间和零次路径搜索。
动态来源、精度、轴、真实基底关系与别名检查保留，重复完整代数回放只在 debug。

X 在每批结果产生后下载已打包的完整 owner，再用 NumPy 拆时间轴，避免每 Term 每
lane 发出 GPU take/copy；外层不重复下载这些已在主机上的结果。CPU 也沿同一拆分。

范围是普通 dense Action 和完整可描述的 identity/spin_spatial/stack/constant 准备。
opaque、sparse/baryon join、矩阵专用 Action 显式拒绝；Dataset 输出仍走原窗口服务，
不接受新 batch 覆盖。CPU 数值测试、真实 GPU gate、真实生产规模分别记录。

验证见 `test_coordinate_path.py`、`test_coordinate_inputs.py` 与
`tests/correlator/execution/test_coordinate_replay.py`；覆盖独立逐坐标参考、复杂源 dagger、
矩形模态、40 汇、尾批、静态/重复坐标、改变输入重跑、预算不足且零 IO。
