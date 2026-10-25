# Correlator 数值结果存储

## 职责与接口

`storage.correlator_results` 用 `DatasetSpec/DatasetWriter/DatasetReader` 发布和读取
数值结果；不保存符号对象、不执行收缩、不搜索、不另建存储/恢复系统。
每 cfg 多字段结果以 HDF5 保存；数值字段按各自真实 shape/dtype 写入具名字段，根 `dataset.json`
登记公共科学 metadata 与 cfg 记录。GPU 执行可以在传输中临时打包；持久结果保持 HDF5 数值字段的真实
shape/dtype。完整 cfg 的专用执行接线位于 `_cfg_correlator_output.py`；收缩编排只组合公开服务。不使用 pickle。

`save_correlator(directory, value, *, configuration=cfg, axes=None, coordinates=None, field_axes=None,
field_coordinates=None, average_tsrc=False, source_axis=None, sink_axis=None,
relative_axis=None, physical_lt=None, boundary=None, missing='error',
coordinate_domain=None, selection=None, source_selection=None, sink_selection=None,
time_layout='absolute', metadata=None, provenance=None)` 返回发布产物 ID。
`load_correlator(directory, *, configuration=cfg)` 返回中立契约 `CorrelatorResult`，字段为
`value/axes/coordinates/field_axes/field_coordinates/metadata/reduction/provenance/data_id`。
收缩编排的同名接口额外接受已执行的 Calculation，自动接线声明的时间和结构化算符坐标。

- `axes` 是所有数值叶共享的尾部扫描轴，接受非负整数或非空字符串，按声明顺序保存
- `coordinates` 精确覆盖这些轴，保存实际有序标签；默认不排序、不去重
- `field_axes` 用 tuple 容器路径到完整轴名 tuple 的映射，长度须等于叶 rank，尾部须等于共享轴
- `field_coordinates` 用路径到该叶轴坐标 mapping 的映射；其共享坐标须与全局一致
- 未声明的前置维度命名为 `operator_axis_0` 等，并使用位置坐标；不得由 shape 猜算符物理标签
- dict/list/tuple、数字 ndarray、object ndarray 容器、NumPy/Python 数值标量、空容器保留类型和形状
- 数值数组保留 dtype/字节序与数值位模式；返回值拥有独立内存，不保留输入 strides 或对象别名
- 默认原样保存非有限数值；平均所用测量贡献必须有限
- dict key 支持 JSON 标量及其 tuple，拒绝任意 Python 对象；metadata/provenance 必须为 JSON 数据（mapping key 为字符串）
- 当前接口消费主机 NumPy 数组，不宣称设备数组、超内存输出或流式写入能力

## 可选源时间平均

时间映射、配对数及数值均值唯一归 TS 的公开纯/流式服务所有。
本模块的 `average_tsrc` 便利入口只遍历数值容器、接线显式时间声明、调用
`sampling.temporal.api.pair_time_differences/count_time_pairs/mean_time_pairs`，
并校验及编码归约元数据；这是限定的 storage→TS.api 单向依赖，TS 不导入 storage/收缩编排。
保存端保持 float/complex 叶的原 dtype，以 `result_type(dtype,float64)` 求和后写回；
整数/bool 平均输出为 float64。TS 数组前端默认的精度提升不改变该保存端政策。

默认 `average_tsrc=False`。启用必须给源角色、另一时间角色和 `boundary='periodic'`。物理域显式 `coordinate_domain='physical', physical_lt=Lt`；槽域显式 `coordinate_domain='slot'` 及有序 selection，由记录提供 Lt。key 名称不赋予物理含义。共享轴恰好为 source/sink 或 source/relative 两轴；额外时间变量、未知域、非周期边界、重复、越界或非整数时间在写入前拒绝。

绝对布局按 `dt=(sink-source)%Lt` 将贡献对齐，再除以每个 dt 的实际对数，
不沿固定 sink 简单取平均。保留所有算符轴，以相对 dt 替代 sink 并删除 source。
相对布局必须声明各源共用的真实物理 dt 坐标，只对 source 轴平均。
源槽×循环差槽显式声明 `time_layout="cyclic_difference"` 与共同 selection，保存器按实际端点换算物理 dt，不把 Nt 当 Lt；未给 mapping 不自动猜测，也不修改生产/收缩时间域。

平均输出覆盖完整 `0..Lt-1`；未测 dt 默认报错。只有显式 `missing='nan'` 才保存 NaN 空档，
对应 counts 为零。reduction 保存模式、输入轴/实际坐标、周期边界、物理域、Lt、贡献计数和缺失策略。
减少源轴是不可逆操作；载入保持已归约语义，重新保存保留历史，再次 source 平均拒绝。
对部分覆盖按实际贡献平均不等于保证无偏重建；采样设计、边界相位和物理估计量由调用者明确。

## 生命周期、验证与所有权

保存先校验全部描述和语义，再将当前 cfg 的数值字段写入 HDF5 并更新根登记；已发布 cfg 不可覆盖，失败不登记为完成。
载入核对字段路径、shape/dtype、容器引用、轴坐标与归约 counts/NaN 空档。没有加载 Python 对象或执行保存内容的入口。
额外内存与写入缓冲按实际执行路径核算；产物的逻辑身份由 `DatasetSpec` 管理。

验收：`tests/storage/test_correlator_results.py` 覆盖异构嵌套/标量/空/object 容器、
整数时间 key、全/部分源覆盖的独立 dt 数值参考、显式缺失策略、相对轴、重复坐标、非有限原样保存、
重复平均拒绝、不可变发布、中断及登记语义检查。
收缩编排交接另由 `tests/test_correlator_frontend_roundtrip.py` 验证。

## 显式紧凑槽坐标

`save_correlator(..., coordinate_domain="slot", selection=record, source_axis=..., sink_axis=..., boundary="periodic")` 保留原始数组与有序物理映射；`average_tsrc=True` 以物理 dt 配对平均。selection 提供物理 Lt，额外 physical_lt 若给出必须相同。`source_selection/sink_selection` 支持绝对布局的非对称覆盖，不能与共同 selection 混用。默认 time_layout="absolute"；历史循环差槽必须改用 relative_axis 与 time_layout="cyclic_difference"，要求相同端点记录。槽坐标可为有序子集，必须是相应 Nt 范围内唯一整数。

归约仍用原 serializer/DatasetWriter，保留所有容器和算符轴；扩展的 reduction 保存 selections/time_layout，load 重新计算 dt 计数以拒绝损坏元数据。缺失默认 error，显式 nan 保留零计数与 NaN。已归约结果不能二次平均或反演成原始时间矩阵。完整布局、示例和限制见[两点重建](../results/two_point.md)。

## 完整 cfg 流式存储服务

公开入口仍为 `storage.correlator_results`：

- `CfgOutputWriter(plan, cfg, directory, binding, *, debug=False, coordinates=None)` 消费冻结的逐点字段布局
- `PackedDtOutputWriter(plan, cfg, directory, binding, *, debug=False, coordinates=None)` 消费冻结的整 cfg dt/one-point 布局
- 两者用 `consume(index, value)` 接收一个已由 R 组装的主机结果，`publish()` 返回产物 ID；失败由调用方 `abort(error)` 清理
- `validate_cfg_output_binding(plan, binding, *, coordinates=None)` 在执行前核验显式物理时间、槽覆盖和唯一配对；不规划或读取数值
- `load_correlator_output(directory, configuration=cfg, path=(), *, dt=False)` 返回字段独立数组；收缩编排同名公开前端直接重导出此服务

`coordinates=None` 从 `plan.instances` 读取坐标，保持完整计划调用的现有行为。窗口执行可传入
完整轻量有序坐标序列；storage 直接校验、消费并记录这些坐标，不要求为窗口伪造计划实例。

`load_correlator_output(directory, configuration=cfg, path=(), *, dt=False)` 按 cfg 和具名字段读回。
HDF5 中保存各字段真实 shape/dtype 数值。reader 的批量访问、句柄复用和临时恢复细节属于实现与验收范围，
不在公共格式中固定物理文件数量。

完整 cfg 的逐点写入按字段轴切成不超过1 MiB的数值块，尾随扫描轴在块内长度为1。
不沿用基础数据“前两轴是时间”的默认分块假设，避免矩阵的每个元素各占一个HDF5块。
逐对、dt及packed字段共用这一物理存储策略；字段身份、坐标、dtype及数值不变，旧文件仍可读取。


`plan` 提供既有 CfgPlan 的字段表、实例坐标和容量；`binding` 提供实际有序
`physical_times/physical_lt`、lattice/ensemble/parents/provenance。storage 不重新搜索输出布局或容量。
P 的 `configure_cfg_output_layout(plan, layout, coordinates, *, host_cap)` 纯推导完整输出预算，
收缩编排只拆包 Calculation 和已声明 cap。字段路径由输出配方提供；档案版本与 HDF5 字段编码由各自契约维护。

完整 cfg 结果使用 HDF5 的具名数值字段保存，同时由数据集根登记 cfg 和字段定义。
dt 坐标、缺失值、单圈字段与字段路径仍遵守各自的科学配方；字段路径由当前 `OutputRecipe`/接口提供，
并以实现验收为准。

默认 CPU 写入流用 TS `TimePairAccumulator`，每字段只常驻 `(Lt,*field.shape)` 原 dtype 求和缓冲及共享 counts，
不构造 source×sink 数值矩阵、不提升其原生逐贡献求和精度。
逐点 dt 使用原 dtype 原位除法且始终检查已覆盖输出有限；packed dt 原 dtype 除法写入尾轴，
保留原 debug-only 有限性扫描策略。one-point 缓冲只保留每时间槽首次值及覆盖标记。

GPU 后处理可在设备传输中使用临时打包；HDF5 持久层保存真实字段 shape/dtype。
NaN/计数、时间坐标、分区累加与 one-point 首次值仍由配方和调用声明决定；所有权交接必须持续到同步写入完成。
是否同时保留逐时间对及 dt 聚合取决于已保存的输出配方，不以磁盘子目录协议表达。
协议替身的科学数值/传输检查分开，详见
[GPU 后处理](../results/gpu_output_postprocessing.md)。

`tests/test_output_architecture_alignment.py` 固定样本冻结两个 cfg 布局和 structured schema 的
全部 spec、具名 HDF5 字段的 shape/dtype/数值字节与预算；另覆盖 dt/one_point 子 write/publish 故障、原 dtype 差异、
NaN/counts 和无 source×sink 常驻值栈。该限定 CPU 证据不代表 GPU/MPI 或生产规模验收。

保存时的 `dt_coordinates` 可在源平均后选取有序的物理时间差子集；缺省仍保存完整 Lt。
计数仍按完整物理 Lt 登记，缺失策略仅检查选定的输出 dt；该选项要求 `average_tsrc=True`。
