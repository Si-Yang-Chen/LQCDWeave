# 流的空间数据与真实端点

只有关联函数涉及流插入才加载。先明确局域流还是守恒/有向流、方向、动量、锚点时间及时间边界。

局域流使用相应局域空间配方，Gamma与Flavor由Operator处理。守恒流保留全部组成路径及真实端点，时间方向包含t±1；流组合系数由ConservedVectorOperator或指定Operator给出，不在空间数据再乘一遍。

有向流复用统一路径入口。构造 ElementalGenerator 时显式设置：

```python
operator_table = [dict(component=d, path=(d,), coefficient=1) for d in range(8)]
generator = ElementalGenerator(
    gauge_reader, basis_reader, momenta, operator_table, lattice_size=lattice,
    path_boundary=boundary, phase_convention="midpoint",
    Ne=Ne,
).load(configuration)
row = generator.calc_path_blocks(time)["v2v"]
```

row 是借用的 `(direction,momentum,bar_mode,field_mode)` 数组，下一次调用前保存或复制。boundary 由任务选择 periodic/open；时间方向使用物理 t±1 实际基底，开放越界端点按既有配方为零。稀疏输入需要显式覆盖全部所需物理端点，否则在交付前报错；不能把相邻槽当相邻时间。中点相位须显式配置。`iter_slices()` 按初始化选择惰性逐片调用 `calc_path_blocks(t)`；默认完整时间，不堆叠输出。需要旧 raw 布局时逐片复制并沿 time 轴拼接，再调用 contracts.current.build_current_raw_contract；布局契约不能替代实际基底身份。严格守恒流不隐式施加 mode blending。

八方向顺序为 `+x,+y,+z,-x,-y,-z,+t,-t`。后端、动量、Ne、实际时间和规范场变换按生成器接口选择。需要保存或登记产品时按[保存与持久交接](production.md)组合数值接口与通用服务；`plan_indexed_directed_current` 只规划，脚本显式调用 `calc_path_blocks(t)`，再用 `DatasetWriter.write(slot, value)` 保存对应实际时间行。

多步路径使用ElementalGenerator的 operator_table 中的显式 GaugePath，时间/空间方向含义沿用已有配方，不从二维空间数组推断跨时间端点。

Elemental 结构只有三种：`general`（一般）、`diagonal`（完全对角）和 `point_diagonal`（P 对角）。完整 V/D 基底指标是 mode 编号；P 的完整指标为 point 加 color。P 对角只保证点相等，颜色矩阵可非对角；自旋/Gamma 另行处理。单位与零矩阵属于完全对角数值情况，不增设结构类型，也不扫描实际数组的零值猜结构。

`ElementalGenerator` 自动给出非一般的 `generator.elemental_structures[(sector, component, momentum_slot)]`。P/P 空间单位路径证明完全对角（可在非零动量槽），闭合 Wilson 路径证明 P 对角。V/V 和 D/D 只在空路径、零动量且有实际正交证明时声明完全对角。`DatasetFieldHandle` 从真实 basis spec 的 `conventions.normalization="orthonormal"` 或 `conventions.basis_orthogonal=True` 读取证明；高级自定义句柄可声明 `basis_orthogonal=True`，调用者负责真实性。裸数组不证明正交。NV 只从已证明的 parent 传播正交性；legacy 单向量 group 即使范数不为 1 也为对角。QR 声明只在同 realization、同一时间内传播；不同实际 V/D 基底之间不推断 overlap。P 对角要求每个配方项的空间端点位移按各空间格长取模为零；时间位移需在同一实际时间或使用显式共享的有序点基底。生产不会增加结构开关。

自定义数据可在 `ArrayElemental(...)`、`from_npy(...)`、`from_binary(...)` 传 `elemental_structures={(sector, component, momentum_slot): structure}`，component 使用实际身份。旧 `point_diagonal_components` 仍兼容且映射到 P 对角；旧 `conventions.point_diagonal_components` 也如此读回。新 Dataset 在自身 sector 的 `conventions.elemental_structures` 保存 `{component: row_slot, momentum: momentum_slot, structure: ...}` 项。没有结构标签（包括只有数值 identity kind 的 legacy record）按一般处理；不会从 values、shape、filename 或 identity record 类型升级为完全对角。左右有序 mode/point selection 不一致时，当前视图降为一般。

由 `gen_correlator` 生成的普通稠密介子关联函数在后端展开时 可对 fully diagonal V/V、D/D 或 P/P 只准备对角系数乘 Gamma，并合并 mode 或 point+color 索引；系数不丢。P 对角仍保留 color matrix 收缩。矩阵专用核与混合 baryon/meson 消费保留 dense 布局。稀疏 P/P 继续走现有 COO bridge；不表示新增 GPU sparse kernel。已计权 sampling 使用结构证明合并 P/P 等点腿时，仍须实际 point-pool ID 和 ordered selection 一致。

流含实际点端时读[点端数据](point-flow.md)，包括确定点集和随机取样。需要 medium D 流时用[V/D 接线](dilution.md)，与点采样共享 elemental_weighted；纯模态流无需点端参数；只有任务选择随机取样时才引入取样策略与估计权重。

## 空间点采样计权

当前 V/P 点抽样的局部逆包含系数可由 `ElementalGenerator(..., points=point_set, elemental_weighted=True)` 在生产时乘入；消费侧用 `calculate(..., elemental_weighted=True)` 处理多 current 图中的剩余系数。若 `blocks` 含点端扇区，`point_set` 必须是有显式 `design`、`pool_ids` 的 `SpatialPointSet`，也可由其 `.points` 满足契约的 `SpatialPointHandle` 提供；带 cfg 的 handle 用 `generator.load(cfg)` 显式加载。数组形状或 seed 不提供池身份。保存读回必须保留实际 realization 的池身份。扇区公式及已有数据适配器的声明方式见[Elemental 计权参考](../../../../docs/modules/operators/elemental_frontend.md#空间点采样计权)；复杂图约定见 correlator skill 的[取样计权](../../lqcdweave-correlator/references/sampling-current.md#elemental-weighted)。
