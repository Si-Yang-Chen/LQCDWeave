# 具体插值算符与流

量子数描述测量通道；实际数据对应哪种插值配方仍需明确。模板已给定常用配方，普通使用者只选择配方与参数。

`lqcdweave.operators.single_hadron.api` 中的 `Insertion` 和 `Operator` 是底层配方定义；常用预制方案优先使用对应绑定。RecipeVertex 通过 `RecipeVertex.symbol(...)` 生成符号行，再由 `RecipeVertex` callable 自动构造对应的 insertion/operator，不需要手写 Dirac×空间项。重子从 `baryon_spin_operators(version="v1")` 选择已联合投影的自旋配方，调用 `recipe.insertion(momentum_dict, component=...)` 后交给同一 `Operator/Baryon`；需要空间表示耦合时把 recipe 传给现有 `Insertion.from_irreps` 或 `InsertionGaugeLink`。显式自定义行仍用 `Insertion.from_rows`，不添加第二种重子 calculator。味道在 `Flavor` 层表达。

重子自旋目录名为 `baryon-spin`，`baryon_spin_preset(version="v1")` 提供 64 个编号行，`baryon_spin_operators` 提供 24 个不可变立方多重态。按 `irrep/parity/exchange_component/copy_index` 选择；`partner` 关联 MS/MA，二者共同组成 S3 二维表示。仅自旋的交换标签不能直接作为含颜色、空间及味道的完整重子标签。完整格式、计数、空间耦合与示例见[64 行自旋目录](../../../../docs/modules/operators/baryon_spin_catalog.md)。旧 `local_spin` 的 12 行保持原定义。与原始收缩程序同型的双夸克 × 自由夸克配方用 `baryon_diquark_operators(version="v1")`：标量双夸克 `diquark_{C,Cg5,Cg4,Cg4g5}_G1{g,u}`，矢量双夸克 `diquark_{Cgi,Cgig5,Cgig4,Cgig4g5}_{G1,H}{g,u}`，共 24 个、64 行完备正交；同为 `BaryonSpinOperator`，是目录同表示多重态的精确组合，在因子网络中秩一/秩三拆分。

与介子相同，先用 `Insertion.from_irreps` 或 `InsertionGaugeLink` 组合 spin×spatial 到目标立方表示，再通过 `combined.project_exchange("A", operator_table=table)` 作组合交换投影，最后在 Hadron 层组合 flavor。`"mixed"` 返回相位对齐的 MS/MA 两个 Insertion，`copy_index` 选择独立交换份数，`slots=(0,1)` 等选择 S2。将结果的 `operator_table` 交给同一 `Elemental` 自动准备，不能继续假定原空间子集覆盖投影后的所有腿位置。接口及颜色 epsilon 约定见[组合交换投影](../../../../docs/modules/operators/baryon_exchange_projection.md)。


Elemental 生产可将同一份 `momentum_dict` 传给 `Insertion` 或 `InsertionGaugeLink`；字典值的插入顺序对应生产与算符的动量槽，key 不作为槽号。普通 `Insertion` 默认 `momentum_identity=False`；需要绑定时核验实际 XYZ 与 reader 动量顺序时，可设 `momentum_identity=True`。`InsertionGaugeLink` 默认 `momentum_identity=True`。

用户选择 meson55 时，默认使用 `contraction.binding.api.meson55`：从 v2 `meson55_operators()` 的冻结 recipe 名称构造符号算符，再将该 callable 绑定器传入 `vertex_map`。它应用 Dirac masks/phases、component 行和对应版本的冻结耦合，并依源端符号 dagger 构造 `Meson(source=True)`；用户不手写 Gamma、路径、CG、component、动量槽或源 dagger。

`meson55_operators()` 返回默认 v2 的 368 个不可变 multiplet、共 880 行，可按 parity、irrep、gamma_name 和 `kernel_origin[0]` 筛选；`version="csns"` 显式读取历史目录。recipe 的 `c_factor` / `reflection` 是空间×Dirac metadata；实际物理 C 仍依赖 `Flavor`，特别是带电 `ud`，而 metadata 不会额外乘进缩并。`momentum_recipe("mom4").momenta` 的顺序必须与实际 elemental 动量轴相同。完整定义、移动动量边界及独立核验见[meson55](../../../../docs/modules/operators/meson55.md)。

多算符矩阵用同一绑定器和目录名称列表构造 `basis`，再调用 `gen_correlator([basis, list(basis)], [0, 1], [False, True], structured=True)`，将完整矩阵交给一次 `calculate`。目录选择与动量约定见[meson55 使用说明](../../../../docs/modules/operators/meson55.md)；联合矩阵与符号参考的 CPU 对照见[structured 回归](../../../../tests/test_structured_correlator.py)。合成输入不用于物理结果或生产性能结论。

高分波选择按[逐壳低分波检查](../../../../docs/modules/operators/meson55.md#低分波投影检查)核对；群投影非零不能代替低分波消除。绑定时保留完整 component 版本，历史目录与修正目录不自动混用。

仅在用户要求目录以外的自定义配方时，才从 `meson55_preset()` 取 `component` 并用 `InsertionTerm` / `Insertion.from_rows` 手工构造耦合行。空间系数已在 elemental 生产端施加，调用系数不能重复包含它。

预制守恒流是 `ConservedVectorOperator`（Operator子类），其组成项包含真实相邻端点。选择方向、Wilson r及流动量后，数据必须覆盖这些端点。局域流与守恒流不是同一数据需求。等价手写Operator只在用户明确要自定义配方时展开。


未定义的粒子或流先确认所需插值配方，不从表示名推导未经确认的自旋系数。用户不需要接触原子缩并或内部张量描述。
