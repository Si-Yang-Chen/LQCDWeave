# Elemental 前端：数据、算符表与时间

本页记录已实现的表格前端；小规模 CPU 数值与示例已验证。GPU 和真实系综规模须另行验收。

## 介子

```python
generator = ElementalGenerator(
    gauge_field, eigenvectors,
    momentum_dict=momentum_dict, operator_table=operator_table,
    Ne=120,
    blocks=("v2v",),
)
result = generator.calc_path_blocks(t)
```

- 初始化提供全部数据、配方和执行选项；计算方法只接收 `t`。
- `momentum_dict` 与 correlator 的 `Insertion` / `InsertionGaugeLink` 共用：值接受整数 XYZ 三元组或三整数字符串，按插入顺序对应动量轴，key 不作为槽号。`generator.momenta` 是规范化后的有序 XYZ tuple，保存时用它记录动量身份。详见[配套动量表](../elemental/elemental_presets.md)。
- 原有第三位置或 `momenta=` 的有序列表调用保留；第三位置也可直接传同一字典。`momenta` 与 `momentum_dict` 只能选一个，动量表和 `operator_table` 均必填。空表、重复 XYZ、非三维或非整数坐标报错。
- `Ne` 可省略，继承输入本征矢的模态数；显式值选择实际有序前缀，不改变基底。
- 格点尺寸从已加载规范场 `(t,z,y,x,direction,color_row,color_column)` 推导。本征矢布局为 `(t,mode,z,y,x,color)`。
- 文件读取属于脚本或适配器。已加载数组可直接计算；延迟读取句柄须提供 `lattice_size` 并调用 `load(configuration)`，不从路径名称猜测数据身份。

### 点端扇区

```python
generator = ElementalGenerator(
    gauge_field, eigenvectors, momenta, operator_table,
    blocks=("v2v", "v2p", "p2v", "p2p"),
    points=point_set,
)
result = generator.calc_path_blocks(t)
```

| 扇区 | 所需数据 | 一个时间片的输出 |
|---|---|---|
| `v2v` | 本征矢 | `(component,momentum,mode_left,mode_right)` |
| `v2p` | 本征矢、点集 | `(component,momentum,mode_left,point_right,color_right)` |
| `p2v` | 点集、本征矢 | `(component,momentum,point_left,color_left,mode_right)` |
| `p2p` | 点集 | `[component][momentum]` 的 SparsePointBlock |

`blocks` 默认只选 `v2v`，可以选择上述任意非空子集。仅 P/P 时 `eigenvectors=None`。统一使用一个 `points`，无 `points_right`；两端在各自实际时间读取同一具名点集。数组点集布局为 `(time,point,xyz)`，也可传 SpatialPointSet。时间路径的右端仍为真实相邻时间，不能用锚点时间替代。

### V/D 分开的向量扇区

已有 medium dilution 向量时，通过 `vector_bases={"V": v_handle, "D": d_handle}` 显式导入；第二位置 `eigenvectors` 仍是 V 的兼容入口。V、D 使用相同的向量数值核，保留不同实际基底身份和轴大小。`blocks` 支持九个扇区：

| 左端／右端 | V | D | P |
|---|---|---|---|
| V | `v2v` | `v2d` | `v2p` |
| D | `d2v` | `d2d` | `d2p` |
| P | `p2v` | `p2d` | `p2p` |

D handle 的 `dilution_source` 保存实际 D、父基底、有序分组与时间随机池。生产与 `calculate` 仍用同一个 `elemental_weighted` 选项控制 D 和 P 的局部权重及整图余权，不增设 dilution 权重开关。当前保存语义为内部 component_inclusion-v1，旧已计权D须从raw重新生产；来源绑定、整组选择和同池相等分区范围见[详细使用说明](../../../.agents/skills/lqcdweave-elemental/references/dilution.md)，D 计权 CPU 回归见[test_dilution_correlator.py](../../../tests/test_dilution_correlator.py)。重子和矩阵专用入口的 D 支持没有随此扩展。

### 空间点采样计权

默认 `elemental_weighted=False` 保留原始输出。对当前支持的均匀无放回空间点抽样，可在生产和消费时分别使用同名的一个布尔选项：`ElementalGenerator(..., elemental_weighted=True)` 将单个 current 的局部逆包含权重乘入点端块，`calculate(..., elemental_weighted=True)` 读取该标记并计算复杂图剩余的采样系数。D/P 混合块还使用同一选项乘入 D 混合列包含补偿；同组 D/D 对角乘 H1=n/m，非对角乘 H2=n(n-1)/(m(m-1))。已证明完全对角时只需 H1。没有额外的系数表需要用户填写。

当 `blocks` 含点端扇区时，计权输入必须是带 `design` 和显式 `pool_ids` 的 `SpatialPointSet`，或其 `.points` 满足同一契约的 `SpatialPointHandle`；带 cfg 的 handle 用 `generator.load(cfg)` 显式加载后再计算。导入的裸坐标没有这些元数据，不能用于 `True`。`sample_spatial_points` 会为每次随机生成及其实际时间分段提供池 ID。保存和读回必须保留实际点池身份：显式点集保留其 `pool_ids`，登记的本地采样数据用真实 `data_id` 与设计分段标识 realization；不能由 seed、shape 或相同坐标重建同池关系。若从已有数组或文件适配器读取已计权值，用 `elemental_weighted=True` 声明数据已计权，避免生产端重复乘权。系数、数据标记和多 current 图的限制见[生产模块说明](../elemental/elemental_production.md#optional-per-elemental-point-compensation)与[展开模块说明](../expansion/expansion.md#elemental-local-compensation-and-graph-residual)。取样权重 CPU 回归见[test_sampling_weights.py](../../../tests/production/elemental/test_sampling_weights.py)。

### Elemental 结构

结构声明仅有三种：`general`（一般）、`diagonal`（完全对角）和 `point_diagonal`（P 对角）。V/D 的完整基底指标是向量编号；P 的完整基底指标是点编号加颜色编号。因此 P 对角只保证左右点编号相等，颜色矩阵仍可非对角。自旋与 Gamma 矩阵另行处理，不包含在这些基底指标内。单位矩阵和零矩阵都属于完全对角的数值情况，不增加结构类别；程序不扫描数据里的零值来猜结构。

`ElementalGenerator` 根据配方和可核实的输入证明，自动提供非一般项的 `generator.elemental_structures`，键为 `(sector, component, momentum_slot)`，值为上述结构字符串。P/P 空间单位路径可证明完全对角，动量槽仍可非零；闭合 Wilson 路径只证明 P 对角。V/V 和 D/D 只有空路径、零动量且输入基底有明确正交证明时才证明完全对角。登记的 `DatasetFieldHandle` 从实际基底的 `conventions.normalization="orthonormal"` 或 `conventions.basis_orthogonal=True` 取得证明；高级自定义句柄可显式声明 `basis_orthogonal=True`，由调用者保证其真实性。裸数组不提供正交证明。NV 正交声明只沿已证明的 parent 传播；传统单向量组即使范数不为 1 也属于对角结构。QR 只在同一 realization、同一时间内传播正交性；不会从 V/D 两套实际基底之间的重叠自动推断。

P 对角支持要求该分量的每个配方项端点空间位移都按各空间格长取模为零；时间位移还需是同一实际时间，或点输入显式共享同一有序基底。登记 P/P convention 还要求左右端绑定同一实际点数据集和相同有序点标签。生产配方不新增前端开关。

导入自定义数组时，可在 `ArrayElemental(...)`、`from_npy(...)` 或 `from_binary(...)` 传 `elemental_structures={(sector, component, momentum_slot): structure}`。旧 `point_diagonal_components` 仍作为兼容入口接受，推荐新数据改用三态结构。登记数据在各 sector 自身的 `conventions.elemental_structures` 中保存 `{component: row_slot, momentum: momentum_slot, structure: ...}` 项；旧 `conventions.point_diagonal_components` 读回映射为 P 对角。缺少声明的旧数据按一般处理。左右使用的有序基底/点选择不一致时，选中视图会降为一般结构。

普通 dense Meson Diagram lowering 消费完全对角的 V/V、D/D 或 P/P 块时，只准备对角系数与 Gamma 的组合，并分别合并 mode 指标或点加颜色指标；所有对角系数都会保留。P 对角仍保留颜色矩阵收缩。矩阵专用核及混合重子/介子消费保留原 dense 布局。稀疏 P/P 继续沿既有 COO 桥接路线消费结构证明，不表示 GPU 稀疏收缩已新增。

### 导数

```python
generator = ElementalGenerator(
    gauge_field, eigenvectors, momenta, derivative_operator_table,
    Ne=120,
)
result = generator.calc_derivative(t)
```

初始化参数仍叫 `operator_table`，没有另一套 `derivative_table` 参数。表格格式相同，但方法决定操作序列的意义：路径方法按有序规范链解释；导数方法按有序空间协变导数解释，方向 0/1/2 为 x/y/z。导数保留现有双侧左右作用、次序和符号，不把它静默改成单侧差分；每步是 forward−backward，不自动除以 `2a`。导数目前只输出 V/V。同一张数值序列表在两个入口下可能表示不同算符，不应混用。

## 算符表

任务脚本读取表格，计算类只消费记录，不自行打开 CSV/JSON 文件。

| 字段 | 含义 |
|---|---|
| `component` | 输出分量标签；相同标签的项累加 |
| `path` | 有序操作序列；介子为一条，重子为三条 |
| `coefficient` | 复系数；可用复数或 `[real,imag]` 保存 |

```python
operator_table = [
    {"component": "local", "path": [], "coefficient": 1},
    {"component": "difference", "path": [0], "coefficient": 1},
    {"component": "difference", "path": [3], "coefficient": -1},
]
```

不再分开传 paths 和 coefficients。输出按 component 首次出现顺序排列，标签通过 `generator.components` 核对；重复路径项按系数累加，零系数不改变定义。非有限系数、空表或缺字段明确拒绝。

已有 55 行 catalog 的 `id/terms` 形式也属于同一张表：每项仍是 path/coefficient，额外物理标签保留在任务清单。可直接传 `catalog["rows"]`，无需另建路径索引或稠密矩阵。meson55 v2 有 901 条不同路径、最大长度 5；旧 CSNS 表为 175 条、最大长度 3。目标输出是 55 个组合核，Gamma 与味道耦合在后续算符层处理。

路径方向采用 v2：0..7 为 +x,+y,+z,−x,−y,−z,+t,−t。只计算表中明确列出的操作，不自动枚举路径或导数阶数。

## 重子

```python
generator = BaryonElementalGenerator(
    gauge_field, eigenvectors,
    momentum_dict=momentum_dict, operator_table=operator_table,
    Ne=120,
)
result = generator.calc_path_blocks(t)
```

默认三腿共用 `eigenvectors`；只有确需不同实际基底时才传 `(vectors0,vectors1,vectors2)`。`Ne` 可省略并分别继承三腿模态数，也可用整数统一截断或 `(N0,N1,N2)`。每个表格项的 path 是三条腿的有序路径；相关三元组先做 epsilon 缩并再按系数求和，不把三腿路径和分别相乘。当前只有 V/V/V，无 blocks、点集或重子导数选项。结果为 `{"vvv": array}`，轴 `(component,momentum,N0,N1,N2)`。

距离平均应先展开到这张路径及系数表：非零距离每腿的六条直线系数为 1/6，零距离为空路径；三腿组合系数相乘，不另设计算入口。

## 显式稀疏时间

不传 selection 时沿用完整时间。明确稀疏化时，同 cfg 的基底和点数组按已保存选择记录紧凑排列：

```python
selection = load_time_selection(selection_file, Lt=gauge.shape[0])
generator = ElementalGenerator(
    gauge, compact_vectors, momenta, operator_table,
    selection=selection,
)
for slot, (t, result) in enumerate(generator.iter_slices()):
    stored[:, :, slot] = result["v2v"]
```

`selection=None` 是默认值。`calc_path_blocks(t)` / `calc_derivative(t)` 中的 t 始终是物理时间，必须在锚点选择内；`iter_slices()` 按记录原顺序返回 `(physical_t, borrowed_result)`。生成器 `Lt` 是物理长度，`Nt` 是输出槽数；规范场不截断，输出不扩回 Lt 或补零。直接数组的默认输入映射沿用 selection；SpatialPointSet 使用自己的 `times`，已有句柄使用 `time_selection`，无该元数据的句柄仍要求完整按序时间。

进阶输入可用 `eigenvector_selection=None` 与介子的 `point_selection=None` 分别声明额外覆盖或不同顺序。重子的 eigenvector_selection 接受一个共同记录或三腿记录三元组；不按最短输入截断。使用全时间原始数组计算稀疏锚点时，显式给完整输入记录覆盖默认映射。句柄已有元数据与显式输入记录不一致时拒绝；流式 `read_modes(key,t,start,stop)` 的 t 仍是物理时间，适配器负责映射到存储槽。

时间路径使用物理 `(t + path.displacement[3]) % Lt` 端点，再分别查找基底与点输入槽。例如选择 `(7,2,5)` 的下一个槽不是 t+1。跨时间流常需额外端点；用输入记录提供这些实际片，缺片在输出交付前明确报错。不会自动补算基底或把缺片当零；几何开放边界既有零路径配方保持不变，只有这类几何上不存在的路径不要求端点输入；有效路径缺片仍报错。

同一时间坐标映射不证明实际基底或点池一致；保存原有身份、端点、模式/点标签及顺序。消费侧的时间轴是 Nt 槽，按该 reader 的 slot API 读取，不再把生产端物理 t 直接当数组下标。

## 执行与保存

常规调用无需填写后端、块大小、相位和权重。需要时这些可选项均放在初始化：array_module、ne_block_size、phase_convention、path_boundary、dilution/is_blending；重子另有 spatial_block_size、cache_bytes、stream_eigenvectors 和 vector_scratch_limit_bytes。SU(3) 投影和 stout 是显式预处理，不暗中改变已有基底或 Dirac 规范场版本。

默认相位沿用已核实约定：介子 midpoint，重子 anchor；显式选项保留，不能以少参数为由舍弃计算自由度。mode blending 是模态对权重，不能用普通路径系数代替。

计算默认返回按扇区标记的字典。借用缓冲在下一次计算前同步保存或复制。完整时间由脚本循环；默认覆盖全部时间，用户明确选择子集才稀疏化。实际基底身份、规范场版本、时间与动量顺序、表格及归一化随产物保存，沿用现有 storage/runtime 的发布和恢复。

可选 write 回调在初始化固定；它只交付数据，不决定文件保存。是否降低输出内存取决于内核是否在分配完整组合结果之前分块交付。一个时间片的 55×33×120×120 complex128 输出约 398.8 MiB；先生成完整结果再回调不会减少这一部分。各分支的实际分块范围随后端验收记录说明，不能由 write 的存在推断全面内存上限。

默认 V/V 空间路径表沿用融合投影内核，直接形成组合行，不分配完整原始路径输出。其它扇区或含时间路径的组合先计算原始路径块，需另计这些临时结果。介子 write 在完整组合之后交付；重子 write 沿用三腿分块回调。

输入所有权：规范场和点坐标在加载时复制；介子本征矢保留借用引用，计算期间须保持有效且不修改。重子非流式读取形成有序前缀快照，三腿共用基底只复制一份；流式读取每次同步消费模态块。

## 验证与适用范围

公开前端由 `tests/production/elemental/test_table_frontend.py` 对照独立逐点矩阵及行列式；稀疏时间由同目录 `test_sparse_time.py` 核对有序槽、四扇区、跨时间正负端点、不同输入顺序、缺片和重子流式物理时间读取；历史接口迁移或平台故障记录不作为当前 API 限制。

规范场仍完整常驻；介子基底借用已有输入，重子 eager 快照为各实际输入 Nt，流式输入按现有模态块暂存。数值、输入、输出、缓存和整体内存分别核对。GPU、MPI、真实系综和生产规模需要各自验收，小规模 CPU 参考不能替代。

## 配套的预制表

动量与规范链接预制表都保留；按[从选表到调用](../elemental/elemental_presets.md)选择需要的行，
同一 `gauge_link_tables` 输出直接供生产和 `InsertionGaugeLink` 使用。读者验证完整
路径/系数组合及所选物理动量，不依靠整数行号碰巧一致。固定动量 key `3` 是 27 项立方表。
