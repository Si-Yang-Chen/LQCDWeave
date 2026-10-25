# meson55 预制算符与数据绑定

默认 meson55 是 `v2` 表。它修正了普通 G 波的 A1g 行 16 与 T2g 行 22–24，满足逐半径低于 L=4 的
投影为零、目标 L=4 非零；该条件不意味着严格纯连续 L=4，更高分波仍可存在。默认绑定从目录符号选算符，
不要求用户额外选择版本。历史 CSNS 表仍原样保留，只有明确使用 `version="csns"` 才会读取；
这四行在两版中的路径身份不同，旧表数据不能冒充 v2。定义与逐行核验见[低分波投影检查](#低分波投影检查)。

常规流程从冻结目录中选一个符号算符，再交给 `RecipeVertex` 预制绑定处理已登记的 elemental。
它会应用目录中的 Dirac 结构、空间行和耦合系数，并生成 `Meson` 顶点；测量脚本不需要
手写 Gamma、路径、CG 系数、component、动量槽或 dagger 规则。

`meson55_preset()` 默认返回 v2 的 55 行 `operator_table`，包括普通 L≤4 的 25 行、磁型 L≤2 的
27 行和电型 S 轨道候选的 3 行。`momentum_recipe("mom4").momenta` 保持原有 33 个 `p²≤4` 动量顺序，按
`(p², px, py, pz)` 排序，零动量在槽 0。

`meson55_preset(*, version="v2")`、`meson55_operators(*, version="v2")` 和
`RecipeVertex(elemental, version="v2")` 均默认使用修正版；`version="csns"` 显式选原历史表。
其它版本会报错，不会静默回退。v2 与 CSNS 的 55 行和 368 个多重态/880 个表示行数量相同，
但只有空间行 16、22–24 的路径身份改变。若要保留这四行的历史算符，显式设 `version="csns"`；
若要使用修正 G 核，则用默认 v2 并重新生产/补充对应 elemental。其余 51 行的路径系数不变；现在仍按目录版本匹配，不自动跨版本复用。

## 用户需要提供什么

| 项目 | 用户提供或选择 | 预制方案负责 |
|---|---|---|
| 已有数据 | elemental、传播子位置及 cfg | 按保存的完整 component 身份匹配空间数据 |
| 测量通道 | 目录算符名称、Flavor、动量、表示行及符号时间标签 | 从目录生成符号所需的 Dirac×空间耦合顶点 |
| 移动算符 | 非零动量的小群 irrep；需要时选择 `copy_index` | 小群投影及对应表示行 |
| 数值计算 | 实际时间坐标、Ne、后端及任务要求的资源设置 | 源端 dagger、相位约定及动量轴匹配 |

每个 elemental 数据源只需声明一次 `RecipeVertex(elemental)`，计算时传 `vertex_map=mesons`。
新增目录算符只改变符号选择，不需要逐个填写绑定字典、Gamma、路径或 CG 系数。
`propagator_map` 仍对应所选味道产生的 Wick 图标签。

生产表与测量目录的职责不同：`meson55_preset()` 给出 55 个空间行，供 elemental 生产；
`meson55_operators()` 默认给出 v2 的 368 个耦合算符多重态、共 880 个表示行，供测量选择。

历史三算符矩阵示例在同一静止系 A1 负宇称、
带电 `ud` 通道选择局域 `gamma5`、`gamma4gamma5` 和导数耦合算符。它只修改目录名称列表，
由同一个 `RecipeVertex` 绑定器构造完整 3×3 矩阵，以 `structured=True` 保留结构，
用 `anytime` 规划并与逐项符号展开的 `none` 路线核对。输入是固定种子的合成复数数组，
仅用于演示接口和数值一致性，不能用作真实谱学结果或生产性能基准。

## 从登记 elemental 构造 correlator

```python
from lqcdweave import preset
from lqcdweave.contraction.binding.api import RecipeVertex
from lqcdweave.operators.correlator.api import (
    Flavor, Hadron, Tag, gen_correlator,
)
from lqcdweave.contraction.correlator import calculate

elemental = preset.Dataset(elemental_dir)
perambulator = preset.Dataset(perambulator_dir)
mesons = RecipeVertex(elemental)
pion = Hadron(
    mesons.symbol("op_00_gamma5_A1", tag=Tag(0, 0))[0],
    Flavor("ud"),
)
correlator = gen_correlator(
    [[pion], [pion]], [0, 1], [False, True], structured=True,
)
sink_coordinates = range(Lt)
source_coordinates = range(Lt)
result = calculate(
    correlator,
    vertex_map=mesons,
    propagator_map={"S^q": perambulator, r"S^q_\mathrm{local}": perambulator},
    configuration=cfg,
    time_axes={0: sink_coordinates, 1: source_coordinates},
    usedNe=Ne,
)
```

这里的 `op_00_gamma5_A1` 是测量者显式选择的目录名称示例；绑定器不会替用户决定物理通道。

`vertex_map=mesons` 把符号图的每个 `HadronIrrepRow` 交给预制绑定；`gen_correlator` 标出的
源端 dagger 由绑定器传给 `Meson(source=True)`。`elemental_dir` 和 `perambulator_dir` 应指向
已有登记数据，`configuration`、`Lt` 和 `Ne` 使用该数据的实际组态、时间覆盖和模态数。示例中的
两条传播子标签来自 `"ud"` 味结构生成的 Wick 图；改变味结构时按图中实际标签提供映射。

这里的 `sink_coordinates`、`source_coordinates` 是与 reader 实际坐标域匹配的扫描序列：完整时间数据
可设为 `range(Lt)`；紧凑 Dataset 需用 `range(Nt)` 这样的槽坐标，并由登记 reader 保留实际
`selection.times` 和 `physicalLt`。不要把物理 `Lt` 直接当作紧凑轴长度；详见[时间稀疏化交接](../temporal_sampling/temporal_production.md)。

`RecipeVertex` 默认动量字典是 `momentum_recipe("mom4").momenta`。字典值顺序必须与 elemental 的存储动量轴完全相同；
若生产时选择了不同顺序或子集，将同一 `momentum_dict` 传给 `RecipeVertex(elemental, momentum_dict=...)`。
预制绑定采用 midpoint 相位和默认周期时间端点约定。空间路径按周期边界生产；
绑定消费已保存的路径定义，不改变或代替生产时的空间边界约定。

目录中的 `c_factor` 和 `reflection` 是预制空间与 Dirac 结构的 C 因子及线性时间反射符号，
不是 flavor-independent 的完整物理 C 量子数；实际味道结构（特别是带电 `ud` 通道）仍决定物理
C。两者是可查询的目录 metadata，不会作为额外系数乘进缩并。源端 dagger 沿
`HadronIrrepRow.dagger → Meson(source=True)` 传递并使用绑定模块既有源端约定。

## 名称、目录查询与移动动量

`meson55_operators()` 返回 368 个不可变 `GaugeOperator` multiplet，合计 880 个表示行。
名称为 `op_{firstkernelid:02d}_{gamma_name}_{restirrep}`，例如
`op_00_gamma5_A1` 和 `op_00_gammai_T1`。可按奇偶性、irrep、Dirac 结构或 `kernel_origin`
筛选；空间来源类型在 `kernel_origin[0]`，原始空间目录编号在 `kernel_ids` 中，实际 reader 槽由完整
component 身份解析。这个低层目录主要用于
枚举谱学基与检查定义，常规绑定直接用 `RecipeVertex.symbol(...)`。

静止系可从目录名称直接生成符号。非零动量必须显式指定小群 irrep；需要同一 irrep 的多个副本时
再指定 `copy_index`：

```python
moving = mesons.symbol(
    "op_00_gammai_T1", (0, 0, 1), irrep="A1", tag=Tag(0, 0),
)
```

当前实现会用小群投影构造移动算符，并检查所选动量、irrep 和副本。两个冻结耦合目录都定义静止系基底；
移动投影由小群接口产生，CPU 测试覆盖的方向和表示范围见下文，不能据此宣称远端实现已独立核验。

## 生产端空间表

```python
from lqcdweave.operators.single_hadron.api import (
    meson55_preset, momentum_recipe,
)
from lqcdweave.production.elemental.api import ElementalGenerator

table = meson55_preset()  # 默认 v2
generator = ElementalGenerator(
    gauge, eigenvectors, operator_table=table, momenta=momentum_recipe("mom4").momenta,
    phase_convention="midpoint",
)
for t, blocks in generator.iter_slices():
    save_slice(t, blocks["v2v"].copy())  # component,momentum,mode_left,mode_right
```

`gauge`、`eigenvectors` 为同 cfg 的实际规范场和有序基底，数组布局及惰性输入用法见
[elemental 前端](elemental_frontend.md)。每次取表返回独立副本，不会重新生成、重新投影或重新归一化。
该方案只选择空间算符，不隐式选择涂抹、Ne、时间或后端。复现原 CSNS 表时显式用
`meson55_preset(version="csns")`。复现已检查的 CSNS 基础数据时，
还需使用其实际 20 步、ρ=0.12 的空间 stout 规范场、120 个实际模态及原有 40 个有序时间；
这些参数不由 `meson55_preset()` 自动施加。

完整表的 `id=0..54` 对应 component 槽。每行的 `component` 是
`ElementalComponent.catalog("meson55", version, id)`，引用固定版本目录中的分量；`kind`、`target_l`、
`family`、`irrep`、`copy`、`row` 等字段保留选取来源。空间系数已经在生产中施加，
调用保存的投影数据时不要重复乘一次。

低层用户若要自定义目录之外的空间或 Dirac 配方，可用 `InsertionTerm` / `Insertion.from_rows`
显式组装 component 行；参见[算符模块中的高级接口](operators.md#配套预制表与物理动量检查)。
该接口适用于自定义配方，常用 meson55 multiplet 不需要手工重建。

裸数组 reader 使用 `generator.components` 与 `generator.momenta`，并明确前三轴布局及 midpoint
相位；登记数据用已有 `Dataset` 入口读取。生产接线见[elemental 前端](elemental_frontend.md)，
correlator 接线见[目录绑定参考](../../../.agents/skills/lqcdweave-correlator/references/binding.md#从-meson55-目录绑定预制算符)，保存格式见
[配套预制表](../elemental/elemental_presets.md)。

## 保存定义

`component` 是内存中的科学身份对象；保存 `elemental_projected` 的 JSON 定义时保留原始
catalog 行，去掉这个派生字段，并保存一次目录及选中编号：

```python
from lqcdweave.contracts.elemental_catalog import catalog_definition

catalog = catalog_definition(table)
rows = [{key: value for key, value in record.items() if key != "component"}
        for record in table]
component_coordinates = [{"projection": record["id"]} for record in rows]
momenta = [list(momentum) for momentum in generator.momenta]
```

完整 55 行保存为 `definition.rows=rows`，component 轴使用 `component_coordinates`，
momentum 轴使用 `momenta`，同时登记实际相位、时间、输入身份和基底。
保存 `definition.catalog=catalog`，`DatasetElemental` 按目录标识、版本和编号恢复相同引用。
选子集或重排行后，存储槽由实际表顺序决定；若保存为 `elemental_projected`，需将保存定义的
`id` 重编为连续 `0..N-1` 并同步坐标，不能将原始编号当作新的数组槽。

直接用 `ElementalGenerator` 生产投影行、没有保存 raw-path 数组时，登记
`elemental_projected` 的 `definition.method="direct_calc_projected"`。除完整 `rows` 外，
保存 `input_sha256` 中实际规范场和有序基底文件的 `gauge`、`eigenvector` SHA256，
`gauge_source_sha256` 与前者一致，以及 `gauge_transform={"steps":20,"rho":0.12,"dimensions":"spatial"}`
这样的实际涂抹记录和 `path_boundary`。这里的 20、0.12 仅复现上述 CSNS 数据，其他任务记录实际选择。
模式轴的 `bases` 绑定实际基底 `data_id` 与有序 indices，并将这些 ID 保留在 `parents`；
不要为未生产的 raw-path 数据填写 `raw_id` 或 `input_views.raw`。通过同一个
`preset.Dataset(directory)` / `RecipeVertex` 入口读取，component 按保存的目录标识、版本和编号恢复。
若从已存 raw-path 产物线性组合，则继续使用 `linear_raw_path_combination` 及其真实 raw 父项和输入视图。
紧凑时间轴保存槽号，并用 `conventions.time_coordinates` 记录实际有序时间与物理 Lt。

## 为什么选这 55 行

普通部分对每个目标轨道分波的每个立方表示保留一个独立代表及其全部行。
有限格距的 L 是夸克对轨道构造标签，不是完整介子的 J，也不保证纯连续 L。

| component 槽 | 类型 / 目标 L | 路径族与空间表示 |
|---|---|---|
| 0 | 普通 S | `n0_000 / A1g+`，局域 |
| 1–3 | 普通 P | `n3_210 / T1u-` |
| 4–8 | 普通 D | `n3_210 / Eg+`；`n2_110 / T2g+` |
| 9–15 | 普通 F | `n3_111 / A2u-`；`n3_210 / T1u-, T2u-` |
| 16 | 普通 G | v2 修正 A1g 核，端点位于 `r²=9` 的 `(3,0,0)` 与 `(2,2,1)` 轨道 |
| 17–21 | 普通 G | 继承历史表的 Eg、T1g 核 |
| 22–24 | 普通 G | v2 修正 T2g 核，端点位于 `r²=9` 的 `(2,2,1)` 轨道 |
| 25–27 | 磁型 S | `B_S / T1g-` |
| 28–36 | 磁型 P | `B_P / A1u+, Eu+, T1u+, T2u+` |
| 37–45 | 磁型 D-T₂ | `B_D_T2 / A2g-, Eg-, T1g-, T2g-` |
| 46–51 | 磁型 D-E | `B_D_E / T1g-, T2g-` |
| 52–54 | 电型 S 候选 | `E_S_axis / T1u-` |

普通核先对同端点的最短路径取平均，再作立方群投影及同族副本组合。
P、D-Eg、F-T₂u 选择 `(2,1,0)` 端点族以降低第一个更高分波的几何投影；
D-T₂g 用 `(1,1,0)`，F-A₂u 用 `(1,1,1)`。v2 的 G-A₁g 与 G-T₂g 使用下述新核；
其余 51 行保持原空间路径系数；目录引用仍按版本区分。

### v2 普通 G 波核

行 16 的 A1g 核使用 `r²=9` 的端点轨道 `O(3,0,0) ∪ O(2,2,1)`。每个轴向端点的总权重为
`4/√120`，每个 `(2,2,1)` 型端点为 `−1/√120`。行 22–24 的三个 T2g 行只使用
`O(2,2,1)`，对端点 `(x,y,z)` 的权重依次为
`yz(x²−2)/√384`、`zx(y²−2)/√384`、`xy(z²−2)/√384`。
每个端点的总权重均分给通往该端点的全部最短路径；`(2,2,1)` 型端点各有 30 条此类路径。
四行在每个半径壳都消除所有 `l<4` 投影并保留非零 `l=4`；它们不是严格纯连续 G 波，允许存在更高分波。

磁型 27 行与电型候选 3 行保持原定义。v2 全表共有 901 条不同路径，最大路径长度为 5；
历史 CSNS 表共有 175 条不同路径，最大路径长度为 3。

磁型核用同端点三链的反序差及轴向 staple 差，保留 S/P/D 各轨道来源的独立副本。
电型候选用横向对称 staple 平均减直链，再对正负轴向端点作差；
它只有空间路径，不是显式局域 F₄ᵢ 插入；磁型与电型行不随 v2 的普通 G 波修正而改变。

冻结来源是 `lattice-scattering` 的
[方案生成器](https://github.com/Si-Yang-Chen/lattice-scattering/blob/c226bb1c22010683c9745c082c0c25ba3e1b3c09/scripts/spectroscopy/build_l4_magnetic_l2_electric_s_plan.py)
及[选取说明](https://github.com/Si-Yang-Chen/lattice-scattering/blob/c226bb1c22010683c9745c082c0c25ba3e1b3c09/docs/spectroscopy/l4-magnetic-l2-electric-s-plan.md)。
原始来源（`tools/presets_raw/meson55.json`、`meson55_v2.json`、
`meson55_bindings.json`、`meson55_bindings_v2.json`）与已检查的 CSNS 四个 cfg 逐行一致；
它们的 SHA256 记录在库内文档的 `sources` 字段中：

- 包内文档：`presets/gauge/meson55_v2.json`、`presets/gauge/meson55_csns.json`，
  自包含 `catalog`（55 行）与 `multiplets`（368 个耦合），并带 `integrity` 摘要。
- 历史 CSNS JSON 的 SHA256 为
  `1043d30f227ac19daf054e2aa4757a6e74de5b923d3e60607f1c1f03e217b9af`；
  v2 catalog 为 `b7bec45f0dd2eb924f9c8632cdcb94e5a8ba9c60deb05078f094862c668af256`；
  v2 binding 为 `9da81b528318dcb7379c8d30bd577cbc2464ba92fe3af29c8045a338a26c5fc3`；
  历史 binding 为 `2ab829bc107b93dd012110c96caeb372bd5944acaf7d8026fcb636017067bad0`。
- 校验命令：`python tools/build_recipe_documents.py --check`（文档摘要与原始来源一致）。
CSNS 参考实现 SHA `0dd9622...` 记录的是原耦合算法约定；v2 不宣称其四个修正行是 CSNS 已生产数据。
离线重建工具依赖 NumPy 与包内置的格点群表示模块，可将输出写入独立目录：

```sh
python tools/build_meson55_revised.py --output-dir /path/to/rebuilt-presets
```

运行库读取冻结文档时不调用重建算法或增加额外依赖。重建资源的 byte-level 复现由
`tools/build_recipe_documents.py --check` 对照 `sources` 摘要检查；其余 CPU 检查不代表新的 GPU/MPI 或生产规模验收。

## 低分波投影检查

对目标轨道分波 L，每个表示行都必须满足所有 `l < L` 的投影为零，并且目标 L 投影非零。
允许存在更高分波。检查对象是实际路径系数，不是目录中的 `target_l`、`leading_l` 或旧 `Q` 标签。

普通核先从路径步序列计算端点，将同端点的复系数相加得到 `w(r)`。每个半径独立检查
`a_lm(R) = Σ_{|r|=R} w(r) Y_lm*(r̂)`。不能将不同半径的角向系数相消当成径向分波消失。
局域原点项单独按 S 处理，不为原点虚构球面方向。

审计工具用球谐加法定理计算非负强度
`S_l(R) = 4π Σ_m |a_lm(R)|² = (2l+1) w† P_l(r̂·r̂′) w`，
独立测试再用球谐振幅直接求和比对。公式见 [NIST DLMF 14.30.9](https://dlmf.nist.gov/14.30.E9)。
以下比值是角向强度比，不是物理态混合概率。
数值零阈值为每壳 `1e-10 × Σ_r |w(r)|²`；未达到目标非零条件的空核不会通过。

| 版本 / 普通行 ID | 目标 L / 表示 | 实测最低 l | 低分波强度 / 目标强度 | 结论 |
|---|---|---:|---:|---|
| v2：16、22–24 | 4 / A1g、T2g | 4 | 每壳低于目标分波为零 | 普通核几何检查通过 |
| v2：其余 21 行 | S、P、D、F；G 的 Eg、T1g | 等于目标 L | 舍入误差范围内为零 | 普通核几何检查通过 |
| csns：16 | 4 / A1g | 0 | 100/21 ≈ 4.76190 | 历史核含 S，不满足条件 |
| csns：22–24 | 4 / T2g | 2 | 3/4 | 历史核含 D，不满足条件 |

历史 csns 的上述表示各只有一个副本；副本旋转或重新归一化不能消除其低分波。
v2 改变了四行的路径身份，因此旧表和已生产 elemental 的身份必须保留，不能沿用旧行号假装兼容。
使用 v2 时应重新生产或补充这四行；其余 51 行的空间定义不变，目录引用仍须匹配对应版本。

磁型与电型核含同端点的路径形状差。把它们当成普通标量端点权重，会令路径差相消，
可能错误地把所有投影都记为零。因此下面工具将这些行明确标为未支持，不能计入分波验收通过。
其轨道与规范场分量的分解须另行检查。

```sh
python tools/check_meson55_partial_waves.py --output /path/to/meson55-angular-audit.json
```

默认命令检查 v2：报告 25 行通过、0 行失败、30 行未支持，并以状态码 2 退出；状态码 2 表示
磁型/电型尚未由该标量端点审计支持。若需核查历史表，可加
`--catalog tools/presets_raw/meson55.json`；它报告 21 行通过、
4 行失败、30 行未支持，状态码 1。审计只检查几何分波，不代表真实 QCD 态、GPU/MPI 或生产规模验收。
检查与独立参考见 `tools/check_meson55_partial_waves.py` 自身的几何判据（对应专项测试已随目录迁移删除）。

## 耦合定义与独立核验

基底耦合参考脚本是 CSNS 冻结前端文件
`/dg_workfs/LQCD/chensiyang/codex_jobs/full-io-production-dispatch-mpi-cfg-20261007/external/meson55/meson55_basis.py`，
参考 SHA256 为 `0dd9622aa2704d68a5749dcc13beb5c8a821ff43b20014535be5f7f1b5d6c655`。
历史 CSNS 绑定资源（`tools/presets_raw/meson55_bindings.json`）的 SHA256 为
`2ab829bc107b93dd012110c96caeb372bd5944acaf7d8026fcb636017067bad0`；其中记录的空间 catalog SHA256 为
`1043d30f227ac19daf054e2aa4757a6e74de5b923d3e60607f1c1f03e217b9af`。本地执行
`build_basis` 用时 0.693 秒，生成 368 个 multiplet 和 880 行。

耦合从空间 catalog 系数恢复 24 个 proper rotations。标量结构使用 A1，矢量结构使用 T1，
按 Clifford 分量与空间核的直积表示构造目标 irrep。Reynolds embedding 为
`U=(1/24) Σ_R D_product(R) seed D_target(R)†`；seed 按行优先 `(i,j)` 枚举，选择首个非零结果，
并归一到 `U†U=I`、首个非零系数为正实数。目标表示顺序固定为 `A1,A2,E,T1,T2`，同一表示的
副本沿原目录次序保留。

冻结项保留八种 Dirac 名称及对应的 masks、phases、宇称、`c_factor` 和 `reflection`。
本地 CPU 核验检查 368 个 multiplet 的 24 rotation carrier、PC metadata 和 16×55 维耦合正交性；
耦合正交检查本身不验证低分波消除；普通行的几何检查由上节工具独立完成，磁型与电型仍未获该标量审计认证。
完整 Wick trace 覆盖静止系 γ₅、γ₄γ₅、γᵢ 的 structured/unstructured 流程，移动动量检查覆盖
A1/E 投影、方向、dagger 和 XYZ 一致性。移动小群示例由当前实现验证；冻结耦合文件定义静止系基底，
上述结果不代表 GPU、MPI、远端生产数据或生产规模验收。

共同目录契约与重子目录见[版本化目录](elemental_catalogs.md)。
