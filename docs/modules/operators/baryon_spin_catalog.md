# 重子 64 行 gamma／自旋目录

`baryon_spin_preset(version="v1")` 完整覆盖有序三夸克 Dirac 自旋空间
`4 × 4 × 4 = 64`；每行是一个归一化三指标系数张量。
目录名为 `baryon-spin`，使用版本和零起始编号，采用 DeGrand–Rossi 基。
它与 [2031 行重子空间目录](elemental_catalogs.md)分别描述自旋和空间，可自由配套。

## 与介子相同的目录形式

| 用途 | 介子 | 重子自旋 |
|---|---|---|
| 编号表 | `meson55_preset(version=...)` | `baryon_spin_preset(version="v1")` |
| 冻结多重态 | `meson55_operators(version=...)` | `baryon_spin_operators(version="v1")` |
| 配方记录 | `GaugeOperator` | `BaryonSpinOperator` |
| 算符行 | `recipe.insertion(momentum_dict)[row]` | `recipe.insertion(momentum_dict, component=...)[row]` |

编号表保留 `id/name/irrep/parity/copy/row/terms`；自旋项使用
`spin=[alpha,beta,gamma]`、`coefficient=[real,imag]`，指标取 0..3。
每行记录 `catalog/version`；重排表不改变其编号。返回独立可编辑副本，
不会修改冻结目录。若要改变系数，作为自定义 `SparseSpinTensor` 使用，不能继续声称原目录编号。

不可变配方包含 `name/irrep/parity/version`、`row_ids`、`rows`、`copy_index`、
`exchange_irrep`、`exchange_component`、`partner`。`rows` 是不可变稀疏自旋张量；
`coefficients` 的轴为 `(立方群行, spin0, spin1, spin2)`。
共 24 个立方群多重态、64 行；普通使用选择配方，不手写投影矩阵。

```python
from lqcdweave.operators.single_hadron.api import (
    baryon_preset, baryon_spin_operators, Operator,
)

spatial = baryon_preset(version="v1").select(("local",))
recipes = {r.name: r for r in baryon_spin_operators(version="v1")}
recipe = recipes["spin_G1g_MA_copy0"]
insertion = recipe.insertion({0: (0, 0, 0)}, component=spatial[0]["component"])
operator = Operator(recipe.name, [insertion[0](0, 0, 0)], [1])
```

空间表用于相同的 `Elemental(...)` 自动准备入口；`operator` 用于普通 `Baryon` 绑定。
省略 `component` 默认使用 `baryon-vvv/v1` 的局域分量；默认动量仅零动量。
配套 reader 必须提供真实动量元数据。目录 CPU 回归见
[test_baryon_spin_catalog.py](../../../tests/correlator/operators/test_baryon_spin_catalog.py)。

## 双夸克低秩配方

`baryon_diquark_operators(version="v1")` 提供与原始收缩程序同型的 `(q0ᵀ CΓ q1) q2` 配方，
Γ 取全部 16 个 Clifford 元（DR 中 `C = gamma(2) gamma(8)`），自由夸克在槽 2。
按双夸克在空间转动下的性质分为两组，共 24 个配方、64 行，构成**完备正交归一**的另一组三自旋基：

| 双夸克（标签） | 交换性质 | 每种宇称的配方 | 每行结构 |
|---|---|---|---|
| C、Cγ5、Cγ4γ5（`C`、`Cg5`、`Cg4g5`） | 反对称 | `G1` | 一个双夸克矩阵 × 一个自由旋量（秩一） |
| Cγ4（`Cg4`） | 对称 | `G1` | 同上（秩一） |
| Cγᵢ、Cγᵢγ4、Cγᵢγ4γ5（`Cgi`、`Cgig4`、`Cgig4g5`） | 对称 | `G1`、`H` | 三个 i 分量之和（秩三） |
| Cγᵢγ5（`Cgig5`） | 反对称 | `G1`、`H` | 同上（秩三） |

名称为 `diquark_{标签}_{G1,H}{g,u}`，例如 `diquark_Cg5_G1g`、`diquark_Cgi_Hu`。
它们与 64 行目录同为 `BaryonSpinOperator`，用法相同：

```python
from lqcdweave.operators.single_hadron.api import baryon_diquark_operators

recipes = {r.name: r for r in baryon_diquark_operators(version="v1")}
recipe = recipes["diquark_Cg5_G1g"]
insertion = recipe.insertion({0: (0, 0, 0)}, component=spatial[0]["component"])
```

- 每个配方都是同表示、同宇称目录多重态的精确组合，`weights` 记录 `(多重态名, 系数)`；
  行处于目录的规范立方行基，可与目录配方混用、做空间耦合与移动系投影。
  全局相位取第一个非零权重为正实数。组合可跨越 S/MS 或 A/MA，因此没有单一交换标签
  （`exchange_irrep`、`exchange_component` 为 `None`）；对称双夸克只落在 S/MS，反对称只落在 A/MA。
- 在[行打包因子网络](../expansion/baryon-factor-network.md)中，标量双夸克是槽 2 上的秩一拆分，
  矢量双夸克是秩三拆分，均不形成稠密 `(R,4,4,4)` 常量。
- `diquark_Cg5_G1g` 的权重为 `A_copy0`、`MA_copy0`、`MA_copy2` 上的 `1/√6, 1/√2, 1/√3`，
  即 LL3/XiN 前端重建原程序 `G13 ⊗ e_r`（Dirac-Pauli 下的 Cγ5）所用的组合；
  G1u 的 `MA_copy0` 变号。原程序以 DP 分量 `r` 编号的行与规范行的对应及符号，仍由调用方的前端处理。

## 双立方群与交换群的联合分解

半整数自旋使用立方群的双覆盖，再加入反演，构成 `O_h^D`。
目录将它与 `S3` 同时约化，并保留重复表示的每一份。
每种宇称的分解如下，g/u 两份合计 64 行：

| 自旋交换类型 | 每种宇称的立方表示 | 每种宇称行数 |
|---|---|---:|
| S | `G1 + 2H` | 10 |
| A | `G1` | 2 |
| MS | `3G1 + H` | 10 |
| MA | `3G1 + H` | 10 |
| 合计 | `8G1 + 4H` | 32 |

S、A 是一维交换表示；MS、MA **共同组成一个二维表示**，不能当作两个一维表示。
在相同 `irrep/parity/copy_index` 下，通过 `partner` 互相定位，固定采用现有
`QuarkExchange("mixed")` 的行基：

```text
D(01) = [[1, 0], [0, -1]]
D(12) = [[-1/2, sqrt(3)/2], [sqrt(3)/2, 1/2]]
```

这里括号内为零起始腿槽，分别交换第 0/1 和第 1/2 条腿。
其余交换由群乘法确定。纯自旋 64 维空间没有 G2；与非平凡空间表示耦合后可以出现 G2。
`copy_index` 固定同一表示的重复份数和相位，不额外宣称唯一的连续角动量标签。
分类方法参考 [Basak 等的双立方群与 Dirac 交换分类](https://arxiv.org/abs/hep-lat/0508018)；
本目录的系数在项目 DR 基中由矩阵单位生成，行基和份数顺序以冻结文件为准，不冒充论文表号。

这些交换标签仅适用于**自旋系数**。完整重子还包含颜色 epsilon、空间和味道；
不能直接把这里的 S/A/MS/MA 填作完整系数的 `quark_symmetry`。
例如局域空间对称、味道对称时，颜色 epsilon 会把自旋 S 的完整系数变成 A；
混合表示也要组合所有因素后使用现有矩阵交换接口。
目录不会自动给完整重子加错误标签，也不会丢弃因具体味道/空间而禁止的自旋基向量。

## 空间耦合与移动系

`Insertion.from_irreps` 和 `InsertionGaugeLink` 的自旋参数既接受原有
`SpinName.G1G/G1U/HG/HU`，也接受任意 `BaryonSpinOperator` 配方：

```python
from lqcdweave.operators.single_hadron.api import Insertion

coupled = Insertion.from_irreps(
    recipe, [[(1, spatial[0]["component"])]], "A1g", "G1", {0: (0, 0, 0)},
)
```

非平凡空间行仍按[既有耦合契约](baryon_irrep_coupling.md)提供。
目标表示中的 `copy_index` 与自旋配方本身的份数分开，分别用于两次不同的约化。
冻结行已处于标准立方行基，`recipe.insertion(...).little_group_projection(...)`
可沿用现有移动系入口；非零动量必须在所传动量字典中明确给出。
组合自旋和空间后的交换表示，可用 `.project_exchange(...)` 自动投影，随后再组合 flavor；
方法与配套生产表见[组合交换投影](baryon_exchange_projection.md)。
源端仍由 `Baryon(source=True)` 完成系数复共轭及每腿 gamma4，不由目录重复处理。
旧 `local_spin` 的 12 行定义保持不变，已有数据和算符不会被改写。

## 构建与验证

运行期与介子目录一样只读打包 JSON，不依赖现场群投影。离线重新生成使用
`tools/build_baryon_spin64.py --output-dir <目录>`，复用包内置的 `row_subduction`
和 `QuarkExchange`，不维护第二套立方群或交换群表。
投影只在离线去除舍入零；用户提供的自旋张量仍保持原有精确零规则。
双夸克配方由 `tools/build_baryon_diquark.py --output-dir <目录>` 从冻结目录离线生成
`presets/baryon_diquark_v1.json`；
[test_baryon_diquark_catalog.py](../../../tests/correlator/operators/test_baryon_diquark_catalog.py)
独立验证 C 的定义、双夸克对称性与 16 个 CΓ 的迹正交、目录组合与 64 行完备正交、
"双夸克 × 自由旋量"结构与宇称及秩一/秩三拆分、与原程序 Cγ5 组合一致、冻结文件重建，
以及空间耦合和移动系接口。

`tests/correlator/operators/test_baryon_spin_catalog.py` 验证 64×64 完备正交、
独立特征标计数、全部 48 个正双群旋转及宇称、全部 6 个交换、MS/MA 相对相位、
冻结目录重建、64 行源汇数值、空间耦合全部目标份数和移动系交接。
这些是 CPU 系数与数值验证，不代表 GPU 或真实系综验收。
