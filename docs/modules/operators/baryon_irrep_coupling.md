# 重子自旋与空间表示的自动耦合

重子沿用介子的调用方式：**自旋表示 × 空间规范核表示 → 目标表示**，然后用
`insertion[row](../../px, py, pz)` 选择行和动量，并交给同一个 `Operator`。
构造算符只计算系数，不读取或生产 elemental。

## 接口

```text
Insertion.from_irreps(
    spin, spatial_rows, spatial_irrep, projection, momentum_dict,
    *, copy_index=0, spatial_matrices=None,
)
```

- `spin`：`SpinName.G1G`、`G1U`、`HG`、`HU`，值为 `G1g`、`G1u`、`Hg`、`Hu`。
  使用归一化的 DeGrand–Rossi 三夸克自旋张量；g/u 是三腿 gamma4 的宇称。
  也可传入 `baryon_spin_operators(version="v1")` 的任意配方，覆盖完整 64 维空间的
  全部交换类型、份数及规范立方行基；见[自旋目录](baryon_spin_catalog.md)。
- `spatial_rows`：按空间表示的行排序，每行是 `(复系数, BaryonComponent)` 的序列。
  `BaryonComponent` 可以指向原始三腿路径，也可以指向已经投影的空间 elemental。
  腿槽 0/1/2 保持原顺序。
- `spatial_irrep`：`A1g/u`、`A2g/u`、`Eg/u`、`T1g/u`、`T2g/u`，允许下划线。
- `projection`：`ProjectionName.G1`、`G2`、`H`。也可写带 g/u 的目标名，宇称必须
  等于自旋和空间宇称的乘积。重子没有这里的 C 本征值，`charge_conjugation=None`。
- `momentum_dict`：非空动量字典，约定同介子。动量选择不会自动进行小群投影。
- `copy_index`：默认 0。同一目标表示出现多次时选择第几份；越界或不存在时报错。
  份数编号不对应唯一的连续总角动量。
- `spatial_matrices`：默认采用内置格点群实现的标准空间行基。若空间行使用另一行基，
  显式传入该基下的旋转矩阵字典，键是标准的 48 个正旋转双群操作名。
  必须与实际空间行相符；接口检查维数、群乘法、特征标及投影一致性，不从 component ID
  猜测其波函数。非正交行基须先整理为正交行基。

自动耦合需要 Python 3.11+ 和项目的 `[correlator]` 可选依赖。
自定义的三自旋行仍可用 `Insertion.from_rows` 显式构造。

## 各表示的乘积分解

- `G1 × A1 → G1`
- `G1 × A2 → G2`
- `G1 × E → H`
- `G1 × T1 → G1 + H`
- `G1 × T2 → G2 + H`
- `H × A1 → H`
- `H × A2 → H`
- `H × E → G1 + G2 + H`
- `H × T1 → G1 + G2 + 2H`
- `H × T2 → G1 + G2 + 2H`

G1、G2、H 分别有 2、2、4 行。表示不自动指定唯一的连续总角动量、轨道分波或味道。
自旋、空间和味道的三夸克置换对称性仍须按目标通道配合，避免 Pauli 禁止的算符。

spin×spatial 完成立方耦合后，可继续调用
`insertion.project_exchange("A", operator_table=table)`，或以 `"mixed"` 得到 MS/MA 伙伴。
它实际投影组合系数并给出配套的 `operator_table`，随后再在 Hadron 层组合 flavor。
完整接口、S2 槽位及颜色约定见[组合交换投影](baryon_exchange_projection.md)。

## 与介子相同的规范核字典入口

```python
from math import sqrt
from lqcdweave.operators.single_hadron.api import (
    BaryonComponent, InsertionGaugeLink, Operator, ProjectionName, SpinName,
)

# 只在第 0 条腿做奇宇称位移；三行依次为 x、y、z。
spatial_rows = [
    [(1 / sqrt(2), BaryonComponent.paths(((axis,), (), ()))),
     (-1 / sqrt(2), BaryonComponent.paths(((axis + 3,), (), ())))]
    for axis in range(3)
]
insertion_dict = {"T1u": [spatial_rows]}
insertion = InsertionGaugeLink(
    SpinName.G1G, "T1u", 0, ProjectionName.H,
    {0: (0, 0, 0)}, insertion_dict,
)
operator = Operator("baryon_Hu_row0", [insertion[0](0, 0, 0)], [1])
```

这里 `gauge_link_idx=0` 选择空间配方，`copy_index` 选择耦合的目标表示份数。
字典键也可沿用带 `+/-` 后缀的介子式名字；重子只取空间表示及 g/u，后缀不赋予重子 C。
三腿不接受介子的二腿 `profile` 参数；径向模态分块见[重子模式分块](baryon_mode_blocks.md)。
规范核系数只施加一次：若 elemental 已存整行空间投影，使用对应的 projected component，
并在这一行中用系数 1 引用它。源端共轭和三腿 gamma4 由现有 Baryon 绑定处理。

## 系数、归一化与小群

- 程序从实际三自旋张量求出自旋表示矩阵，保留原 DR 基和腿槽。
- 使用内置 `row_subduction` 的矩阵单位：
  `P(r,0) = d/|G| Σ_g D_target(g)[r,0]* U_product(g)`。
  星号表示复共轭。行 0 的投影列按原坐标顺序正交化，再用矩阵单位生成其他行，固定份数和相位。
- `insertion.coupling_coefficients` 的形状是 `(n_spin*n_spatial, n_target)`，
  产品坐标先自旋行、后空间行；列正交归一。它是只读数组，`multiplicity` 给出总份数。
  这不会改变用户空间波函数的归一化，也不添加体积因子。
- 需要非零动量小群时调用
  `moving = insertion.little_group_projection(momentum, irrep_name, idx=0)`。
  自动耦合的重子行按目标表示的标准行基作投影，返回一个新 `Insertion`，保留原始静止行。
  任意 `from_rows` 或局域自旋行没有标准行基标签，不能据此自动投影。

空间路径如何计算和保存见[重子 elemental](../elemental/baryon_production.md)；已有数据和按需数据均
可用于同一算符，后者见[按需重子 elemental](../elemental/baryon_on_demand.md)。

## 验证

- 5 项新增测试覆盖全部自旋与空间表示、宇称组合、重复份数、旋转协变、
  自定义行基和非零动量小群投影。算符模块共 19 项测试通过。
- 隔离 CPU 全量回归运行 1576 项；相对 1571 项基线无新增失败或错误。
  两者均有原有的 3 个失败、9 个错误及 24 个跳过，不能据此声称全量通过。
- 同机 CPU ASV（Ne=64，两种复数精度，各 3 轮）未检出显著变化。
  测量覆盖既有缩并核，不包含新增耦合初始化和数据读取。
- 小规模 CPU 参考不证明 GPU、真实组态或生产规模通过。
