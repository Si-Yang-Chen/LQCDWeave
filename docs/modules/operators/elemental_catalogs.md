# Elemental 版本化目录

介子和重子的预制分量统一引用 **目录标识、版本、分量编号**。
完整空间路径、复系数、腿顺序和归一化由目录规定，不再构造包含完整路径系数的
`spatial_projection` 分量身份。存储槽与目录编号分开；选取子集或重排不改变目录编号。

`meson55_preset(version="v2")` 和 `gauge_link_tables(...)` 返回的生产表均保留目录。
使用 `table.select((number0, number1))` 或切片选择，生产和算符引用同一表的 component。
meson55 的历史 `csns` 与 `v2` 为两个目录版本，即使某行系数相同也不自动跨版本匹配。
历史无目录记录的显式自定义投影可作为内嵌自定义目录读取；不会从行数或 shape 猜测预制版本。

## 重子 v1 目录

```python
from lqcdweave.operators.single_hadron.api import baryon_preset

table = baryon_preset(version="v1")
selected = table.select(("local", "distance/0,0,1", "s3/0,1,2/mixed0/MS"))
component = selected[0]["component"]
```

目录标识为 `baryon-vvv`。路径编号 0 是局域，1–6 分别为 +x、+y、+z、−x、−y、−z 单链。
目录列出可选配方，不替测量选择动量、实际基底、Ne、自旋、物理味道或涂抹参数。

| 名称格式 | 行数 | 内容 |
|---|---:|---|
| `local`、`paths/i,j,k` | 343 | 七个局域/单链路径的全部有序三腿组合 |
| `distance/d0,d1,d2` | 8 | 每条腿距离为 0 或 1，非零距离独立平均六个方向 |
| `s3/i,j,k/S`、`A`、`mixed0/MS`、`mixed0/MA`、`mixed1/MS`、`mixed1/MA` | 504 | 无序路径种子的 S3 投影，包含二维混合表示的两个出现次数 |
| `s2/ab/i,j,k/S`、`A` | 1176 | 全部三种交换腿位置的 S2 投影 |

共 2031 行。单腿空间更长路径及其它线性组合仍可通过显式目录配方定义，不能修改现有
`v1` 的行定义后继续使用原版本。目录数据通过共同 `ElementalCatalog` 契约保存和核对。

这里的交换特征描述含颜色 ε 的**空间三腿张量**，要求被交换腿使用相同的实际有序基底。
完整重子的交换标签还要结合自旋和味道；不能自动把空间标签当成完整系数的标签。
S3 投影使用标准矩阵单位：系数为 `sign(p)*D(p)[copy,component]/sqrt(3)`，
S/A 的系数分别为 `sign(p)/sqrt(6)`、`1/sqrt(6)`。S2 系数为 `(1,-s)/sqrt(2)`，
其中 s=+1 对应空间 S、s=−1 对应空间 A。重复路径保留这些既定归一化和相消，
不逐行重新归一化。数值验证见 `tests/test_baryon_catalog.py` 的独立空间行列式和交换矩阵检查。

## 保存与自定义目录

目录定义保存一次，选中分量只保存编号。登记投影产物时把
`contracts.elemental_catalog.catalog_definition(table)` 保存为 `definition.catalog`；
数值行的 `definition.rows` 和存储 component 坐标继续按实际存储槽连续排列。
已有 raw-to-projected 规划入口直接传保留目录的 table，会自动携带该记录。
读回时核对选中编号的定义，不由数组位置推断目录编号。

重子独立登记时把同一 table 传给 `plan_indexed_baryon_elementals(...,
product="projected", components=table)`，自动保存目录和选中编号；数值配方记录在
`definition.projected_rows`。`DatasetBaryonElemental` 读回后恢复相同的目录 component。
常规测量直接通过[受管 Elemental](../correlator/managed_elemental.md)自动准备，无需手工调用这些保存步骤。

显式自定义目录使用 `ElementalCatalog.from_rows(name, version, rows, rank=2或3)`，
再通过 `.table()` 给出配套生产表。目录名和版本属于配方声明，不能用同一名字和版本
承载不同定义。构造时检查表和目录一致，正常数值读取不逐次重复检查完整目录。
