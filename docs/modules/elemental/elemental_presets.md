# 预制动量与规范链接表：从生产到调用

两张表一起使用：动量表规定有序 `(px,py,pz)` 轴，规范链接表规定每一行实际路径和复系数。
分量以目录标识、版本和目录内编号引用；数组槽只表示存储位置。生产、保存与调用共享同一目录。

默认 `meson55_preset()` 返回修正版 v2 的 55 行表；配套 33 动量顺序仍为 `momentum_recipe("mom4").momenta`。
历史 CSNS 表可用 `meson55_preset(version="csns")` 显式选择。完整脚本写法、component 对应及
版本边界见[meson55 预制算符与数据绑定](../operators/meson55.md)。它与下面按最大链长取全表的入口独立。

已生产并登记 meson55 elemental 后，常规 correlator 直接用
`oontraotion.binding.api.meson55` 按目录名称选择算符。它将空间核、Dirao 结构和冻结耦合一起绑定；
动量顺序和详细用法见[meson55 流程](../operators/meson55.md#从登记-elemental-构造-correlator)。本页 `InsertionGaugeLink`
示例针对下面的通用规范链接表，属于自定义路径表时使用的低层接口。

## 选择方案

| 用途 | 生产表 | correlator 消费入口 |
|---|---|---|
| 使用 meson55 目录算符 | `meson55_preset()` 默认 v2，55 个空间行 | `RecipeVertex(elemental)` 默认 v2，按目录名称选择符号 |
| 使用通用规范链接表 | `gauge_link_preset(0/1/2/3)` 经 `gauge_link_tables` 转换 | 配套 `insertion_dict` 与 `InsertionGaugeLink` |

meson55 不需要额外填写 `insertion_dict`。两种方案均保存生产时的实际动量顺序；
meson55 默认配套 `momentum_recipe("mom4").momenta`，通用链接表独立选择动量字典。
要消费历史 CSNS 的四个 G 行时用 `RecipeVertex(..., version="csns")`；v2 修正 G 行需重新生产或补充对应 elemental，其余 51 行的路径系数保持不变；分量仍按目录版本匹配，不自动跨版本复用。
下面第 2–4 步是通用规范链接表流程，meson55 的生产、保存与绑定统一见[专用流程](../operators/meson55.md)。

## 1. 选动量表

```python
from lqcdweave.operators.single_hadron.api import (
    momDict_mom3,
)
momentum_dict = momDict_mom3
```

这是固定预制表，不是任意的 `p²` 上限生成器：`1` 是单个零动量，`2` 是
`p²<=2` 的 19 个动量，顺序与 `momDict_mom3` 前 19 项完全一致；`3` 是
`{-1,0,1}³` 的 27 个动量，`9` 是 123 个动量，`0` 是 23 项测试表。
`momDict_mom2` 是该 19 项表；`momDict_mom1` 仍只包含零动量，不表示 `p²<=1`。
保留表中原有顺序。只需子集或重排时，两端共用同一字典：

```python
momentum_dict = {19: momDict_mom3[19], 13: momDict_mom3[13]}
```

不能生产时重排，调用时仍使用未重排的原始字典。

Elemental 的介子、重子前端与 correlator 的 `Insertion` / `InsertionGaugeLink`
直接共用这一份 `momentum_dict`。值可以是整数三元组，也可以是 `"0 0 1"`
形式的字符串；按字典值的插入顺序对应动量轴，key 不作为存储索引，也不排序。
上例 key=19 对应槽 0、key=13 对应槽 1。空表、重复 XYZ 或非整数三元组拒绝使用。
`mom_dict_to_list(momentum_dict)` 可取得相同顺序的列表；原有 `mom_dict_to_list(3)` 等
预制表编号调用及 elemental 的 `momenta` 列表参数继续可用。两种参数不能同时提供。

## 2. 选规范链接表，并取得配套调用表

```python
from lqcdweave.operators.single_hadron.api import (
    gauge_link_preset, gauge_link_tables,
)
preset = gauge_link_preset(1)
selected = {
    "T_1u-": preset["T_1u-"],
    "A_1g+": [preset["A_1g+"][0]],
}
operator_table, insertion_dict = gauge_link_tables(selected)
```

`gauge_link_preset(max_length=1)` 支持固定长度 `0/1/2/3`，分别有 `1/7/37/187`
个投影行。结构是 `irrep → occurrence → row → {path, coefficient}`；路径用
`0,1,2,3,4,5 = +x,+y,+z,-x,-y,-z` 的显式步序列，系数为 `[real, imag]`。
这些是包含不超过指定长度、无相邻立即折返空间路径的预制立方群行，不包含时间路径。
每次调用返回独立副本。保留给定行基、符号和归一化，不自动重新归一。

`gauge_link_tables` 延续“生产行 + 调用重映射”机制，输出直接用于现有
`operator_table` 和 `InsertionGaugeLink`。生产后的分量引用 `gauge-link/v1` 目录及稳定编号，完整路径与系数保留在目录中；调用表的系数为 1，不会重复乘入已在生产端组合的系数。
可选取、重排 irrep 或 occurrence；重复的完全相同行只生产一次。不得把未知整数 ID
当作路径；导入其他编号方案时必须给出实际步序列。

## 3. 生产、保存

先按实际登记的规范场、基底和同一有序表构造 `elemental_projected` 的 `spec`，
轴为 `(time,component,momentum,mode_left,mode_right)`，定义保留实际路径、系数、动量、相位和父身份。
相关规划与行写入见[生产组合](elemental_persistence.md)；下面只展示数值数组与保存格式。

```python
import numpy as np
from lqcdweave.production.elemental.api import ElementalGenerator
from lqcdweave.storage.dataset import DatasetWriter

generator = ElementalGenerator(
    gauge, eigenvectors, momentum_dict=momentum_dict, operator_table=operator_table,
)
momenta = generator.momenta  # 保存定义和动量轴使用这一有序 XYZ tuple
values = np.stack([
    generator.calc_path_blocks(t)["v2v"].copy()
    for t in generator.times
], axis=0)  # time,component,momentum,mode_left,mode_right
with DatasetWriter("elemental", spec, provenance=provenance) as writer:
    for slot in writer.missing:
        writer.write(slot, values[slot])
    writer.publish()
```

这是小规模整数组示例；`.copy()` 保证下次计算不覆盖已保存的借用缓冲。
每 cfg 保存一份完整 NPY，共同科学定义及 cfg 信息由 `elemental/dataset.json` 登记。
显式时间稀疏化使用同一 `selection` 构造 spec 和 generator，不能从数组长度猜测物理时间。

需要持久身份时使用已有 `DatasetSpec/DatasetWriter`。已有原始路径产物可用
`plan_indexed_elemental_projection` 做只读规划，`rows` 参数直接传保留目录的 `operator_table`；脚本读取 raw 数值、调用 `re_combine`，再用 `DatasetWriter.write` 保存。
保存产物 `elemental_projected` 保留 `definition.rows`、实际 raw parent/input view、
有序 momentum 轴及相位。`DatasetElemental`（或 `preset.Dataset`）从这些元数据重建
同一目录引用，保存 `definition.catalog` 后按目录编号解析；不根据存储槽猜路径。缺少行定义或父身份时拒绝读取。

## 4. 高级：手工调用通用规范链接表

常规 meson55 使用不需要手工写 `InsertionGaugeLink`。下面保留通用规范链接表的显式入口：
它适合自定义的 `gauge_link_tables` 行定义；meson55 的完整 Dirac×空间耦合走
[`RecipeVertex` 预制绑定](../operators/meson55.md#从登记-elemental-构造-correlator)。

```python
from lqcdweave.preset import Dataset
from lqcdweave.operators.single_hadron.api import (
    InsertionGaugeLink, GammaName, ProjectionName, Operator,
)
from lqcdweave.contraction.binding.single_hadron import Meson

source = Dataset("elemental").load("cfg")
ins = InsertionGaugeLink(
    GammaName.A0, "T_1u-", 0, ProjectionName.T1,
    momentum_dict, insertion_dict,
)
operator = Operator("x-link", [ins[0](*momenta[0])], [1j])
meson = Meson(source, operator, phase_convention="midpoint").load("cfg")
value = meson.get(0)
```

`InsertionGaugeLink` 默认 `momentum_identity=True`，保留物理动量三元组。
绑定时逐项检查读者该位置的 `momenta`；缺少、重排或不一致都会报错，不静默改位置。
显式老式位置调用可选 `momentum_identity=False`，这时只检查位置覆盖，不能证明物理动量一致。
普通导数 `Insertion` 保留旧位置默认；需要同样检查时传 `momentum_identity=True`。
`OperatorTerm(..., momentum_vector=(px,py,pz))` 可为手写项声明同样的检查。

算符仍使用原有 cyclic Gamma 到 bitmask 的一次符号转换；预制表不会改变相位、Gamma、
源端共轭、路径方向或归一化。换了行的路径/系数，即使复用了行号也无法匹配；相位不一致
同样报错。CPU 小规模端到端覆盖见 `tests/test_elemental_preset_matching.py`；GPU、MPI、
真实系综与生产规模不由此证明。

介子与重子的共同目录格式、重子 v1 的可选行及版本迁移见[版本化目录](../operators/elemental_catalogs.md)。
