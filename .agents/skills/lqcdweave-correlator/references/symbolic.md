# 粒子、投影与时间

关联函数的唯一构造入口是 `gen_correlator`：先声明物理算符、味道和源汇共轭，再由它生成 Wick 收缩。不要直接构造内部图对象、手写夸克连线或用低层拓扑替代物理算符声明。生成后的关联函数可交给公开的贡献选择、投影和分区接口。

用于构造任务的符号测量。目标或投影未定时读[候选与选择](measurement-choice.md)；已有明确测量直接构造，模板只作参考。

```python
from lqcdweave.operators.correlator.api import (
    Tag, Flavor, HadronIrrep, Hadron,
    hadron_little_group_projection, gen_correlator,
)

pi = HadronIrrep("pi", [0, 0, 0], "A_1", -1, Tag(0, 0))[0]
channel = Hadron(pi, Flavor("ud"))
correlator = gen_correlator([[channel], [channel]], [0, 1], [False, True])
```

`HadronIrrep` 参数依次是名称、整数动量、表示、宇称、粒子槽位/符号时间；`[row]` 选择表示行。味波函数使用Flavor的和、积和系数，沿用任务中明确的夸克/反夸克约定。

多粒子将各单粒子表示传给 `hadron_little_group_projection`，选择总表示、行和宇称，然后与味波函数组成Hadron。再交给同一 `gen_correlator`，不单独手写 Wick 接线。

`gen_correlator(halves, times, dagger, degenerate=True)` 将各通道组合成矩阵；False保留非简并味道标签。非负整数或非空字符串时间键把整个组合放到该符号时间；None保留组合中粒子的个别时间，带跨时间流的三点尤其需要这一点。输入列表会被更新，重复构建使用新列表。同名图遵守既有共享/合并语义。

已有重子三点示例将汇重子和流联合投影，源重子留在外面，再 gen_correlator；其它多点按物理定义组织。源 dagger 与守恒流真实端点保持各自约定，不能为了方便把流和重子移到同一时间。

符号行保持各自独立的时间 Tag；创建或重写另一个同名行不会改变已有行。`set_time` 同时重写位置与味结构，并保留 dagger。
重子伴随同时反序三条味道腿：`Flavor("uud").conjugate()` 是 `Flavor("bar{duu}")`；
显式构造的 `bar{...}` 按字面顺序保留。此顺序与源端系数及交换矩阵的 `r p r` 一致。

重子已知完整三夸克系数的交换关系时，可在 `Hadron(row, flavor, quark_symmetry="antisymmetric")`
声明；部分交换如 `quark_symmetry={(1,0,2):-1}`。完整系数包含颜色 epsilon，Wick 的
费米子负号另由原算法计算。普通/结构化入口会合并等价图并累计重数；不从相同味道或
自旋名称猜测标签。标签必须在实际基底、时间和模态选择下成立；需要独立有序模态块时
生成不带标签的关联函数。先对每个完整重子行声明，再构造线性组合或乘积；位置与味结构应保留相同顺序的逐重子标签。
后加混合 flavor 时，先前 spin×spatial 的标签不自动成为完整标签；相关耦合 `sum(C_ab B_a F_b)` 不能写成两个独立求和的乘积。当前矩阵族保持固定味道词，不自动利用 `uud/udu/duu` 之间的关系；三部分组合与图选基的区别见下面的交换说明。
定义、兼容检查与可运行示例见[重子交换标签](../../../../docs/modules/operators/baryon_quark_exchange.md)。

MS、MA 分量会在交换下相互混合时，用
`ms, ma = Hadron.multiplet([ms_hadron, ma_hadron], quark_symmetry="mixed")` 一起声明，
再交给普通或结构化 `gen_correlator`。各分量用不同名称，保留相同量子数、小群行、Tag
和固定味道词；绑定器须提供约简使用的伙伴系数。自定义精确矩阵通过 `QuarkExchange`
给出，`tensor_product` 可组合同时交换的混合因子。同味纯混合的完整系数应为零；非零混合
自旋/味道/空间因子须先组合为实际完整系数。矩阵方向、复伴随、归一化和非零数值示例
见上面的交换说明，不能只给一个 MS/MA 分量套正负号代替整族关系。

需要完整目录时，`Hadron.multiplet([s, a, ms, ma], quark_symmetry="all")` 按 S/A/MS/MA
声明；这些仍是实际完整系数，MS/MA 是一个二维表示的两个分量。重子中只有两条同味腿时，
用 `QuarkExchange.pair(slots=(0,1))` 配套 S/A 两个 Hadron；单分量可选 `"symmetric"`
或 `"antisymmetric"`。槽也支持 `(0,2)`、`(1,2)`，第三个夸克味道不同，不是另造二夸克对象。
程序联合使用标量、矩阵、源汇及相同顶点关系，对整个输出共用精确的原始图基；只保证
已声明且保持实际味道词的全部交换关系下独立，不推断实际数值张量的额外低秩。纯表示的
分量数、点态独立图数和完整 Wick 和相消后的图数须区分，计数及证明见交换说明。
