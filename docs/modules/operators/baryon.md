# 重子 Insertion 与顶点

`Baryon` 从已计算的三模态 elemental 构造单重子顶点。它与 `Meson` 一样采用“定义 insertion、组合 operator、加载 elemental、读取顶点”的流程。当前支持等时三条模态腿，不执行传播子收缩或 Wick 展开。空间输入的生产入口见[重子 elemental](elemental_frontend.md#重子)。

```python
from lqcdweave.contracts.elemental import BaryonComponent
from lqcdweave.adapters.elemental import ArrayBaryonElemental
from lqcdweave.operators.single_hadron import Insertion, Operator
from lqcdweave.contraction.binding.api import Baryon

component = BaryonComponent.paths()  # 三条空路径
insertion = Insertion.local_spin("H", "g", {0: (0, 0, 0)}, component)
# 行采用零起始编号；这里选择第二行，保留六个非零自旋项。
operator = Operator("H_row2", [insertion[1](0, 0, 0)], [1])

with ArrayBaryonElemental.from_npy(
    {"vvv": "baryon.npy"}, [component],
    axis_order=("component", "momentum", "time"),
    phase_convention="anchor", cfg="cfg",
) as reader:
    sink = Baryon(reader, operator).load("cfg")
    source = Baryon(reader, operator, source=True,
                    spin_basis="degrand_rossi").load("cfg")
    for factor in sink.factors(0):
        print(factor.spin.entries, factor.spatial.shape)
```

上述定义对应零动量、正宇称的局域三夸克插值场：腿味道顺序为 `(u,d,u)`，
使用归一化的 DR `G1g` 第一自旋行，并取三条空路径。其自旋系数为
\((e_{121}-e_{211}+e_{343}-e_{433})/2\)（此处指标为 1–4）。
这不是旧 DP 自旋构造块的基变换，不再将其标作完整的 `C gamma5` 插值场。
规范场和 Laplace 向量分别使用 `(Lt,Lz,Ly,Lx,4,3,3)` 与 `(Lt,Ne,Lz,Ly,Lx,3)` 布局，
读取方式见[基础数据](../data_readers/data_readers.md)。生产时应提供与向量基底一致的规范场和本征向量；在应用 stout 或 SU(3) 投影时须让生产重子 elemental 所用规范场版本与本征向量所用版本兼容。
输出目录包含每 cfg 一份完整 NPY 与共享 `dataset.json`；数组保持 `(component,momentum,time,mode0,mode1,mode2)`。
裸输入没有已登记的规范场和基底身份，登记时应保留这些身份未知，不能作为严格持久复用的已知输入声明。

## 生产 elemental

使用[Elemental 表格前端](elemental_frontend.md#重子)：初始化提供规范场、一个默认三腿共享的本征基底、动量和包含三腿路径及系数的表格，计算只调用 `calc_path_blocks(t)`。独立三腿基底和 Ne 三元组只在需要时显式提供；Ne 省略时分别继承输入。空间投影和距离平均都表达为路径及系数表，不新增计算入口。相关路径三元组必须保持相关性，不能拆成三腿各自投影后相乘。后端适配完成前，此页新前端不代表已完成生产验收。

需要在读取时只计算一个分量、动量、时间和模式选择时，可用[按需重子 elemental reader](../elemental/baryon_on_demand.md)；它不登记或写出持久产物。

自旋与味道不进入空间 elemental 计算；后续 Insertion 与顶点使用方式保持以下约定。

## 自旋与行

完整预制优先使用[64 行自旋目录](baryon_spin_catalog.md)：与介子一样提供版本化编号表
和不可变多重态，已联合投影到双立方群、宇称和 S3。下面的 `local_spin` 是保留原定义的
12 行局域构造块，不覆盖全部自旋配方。

`SparseSpinTensor` 保存 `(alpha,beta,gamma,value)` 项，三个自旋指标都取 0..3。它允许任意 `(4,4,4)` 张量，不限制任何腿的非零分量数。相同指标精确合并，不按阈值丢弃小系数。`dense()` 显式恢复小型自旋张量。

自定义三腿行通过 `Insertion.from_rows(rows, momentum_dict)` 提供，每个 `InsertionTerm` 显式带有 `component`、`spin` 和可选复系数；同一行可组合不同空间分量。物理味道由 correlator 的 hadron/Wick 定义，并在传播子绑定时解析；`BaryonComponent` 的路径键只表示有序空间腿，旧 `flavors` 元数据不定义物理味道。静止系的常用 G1/H 自旋行可用 `Insertion.local_spin` 构造；一般 spin-irrep 与 spatial-irrep 的耦合见[重子不可约表示耦合](baryon_irrep_coupling.md)。

```python
from lqcdweave.operators.single_hadron import Insertion, InsertionTerm, SparseSpinTensor

spin = SparseSpinTensor([(0, 1, 2, 1), (2, 3, 0, 1j)], basis="degrand_rossi")
insertion = Insertion.from_rows(
    [[InsertionTerm(component, spin=spin)]], {0: (0, 0, 0)},
)
```

`local_spin` 提供静止系 G1/H、g/u 的归一化自旋构造块，直接采用 `degrand_rossi` 基。
令 \(\eta=+1\) 对应 g、\(\eta=-1\) 对应 u；每行由上块的自旋组合及其三腿 gamma4
像构成。下面公式的自旋指标为 1–4，API 的行号和自旋指标均从 0 开始。

| 表示 / API 行号 | 自旋张量 | 非零元数 |
|---|---|---:|
| G1 / 0 | \((e_{121}-e_{211}+\eta e_{343}-\eta e_{433})/2\) | 4 |
| G1 / 1 | \((e_{122}-e_{212}+\eta e_{344}-\eta e_{434})/2\) | 4 |
| H / 0 | \((e_{111}+\eta e_{333})/\sqrt2\) | 2 |
| H / 1 | \((e_{112}+e_{121}+e_{211}+\eta e_{334}+\eta e_{343}+\eta e_{433})/\sqrt6\) | 6 |
| H / 2 | \((e_{122}+e_{212}+e_{221}+\eta e_{344}+\eta e_{434}+\eta e_{443})/\sqrt6\) | 6 |
| H / 3 | \((e_{222}+\eta e_{444})/\sqrt2\) | 2 |

每行满足 \(\|G\|^2=1\) 和 \(\gamma_4^{\otimes3}G=\eta G\)。
自旋构造不执行味道投影；味道由 correlator 的 hadron/Wick 定义并在传播子绑定时解析。调用方仍须确认空间结构与所选自旋/味道交换对称性相容。任意手写行的不可约表示归属由调用方负责；局域 G2 不受支持。
此配方替换了原来在 DP 基内选块再转到 DR 的配方，改变了具体算符。已有显式
`SparseSpinTensor` 不会被重写；使用旧定义保存的算符或计划不能当作新定义复用，应显式重建。

`Baryon` 默认输出项目的 `degrand_rossi` 基，与现有 VSV 自旋约定相接，并保留上述预定义行的稀疏性。可以显式选择 `dirac_pauli` 转换自旋系数，但后续使用的传播子也必须处于该基；此对象不转换传播子。输出因子的 `spin.basis` 标明实际基，转换不会丢弃分量。两者关系为 q_DP=U q_DR，其中 U=[[I,I],[I,-I]]/sqrt(2)。

## 数据与所有权

数组形状为 `(component,momentum,time,N0,N1,N2)`，三个模态维度可以不同。`ArrayBaryonElemental` 复用 NPY 映射读取，`axis_order` 显式说明文件的前三轴。`BaryonComponent.paths((P0,P1,P2))` 保留有序空间路径；也可用 `BaryonComponent(key)` 标记已投影的分量，键的物理定义由数据提供者维护。

相位为共同锚点的 `anchor`，或明确不含相位的 `none`，读取器与顶点必须一致。`load(cfg, usedNe=n)` 同时截取三腿；传三元组可分别指定，超出数据覆盖时报错。

`factors(t)` 输出独立于读取器临时缓冲的空间数据。同次遍历的相邻相同分量可能共享此输出数组，请视为只读；需要修改时复制。关闭读取器后，已取得的因子仍有效。

`get(t)` 显式展开为 `(4,4,4,N0,N1,N2)`，仅适合小规模检查。大数据使用 `factors(t)`，避免自旋与空间张量的外积存储。

## 源端约定

源端对系数及 elemental 取共轭，并在三条腿上分别作用 gamma4。模态和自旋轴仍按原始腿 0、1、2 标记；`field_order=(2,1,0)` 表明厄米共轭后的反夸克场顺序，汇端为 `(0,1,2)`。此处不添加 Wick 重排符号，后续收缩必须依据字段顺序处理。所有腿时间均为所选 t。

关联函数模态分块保留这三个原始空间槽顺序，源端 dagger 不反转模式轴；`field_order` 只用于传播子字段映射。`B.with_mode_indices` 选择原始槽，VVV elemental 与 VSV propagator 按实际模式索引读取。全对称径向三元组投影及其适用边界见[重子关联函数模式分块](baryon_mode_blocks.md)。

重子通过独立的 `ElementalWorkspace` 类型复用规范场准备、几何相位及缓冲分配，不再创建介子生产器或访问其私有状态。加载或变换失败后需重新加载。详见 [elemental 工作区说明](../elemental/elemental.md#共享工作区与失败恢复)。

## 按量子数组计算关联矩阵

调用方按实际量子数准备各组的 `Hadron` 列表，以及对应的 `Baryon` 原生绑定。
完整汇×源矩阵由同一符号和计算入口处理，组名只作为输出键，不自动投影同位旋或补齐缺失算符：

```python
from lqcdweave.operators.correlator.api import gen_correlator
from lqcdweave.contraction.correlator import calculate

requests = {name: gen_correlator([basis, list(basis)], ["sink", "source"],
                               [False, True], structured=True)
            for name, basis in groups.items()}
planned = calculate(requests, vertex_map=vertex_map, propagator_map=propagators,
                    configuration=cfg, time_axes={"sink": sink_times, "source": source_times},
                    execute=False)
values = planned.run().value
```

`groups`、`vertex_map` 和 `propagators` 来自已确认的物理定义与当前 cfg readers。
三腿自旋、空间基底、源端 dagger 与味道映射见[数据绑定](../../../.agents/skills/lqcdweave-correlator/references/binding.md)。
所有输入在执行期间保持有效；结果保留原组名与算符顺序。兼容 cfg 的生产循环读取已保存的完整计划，
具体归档与重绑定见[计划复用](../../../.agents/skills/lqcdweave-correlator/references/plans.md)。
统一前端的 CPU 接线回归见[重子 Diagram 测试](../../../tests/test_baryon_diagram_frontend.py)，不作为真实 GPU 或生产规模验收。
