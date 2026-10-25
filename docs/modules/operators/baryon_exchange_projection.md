# 先组合 spin×spatial，再作交换群投影

重子沿用介子的组合顺序：先构造自旋与空间行，耦合到目标双立方群表示，
再投影组合系数的交换表示；最后在 `Hadron` 层组合 flavor。
算符构造只处理配方和系数，不读取规范场、基底或 elemental 数值。

```python
combined = Insertion.from_irreps(
    spin_recipe, spatial_rows, "T1u", "H", momentum_dict,
)
projected = combined.project_exchange("A", operator_table=spatial_table)
operator = Operator("baryon_Hu", [projected[0](0, 0, 0)], [1])
```

`spatial_rows` 必须是真正的空间表示行，约定同[既有立方群耦合](baryon_irrep_coupling.md)。
立方旋转与夸克槽置换彼此对易，交换投影保留原来的立方行顺序、宇称和移动系入口。
它也可用于 `Insertion.from_rows` 的显式三腿系数；该情况下不会凭空赋予立方表示。
构造及独立投影 CPU 回归见 [test_baryon_exchange_projection.py](../../../tests/correlator/operators/test_baryon_exchange_projection.py)。

## 投影接口

```text
insertion.project_exchange(
    target, *, operator_table=None, copy_index=0, slots=None,
)
```

| 参数 / 返回 | 含义 |
|---|---|
| `target="S"`、`"A"` | 返回对称或反对称的一个新 Insertion |
| `target="mixed"` | 返回同一份二维表示的 `(MS, MA)` 两个 Insertion |
| `target="MS"`、`"MA"` | 单独选择该伙伴；与 mixed 入口的相位、份数一致 |
| `operator_table` | 与输入 component 对应的空间生产表；不读取数值 |
| `copy_index=0` | 选择此次组合中独立的交换表示份数，不是立方耦合的份数 |
| `slots=None` | 完整 S3；给 `(0,1)`、`(0,2)` 或 `(1,2)` 时仅投影相应 S2，支持 S/A |

内置 `baryon-vvv/v1` 分量和显式 `BaryonComponent.paths(...)` 可省略 operator_table；
自定义目录及不透明 component 必须提供匹配表。未知目标、不存在的份数或零支持明确报错。
混合表示最多保留两个由当前种子生成的独立份数；输入若只含其中一份，越界选择会报错。

返回对象还记录 `exchange_irrep`、`exchange_component`、`exchange_copy_index`、
`exchange_multiplicity`、`exchange_slots` 和 `exchange_matrices`。
MS、MA 共同组成二维表示，不能当作互不相关的两个一维符号。
其中 `exchange_matrices` 采用现有 `QuarkExchange` 的精确矩阵和列约定。

## 配套空间表与自动准备

交换作用可能将原来第 0 条腿的路径移到另外两条腿。返回的
`projected.operator_table` 自动覆盖实际用到的这些路径，应交给原有入口：

```python
elemental = Elemental(
    gauge, basis, projected.operator_table, momentum_dict,
    directory=shared_elemental_directory,
)
vertex = Baryon(elemental, operator)
```

已有投影空间分量先按其真实路径系数展开。若原目录包含所需单位路径三元组，继续使用
同一目录、版本和编号；否则形成共同契约支持的内嵌自定义目录。组合系数保留在自旋项中，
新空间表为单位路径项，生产时不重复施加系数。原表、原 insertion、已保存数据均不修改。
后续复用、缺项补产、保存和恢复沿用同一 `Elemental → calculate` 工作流。

被交换腿须使用相同的实际有序基底、时间和截断，才能把此关系用于模态张量交换及图约简。
不同腿的独立基底或不同有序模态块不能据此视为相同空间。图级交换声明继续执行既有端点检查。

## 颜色、flavor 与缩并标签

本项目空间 elemental 是颜色 epsilon 收缩后的三腿张量，因此这里实际投影的是
`spin × epsilon-contracted spatial`。每次槽交换只计一次颜色奇偶性，不能再额外乘 epsilon 符号。
例如对称自旋乘局域 elemental，组合系数为 A；投影到 S 应无支持。
若按不含颜色的纯 `spin×space` 讨论，对应标签需再乘颜色 A。

**flavor 在上述组合投影完成后加入**，与介子的分层方式一致：

```python
hadron = Hadron(symbolic_row, Flavor("uuu"), quark_symmetry="antisymmetric")
```

这个声明适用于上面得到 A、并且后加味道为对称 `uuu` 的完整系数。
其它味道仍按其真实组合和保持味道排列的子群处理。投影方法不把自旋或空间某个因子的标签
自动冒充为完整重子标签；只有最终成立的关系才用于 `Hadron.multiplet` 和图约简。

## 算法与验收

使用矩阵单位 `P_ab = d/|G| sum(conjugate(D(p)[a,b]) U(p))`。
`U(p)` 同时置换自旋、三条路径及三腿端点偏移，并计入空间的颜色奇偶性。
S/A 是标准线性幂等投影，不额外重归一化。混合表示先对行零的两个候选种子列作精确
Gram–Schmidt 去依赖，再用同一组系数运输所有立方行及 MS/MA 伙伴；保留输入幅度和相位，
不将不同份数强制归一化。份数按候选列首次独立出现的顺序确定。

精确小矩阵来自既有 `QuarkExchange`；使用 SymPy 精确表达输入浮点值及根式，
只在交付普通数值自旋系数时转换。独立性检查不采用数值秩阈值，不丢弃用户的微小非零系数。
投影只在构造阶段执行，数值收缩阶段不反复做群求和或符号消元。

`tests/correlator/operators/test_baryon_exchange_projection.py` 对照独立颜色行列式和完整
自旋×模态张量的逐指标置换，验证 S/A 幂等、两份 MS/MA 的全部 6 个交换、三种 S2 位置、
自定义路径闭包和微小系数；另核对全部双群旋转的协变性。
端到端使用后加 flavor 的普通/结构化关联函数，比较全 Wick 枚举、交换约简和独立 Fock 行列式，
并实际触发配套空间表的自动准备。验证范围为小格点 CPU，不代表 GPU 或生产规模。
