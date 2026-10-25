# V/D elemental 与统一计权

V 是实际 Laplace 基底，D 是已有 dilution 混合向量。两者的求解、投影和空间张量使用相同数值内核；端点身份、向量数、随机分组及估计量语义分别保存。low 的 V 和 medium 的 D 不以相同 shape、seed 或数组下标合并。

## 导入实际 D

实际 D 来自 [NV](../../lqcdweave-noisevector/SKILL.md)。D 数据发布后，用实际 `data_id` 绑定完成来源；NV 的 `describe()["time_pools"]` 保留本次 draw 在各物理时间的随机池，每次 `calc` 都重新生成池 ID，即使 seed 相同。

```python
from lqcdweave.contracts.dilution import DilutionSource

d_source = DilutionSource.from_description(
    noise_generator.describe(), data_id=actual_d_data_id,
)
```

`d_source.document` 是可持久化 JSON；`DilutionSource.from_document(document)` 读回同一来源，缺失记录返回 `None`。它保存父基底实际 ID、每组有序 parent modes、n/m、科学向量标签 `(block,in_block)`、实际 qr/exact/u1 分支、raw 协方差和时间池。`source.ordered_indices` 为张量 D 轴的展平整数标签；科学标签由 source 显式映射。实际 D 自身保存 completed NV 描述到 `provenance.generator`，发布后由 `DatasetFieldHandle` 用真实 dataset ID 构造 `dilution_source`；引用 D 的 elemental 再保存完整 `source.document`，避免在 D 自身定义中循环引用其 ID。不能从 vector axis、mode_offset 或 seed 重建。

D handle 提供 `load(cfg)`、`Ne`、`basis_id`、有序 `mode_labels` 和 `dilution_source`；可以是适配已有实际数组的高级句柄。使用流式读取时沿用 `read_modes(cfg, physical_t, start, stop)`。声明 `basis_orthogonal` 时由调用方保证实际证明。裸数组可以生产 raw 张量，但不会自动携带 D 来源或正交证明。

```python
generator = ElementalGenerator(
    gauge_reader, None, momentum_dict=momentum_dict,
    operator_table=operator_table,
    vector_bases={"V": v_handle, "D": d_handle},
    blocks=("v2v", "v2d", "d2v", "d2d"),
    elemental_weighted=True, lattice_size=lattice,
).load(configuration)
blocks = generator.calc_path_blocks(physical_time)
```

`Ne` 可按 kind 显式传 `{"V": Nv, "D": Nd}`，默认各用实际全轴；`eigenvector_selection` 也可按 V/D 分别保存实际时间映射。只生产 D/D 时可仅给 D，第二位置仍传 `None`。九个扇区是 `v2v/v2d/v2p/d2v/d2d/d2p/p2v/p2d/p2p`；D/P、P/D 只把向量腿换为 D，点加颜色轴及 sparse P/P 表示沿用原约定。导数入口及重子入口保持各自原有范围。

## 保存和读取端点

生产器的 `basis_endpoints` 和 `dilution_endpoints` 可直接交给 `ArrayElemental`。两者键均为 `(sector, "bar"|"field")`，bar 是 raw 左端，field 是 raw 右端。`basis_endpoints` 的值为 `(actual_data_id, ordered_axis_labels)`；`dilution_endpoints` 的值为完整 `DilutionSource`。source dagger 由绑定层交换真实端点，调用方不手动再交换一次。

```python
elemental = ArrayElemental(
    saved_arrays, generator.components, phase_convention="anchor",
    basis_endpoints=generator.basis_endpoints,
    dilution_endpoints=generator.dilution_endpoints,
    elemental_structures=generator.elemental_structures,
    elemental_weighted=True,
)
```

`saved_arrays` 按 `(component,momentum,time,...)` 保存；copy 借用块后再继续计算。裸 NPY/Binary 必须另存以上声明。登记 Dataset 的 `definition.logical_sector` 保留具体 V/D 扇区，`conventions.dilution_endpoints` 以 bar/field 存来源 document，实际 `bases` 绑定轴身份。`DatasetElemental` 从登记规则读回，不把通用 `meson_elemental` 存储 kind 自动称为 V/V。向量间产品可用 `elemental_dataset_document(..., product=sector, right_basis=..., right_basis_id=..., right_modes=..., elemental_weighted=...)`，点端 planner 支持 d2p/p2d；共同存储仍由 DatasetWriter/Reader 管理。

传播子保持六个保存代表、九个读取视图。`PropagatorPairReader(existing_pair_reader)` 将实际 `PropagatorReader` 接到 `calculate` 的 reader 协议；多个实际端点组合时显式给 `endpoints={"vsd": (v_endpoint,d_endpoint), ...}`。传播子的 D 端点实际 ID 和展平标签须与 elemental 一致，Gamma5 推导交换端点与物理时间。见 [PR 说明](../../lqcdweave-perambulator/references/dilution.md)。

## 同一个 elemental_weighted

生产与消费都使用 `elemental_weighted`，并与保存的数据标记一致。它决定权重放置，raw 与 preweighted 路径最终使用同一混合列分量估计量。令 n 为该组完整父空间的列数，m 为保留的 QR 混合列数，a、b 是**实际 D 混合列标签**。同组的相等标签只算一次，r 个不同标签的包含补偿为

\[
H_r=\frac{(n)_r}{(m)_r},\qquad H_0=1.
\]

这是完整 QR 旋转基底中可交换列的包含系数。它作用于混合列轴，与旧入口直接抽取 Laplace 父模态的选模语义不同；`D=VQ` 的 raw 振幅仍不缩放。V/D 使用同一数值核不意味着二者需要相同的随机计权。

| Elemental 的 D 分量关系 | 局部包含权重 |
|---|---:|
| 单个 D 端点 | `H1=n/m` |
| D/D，同一 pool 和组，a=b | `H1=n/m` |
| D/D，同一 pool 和组，a≠b | `H2=n(n-1)/(m(m-1))` |
| D/D，两个独立组或池 | `H1_left * H1_right` |
| exact 或 m=n | `1` |
| legacy u1，单端一阶 | `1` |

因此同组 D/D 的对角项乘 H1，非对角项乘 H2；不能给整个 D/D 块统一乘 `(n/m)^2`。跨组项使用各组的一阶因子乘积。D/P、P/D 同时包含 D 和 P 的局部系数，仍共用同一个选项。已证明完全对角时，只需处理有支持的 a=b 分量，不要求为不存在的不同列项计算 H2。

`elemental_weighted=False` 生产 raw，correlator 加完整图包含补偿；`True` 在生产放置当前 elemental 的局部系数，correlator 按实际相等分区除去这些局部系数并计算整图余权。单个 elemental 在同一池有多个 D 腿时，它的局部因子是 `H_local_distinct`，并非每条腿的一阶因子外积。物理算符、Gamma、味道和 Wick 系数照原约定处理。

## 同池混合列包含系数与范围

同一实际 parent basis、同一声明的 pool UUID、同一 ordered parent group 的多个 D 分量按照混合列标签的相等关系展开；物理时间不拆分显式共享的池。一个自由求和场景里的不同求和标签仍可以取相同数值，因此直接给整个自由和乘 H_k 会重复计数。图展开复用 P 的 `SamplingGroup/expand_sampling_scenes`：以精确 Fraction 做 labelled-partition 的 zeta 逆变换，并在已知 D/D 对角支持下先合并相等腿。

例如两腿 raw 同池展开为：

\[
H_2\sum_{a,b}F(a,b)+(H_1-H_2)\sum_a F(a,a).
\]

这里两个分量用同一个自由和及等标签和表达，保留所有原传播子、自旋和顶点连接。若此 D/D elemental 已经在 a=b 处乘 H1、a≠b 处乘 H2，单流消费的余权为 1。多个已计权 elemental 共用一池时，剩余系数按全图各分区的 H_r 除去每个局部 `H_local_distinct` 后再作 zeta 逆变换，不能统一除以 `(n/m)^k`。

默认 NV 为不同时间生成不同 UUID，实际 source 内的这些时间池及不同分组由其算法独立契约确定。高级输入若在 `time_pools` 的不同物理时间显式声明同一 UUID，则表示同一组完整混合列中的共同列选择：跨时间的相等标签只计一次，不能把两端 H1 相乘。各时间仍绑定各自的实际向量和传播子，共享 UUID 不声明向量数值相同。不同实际 D 来源、不同随机池需在 calculate 沿用已有 `groups_independent=True` 声明，各组和池分别展开；调用方须保证独立性声明真实，不能靠改 UUID 将相关数据变成独立数据。同 seed 复现不提供额外独立样本，复用一份已保存实现时保留原 source 与 time_pools。同时包含随机 D 与随机空间 P 的图也需要已有独立性声明；确定的完整空间点集不增加此要求。

以每个时间的完整三列选择两列为例，平均范围取决于实际选择关系：

| 两个时间的列选择 | 穷举组合数 | 整图包含关系 |
|---|---:|---|
| 显式共享 UUID，共用同一子集 | 3 | 两个时间联合按相等/不同列用 H1/H2 |
| 不同 UUID，各自独立选子集 | 3×3=9 | 两个池各自计权后相乘 |

零动量局域 D/D 若在两个时间各自有完全对角证明，四条 D 腿先各合并为一列，整图只剩两个自由标签，m=2 可估计。一般四条自由标签不能据此用 m=2 估计。三方案和九组合的平均各自对应其声明的选择关系；不能拿三份相关样本作为独立池的穷举基准。

两个流位于同一时间时，同样按实际 pool UUID 和有序分组联合计权，不按流分别认作独立样本。若图中其余端点为 V，且两个 D/D 流都有完全对角证明，仍只有两个自由标签。以 n=3、m=2 为例，raw 自由和的系数为 3、等标签和的修正为 −1.5；两个流分别预计权 H1=1.5 后，correlator 的自由和余权为 4/3、等标签和修正为 −2/3。这分别给同标签总权 H1、不同标签总权 H2。一般双 D/D 流没有这些对角约束，完整三列中可以出现三个不同标签，三选二无法覆盖而拒绝。实际 weak_field 同时间双流验收见[测试说明](../../../../tests/README.md)。

一般双 D/D 流在同一池可需要四个不同标签，因此五取四可估计，整图按实际不同标签数使用 H1=5/4、H2=5/3、H3=5/2、H4=5。预计权路径按分区扣除两个 elemental 各自的局部 H1/H2，不能统一除以 H1 的四次方。weak_field 验收穷举五种选取方案，并核对一般空间路径、非零动量、固定 complex QR 混合和非零四标签全不同贡献；四取三则不能覆盖四标签全不同的项。

可估计性由商化后**有效不同混合列标签数**决定：需要的 r 若大于 m，包含概率为零，明确报错；已证明完全对角 D/D 的两腿共用一列时 m=1 也能估计，不按原始腿数或 projector 数量一律拒绝。通用服务沿用已有 partition 枚举策略，没有固定的五阶上限。exact 或全秩 QR 返回一个不限制相等关系的自由求和场景，系数 1，不限阶也不枚举所有分区。legacy u1 的一阶协方差为 1；同池高阶拒绝，不能套用正交混合列的包含公式。

保留完整组及其组内/组间重排可用，部分组选择尚无当前消费契约而拒绝。缺少实际 D、父基底或所需 pool 的旧数据可读取原始张量，但不能补猜随机估计量。同一 actual D ID 对应不同完整 source document 时明确拒绝，不能重新命名随机池来改变同一份数据的统计关系。

## 计权语义与旧数据

登记 D elemental 的 `conventions.dilution_weight_scheme="component_inclusion-v1"` 是内部保存的语义标签，由生产 planner 自动写入，不是新的前端选项。`elemental_weighted` 仍是用户控制权重放置的唯一开关。DatasetElemental 读取已计权 D 时要求这个标签；缺标签或声明旧方案的数据明确拒绝，须从 raw 数据重新按对角 H1、非对角 H2 生产预权。

`ArrayElemental(..., elemental_weighted=True)` 声明调用方已经应用**当前**混合列包含语义。裸 NPY 和旧 `(n/m)^2` 数组不能仅通过改标记变为新数据，应保留原 raw 并重新计权。来源完整的未计权 raw D 可以按当前规则重新规划；旧方案的保存计划已固化不同估计量，即使原值 raw 也不能直接重放，恢复时来源 identity 的方案版本会拒绝不匹配。重新生成计划不改变实际 D 向量和传播子。

结构声明仍只有一般、完全对角、P 对角，缺少标签的旧数据按一般。同一已证明正交基底的零动量局域 V/V 和 D/D 可对角优化，V/D 交叉块不由 shape 推断对角。high 补空间仍用完整 low+medium Laplace 集合定义，不能用随机低秩 D D† 替代。由 `gen_correlator` 生成的普通介子关联函数支持此 D 路线；重子混合图和矩阵专用 D 路线没有扩展。

raw 与 preweighted 的同输入 CPU 对照见[关联函数计权回归](../../../../tests/test_dilution_correlator.py)，九扇区的独立显式矩阵参考与登记读回见[elemental dilution 回归](../../../../tests/production/elemental/test_dilution_sectors.py)。这些合成测试不证明真实 LQCD、GPU/MPI、性能或生产规模验收。相关数学验证见[测试说明](../../../../tests/README.md)。
