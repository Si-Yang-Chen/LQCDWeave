# 单介子多算符、多谱截断矩阵缩并

`lqcdweave.contraction.meson_matrix` 从旧版 `lattice/correlator/meson_matrix.py` 复制数值核，计算一个实际源汇时间对的连通介子矩阵。输入数据及基底兼容性由调用方检查；该模块不读取文件、不寻找缺失的 elemental，也不推断不同 Ne 文件是否同一有序基底。

```python
from lqcdweave.contraction.meson_matrix import meson_matrix_from_operator_data

values = meson_matrix_from_operator_data(
    forward, backward_raw, operator_data,
    source_time, sink_time, gamma4, gamma5,
    (23, 46, 70),  # backward mode cuts
    (23, 46, 70),  # forward mode cuts
    strategy="auto",
)
# values.shape == (N_sink_operator, N_source_operator, 3, 3)
```

`forward` 的轴为 `(sink_spin,source_spin,forward_mode,forward_mode)`，`backward_raw` 为同样的自旋轴及 backward 模态轴。`operator_data` 中每个算符为 `(spin_terms, elemental_terms)`：分别有 `(term,4,4)` 和 `(term,time,backward_mode,forward_mode)` 轴。一个算符可以包含多个 Gamma×空间行。前后向模态数可不同，适用于轻粲混合矩阵。

两个 `cuts` 列表长度相同、严格递增，末项分别等于 backward/forward Ne。第 (p) 个 profile 在每一条模态腿上取前缀；输出包括全部 (p,q) 交叉 profile。源端顶点使用 `gamma4 Gamma† gamma4` 和空间矩阵共轭，反向传播子由提供的 `backward_raw` 作 gamma5 Hermiticity 变换；连通环乘负号。当前入口不做源时间平均、味道因子、断开图或额外归一化。

`auto` 在 NumPy 选择 `blocks`，在 CuPy 选择 `batched`；还可显式选 `weighted`。`tests/contraction/meson_matrix/test_meson_profile_matrix.py` 对照独立完整缩并，覆盖不同前后模态数、全部交叉 profile、多项算符及源端/反向传播子约定。目标设备的吞吐和内存需按实际算符数与 Ne 测量。

三个公开入口都接受 keyword-only 的 `array_module=np` 与 `backend='numpy'`。`array_module` 提供分配、前缀累计和堆叠操作，`backend` 原样传给每个 `opt_einsum.contract`；两者须对应兼容的数组实现。已有 CPU 调用继续使用 NumPy 默认值。调用 CuPy 时显式传入 `array_module=cupy, backend='cupy'`；`strategy='auto'` 按声明的 backend 选择：仅 `cupy` 使用 `batched`，其他 backend 使用 `blocks`。不接受 `backend='auto'`，避免 opt_einsum 按数组 Python 类型推断并导入后端。核心不会从输入数组类型发现 backend。

消费旧产物前须核对组态、源汇时间、动量、38/40 行定义、实际有序本征基底、相位、自旋基和边界约定。相同 Ne 或数组 shape 不足以证明可直接截取。带来源的完整矩阵发布、恢复、断开图及味道组装属于调用方 流程，不能由这个单时间对数值核替代。

<a id="entry-routing"></a>
## 专家入口：按已持有的数据选一层

普通完整物理测量仍使用 `calculate`。只有要直接计算单时间对连通矩阵时才选择：

1. 已有 `(spin_terms,elemental_terms)` operator_data：`meson_matrix_from_operator_data`，使用本页首例；由此层完成源端 gamma4/空间共轭及 backward_raw 的 gamma5 变换
2. 已有独立 elemental + gamma：`contract_meson_matrix_blocked`；backward 已做 gamma5 变换、gamma_source 已做源端自旋变换，函数内部共轭 source_elementals，勿重复
3. 已有完整 `(operator,4,4,NeA,NeC)` 自旋空间顶点：`contract_meson_vertices_blocked`；源端所有 dagger/相位/复系数已经正确进入 source，函数不再替用户变换

```python
from lqcdweave.contraction.meson_matrix import contract_meson_matrix_blocked
matrix = contract_meson_matrix_blocked(
    forward, backward, sink_elementals, source_elementals,
    gamma_sink, gamma_source, cuts_a, cuts_c,
    strategy="auto", array_module=np, backend="numpy",
)
```

此层 elemental 为 `(operator,NeA,NeC)`、gamma 为 `(operator,4,4)`；输出
`(sink_operator,source_operator,sink_profile,source_profile)`。两组严格递增 cuts 分别以 NeA/NeC 结束且长度一致。
保留算符与 profile 次序、连通负号，不添加味因子、断开图、源平均或归一化。
多项复系数先按所属 operator/vertex 组合；嵌套 cutoff 不是一般 radial profile。
`auto` 依据显式 backend 而非自动探测设备；batched 可能增加内存，不保证所有规模更快。
