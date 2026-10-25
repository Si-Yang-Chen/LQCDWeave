# 单流独立外腿数值入口

本页仅供用户明确选择完整另一腿的专家研究。有限点/向量传播子的方法验收必须使用
[有限抽样的比较约定](blending_comparison.md)，不能自动复用本页的额外算符作用源。
历史脚本`run/run-fields`现在要求显式`--complete-other-leg`，避免误作生产验收入口。

`contraction.external_current` 是类似单时间对 meson_matrix 的专家数值入口；
普通多流图仍走 `contraction.correlator.calculate`。此处只计算已指定的连通介子单流迹，
不提供通用 Wick 展开、断开图或第二个流，也不把它冒充普通联合 Diagram 档案回放。
任务编排层 `contraction.external_current_measurement` 共用这些内核生成逐腿贡献并装配参数点。

输入 X 是完整流时间片上的 `S(t_current,t_meson) V_meson`，轴为
`(current_spin,meson_spin,z,y,x,color,meson_mode)`。介子端顶点已经包含实际
Gamma、elemental、系数和相位；源流按既有 gamma4 dagger 和 anchor phase 约定。
内核复用 PR 的 Q 点提取和 EL 的路径输运，而不是只读取入选点之间的 gauge link。

设源流 dagger 后为 A，介子顶点为 B。Gamma5 厄米性给出完整连通迹
`-Tr[B gamma5 X† gamma5 A X]`。只在 `X†` 与 `gamma5 A` 之间插入一次空间单位阵。
确定性 V 与 QR medium 使用同一向量投影两侧；high 点贡献使用 `(Q X)(x)`
的共轭腿及完整 `(gamma5 A X)(x)` 操作腿。**不在两条腿都施加 Q**，也不把另一条腿
截为相同 P 子集。这样完备性是 `V V† + sum E[D D†] + Q = I`，
单点及单组权重分别为 `Ns/Np` 和 `n/m`，由原生 expected_weight 提供。
这是单插入估计量，不能复用旧双插入估计量的同分量/异分量二阶权重。
这里的 high 抽样向量同样称为 H=Q eta，但只对应一次单位阵插入中的投影腿。
普通 HSV/HSD 可以提供相应交叉块，不能代替完整算符作用腿，也不能把两腿都改成 HSH。
H 的[生产路线](../perambulator/point_highmode.md#h-contract)与本节单插入估计量需要分别核对。

所有方向的逐 V、D、点贡献保留复数及有序标签，参数扫描在这些贡献上做线性装配。
它与旧双插入测量具有相同的抽样期望，单次样本的方差和结果一般不同。
不能把两者的单次结果相等作为测试条件，或将其混入同一统计定义。

CPU 独立矩阵测试覆盖复系数、非零动量、非平凡规范链、全空间迹、
两父组各4列中取2/3/4列的全部方案平均，以及不同 P 子集的全部方案平均。
也验证单个入选锚点具有未入选的位移邻点贡献。此验证不等于真实 QUDA/DCU 验收。

同一外腿还可以从全部 RHS 在流时间0的 `V(t)† S(t,0) xi`、
`V(t)† S(t,0) O xi` 及 gamma5 厄米性构造，不需要保存完整 X。
`reverse_flow_leg`、`flow_operator_leg` 提供该表示；源端空间 O 与 dagger 后流核的
关系不能任意交换。采用完整另一腿的研究路线时，
原始完整父辅助块支持运行时 Q 派生。独立逆矩阵测试逐腿核对两种表示，并检查所有 RHS 在0；
另验证利用实际 D 列和仅剩余正交方向精确恢复完整父传播子。
