# 使用空间取样的关联函数

实际测量使用随机空间点取样时加载，包括流及其它已支持点端任务；确定点集只核对真实端点/坐标和输入，常规模态测量或单纯时间扫描不自动抽样。

绑定实际有序点集、跨时间共用或独立策略，以及参数扫描之间的共享池。实际 realization 的池身份须随保存和读回保留；同 seed 的新生成不等于同一池。点生成、真实流端点与局部结构由 [elemental 点端参考](../../lqcdweave-elemental/references/point-flow.md)和[流参考](../../lqcdweave-elemental/references/current.md)负责；需要补数据时才加载。未支持的端点明确报告，不把高模重子能力推广为任意采样流重子多点。

## Elemental weighted

默认 `elemental_weighted=False`。受管 elemental 在测量入口设置一次；已有预权数据则使用与其标记相符的同一选项：

```python
result = calculate(
    correlator,
    vertex_map=vertices,
    propagator_map=propagators,
    elemental_weighted=True,
)
```

P 的局部系数和适配器声明统一见[空间点计权](../../../../docs/modules/operators/elemental_frontend.md#空间点采样计权)，D 的父分组、混合列包含系数和支持范围统一见[D 计权](../../lqcdweave-elemental/references/dilution.md)。不在消费脚本手写局部系数或重复乘权。`ArrayElemental` 的同名标记表示输入值已计权，不会修改 block；Dataset 各 sector 的标记须一致。COO values 同样按已计权理解，所需系数不为 1 时不能用无 values 的 identity 记录替代。

结构证明只用 `general`、`diagonal`、`point_diagonal`，定义和来源见[结构契约](../../lqcdweave-elemental/references/current.md)。无标签旧数据按一般处理。expected compensation 下，实际 pool ID 和有序选择相同的对角 P/P 腿可先合并；不同池、observed compensation 或混有未证明的一般项不能使用该简化。D 同样依实际组/池与结构证明展开，不按时间不同就推断独立。

Correlator 根据已乘局部权重，用精确 zeta 逆变换补全图剩余系数。两个 current 共用一池时不能直接相乘局部权重；显式独立池还须声明 `groups_independent=True`。`usedNp` 缩小原保存点集时，余权按原始 `sampled_points` 与实际选择数修正。普通 D reader 只支持整组保留/重排；组内减少列数需要显式联合列选择估计量，不能裁数组后沿用原 m。

完整 `plan_archive` 固化计权选项、结构和剩余系数；恢复时无需再传该选项，但新 reader 的结构须匹配。改变计权或结构需重新规划。`False` 保留未预权数值路径，已证明结构仍可减少场景。缺失实际池或父基底来源时不补猜；旧 D 计权方案须从 raw 重产，不能重放不兼容的估计量档案。

空间点预计权限于 expected 简单均匀无放回抽样；其它设计不冒用该公式。实现细节见[局部生产计权](../../../../docs/modules/elemental/elemental_production.md#optional-per-elemental-point-compensation)与[图级余权](../../../../docs/modules/expansion/expansion.md#elemental-local-compensation-and-graph-residual)。
