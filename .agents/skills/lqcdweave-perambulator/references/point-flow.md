# 点端传播子

仅当任务实际需要 PSV、VSP/DSP、PSP 或投影点 H 时读取。生成器在 `calc` 中接收实际源和按序汇；同一源的一次求解可投影到多个不同汇，返回项与 `sink` 顺序一致。

| 产品 | 源输入 | 汇输入 |
|---|---|---|
| PSV / PSD | `VectorBasis(V/D)` | `PointBasis(P)` |
| VSP / DSP | `PointBasis(P)` | `VectorBasis(V/D)` |
| PSP | `PointBasis(P)` | `PointBasis(P)` |

例如，点源同时投影到 V 汇与点汇会共享同一次点源求解，并按顺序返回 VSP 与 PSP：

```python
from lqcdweave.production.perambulator.api import PointBasis, VectorBasis

vsp, psp = generator.calc(
    source_time,
    src=PointBasis(source_points, count=N_source),
    sink=[
        VectorBasis(vector_sink, count=N_vector),
        PointBasis(point_sink, count=N_sink),
    ],
)
```

产品由两个实际 endpoint 决定：首字母是汇端，末字母是源端。结果 `.values` 第一轴按 `sink_selection.times` 顺序排列；`.at(t_sink)` 返回对应物理时间对的 `PropagatorBlock`。结果拥有自己的投影数组。

## 高模点 H

投影点 `H=Q eta` 仍以 `PointBasis` / P endpoint 编码，完整父空间通过 `PointBasis(..., highmode_basis=high)` 指定。`Q=I-U U†` 使用全部 low+medium 父模式，在点源求解前或点汇采样前应用；D dilution 向量不能代替完整 low/medium 空间。父向量须声明正交归一，实际父基底和分区记录保留在端点 metadata。

```python
projected_points = PointBasis(point_handle, highmode_basis=high)
result = generator.calc(
    source_time,
    src=projected_points,
    sink=[VectorBasis(vector_sink)],
)[0]  # HSV 或 HSD，取决于实际 vector endpoint
```

固定点池时比较直接 H 源、共享 V/D 解场后取 H 汇、以及用完整父传播子事后派生交叉块的成本。后者需要对应的完整传播子与实际 `U†SV`/`U†SU` 辅助块；不得因需要 H 自动补算父传播子。HSV/HSD 保留全有向时间覆盖；只有同一实际端点的 HSH 可按自反产品使用 Gamma5 时间三角。完整路线与验证边界见[H 定义与路线选择](../../../../docs/modules/perambulator/point_highmode.md#h-contract)和[三分区闭环](../../../../docs/modules/perambulator/three_sector_production.md)。

## Gamma5 反向与保存

直接计算 VSP/DSP 不依赖 Gamma5 Hermiticity。若生成器所用后端声明了 `gamma5` 与 `gamma5_hermitian=True`，`PropagatorResult.reverse(t_sink)` 可反向一个时间对；不传时间时返回每个原汇时间对应的反向 block 列表。反向会交换物理源汇时间和实际 endpoint，并按 Gamma5 变换自旋及完整端点轴，不是整行转置。

`PropagatorWriter` 可把 VSP、DSV、DSP 转存为对应保存代表 PSV、VSD、PSD；`PropagatorReader` 可从六类保存产品提供九类读取视图。PSP/HSH 的时间三角和 HSV/HSD 的有向覆盖要求见[存储说明](../../../../docs/modules/storage/data_storage.md#h-storage)及[PR L2](../../../../docs/modules/perambulator/perambulator.md#六类保存和九类读取)。

点句柄与标准 `SpatialPointSet` 坐标均使用 `(time,point,xyz)`；`SpatialPointHandle.load(cfg)` 返回 `(Nt,Np,3)` 的只读 `int64` 数组。使用 `SpatialPointHandle` adapters 保留真实坐标、有序来源与 compact 时间映射，不按点数或 seed 推断点身份。随机空间取样权重属于最终估计量，不在传播子中乘体积或 `Np`。
