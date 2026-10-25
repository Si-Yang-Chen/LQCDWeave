# 两点关联函数：从完整时间到紧凑槽重建

## 默认完整时间

未显式选择时间子集时，保持全部物理时间片。普通 `calculate(time_axes=...)` 按声明顺序返回源×汇坐标，不自动平均、不自动启用时间稀疏化。保存原始结果使用 `save_correlator(path, calculation)`；需要平均时明确源轴、汇轴、周期边界和物理 Lt。

## 明确选择时间子集

同一 cfg 的生产、读取和收缩复用同一已保存 `TimeSelectionRecord`。它记录物理 Lt 和实际有序时间，允许跨周期边界；不能对记录单独排序或再次随机平移。紧凑 reader 的 `.Lt/.Nt` 是计算槽数，`time_selection.Lt` 保留物理长度，数值读取只用槽索引。

`calculate(time_axes={source: range(selection.Nt), sink: range(selection.Nt)})` 的普通两点结果为源槽×绝对汇槽。默认重建直接接受这一布局：

```python
from lqcdweave.storage.time_selection import load_time_selection
from lqcdweave.sampling.temporal.api import reconstruct_two_point_from_coordinates

selection = load_time_selection(selection_file, Lt=128)
# data.shape = (selection.Nt, selection.Nt)，依次为源槽、绝对汇槽
result = reconstruct_two_point_from_coordinates(data, selection)
values, counts = result.values, result.counts
```

每个物理 `dt=(t_sink-t_source)%Lt` 单独累计并除以实际配对数。结果长度为物理 Lt；这是一份采样平均估计，不能恢复未计算的逐源数据，也不保证单 cfg 等于全源平均。不同 cfg 使用各自的记录分别重建。

保存并平均嵌套结果不必拆开容器：

```python
from lqcdweave.contraction.correlator import save_correlator, load_correlator

save_correlator(output, calculation, average_tsrc=True,
                source_axis=0, sink_axis=1, coordinate_domain='slot',
                selection=selection, boundary='periodic')
restored = load_correlator(output)
values = restored.value
counts = restored.reduction['counts']
```

保留 dict/list/tuple、算符轴和所有字段；只消去源轴，汇轴改为物理 dt。省略 `average_tsrc` 保存原始槽结果及有序映射。已平均结果保留归约记录，禁止再次平均或声称恢复原始时间矩阵。

## 高级：轴、布局和非对称覆盖

重建的 `tsrc_axis=0, dt_axis=1` 指输入两条时间轴；历史名字 `dt_axis` 在默认绝对布局下指汇槽轴，输出才是物理 dt。其他轴可在任意位置；输出 `result.dt_axis` 给出消去源轴后的轴号。

源汇覆盖不同可使用 `reconstruct_two_point_from_coordinates(data, source_selection, sink_selection=sink_selection)`。保存器对应 `source_selection=...` 和 `sink_selection=...`，不同时再传共同 `selection`。两个记录必须有同一物理 Lt，各自维度和顺序必须匹配数据。原生同时间槽 Diagram 按每条边的源/汇角色分别核对有序选择；正反传播子必须与各自顶点对应，不能用相同 Nt 替代核验。

旧源槽×循环差槽布局必须显式声明：

```python
result = reconstruct_two_point_from_coordinates(
    old_data, selection, time_layout='cyclic_difference')
```

此时 `(i,j)` 指汇槽 `(i+j)%Nt`，j 不是物理 dt；源汇记录必须完全相同。迁移旧调用时补上这一参数，不能仅凭两个轴长度都是 Nt 来猜测布局。旧 `reconstruct_two_point(..., Lt=..., selected_times=..., tsrc_axis=..., dt_axis=...)` 保持该历史循环槽语义。

已有真正源×物理差值数据可省略 selection，显式传 `Lt, tsrc_times, dt_values`，后者形状为 `(Ndt,)` 或 `(Nsrc,Ndt)`。可选 `sink_times` 只筛选已计算的物理汇。禁止与 slot selection 混用。

## 缺失、资源和恢复

缺失任一 dt 默认报错。数组重建 `missing='nan'`、保存器 `missing='nan'` 明确保留 NaN 与零计数；不补零、插值、反射、共轭或自动修正边界相位。重建累计至少使用 float64/complex128；保存器保持浮点/复数字段 dtype。

输入不会被修改；输出与计数由调用方持有。此数组接口仍需输入和输出可放入主机内存，不宣称流式或超内存重建。数值结果使用现有 DatasetWriter 发布/校验流程，未完成产物不能加载。计划重用核对完整有序选择，不以 Nt 相同代替科学身份。

绝对槽和显式循环差槽与独立物理配对参考的 CPU 对照见[槽布局重建回归](../../../tests/sampling/temporal/test_slot_reconstruction.py)，不作为物理系综或生产规模验收。完整生产约定见[时间流程](../temporal_sampling/temporal_production.md)。
