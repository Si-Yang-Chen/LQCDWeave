# 已有数据上的介子两点收缩

本模块提供介子行计算和只读规划，不提供固定的持久化生产入口。调用者按当前数据、算符和输出需求组合输入读取、`prepare_meson_stage` / `write_meson_rows`、通用 `DatasetWriter`，或直接使用 `contraction.correlator.calculate`。简单数组默认每 cfg 一个完整 NPY，多字段结果使用 HDF5；公共 metadata 与 cfg 登记写入共享根 `dataset.json`。输入生产、输出目录和字段选择由调用脚本明确决定；格式细节见[数据保存说明](../storage/data_storage.md)。

## 数据契约

`DatasetSpec` 保存数组的稳定科学契约：数据类型、系综/组态/格点、具名有序坐标、dtype、实际基底和点父项、物理定义、约定及已知来源。`data_id` 表示实际数据身份；相同 shape、seed 或 Ne 不足以证明实际数据兼容。文件位置不参与科学身份，移动文件时仍须保持共享登记有效。

介子收缩接受完整物理时间和 DeGrand–Rossi 自旋轴。VSV 时间可按汇优先或源优先存储；elemental 支持原始空间路径布局。实际基底、模态标签、动量和时间顺序必须按坐标绑定，缺失身份不能由文件名推断。

输入可用 `DatasetReader` 或 `DatasetIndex.open` 读取。读取结果在 reader 生命周期内是借用视图，写入前需同步消费或复制。`DatasetWriter` 按输出 `DatasetSpec` 检查每个首轴槽；槽数据写入同一个完整 cfg 文件，完整覆盖后再登记为完成。逐槽计算回调不对应独立持久文件。

## 计算约定

```text
C(s,t) = -Tr[Phi_sink(t) tau_a(t,s) Phi_source(s) tau_b(s,t)]
```

源顶点由 `Meson(source=True)` 按既有 dagger 约定构造：自旋取 `gamma4 Gamma† gamma4`，空间矩阵取厄米共轭，系数取复共轭；收缩核不再次共轭源顶点。反向传播子必须来自明确输入，不能从正向数据推断 gamma5 对称性。结果坐标仍是绝对 `(source_time,sink_time)`。

公开共享写出函数的形式如下：

```python
from lqcdweave.contraction.meson_measurement import prepare_meson_stage, write_meson_rows

prepared = prepare_meson_stage(opened_datasets, sink_operator, source_operator)
# 调用者依据自己的输出契约创建 DatasetWriter，并管理 reader/writer 生命周期。
write_meson_rows(opened_datasets, sink_operator, source_operator, writer)
```

`write_meson_rows` 只同步写入 `writer.missing`，不关闭借用输入，也不单独发布数据集登记。使用 `calculate` 时，计算和结果结构由相关器 API 决定；若要持久化，按所需坐标与输出 recipe 另行组合通用存储接口。没有固定恢复或自动补数据流程。

## 验证范围

`tests/test_dataset_流程.py` 和 `tests/test_operator_dataset_reuse.py` 覆盖独立全空间迹、复系数、动量、dagger、重排时间、身份拒绝和 raw 路径重组。当前格式的完整 cfg 文件与共享登记由 storage 测试验证；高级恢复、锁和跨进程生命周期能力需另按实现验收，不代表介子模块提供固定运行器。
