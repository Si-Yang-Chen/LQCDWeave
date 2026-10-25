# 已有传播子的增量复用

## 历史源时间行的三角区归档

`tools/campaigns/archive_perambulator_triangle.py` 是限定历史数据的迁移工具，不参与
新传播子的求解。它消费明确指定的 cfg、完整 source/rank 路径清单、原始 manifest 和
输入元数据，使用已有 `propagator_pair_spec`、`DatasetWriter` 保存 `t_sink >= t_source`
的物理时间对。每个产品每 cfg 一个 complex128 NPY，由共享 `dataset.json` 登记；
写数值前使用 POSIX 分配实际磁盘块。VSV、PSP 的对角时间块完整保存，PSV 不裁剪。

迁移按 cfg 串行进行。该 cfg 的两种产品均写出后，重新打开原始文件和已发布 Dataset，
检查全部时间对：保留区比较复数的原始位模式，省略区通过原生 `PropagatorReader`
的 Gamma5 反向操作逐项对照旧数据。随后 fsync 并记录归档文件 SHA-256、data_id、
完整检查数和误差回执。仅显式启用 `--delete-verified-originals` 时，且两种产品均通过、
归档哈希未改变，才逐文件删除清单内的该 cfg 旧数值文件。迁移完成并通过数值检查后补登记。原始 manifest、
输入元数据、PSV、其它生产目录不删除。当前 Slurm 任务定义若引用旧批次，则拒绝迁移。

读取使用 `DatasetReader` 配合 `PropagatorReader`，或缩并用 `PropagatorPairReader`；
二者均支持对角产品的反向时间对。旧的 dense source/rank 文件路径不会继续提供完整行，
历史脚本需切换到归档读取器。归档中的 legacy 输入身份保留旧来源记录，不宣称与另一套
相同维数的 V 基底等价。数值恢复受旧求解容差影响，并非被删除块的逐位无损压缩。

中断后沿用同一迁移配置和输出根目录：已发布产品重新验证，有完整验证回执时重新核对
哈希后继续清理。失败不会删除尚未通过双产品验证的 cfg。迁移工具应作为独立 CPU 作业
运行；配置、任务路径、实际容差与回执放在对应 campaign，不作为通用测量默认。

复用按实际产品、端点和物理时间对合并读取覆盖。已有传播子与新计算结果共同交给
`PropagatorReader`；缺失覆盖由调用方显式补算。原始幅度保持不变，重叠结果不能直接相加。
作用量、规范场、自旋基、Gamma5 与实际 V/D/P 身份必须一致。不同质量、不同随机 D 实现，
或者同模态数但不同相位/顺序的 V，不能仅凭数组大小视为同一端点。

## 显式接入源时间行

对于 VSV、DSD、PSP，reader 也可通过已声明的 Gamma5 Hermiticity 从反向物理时间对
读取同产品，交换实际端点；`coverage` 与 `missing` 计入这些可派生时间对。
例如相同实际 P 端点的 PSP 只存 `t_sink >= t_source`，即可读取完整时间覆盖。
直接登记的时间对优先；端点身份不同的另一套数据不会因此变成可用覆盖。

`SourceTimePropagatorReader` 接受一个已确认定义的 canonical pair spec 与 `load_row(t_source)`
回调。回调返回 `(row_time, sink_spin, source_spin, ...)` 数组；解码、缓存、可信来源与数组
生命周期归调用方。adapter 不执行 pickle，不发现文件，也不求解。直接读只借用单个块，
Gamma5 转换只为该块分配结果；整个旧行的读取内存仍取决于回调。

```python
from lqcdweave.adapters.perambulator_rows import SourceTimePropagatorReader
from lqcdweave.production.perambulator.api import PropagatorReader

legacy = SourceTimePropagatorReader(
    vsv_spec, load_vsv_row, product="VSV", source_times=old_source_times,
    row_times=old_row_times, time_convention="absolute", data_id=old_product_id,
)
access = PropagatorReader([legacy, *our_existing_readers])
missing = access.missing("VSV", requested_pairs, sink=v_sink, source=v_source)
```

`vsv_spec` 由 `propagator_pair_spec` 声明准确的旧覆盖及共同科学定义。
`row_times` 的顺序就是行首轴顺序：`absolute` 表示物理汇时间；`relative` 表示
`(t_sink-t_source) % Lt`。必须显式选择，不从 shape 或文件名猜测。
非规范产品 DSV/VSP/DSP 的 spec 则登记反向产品、反向时间对与交换端点。

用 `missing` 的非空结果定义一个新增 pair 数据集。只计算对应的源时间和所需产品，
将结果中所需物理时间对交给 `PropagatorWriter`；普通生成器仍返回 dense 源时间行，
这一步减少已完成源时间的求解，尚不提供任意缺失单块的独立求解。
再次把旧 reader、自己的已发布 reader 与新增 reader 一起接入，下一次同范围的
`missing` 应为空。新数据集的中断恢复继续使用通用 storage writer，不重写旧文件。

已有行与补齐产品的接线见[传播子前端 CPU 回归](../../../tests/test_propagator_frontend.py)。
这些小格点与协议检查不代表 C24P29 实际数据验收。

## 实际数据接入前需要的记录

- 我们自己的保存路径、cfg、源时间和产品覆盖。
- 两边相同的规范场与完整 Dirac 定义，边界、自旋基、求解精度及输出 dtype。
- 实际 V/D 文件和有序标签；D 的父基底、分组、realization/pool 与 raw 幅度约定。
- 两端点集 P 的实际坐标与顺序，以及旧行的绝对/相对时间记录。

数量和目录参数只能帮助定位。缺少这些记录时可以实现读取接口和小规模验证，但不能声称
两套实际生产数据已经兼容或合并。
