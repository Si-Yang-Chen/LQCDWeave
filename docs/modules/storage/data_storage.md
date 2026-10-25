# 数据保存格式

默认的持久格式是一份共享登记文件，加上每个 cfg 一份完整数据文件：

```text
perambulator/
  dataset.json
  2000.npy
  2001.npy

correlator/
  dataset.json
  2000.h5
  2001.h5
```

公共科学元数据在 `dataset.json` 中登记一次，各 cfg 的实际身份、文件、来源和特有信息按配置记录。配置名来自 `DatasetSpec` 的 `configuration.configuration`，读取时显式选择 cfg。目录名和文件名不构成科学身份。

多任务可分别生产不同 cfg。每 cfg 的写入锁只保护该 cfg 的数据；发布时短暂获取目录登记锁，增量加入当前记录。公共信息不变时保留已有压缩记录；公共信息缩小时只调整受影响的差异，不展开所有旧 cfg。登记仍以原子替换和同步落盘更新，同版本旧登记文件无需迁移。读取指定 cfg 也只恢复这一份元数据。

一个 cfg 对应一个完整数组时默认用 NPY。NPY 保留整块数组的 shape 和 dtype，可由读取器按需映射与切片。多字段数据或需要 HDF5 分块读写的数据使用 HDF5，各字段保留自己的 shape 和 dtype。Correlator 结构结果和完整 cfg 结果使用 HDF5 原生数值字段；GPU 可以在传输期间打包数据，磁盘文件仍保存原生字段。

`cpu` 和 `correlator` 安装包含 HDF5 依赖；最小安装按需加选 `hdf5`，命令见[安装说明](../../installation.md)。

## 保存与读取整块 NPY

```python
from lqcdweave.contracts.dataset import DatasetSpec
from lqcdweave.storage.dataset import DatasetWriter, DatasetReader

spec = DatasetSpec(document)  # axes、dtype、configuration 与科学约定均显式给出
with DatasetWriter(output_dir, spec, provenance=provenance) as writer:
    for slot in writer.missing:
        writer.write(slot, make_array_for_slot(slot))
    data_id = writer.publish()

with DatasetReader(output_dir, cfg=cfg) as reader:
    values = reader.read(time=physical_time)
```

`DatasetWriter` 默认写 `{cfg}.npy`。`write(slot, array)` 将一个首轴槽写入同一个完整数组文件；已发布数据按 cfg 登记，完整发布后临时生产状态清除。读取时传入共享目录和 cfg；`read` 使用具名坐标返回一个首轴切片，首轴坐标仍按 `DatasetSpec` 中的物理值解释。

需要保存多字段结果时，使用提供的结果入口；例如 `save_correlator(directory, value, configuration=cfg)` 和 `load_correlator(directory, configuration=cfg)` 以 HDF5 保存和读取结构化数值结果。底层 HDF5 writer 支持具名字段及各自的 shape/dtype。字段路径、轴和坐标仍由科学定义明确声明。

## 登记与科学身份

`DatasetSpec` 记录数据类型、组态与格点、具名且有序的轴坐标、dtype、定义、约定和已知父数据。`data_id` 标识一次实际数据；模式或点轴的绑定保存实际 `data_id` 与坐标顺序。相同文件尺寸、shape、cfg 名或求解参数都不能证明数据科学兼容。

未知的基底、点集、规范场版本或来源保持未知。导入文件时只登记调用方明确提供的科学定义与来源信息；未提供的信息不会从文件名、随机 seed 或数组形状推断。严格测量仍按实际有序身份和模块约定检查输入。

旧的裸 NPY、传播子 NPY 和时间切片 NPY 读取方式继续支持，供已有输入显式导入或消费。它们不改变新产物默认每 cfg 一个整块 NPY/HDF5 文件、并由 `dataset.json` 统一登记的格式。

生产过程中的临时状态位于 `.state`，不作为已完成数据集对外读取；成功发布后清除。恢复与项目级索引的高级行为应以相应公开接口和验收为准，本页只定义数据布局与基本读写入口。

## V/P/D 传播子的六类保存与九类读取

[PR V/P/D 统一模块](../perambulator/perambulator.md#vpd-target)将直接生产产品与持久代表分开：
`VSV, PSV, PSP, VSD, PSD, DSD` 六类可直接保存；`VSP, DSV, DSP` 可由反向的
`PSV, VSD, PSD` 按 Gamma5 Hermiticity 读取。调用方必须声明对应作用量的 Gamma5 能力，
并提供物理时间对、实际端点和有序标签完全匹配的反向数据。

`propagator_pair_spec(...)` 创建 `DatasetSpec(kind="perambulator_pairs")`，编码
`pr-six-pairs-v1`。首轴是 `time_pair`，每项坐标显式为 `[sink_time, source_time]`；每个首轴
槽存一个完整的单时间对数组。此布局不是旧 indexed planner 的 `source_time` 行及整条汇时间
数组。`PropagatorWriter` 把实际 `PropagatorBlock` 规范化后交给调用方提供的通用
`DatasetWriter`；`PropagatorReader` 从调用方提供的 readers 返回九类 block 视图。实际
反向转换会交换物理时间和整组端点，不只改产品名或数组轴。通常的共享 `DatasetWriter`/
`DatasetReader` 仍负责文件 IO 与覆盖；PR 调用方管理 reader/writer 生命周期、发布及恢复。

D 端点记录必须保存实际 D `data_id`、向量索引及 `vector_labels`、`mode_groups`、
`parent_basis_id`、`realization_id`、`noise_pool_id` 与 `normalization="raw"`。PR 不对原始
D 幅度施加权重或归一化。旧 V/P 文件仍按其既有格式读取；新 pair schema 是显式新增的存储
定义，不会改写已有数据。`plan_indexed_perambulators` 仅预览旧 V/P 源时间行布局；它不支持 D 及该 `time_pair` 布局。
源时间行由脚本调用 `calc` 后交给 `DatasetWriter.write`，pair writer 单独处理已有结果的实际 endpoint 和物理时间对。

<a id="h-storage"></a>
## H 产品、时间三角与实际占用

H 的[定义](../perambulator/point_highmode.md#h-contract)保存在 P 端点的 `metadata.high_mode_projection`；
实际父基底 ID 同时进入 parents。HSV/HSD/HSH 分别沿用 PSV/PSD/PSP schema，
点标签、全颜色轴、dtype 和物理时间对布局不变；raw P 与 H 的缓存和数据身份必须区分。

Gamma5 恢复满足 `A† S(t,s) B = gamma5 [B† S(s,t) A]† gamma5`，同时交换端点、
物理时间及投影记录。VSV、DSD，以及两端点池和投影定义相同的 HSH，可保存
`t_sink >= t_source`，含时间对角线。Lt 个时间对应 Lt(Lt+1)/2 对；不同两端定义不适用。
HSV 的半三角只能恢复 VSH 的反向半三角，不能恢复 HSV 的另一半；HSD/VSD 同理。
因此交叉产品的全时间覆盖仍需 Lt² 对，反向产品由 reader 恢复，不另存一份。

载荷预算为 `Npair × 16 × d_sink × d_source × complex_itemsize` 字节，16 是自旋对数，
全颜色点端 `d=3Np`，V/D 端分别为实际 Nv/Nd。另计文件头、元数据、临时文件和并行任务
尚未分配的输出；软链接入口不重复计费。若项目要求实际预分配磁盘块，须另行核验分配策略，不能仅由通用 NPY 文件长度推断。
预分配大小不表示完成度；进度读取 `.state` 的完成行或源时间回执，完整发布后才供常规消费。
