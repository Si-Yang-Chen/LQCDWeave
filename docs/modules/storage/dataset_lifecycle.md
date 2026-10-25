# Dataset 读写与发布

持久布局和默认格式见[数据保存格式](data_storage.md)。基本约定是一份共享 `dataset.json`，每个 cfg 一份完整 NPY 或 HDF5 文件。生产中的临时状态存于 `.state`；它不等于已登记的完整数据集。

## 数组契约与身份

`DatasetSpec` 描述 kind、系综与 cfg、格点、具名有序轴、dtype、科学定义、约定、父数据及已知的基底/点集绑定。它是科学定义，不包含目录路径或后端布局。坐标顺序有物理含义；不能仅凭 shape、cfg 文件名或求解参数断言输入兼容。

`data_id` 识别实际数据集。派生数据的父身份和模态/点标签按真实有序坐标登记。provenance 记录生产或导入来源；缺少的科学身份和来源信息保持 unknown。登记 unknown 不会让严格测量自动接受它。

数据集完成与科学验证是两个状态。只有首轴覆盖完整且登记成功，reader 才把数据作为已完成数据提供。科学验证状态不会从文件形状或发布动作推断。

## 写入和读取

```python
from lqcdweave.contracts.dataset import DatasetSpec
from lqcdweave.storage.dataset import DatasetWriter, DatasetReader

spec = DatasetSpec(document)
with DatasetWriter(output_dir, spec, provenance=provenance) as writer:
    for slot in writer.missing:
        writer.write(slot, produce_row(slot))
    data_id = writer.publish()

with DatasetReader(output_dir, cfg=cfg) as reader:
    row = reader.read(time=physical_time)
```

默认 writer 为每 cfg 写一个完整 NPY。多字段数值结果使用 HDF5；字段按明确路径保存各自的 shape 和 dtype。`save_correlator`/`load_correlator` 是 Correlator 结构结果的高层接口。使用方式和格式选择见[数据保存格式](data_storage.md)。

发布沿用 v1 的公共元数据与逐 cfg 差异格式。storage 在目录登记锁内读取压缩表，仅比较公共记录与当前新记录；公共记录缩小时，将移出的部分叠加到旧 cfg 的差异上。已发布 cfg 仍不可覆盖，登记写入失败仍将数据文件移回暂存位置。比较保留 JSON 类型与负零，不反复序列化相同子树。

登记锁按 10–100 毫秒逐步退避，最多等待 60 秒，容纳多个亚秒登记排队；不自动回收进程留下的锁。Windows 原子替换碰到读句柄短暂占用时重试至多 1 秒，其它写入错误直接传播。完整表重建只用于显式整体改写；指定 cfg 的读取和正常追加不恢复其它 cfg。等待竞争与登记自身耗时分别测量，见[登记基准](../../../benchmarks/README.md#dataset-registration)。

reader 按 `DatasetSpec` 的具名坐标读取，不从轴名或下标推断科学含义。HDF5 字段通过 `read_field(path, selection=...)` 读取。reader/view 返回的数据由调用方在 reader 关闭前消费或复制。裸 NPY 和旧时间切片 NPY 的读取兼容仍可用于显式输入，详见[已有数据读取](../data_readers/data_readers.md)。

## 复用与依赖图

`DatasetIndex` 与 `DatasetRequest` 可用于已登记数据的兼容查询和多阶段依赖规划。规划按有序坐标、实际父身份、definition 和 conventions 判断复用；不从文件名猜测身份，也不拼接多个数据集来制造一份更完整的输入。首轴部分覆盖只报告已覆盖和待计算位置，不能作为完整请求直接消费。

流程 可通过 `run_dataset_requests` 将显式的 `DatasetRequest` 图、输出目录和 producer 连接起来。producer 接收已解析的 `DatasetSpec`、只读输入视图和 writer，并写入缺少的首轴槽。数值算法仍由调用方选择。依赖图与坐标子视图的科学匹配规则见[数据集规划](../dataset_planning/dataset_planning.md)。

## 项目回执

项目回执保存显式请求、解析后的定义、依赖关系和实际 `data_id`，用于后续将新需求图与已保存输入关联。回执是项目引用记录，不替代 `dataset.json`，也不证明数据数值通过了科学验证。项目级查询与序列入口见[数据集规划](../dataset_planning/dataset_planning.md)。

本文只规定默认文件格式、科学身份与基本读写/复用边界。高级恢复、managed retention 和跨进程清理的行为及入口需在实现与验收核对后另行说明。
