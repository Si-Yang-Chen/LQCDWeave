# Elemental 生产脚本与通用持久化

使用压缩时间输入前，先核对[共同时间交接约定与当前接线限制](../temporal_sampling/temporal_production.md)；不从 Nt×Nt 的 shape 推断输入或重建布局。

数值计算由公开 elemental calculator 完成；生产脚本选择配方、实际时间、输出字段和共享数据集目录，再按需要组合通用 storage/runtime。简单数组默认每 cfg 一个完整 NPY，多字段结果使用 HDF5，并由根 `dataset.json` 登记；详见[数据保存格式](../storage/data_storage.md)。`production.elemental.planning` 提供输入准备和只读规划，不提供固定 elemental 生产 runner。测试拥有的显式生成/保存接线见 [elemental_script.py](../../../tests/fixtures/elemental_script.py)；它不是用户生产 runner。

## 直接计算

```python
from lqcdweave.production.elemental.api import ElementalGenerator

calculator = ElementalGenerator(
    gauge, vectors, momenta,
    [dict(component=0, path=(), coefficient=1)],
    Ne=Ne, phase_convention="anchor",
)
for time in physical_times:
    row = calculator.calc_path_blocks(time)["v2v"]
    save_or_consume(time, row)  # 同步消费，或在下次计算前复制
```

`ElementalGenerator` 接收用户选择的实际基底、动量、component 表、模态截断、sector、空间/模态块与 array backend 等选项。`calc_path_blocks(t)` 的 `t` 始终是规范场和输入基底的物理时间。返回数组可能借用 calculator 的工作区；必须在下一次计算覆盖之前同步写完或复制。

重子计算使用同一公开模块中的 `BaryonElementalGenerator`。三腿基底、三腿截断、component 表和逐腿块选择由脚本显式指定；数值入口为 `calc_path_blocks(t)`，当前输出扇区为 VVV。表格式和配方见[重子 elemental 说明](baryon_production.md)。

若输入来自 DatasetReader，可使用 `prepare_elemental_stage(gauge_data, basis_data, ..., selection=None)` 准备 calculator、输出 spec、provenance、实际物理 times 与 configuration。默认输出完整时间；显式 selection 时，spec 的 time 轴是 `0..Nt-1` 槽，返回的 times 保持记录中的物理顺序。它不打开额外输出目录、不创建 writer，也不发布；调用方掌管 reader 生命周期和写出。

## 使用通用 DatasetWriter

以下示例描述完整物理时间的 V/V raw elemental。数组按 cfg 写入同一共享目录中的完整 NPY 文件，并由 `dataset.json` 登记：

```python
from lqcdweave.storage.dataset import DatasetWriter
from lqcdweave.production.elemental.planning import prepare_elemental_stage

# gauge 与 basis 是调用方已经打开的 DatasetReader。
calculator, output_spec, provenance, physical_times, cfg = prepare_elemental_stage(
    gauge, basis, paths=paths, momentum_list=momenta,
    usedNe=Ne, phase_convention="anchor",
)
calculator.load(cfg["configuration"])
with DatasetWriter(output_path, output_spec, provenance=provenance) as writer:
    for slot in writer.missing:
        writer.write(slot, calculator.calc_path_blocks(physical_times[slot])["v2v"])
    data_id = writer.publish()
```

`DatasetWriter.write` 将一个首轴槽同步写入同一个 cfg 文件；默认 `.npy` 格式、共享登记和临时 `.state` 见[数据保存格式](../storage/data_storage.md)。默认完整物理时间。明确稀疏化时，先读取同 cfg 已保存的 selection，再给上述 `prepare_elemental_stage` 传 `selection=selection`；其返回的 spec 自动携带 Nt 槽及真实时间映射，循环仍使用返回的 physical_times。底层 `elemental_dataset_document(..., selection=None)` 同样支持显式选择，无需另造输出契约。不能把槽下标当物理时间，也不能将缺失槽补零冒充已完成覆盖；规范场及 Dirac 求解仍使用物理 Lt。高级恢复和跨进程协调须按当前实现与验收核对。详情见[共同生产与保存约定](../temporal_sampling/temporal_production.md)。

## 只读规划

以下函数只承担只读规划工作。它们不会替调用方执行完整生产事务、创建输出位置、关闭输入或发布数据集登记。

| 辅助接口 | 用途 |
|---|---|
| `plan_indexed_point_elementals(index, gauge_id, basis_id, points_id, ...)` | 根据实际 gauge、基底、点集 ID 与所选 sectors/path 建立 point elemental 计划。 |
| `plan_indexed_elemental_projection(index, raw_id, rows, ...)` | 规划从一个已登记 raw 父项派生的空间路径线性组合。 |
| `plan_indexed_directed_current(index, gauge_id, basis_id, ...)` | 规划已登记规范场/基底对应的八方向 V/V 产物。 |
| `plan_indexed_baryon_elementals(index, gauge_id, basis_ids, ...)` | 规划 raw_paths、projected 或 distance 重子产品；数值计算由 `BaryonElementalGenerator` 执行。 |

Planner 输出和接受的配方由当前函数签名与对应契约定义。计划不执行计算或保存；脚本直接调用数值接口和 `DatasetWriter.write`。需要共享登记与消费者验证时，由用户脚本显式使用通用 runtime/storage 接口；高级恢复与 retention 需按当前代码和验收范围核对。

## 输入身份与时间覆盖

已登记输入必须携带实际产物 ID、组态、格点、具名轴与可用时间覆盖。生产脚本不能用 seed、文件名或 shape 替代基底、点集或规范场身份。行写入辅助会将 writer definition 与实际输入绑定，避免把其他输入的数值写入当前产物。

`prepare_elemental_stage`、point/current/baryon planner 的 `selection=None` 默认完整时间，显式选择则建立保存实际时间映射的紧凑输出。point/current row writer 直接读取 writer 契约中的映射，将缺失槽转成物理锚点，再在各父基底或点集的独立映射里查找真实端点。父输入可比锚点覆盖更大，也可有不同顺序；有效路径缺少端点时明确报错，槽相邻不代表物理相邻。投影派生保持原始数据集的时间映射；中断后的恢复行为以当前 storage/runtime 实现和验收为准。

模块的数值实现见 [`production/elemental`](../../../src/lqcdweave/production/elemental)。接口边界与模型细节见[Elemental production 模块说明](elemental_production.md)、[元素计算前端](../operators/elemental_frontend.md)和[dataset 生命周期](../storage/dataset_lifecycle.md)。
