# 保存与持久交接

关联函数测量中的 elemental 由[后端自动准备](../../../../docs/modules/correlator/managed_elemental.md)选择生产服务、检查覆盖并保存，用户不选择补产步骤或回调。独立 elemental 生产使用主 skill 的生成器入口，由代理编写脚本组合生成器与公共存储服务；本页说明其保存交接，也供内部开发参考。数值结果、写出的文件和验证发布的产物属于不同阶段。

## 直接保存数值结果

任务仅需数组/文件时，调用方可使用已有文件格式和 reader 保存每个实际时间的结果，并同步保存配方、轴、时间、动量、实际基底及规范场来源。借用结果及时消费；GPU 数据采用明确的同步主机转移或已有支持设备数组的保存服务。不将普通文件自动标成已登记完成产物。

初始化的 `write(component,momentum,sector,slices,value)` 只交付数值，不管理输出路径或 cfg 完成登记。介子 slices 覆盖该 component/momentum 的整个密集结果；P/P 为 `()`。重子 slices 是计算中的三腿模态块。保存回调按 cfg 将完整数组写入一个 NPY，或将异构字段写入一个 HDF5；共享根 `dataset.json` 登记 cfg。计算 slice 粒度不决定持久文件布局。格式细节见[默认数据格式](../../../../docs/modules/storage/data_storage.md)。

## 显式计算与保存范例

下面使用已打开的实际 `gauge_data` 和 `basis_data`，保存 raw V/V 空间路径结果。
准备函数只构造计算器和科学定义，不求解、不保存；脚本决定实际计算范围：

```python
from lqcdweave.production.elemental.planning import prepare_elemental_stage
from lqcdweave.storage.dataset import DatasetWriter

calculator, spec, provenance, physical_times, cfg = prepare_elemental_stage(
    gauge_data, basis_data, paths=paths, momentum_list=momenta,
    usedNe=Ne, phase_convention="anchor",
)
calculator.load(cfg["configuration"])
with DatasetWriter(output_directory, spec, provenance=provenance) as writer:
    for slot, t in enumerate(physical_times):
        value = calculator.calc_path_blocks(t)["v2v"]
        writer.write(slot, value)
        del value
    data_id = writer.publish()
```

直接投影表使用主 skill 的 `ElementalGenerator` 范例和相应 projected spec，不能套用 raw spec。
点端和有向流同样显式调用所需 `calc_path_blocks(t)`，再保存返回结果；导数改用
`calc_derivative(t)`。V/P、P/V 为密集数组；P/P 需按选定格式保存稀疏块的
`rows/columns/values/point_shape`，不能把它当密集数组交给 writer。重子使用
`BaryonElementalGenerator.calc_path_blocks(t)` 或同步数值块回调，保存仍由脚本完成。
GPU 数组先由脚本显式同步转到主机；借用结果在下一次计算前写完。

已有 raw-path 数据的线性组合也由脚本分开执行；`combinations` 是按实际 raw component
顺序解析的复系数/位置列表，`projected_spec` 由同一 raw 父项和配方构造：

```python
import numpy as np
from lqcdweave.production.elemental.api import re_combine

with DatasetWriter(projected_directory, projected_spec, provenance=provenance) as writer:
    for slot, time in enumerate(raw_times):
        raw_values = np.stack([raw_reader.read(time=time, component=c) for c in components])
        projected = re_combine(raw_values, combinations, 0)
        writer.write(slot, projected)
        del raw_values, projected
    projected_id = writer.publish()
```

这里 `raw_times` 使用 raw reader 的实际存储坐标，紧凑时间为槽；输出登记保留原物理时间映射。
独立基础产物没有计算并保存的库入口。关联函数内部的受管 elemental 自动准备仍保留。

## 需要登记、复用或恢复

从实际科学定义构造 `contracts.dataset.DatasetSpec`，通过 `storage.dataset.DatasetWriter` 查询缺失的首轴槽、写入并发布 cfg 数据；输入 reader、产物依赖及执行生命周期沿用既有 adapters、planning、runtime。核对当前 writer 的参数、来源记录和发布验证要求，不以 shape、seed 或文件名捏造实际身份。默认格式见[数据保存说明](../../../../docs/modules/storage/data_storage.md)。

介子空间表直接投影、未保存 raw-path 数组时，`elemental_projected` 使用
`definition.method="direct_calc_projected"`，登记完整 rows、实际规范场/基底文件 SHA256、
空间 stout 记录、边界及实际模式轴父身份；不虚构 raw 父项。字段及常规 Dataset/meson55
读取见[直接投影保存](../../../../docs/modules/operators/meson55.md#保存定义)。已有 raw 数据线性组合仍沿用原 raw 父项契约。

数值回调可能一次计算部分时间或扇区，而 writer 按首轴槽写入同一 cfg 文件。核对输出轴、槽位和完整性条件，必要时使用已有适配服务；若服务尚未支持所需时间/扇区/配方，明确需要补齐的交接并验证消费者，不能只拼出 writer 调用便宣称持久链可用。

普通全时间空间介子/流及受支持目录的 VVV 重子，由关联函数工作流中的 `Elemental` 声明自动准备，支持范围见[受管 elemental](../../../../docs/modules/correlator/managed_elemental.md)。这类测量不要求用户组合生产器、writer 或 runtime；独立基础生产由代理按本页接线，不能把内部准备函数再列为普通测量的任选入口。

已有登记规范场/基底的空间 V/V 可用 `prepare_elemental_stage(gauge_data, basis_data, array_module=cp, ...)`，
返回 generator、spec、provenance、物理 times 和 cfg。默认仍为 NumPy；显式 CuPy 原样进入数值 kernel，
provenance 记录选定后端。计算后传给主机 `DatasetWriter.write` 的值显式使用
`cp.asnumpy(generator.calc_path_blocks(t)["v2v"])`，并在下一次计算前完成写入。
这个参数只补齐该 V/V 输入准备入口；其它扇区同样由脚本显式选择后端和保存方式。
内部P/P复系数路径合并在设备端分别累加实部与虚部，兼容CuPy不支持复数scatter-add的
CUDA/HIP实现；重复点对仍求和，不能直接覆盖。
数值支持、设备编译、完整生产和消费者衔接分别验收；不从旧课题或历史作业继承本次修改的设备通过结论。

相关现有位置（相对仓库根目录）：

| 职责 | 当前位置 |
|---|---|
| 数值前端及选项 | `src/lqcdweave/production/elemental/api.py`；`docs/modules/operators/elemental_frontend.md` |
| 产物定义、写入、读取 | `src/lqcdweave/contracts/dataset.py`；`src/lqcdweave/storage/dataset.py` |
| 生命周期及请求执行 | `src/lqcdweave/runtime/dataset_stage.py`；`src/lqcdweave/runtime/scheduler.py` |
| 输入准备与只读产品规划 | `src/lqcdweave/production/elemental/planning.py`；`src/lqcdweave/production/elemental/baryon_planning.py`；`docs/modules/elemental/elemental_production.md` |
| 已有产品到消费接口 | `src/lqcdweave/adapters/`；`docs/modules/binding/binding.md` |

这些位置用于定位配套服务；关联函数测量使用 `calculate` 管理自动补产，独立生产脚本组合主 skill 的生成器与公共存储服务。准备函数与持久 planner 的时间坐标和槽映射有各自契约；接线按 selection 对应的物理时间计算、用 `Nt` 槽写入并登记同一 selection。不能把稀疏槽标成完整物理时间产物。简单数组为完整 cfg NPY，多字段为 HDF5。

## 显式稀疏锚点的公共辅助入口

普通全时间调用不变。需要时间稀疏化时，prepare_elemental_stage、plan_indexed_point_elementals、plan_indexed_directed_current 和 plan_indexed_baryon_elementals 可传 `selection=selection`。输出具名 time 轴为 `0..Nt-1`，保存实际物理映射；规范场保持完整 Lt，父基底与点产物保留各自真实时间顺序及身份。

独立生产脚本从已保存的 selection 取得槽对应的物理锚点，显式调用数值核并保存；跨时间端点在各父输入映射中独立查找。此处的稀疏锚点重子交接由公开 kernel 和调用方 writer 组合，不能从全时间受管 VVV 推断稀疏时间 runner 已接通。缺少有效物理端点时拒绝生产；未支持的跨时间收缩不能从 Nt 槽邻接猜测。

## 完成与恢复

恢复只复用科学定义、实际父项和覆盖一致且通过验证的 cfg/槽位；新定义使用新输出位置。输出检查须覆盖真实时间、component/动量顺序、稀疏点坐标与数值布局。数据集登记与消费者重读按任务需要验证，不能以进程成功或目录存在代替发布完成。

写回能否降低峰值由实际分配与交付顺序决定。分别核对规范场/基底常驻、运输缓存、raw 临时量、组合结果、共轭、模态暂存及主机转移；数值支持 GPU 不代表持久预算或消费者已通过 GPU 验收。


确需跨多个已有产物图续算时用[有限项目恢复](../../../../docs/modules/dataset_planning/dataset_planning.md#project-resume)。普通数组保存不引入项目回执；回执历史与受管缓存清理只在该任务明确需要时读取，恢复成功不自动删除数据。
