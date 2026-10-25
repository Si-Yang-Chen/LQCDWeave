---
name: lqcdweave-spatial-sampling
description: "为明确需要随机空间取样的 LQCD 测量生成实际有序点集，并按用户选择的格式保存和读取坐标；已有确定点集直接消费。"
---

# LQCDweave · 空间取样与实际点集

共同任务路由、前端边界、保存/身份及执行范围统一见[技能共同约定](../README.md)。只读取当前任务的条件参考。

多 cfg 的任务分配、循环前共同准备及完成核验统一按 [Dispatch skill](../lqcdweave-dispatch/SKILL.md)；循环内生成或读取当前 cfg 的实际点集，本 skill 负责取样策略、计权与坐标约定。

只读取当前任务需要的参考；不要求通读其它产品或资源模块。

## 参数与执行范围

核对格点、实际点集和顺序、每片 `Np`、temporal_policy 段长与点集共享策略、点池共用关系及估计权重。seed 描述随机规则，不能代替实际坐标。

核对实际科学参数与[执行范围](../README.md#execution-scope)，沿用已有授权。

用户明确选择随机空间取样时使用，包括流或其它适用的点端测量。已有确定坐标直接读取或消费，不自动重新抽样；纯模态测量不引入点端。具体消费者能否支持该图及估计量另行核对。

唯一采样入口是 `sample_spatial_points(SpatialSamplingDesign(...), seed=..., selection=...)`，返回 `SpatialPointSet`；所有坐标都采用 `(time,point,xyz)`，`coordinates` 为只读 `int64`，并保留实际坐标、设计和池身份。已有坐标可直接构造 `SpatialPointSet(lattice_size=lattice, times=tuple(range(Lt)), coordinates=coords)`；`SpatialPointHandle` 用于读取或适配已有数据，它不是采样入口。

给定格点 `(Lx,Ly,Lz,Lt)`、每片 `Np`、取样策略和 seed 时，使用数值 sampler：

```python
from lqcdweave.sampling.spatial.api import (
    SpatialSamplingDesign, sample_spatial_points,
)

design = SpatialSamplingDesign(lattice, Np, temporal_policy=1)
points = sample_spatial_points(design, seed=seed)
coordinates = points.coordinates
```

计算完成后再显式保存实际点集，不调用内置生产流程。下面的 `configuration` 为实际
cfg 的科学记录，输出保留刚才生成的坐标、顺序与采样身份：

```python
from lqcdweave.planning.spatial_points import validate_complete_spatial_point_set
from lqcdweave.runtime.provenance import local_provenance
from lqcdweave.storage.dataset import DatasetWriter

spec = validate_complete_spatial_point_set(points, configuration=configuration)
with DatasetWriter(output_directory, spec, provenance=local_provenance()) as writer:
    for slot, row in enumerate(points.coordinates):
        writer.write(slot, row)
    data_id = writer.publish()
```

显式稀疏时间时，sampler 和 spec builder 同时传入同一 `selection`。
通用 runtime 不再内置“采样并保存”的空间点集生产桥；来源策略仍按实际来源记录判断。

`temporal_policy=b` 指定 1 到 `Lt` 的正整数段长，默认 `1` 逐时间片独立抽样，`Lt` 在全部时间片复用同一有序点集；段内相同、段间独立。按物理时间从 0 分段，末段不足 b 时保留短段。旧字符串 `"independent"` 和 `"same"` 分别等价于 1 和 Lt，继续兼容。sampler 默认返回完整 `Lt` 行，不自动稀疏时间；每片均匀无放回抽样且 `Np` 不超过空间体积。坐标布局为 `(time,point,xyz)`，`xyz=(x,y,z)`。sampler 支持 `seed=None`；使用熵时应保存实际坐标和顺序以供复用。

需要保存时，简单坐标数组默认按 cfg 写为完整 NPY，多字段数据用 HDF5，由共享根 `dataset.json` 登记公共 metadata 和 cfg 差异；同时保留物理时间顺序、点顺序、实际坐标和必要的采样元数据。读取时消费已保存的实际坐标。sampler、DatasetSpec builder 和只读 `plan_indexed_spatial_points` 都不执行文件写出；design/seed 也不证明生产来源已知。来源策略根据实际登记的来源记录判断，fresh 请求或已有 unknown 数据不能仅凭抽样意图变成 known。通用 runtime 来源服务仍可由上层按需组合；没有空间点集专用发布/恢复 流程。格式细节见[数据保存说明](../../../docs/modules/storage/data_storage.md)。

## 联合时间稀疏化

仅在用户明确要求时，先调用 load_time_selection(selection_file, Lt=Lt) 读取同 cfg 实际记录；不重新选择或平移。唯一通过 `sample_spatial_points(design, seed=seed, selection=selection)` 采样，按记录原顺序得到 Nt 行，保存相同映射。段归属按实际物理时间计算，不按 selection 槽号重新分段。仍消费全部物理段的 RNG 序列，使固定 seed 的选中行与同策略的 full 参考逐项相同；selection 减少常驻输出，不减少抽样次数。规范场和格点长度保持物理 Lt。

SpatialPointHandle(points, cfg=cfg) 保留 points.times；compact NPY 用 from_npy(..., time_selection=selection)。PR 传相应 source_selection/sink_selection，输出直接为所选汇时间槽。进一步需要身份/恢复时，sampled_spatial_points_spec 与 validate_complete_spatial_point_set 的 selection 参数保留物理格点和时间映射，复用原 writer/reader；不能把 Nt 行声明为全物理覆盖。遵守[共同生产与保存约定](../../../docs/modules/temporal_sampling/temporal_production.md)。

实际坐标和顺序必须随数据保留；相同 seed 或 `Np` 不证明文件包含相同取样点。导入数据不编造采样策略或点池身份。

采样模块不自动乘权。线性单点空间求和可使用体积/`Np`；同池点乘积、不同参数共享取样及跨时间项需按相应联合包含概率核验，不能统一套用一个因子。分段固定时段内时间共享点集，跨段独立；点池身份仍由实际数据绑定确认。实际计算交给适用的绑定和采样展开流程，避免重复计权。

PR 使用 `adapters.spatial_points.SpatialPointHandle(points, cfg=cfg)`，或 `SpatialPointHandle.from_npy(path, lattice_size=lattice, cfg=cfg)`；通用完整产物用 `SpatialPointHandle.from_dataset(directory, cfg=cfg)`，登记中只有一个 cfg 时可省略 cfg。所有输入和输出坐标均为 `(time,point,xyz)`；handle 的 `.load(cfg)` 返回只读 `(Nt,Np,3)` `int64` 数组。裸 NPY 不能保存 design/seed/algorithm；新保存的登记数据集使用每 cfg 整块 NPY 或 HDF5，并在根 `dataset.json` 登记公共 metadata 与 cfg 信息。来源 unknown 不升级。EL 直接用对象或规范布局数组。精确接口与保存示例见[空间说明](../../../docs/modules/spatial_sampling/spatial_sampling.md)和[默认格式说明](../../../docs/modules/storage/data_storage.md)。

需要 elemental 或传播子的点端数据时按需读取对应 skill 的 point-flow 说明。
