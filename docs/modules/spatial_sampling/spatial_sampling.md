# 空间稀疏化

## 保留实际点集与采样记录

使用 `SpatialSamplingDesign` 指定格点尺寸、每个时间片的点数和采样策略，
统一调用 `sample_spatial_points` 获得保留实际坐标、时间顺序、采样记录和池身份的 `SpatialPointSet`。
所有空间点坐标统一采用 `(time,point,xyz)` 布局；已有坐标直接构造成同一契约，不做轴转换。

```python
from lqcdweave.sampling.spatial.api import SpatialSamplingDesign, sample_spatial_points

design = SpatialSamplingDesign((24, 24, 24, 72), 216, temporal_policy=1)
points = sample_spatial_points(design, seed=42)
assert points.coordinates.shape == (72, 216, 3)
assert points.times == tuple(range(72))
```

- 格点尺寸顺序为 `(Lx, Ly, Lz, Lt)`，坐标分量为 `(x, y, z)`。
- 坐标布局为 `(time, point, xyz)`，`coordinates[i, p]` 对应物理时间 `times[i]` 的第 p 个点。
- 每个时间片从空间体积中均匀无放回抽取 Np 个点；必须 `1 <= Np <= Lx*Ly*Lz`。
- 正整数 `temporal_policy=b` 将物理时间从 0 起划分为 `[k*b, min((k+1)*b, Lt))`；段内复用同一有序点集，段间独立无放回抽样。`b=1` 为逐片独立，`b=Lt` 为全时间共享。
- 旧字符串 `"independent"` 与 `"same"` 仍可使用，分别等价于 `1` 与 `Lt`；设计元数据保留旧字符串原值。
- 生成入口默认覆盖全部 Lt 个时间片；显式 selection 才保留选中 Nt 行。`same` 不表示时间稀疏化。
- 点和时间顺序定义基底，不自动排序。点集对象复制输入并保存为只读 int64 坐标。

采用局部 NumPy PCG64 和部分 Fisher–Yates 算法，辅助采样空间 O(Np)，不构造全空间体积的排列。
输出本身需要 O(Nt*Np) 空间，默认 Nt=Lt，校验按时间片检查重复点；当前尚无流式生产接口。
结果记录设计、seed 和算法名 `numpy.PCG64/partial-fisher-yates-v1`。
同一软件环境和 seed 下可复现，不修改 NumPy 全局随机状态。`seed=None` 使用系统熵，
不承诺从元数据重建随机序列。跨版本或生产复用均应保存实际坐标和顺序、NumPy 版本及 cfg 身份，
不能仅凭 seed 判断基底相同。sampler 不写文件，也不替坐标声明生产来源。

已有坐标可直接构造点集。构造器复制坐标，并将其保存为规范只读 `int64`：

```python
from lqcdweave.sampling.spatial.api import SpatialPointSet

lattice = (24, 24, 24, 72)
Lt = lattice[3]
# coords 是已有的规范 (Lt,Np,3) 整数 XYZ 坐标数组。
points = SpatialPointSet(
    lattice_size=lattice,
    times=tuple(range(Lt)),
    coordinates=coords,
)
```

导入不推断 seed 或采样策略，`design/seed/algorithm` 默认为 None。
若外部数据只包含部分时间片或时间顺序不同，必须显式传入对应的有序 `times`；坐标逐项保留，
不会自动补齐时间覆盖。点坐标不进行布局转换。

本模块只定义和生成点基底，不自动乘归一化权重。单个时间片线性空间求和的均匀抽样估计可用
`volume/Np` 加权，但多点乘积及跨时间关联的权重取决于联合包含概率，不能统一套用此因子。
传播子、overlap 与 elemental 必须绑定实际点集；各数值入口和只读规划的职责见[点端生产说明](../perambulator/point_highmode.md)及[通用生产说明](../../architecture/generic_production.md)，不自动对任意多点估计量套用同一权重。

## 直接交给传播子生产器

```python
from lqcdweave.adapters.spatial_points import SpatialPointHandle
from lqcdweave.production.perambulator.api import PointBasis

point_handle = SpatialPointHandle(points, cfg="cfg-1")
# generator.load("cfg-1")  # PerambulatorGenerator 已配置 gauge 与 backend
# result = generator.calc(t_src, src=PointBasis(point_handle),
#                        sink=[PointBasis(point_handle)])[0]
```

句柄公开 `.Np` 和 `.load(cfg)`，返回不可写 `(Nt,Np,3)` `int64` 快照；不需要用户自定义类或手动转置。
保留 points.times 的实际顺序和物理 Lt，通过 `.time_selection` 显式暴露映射。`.read_points(cfg, physical_time, start=0, stop=None)` 可读取一个物理端点。`cfg=None` 默认不绑定组态，
显式 cfg 则拒绝不匹配的 load；不会从路径、seed 或 shape 推断组态/基底身份。
已有裸 NPY 使用规范轴顺序，通过以下入口读取并显式给出格点：

```python
from lqcdweave.preset import save

coordinates = points.coordinates
save("points.npy", coordinates, kind="points")
point_handle = SpatialPointHandle.from_npy(
    "points.npy", lattice_size=(24,24,24,72),
    cfg="cfg-1",
)
```

裸 NPY 只保存数组，不能编码 seed/design/algorithm/cfg；读回 `.points` 中生成历史为 None。
此类元数据另行保留，文件名不是证据。`preset.load/save(kind="points")` 省略 `axis_order` 时采用
`("time","point","xyz")`；显式提供时也只接受该顺序。`SpatialPointHandle.from_npy` 不接收 `axis_order`。
句柄读取时复制坐标并关闭文件映射，不需要用户管理 close；`.points` 保留规范只读 int64 对象。
Elemental 的 `points=` 消费此对象或规范布局数组。

## 脚本保存和读取

仅需坐标数组时直接调用 sampler。需要保存或复用时，由脚本按用户选择的文件格式或通用 storage 接口写出坐标，并一并保留物理 `times`、cfg、点顺序和采样记录；读取时逐项核对实际坐标，不用 seed、Np 或 shape 替代数据身份。`plan_indexed_spatial_points` 是只读的 dataset 定义与覆盖预览，不抽样、不保存；`source_policy="require_known"` 依据已有登记来源判断，fresh 请求和已有 unknown 数据不会因 design/seed 而变成 known。来源语义见[点端生产说明](../perambulator/point_highmode.md#空间点集与来源)。

```python
from lqcdweave.sampling.spatial.api import SpatialSamplingDesign, sample_spatial_points

design = SpatialSamplingDesign((24, 24, 24, 72), 216, temporal_policy=1)
points = sample_spatial_points(design, seed=42)
# 由调用脚本选择保存后端，并同时保存 points.times 与 points.coordinates。
```


需要文件内保留采样记录时复用已有 DatasetWriter/Reader。点坐标是单数组数据，按 cfg 保存为完整 NPY，并由共享根 `dataset.json` 登记；格式和发布边界见[数据保存说明](../storage/data_storage.md)。例如有明确 seed 的生成点集：

```python
import numpy as np
from lqcdweave.planning.spatial_points import sampled_spatial_points_spec
from lqcdweave.storage.dataset import DatasetWriter

spec = sampled_spatial_points_spec(design, configuration={
    "ensemble": "my-ensemble", "configuration": "cfg-1", "lattice": list(design.lattice_size),
}, seed=42)
with DatasetWriter("points-dataset", spec, provenance={"numpy_version": np.__version__}) as writer:
    for row in writer.missing:
        writer.write(row, points.coordinates[row])
    writer.publish()
point_handle = SpatialPointHandle.from_dataset("points-dataset", cfg="cfg-1")
```

上例使用前文同一 design/seed 实际采出的 points；定义不验证数据确实由该随机过程生成。
`from_dataset(directory, cfg=configuration)` 从共享登记中读取指定 cfg 的空间点数据，保留来源、provenance、data_id 和显式本地采样记录；仅包含一个 cfg 时可省略 cfg；
`.cfg` 绑定该 cfg 的登记信息，`.load(cfg)` 必须匹配。它不将 unknown 来源升级为 known，
不重新抽样，也不从 filename 推断缺失元数据。每 cfg 整块 NPY/HDF5 与共享根登记规则见[数据保存说明](../storage/data_storage.md)。已有坐标可使用 `validate_complete_spatial_point_set` 构造实际坐标身份，
并在通用 provenance 保存已知历史；此类任意 provenance 保留原样，不自动解释为 sampling design。

## 可选：联合时间选择

默认脚本不需要时间参数。明确启用后，读取同 cfg 已保存的实际 selection，再传给采样器：

```python
from lqcdweave.production.perambulator.api import PointBasis, VectorBasis

points = sample_spatial_points(design, seed=42, selection=selection)
point_handle = SpatialPointHandle(points, cfg="cfg-1")
assert points.times == selection.times
# generator 使用 sink_selection=selection 构造
# result = generator.calc(t_src, src=VectorBasis(basis_handle),
#                         sink=[PointBasis(point_handle)])[0]
```

selection 按其 `times` 顺序保留输出行，段归属按每个实际物理时间计算；固定 seed 时选中行与全时间结果中相同物理时间的行逐项相同。采样器仍消费全部物理段的 PCG64 序列，抽样次数为 `ceil(Lt/b)`，其中 `b` 是实际段长（字符串策略映射后取 1 或 Lt），不因 selection 改变；只保留选中输出行。周期环绕顺序保持原样，不排序。Dirac 格点仍为物理 Lt。

已有 compact NPY 读取时给 `SpatialPointHandle.from_npy(..., time_selection=selection)`；未给记录表示完整 canonical Lt，不从 shape/文件名猜物理时间。需要持久身份时，`sampled_spatial_points_spec(..., selection=selection)` 或 `validate_complete_spatial_point_set(..., selection=selection)` 复用原 DatasetSpec、writer、reader，将存储轴登记为 0..Nt-1，并保存完整物理格点和实际有序映射。from_dataset 自动恢复该映射，不重抽样或升级来源。

点采样规则、实际坐标、源/汇点池和时间槽顺序分别记录。准确接口和范围见[空间采样 L2](spatial_sampling-api.md)及[共同生产与保存约定](../temporal_sampling/temporal_production.md)。

点集设计与坐标的 CPU 核验见[空间采样回归](../../../tests/sampling/spatial/test_spatial_sampling.py)。当前尚未进行集群规模性能验证。
