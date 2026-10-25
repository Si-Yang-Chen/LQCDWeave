# 已有数据读取

常规脚本优先使用[preset 数据源入口](../storage/preset.md)：声明 prefix/suffix，
调用 `load(cfg)`，再传给计算。以下为显式格式和低层读取接口，后端校验仍然适用。

读取器只解释已知数据的字节和轴布局，不推断组态、基底或物理约定。传入文件时须显式给出时间约定、相位、形状和存储字节序等无法从文件本身可靠推出的信息。

## 传播子

历史 projected elemental 已保存完整有序路径系数但未保存目录标签时，可以显式使用
`DatasetElemental(reader, catalog=meson55_preset(version="v2"))`。
此绑定逐行严格比较全部路径系数及顺序；冲突立即拒绝，不按文件名或行数猜测版本。
它只改变当前reader的分量标签，不修改已登记数据、父项或data_id。
已有目录声明时不得用不同目录覆盖；原始路径和复合扇区不接受此选项。
V/V模式选择使用统一的`mode_left/mode_right`；历史Dataset的`sink_mode/source_mode`
按实际左右槽位映射，不改变存储轴、源端dagger或模式标签。重复指定同一轴的别名会报错。

`ArrayPropagator` 支持九类 V/D/P 传播子的内存数组、整文件 NPY、裸二进制和每个源时间一个 NPY。整文件数组的前两轴默认为 `(source_time, sink_time_or_relative_time)`；其后依次为两个 spin 轴和：

| sector | 末尾轴 |
|---|---|
| VSV | sink_mode, source_mode |
| PSV | sink_point, sink_color, source_mode |
| VSP | sink_mode, source_point, source_color |
| PSP | sink_point, sink_color, source_point, source_color |

D 腿与 V 腿使用相同的 mode 轴名称，但绑定各自实际有序向量身份。
`time_convention="relative"` 表示文件的第二轴是循环槽差；`"absolute"` 表示第二轴直接是汇槽。
`block(sink_time, source_time, sector)` 接收整数槽；未声明时间选择时，槽与规范物理时间相同。
声明 `source_selection/sink_selection` 时，实际物理坐标由各自的 `TimeSelectionRecord.times` 保存。
按源时间分片的 NPY 每片省略第一个轴，文件路径按源槽排列。

```python
from lqcdweave.adapters import ArrayPropagator

with ArrayPropagator.from_npy(
    {"vsv": "cfg.peram.npy", "psv": "cfg.psv.npy"},
    time_convention="relative",
) as data:
    tau = data.block(sink_time=3, source_time=1, sector="psv")

with ArrayPropagator.from_npy_timeslices(
    {"psp": [f"cfg.t{t:03d}.psp.npy" for t in range(Lt)]},
    time_convention="relative",
) as data:
    point_to_point = data.block(3, 1, "psp")
```

只取部分时间与模态时，用物理时间索引和 `usedNe`（每条 V 腿的前 N 个模态）：

```python
with ArrayPropagator.from_npy({"vsv": "cfg.peram.npy"},
                              time_convention="relative") as data:
    subset = data.read_times(sink_times=[0, 3], source_times=[1, 5],
                             sector="vsv", usedNe=16)
    # subset: (2 个源时间, 2 个汇时间, 4, 4, 16, 16)
    for t_sink, t_source, block in data.iter_blocks([0, 3], [1, 5],
                                                     usedNe=16):
        consume(t_sink, t_source, block)
```

`block(..., mode_indices={"sink_mode": [3, 1], "source_mode": [0, 2]})` 可按指定顺序选模态；`usedNe` 与 `mode_indices` 不能同时使用。`read_times` 只物化选中的时间对，`iter_blocks` 一次给一个块。分片文件可在构造时传 `source_times=[1,5]`，只打开对应源时间文件；未打开的源时间读取时报错。

### 传播子的惰性时间视图

完整父数据不需要先转换、写出为紧凑时间方阵。用实际有序选择创建只读视图，直接传给
普通 `gen_correlator/calculate` 的 `propagator_map`：

```python
from lqcdweave.adapters.propagator import ArrayPropagator, select_propagator_times
from lqcdweave.contracts.time_selection import TimeSelectionRecord

selection = TimeSelectionRecord(128, (127, 0, 7, 15))
with ArrayPropagator.from_npy({"vsv": "cfg.full-relative.npy"},
                              time_convention="relative", cfg=cfg) as original:
    data = select_propagator_times(original, selection)
    block = data.block(1, 0)  # 实际 sink=0、source=127，父 relative 索引为1。
    # calculate(..., propagator_map={"S^q": data}, ...) 使用相同实际 selection 的顶点。
```

构造和 metadata 描述不读数值、不创建临时文件；每次 block/slab 只访问实际请求。
视图的 `Lt/Nt/source_Nt/sink_Nt` 为入选槽数，`physical_Lt` 保留原物理128；其公开
`time_convention="absolute"` 表示请求的两个入选槽是端点，不是重新定义父数据的时间差。
父 reader 按原有绝对/相对规则解释映射后的端点。紧凑父数据按其真实有序记录查找，
缺少所需物理端点立即拒绝；不从 shape 猜测坐标，不改变 spin、mode/point 轴、dtype、
端点基底或数值。slab 的顺序、重复和模态/点选择继续沿父读取服务。

视图借用父 reader，关闭视图不会关闭父文件；父 reader 须在计算完成前保持打开。
`data_identity/cfg`、基底和其它科学来源保持父声明；`sectors` 若由父 reader 暴露，仍是
原始存储，不能把它的完整 shape 当成视图覆盖。查询视图时间覆盖和 `describe_block`。

新计划在既有 `SourceReadRequest` 的 `propagator_time_view` convention 中保存父物理 Lt、
absolute/relative 解释及有序父记录；block metadata 同时保留父选择身份。回放核验当前
描述、实际有序选择、dtype、端点与来源共享关系，继续显式绑定本次 reader，不恢复文件
句柄或数值缓存。既有 absolute 紧凑档案可绑定同一真实端点选择的视图；这种档案原先未
记录父布局，仍沿它原有的契约核验。保存/加载及 Dispatch 的固定安排无需新增入口。

### 只持有实际需要的时间对

内存中只有部分时间对时，每个 sector 可直接传 `{(sink_slot, source_slot): block}`。
此时必须使用 `time_convention="absolute"` 并声明时间选择，块中省略两个时间轴。
例如物理时间 0/12 的连通两顶点图只需要两个交叉方向：

```python
from lqcdweave.contracts.time_selection import TimeSelectionRecord

selection = TimeSelectionRecord(72, (0, 12))
data = ArrayPropagator(
    {"vsv": {(1, 0): vsv_12_0, (0, 1): vsv_0_12}},
    time_convention="absolute", source_selection=selection, sink_selection=selection,
    basis_endpoints=actual_endpoints,
)
```

字典键顺序是汇槽、源槽，与 dense 数组前两轴的源槽、汇槽顺序不同。
每个 sector 可有不同的实际时间对覆盖；缺失块在 `describe_block/describe_slab`
规划阶段就报错，不能用零填充冒充覆盖。描述只访问 shape/dtype，不读取数值。
普通选择、端点绑定和批量读取仍走相同接口；slab 保留请求顺序与重复坐标。
reader 固定构造时的时间对成员，数值数组继续借用；此接口不生产、发布或自动寻找数据。

含点传播子还可传 `usedNp`（各点端前 N 个点）或 `point_indices`（精确点编号，顺序保留）。两点端可用二元组分别设置前缀；独立列表可分别指定汇点和源点：

```python
block = data.block(3, 1, "psp", point_indices={
    "sink_point": [12, 2, 7], "source_point": [9, 4],
})  # (4,4,3,3,2,3)
mixed = data.block(3, 1, "psv", usedNp=3, usedNe=16)
```

PSV 只有 `sink_point`，VSP 只有 `source_point`；VSV 不含点轴，传选点参数会报错。这里的点编号是文件点轴的索引；如需对应空间坐标，须按相同顺序选取点集。

发布的 VSV、PSV、VSP、PSP `DatasetReader` 也可按同一传播子读取接口逐块消费，不会在绑定时载入完整时间数据：

```python
from lqcdweave.adapters.meson_dataset import DatasetPropagator
from lqcdweave.storage.dataset import DatasetReader

with DatasetReader("perambulators", cfg="cfg") as dataset:
    data = DatasetPropagator(dataset)
    block = data.block(3, 1, "psv", usedNp=2, usedNe=16)
    # (sink_spin, source_spin, sink_point, sink_color, source_mode)
```

`DatasetPropagator` 也接受实际扇区映射，例如 `{"vsv": vsv_reader, "psv": psv_reader, "vsp": vsp_reader, "psp": psp_reader}`；也可直接传生产 流程 返回的 `datasets` 映射，其中 `basis`、`gauge`、`points` 是输入依赖而不是传播子扇区。复合绑定逐扇区保留 absolute/relative 时间规则及实际覆盖；要求组态、规范场、Dirac、solver、spin basis、时间边界和归一化一致，并按 sink/source 角色核对实际基底 ID 与有序标签。汇点与源点可绑定不同实际点产物，但同一端角色跨扇区必须一致。单个 reader 入口与 VSV 行为保持兼容。

通过 `gamma5_hermiticity` 派生的 VSP 必须声明一个实际 PSV 父产物。只有组合中 PSV 的汇点/源模态与 VSP 的源点/汇模态具有相同实际 ID 和兼容有序前缀时，才要求它就是组合中的 PSV。源汇点集不同时，VSP 绑定源点集对应的反向 PSV，不能强制指向汇点集的正向 PSV。读取器检查声明和同角色兼容性；严格的索引测量入口还须重新验证实际父产物。

`describe_block` 和 `describe_slab` 还会在可选 `axis_bindings` 映射中为每个 mode/point 轴返回 `(actual_data_id, ordered_selected_labels)`。键沿用对应 reader 的 named axis；标签使用 dataset 的坐标值和请求顺序，gather/prefix 会反映在该 tuple 中。自旋、颜色轴不在此映射内；slab 保留尾轴绑定，不把时间批次当成基底轴。

数据集登记中的 `kind` 必须与请求的 sector 一致；轴名必须准确声明两端自旋、模态/点和颜色。`sink_time` 表示绝对汇时；`relative_sink_time` 或显式 `time_convention="relative"` 表示 `(sink_time-source_time) % Lt`。PSV 绑定 `sink_point` 与 `source_mode`，VSP 绑定 `sink_mode` 与 `source_point`，PSP 绑定两个点端；每个模态/点轴都须带有非空实际 `bases[axis].data_id`，`indices` 须逐项等于该轴的有序标签。点轴标签是该点产物的坐标标签，adapter 将它们保留在描述身份中，不把实际点产物 ID 当作采样池，也不推断两端是否为同一组点或与 elemental 的坐标对齐。

PSV/VSP/PSP 还要求显式声明 `spin_basis="degrand_rossi"`、`temporal_boundary="periodic"` 或 `"antiperiodic"`、`normalization="unscaled"` 及实际 `definition.gauge_id`；未知约定报错。两端颜色轴必须依序为 0、1、2。这些检查只核对单个产物中声明的值，不解引用 ID，也不证明它与另一传播子或 elemental 使用同一规范场/点集；跨产物兼容性仍由既有 planning 与测量调用方负责。`configuration.ensemble` 或 `configuration.configuration` 为 `null` 时，adapter 保留未知值供读取，但这不构成严格测量兼容认证；规划层会阻止未知身份的复用。`usedNp` 与 `point_indices` 仍使用 ArrayPropagator 的侧向参数；`point_indices` 中的整数是文件点轴位置，列表顺序保留。基本切片返回借用的只读块，精确 gather 返回拥有数据的块；reader 由调用方持有，并须保持打开至消费完成。

裸二进制使用 `from_binary(files, shapes=..., dtypes=..., time_convention=...)`；`dtypes` 应包含明确字节序，如 `">c16"` 或 `"<c8"`。文件长度必须和给定形状、dtype 及可选的 `offsets` 完全吻合。无头格式不会自带轴语义。

## Elemental

`ArrayElemental.from_npy` 和 `ArrayElemental.from_binary` 支持稠密 V/V、V/P、P/V、P/P；`ArrayBaryonElemental` 复用这两个入口读取 V/V/V。前三轴顺序通过 `axis_order` 声明，分量及相位约定由调用者给出。例如：

```python
from lqcdweave.adapters.elemental import ArrayElemental

with ArrayElemental.from_binary(
    {"v2v": "cfg.meson"}, components,
    shapes={"v2v": (len(components), num_momentum, Lt, Ne, Ne)},
    dtypes={"v2v": "<c16"},
    axis_order=("component", "momentum", "time"),
    phase_convention="midpoint",
) as data:
    block = data.block(components[0], 0, 0, "v2v")
```

### 已发布点端 elemental

`DatasetElemental` 可继续读取原有单个 V/V dataset，也可组合已登记的 `v2v`、`v2p`、`p2v`、`p2p` 扇区：

```python
from lqcdweave.adapters.meson_dataset import DatasetElemental

elemental = DatasetElemental(
    {"v2v": vv_reader, "v2p": vp_reader, "p2v": pv_reader, "p2p": pp_reader},
    point_dataset=points_reader,
    sampling_pool_ids={(component, 0, t, "bar"): "pool-A" for t in range(Lt)},
    sampling_groups={"pool-A": pool_spec},
)
block = elemental.block(component, 0, 3, "v2p", usedNe=8, usedNp=4)
```

V/P 与 P/V 只读取请求的物理时间、路径、动量和端点标签；完整时间覆盖由数据集登记按物理坐标明确声明，储存行不用于推定遗漏的时间。两类扇区须声明实际规范场、模式/点轴绑定、有序标签、`normalization="unscaled"`、明确 `phase` 和空间周期路径约定。复合读取器校验共同配置、路径/动量定义、相位和同一端角色的实际基底 ID/标签；不同 shape 或相同 shape 本身都不能证明端点相同。未知配置/系综身份可留给只读注册，但这不授予严格测量兼容性。

P/P dataset 使用 `elemental_p2p_row_sparse`：持久数组的 `point_left` 轴是行槽，末两轴是左右颜色，不存储稠密右点轴。构造时传入左端 `spatial_points` reader/view；新式双点集定义还可传 `point_right_dataset`，供右端坐标与标签读取。adapter 每次只读取请求物理时刻的 XYZ 坐标，结合有序空间路径和周期边界重建每行唯一右点列，再将 row-slot 值作为稀疏 `{"type": "sparse", "indices": (nnz, 2), "values": (nnz, 3, 3)}` 返回。零值配对仍保留在结构中；没有坐标支持的行不产生条目。`usedNp` 和 `point_indices` 可分别选择左右标签及其顺序，COO 索引改为所选列表的局部位置。P/P 按逐物理时刻 `block` 消费，不提供多时 dense slab，也不会隐式生成 `Np×Np` 数组。

左端 `point_dataset` 的 ID 必须等于 P/P 定义中的左端实际点 ID。旧定义用 `point_right_selection="same_ordered_prefix_as_point_left"` 声明右端复用同一实际点集及左端有序标签；未传 `point_right_dataset` 时，adapter 只允许在右端实际 ID 与左端 ID 相同且真实标签覆盖右端定义时复用左 reader。双端定义用 `point_right_selection="explicit_ordered_labels"`、`point_right_id` 和有序 `point_right_labels` 声明真实右端选择，并须传入 ID 匹配的 `point_right_dataset`。两端 reader 的 `time,point,xyz` 坐标轴必须覆盖请求的物理时刻，XYZ 顺序为 `x,y,z`，并显式声明 `coordinate_delta`/`unit_delta`；标签和坐标从各自的实际 reader 获取，不能由 shape 推断两侧同点。

未传 `usedNp` 时，左右轴各保留各自实际 view 的点数；单个 `usedNp` 仍按两端共用的有序前缀语义裁剪。`point_indices` 可分别选择左右标签及其顺序，COO 索引改为对应列表的局部位置。适配器返回的 actual dataset ID 仍与采样池 ID 分开：如 Diagram 需要采样补偿，调用方另传每端 `sampling_pool_ids` 和 `SamplingPoolSpec`，池 ID 必须显式登记，不能由点产物 ID、Np 或数组形状生成；不同池的抽样分组与笛卡尔乘积由显式声明控制。

`DatasetElemental.describe_block` 与 dense `describe_slab` 对每个 mode/point 轴也提供 `axis_bindings[axis] = (actual_data_id, ordered_selected_labels)`。P/P 稀疏逻辑视图分别为 `point_left`、`point_right` 保留各自所选坐标标签；同池旧定义可绑定同一个显式 point dataset ID，独立双点集定义则保留左右各自的实际 ID 和标签顺序。颜色轴不绑定到 basis/point identity。

`describe_block` 对 P/P 返回稀疏布局、所选点 shape 与坐标重建的支持数；`describe_sparse_record` 描述完整当前点视图，`describe_selected_sparse_record(..., usedNp=...)` 描述选择后的实际 COO 记录。基础连续切片可借用只读 DatasetReader 内存映射，点/模态 gather 与请求 slab 拥有独立数组；调用方持有所有 reader，并保持打开至消费完成。适配器会在各自输入内核验声明的实际 endpoint ID 和标签；共同 E 若两端都声明该身份，会逐稀疏 edge 比较实际 ID 与有序标签，匹配标为 `verified`，冲突在展开时拒绝，任一侧缺声明则保留 `unknown`。严格收缩编排测量还要求相关输入的已知身份匹配。此检查不证明点采样随机设计、跨 reader 的全部物理参数、规范场或未声明边界约定兼容，也不替代生产或数值 MPI/GPU 验收。

`block(..., usedNe=16)` 仅取模态前缀；三腿重子数据可传 `usedNe=(n0,n1,n2)`。`mode_indices={"mode_left": [2,0], "mode_right": [1]}` 可选任意有序子集。`read_times(component, momentum, [0,4], sector, usedNe=...)` 返回所选时间数组；`iter_time_blocks(...)` 逐时间产出，避免一次物化多个时间块。稀疏 P/P 也提供 `iter_time_blocks(...)`。

旧 `CurrentElementalV2P`／`CurrentElementalP2V` 的 NPY 分别使用 `(Lt, num_disp, Ne, Np, 3)`／`(Lt, num_disp, Np, 3, Ne)`，保存的是没有 Fourier 相位的空间 gaugelink 块。使用专用的 `LegacyGaugeLinkElemental`；`ArrayElemental` 仍要求文件有真实动量轴。需提供按旧文件位移顺序排列的空间 `GaugePath` 分量、实际点集和物理动量表。例如：

```python
from lqcdweave.adapters.elemental import LegacyGaugeLinkElemental
from lqcdweave.sampling.spatial.api import SpatialPointSet

lattice = (Lx, Ly, Lz, Lt)
points = SpatialPointSet(
    lattice_size=lattice,
    times=tuple(range(Lt)),
    coordinates=point_coordinates,
)
with LegacyGaugeLinkElemental.from_npy(
    {"v2p": "cfg_v2p.npy", "p2v": "cfg_p2v.npy"}, components,
    points=points, momentum_list=((0, 0, 0), (1, 0, 0)),
    phase_convention="midpoint",
) as data:
    block = data.block(components[0], 1, 4, "v2p")
```

`point_coordinates` 为与文件同顺序的规范 `(Lt,Np,3)` 整数 XYZ 坐标；动量参数是 `momentum_list` 中的索引。读取器保留原文件内存映射，到 `block`／`read_times` 消费时才乘 `exp(+2πi p·x/L)`：默认 `midpoint` 对 P/V 取点坐标 `x+Δ/2`，对 V/P 取 `x-Δ/2`；`anchor` 分别取 `x` 与 `x-Δ`。`Δ` 是声明路径的空间位移。点集必须覆盖全部物理时间且与文件点轴一致；选点后相位按所选点的原始坐标计算。旧 `GaugeLink` 整数编号与新 `GaugePath.idx` 不同，须按文件中每个旧编号对应的实际路径步骤构造 `components`，不能直接复用编号。该读取器不执行空间求和或采样归一化，也不从文件推断点集、路径顺序及实际本征基底身份。

V/P、P/V 和稠密 P/P 分别使用 `point_right`、`point_left`、双点端筛选。例如 `block(component, 0, 4, "p2p", point_indices={"point_left": [5,1], "point_right": [3,8]})`；`usedNp=(2,3)` 可分别取两端前 2/3 个点。`usedNp` 与 `point_indices` 互斥；模态和点的筛选可以同时使用，颜色轴始终保留完整 3 色。

稀疏 P/P 使用已发布的 `elemental_p2p_row_sparse` Dataset，通过 `DatasetElemental` 读取；其点选择、稀疏记录和所有权见上文。

## 规范场与本征矢

普通 NPY 或裸二进制输入可用 `DenseFileHandle(prefix, suffix, format="npy"/"binary", shape=..., dtype=..., Ne=..., Np=...)`。NPY 的 `shape`、`dtype` 可省略并由文件头读取；显式提供时仍严格校验。裸二进制仍必须提供两者。它提供生产器要求的 `load(cfg)`；历史文件的头部轴与逻辑轴不一致时，用显式 `reshape=` 和 `transpose=` 转为目标布局。裸二进制的 dtype 必须指定字节序，文件长度会校验。规范场目标布局为 `(Lt,Lz,Ly,Lx,4,3,3)`，本征矢为 `(Lt,Ne,Lz,Ly,Lx,3)`，点集为 `(Lt,Np,3)`。

`IldgGauge(prefix, suffix=".lime").load(cfg)` 解析 ILDG/LIME 的 `ildg-format` XML 和 `ildg-binary-data`，校验格点尺寸、精度及负载长度，返回 `(Lt,Lz,Ly,Lx,4,3,3)` 的只读数组。`QDPTimesliceField(prefix, suffix, role="gauge").load(cfg)` 读取 QDP 的逐时间、逐方向记录；`role="eigenvector", count=Ne` 读取逐时间、逐模式记录，形状为 `(Lt,Ne,Lz,Ly,Lx,3)`。QDP 记录按需读取；对规范场使用 `[:]` 会物化完整规范场。

QDP 句柄的 `read_times([1,4], records=[0,2])` 只读取指定时间及方向／模式记录；`[t, mode]` 可直接读取一个本征模。普通 NPY、裸二进制和 ILDG 返回内存映射，直接对返回值切片即可。以上筛选均由调用者显式选择；生产器仍默认遍历全部时间片。

QDP 本征矢的 `[t, mode_slice]` 和 `[time_slice, mode_slice]` 会先按模态切片选择记录，再读取数据；支持步长、反向和空切片。返回值是独立数组，不借用可被下一次读取覆盖的缓冲。Elemental 路径分块因此只读取当前块需要的模式；重子加载的 `[:, :usedNe]` 也不会读取未选模式。不过重子仍保留全部时间的已选模态快照，传播子仍复制全部时间基底及汇端共轭副本。这些读取优化不代表生产端已有统一内存预算或全面超内存支持。格式读取不赋予实际基底身份，持久身份须由数据集登记提供。

## 登记数据集的只读读取

旧裸 NPY、分片 NPY 和外部二进制仍可通过前述专用 reader 作为输入读取；读取不会自动赋予已知科学身份。新保存的数据集以共享根 `dataset.json` 登记公共 metadata 与各 cfg 记录：简单数组每 cfg 保存一个完整 NPY，多字段结果使用 HDF5。`DatasetReader(directory, cfg=cfg)` 按 cfg 读取新登记的 NPY/HDF5，可按具名坐标选择数组视图。写入规则见[默认存储格式](../storage/data_storage.md)。计算脚本组合公开适配器、数值入口和所选保存方式，具名物理轴必须与数据一致，不能把相对汇时间当作绝对时间。V/P/P 扇区与稀疏 P/P 按各自适配器及 calculator 的真实契约消费，详见 [correlator skill](../../../.agents/skills/lqcdweave-correlator/SKILL.md)。

`DatasetReader.read(first_axis=..., select={"source_mode": [0,2], "sink_mode": slice(0,4)})` 可选其余轴；列表使用登记的物理坐标，slice 使用数组中的轴位置。`iter_read([...], select=..., ...)` 逐首轴坐标给出数组视图。`DatasetIndex.open_selected(actual_id, coordinates)` 和 `DatasetReader.selected_coordinates` 限制读取视图；它们不表示产物只完成了部分生产。具名 HDF5 字段通过 `read_field(path)` 读取。视图只描述消费范围，不替代 cfg 完成状态或实际 scientific identity。

## 选点性能

稠密 PSV/VSP 的返回数据量随所选点数近似线性下降；PSP 与稠密 P/P 的两点端分别缩小，数据量按 `(所选左点/总左点) × (所选右点/总右点)` 缩放。连续前缀是 memory map 上的视图；任意顺序的点列表需要聚集复制。为减少临时数组，读取器先应用切片，再优先聚集比例较小的轴。实际磁盘页读取还受行主序、缓存和点位置影响，不一定按返回字节数同比下降。

本机热缓存示例（[可复跑脚本](../../../benchmarks/point_read.py)，64×64 点 PSP、每端选 8 点）：

| 数据 | 返回数据量 | 单次读取并消费的中位时间 |
|---|---:|---:|
| PSP 全点 | 9.0 MiB | 0.195 ms |
| PSP 精确选 8×8 点 | 0.141 MiB | 0.080 ms |
| PSP 前缀 8×8 点 | 0.141 MiB | 0.019 ms |

表中时间是本机合成数据的热缓存中位数，不能当作集群文件系统性能承诺。当前 row-sparse Dataset 的读取量和性能应按其实际布局单独测量。


## 只读取产物元数据

`DatasetIndex.observe_metadata(actual_id)` 只读取已登记产物的元数据声明，不打开数组内容。它适合读取 shape、dtype、坐标、provenance 和 execution 信息；需要消费数据时使用 `open` 或 `open_selected`。元数据观察不证明文件存在或其中数值未被修改，reader 会在实际读取时继续检查结构、shape、dtype 和布局。

完整 cfg 的 `load_correlator_output` 如何复用 reader/映射、资源何时释放，属于实现级性能与生命周期细节；在对应行为通过验收前，不将其写成公共接口保证。调用方按本节指定的 cfg、字段和坐标读取。


<a id="shared-npy"></a>
## 高级性能任务：共享 NPY 前缀读取

仅在多个请求共用已核验的同一实际 NPY、有序基底及嵌套前缀时使用。
普通 preset/load(cfg) 不需要改成低层 scene。独立字段与共享准备的接线见
[共享数据回归](../../../tests/storage/test_shared_data_contract.py)：请求 → 合并 → SourcePreparation → solve_storage → loader → replay。

```sh
python -m unittest discover -s tests/storage -p test_shared_data_contract.py
```

`NpyPrefixRequest(dataset, basis_identity, path, shape, prefix_shape, dtype)` 显式声明身份和完整头部；
`merge_npy_prefixes(requests, backend="numpy", read_mode="whole_file")` 返回 blocks、bindings、aliases，尚不读文件。
同一文件的身份冲突、不同有序基底或越界前缀拒绝合并；不能因 shape、seed、路径相同就认定相同科学输入。
保留非截断轴及全部物理时间，不能为共享读取偷偷时间稀疏化。

whole_file 默认计整文件 payload、host staging 和选取后的传输字节，不是选择性磁盘读取；mmap 也不证明全面内存受控。
`NpySceneBlockLoader` 在实际读取核验 header，按公开回放生命周期释放；materializer 只转换已读块，不能再次读外部数据。
数值内容的基底身份仍是调用方核验的声明，loader 不作物理认证。例中统计实际 read 事件和字节，并与直接切片参考比较。
任意区域读取才需要 `adapters.scene_basis_region.MaxBasisRegionLoader`，不是本路线的前置步骤。
验收见 `tests/test_scene_storage_prefix.py`，CPU fixture 不证明 GPU、MPI、真实存储吞吐或生产规模。
