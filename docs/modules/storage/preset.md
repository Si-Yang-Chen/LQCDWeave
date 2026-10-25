# 常用数据加载与保存

单个已有稠密 NPY 可直接指定数据用途和路径；文件头已有的 shape、dtype 不必重复填写：

```python
from lqcdweave import preset

gauge = preset.load("/data/gauge.npy", kind="gauge")
vectors = preset.load("/data/vectors.npy", kind="eigenvector")
values = preset.load("/data/eigenvalues.npy", kind="eigenvalues")
saved = preset.save("/results/vectors.npy", vectors, kind="eigenvector")
```

这是读取裸 NPY 数组的便捷入口，不增加数据集登记、恢复或生产框架。正式持久数据使用
`DatasetWriter` 保存整块 cfg 文件，并由共享 `dataset.json` 登记；读取已登记数据使用
`preset.Dataset(directory).load(cfg)`。旧的逐 cfg NPY 源仍由专用兼容 reader 提供。

## 单文件 `load` / `save`

```text
load(path, *, kind, shape=None, dtype=None, axis_order=None,
     time_convention=None, components=None, phase_convention=None, cfg=None)
save(path, data, *, kind, axis_order=None, time_convention=None,
     components=None, phase_convention=None, overwrite=False)
```

- 两个函数都只处理一个完整稠密 NPY 数组。`kind` 必填，取下列小写名称；不从路径猜用途。
- `load` 从 NPY 头读取 shape/dtype，给出 `shape=` / `dtype=` 时严格检查**磁盘布局**，不裁剪、不转换精度或字节序。加载为只读映射，不复制整个数组；检查文件类型、完整长度、rank、dtype 类别、非空轴和固定颜色/自旋轴。
- `save` 只接受稠密、非对象、非 masked 的主机 NumPy 数组（包括 memmap），由数组导出 shape/dtype。保存输入轴顺序、数值、精度和字节序，不按 canonical 布局改写磁盘数组；NPY 不保留任意视图 strides。不会隐式转换 CuPy、列表、稀疏对象或 reader。
- 返回值是实际写入的 `pathlib.Path`。使用传入的精确路径，即使没有 `.npy` 后缀也不会追加；父目录须已存在。默认独占创建，已有文件时报 `FileExistsError`。`overwrite=True` 才允许替换，先写完临时文件再替换目标，写入失败保留原文件。此行为不提供生产产物发布、续算或断电持久性保证。
- 缺少声明、非法布局/精度类别、NPZ/非 NPY、截断或多余字节会拒绝。只核对布局与声明一致性，不检验 SU(3)、向量正交性、物理身份、坐标是否在格内或数值有限性；这些仍由对应科学契约和计算入口检查。

### 场数组

返回只读 NumPy 映射视图，轴如下。省略 `axis_order` 表示调用者确认文件使用表中 canonical 布局；其它 `kind` 的排列必须显式列出**所有轴**，读入时转换为 canonical 视图。`points` 默认采用规范顺序 `(time,point,xyz)`；显式提供 `axis_order` 时也只接受该顺序，不转换其它排列。

| kind | 返回的 canonical 轴 | dtype 类别 |
|---|---|---|
| `gauge` | `(time,z,y,x,direction,color_left,color_right)`，末三轴 `(4,3,3)` | complex |
| `eigenvector` | `(time,mode,z,y,x,color)`，color=3 | complex |
| `eigenvalues` | `(time,mode)` | float |
| `points` | `(time,point,xyz)`，xyz=3，坐标分量按 x/y/z | integer，排除 bool |
| `overlap` | `(time,mode,point,color)`，color=3 | complex |

`points` 省略 `axis_order` 时使用规范顺序；也可显式声明 `("time", "point", "xyz")`：

```python
points = preset.load("/data/points.npy", kind="points")
p = preset.save("/results/points.npy", points, kind="points")
```

这些返回值是数组，不是 `SpatialPointSet`、科学身份或 `load(cfg)` 句柄。裸 NPY 交给传播子时使用 `SpatialPointHandle.from_npy(path, lattice_size=lattice, cfg=cfg)`；已有规范坐标可直接构造 `SpatialPointSet(lattice_size=lattice, times=tuple(range(Lt)), coordinates=coords)`，再包装为 `SpatialPointHandle`。点句柄提供 `.Np/.load(cfg)`，详细来源与布局见[空间点集](../spatial_sampling/spatial_sampling.md)。其他要求句柄的生产器使用下面的数据源类或 `DenseFileHandle`。
本入口不重排 XYZ 分量，也不为四轴 flat-volume 本征矢猜格点尺寸；这种历史本征矢继续用 `EigenvectorNpy(..., lattice_size=...)`。
`overlap` 的 kind 声明沿用现有 `eta_dagger_xi` 排列，不额外共轭或归一化；调用者负责确认文件约定。

### 传播子

`kind="vsv"/"psv"/"vsp"/"psp"` 返回已有 `ArrayPropagator`，`time_convention` **必填**：
`"absolute"` 表示物理汇时间，`"relative"` 表示相对源时间的循环差。
`axis_order` 只描述前两轴，默认 `("source_time","sink_time")`，也可显式交换。
之后固定为 `(sink_spin,source_spin,...)`，两个 spin 均为 4，两个 time 轴均为完整 Lt。

| kind | spin 后的尾轴 |
|---|---|
| `vsv` | `(sink_mode,source_mode)` |
| `psv` | `(sink_point,sink_color,source_mode)` |
| `vsp` | `(sink_mode,source_point,source_color)` |
| `psp` | `(sink_point,sink_color,source_point,source_color)` |

所有传播子要求 complex dtype，color 均为 3；源汇 mode/point 数量可不同。

```python
with preset.load("/data/vsv.npy", kind="vsv",
                 time_convention="absolute", cfg="2000") as p:
    tau = p.block(sink_time=3, source_time=0, usedNe=16).copy()
```

`time_convention` 必须显式声明。`cfg=` 只绑定当前传播子/elemental reader，不从文件名推断，也不证明文件属于此组态；场数组不接受这个参数。

### 稠密 elemental

`kind="v2v"/"v2p"/"p2v"/"p2p"` 返回已有 `ArrayElemental`；`"vvv"` 返回 `ArrayBaryonElemental`。
必须显式传 `components=`、`phase_convention=` 和 `axis_order=`；不从 shape 生成组件、不猜相位。
`axis_order` 只描述前三轴，是 `("component","momentum","time")` 的排列。
读取器返回 canonical 的前三轴；尾轴不可通过此参数改变：

| kind | 前三轴之后的尾轴 |
|---|---|
| `v2v` | `(mode_left,mode_right)` |
| `v2p` | `(mode_left,point_right,color_right)` |
| `p2v` | `(point_left,color_left,mode_right)` |
| `p2p` | `(point_left,color_left,point_right,color_right)` |
| `vvv` | `(mode0,mode1,mode2)` |

dtype 可为 float 或 complex，color 均为 3。组件须为不重复的实际 `ElementalComponent` 定义，数量和文件 component 轴一致；vvv 使用 `BaryonComponent`。
普通 elemental 相位可为 `"midpoint"`、`"anchor"`、`"none"`，vvv 仅 `"anchor"` 或 `"none"`。
例如已知单条空路径、canonical 排列的 V/V 数组：

```python
from lqcdweave.contracts.elemental import ElementalComponent

declarations = dict(components=[ElementalComponent.path(())],
                    phase_convention="anchor",
                    axis_order=("component", "momentum", "time"))
# elemental_values 已有 shape (1,num_momentum,Lt,Ne_left,Ne_right)
path = preset.save("/results/v2v.npy", elemental_values,
                   kind="v2v", **declarations)
with preset.load(path, kind="v2v", **declarations) as e:
    block = e.block(declarations["components"][0], 0, 0, "v2v").copy()
```

NPY **只保存数组头和数值**。上述 `kind`、具名轴、组件定义、相位、时间约定、cfg、动量表、实际基底/点集身份等不会写入文件；调用者须保留原声明并在重新加载时提供。`save` 校验声明并不代表 NPY 能恢复它们，也不生成 sidecar。
未知科学身份保持未知，严格复用/发布继续使用既有 storage/planning 契约。

场数组在最后一个映射引用释放时由 NumPy 释放。传播子/elemental reader 可用 `with` 或 `.close()`；借用块在关闭后不可使用，需要跨生命周期保留时显式 `.copy()`。
Windows 上仍被映射的目标文件不能替换，`overwrite=True` 此时报 `PermissionError`，保留原文件及现有视图，并清理临时文件；释放映射后可覆盖。不隐式关闭调用方的映射，也不为覆盖复制整个输入数组。
加载期间不要外部修改输入文件。此入口只覆盖全时间稠密数组，不解析时间子集记录、稀疏 P/P、NPZ/HDF5、裸二进制或已发布产物；这些不能由 NPY shape 推断。
CPU 小数组验证不代表 GPU/MPI、整体内存受控或真实生产规模验收。

## 按 cfg 读取已登记数据

简单数组的新输出默认是每 cfg 一个完整 NPY，多字段数据使用 HDF5；所有 cfg 由目录根部的
`dataset.json` 共同登记。使用 `preset.Dataset` 传入共享数据目录，再按 cfg 加载：

```python
from lqcdweave import preset

dataset = preset.Dataset("/results/vsv")
vsv = dataset.load("2000")
```

文件布局、科学身份、时间坐标与来源由登记记录描述。未知科学身份仍为 unknown；`preset.Dataset`
不会从目录名、文件名或数组 shape 推断基底、点集或来源。详细写出方式见[数据保存格式](data_storage.md)。

## 兼容读取已有逐 cfg NPY

已有的 perambulator、elemental 与本征矢 NPY 可通过专用 reader 读取旧文件。以下调用仍按
`prefix + cfg + suffix` 找到输入文件，既可调用 `load(cfg)`，也可交给原生
`calculate(configuration=cfg)` 自动绑定。文件检查、时间坐标换算、
模态范围检查和已有数据集校验由后端执行。这个兼容输入路径不改变新数据默认的共享登记格式；
用户不必为读取旧文件手工管理 `DatasetSpec` 或 reader。实现位于 `adapters/preset.py`。

## 已有 NPY

```python
from lqcdweave import preset

perambulator = preset.PerambulatorNpy("/data/cfg_", ".peram.npy",
                                      [128, 128, 4, 4, 120, 120], 120)
elemental = preset.ElementalNpy("/data/cfg_", ".meson.npy",
                               [3, 27, 128, 120, 120], 120)
p = perambulator.load("2000")
e = elemental.load("2000")
tau = p[0, 3, :, :, :16, :16]  # 旧存储轴索引；返回独立 NumPy 数组
block = p.block(sink_time=3, source_time=0, usedNe=16)
```

prefix/suffix 由调用者提供。`shape`、`totNe` 可省略，由后端读取 NPY 头；
显式给出时作为检查，不能以较小的 shape 假装文件已裁剪。complex64/complex128
取实际文件精度；可传 `dtype=` 要求严格匹配。加载只映射文件，不复制整个数据集。
同一声明、相同 cfg 与未变文件复用已加载对象；另一个 cfg 返回另一个绑定，
不关闭仍被计算引用的旧对象。不要在计算期间改写或删除输入文件。

未声明 selection 时，`p.block` 使用物理时间，默认旧 perambulator 的相对汇时间约定；绝对时间文件
须声明 `time_convention="absolute"`。旧式 `p[...]` 直接索引文件轴，不换算时间。
`e[...]` 的轴为 `(component,momentum,time,mode_left,mode_right)`。

`ElementalNpy` 默认组件标签为 `ElementalComponent(0), ElementalComponent(1), ...`，
只表示文件行号，不证明对应哪个规范路径。需要路径算符时传 `components=`，
其顺序须与文件一致。默认 `phase_convention="midpoint"` 是这个 preset 的约定；
其它文件可显式声明 `"anchor"` 或 `"none"`，后端核对与算符是否一致，不修补相位。

## 直接接入计算

原生 `calculate` 可直接消费 preset 数据源声明及未加载的顶点。
`configuration=cfg` 在这次计算中加载并校验组态；显式 reader 则保持已有绑定。
数据源、算符、图和共同选项可在循环外定义。

紧凑输入显式使用 `ElementalNpy(..., selection=record)` 和 `PerambulatorNpy(..., selection=record, time_convention="absolute")`。独立源汇覆盖可给 `source_selection/sink_selection` 代替共同 selection。ArrayElemental/ArrayBaryonElemental 与 `preset.load/save` 同样接受 selection。裸 NPY 不内嵌科学元数据，必须保存并重新提供同 cfg 的有序记录；Dataset 从共享登记读取已登记数据。紧凑 reader 的 `.Lt/.Nt` 是槽范围，`.physical_Lt` 与 `.time_selection` 保留物理身份；`time_axes` 用槽，不传 selection.times。旧 relative 文件在共同 selection 下明确表示循环槽差，不能当物理 dt。详见[共同时间约定](../temporal_sampling/temporal_production.md)。

生产端的 `EigenvectorNpy(..., selection=record)` 是不同边界：磁盘按 Nt 保存，但 `read_modes(cfg, physical_time, start, stop)` 显式把物理时间映射到槽，缺失坐标报错。规范场/Dirac 仍保持完整物理 Lt。

含时间偏移的原生 Meson/Current 在绑定时按真实物理偏移生成端点槽表，传播子需覆盖这些端点。若显式 sampling_pool_ids 覆盖额外端点槽，ArrayElemental/DatasetElemental 使用 `sampling_selection=record` 声明池时间顺序；不把锚点槽和采样端点槽混用。

```python
from lqcdweave.contraction.binding.api import Meson
from lqcdweave.contraction.correlator import calculate, save_correlator
from functools import partial
import numpy as np

source = Meson(elemental, operator, True)
sink = Meson(elemental, operator)
compute = partial(calculate, diagram, propagator_map={"q": perambulator},
                  vertex_map={"source": source, "sink": sink}, usedNe=16,
                  time_axes={0: range(128), 1: range(128)})
for cfg in ("2000", "2004"):
    values = compute(configuration=cfg).value
    save_correlator("/results/two_point", values, configuration=cfg)
```

同一声明在一次计算内加载一次，原 `source/sink` 不被修改；旧计划持有自己的组态绑定，
可在后续 cfg 计算后执行。`execute=False` 可以读取 NPY 文件头和数据集登记，但不读取数值数据。
显式加载写法继续支持；与请求 cfg 不符的 reader 会报错。自定义 `diagram_adapter` 不走
自动解析，仍由它处理自己的输入。此接线只支持上述测量数据源及 `Dataset` 当前支持的类型；
不隐式加载任意带 `load` 方法的对象。

`diagram` 的顶点/传播子标签应与上述映射相同。完整输入与计算绑定约定见
[数据绑定参考](../../../.agents/skills/lqcdweave-correlator/references/binding.md)。默认保留全部时间，
普通脚本不操作 mmap 生命周期；数据源/计算绑定保留映射引用，最后一个引用释放时
由 NumPy 释放映射。低层 `block` 可以借用映射，读取器关闭后不可再使用借用块；
旧式 `[...]` 得到独立数据，适合跨读取器生命周期保存。对整文件使用 `[...]`
仍会复制整个文件，大数据应按块消费。

两类文件的后缀由各自的 NPY reader 构造参数指定。组件、动量行与 gamma 配方须来自实际定义，
不根据行号或文件名推断物理动量和路径。

## 本征矢与已发布数据

```python
vectors = preset.EigenvectorNpy("/data/cfg_", ".eig.npy", totNe=120)
mode_block = vectors.read_modes("2000", time=0, start=0, stop=16)

published = preset.Dataset("/results/vsv")
p = published.load("2000")
```

`EigenvectorNpy` 默认读取规范 `(time,mode,z,y,x,color)` 轴。旧四轴
`(mode,time,volume,color)` 文件另传 `lattice_size=(Lx,Ly,Lz,Lt)`；后端转换视图，
不猜体积形状。其它排列可显式声明 `axis_order=`。生产器可直接接收这个
`load/read_modes` 句柄；构造生产器前 `.Ne` 必须已知。给出 `totNe`，或给出 `shape`（按 `axis_order` 定位 mode 轴）可直接确定计数，实际文件头仍在装载时核验；两者都省略时先调用 `vectors.load(configuration)` 解析配置文件头，再构造生产器。读取器使用 memory map，不在这一步复制完整基底。`CurrentElementalV2V(prefix,suffix,shape,Ne)` 读取旧
`(time,component,momentum,Ne,Ne)`，自动提供 ElementalNpy 的读取顺序。

`Dataset(directory)` 读取共享目录中已登记的数据，支持 VSV/PSV/VSP/PSP、稠密 V/V
elemental、Laplace basis 和 gauge。后端持有读取器并使用原有适配器；数据集登记不完整、
实际 cfg 不符、未知/错误基底绑定或物理约定仍按既有校验拒绝。
它不登记 raw 文件、不生产缺失数据；其它产物继续使用其公开专用接口。

## 接口与范围

- `PerambulatorNpy(prefix,suffix=".npy",shape=None,totNe=None,*,time_convention="relative",dtype=None)`。
- `ElementalNpy(prefix,suffix=".npy",shape=None,totNe=None,*,components=None,phase_convention="midpoint",dtype=None)`。
- `CurrentElementalV2V(prefix,suffix="_v2v.npy",shape=None,Ne=None,**options)`，options 同 ElementalNpy。
- `EigenvectorNpy(prefix,suffix=".npy",shape=None,totNe=None,*,lattice_size=None,axis_order=None,dtype=None)`。
- `Dataset(directory)` 通过共享登记查找指定 cfg；旧 NPY reader 继续提供 `load(cfg)`，本征基底另有 `read_modes(cfg,time,start,stop)`。

省略 suffix 默认 `.npy`。
`totNe` 表示文件实际模式数量，计算时 `usedNe` 才是截取数量。
NPY 后端检查头、非对象复数 dtype、文件长度、声明形状及 Ne，并复用现有
传播子/elemental 的轴、颜色、自旋和范围校验。裸文件仍没有科学基底身份；
路径和形状不会自动成为持久身份。即时读取可以使用未知身份，严格持久复用仍由
既有 planning/storage 校验阻止；不能通过 preset 绕过。

此入口支持常用数据声明、CPU NumPy 原始读取及计算入口。
GPU/MPI 和真实生产规模的本入口验收另行进行。

预制规范链接投影的 NPY 还需 `ElementalNpy(..., components=generator.components,
 momentum_list=momenta)`；`ArrayElemental`、`from_npy/from_binary` 同样接受可选
`momentum_list` 并校验轴长度和唯一 XYZ 整数坐标。省略时身份保持未知，严格物理动量绑定
报错。已发布 `elemental_projected` 可用 `preset.Dataset` 读取保存的路径/系数行及动量。
完整例子见[配套预制表](../elemental/elemental_presets.md)。

紧凑 SpectralProfile 加权项目前明确拒绝，因为 profile 没有独立的有序时间域记录；不能从行数猜测其物理/槽含义。默认全时间 profile 与不使用 profile 的模式分区保持原行为。
