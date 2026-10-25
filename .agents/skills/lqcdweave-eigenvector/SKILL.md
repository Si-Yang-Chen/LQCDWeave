---
name: lqcdweave-eigenvector
description: "为 distillation 测量从指定规范组态计算三维规范协变 Laplace低模，选择模态数、规范场处理与求解精度，并保存实际有序基底供后续测量使用。"
---

# LQCDweave · Laplace低模基底

共同任务路由、前端边界、保存/身份及执行范围统一见[技能共同约定](../README.md)。只读取当前任务的条件参考。

保留旧前缀时先核验同一算符下的正交性、残差和谱顺序，在实际旧空间的正交补求新增模式；
新完整基底记录新身份及旧前缀来源，不把独立重算的向量直接拼接。仅在用户明确要求续接已有低模时启用
`existing_basis`；接口与保存细节见[保留旧低模前缀](references/production-script.md#保留旧低模前缀)。

多 cfg 的任务分配、循环前共同准备及完成核验统一按 [Dispatch skill](../lqcdweave-dispatch/SKILL.md)；循环内消费当前 cfg 的规范场和实际时间记录，本 skill 负责基底计算与保存约定。

## 参数与执行范围

特别核对规范场版本及变换、实际时间与槽顺序、Ne、tol、输出精度与实际模式顺序。

核对实际科学参数与[执行范围](../README.md#execution-scope)，沿用已有授权。

用户参数不多，直接说明用法。需要规范组态、格点(Lx,Ly,Lz,Lt)、Ne、tol，以及是否对Laplace使用的规范场做明确变换。

```python
from lqcdweave.production.eigenvector.api import LaplaceEigenvectorGenerator
from lqcdweave.storage.dataset import DatasetWriter

generator = LaplaceEigenvectorGenerator(lattice, gauge_reader, Ne=Ne, tol=tol, backend="numpy")
generator.load(cfg)
with DatasetWriter(basis_directory, basis_spec, provenance=provenance) as basis_writer, \
        DatasetWriter(values_directory, values_spec, provenance=provenance) as values_writer:
    for slot, t in enumerate(range(lattice[3])):
        vectors, eigenvalues = generator.calc(t)
        basis_writer.write(slot, vectors)
        values_writer.write(slot, eigenvalues)
        del vectors, eigenvalues
    basis_id = basis_writer.publish()
    values_id = values_writer.publish()
```

`basis_spec`、`values_spec` 来自同一规范场、实际物理时间及模式顺序的科学定义，
`provenance` 记录实际来源；完整构造和续算示例见[生产脚本](references/production-script.md)。
库只提供计算与保存各自的接口，不再提供将两者绑定的基础产物生产入口。

gauge_reader从现有格式入口取得，不要求用户写load回调。NPY 用 `DenseFileHandle(prefix, suffix, format="npy")` 即可从头部读取 shape/dtype，显式提供时作为核验；ILDG 用 `IldgGauge` 从 XML 读取。vectors是(Ne,Lz,Ly,Lx,3)，eigenvalues与实际模式一一对应；保存组态、规范场版本、时间与顺序。相同Ne或生成参数不能代替实际基底身份。

用户明确要求时在load后用project_SU3或stout_smear处理Laplace的空间链接，不把它自动施加于Dirac或elemental的规范场。统一构造 `LaplaceEigenvectorGenerator`，`backend` 必填且不接受 `None`：字符串 `"numpy"/"cupy"/"quda"` 选择已有后端，否则传入实现 `LaplaceBackend` 协议的用户对象。字符串自动接好数组模块；自定义后端默认使用 NumPy，设备对象同时显式传 `array_module`。此入口不初始化 PyQUDA，也不自动回退。GPU用已支持的显式后端与环境，不从旧运行报告声称当前改动已验收。当前MPI域分解未接入。

生产脚本直接使用上述 load/calc 调用，并在下一片计算前保存基底和本征值。完整时间默认逐片计算；可选 `for t, vectors, eigenvalues in generator.iter_slices()` 是惰性时间循环，未传 selection 时为全 Lt，不堆叠输出，原有 `calc(t)` 保留；计算低模不隐式选择时间子集。tol是求解精度，输出存储精度另说明。生成器返回数组，文件保存由脚本完成；编写实际生产脚本时读取[脚本与保存示例](references/production-script.md)，不让用户填写内部工作区或后端协议实现。

用户明确启用时间稀疏化时，先用 `load_time_selection(selection_file, Lt=lattice[3])` 读取该 cfg 的实际选择文件；本 skill 不重新选择或平移，每个 cfg 使用自己的实际有序记录。再按记录中的 `times` 在真实物理时间计算；基底及本征值按 selection 顺序存入 `Nt` 个逻辑槽，槽 i 对应 `times[i]`。`Nt` 是存储时间轴长度，不是物理格点长度；实际规范场格点、传给生成器的 `lattice[3]` 和 Dirac 求解 `Lt` 都保持物理 `Lt`。生成器可直接传 `selection=selection`，随后 `iter_slices()` 只计算记录内的物理时间；下游 reader 必须消费相同映射；共同时间约定与限制见[生产与保存说明](../../../docs/modules/temporal_sampling/temporal_production.md)。

```python
from lqcdweave.storage.time_selection import load_time_selection

selection = load_time_selection(selection_file, Lt=lattice[3])
for slot, (t, vectors, eigenvalues) in enumerate(generator.iter_slices(selection=selection)):
    save_slice(slot, t, vectors, eigenvalues)  # 调用方同步保存同一槽的两项数据
```

生成与保存的接线见[生产脚本用法](references/production-script.md)；小格点 CPU 数值参考见[Laplace 回归](../../../tests/production/eigenvector/test_laplace_eigvector.py)，不作为物理系综数据或生产规模效率证明。

基底和本征值保存为同一 cfg、同一实际时间及模式顺序的一对输出；简单数组按每 cfg 一个完整 NPY 保存，例如布局分别为 `(Nt,Ne,Lz,Ly,Lx,3)` 和 `(Nt,Ne)`，并由共享根 `dataset.json` 登记。全时间时 Nt=Lt，稀疏时 Nt 为读回记录的长度。已有输出不能仅凭路径存在认定完整，也不能把新求解的本征值与旧基底直接拼接。默认格式和恢复边界见[数据保存说明](../../../docs/modules/storage/data_storage.md)；数值与平台限制见 [EV L2](../../../docs/modules/eigenvector/eigenvector.md)。

## 已有基底与点集的 overlap

从已保存的实际基底与点集读取同 cfg、同时间及原有顺序的数据；按需调用 `calc_overlap_row(eigenvectors, points, usedNe=None, usedNp=None, *, array_module=np)`，输入分别为 `(Ne,Lz,Ly,Lx,3)` 与 `(Np,3)` 的 XYZ 坐标，返回 owned `(Ne,Np,3)`。完整数组可用 `get_overlap_matrix`，其基底轴为 `(Nt,Ne,Lz,Ly,Lx,3)`、点轴为 `(Nt,Np,3)`，返回 `(Nt,Ne,Np,3)`。按所选格式保存，并在下游读取已保存结果；不额外共轭、归一化或乘抽样权重，不凭 Ne/Np 判断实际基底或点池身份。CPU/GPU 数组及保存方式由适用后端与格式选择，NPY 只是示例。

已有 `points` 为同 cfg、同时间槽顺序的 `SpatialPointSet` 时，直接交接其规范坐标：

```python
from lqcdweave.production.eigenvector.api import get_overlap_matrix

vectors = basis_reader.load(cfg)[:]
overlap = get_overlap_matrix(vectors, points.coordinates, usedNe=Ne, usedNp=Np)
```

只校验并读取请求的模态/点前缀；两入口都会拒绝负数、越界或同行重复坐标及所选向量中的 NaN/Inf。完整调用在分配结果前检查各时间行坐标，内部复用已接受的行，无需用户增加预检调用。

## 按需扩展

只在用户明确要求保留已有低模前缀并扩展最终 `Ne` 时，读取[保留旧低模前缀](references/production-script.md#保留旧低模前缀)。默认独立求解流程不读取旧基底。

H 点端投影需要完整 low+medium 父基底的实际正交归一向量、标签和物理时间；无需在 EV
阶段求解传播子。父基底声明 `normalization="orthonormal"`，由 PR 的 HighmodeBasis 消费。
按时间片读取、直接 H 源及事后派生的边界见[H 契约](../../../docs/modules/perambulator/point_highmode.md#h-contract)。

只读与当前任务有关的参考。普通恢复须保持同 cfg、实际基底及本征值配对、时间与有序模态覆盖；不以文件存在跳过未完成数据。
- 需要通用项目的跨阶段恢复：[项目恢复](../../../docs/modules/dataset_planning/dataset_planning.md#project-resume)
- 仅设备目标尚未登记时：[目标档案与生产内存估算](../../../docs/architecture/generic_production.md#unregistered-target)。不在默认 load/calc 增加独立预检调用；已登记设备的必要运行期内存检查不因此取消
