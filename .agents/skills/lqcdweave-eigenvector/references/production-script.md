# 本征基底生产脚本

代理使用 `LaplaceEigenvectorGenerator` 创建同一数值生成器，再组合 load/calc 与现有 writer；用户不编写底层回调。先从[数据格式入口](../../../../docs/modules/data_readers/data_readers.md#规范场与本征矢)取得 gauge_reader；不要要求用户编写 load 回调。NPY 读取可用 `DenseFileHandle(prefix, suffix, format="npy")`，无需重复写 shape/dtype；ILDG 用 `IldgGauge(prefix, suffix)`。需要统一显式选后端时，用 `LaplaceEigenvectorGenerator(lattice, gauge_reader, Ne=Ne, tol=tol, backend="numpy"/"cupy"/"quda")`，仍返回同一个生成器。配置 cfg、物理 lattice、Ne、tol、规范场变换及输出路径。默认全部时间；只有明确启用时间稀疏化时，才指定同 cfg 已保存的 selection_file。新建数据格式见[数据保存说明](../../../../docs/modules/storage/data_storage.md)。

新建数据默认按 cfg 保存完整 NPY，简单基底/本征值数组各自使用一个 NPY 文件，并通过共享根 `dataset.json` 登记；多字段数据用 HDF5。下面的片段分别使用基底与本征值的 `DatasetSpec` 和完整数组 writer。旧输入格式仍可读取，详见[默认数据格式](../../../../docs/modules/storage/data_storage.md)。

## NumPy / NPY 示例

下面是 CPU NumPy 新建输出的脚本主体。调用前提供 `cfg`、`lattice=(Lx,Ly,Lz,Lt)`、`gauge_reader`、`Ne`、`tol`、两个由当前科学定义构造的 `DatasetSpec`、共享的基底/本征值数据集目录和 provenance；两个 spec 都需绑定同一 cfg 与有序时间。全时间设 `selection_file=None`，稀疏时间设为该 cfg 的实际选择文件。

本片段用于新产物或同定义、同来源的未发布续算，显式启用 `resume=True`。双方已发布时核对登记后用 `DatasetReader` 读取复用，不再进入本片段；只发布了一侧时先恢复配对，不能重新打开已发布 writer 或只重解另一侧。

```python
import numpy as np

from lqcdweave.contracts.dataset import DatasetSpec
from lqcdweave.production.eigenvector.api import LaplaceEigenvectorGenerator
from lqcdweave.storage.dataset import DatasetWriter
from lqcdweave.storage.time_selection import load_time_selection

Lx, Ly, Lz, Lt = lattice
if selection_file is None:
    selection = None
    times = tuple(range(Lt))
else:
    selection = load_time_selection(selection_file, Lt=Lt)
    times = selection.times
Nt = len(times)

if not isinstance(basis_spec, DatasetSpec) or not isinstance(values_spec, DatasetSpec):
    raise TypeError("basis_spec and values_spec must be DatasetSpec values")
if (basis_spec.document["configuration"]["configuration"] != cfg
        or values_spec.document["configuration"]["configuration"] != cfg):
    raise ValueError("both DatasetSpec values must be bound to cfg")

generator = LaplaceEigenvectorGenerator(lattice, gauge_reader, Ne=Ne, tol=tol, backend="numpy", selection=selection)
generator.load(cfg)
# 明确需要空间 SU(3) 投影或 stout 时，在这里按约定顺序执行。

with DatasetWriter(basis_directory, basis_spec, provenance=provenance, resume=True) as basis_writer, \
        DatasetWriter(values_directory, values_spec, provenance=provenance, resume=True) as values_writer:
    basis_missing = set(basis_writer.missing)
    values_missing = set(values_writer.missing)
    if basis_missing != values_missing:
        raise ValueError("基底和本征值完成覆盖不一致；先恢复同一次求解的配对产物")
    for slot, t in enumerate(times):
        if slot not in basis_missing:
            continue
        vectors, values = generator.calc(t)
        if vectors.shape != (Ne, Lz, Ly, Lx, 3) or values.shape != (Ne,):
            raise ValueError("求解结果与输出布局不一致")
        basis_writer.write(slot, vectors)
        values_writer.write(slot, values)
    basis_id = basis_writer.publish()
    values_id = values_writer.publish()
```

此示例保留生成器的模式排序和相位约定；两个 spec 分别明确所需向量与本征值 dtype。规范场仍用物理 Lt；只把输出文件的时间轴按 Nt 槽组织。输出基底首轴为 Nt、本征值为 `(Nt,Ne)`，不补齐未选时间。所选物理时间、环绕顺序直接来自选择文件，脚本不再 select 或抽平移。读取 NPY 时保持相同槽顺序，见[共同稀疏时间约定](../../../../docs/modules/temporal_sampling/temporal_production.md)。

## 保留旧低模前缀

只有用户明确要求扩展已有基底时，才传 `existing_basis`。`Ne` 是扩展后基底的最终总模式数，旧基底模式数 `Nold` 必须满足 `0 < Nold <= Ne`。旧向量由 reader 提供，支持两种入口：

- 流式 reader 暴露 `Ne` 和 `read_modes(key, physical_t, start, stop)`；生成器按当前 cfg、实际物理时间读取旧模式。
- 数组 reader 暴露 `load(key)` 并返回 `(Nt,Nold,Lz,Ly,Lx,3)`；可额外提供 `time_selection`、`basis_id` 和有序 `mode_labels`。若提供 `mode_labels`，数量必须等于实际 `Nold`；NumPy 标量标签在扩展记录中转成 Python 标量。紧凑时间轴沿用它实际登记的时间映射，不从 `Nt` 推断物理时间。

```python
generator = LaplaceEigenvectorGenerator(
    lattice, gauge_reader, Ne=Ne_total, tol=tol,
    backend="numpy", selection=selection,
    existing_basis=old_basis_reader,
)
generator.load(cfg)
for slot, t in enumerate(times):
    vectors, values = generator.calc(t)
    # 在下一次 calc 前同步保存这一时间片及 describe()['extension']。
```

保留旧前缀、Gram 矩阵和独立稠密谱的 CPU 核验见[基底扩展回归](../../../../tests/production/eigenvector/test_extension.py)。它使用合成规范场，不提供物理系综或生产规模验收。

对每个实际物理时间，生成器先从旧 reader 取得原有有序模式，再按 `complex128` 建立副本；旧模式的顺序和相位保留，只有新增模式应用所选定相规则。它用当前已加载并完成空间变换的规范场重算旧模式的 Rayleigh 商，并验证旧基底 Gram 矩阵、逐模式归一化残差、旧谱升序以及旧子空间确为低谱前缀。扩展结果还要通过完整基底正交性与新增模式残差检查。Gram 谱范数误差及归一化残差阈值为 `max(10*tol, 1e-11)`；谱比较的松弛量是该阈值乘以 `max(1, max(abs(old eigenvalues)))`。投影到正交补后的新增求解向量还须通过最小奇异值 `>= 0.5` 的独立性检查。QR 只用于内部构造旧空间投影，不会改写输出中的旧向量。若输入与当前规范场或算符不匹配，生成器报错；它不通过正交化、排序或重新定相修补输入。

默认 NumPy/CuPy 数组路径在旧空间正交补中求新增模式。自定义 backend 或 QUDA backend 只有实现正交补求解扩展方法时才支持该操作；缺少该方法会明确报 `NotImplementedError`，不会回退到独立全空间求解。保留旧前缀时不支持 Chebyshev 加速。

`describe()['extension']` 记录最近一次成功 `calc(t)` 的旧基底来源和该时间片数值验证结果，包括 `method='preserved-prefix-complement-v1'` 和 `new_phase`。`new_phase` 为 `apply_renorm_phase=True` 时的 `'first_component_real'`，或关闭时的 `'solver_phase'`；旧前缀始终保留输入相位。`load`、规范场变换以及新一次 `calc` 开始时会清除该记录；因此每片算完后、下一片开始前就保存记录。未知的旧 `basis_id` 保持未知，不从路径、`Ne` 或 shape 补造。扩展后的完整基底是新数据集：其科学定义须登记实际旧数据集身份和保留前缀数量，发布时取得新的实际 `data_id`。不能沿用未登记前缀来源的普通 spec。

## 保存与续算的边界

基底和本征值必须来自同一次求解的同一模式顺序；同时保留 cfg、原物理格点、实际规范场版本及变换、求解参数、输出精度和对应选择文件。已有格式具备元数据或登记方式时沿用它，路径或参数不能代替实际基底身份。没有登记的 NPY 不自动获得 DatasetSpec 身份或来源证明。

writer 在生产期间把未完成范围放入临时 `.state`，成功发布后清除；只有登记为完成的 cfg 才能作为产物读取。高级崩溃恢复、跨进程并发与 retention 边界需按当前实现和验收核对，不能仅凭文件、shape 或有限值认定完整。不能只重新求解本征值并与旧基底拼接，简并子空间可能旋转。

逐时间片写出不会一次物化全部输出，但生成器仍常驻完整规范场，求解器另需空间工作区；内存映射不是整体内存上限。GPU 脚本显式配置数组模块和后端，并在写出前同步转为 NumPy；不自动初始化设备或提交集群作业。当前 GPU/MPI 验收范围见 [EV L2](../../../../docs/modules/eigenvector/eigenvector.md#后端与平台范围)。

两份产物的完成槽不同可能来自中断；先核对并恢复同一次求解的配对数据，或在新输出中一起重产。不能只重解缺失的一侧并与旧侧拼接。本例明确拒绝这种不一致，不假装自动修复。
