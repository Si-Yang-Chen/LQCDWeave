# EV：Laplace 本征矢生产 L2

EV 数值 API 由用户脚本通过 `LaplaceEigenvectorGenerator.load/calc` 调用，并按既有文件格式保存（脚本指南使用 NPY）；全时间与明确稀疏时间示例见[EV 生产脚本指南](../../../.agents/skills/lqcdweave-eigenvector/references/production-script.md)。`plan_indexed_laplace_basis` 仅作只读预览，Wilson 参考脚本同样显式组合计算和保存，见[生产脚本与保存](../../architecture/generic_production.md#laplace-入口)。

L0/L1 交接见 [production-l0-l1.md](../../architecture/production-l0-l1.md#ev规范场--实际有序基底)。本模块接收显式规范场句柄、必填后端名称或 `LaplaceBackend` 对象，执行三维规范协变 Laplacian 的空间变换与低模求解。构造时仅通过 `adapters.laplace.resolve_laplace_backend` 接线；它不定位文件、不解析外部格式、不初始化设备、不登记持久产物，也不导入 Elemental 或 Perambulator 的实现。

## 源码和公共入口

| 路径 | 职责 |
|---|---|
| `src/lqcdweave/production/eigenvector/api.py` | 唯一公开模块 API |
| `src/lqcdweave/production/eigenvector/generator.py` | `LaplaceEigenvectorGenerator` 状态、加载、空间变换、后端调用与排序/定相 |
| `src/lqcdweave/production/eigenvector/extension.py` | 旧前缀质量门槛、正交补求解、扩展模式验证 |
| `src/lqcdweave/production/eigenvector/laplacian.py` | 数组形式空间 Laplacian |
| `src/lqcdweave/contracts/laplace.py` | `LaplaceBackend` 的唯一协议定义 |
| `src/lqcdweave/adapters/laplace.py` | `resolve_laplace_backend` 解析名称/对象，不初始化 runtime |
| `src/lqcdweave/adapters/laplace_quda.py` | QUDA 原生对象/QIO 路径转换及单进程 PyQUDA 执行 |

推荐导入：

```python
from lqcdweave.production.eigenvector.api import LaplaceEigenvectorGenerator
```

包级 `lqcdweave.production` 与模块 API 导出同一个 `LaplaceEigenvectorGenerator` 类。

## 类、构造参数和状态字段

### `LaplaceEigenvectorGenerator`

```text
LaplaceEigenvectorGenerator(
    latt_size, gauge_field, Ne, tol, *,
    backend: str | LaplaceBackend,
    array_module=None,
    selection=None,
    existing_basis=None,
)
```

`latt_size` 必须有四个正整数，顺序是 `(Lx,Ly,Lz,Lt)`；布尔值不作为整数接受。空间维数为 `D=3*Lx*Ly*Lz`，`Ne` 必须满足 `1 <= Ne <= D-2`。`tol` 必须是有限浮点数且 `0 < tol < 1`。`backend` 必填且不接受 `None`：字符串仅支持 `numpy/cupy/quda`，其它值作为用户后端对象。名称决定数组模块；显式 `array_module` 与名称冲突时报错。用户后端对象未指定数组模块时使用 NumPy，指定时其 `__name__` 必须是 `numpy` 或 `cupy`。

`gauge_field` 是显式句柄，至少提供 `load(key)`；NumPy/CuPy 数组求解路径中其返回值必须能按数组切片，形状为 `(Lt,Lz,Ly,Lx,4,3,3)`，dtype 为复数。可选 `gauge_id` 只作为描述信息快照读取；缺失时保持 `None`。用户提供的 `backend` 对象必须实现 `contracts.laplace.LaplaceBackend` 的 `load`、`invalidate`、`stout_smear`、`solve` 四个方法。核心不会检查 QUDA 对象类型或文件扩展名。

构造完成后公开/内部字段如下；带下划线字段是实现状态，不是消费接口：

| 字段 | 初始值/含义 |
|---|---|
| `latt_size` | `(Lx,Ly,Lz,Lt)` 整数元组 |
| `gauge_field` | 调用者提供的句柄引用 |
| `Ne`、`tol` | 已验证的低模数和求解容差 |
| `xp` | `numpy` 或 `cupy` 模块对象 |
| `backend` | 后端对象；缺省时核心直接用数组模块求解 |
| `existing_basis` | `None` 表示普通独立求解；非空时作为旧有序模式的 reader，`Ne` 表示扩展后总数 |
| `_links` | `None`，成功 `load` 后为方向优先、复数、生成器持有的完整规范场 |
| `_key`、`_gauge_id` | `None`，成功加载后分别记录调用 key 和 gauge 身份快照 |
| `_transforms` | 空列表；成功应用的空间变换按调用顺序记录 |
| `_extension` | `None`，最近一次成功扩展计算的来源与数值验证记录；普通求解时保持 `None` |

类不生成新本征基底 ID，也不把 `(Ne,tol,seed/shape)` 组合解释为基底身份。`describe()['extension']` 可带有 reader 声明的旧 `basis_id`，未知时保持 `None`；它是最近一次计算的诊断/来源记录，不是产物清单或持久登记。用户脚本按既有文件格式保存实际有序向量、扩展来源和逐时间记录；发布完整新基底时由实际 Dataset 发布生成新身份。

## 显式时间覆盖

`LaplaceEigenvectorGenerator(..., backend="numpy", selection=None)` 接受共享时间选择记录。物理 `Lt` 必须等于 `latt_size[3]`；`Nt` 为记录长度，`times` 与 `time_selection` 暴露实际顺序。未传 selection 时为全部物理时间，旧数值默认不变。`describe()` 包含 `time_selection` 文档及 `Nt`，但不是持久基底身份。

`iter_slices(selection=record)` 可覆盖本次迭代顺序，不与旧 `times` 混用；逐片直接调用原 `calc(physical_t)`，没有全时间求解后截取或补零。`calc(t)` 仍可计算任意合法物理时间。完整规范场、后端和空间求解内核不变，输出所有权仍是 owned；循环保存时以枚举槽存成 `(Nt,Ne,Lz,Ly,Lx,3)` 与 `(Nt,Ne)`，保留同一记录。

`tests/production/eigenvector/test_sparse_time.py` 用独立全时间稠密 Laplacian 本征系统选择参考，核对环绕槽序、只调用所选时间、谱与子空间；平台范围为合成小格点 CPU，未增加 GPU/MPI 验收。

## 保留已有低模前缀

默认入口保持独立求解。用户明确要求延长旧基底时，才传 `existing_basis`；构造器的 `Ne` 是最终总模式数，旧模式数 `Nold` 由 reader 提供且须满足 `0 < Nold <= Ne`。reader 可选择以下一种数据入口：

- 流式入口须提供整数 `Ne` 和 `read_modes(key, physical_t, start, stop)`，返回 `(stop-start,Lz,Ly,Lx,3)`；每次 `calc(t)` 按当前 key 和物理时间读取完整旧前缀。
- 数组入口提供 `load(key)`，返回 `(Nt,Nold,Lz,Ly,Lx,3)`。reader 可声明 `time_selection`、`basis_id`、`mode_labels`；若时间轴为紧凑子集，须提供其物理时间映射，未声明时按完整 `Lt` 处理。若提供标签，其数量须等于实际 `Nold`；NumPy 标量标签在扩展记录中转为对应的 Python 标量。

每片计算时，旧向量转换为生成器数组模块的 `complex128` 副本，保留原顺序和相位；对 `complex128` 输入数值原样复制。生成器使用**当前实际规范场以及已完成的空间变换**重算旧模式 Rayleigh 商，并在其实际子空间的正交补中求新增低模。它核验旧基底及最终基底的 Gram 矩阵、旧/新增模式的归一化算符残差、旧模式升序，以及旧空间确为最终低谱的完整前缀。用于 Gram 谱范数误差和归一化残差的阈值为 `max(10*tol, 1e-11)`；谱前缀比较的松弛量为此阈值乘 `max(1, max(abs(old eigenvalues)))`。新增求解向量投影入旧空间正交补后，其最小奇异值还须至少为 `0.5`，以拒绝失效或重复的 complement 结果。QR 只构造内部投影空间，不会归一化、重排或重新定相旧输出。错规范场、非正交模式、残差超限、谱顺序错误或非完整低谱前缀都会报错，不会修补旧输入。

`describe()['extension']` 是最近一次成功扩展 `calc(t)` 的记录，包含 `method='preserved-prefix-complement-v1'`、`new_phase`、`prefix_count`、`added_count`、`threshold`、`prefix_residual`、`new_residual`、`gram_error`、`source_basis_id`、`source_mode_labels` 和 `physical_time`。`new_phase` 在 `apply_renorm_phase=True` 时为 `'first_component_real'`，否则为 `'solver_phase'`；两种情况下旧前缀相位都保持不变。`load`、规范场变换以及每次新 `calc` 开始时清空该记录；脚本须在下一片计算前保存当前时间记录。未知 `basis_id` 不由路径、shape 或模式数推断。扩展完整基底时，输出科学定义须登记实际旧数据集身份和保留前缀数量；发布时获取新 `data_id`，不能沿用未登记前缀来源的普通 spec。

NumPy/CuPy 数组后端直接在正交补中求解。外部 backend 必须额外提供 `solve_complement(links, latt_size, t, count, tol, prefix)`；缺少此扩展方法时会抛 `NotImplementedError`，不会回退到全空间独立求解。扩展路径不接受 Chebyshev 参数。接口说明见[EV 生产脚本指南](../../../.agents/skills/lqcdweave-eigenvector/references/production-script.md#保留旧低模前缀)，保留前缀/独立谱 CPU 回归见 [test_extension.py](../../../tests/production/eigenvector/test_extension.py)。

## 数组轴、数值约定和解法

输入 `latt_size=(Lx,Ly,Lz,Lt)` 是几何元组；原始 gauge 数组布局为 `(Lt,Lz,Ly,Lx,4,3,3)`，方向轴次序为 x、y、z、t，末两轴是颜色矩阵。无外部后端时 `load` 将其转成方向优先 `(4,Lt,Lz,Ly,Lx,3,3)`，并复制为 `complex128`。后端的 `load(handle,key,latt_size)` 须直接返回此方向优先形状的 owned complex 数组。

每次 `calc(t)` 只选取一个时间片的三个空间方向，空间场布局为 `(3,Lz,Ly,Lx,3,3)`。周期空间边界下使用

```text
L V(x) = 6 V(x) - Σ_{i=x,y,z} [ U_i(x) V(x+i) + U_i(x-i)† V(x-i) ]
```

邻居由 `roll` 取值。CPU 默认路径构造复数 `LinearOperator` 并调用 `scipy.sparse.linalg.eigsh(k=Ne, which="SA", tol=tol)`；CuPy 默认路径对应 `cupyx.scipy.sparse.linalg.eigsh`。后端路径把整个方向优先 gauge、几何、`t`、`Ne`、`tol` 与 Chebyshev 参数交给 `LaplaceBackend.solve`。

求解结果按本征值升序排序，向量按相同置换排序。默认 `apply_renorm_phase=True` 时，每个向量展平后以第一个分量定相：乘以 `exp(-i angle(v[0]))`；首分量为零时 NumPy/CuPy 的 `angle(0)=0`，不会另选非零分量。此规则固定整体相位，不固定严格简并或近简并子空间内的向量方向；求解器变化时该子空间基底可以旋转。返回的向量被强制复制为 `complex128`；本征值仅取实部、不额外 cast，当前 SciPy/CuPyX/QUDA 路径返回 `float64`。返回数组的 mode 顺序是已排序顺序，不能仅以 `Ne` 判断与已有基底相同。

`spatial_laplacian(field, links, xp=numpy)` 是公开数组算符：`links` 形状 `(3,Lz,Ly,Lx,3,3)`；`field` 可为 `(Lz,Ly,Lx,3)` 或 `(Lz,Ly,Lx,3,Nrhs)`。有 `Nrhs` 轴时它被保留，内部 `einsum` 一次处理全部 RHS。生产 eigensolver 的 `eigsh` 输入是一列向量；该多 RHS 形式是算符能力，不表示当前 eigensolver 批量求多套基底，也不代表按模态或时间分块加载。

## 方法接口与返回值

| 方法 | 签名 | 行为/返回 |
|---|---|---|
| `load` | `load(key)` | 清空旧状态，加载并复制/接收完整 gauge；成功返回 `self`。不做持久化。 |
| `project_SU3` | `project_SU3()` | 对全部空间链接做 SVD 极分解并修正行列式相位；时间方向链接不变。成功返回 `self`。 |
| `stout_smear` | `stout_smear(nstep, rho)` | 按空间 staple 顺序更新空间链接，保留时间方向链接；成功返回 `self`。`nstep=0` 或 `rho=0` 时不做变换、不添加记录。 |
| `calc` | `calc(t, apply_renorm_phase=True, polynomial_degree=0, lambda_cut=0.0)` | 返回 `(vectors, values)`；形状分别为 `(Ne,Lz,Ly,Lx,3)` 和 `(Ne,)`。普通路径按谱升序；扩展路径保留旧前缀顺序，并追加其后新增模式。 |
| `iter_slices` | `iter_slices(times=None, *, selection=None, apply_renorm_phase=True, polynomial_degree=0, lambda_cut=0.0)` | 惰性逐次返回 `(t,vectors,values)`；默认按构造时的 selection（未指定为 `range(Lt)`），显式时间按原顺序；times 与 selection 互斥。委托 `calc`，不缓存输出；扩展记录须在下一次迭代前读取/保存。 |
| `describe` | `describe()` | 返回当前配置参数、key、gauge ID 快照、变换记录副本、`loaded` 标志和最近一次扩展记录；不返回数组或新基底 ID。 |
| `_require_loaded` | `_require_loaded()` | 内部检查；未加载或变换失败后抛 `RuntimeError`。 |
| `_solve_array` | `_solve_array(links)` | 内部 CPU/CuPyX `LinearOperator` 与 eigsh 路径，接收单时间片空间链接。 |

`calc` 的 `polynomial_degree` 默认 0 且必须为非负整数；`lambda_cut` 默认 0 且必须有限、非负且小于 12。二者必须同时为零，或同时为正。正值只有显式 backend 时可用，核心无后端路径抛 `NotImplementedError`。该参数组合本身不保证任意 backend 实现 Chebyshev；具体能力属于 `LaplaceBackend` 合同实现，当前 QUDA adapter 提供 Lanczos/Chebyshev 分支。

模块内部辅助函数签名为 `_integer(value, name, minimum=0)`、`_scalar(value)`。它们负责整数参数验证和同步读取数组标量，不是公开入口。

## 状态、异常与失败语义

| 操作 | 成功后的状态 | 失败时的状态 |
|---|---|---|
| 构造 | 未加载；`_links/_key/_gauge_id=None`，变换列表为空 | 参数非法抛 `ValueError`；对象不构造 |
| `load(key)` 开始 | 先清除链接、key、gauge ID、变换记录、已读旧基底和扩展记录；后端存在时先 `invalidate()` | 任何读取、shape/dtype 或转换错误均不恢复旧 cfg；后端读取/布局错误再次 `invalidate()` 后原样抛出 |
| `load` 成功 | `_links` 有完整 cfg 链接，key 和 gauge ID 快照更新，返回 `self` | — |
| 变换参数错误 | 已加载的状态保持原样；例如负 `nstep`、非有限 `rho` 在执行前报错 | `ValueError`，不清除当前 gauge |
| SU(3)/stout 数值执行成功 | `_links` 更新；在 `_transforms` 追加成功变换记录；清除此前扩展记录；后端缓存按协议失效/替换 | — |
| SU(3)/stout 执行异常 | — | `_links=None`，backend `invalidate()`；异常向上传播。key/gauge ID 与此前成功变换记录仍留作诊断，但 `loaded=False`；失败变换不会追加记录 |
| `calc(t)` 开始/成功 | 开始先清除前一片扩展记录；普通求解返回新结果，扩展求解在成功后把来源和质量记录写入 `describe()['extension']`。两者均不标记持久产物完成 | — |
| `calc` 时间/参数错误或求解异常 | 当前已加载 gauge 和变换状态保留；失败时扩展记录为空，可由调用者决定是否重试 | 参数错误 `ValueError`，未加载 `RuntimeError`，依赖/求解/backend 原异常向上传播；没有结果对象返回 |

原始数组路径检查复数 dtype 和精确轴形；backend 的 `load` 还检查返回链接的复数 dtype 与布局。普通独立求解的 `calc` 不会额外验证 backend 返回向量/本征值的正交性、收敛标志或 Ritz 残差。CPU `eigsh` 的未收敛异常会传播。扩展路径则按上文阈值检验旧前缀和新增结果；`describe()['extension']` 仅记录通过质量门槛的最近一次扩展片，不代表数据已发布。backend 必须遵守 `LaplaceBackend` 协议并自行处理自身 solver 状态。

## 所有权、输出与内存边界

无 backend 的 `load` 将输入转换成生成器独立持有的 `complex128` 链接数组，不修改输入。backend 路径依赖 `LaplaceBackend.load` 的 owned-return 合约。SU(3)/stout 只改生成器持有的链接；backend stout 必须返回 owned 数组，生成器保留旧时间链接值。普通 `calc` 将排序后的向量强制复制为 `complex128`；扩展 `calc` 同样返回独立数组，旧前缀只转为 `complex128` 副本，不做正交化、重排或重新定相。调用者拥有返回值，可保存或长期保留，不会被下一次 `calc` 覆写。`spatial_laplacian` 的结果是运算结果，不修改输入场或链接。`describe()` 每次新建字典；`gauge_id`、`transforms` 与扩展记录是深复制，参数字段是标量/元组，`key` 字段保留原对象引用。

CPU 无 backend 的 stout 把生成器拥有的完整 `(direction,time,z,y,x,color,color)` 链接交给 `production.gauge.stout_smear`，显式使用 NumPy 和 `inplace=True`。共享核保留 time 轴，空间轴索引为 `3-direction`。加载时链接已转换为生成器独立持有的 `complex128` 数组，但不保证 C 或 Fortran 连续；共享核原位更新该私有数组、保留时间方向，并避免另建完整四方向输出；每一步仍需空间链接快照及矩阵工作区，不能视为无临时内存的路径。若原位变换中途失败，生成器丢弃 `_links`，不会继续使用部分更新的数据。QUDA 路径仍要求 backend 返回独立结果。

时间轴是**按调用计算**，不是**按时间片读取**：一次 `load` 常驻一个 cfg 的完整 `(4,Lt,Lz,Ly,Lx,3,3)` gauge；`calc(t)` 的 solve 输入与低模输出限于一个 `(Lz,Ly,Lx)` 时间片，但求解器 Krylov/Lanczos、算符和临时向量工作区仍按该空间体积和模态数量增长。脚本默认逐 `t in range(Lt)` 计算；基底与本征值作为简单数组分别按 cfg 保存完整 NPY，并由共享 `dataset.json` 登记。明确稀疏化时，先用 `load_time_selection(path, Lt=Lt)` 读取同 cfg 记录，再按其实际 `times` 调用 `calc(t)`，并按同槽顺序保存 Nt 行。规范场和求解器仍使用物理 Lt。EV 本身没有写回 callback、时间块读取或高级恢复协议。空间 stout 对完整链接场更新并需临时空间数组；不能把每次单 slice solve 描述为全链路超内存生产。脚本前端及保存示例见[EV 生产脚本指南](../../../.agents/skills/lqcdweave-eigenvector/references/production-script.md)与[数据保存说明](../storage/data_storage.md)。

`planning.laplace.estimate_laplace_memory` 是独立的纯逻辑估算器，按 gauge metadata、Ne、basis/eigenvalue 行、完整 gauge、全 Lt 本征值缓存及已知 operator/copy 临时量估算；它不读取 payload，不由 EV 脚本自动调用，也不执行资源准入或恢复。当前模型描述普通独立求解，不计入旧基底扩展额外的投影、Gram 矩阵、QR/SVD 和旧模式副本工作区。ARPACK、CuPyX、QUDA 或 custom backend 的原生工作区也不由生成器测量，所以估算不是 RSS 或 solver scratch 硬上限。完整估算边界见[Laplace 生产内存模型](../../architecture/generic_production.md#laplace-生产内存模型)。

## 后端与平台范围

`LaplaceBackend` 是唯一后端协议，方法为：

```text
load(handle, key, latt_size) -> links[4,Lt,Lz,Ly,Lx,3,3]
invalidate() -> None
stout_smear(links, latt_size, nstep, rho) -> owned_links
solve(links, latt_size, t, count, tol, polynomial_degree, lambda_cut)
    -> (vectors[count,Lz,Ly,Lx,3], values[count])
```

旧基底扩展不改变 `LaplaceBackend` 基本协议；选择 `existing_basis` 时，backend 还须显式提供 `solve_complement(links, latt_size, t, count, tol, prefix)`，返回 `(vectors[count,Lz,Ly,Lx,3], values[count])`。缺少此方法时报 `NotImplementedError`，不回退到 `solve` 的独立全空间求解。数组后端使用 NumPy/SciPy 或 CuPy/CuPyX 正交补求解。扩展与 Chebyshev 不兼容；不能据此推断 QUDA 已实现或通过 GPU 验收。

backend instance 由一个 generator 独占，避免共享可变 native cache。构造器仅向 `adapters.laplace.resolve_laplace_backend` 委托后端接线，不探测 native QUDA 对象、不解析文件名、不做 QIO 读取、不创建 CUDA context。`QudaLaplaceBackend` 把这些操作留在 `adapters/laplace_quda.py`；在用户显式构造 adapter 并提供 CuPy，或构造 `LaplaceEigenvectorGenerator(..., backend="quda")` 时走该路径。

- **CPU：** NumPy + SciPy 路径；独立稠密矩阵、低模本征值、正交性、算符残差、空间 stout、SU(3) 投影和失败状态由 CPU tests 覆盖。SciPy 为可选 `laplace` 依赖。
- **GPU：** CuPy 可执行 CuPyX eigsh；单进程 PyQUDA adapter 需要已经配置的 runtime，且当前文档要求偶数 `Lx`、匹配本地格点。当前源码的真实 GPU 数值和性能仍待验收。
- **MPI：** 尚未接入域分解或 rank 间归约；QUDA adapter 只支持单 rank。不能据 CPU 结果推断 MPI 能力。

当前纯 EV 独立参考在 `Lx=Ly=Lz=Lt=2` 的随机 SU(3) 小格点上取 `Ne=3`：本征值与直接构造的稠密矩阵低谱比较（绝对容差 `1e-9`），Gram 矩阵与单位阵比较（`1e-8`），逐向量 `L v - λv` 与零比较（`1e-8`，`numpy.testing.assert_allclose` 的默认相对容差仍生效）。单位链接简并零模另比较特征值及残差（`1e-9`）；模拟 backend 路径也核对算符残差。这里的残差检查是测试参考，不是生成器的运行时接受阈值；求解器的 `tol` 与测试残差容差不等价。

## 来源和测试归属

纯 EV 数值测试位于 `tests/production/eigenvector/test_laplace_eigvector.py`；可用 `python -m unittest discover -s tests/production/eigenvector -t tests -v` 单独运行。`tests/fixtures/laplace_reference.py` 提供共享的 `GaugeHandle`、`random_gauge`、`dense_laplacian` 和 `scipy` 数值参考 fixtures，供通用后端测试、`test_wilson_pipeline.py` 与 rho–ππ CPU 参考测试使用。通用后端与失败失效测试在 `tests/production/eigenvector/test_laplace_backend.py`，QUDA adapter 模拟测试在 `tests/adapters/test_laplace_backend.py`，共享后端夹具在 `tests/fixtures/laplace_backend.py`；真实 weak-field/QUDA 平台测试保留原位，执行结果须按实际环境另行核验。

旧前缀扩展测试位于 `tests/production/eigenvector/test_extension.py`，覆盖独立稠密谱、前缀逐元素保留、错误前缀拒绝、简并相位、稀疏物理时间映射、backend 能力边界及紧凑 Dataset 发布/读回。它们是小格点 NumPy 验证，不代表 GPU 或 MPI 验收。

## Overlap 公开接口

calc_overlap_row 和 get_overlap_matrix 共用一个 gather，返回独立数组，保持 eta-dagger-xi 的无共轭、无缩放定义。脚本保存实际基底和点集的有序视图；只读规划/可选 storage 校验覆盖与实际身份；gather 不要求正交，投影消费方另检查 orthonormal。精确签名、异常和验收见[Overlap](../perambulator/point_highmode.md#overlap)。

两个公开入口都保留首次输入校验。完整调用在结果分配前逐时间行验证所选点的范围与唯一性，
随后通过私有 gather 复用该调用内已接受的坐标，不再绕回公开逐行入口重复扫描。
逐行独立调用仍自行验证坐标；所选模态的有限性检查、前缀以外不读取及输出所有权保持不变。
`tests/production/eigenvector/test_overlap.py` 同时计数每时间行一次坐标检查、检查尾行坏坐标
在结果分配前失败，并与独立逐模态/点/颜色标量索引比较；这是 CPU 行为验证，不是 GPU 测速。

## 简洁后端前端与读取边界

`LaplaceEigenvectorGenerator(latt_size, gauge_field, Ne, tol, *, backend, ...)`
统一接收显式后端：`numpy` 对应 NumPy/无注入后端，`cupy` 对应 CuPy/无注入后端，
`quda` 对应 CuPy/新建 `QudaLaplaceBackend`。未知名称抛 `ValueError`，依赖或后端错误
原样传播，没有自动选择和 fallback。非字符串按现有协议接收用户后端对象。解析服务不负责配置 PyQUDA
runtime、加载 cfg、smearing、保存、登记或调度。用户脚本统一直接构造
`LaplaceEigenvectorGenerator`；包级/API 导出同一类，且都要求显式 `backend`。

QUDA 仅对声明 `format="ildg"` 或 `"qio"` 的句柄读取文件路径；普通 `file` 属性或
扩展名不证明 QIO 格式。`IldgGauge.format` 固定为 `"ildg"`；NPY、裸 binary 和 QDP
通过其数组读取结果转换为方向优先的 owned links，仍验证精确 shape/复数 dtype。
`DenseFileHandle` 的 NPY shape/dtype 可省略，从头部读取；显式声明继续严格核对。
裸 binary 仍必须指定 shape/dtype；reshape/transpose 始终显式，不从轴长度猜语义。

`iter_slices` 不创建输出容器，不改变 `calc` 的升序、定相或所有权。每次推进才计算下一片，
参数/加载/求解异常在相应迭代时传播；空显式时间序列不触发计算。迭代器读取生成器的当前
状态，因此调用者须在遍历前完成加载与变换，不能在迭代中切换或变换 gauge。
专项测试位于 `tests/adapters/test_laplace_frontend.py`、`tests/adapters/test_laplace_backend.py`
与 `tests/production/eigenvector/test_iteration.py`。模拟 CuPy/QUDA 仅验证显式接线，
不构成真实 GPU/MPI 数值或性能验收。
