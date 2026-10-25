# Laplace 本征系统生产

用户生产脚本通过 `LaplaceEigenvectorGenerator.load/calc` 计算并按既有文件格式保存基底和本征值（脚本指南以 NPY 为例）；默认覆盖全 `range(Lt)`。明确稀疏时间时，先读取同 cfg 的选择记录，再按其中 `selection.times` 的实际物理时间计算，并按原顺序保存 Nt 个逻辑槽。物理规范场与求解仍使用原 `Lt`。完整脚本见[EV 生产脚本指南](../../../.agents/skills/lqcdweave-eigenvector/references/production-script.md)。下文说明数值生成器及其后端。

`LaplaceEigenvectorGenerator` 对每个时间片求三维规范协变 Laplacian 的最低 `Ne` 个本征对，采用周期空间边界。算符是

统一从 `lqcdweave.production.eigenvector.api` 导入 `LaplaceEigenvectorGenerator`；`backend` 必填且不接受 `None`。包级 `lqcdweave.production` 也导出同一个类。类状态、异常失效语义、所有权和单时间片内存边界见 [EV L2](eigenvector.md)。

```text
L V(x) = 6 V(x) - sum_i [U_i(x) V(x+i) + U_i(x-i)† V(x-i)]
```

`latt_size` 顺序为 `(Lx,Ly,Lz,Lt)`；`gauge_field.load(key)` 须返回复数数组 `(Lt,Lz,Ly,Lx,4,3,3)`，最后两个轴为颜色矩阵。第 5 轴的方向顺序为 x、y、z、t。生成器复制输入，空间 stout 或 SU(3) 投影不修改加载器持有的数组，也不修改时间链接。默认计算全 `Lt` 个时间片；调用者逐时间片运行并及时保存输出。

```python
from lqcdweave.production.eigenvector.api import LaplaceEigenvectorGenerator

gen = LaplaceEigenvectorGenerator((4, 4, 4, 8), gauge_field, Ne=20, tol=1e-9, backend="numpy")
gen.load("weak_field")
gen.stout_smear(10, 0.12)  # 可选；作用于当前已加载的空间链接
for t in range(8):
    vectors, values = gen.calc(t)
    # vectors: (Ne,Lz,Ly,Lx,3), complex128
    # values: (Ne,), float64，升序
    # 在这里按现有生产文件格式保存；不要仅凭 Ne/参数推断实际基底身份。
```

### 显式稀疏时间

默认保持完整时间。只有需要稀疏时间时才读取同一 cfg 已保存的记录：

```python
selection = load_time_selection(selection_file, Lt=lattice[3])
gen = LaplaceEigenvectorGenerator(lattice, gauge_reader, Ne=Ne, tol=tol, backend="numpy",
                               selection=selection).load(cfg)
for slot, (t, vectors, values) in enumerate(gen.iter_slices()):
    basis[slot] = vectors
    eigenvalues[slot] = values
```

输出首轴为 `gen.Nt`，第 `slot` 项对应 `selection.times[slot]`；`gen.Lt` 和规范场仍是物理时间长度。迭代只求所选片，不先生成全时间输出再截取。不指定构造参数时，也可显式使用 `gen.iter_slices(selection=selection)`；它和旧的 `times` 参数不能同时给出。`calc(t)` 始终接收单个物理时间，保留旧行为。

保存同一份选择记录及实际基底身份，后续生成器用其映射读取紧凑输入。选择覆盖、真实基底、精度与求解容差是不同信息；规范场仍完整常驻，稀疏迭代不保证总体内存上限。

### 文件读取与后端选择

NPY 的 shape/dtype 已在文件头中，不必重复填写；需要核对已有声明时仍可显式传入。
ILDG 已从 XML 读取格点与精度。物理 `lattice`、`Ne`、`tol` 仍由计算脚本明确给出：

```python
from lqcdweave.adapters import DenseFileHandle
from lqcdweave.production.eigenvector.api import LaplaceEigenvectorGenerator

gauge_reader = DenseFileHandle("/data/gauge.", ".npy", format="npy")
gen = LaplaceEigenvectorGenerator(lattice, gauge_reader, Ne=Ne, tol=tol, backend="numpy")
gen.load(cfg)
# gen.stout_smear(nstep, rho)  # 仅在明确需要时调用
for t, vectors, values in gen.iter_slices():
    # 按现有格式保存同一时间、同一模式顺序的两组数组
    pass
```

`backend` 没有默认值：`"numpy"` 选择 CPU NumPy/SciPy，`"cupy"` 选择 CuPy/CuPyX，
`"quda"` 选择 CuPy 与新建的 `QudaLaplaceBackend`。字符串入口自动接好数组模块；
显式 `array_module` 与名称冲突时拒绝。不自动探测设备、初始化 PyQUDA 或在失败时换求解器。
`quda` 要求调用前已有配置好的单 rank PyQUDA 环境；缺失依赖的异常直接传播。
非字符串 `backend` 作为用户提供的后端对象，须实现 `LaplaceBackend` 协议；默认以 NumPy
交付数组，设备对象需同时传入相应 `array_module`。用户脚本统一直接构造
`LaplaceEigenvectorGenerator`，并显式指定 `backend`。

`iter_slices()` 是可选的惰性时间循环，按生成器的 `times` 逐次返回 `(t, vectors, values)`；未配置 selection 时，才等价于对 `range(Lt)` 调用 `calc(t)`。它不会堆叠全时间输出。显式 `iter_slices(times)` 保留给定
物理时间的顺序，不自动选择稀疏时间。相位与 Chebyshev 选项同 `calc`，以关键字传入。
先 load/变换，再消费迭代器；不要在两次 yield 间切换或变换规范场。
也可显式使用 `for t in range(Lt): vectors, values = gen.calc(t)`。

`calc(t, apply_renorm_phase=True)` 默认约定：本征值升序排列，每个向量用第一个分量定相。简并子空间内的逐向量方向仍可能随求解器变化。`describe()` 返回已知的 gauge 身份、变换记录和加载状态；它不是持久产物清单。生产求解不额外扫描全体本征向量的残差或正交性；这些数值性质由独立参考测试验收。

CPU 求解需要 SciPy，可用 `python -m pip install -e ".[laplace]"` 安装。传入 `backend="cupy"` 时使用 CuPy/CuPyX 数组和求解器；导入软件包本身不会初始化 GPU。CuPy 上执行非零 stout 步数需要 QUDA 或支持该操作的自定义后端；NumPy 仍可直接执行 stout。已有单进程 PyQUDA 环境可这样配置：

```python
gen = LaplaceEigenvectorGenerator(
    latt_size, gauge_field, Ne=120, tol=1e-9,
    backend="quda",
)
```

该适配器默认用 QUDA Laplacian 加 CuPyX eigsh；若在 `calc` 中同时给出正的 `polynomial_degree` 和 `lambda_cut`，则使用 QUDA Lanczos 与 Chebyshev 加速。它要求偶数 `Lx`，并仅支持单 MPI rank。

用户提供的 `backend` 对象实现 `contracts.laplace.LaplaceBackend`：`load(handle,key,latt_size)` 返回独立持有的方向优先链接，`stout_smear` 和 `solve` 执行数值操作，`invalidate()` 丢弃过期的原生表示。每个生成器独占一个后端实例。例如：

```python
gen = LaplaceEigenvectorGenerator(
    latt_size, gauge_field, Ne=Ne, tol=tol,
    backend=my_solver, array_module=my_array_module,
)
```

QUDA 原生对象识别、句柄中的文件定位以及 QIO 加载只在适配器内进行；只有明确声明 `format="ildg"` 或 `"qio"` 的句柄使用 QIO 文件路径，其它 NPY/raw/QDP 句柄使用其数组读取结果。`IldgGauge` 已自带格式声明。生产器不再访问这些对象或路径。SU(3) 投影、加载或变换失败会使后端缓存失效；连续 stout 可以复用有效原生表示。传入原生 QUDA gauge 时，后端接管其变换用途，调用者不得同时使用或修改它；普通数组输入仍会复制。

CPU 独立矩阵与模拟原生对象的生命周期测试已覆盖该边界；当前源码的真实 GPU 数值、性能与 MPI 仍待验收。

`project_SU3()` 可选，使用极分解和行列式校正投影空间链接。变换顺序会改变实际基底；下游传播子和 elemental 必须引用同一次生成的基底。输出按已声明选择组织为 `(Nt,Ne,Lz,Ly,Lx,3)`；未选择时 Nt=Lt。下游需要同一实际时间映射。用户脚本按既有文件格式保存结果，并保留其 cfg、时间顺序和槽覆盖信息；数值生成器不发布产物或自动恢复缺失时间片。
