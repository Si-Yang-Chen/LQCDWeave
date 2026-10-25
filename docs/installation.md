# 安装与依赖层级

安装命令均从仓库根目录执行。`pyproject.toml` 定义按功能选择的依赖，
`constraints/` 固定特定环境的解析版本；约束文件本身不会安装这些包。
默认安装不拉取 GPU、QUDA、MPI、符号前端或性能工具。

安装包名和 Python 导入名均为 `lqcdweave`，性能命令入口为 `lqcdweave-profile`。

## 选择需要的层级

| 层级 | 命令/入口 | 适用范围 |
|---|---|---|
| 最小 CPU | `python -m pip install -e .` | NumPy、opt_einsum；已有数组、基础数值及纯结果 API；不是全部测试依赖 |
| Laplace CPU | `python -m pip install -e '.[laplace]'` | 增加 SciPy |
| HDF5 保存与读取 | `python -m pip install -e '.[hdf5]'` | 增加 h5py；保存原生多字段与分块数组 |
| 符号 Correlator | `python -m pip install -e '.[correlator]'` | SymPy、SciPy、lazy-object-proxy、h5py；群表示模块随包内置；Windows 增加 psutil |
| 完整单进程 CPU | `python -m pip install -e '.[cpu]'` | 合并 laplace/correlator；不包含 MPI 或 GPU |
| 开发测试 | `python -m pip install -e '.[cpu,dev]'` | CPU 功能与 pytest；标准回归仍是 unittest discovery |
| CPU 性能工具 | `python -m pip install -e '.[perf]'` | ASV、pyperf；不会运行基准或建立硬件基线 |
| HPC 性能工具 | `python -m pip install -e '.[perf-hpc]'` | ReFrame；调度器、站点配置、GPU 软件栈另行提供 |
| MPI | `python -m pip install -e '.[mpi]'` | mpi4py；实际 MPI 动态库和与 QUDA 的 ABI 须匹配 |
| GPU / QUDA | 见下文站点层 | 显式选设备、CuPy 构建、QUDA/PyQUDA 和 MPI；没有可保证兼容所有站点的通用 extra |

包的最低 Python 声明保持 3.10。符号前端和完整 CPU 功能的受支持范围仍为
Python ≥3.11；不要把最小层在 Python 3.10 可安装等同于完整 CPU 在该版本受支持。
项目依赖范围是解析边界，不表示其中每个版本组合都经过科学数值验收。

## 固定版本的 CPU 重现步骤

`constraints/cpu-py312-linux.txt` 来自 2026-10-07 实际环境：CPython 3.12.14、
Linux x86_64，NumPy 1.26.4、opt_einsum 3.4.0、SciPy 1.16.3、SymPy 1.14.0、
lazy-object-proxy 1.12.0、pytest 9.1.1；格点对称模块已随 本软件 内置，
不再单独解析发行包。文件同时固定其 Linux 传递依赖。
`constraints/build-py312-linux.txt` 单独固定安装/构建工具。
该快照不覆盖 Windows 的 psutil、Python 3.10/3.11、MPI、GPU/QUDA 或性能工具。
新格式使用的 h5py 3.16.0 于 2026-10-10 加入约束，已在本地 Windows 的 HDF5 读写回归中使用；
Linux 数值验收由 CI 单独运行，不计入上述 2026-10-07 的历史结果。`cpu` 和 `correlator` 安装包含 HDF5 支持；最小安装需要 HDF5 时加选 `hdf5`。

```sh
python3.12 -m venv .venv
. .venv/bin/activate
python -m pip install -r constraints/build-py312-linux.txt
python -m pip install --no-build-isolation \
  -c constraints/cpu-py312-linux.txt -e '.[cpu,dev]'
python -m pip check
python -m unittest discover -s tests -v
```

严格复现解释器时先确认 `python -VV` 为上述版本。其他 Python/OS 可使用上表的
功能 extra 自行解析，但须另做验证，不把这个快照当作跨平台 lock。
`--no-build-isolation` 在先装固定构建工具之后使用，避免隔离构建环境另选 setuptools。
只需重现最小 CPU 时，上述安装行换为 `-e .`，仍可使用同一个 CPU 约束文件；
它不会把 SciPy、符号依赖或测试工具装进去。

此处保证依赖版本选择可重现，不保证远端包源永远可用或不同平台轮子逐字节相同。
它不是完整离线 wheelhouse/哈希锁。需要离线部署时，在相同目标平台提前下载这些
固定版本并保存原始 wheel、来源与哈希；不能临时把解析不到的版本替换成最新版。

### 内置格点对称模块

`lqcdweave.symmetry` 是随本软件分发的正式模块，无需安装独立对称性包。
接口与依赖边界见[格点对称性](modules/symmetry/README.md)。
安装符号功能时只需执行上表的 `.[correlator]` 命令。

## GPU、QUDA 与 MPI：独立站点层

CPU 快照不提供统一保证的 CUDA/ROCm/QUDA 版本组合。使用已有站点环境或按
上游文档建立新环境，然后在同一解释器中安装 本软件 的所需功能 extra。
不要将本地 CPU 安装、mock adapter 测试或历史设备报告当作当前 GPU 验收。

1. CuPy 构建必须匹配所选 CUDA/ROCm 栈。CUDA wheel 名如 `cupy-cuda12x` 是
   构建选择，不是硬件自动探测；同一环境不能混装多个 CuPy 发行包。
   以 [CuPy 安装说明](https://docs.cupy.dev/en/stable/install.html) 为准
2. PyQUDA 及 `pyquda_utils` 的安装、QUDA 构建和运行库按
   [PyQUDA 官方安装](https://github.com/CLQCD/PyQUDA/wiki/Installation) 执行。
   本库不代为初始化 PyQUDA，不自动替换求解器或把单进程接口升级为域分解
3. mpi4py 必须链接/加载与实际运行环境兼容的 MPI；参见
   [mpi4py 安装说明](https://mpi4py.readthedocs.io/en/stable/install.html)。
   单独 `pip check` 或 `import mpi4py` 不证明原生 MPI/QUDA ABI 可运行

在已分配的目标设备内记录 Python、NumPy、CuPy/PyQUDA/QUDA、MPI 版本、构建来源、
驱动和运行库，再运行适用的[目标平台测试](../tests/README.md)及
[基准](../benchmarks/README.md)。`cupy.show_config()` 可补充软件栈诊断。
未分配设备时不启动设备计算；缺少目标平台证据保持“未验收”。

## 开发和性能环境

该记录只验证导入，不代表 CUDA 初始化或 GPU 数值测试通过。

标准 CPU 功能回归使用 `.[cpu,dev]`，不需要安装性能工具。完整 unittest discovery
中的可选 GPU/MPI 跳过须单列，不能记为平台通过；验收复用与冻结流程见
[验收说明](../tools/acceptance.md)。

`constraints/perf-py312-linux.txt` 是同日 CPython 3.12 Linux 的独立工具解析快照，
固定 ASV 0.6.6、pyperf 2.10.0、ReFrame 4.10.4 及传递依赖。不要把它当作 GPU
软件栈锁或吞吐基线。推荐为性能驱动器使用独立环境：

```sh
python3.12 -m venv .venv-perf
. .venv-perf/bin/activate
python -m pip install -r constraints/build-py312-linux.txt
python -m pip install --no-build-isolation -c constraints/cpu-py312-linux.txt \
  -c constraints/perf-py312-linux.txt -e '.[perf,perf-hpc]'
python -m pip check
asv --version
python -m pyperf --help
reframe --version
```

工具可启动不等于基准验收。现有 ASV 配置仍显式使用 Python 3.11 与 NumPy 1.26.4
作为其独立 benchmark 环境；启动 ASV 的 Python 3.12 工具环境不会改写它。
ReFrame 的 site config、队列与同设备参考必须由实际站点提供。
具体命令和计时边界仅在[基准说明](../benchmarks/README.md)维护。

## 依赖更新与验证

更新依赖先修改功能声明和对应约束，再做干净安装、`pip check`、针对性测试及完整 CPU 回归；改变 GPU/QUDA/MPI 栈时补充目标平台验证。CI 中不同 Python 版本及约束解析分别验证，不沿用旧平台结论。

`tests/test_dependency_tiers.py` 检查最小依赖边界、CPU extra 的一致性、约束满足声明、
内置模块进入 wheel/sdist 的内容与 MIT 许可证，以及 CI 的层级选择。它不把未测试版本自动列入兼容矩阵。
