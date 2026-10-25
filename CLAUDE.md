# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

协作约定（正确性核验、校验放入 `debug=True`、skill 同步、文档链接用相对路径等）以 [AGENTS.md](AGENTS.md) 为准：

@AGENTS.md

## 常用命令

安装（完整单进程 CPU 层 + 测试依赖；其它 extras 见 `pyproject.toml` 与 [docs/installation.md](docs/installation.md)）：

```bash
python -m pip install --find-links .github/test-wheels -e ".[cpu,dev]"
```

测试以标准 `unittest` 为主（CI 即运行此命令，Python 3.11/3.12）：

```bash
python -m unittest discover -s tests -v
```

- 单个文件：`python -m unittest discover -s tests -p test_wilson_pipeline.py -v`
- 单个子目录：`python -m unittest discover -s tests/correlator/planning -t tests -v`
- 部分目录含普通 pytest 函数，不在 unittest 计数中，需用 pytest：`python -m pytest tests/correlator/execution -q`
- 分层运行：`python tools/test_tiers.py --tier cpu-unit|cpu-integration|scientific-reference|gpu|slow`（`--list` 只列清单）。`slow`（完整 4³×8 weak_field）仅显式运行；GPU 需 `ED_RUN_GPU_BENCHMARK=1` 与真实设备；QUDA/MPI/性能走 `benchmarks/` 下的外部入口。`OK (skipped=N)` 不代表被跳过的平台已验收。

文档与公开 API 门禁（修改导出或文档后运行）：

```bash
python tools/check_public_api.py
```

```bash
python tools/check_docs.py
```

对应测试为 `test_public_api.py`、`test_documentation.py`；`test_architecture*.py` 与 `test_*_module_boundaries.py` 检查模块依赖方向并阻止已删除入口重新出现。

## 架构概要

源码位于 `src/lqcdweave/`。顶层模块职责与依赖方向见 [ARCHITECTURE.md §6](ARCHITECTURE.md)；`contracts` 在最底层不依赖其它模块，`production` 不依赖 `sampling`，adapters/storage 由高层入口或用户脚本注入，核心算法不自行发现文件。

**基础计算与稀疏化（六块，[L0/L1](docs/architecture/production-l0-l1.md)）**：TS `sampling/temporal/`、SS `sampling/spatial/`、EV `production/eigenvector/`、NV `production/noisevector/`、EL `production/elemental/`、PR `production/perambulator/`。库**不提供**基础产物的"计算并保存"runner：用户脚本显式调用生成器（如 `LaplaceEigenvectorGenerator(backend=...)`、`PerambulatorGenerator`、`sample_spatial_points`、`select_t`），再单独调用 storage 保存。不要重新包装此类 runner。

**Correlator（[L0/L1](docs/architecture/correlator-l0-l1.md)）** 以单字母模块命名，文档与代码注释大量使用这些字母：

| 字母 | 目录 | 职责 |
|---|---|---|
| S | `operators/correlator/` | 抽象粒子、群投影、Wick 图 |
| O | `operators/single_hadron/` | 两腿/三腿插入与算符组合（预设 JSON 在 `presets/`） |
| B | `contraction/binding/` | 算符与已有数据绑定，注入 reader |
| E | `contraction/expansion/` | 全场景网络、采样系数、坐标批次 |
| P | `contraction/planning/` | 联合路径搜索、模板复用、存储/预读规划 |
| X | `contraction/execution/` | 按计划执行（CPU/CuPy）、数组生命周期 |
| R | `contraction/results/` | 输出装配与系数回填 |
| M | `contraction/performance/` | 成本模型与观测 |
| A | `storage/contraction_plans/` | 完整计划编码/保存/加载 |
| C | `contracts/correlator/` | 跨模块纯记录与协议 |

流程：`Hadron/Flavor → gen_correlator → contraction.correlator.calculate`（收缩编排）→ B→E→P→X→R，返回统一的 `Calculation`。精度在 E 之后、P 之前按输入最低浮点精度统一。受管 `Elemental` 声明由 `calculate` 自动复用/补产，这是唯一保留的自动生产。E/P/X/R 低层入口仅供后端开发，不要作为普通任务的并列入口。

**数据层**：`storage.dataset` 以共享根 `dataset.json` 登记、每 cfg 一个完整 NPY/HDF5（[docs/modules/storage/data_storage.md](docs/modules/storage/data_storage.md)）；科学兼容判断在 `planning.dataset`，依赖图执行在 `runtime.dataset_stage`，跨 cfg 任务领取用 `runtime.dispatch.Dispatch`。身份由实际 cfg、基底、点/时间顺序、规范场版本等决定，不由 shape、seed、文件名推断；未知来源保持未知。空间点坐标一律 `(time, point, xyz)` 且为 `int64`。

**文档地图**：使用入口按任务路由见 [.agents/skills/README.md](.agents/skills/README.md) 与 `.agents/skills/`（8 个 skill）；模块 L2 详细设计在 `docs/modules/`；测试归属与各专项验收说明在 [tests/README.md](tests/README.md)。改动实现时须同步这些文档与 skill。
