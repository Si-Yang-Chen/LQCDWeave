# 测试

按改动范围运行模块测试及独立数值参考；涉及公共契约或跨模块行为时补充集成测试。
测试结果记录在本次验收日志，本文维护入口、覆盖与平台边界，不保存历次作业流水账。

## 完整回归与文档检查

在仓库根目录、已安装项目及所需依赖的环境运行：

```sh
python -m pip install -e '.[laplace,correlator]' pytest
python -m pytest tests -q
python -m unittest discover -s tests -v
python tools/check_docs.py
python tools/check_public_api.py
```

pytest 包含普通测试函数与 unittest 用例；unittest 不收集普通 pytest 函数，不能以其
结果替代全部 pytest 回归。可选环境要求见[安装说明](../docs/installation.md)。

## 测试层级与结果分类

`tools/test_tiers.py` 仅包装标准 unittest discovery/runner，按现有用例 ID
选择层级，不改变断言、skip 条件或执行后端，不提交集群任务、不创建新的测试框架。
可先 `--list` 检查本次实际清单；新增平台用例时维护工具的明确 GPU selector。
实际 CuPy 设备用例按明确的类或单个方法归入 GPU 层；同类的 CPU 测试及依赖主机 C++ 编译器的核体参考仍在 CPU 层，缺少编译器时归环境依赖跳过，POSIX 文件系统要求归文件系统环境跳过；均保留原始原因和未完成状态。
CPU 单元/集成/科学参考/GPU 四组组成常规 unittest 发现用例的无重叠分区；
科学参考组是固定跨模块独立 oracle 的精选入口，单元组同样含数值参考。
所有 discovery 导入错误在每个分层入口都保留为错误，不能因过滤而消失。
普通 pytest 函数仍须用对应目录的 pytest 入口运行，它们不在 unittest 计数中。
`slow` 是额外的 opt-in 清单，不属于 `--tier all` 或常规开发默认运行；
`tests/slow/weak_field_pipeline.py` 只在明确指定 `--tier slow` 时收集。
`temporary` 同为 opt-in：`tests/temporary/tmp_*.py` 是从常规模块暂时移出的
单个慢用例（每个在 CSNS CPU 上约半分钟以上），只在 `--tier temporary` 时收集；
改动对应模块时仍须运行，清单与耗时见 [tests/temporary/README.md](temporary/README.md)。
既有科学参考组成员及分类保持不变。

| 层级 | 执行入口 | 前提与验收边界 |
|---|---|---|
| CPU 单元 | `python tools/test_tiers.py --tier cpu-unit` | 模块目录中的 CPU 用例，含协议替身；不证明真实 GPU/MPI |
| CPU 集成 | `python tools/test_tiers.py --tier cpu-integration` | 根级科学交接、共享数据集登记与架构检查；本地进程和文件系统权限必须满足 |
| 科学参考 | `python tools/test_tiers.py --tier scientific-reference` | SciPy/符号 extra、固定 weak_field 文件；独立谱/逆矩阵/全空间迹对照 |
| GPU/CuPy | `ED_RUN_GPU_BENCHMARK=1 python tools/test_tiers.py --tier gpu` | 已分配的真实 CUDA 设备、兼容 CuPy、固定样本；含现有 GPU 小规模效率检查，不能替代 QUDA 或跨提交性能比较 |
| 慢速 CPU gate | `python tools/test_tiers.py --tier slow` | 额外的完整 4³ × 8 CPU Wilson gate；仅显式运行，不纳入普通开发回归 |
| 临时慢测 | `python tools/test_tiers.py --tier temporary` | 从常规模块暂时移出的单个慢 CPU 用例；仅显式运行，改动对应模块时须运行 |
| QUDA | `python benchmarks/csns_quda_stream.py --out /external/quda-stream.json` | 已授权的分配节点、实际 PyQUDA/CuPy 和匹配 MPI 库；同输入 eager/streamed VSV 对照，不是独立作用量 oracle |
| MPI | `python benchmarks/launch_mpi.py --old-root /path/to/legacy --data-root /path/to/weak_field --output /external/mpi-quda.json` | CSNS 指定匹配的 OpenMPI/PyQUDA，实际 legacy 和固定输入；现有入口只启动 1 rank，不能宣称多 rank 数值或共享文件系统验收 |
| 性能 | `asv check`、`asv machine`、`asv continuous BASE_COMMIT HEAD`；GPU 用 `reframe -C SITE_CONFIG -c benchmarks/reframe --system csns:dggpu --repeat 3 --run` | 同硬件、精度、输入哈希和计时边界；GPU 经授权调度，CPU 正确性通过不证明速度回归通过 |

QUDA/MPI/性能保留已有独立 runner，细节见 [性能基准](../benchmarks/README.md)。
`python tools/test_tiers.py --tier quda --list`（或 `mpi`、`performance`）只列前提和命令；
不带 `--list` 仍不启动它们，并以退出码 2 明确报告 NOT RUN。当前无通用多 rank
科学正确性 gate；该范围保持未验收，不能以 Dispatch 的 CPU mock、QUDA layout
转换或单 rank launcher 替代。不要在登录节点运行 GPU 基准。

```sh
python tools/test_tiers.py --tier scientific-reference --list
python tools/test_tiers.py --tier scientific-reference --json /external/scientific-reference.json
python tools/test_tiers.py --tier slow --list
python tools/test_tiers.py --tier slow --json /external/weak-field-audit.json
```

慢测清单见 [tests/slow/README.md](slow/README.md)。查看清单不会运行计算；
普通开发与默认 unittest/pytest 不会收集这项慢测。

分层工具严格退出码：0 = 实际选中的用例全部通过且无 skip/expected failure；
1 = 任一 failure/error/unexpected success；2 = 有 skip/expected failure、没有用例或
外部层级未执行。原始 unittest 的 `OK (skipped=N)` 语义保持不变，但不能当作
被跳过平台已经验收。JSON 的 `successful` 保留 unittest 原生意义，另看
`status`/`exit_code`；`passed` 直接数 `addSuccess`，不从 `testsRun-skipped` 推导，
因为 `setUpClass` 跳过不计入 testsRun，subtest failure 也有自己的计数语义。
JSON 的 `audits` 按 `test.id()` 保存用例提供的 `audit_metrics`。

诊断分类只补充原因，不把失败变成 skip：

- 缺少 SciPy/可选 extra、CUDA/CuPy、平台专属目录功能或固定参考数据：保留实际 skip
  和原文，分别标记 dependency/platform/filesystem/fixture；未知 skip 保持 unclassified
- `multiprocessing` 的 socket 创建因 EACCES/EPERM 被拒：只有真实 traceback 同时含
  multiprocessing 和 socket 帧才标记 environment-restriction，结果仍是 error、exit 1
- 其后的 worker 清理断言、数值误差、布局/身份不符、损坏、未知导入或一般权限错误：
  保留 failure/error（默认 test-failure），不能凭“可能与环境有关”豁免
- 平台缺失应报告“未运行”；没有真实硬件或依赖，不建立假的 QUDA/MPI 通过记录

工具不重新解释历史日志。报告原生总数、直接记录的成功、skip、failure/error 各类，
保留错误日志；环境受限的运行不是全绿。受限主目录可显式设置
`LQCDWEAVE_PROFILE_DIR` 到可写目录，不能静默关闭测试或降低科学精度。

## 固定小格点完整科学 gate

```sh
python -m unittest discover -s tests -p test_small_scientific_pipeline.py -v
```

`test_small_scientific_pipeline.py` 固定 seed=94 的合成 SU(3) 场，
`(Lx,Ly,Lz,Lt)=(2,1,1,3)`、Ne=2、complex128；Wilson 质量 0.4、各向同性、
无 clover、r=a=1、反周期夸克时间边界，空间周期、传播子 unscaled、DR 自旋基。
两项 gate 都通过公开 EV→EL→PR→Diagram→save/load 接口，PR 使用真实的
`DenseWilsonBackend(max_dimension=72)`。源端 dagger、connected Wick 负号、
非零动量 `(1,0,0)` 的 anchor 相位和复系数 `0.7+0.2j` 均显式核验。

- 默认未传 selection：完整 3 时间片、全部 9 个源汇对，物理 dt 配对数 `[3,3,3]`
- 显式时间稀疏分支：实际 selection `(2,0)` 保存/读回，只生产两个逻辑槽和四个源汇对；
  Dirac 格点仍为物理 Lt=3，保存映射，物理 dt 配对数 `[2,1,1]`
- 每个阶段先发布再重新打开，核对 actual ID/父基底、EV 谱/残差/正交性、EL 相位矩阵、
  PR 独立 Wilson 逆矩阵投影；关联函数对照独立全部物理片的完整空间迹
- oracle 复用既有 `dense_laplacian` 的逐格点索引参考和 `stencil` 的数组平移参考，
  EV oracle 另用 NumPy 稠密对角化，不调用生产 Wilson 矩阵或收缩核生成真值
- 原始 correlator 读回逐元素保持，源时平均另存并用显式坐标循环核对归一化、dt 与计数；
  稀疏平均仅代表覆盖的配对，不宣称等于完整总体平均或一般抽样无偏

这是固定 72 维 CPU 教学/回归问题，不是物理系综、生产规模、GPU/QUDA/MPI 或
性能验收。显式 `profile="default"` 仅固定测试成本模型并避免无关自动校准；
不跳过数值生产/收缩，也不将未测量模型当作性能证据。

## meson55 分波核验

```sh
python tools/check_meson55_partial_waves.py --output /path/to/angular-audit.json
```

单元测试核对审计方法：Legendre 加法定理与独立 SciPy 球谐振幅、逐半径检查、系数变化及目标分波非零。
它验证默认 v2 的普通行通过，也验证历史 CSNS 表中的坏行被正确拒绝；单元测试通过不表示磁型、电型已完成分波验收。
独立 SciPy 对照缺依赖时按原条件跳过，不能计为已验收。

审计工具默认读取 v2，返回 2：25 个普通行通过、无失败，30 个磁型/电型行明确记为未支持。
显式指定历史 `meson55.json` 时返回 1：普通行 16、22–24 的低分波投影非零，且 30 行未支持：
`python tools/check_meson55_partial_waves.py --catalog src/lqcdweave/operators/single_hadron/presets/meson55.json`。
若没有失败但有未支持项，返回 2；全部行实际检查通过才返回 0。几何条件和结果见
[meson55 低分波投影检查](../docs/modules/operators/meson55.md#低分波投影检查)。


## 测试目录与覆盖

| 目录 | 主要覆盖 |
|---|---|
| `correlator/contracts/` | 中立记录、身份、整数轴、精度及资源契约 |
| `correlator/operators/`、`correlator/symbolic/` | 算符、群投影、Wick 与独立代数参考 |
| `correlator/expansion/`、`adapters/` | 实际输入身份、源汇变换、展开与读取适配 |
| `correlator/planning/` | 搜索、固定计划、共享、预算、生命周期和独立资源审计 |
| `correlator/execution/` | 数值核、固定回放、当前输入、缓存所有权、设备及并发协议 |
| `correlator/results/`、`correlator/plan_archive/` | 结果装配、档案兼容、损坏拒绝、共享身份与固定回放 |
| `sampling/`、`production/` | 时空取样、低模、混合向量、elemental 与传播子生产 |
| `planning/`、`storage/`、`runtime/` | 数据依赖、持久化、恢复、任务分配与资源生命周期 |
| `workflows/`、根级 `test_*.py` | 公共入口、跨模块科学交接、完整任务及架构约束 |
| `fixtures/`、`data/` | 共享参考实现及固定样本；不得以待测实现生成真值 |

单模块示例：

```sh
python -m pytest tests/correlator/planning -q
python -m unittest discover -s tests/production/eigenvector -t tests -v
```

## 独立参考与重点回归

- 小格点完整参考见上面的科学 gate；Wilson/Laplace 使用独立全空间算子、稠密逆矩阵或对角化核对。
- 介子、重子与稀疏桥保留独立迹、逐指标求和、行列式和 COO 枚举参考；整体顶点、行打包与合批不能仅互相比对。
- 固定档案回放检查加载/执行不重新展开或搜索，改变 cfg/输入数值后重新计算，保持输出及借用所有权。
- 资源与缓存测试分别核对正常/debug 路线、同值不同身份、预算拒绝、最后使用和异步完成后释放。
- 双夸克冻结资源的结构、顺序和非零支持严格一致，SVD 系数按绝对误差 `1e-14` 对照独立代数参考。
- `test_documentation.py`、架构测试及 `test_public_api.py` 覆盖链接、示例签名、模块边界和公开导出。

## 平台与验收边界

CPU 元数据、协议替身和生成核体参考不证明真实 GPU 执行；GPU 结果必须记录实际设备、
输入和精度。`RawDriverArgumentsCuPyTests` 明确要求可用 CuPy/CUDA，缺环境时报错而非跳过。
GPU 输出后处理另按[验收范围](../docs/modules/results/gpu_output_postprocessing.md)执行，不扩大到未测结构。
生产性能比较固定输入、源码、设备、线程和计时边界，分别报告规划、加载、首次及暖运行。
模型峰值、进程 RSS 与设备实测峰值不得混用；单核或小样本结果不能外推完整 cfg 性能。

模块职责见[Correlator L2](../docs/modules/README.md)与[基础计算 L0/L1](../docs/architecture/production-l0-l1.md)。
冻结源码与安装包的复核流程见[本地验收工具](../tools/acceptance.md)。
