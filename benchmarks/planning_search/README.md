# 固定工作量 CPU 搜索基准

`run.py` 由本轮实际使用的 `extended_worker.py` 整理而来，`path_fingerprint.py` 保留原 DAG/顺序摘要算法。显式指定源码 checkout、冻结计划档案、SHA256 和仓库外报告路径；不依赖历史快照目录或特定主机名。每次运行用新进程，报告不可覆盖。

```sh
PYTHONHASHSEED=0 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
python benchmarks/planning_search/run.py \
  --source /path/to/checkout \
  --archive /path/to/frozen.storage.json.gz --sha256 SHA256 \
  --output /path/to/evidence/run-0.json
```

Linux 自动固定到当前允许集合中的最小 CPU，记录 `/proc` RSS/HWM；其它平台报告内存不可用，需另外固定 CPU 并测量内存。记录源码提交、工作区状态、Python、平台、线程环境。计时前加载档案并核对哈希。代码路径从 `--source/src` 导入。

默认 `root_width=4, branch_width=4, include_ties=True, scorer=time_per_term, component_workers=1, kernel_batching=False`，`max_states=1536`，deadline 600 秒。首次可行路径后的 DFS 和回溯必须保留。时间预算若先触发，不能当作相同工作量成绩：检查 `statistics.expanded_states`、`complete_plans`。`--first-path` 仅作单独的首路径诊断。`--profile` 插桩的 inclusive/exclusive 时间与未插桩正式时间分开，不比较跨模式成绩。

通过截获 `finish_fixed_storage`，在完整最佳路径展开、执行表生成和原有审计之后结束；不做存储排期、传播子 IO 或数值缩并。报告分别记录分类、搜索 advance、最佳操作展开、Action materialization、到审计完成的总时间；档案加载和 DAG 摘要在该总时间之外。`compute_seconds` 是代价模型预测值，绝不能当作实测缩并耗时。Linux HWM 是进程累计峰值，包含输入加载；profile 的 retained-size 测量有额外成本。

比较冻结版本和候选版本时，使用同一脚本、输入、主机、核心、环境和预算，串行交错至少三次，比较中位数。候选路线不变时要求 DAG 和顺序摘要、预测成本、操作数和输出结构一致；改变合法路线时摘要可以变化，但必须补独立数值验证和搜索语义证据，不能只靠摘要宣布正确。运行 planning 全套与 CPU execution 全套：

```sh
PYTHONPATH=src:tests python -m pytest tests/correlator/planning -q
PYTHONPATH=src:tests python -m pytest tests/correlator/execution -q -k 'not cupy'
```

Windows 的 PYTHONPATH 分隔符为分号。小规模独立复数 NumPy 对照不能替代真实生产数据验收。大任务元数据基准不构成真实 CPU/GPU 缩并正确性证明。

## 搜索内存归因（诊断）

`memory_attribution.py` 在同一进程内运行归档的整数图 DFS，并按固定类别顺序贪心划分保留对象图，
对比两个状态预算得到每类别的字节/状态斜率；同时统计 registry 中已应用与从未应用的候选配方。
它不运行上面的正式基准、不做存储排期、不测量缩并时间，也不覆盖或替代正式计时报告。

```sh
PYTHONPATH=src:tests python benchmarks/planning_search/memory_attribution.py \
  --archive <frozen.storage.json.gz> --source . --states 1536 3072 --json /path/outside/repo.json
```

类别顺序即归因顺序：增长中的每状态表在前，共享的固定部分在后；两个预算用同一顺序，差值才代表边际增长。
每次运行同时报告采样调度（walks/visited_objects/seconds），用 `--sampling adaptive` 与 `--sampling fixed`
（原固定 256 状态节奏）或 `--source` 指向旧 checkout 即可对比采样总耗时随状态数的斜率。
每个预算只报告自身进程的 RSS/峰值工作集，跨预算的进程内存对比要在各自独立进程中运行（`--states` 单值）。
测量方法本身不进入软件仓库，真实归档及其结果留在仓库外。

竞赛任务及外部输入位置由发起者提供；真实课题档案、运行结果、性能报告和机器私有配置不入软件仓库。竞赛提示词模板见 [PROMPT.md](PROMPT.md)。
