# 性能回归基准

性能基准与默认 `unittest` 分开运行。常规正确性回归继续使用：

```sh
python -m unittest discover -s tests -v
```

## CPU：ASV

`asv.conf.json` 固定 Python 3.11 和 NumPy 1.26.4；`benchmarks/asv/`
四项基准统一使用固定 `weak_field`：时间选择与空间采样从规范场读取
4³×8 几何；两点重建使用归档 elemental 的四个分量构造全时间输入；
elemental 生产读取规范场和 Ne=20 本征矢并计算全部 8 个时间片。
两点重建的输入只用于测量该数据处理步骤，不解释为物理关联函数。
ASV 会为各提交建立环境，结果保存在被忽略的 `.asv/`
目录。

`benchmark_baryon_atomic.py` 另测合成 dense-V 重子端点的重复原子操作，
覆盖 Ne=64 和 complex64/complex128。计时包括已选固定路径的缩并与
自有结果快照；输入生成、路径选择和独立输出坐标的 CPU 数值对照在 setup 中完成。
此项用于比较结构缓存的 CPU 开销，不代表真实系综或完整生产流程性能。

`baryon_vertex_scope.py` 对重子降阶适用范围的测试样例（模式数 2–3，含介子、胶球、交换多重态、模态分块、一行多个 Φ/核心分组）在同一进程规划全部五种降阶，再交替顺序运行预热 + `--repeat` 轮，报规划、首次与暖机中位/四分位墙钟，以及对参考和对 joined 的相对 L2；`--backend cupy` 在 GPU 上运行。规模很小，墙钟主要反映规划与逐操作开销。`--ne N` 保留请求与绑定、换成 N 个模态的 complex64 随机数据；`--phase plan` 在 CPU 节点按声明的目标硬件（`--device-bytes`、`--profile`）规划并保存完整档案，`--phase run` 在设备上只载入档案运行。Ne=100 结果与生产规模的比较见[重子行打包因子网络](../docs/modules/expansion/baryon-factor-network.md)。

`baryon_vertex_structure.py` 按行族自旋结构（秩、稀疏度、基底、混合族、单行、每行两个 Φ）在同一进程规划五种降阶，同卡交替运行预热 + `--rounds` 轮并报中位数与四分位距；所有回退条件拆除，每种形式都实际运行。默认 `auto` 的选择（规划候选形式、取预测耗时最小者）用它与 `baryon_vertex_scope.py` 核对，结果见同一页“按结构的最优降阶”。

```sh
asv check
asv machine
asv run HEAD^!
asv continuous BASE_COMMIT HEAD
asv run NEW
```

第一次先运行 `asv machine` 登记这台 CPU，再用 `asv run HEAD^!` 建立
当前提交的基线；在分支上用
`asv continuous BASE_COMMIT HEAD` 比较基线与候选版本；夜间用
`asv run NEW` 更新历史。性能图和提交间的统计比较由 ASV 管理。


pyperf 保留为 CPU 微基准诊断工具，不加入默认依赖，也不重复管理 ASV
的提交历史。需要分析单个 Python 热点时，可用 `pyperf timeit` 或独立
`pyperf.Runner` 脚本采集分布，再用 `pyperf compare_to` 判断差异是否显著。

<a id="dataset-registration"></a>

## Dataset 增量登记

`dataset_registration.py` 读取一份已冻结的真实 `dataset.json` 快照，在指定 scratch
目录的独立临时目录中比较原实现与增量登记。它不读取或改写真实数值文件。
计时包含记录规范化、锁获取、压缩表读取、差异计算、原子替换与 fsync；
读取输入样本和逐项正确性核验在计时外。默认要求每次无竞争的完整登记小于 1 秒。
并发压力测量另外报告等待后的总耗时和追加持锁阶段耗时，不能混作无竞争计时。

```sh
python benchmarks/dataset_registration.py \
  --snapshot /path/to/frozen-dataset.json --scratch /path/to/audit/scratch \
  --reference-registration /path/to/original/registration.py \
  --repeats 3 --concurrent 40 --report /path/to/audit/registration.json
```

参考源码参数可省略。报告包含主机、Python 版本、源码与快照哈希、cfg 数量、单记录大小、
全部旧与新记录的精确核验和各次登记耗时。快照与报告必须放在仓库外；scratch 应放在
实际生产使用的文件系统上，硬件与文件系统范围不能从本机结果推广。

## CPU elemental 投影对照

当前与上一版共享投影核的对照可使用 `--reference-projection /external/projection.py`，
与旧的 `--reference-generator` 二选一。`--weight-mib` 仅配合 reference-projection，
在两版上设置相同内部加权场目标，
用于小格点复现工作区受限时的循环行为；不等价于真实大格点或整体内存限制。
预热后按 AB/BA 交替采样，记录两版投影源码哈希、全部逐次秒数及同输入数值误差；
归档源码和 JSON 留在仓库外。默认目标和受限目标分别报告，不能只选择提速样本。

2026-10-15 本地 NumPy 2.3.5、complex128、单 BLAS 线程、33 动量/四个 G 行的交替采样：
L8/Ne24、1 MiB 加权场目标，旧/新中位数 13.668/3.060 s（3 次，4.47 倍，最大差 1.02e-16）；
L6/Ne24、默认 256 MiB，0.675/0.651 s（5 次，逐元素相同），主机波动较大，不宣称默认规模
稳定加速。受限目标是在小数组上复现批宽压力，不代表真实大格点、GPU 或端到端生产验收。

`elemental_projection.py` 用相同输入、精度、进程和预热，对照指定历史 `generator.py`
与当前 CPU kernel 的 meson55 v2 四个 G 行。原文件放在工作区外或独立源码快照中。
计时仅包含投影；不包括输入、涂抹、保存与独立验证，也不据此宣称完整生产加速。

```sh
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONPATH=src python benchmarks/elemental_projection.py --reference-generator /path/to/original/generator.py --size 6 --ne 24 --repeats 3 --output /path/to/results.json
```

## 固定工作量搜索优化

[planning_search](planning_search/README.md) 提供冻结档案的 CPU 持续搜索基准、可选热点插桩与优化竞赛提示词模板；不运行存储排期或数值 IO，结果写入仓库外。
