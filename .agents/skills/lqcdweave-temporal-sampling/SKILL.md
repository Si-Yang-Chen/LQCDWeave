---
name: lqcdweave-temporal-sampling
description: "为明确要求时间稀疏化的 LQCD 测量选择时间方案，指导基底、点集、elemental 和传播子按 Nt 时间槽生产、保存及消费，并重建两点时间差关联函数。"
---

# LQCDweave · 时间取样与两点重建

共同任务路由、前端边界、保存/身份及执行范围统一见[技能共同约定](../README.md)。只读取当前任务的条件参考。

多 cfg 的任务分配、循环前共同准备及完成核验统一按 [Dispatch skill](../lqcdweave-dispatch/SKILL.md)；循环内生成或读取当前 cfg 的实际有序时间记录，本 skill 负责时间选择与重建约定。

只读取当前任务需要的参考；不要求通读其它产品或资源模块。

## 参数与执行范围

特别核对Lt/Nt/loop、实际时间及平移、所需dt覆盖、重建轴与配对计权；方案坐标不能代替实际数据顺序。

核对实际科学参数与[执行范围](../README.md#execution-scope)，沿用已有授权。

默认测量全部时间片；只有用户明确要求才选择子集。涉及基础数据生产、保存、读取、收缩或续算时，先读取[共同时间交接约定及当前支持范围](../../../docs/modules/temporal_sampling/temporal_production.md)，并按产品使用对应生产 skill。共同约定的正文维护在用户文档，下面仅给出选择记录与重建的入口。

生产创建新选择时，可在 cfg 循环外用 select_t 搜索模板，再用 `realize_time_selection(template, cfg=cfg)`；单 cfg 也可用 `select_t(..., cfg=cfg)` 取得该 cfg 的实际有序时间记录，并立即用 `save_time_selection` 独占保存。已有选择文件一律直接 `load_time_selection` 读取，不再次搜索、平移或覆盖。selection 重建默认源槽×绝对汇槽；物理坐标不能直接索引 Nt 槽数据。旧源槽×循环差槽必须显式 time_layout="cyclic_difference"。压缩 reader、生产器及 `calculate` 的接线能力必须按共同文档的当前支持范围核对。

用户给物理时间长度Lt、所选时间片数Nt、所需时间差范围及评分目标。loop表示重复周期，必须整除Lt；周期内选择m=Nt*loop/Lt个点。不要把重复周期当重复次数。

若该 cfg 的选择文件已存在，直接用下方消费块读取。仅在创建新选择文件时运行生产块；每个 cfg 保留自己的路径。

```python
from lqcdweave.sampling.temporal.api import select_t, exponential_pair_q
from lqcdweave.storage.time_selection import save_time_selection

score = exponential_pair_q(ell=ell, differences=desired_differences)
selection = select_t(Lt=Lt, Nt=Nt, key=score, cfg=cfg)
if selection is None:
    raise RuntimeError("没有满足要求的时间方案")
save_time_selection(selection_file, selection)
```

生产、恢复、收缩和重建都先从同 cfg 的选择文件读取实际时间顺序：

```python
from lqcdweave.storage.time_selection import load_time_selection
from lqcdweave.sampling.temporal.api import reconstruct_two_point_from_coordinates

selection = load_time_selection(selection_file, Lt=Lt)
# data 是源槽 × 绝对汇槽；i/j 分别对应 selection.times[i/j]。
result = reconstruct_two_point_from_coordinates(data, selection)
values, pair_counts = result.values, result.counts
```

评分尺度ell和差值范围由测量决定，不固化研究例值。`cfg` 决定可复现的周期随机平移，返回的 `TimeSelectionRecord` 保持实际槽序；算法细节见[TS L2](../../../docs/modules/temporal_sampling/temporal_sampling.md)。选择 NPY 只保存实际有序坐标；读取必须显式传入物理 `Lt`，API 和格式见[生产与保存约定](../../../docs/modules/temporal_sampling/temporal_production.md)。测量数值数组按 cfg 使用整块 NPY 或 HDF5，并由 `dataset.json` 登记，详见[数据保存说明](../../../docs/modules/storage/data_storage.md)。默认重建轴为tsrc_axis=0、dt_axis=1，额外轴布局按需指定。物理时间差映射、配对计数和平均由模块内部完成，普通生产脚本不生成转换表。Nt是选中数据的槽数，不改变规范场或Dirac算符的物理Lt。

周期模板 `TimeSelection.times` 是规范升序；cfg 平移后的 `TimeSelectionRecord.times` 保留跨边界的实际槽顺序。读取实际记录后直接传入重建。只有旧数据没有选择记录时，才用兼容入口 `reconstruct_two_point(data, Lt=Lt, selected_times=actual_times, tsrc_axis=..., dt_axis=...)` 提供已知的实际坐标。

自定义合法范围使用covers_differences；find_min_loop找最小合法周期，iter_legal_schemes枚举候选，select_best选择评分最小者。默认覆盖全部模时间差。穷举规模按组合数增长，超过max_candidates会报SearchLimitExceeded，不称部分搜索为严格最优。

重建对同一真实dt按实际配对数取平均，不恢复未计算逐源数据，也不保证单组态等于全源平均。默认源槽×绝对汇槽可直接交给 selection；物理坐标矩阵与旧循环差槽必须分别显式声明域和布局，不能仅凭 shape 混用。旧显式坐标调用仍兼容，但不能与selection混用；细节见[使用文档](../../../docs/modules/results/two_point.md)。不在skill中添加新的重建算法。

交付方案、实际时间、配对计数与估计量说明。缺失时间差明确报告，不能补零冒充已测量。

公开重建入口保留坐标、唯一性、轴和缺失差值检查；同次调用的内部计数与平均复用已接受的配对几何，无需用户先调用几何/计数服务来预检。独立再次调用时仍重新检查输入。
