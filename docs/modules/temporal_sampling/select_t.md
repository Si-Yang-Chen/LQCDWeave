# 时间选择程序 select_t

本模块仅在用户明确要求时间稀疏化时使用。默认计算采用全部时间片（Nt=Lt），不调用方案搜索。推荐从 `lqcdweave.sampling.temporal.api` 导入；旧导入路径保留为兼容入口。以下示例演示显式启用后的用法。

在生产或收缩脚本中使用选择记录前，先阅读[共同时间交接约定与当前接线限制](temporal_production.md)。稀疏生产、压缩 reader 与结果重建通过显式选择记录接线；`selection.times` 是物理坐标，不能直接索引 Nt 槽数据。

`Lt` 是时间总长度，`Nt` 是整个时间方向选中的时间片总数，`loop` 是重复周期，不是重复次数。当前支持周期边界，要求 `loop` 整除 `Lt`。

一个周期选 `m = Nt * loop / Lt` 个 offsets，在整个格点上重复 `Lt / loop` 次。`m` 非整数时该组合没有方案。`TimeSelection` 保存 Lt、loop、升序 offsets，并提供周期模板的升序 `times`。生产调用在 `select_t` 中提供 `cfg` 时，会把选中的模板按该 cfg 可复现地周期平移，并返回保留实际槽顺序的 `TimeSelectionRecord`。

生产开始后立即把实际 `TimeSelectionRecord` 保存为该 cfg 专用的 NPY 文件。生产、恢复、收缩和重建都从同一文件读取这份实际时间顺序；已存在选择文件时直接读取，不重新选择或覆盖。新收缩结果按 Nt 个源槽和绝对汇槽组织数据，随后直接调用 `reconstruct_two_point_from_coordinates(data, selection)`；模块内部还原物理时间差，不需要前端转换。完整调用与数组约定见[两点重建](../results/two_point.md)。

## 三步接口

```python
from lqcdweave.sampling.temporal.api import (
    find_min_loop, iter_legal_schemes, select_best, select_t,
    exponential_pair_q,
)

loop = find_min_loop(Lt=128, Nt=40)  # 16
schemes = iter_legal_schemes(Lt=128, Nt=40, loop=loop)
score = exponential_pair_q(ell=2 ** (29 / 8), differences=range(64))
best = select_best(schemes, key=score)

# 同样可一次完成；传入 loop 则直接在指定周期搜索。这里不传 cfg，返回周期模板。
best = select_t(Lt=128, Nt=40, key=score)
```

## 生产记录与复用

`cfg` 接受非空字符串或非 bool 整数。模块先按 criteria 和 key 选出未平移的周期模板，再将 `str(cfg)` 按 UTF-8 编码并用 SHA-256 摘要生成大端 seed，由局部 `numpy.Generator(PCG64(seed))` 从 `[0, Lt)` 抽取一个平移量；整型 `2000` 与字符串 `"2000"` 相同。平移按 `(t + shift) % Lt` 应用于模板的 `times`，原槽顺序保持不变，不重新排序，也不重新评分。默认全差覆盖在平移下保持成立；自定义 criteria 不会对平移后的时间记录重新检查。不同 cfg 可能抽到相同平移；这里不保证平移唯一。`cfg=None` 保持原行为并返回 `TimeSelection`。

```python
from lqcdweave.sampling.temporal.api import exponential_pair_q, select_t
from lqcdweave.storage.time_selection import save_time_selection

score = exponential_pair_q(ell=ell, differences=desired_differences)
selection = select_t(Lt=Lt, Nt=Nt, key=score, cfg=cfg)
if selection is None:
    raise RuntimeError("没有满足要求的时间方案")
save_time_selection(selection_file, selection)
```

`save_time_selection(path, selection)` 独占创建新文件，自动建立父目录，已有文件会报错而不会覆盖。文件只保存一维 `int64` 的实际有序时间坐标，不包含物理 `Lt`。例如 CSNS 旧格式参考 `01.select_t5x8/2000.npy` 是 40 个坐标的一维 `int64` 数组，开头为 `104, 105, 108, 113, 115, 120, 121, 124, 1, 3, ...`；`2010.npy` 示例从 `109` 开始。这些仅说明文件形状和顺序格式，各 cfg 的平移由各自 `cfg` 决定。

读取时必须显式传入物理时间长度，API 返回保留文件顺序的 `TimeSelectionRecord`：

```python
from lqcdweave.storage.time_selection import load_time_selection

selection = load_time_selection(selection_file, Lt=Lt)
```

继续生产、恢复、收缩和重建时均复用这个读取结果及对应的同 cfg 文件。保存的 NPY 是时间坐标表，不是 DatasetSpec 发布记录、产物身份或覆盖凭据。

`find_min_loop` 从小到大检查所有整除 Lt 的周期；找到实际合法方案才返回该周期。无方案返回 `None`。必要的计数下界不能单独作为存在性证明。

`iter_legal_schemes` 流式枚举全部合法 offsets 子集，按字典序输出，不将平移、反射等方案合并。方案内的时间点是集合，不枚举点的排列。指定 loop 表示允许的重复周期，不要求它是每个方案的最小基本周期。

## 合法标准：一个函数，一个 bool

```python
from lqcdweave.sampling.temporal.api import covers_all_differences, covers_differences

def includes_zero(selection) -> bool:
    return 0 in selection.offsets

rules = (covers_all_differences, includes_zero)
schemes = iter_legal_schemes(128, 40, 16, criteria=rules)

# 也可替换全覆盖，只要求给定的时间差。
rules = (covers_differences(range(20, 53)),)
```

默认标准 `covers_all_differences` 要求每个周期时间差都有源汇配对。多个标准按 AND 组合；空标准元组接受所有结构有效的方案。标准必须返回 Python `bool`，并应是确定性的纯函数。结构有效性（整数、范围、无重复等）由数据对象与搜索入口检查，不是可关闭的科学判据。

## 最优标准

`key(selection)` 返回可比较的分数，默认取最小值；`maximize=True` 取最大值。支持元组表达多级排序，完全同分保留输入中的第一个方案。评分函数应确定、无副作用，数值分数必须有限。`select_best` 只保证在传入候选中最优；传入完整枚举才能保证在全部合法方案中最优。

附带指数核 Q 实现，核尺度及时间差范围必须明确提供。它按每个时间差的配对源集合计算核协方差（含对角项），再相对完整时间平均归一化。周期约简避免构造完整 Lt×Lt 矩阵。自定义合法规则允许的方案若不满足 Q 所需配对覆盖，Q 会报错。

## 搜索规模与完成性

候选总数为 `C(loop, m)`。默认每个周期最多检查 1,000,000 个候选；可以显式设置 `max_candidates=None` 完整无上限搜索。超过限制抛出 `SearchLimitExceeded`，不返回伪装完整的方案集或最优解；已经消费的生成器结果只是部分结果。最小周期搜索遇到未完成的小周期也会抛出异常。

当前提供精确串行搜索；大周期组合枚举可能非常昂贵。

## 运行

覆盖判据使用实际模差，支持奇数和偶数周期；loop=1 的完整时间方案也合法。

```sh
python -m pip install -e .
python -m unittest discover -s tests -v
python -m unittest discover -s tests/sampling/temporal -p test_select_t.py
```

实际记录用上文的选择与保存 API 创建，按本次 cfg 保存后复用，不依赖项目脚本的默认 cfg 或输出参数。

## 模板复用与持久槽契约

`realize_time_selection(template, *, cfg)` 直接从已选 `TimeSelection` 生成 cfg 的
`TimeSelectionRecord`，算法与 `select_t(..., cfg=...)` 相同，不重复搜索。模板可放在 cfg
循环外；实际记录首次保存后，后续生产/恢复/消费复用读取值。

`TimeSelectionRecord` 提供 `slot(physical_time)`、`physical_time(slot)`、
`require_times(times)`、`document` 和 `from_document(document)`；缺失物理端点抛
`KeyError`，非法类型、重复/越界坐标或非法槽抛 `ValueError`。映射保持原始顺序。
`normalize_time_selection(selection, *, Lt=None)` 统一模板与实际记录；
`selection=None` 只有同时提供 Lt 时表示完整规范顺序。

正式产物的 `conventions.time_coordinates` 使用已有 DatasetSpec 字段，保存 slot 域、
absolute/cyclic_difference 布局和每条时间轴的 `{Lt,times}`。
`bind_time_selections(document,selections,*,layout="absolute")` 返回转换后的副本，
输入物理轴须精确等于给定选择顺序；已有不同映射不能重绑。
`dataset_time_selection(spec,axis="time")` 读取显式映射，旧物理产物返回 None；
`dataset_physical_times(spec,axis="time")` 解析有序物理坐标。
选择记录进入现有稳定定义身份，不能仅凭 Nt 复用其他平移或顺序的产物/计划。

selection 重建默认改为绝对源槽×汇槽；旧循环差槽调用必须明确 time_layout。
不同源/汇记录通过 sink_selection 明确，物理 Lt 必须相同。未知布局拒绝，缺失 dt
默认报错，只有显式 nan 生成 NaN；保留非时间轴，不恢复缺失逐源数据。
完整生产与保存用法见[共同时间交接](temporal_production.md)。
