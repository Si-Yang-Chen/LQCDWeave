# TS：时间选择与两点时间重建（L2）

状态：精确串行方案搜索、几何判据/评分、cfg 平移记录、selection 直接重建及兼容显式坐标重建已实现。该模块是可选能力；常规工作流默认使用全时间片，只有调用方明确请求时间子集时才进行稀疏化。模块不启动生产、调度、存储、恢复或集群任务。

## 职责与边界

唯一实现目录为 `src/lqcdweave/sampling/temporal/`。公开入口是 `lqcdweave.sampling.temporal.api`，它组合搜索、判据、目标函数和重建，并重导出唯一共享对象 `TimeSelection` 与 `TimeSelectionRecord`。实际选择记录的 NPY I/O 归 `lqcdweave.storage.time_selection`；TS 不读写文件、不选择后端、不生产传播子，也不改算符时间语义。

默认完整覆盖仍由上层流程决定（`Nt=Lt`）。TS 不会因为模块存在而自动搜索或启用子集。`TimeSelection(Lt, 1, (0,))` 表示完整时间覆盖。无 `cfg` 的搜索结果是周期模板；生产调用提供 `cfg` 时返回确定性平移后的 `TimeSelectionRecord`，调用方通过 storage 保存和读取该记录，再把实际有序坐标传给消费者。

```python
from lqcdweave.sampling.temporal.api import (
    TimeSelection,
    covers_all_differences,
    exponential_pair_q,
    find_min_loop,
    iter_legal_schemes,
    reconstruct_two_point,
    reconstruct_two_point_from_coordinates,
    select_best,
    select_t,
)

score = exponential_pair_q(ell=2 ** (29 / 8), differences=range(64))
selection = select_t(Lt=128, Nt=40, key=score, cfg="cfg-001")
```

`exponential_pair_q` 是显式提供的几何代理；`ell` 与 `differences` 必须由调用方给定。本模块不使用研究资料中的固定尺度、panel 池、统计估计器或论文优化流程作默认值。cfg 平移使用下面定义的可复现规则，不采用固定的 cfg-to-shift 表。

## 共享时间契约与顺序

`TimeSelection` 在 `contracts/time_selection.py` 中唯一实现，本模块只导入并重导出它，不创建平行类型。构造签名为 `TimeSelection(Lt: int, loop: int, offsets: tuple[int, ...])`，三个字段都没有默认值。

| 字段/属性 | 含义 |
|---|---|
| `Lt` | 周期格点的时间长度，正整数。 |
| `loop` | 重复周期长度，正整数且必须整除 `Lt`；不是周期重复次数。 |
| `offsets` | 周期内选中偏移的非空、唯一坐标，范围 `[0, loop)`。初始化时规范化为升序 tuple。 |
| `Nt` | 只读派生属性：`len(offsets) * (Lt // loop)`。 |
| `times` | 只读派生属性：按 `base=0, loop, 2*loop, ...` 外层、升序 `offsets` 内层生成 `base + offset`；返回长度 `Nt` 的升序 tuple。 |

`TimeSelectionRecord(Lt: int, times: tuple[int, ...])` 是生产 cfg 的实际时间记录。它是冻结 dataclass，`Lt` 为正整数，`times` 为 `[0,Lt)` 内非空唯一整数 tuple；不按周期展开或排序。`Nt = len(times)`；`times` 的顺序就是生产数据槽顺序，可以跨周期边界回绕。

`TimeSelection` 对象为冻结 dataclass，字段不能重新赋值，`offsets` 是不可变 tuple。正整数检查拒绝 `bool`。`offsets` 必须可迭代；坐标类型、范围、空集或重复坐标不合法时抛 `ValueError`，不可迭代对象会由 `tuple(...)` 引发 `TypeError`。它的偏移会规范化为升序。`TimeSelectionRecord` 同样冻结，但保留传入时间顺序。消费已保存的 cfg 选择时使用记录对象；只有旧数据没有实际记录时才使用兼容坐标入口。

## 搜索、判据与评分 API

### 搜索与排名

| 名称 | 精确签名与默认值 | 行为 |
|---|---|---|
| `SearchLimitExceeded` | `SearchLimitExceeded(RuntimeError)` | 达到候选上限仍未穷尽时表示搜索不完整；异常不能解释为无解或最优。 |
| `DEFAULT_CRITERIA` | `(covers_all_differences,)` | 默认要求每个周期时间差都有源汇配对。 |
| `iter_legal_schemes` | `iter_legal_schemes(Lt: int, Nt: int, loop: int, *, criteria: Iterable[Criterion] = DEFAULT_CRITERIA, max_candidates: int \| None = 1_000_000) -> Iterator[TimeSelection]` | 惰性枚举指定 `loop` 下全部符合结构和判据的偏移组合。 |
| `find_min_loop` | `find_min_loop(Lt: int, Nt: int, *, criteria: Iterable[Criterion] = DEFAULT_CRITERIA, max_candidates: int \| None = 1_000_000) -> int \| None` | 按递增整数检查整除 `Lt` 的周期，返回第一个有合法方案的 `loop`；完整搜索无解才返回 `None`。 |
| `select_best` | `select_best(schemes: Iterable[TimeSelection], *, key: Callable[[TimeSelection], Score], maximize: bool = False) -> TimeSelection \| None` | 对传入候选流按 `key` 排序；默认取小值，`maximize=True` 取大值，同分保留最先方案。空流返回 `None`。 |
| `select_t` | `select_t(Lt: int, Nt: int, *, key: Callable[[TimeSelection], Score], loop: int \| None = None, criteria: Iterable[Criterion] = DEFAULT_CRITERIA, maximize: bool = False, max_candidates: int \| None = 1_000_000, cfg: str \| int \| None = None) -> TimeSelection \| TimeSelectionRecord \| None` | `loop=None` 时先求最小可行周期，再在该周期内找最优模板；显式给出 `loop` 时只搜索该周期。`cfg=None` 返回原周期模板；提供 `cfg` 时返回该 cfg 的实际有序平移记录。 |

`cfg` 必须是非空 `str` 或非 bool 整数；非法值抛 `ValueError`。先完成模板搜索并选出 `best`，criteria 与 key 都只作用于原周期 `TimeSelection`；之后才执行 cfg 平移，不对平移结果重新评分或排名。实现用 `str(cfg).encode("utf-8")` 的 SHA-256 摘要，再把摘要字节解释为大端整数 seed，创建局部 `numpy.random.Generator(numpy.random.PCG64(seed))`，并调用 `integers(Lt)` 取平移量 `shift`。结果为 `TimeSelectionRecord(best.Lt, tuple((t + shift) % best.Lt for t in best.times))`，按模板槽序保留坐标，不排序。默认 `covers_all_differences` 在周期平移下保持成立；自定义 criteria 可能依赖绝对偏移，平移后不会重新检查。整数 `2000` 与字符串 `"2000"` 归一成同一 cfg 值；不同 cfg 不保证得到不同平移量。cfg 平移不改变全局 NumPy 随机状态。

给定周期需满足 `loop | Lt`。每周期所需偏移数为 `m = Nt * loop / Lt`；若非整数，该周期没有方案，枚举为空，不视为异常。候选偏移组合按 `itertools.combinations(range(loop), m)` 的字典序产生；不合并平移、反射或其它等价方案，也不排列同一集合内坐标。`loop` 是允许的重复周期，不保证是某个 offsets 集合的最小周期。因为 `iter_legal_schemes` 是生成器，它的参数校验在首次迭代时发生，而不是在创建生成器对象时发生。

`iter_legal_schemes` 完整耗尽才证明该周期枚举完成。默认上限为每个被检查周期 1,000,000 个组合；`max_candidates=None` 取消上限，正整数限制候选数。检查第 `max_candidates+1` 个组合时抛出 `SearchLimitExceeded`。此前已从生成器取到的值仍只是部分结果。`find_min_loop` 的预算逐周期计算；较小周期搜索不完整时异常传播，不跳过该周期冒称最小值。

默认全差覆盖时，搜索先使用必要条件 `m * (m - 1) + 1 >= loop` 排除不可能周期；该下界只在判据元组中直接包含默认函数对象 `covers_all_differences` 时应用，不为自定义判据推导假设。之后逐组合构造 `TimeSelection` 并求判据 AND。搜索以生成器流式工作，不保留全部候选。对 `C(loop,m)` 个组合，结构检查约为 `O(C * m log m)`；默认差覆盖判据每个候选约 `O(m²)` 时间和 `O(min(loop,m²))` 集合空间。用户自定义判据的成本另计。

`select_best` 和 `select_t` 不会缓存完整候选；对 `C` 个输入方案，排名开销为 `O(C * (key_cost + comparison_cost))`，除当前最优外使用常数个方案的空间。精确搜索可能很昂贵；本模块不并行搜索。评分由调用方负责确定性、可比较性和有限值。实现不主动拒绝 NaN/无穷值；key 调用异常或不可比较值引发的比较异常直接传播，NaN 不应作为 key。

## 实际选择文件（storage）

文件 I/O 属于 `lqcdweave.storage.time_selection`，不属于 TS 自身：

| 函数 | 精确签名 | 行为 |
|---|---|---|
| `save_time_selection` | `save_time_selection(path, selection: TimeSelection \| TimeSelectionRecord) -> None` | 创建新的一维 `int64` NPY，只保存 `selection.times` 的顺序；自动建立父目录，以独占创建方式写入，目标已存在时失败且不覆盖。 |
| `load_time_selection` | `load_time_selection(path, *, Lt: int) -> TimeSelectionRecord` | 用显式物理 `Lt` 读取 NPY，`allow_pickle=False`；只接受一维整数数组，且坐标非空、唯一并位于 `[0,Lt)`，保留数组顺序。 |

NPY 不保存物理 `Lt`，加载时不会从文件名或最大时间坐标推断它。选择文件只记录实际坐标，不发布 DatasetSpec、不建立产物身份，也不充当覆盖或来源凭据；正式数据仍走现有 storage/runtime 机制。

### 判据

`Criterion = Callable[[TimeSelection], bool]` 是类型别名。

| 函数 | 签名 | 行为 |
|---|---|---|
| `covers_all_differences` | `covers_all_differences(selection: TimeSelection) -> bool` | 计算所有偏移有序对 `(a,b)` 的 `(b-a) % loop`；要求差集包含周期内所有 `0..loop-1`。 |
| `covers_differences` | `covers_differences(differences: Iterable[int]) -> Criterion` | 先固化显式差值序列，返回只要求其中每个差值模 `loop` 可配对的判据。空差值序列恒满足。 |
| `is_legal` | `is_legal(selection: TimeSelection, criteria: Iterable[Criterion]) -> bool` | 按输入顺序短路 AND；空判据接受任意结构有效的方案。 |

`covers_differences` 拒绝 bool 或非整数差值（`ValueError`），接受负整数并在调用时按周期取模。每个判据必须返回 Python `bool`；`is_legal` 对非 bool 返回值抛 `TypeError`。判据本身及其异常不被捕获；判据应为确定性的纯函数。

默认覆盖判据对一个候选约需 `O(m²)` 时间，并保存差值集合。自定义判据的算法和空间不在模块保证范围内。

### 目标函数

| 名称 | 精确签名与默认值 | 行为 |
|---|---|---|
| `exponential_pair_q` | `exponential_pair_q(*, ell: float, differences: Iterable[int]) -> Callable[[TimeSelection], float]` | 构造一个几何评分闭包；较小值较优。没有隐含的 `ell` 或差值默认值。 |

闭包对完整格点差值 `d=0..Lt-1` 使用周期最短距离核 `K(d)=exp(-min(d,Lt-d)/ell)`，对每个 `d mod loop` 将 `Lt // loop` 个重复值作平均。对每个需求差值，选出偏移 `a` 使 `(a+d) % loop` 也在偏移集合中，再对这些可用源之间的所有有序源对计算核均值；最后按调用方提供的 `differences`（包括重复差值）等权平均，并除以完整时间的核均值后开平方。对角源对包括在内。此数值是几何代理，不是从本次数据估计的方差或无偏性证明。

工厂要求 `ell` 有限且大于零；`differences` 必须是非空整数序列，bool 不算整数，检查失败为 `ValueError`。不能传给 `math.isfinite` 的 `ell` 类型会引发 `TypeError`。重复差值保留最终均值中的重复权重，但计算时相同模差只算一次。若某差值没有可配对源，调用评分闭包时抛 `ValueError`。核值按 `(Lt, loop)` 在该闭包的 LRU 缓存中复用，最多 128 个键；单次核预计算 `O(Lt)` 时间/空间，单次评分约 `O(D_unique*m² + D)` 时间、`O(Lt + D_unique + m)` 空间，其中 `D` 为差值条目数。

## 两点重建 API

### 返回对象

`TwoPointResult` 签名为 `TwoPointResult(values: np.ndarray, counts: np.ndarray, dt_axis: int)`，三个字段均为必填，无默认值。它是冻结 dataclass：不能重新绑定字段，但其中的 NumPy 数组保持可写。`values` 是新分配的输出；`counts` 是长度 `Lt` 的整数数组，记录每个真实 `dt` 的全体有效源汇配对数；`dt_axis` 是输出中展开到长度 `Lt` 的时间差轴位置。

### 面向生产脚本的 selection 入口

```python
reconstruct_two_point_from_coordinates(data, selection)
```

`selection` 接受唯一共享 `TimeSelection` 或 `TimeSelectionRecord`，可位置传入或使用 `selection=`。默认 `tsrc_axis=0, dt_axis=1, missing="raise", time_layout="absolute", sink_selection=None`。源轴长 `selection.Nt`，汇轴默认相同；显式 sink_selection 时长 sink_selection.Nt，默认源槽 `i`、绝对汇槽 `k` 对应实际 `(times[i], times[k])`；旧循环差槽必须显式 `time_layout="cyclic_difference"`，此时汇槽为 `(i+j) % Nt`，与按 Nt 个逻辑时间槽收缩相同。物理格点/Dirac 算符的 Lt 不变。内部使用 `selection.Lt` 和 `selection.times` 映射物理差值并调用既有平均核；对于 `TimeSelectionRecord` 会按保存顺序解释槽，不要求脚本转换坐标，数值估计量不变。

含 spectator 轴时指定 `tsrc_axis`、`dt_axis`（支持负索引），输入不修改，返回行为与下述兼容入口一致。非 `TimeSelection` / `TimeSelectionRecord` 的 selection 抛 TypeError；与任一显式 `Lt/tsrc_times/dt_values/sink_times` 混用抛 ValueError。省略 selection 时须完整提供旧显式坐标参数；两种输入都未完整提供时抛 TypeError。

### 兼容索引差入口

```text
reconstruct_two_point(
    data,
    *,
    Lt: int,
    selected_times,
    tsrc_axis: int,
    dt_axis: int,
    missing: str = "raise",
) -> TwoPointResult
```

此入口适用于数据第二时间轴保存稀疏集合内的循环索引差 `j`：源索引 `i` 的汇索引是 `(i+j) % Nt`，真实差值为 `(selected_times[(i+j) % Nt] - selected_times[i]) % Lt`。`selected_times` 必须是与存储完全同序的一维非空、唯一、整数坐标数组，范围 `[0,Lt)`；函数按给定顺序解释下标，不排序坐标。数据的源轴与索引差轴都必须长 `Nt=len(selected_times)`。

算法先构造 `(Nsrc,Ndt)` 的真实差值矩阵，再交给唯一内部显式坐标平均核 `_reconstruct_from_physical_coordinates`。显式 cyclic_difference 的 selection 分支委托此索引映射；默认 absolute 分支直接从两端实际时间构造差值，不复制聚合核。这样周期回绕和任意实际时间排列会按真实物理 `dt` 处理；不会把 `j` 当作物理差值。

### 完整签名与显式坐标兼容

```text
reconstruct_two_point_from_coordinates(
    data,
    selection: TimeSelection | TimeSelectionRecord | None = None,
    *,
    tsrc_axis: int = 0,
    dt_axis: int = 1,
    missing: str = "raise",
    Lt: int | None = None,
    tsrc_times=None,
    dt_values=None,
    sink_times=None,
    time_layout: str = "absolute",
    sink_selection: TimeSelection | TimeSelectionRecord | None = None,
) -> TwoPointResult
```

省略 selection 时保留旧显式物理坐标调用。`tsrc_times` 给出实际源坐标，长度必须等于数据源轴。`dt_values` 必须是物理时间差：形状 `(Ndt,)` 表示所有源共享坐标，或 `(Nsrc,Ndt)` 表示逐源差值；后者两个维度固定为源、差值，不跟随 `data` 轴位置。每行时间差必须非空且唯一。可选 `sink_times` 是唯一、非空的实际汇坐标列表；传入时仅保留 `(source + dt) % Lt` 在该列表内的配对。省略时所有显式源/差值对有效。普通源×绝对汇时间矩阵不能仅凭相同 shape 当作源槽×循环差槽输入。

两个时间轴均须存在且不能相同，支持负轴号。数据须为整数、无符号整数、浮点或复数 ndarray；源坐标、时间差与汇坐标必须是 `[0,Lt)` 内整数坐标，且源坐标唯一。仅源轴被平均消去；其余 spectator 轴保持相对次序，原 `dt_axis` 扩展至长度 `Lt`。输出 `dt_axis = normalized_dt_axis - int(normalized_source_axis < normalized_dt_axis)`。

每个真实 `dt` 将所有有效源汇配对值相加后除以该 `dt` 的配对数；不同 `dt` 使用各自计数。此处不额外平均 spectator 轴、不按源重数、采样概率或研究评分加权、不折叠时间、不施加共轭或边界相位修正。已有 NaN 会通过求和传播，配对数不据此调整。

`missing="raise"` 是默认值；若任一 `dt` 没有有效配对则抛 `ValueError`，错误消息列出缺失差值。`missing="nan"` 将缺少的 `dt` 保留为 NaN、计数为零，不插值、不填零。其它 `missing` 值抛 `ValueError`。结果 dtype 至少为 float64；复数输入的求和和结果至少为 complex128。

轴、长度、范围、唯一性、数值 dtype 或坐标形状不满足约定时抛 `ValueError`。轴不是合法整数索引（包括 bool）或两轴相同也抛 `ValueError`。`Lt` 必须为正整数。底层 NumPy 转换/分配异常继续传播。

重建不修改输入。`np.asarray` 对 ndarray 通常复用输入，`moveaxis` 得到视图；输出 `values` 和 `counts` 新分配并由结果持有。缺失掩码和真实差值表占 `O(Nsrc*Ndt)`，输出占 spectator 元素数乘 `Lt`，对每个出现的差值筛选配对并求和，时间约 `O(Lt*Nsrc*Ndt + output_size)`，另有每个差值的高级索引临时数组。输入本身不计入此额外内存。实现不提供分块、流式输出或异步消费协议。

若不同组态具有不同时间平移或实际顺序，调用方必须逐组态加载匹配的 `TimeSelectionRecord` 并分别调用后再堆叠；没有记录的旧数据可提供匹配的显式坐标。重建器不从一个时间方案推断其它组态坐标。该重建生成逐组态的配对平均估计，不恢复未计算的逐源数据，不保证其等于完整源时间平均。

## 状态、所有权与异常汇总

TS 没有任务生命周期或持久状态。`TimeSelection` 与 `TimeSelectionRecord` 都是不可变值对象，前者描述周期模板，后者保留实际物理时间及槽顺序；搜索生成器的完成性由其是否正常耗尽决定；搜索或判据异常原样传给调用方。`exponential_pair_q` 返回纯计算闭包，只有闭包内核缓存。`TwoPointResult` 只冻结字段绑定而不冻结 ndarray 内容。storage 单独持有文件 I/O 职责；TS API 不持有、修改或发布生产输入。

| 位置 | 可见异常 |
|---|---|
| `TimeSelection`/搜索参数 | `ValueError`：非正整数、`Nt > Lt`、不整除的 `loop` 或非法 offsets；搜索候选上限非法也为 `ValueError`。 |
| `iter_legal_schemes` / `find_min_loop` | `SearchLimitExceeded`：候选枚举不完整。结构参数非法为 `ValueError`。算术上无候选只返回空迭代器/`None`。 |
| 判据 | `ValueError`：`covers_differences` 中差值类型非法；`TypeError`：判据返回值不是 Python `bool`。用户判据自己的异常传播。 |
| `select_best` / `select_t` | key 异常与比较异常传播；没有方案返回 `None`。评分有限性由调用方负责。 |
| `exponential_pair_q` | `ValueError`：ell 或差值定义非法、或某个评分所需差值没有配对源。 |
| 重建函数 | `TypeError`：selection 不是 `TimeSelection`/`TimeSelectionRecord` 或缺少完整输入；`ValueError`：混用 selection/显式坐标、非法 Lt、轴、dtype、坐标、shape、重复、覆盖或 missing 值；NumPy 运行异常传播。 |

## 实现与兼容入口

搜索、判据、目标评分和重建分别位于 `sampling/temporal/search.py`、`criteria.py`、
`objective.py` 和 `reconstruction.py`。`contracts/time_selection.py` 唯一定义
`TimeSelection` 与 `TimeSelectionRecord`。

旧导入入口 `sampling.select_t`、`sampling.time_rules`、`sampling.time_objectives` 和 `contraction.two_point` 仅导出新实现中的对象，保持必要调用兼容，不保有算法副本。新代码及文档应导入 `sampling.temporal.api`。

## 测试与验证范围

TS 独立 CPU 测试位于 `tests/sampling/temporal/`，目录 `__init__.py` 保证 `python -m unittest discover -s tests` 可递归发现：

- 搜索：小 Lt 上与独立全组合枚举比对、参考平移轨道计数、最小周期、空/自定义判据、排序与并列、预算中断、Q 的直接定义和参考数值、奇数 Lt 与非法输入。
- 重建：selection 直接调用的 40×40→128 独立物理配对参考、全时间退化、所有 spectator 轴位置、缺失策略、混用参数及类型/shape 拒绝；原兼容用例继续覆盖稀疏索引差、显式坐标、非均匀配对数、sink 过滤、复数、无序坐标、已知周期信号及输入不变。
- 兼容性：新 API 与四个旧路径的函数对象身份一致，且 `TimeSelection` 与共同契约身份一致。

当前可执行用例位于 tests/sampling/temporal；新增 absolute/cyclic/asymmetric 配对参考、产物/恢复和 cfg 模板复用测试。

这些 CPU 用例验证小格点搜索和数组级配对算法，不是物理系综或生产规模验收。GPU/QUDA、MPI、生产规模及集群作业不属于本次 selection 前端验收。搜索超大组合规模的运行时间、广泛 dtype/array-like 后端行为和端到端稀疏估计统计性质也不由这些用例覆盖。

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

## 统一时间归约服务（供现有输出入口组合）

时间映射、计数、均值均归 `sampling.temporal.api`；没有文件 I/O 或数值容器格式依赖。
已有 selection/显式坐标前端保持原签名与校验，改为使用同一纯服务。

公共重建、`mean_time_pairs` 和 `count_time_pairs` 各自在首次接收配对几何时验证范围；
同一次调用内，私有平均/计数 helper 复用已接受的几何，不再次执行完整坐标范围扫描。
没有跨调用缓存，也没有公开跳过校验参数。原始坐标唯一性、几何 shape、轴、数值 dtype、
选择掩码及缺失 dt 政策仍按各入口原有契约检查。
`tests/sampling/temporal/test_validation_boundaries.py` 计数显式/绝对槽/循环差槽重建
各一次配对几何扫描，并用独立逐配对累加比较值与计数；坏公共输入仍失败。

- `periodic_time_difference(source_time, sink_time, *, Lt)` 对已验证的物理标量或可广播数组求周期差
- `pair_time_differences(rows, columns, *, Lt, source_times=None, sink_times=None, time_layout='absolute')`
  返回 source×other 物理差几何表；无端点表时 rows/columns 是物理坐标，`relative` 表示共享物理 dt；
  有显式有序端点表时为槽坐标，支持 absolute 或同序端点的 cyclic_difference。几何表不包含贡献数组
- `count_time_pairs(differences, *, Lt, valid=None)` 按显式可用配对计数，不读取科学数值
- `mean_time_pairs(data, differences, *, Lt, source_axis, other_axis, valid=None, missing='raise', dtype=None, sum_dtype=None)`
  返回 TwoPointResult，只删除源轴，以物理 dt 替代另一时间轴，其余轴相对顺序不变。
  默认 `dtype=result_type(data.dtype,float64)`，默认 sum_dtype 同输出；调用方可显式指定原保存精度。
  几何 shape 必须与时间轴一致，缺失可选 raise/nan，不能隐式补零
- `TimePairAccumulator(fields, *, Lt)` 的 fields 映射字段 key 到 `(shape,dtype)`。
  `add(source_time, sink_time, values, *, reject_equal=False)` 立即消费 `(key,array)` 流并增加共享 counts，
  values 不被保留；输入时间与字段由其输出 binding/plan 验证。
  `mean(key, dt, *, inplace=False)` 返回已覆盖原 dtype 均值；inplace=True 会消耗相应 sums bin，不能二次平均。
  `write_means(key, coordinates, out, *, missing='nan')` 写入调用方尾部 dt 轴，保留原生除法精度；缺失也可 raise。
  `counts` 与 `sums` 是调用方可检查的工作状态，buffers 按 `(Lt,*shape)` 分配，无 source×sink 数值栈

三条现有入口的政策有意保留：TS 数组前端默认 missing='raise'、float64/complex128 提升；
`storage.save_correlator(average_tsrc=True)` 默认 missing='error'，float/complex 保留原输出 dtype、
求和提升至 result_type(dtype,float64)，整数/bool 输出 float64；完整 cfg 流式输出原 dtype 累加与除法，
未测 dt 使用 counts=0/NaN。storage 只遍历容器和序列化显式归约描述，不再维护第二套配对/求和公式。
这次职责归并没有改变估计量、循环差槽解释、非对称选择或缺失策略，也不保证稀疏样本无偏。

验收补充：`tests/test_output_architecture_alignment.py` 以独立双循环分别对照提升精度和原 dtype
逐项累加，比较三种入口的值/counts/缺失拒绝和 NaN，并检查流式常驻布局；既有 selection 重排、
非对称选择、循环差及各算符轴位置仍由 `tests/sampling/temporal/` 和槽保存测试覆盖。


## 可选设备驻留的完整 cfg 时间归约

`DeviceTimePairAccumulator(fields, *, Lt, backend, one_point_slots=0, debug=False)`
消费纯 CfgOutputField 描述与显式已驻留 backend 数组，保存 `(Lt,*shape)` 原 dtype
求和缓冲以及 `(slots,*shape)` 首次 one-point 值。`counts`、`seen` 和物理时间/逻辑槽
映射仍是小型 CPU 元数据；不把源×汇贡献数组堆叠到 CPU 或设备。

`add(source_time, sink_time, values, *, slots=None, reject_equal=False)` 按调用次序原位累加，
无原子乱序归约；调用方须保持同一有序流或建立依赖，并让借用数组存活至消费完成。
`means(path, dts, *, missing='nan')` 与 `one_point(path)` 返回独立拥有、时间轴在尾部
的设备数组；`finalize(dts, *, missing='nan')` 返回全部字段，不修改 sums 或重复平均。
未测 dt 可返回 NaN/零计数，但所有声明的 one-point 时间槽必须覆盖；缺槽明确拒绝。
`close()` 释放内部引用，不使已交付数组失效，也不负责流同步。

`estimate_device_accumulator_bytes(fields, Lt, slots, dt_coordinates)` 在分配前纯计算
常驻求和/首次值及全部最终输出的逻辑字节；字节打包、D2H、在途所有者、allocator rounding
与 runtime/context 由调用者另外预算。TS 不打包、不读写 storage，也不探测设备额度。
正常路径只检查小型 metadata；debug 有限性检查可同步设备。NumPy backend 只用于协议/
数值参考，真实 CuPy tests 和 [GPU 后处理验收](../results/gpu_output_postprocessing.md)
才提供指定设备的证据，不能由 CPU 通过推断 CUDA/HIP 通过。
