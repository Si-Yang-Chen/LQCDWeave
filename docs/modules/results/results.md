# R：结果恢复（L2）

日期：2026-09-29。职责按 [L0/L1](../../architecture/correlator-l0-l1.md) 与 [H03/H07](../../architecture/correlator-target-architecture.md)。R 把 C 的 `OutputRecipe`、S.api 的符号 Diagram 和 X 交付的 TermValue 转回用户表达式，也按已存 PlanBundle 的窄型 scalar Meson recipe 恢复工作负载输出；不搜索、不读取输入、不执行张量缩并，也不导入 P/X/B 实现。

实现位于 contraction/results/core.py，公开边界为 contraction/results/api.py。OutputNode、OutputRecipe 由 C 唯一定义；TermValue 复用 contracts.execution。调用方使用 R.api；`assemble_workload_outputs` 保留 PlanBundle metadata 与既有函数语义，和 `OutputRecipe` 的组装分别处理。

固定单点配方由 `results/bound.py` 一次绑定节点调用、term 索引和数值系数。
同次组装只复用不可变标量的计算结果；数组算术仍创建原来的独立 owner，matrix/select
保持已有选择缓存。纯 Python 数值常量表达式按原有顺序计算，不能把算术异常提前到绑定期。
每次调用使用独立临时缓存，结束后不保留输入或结果；不重排加法或乘法，精度由 C 的配方
显式决定，扫描轴保持原序。

`OutputRecipe.dtype` 声明最终字段 dtype；`intermediate_dtype=None` 保留原逐节点规则。
普通 calculate 新规划的 float32/complex64 输出另记录 float64/complex128 中间精度：term
值与系数在输出算术前转换，乘积、求和、幂和矩阵布局节点保持该精度，直到完整数值字段末尾
才转回 `dtype`。单纯宽化末求和不能恢复此前乘积的舍入损失。输入、准备与 GEMM 的 dtype
不随此输出合同升级。普通绑定和 debug 解释器遵守同一规则，嵌套容器分别转换数值字段。
元数据估算计入宽 term 转换、算术 owner 及最终窄字段同时存活；GPU 分段输出的 scratch
保持宽 dtype，末字段仍窄，所有 owner 和指针表经原 P 服务收费。旧档案缺少新字段时补 None，
不升级其数学/舍入语义；新档案固定保存该选择，加载不重新规划。
乘法仍从初始 1 按原顺序逐项计算；开头连续的纯 Python 数值常数在绑定期执行同一段左乘，
逐点从该段的结果继续，遇到第一个非常数节点即停止预计算，不跨过它合并后续常数。
若该常数段发生算术异常，保留原调用期运算。不把加法改成分段求和，也不因零常数
省略后续可能报错的操作。绑定完成后断开递归绑定函数的自引用，立即释放只在绑定期间使用的
节点索引和常数描述；不等待循环垃圾回收，也不修改全局 GC 开关。
高频的 term 与 multiply 节点把本次缓存访问和原有数值运算绑定为同一个调用，
不先调用缓存包装器再转调运算函数；只在结果为不可变标量时记入缓存。
term 绑定直接创建最终求值器，不先创建再丢弃一个原始运算闭包；项索引和系数
只由该 term 的求值器捕获，其它节点的绑定帧不为它们创建闭包单元。
乘法开头的字面量节点不再生成随后不用的逐叶求值函数；剩余子节点仍依次绑定，
非纯常数乘法直接创建最终带缓存求值器，不先建立再丢弃一个原始乘法闭包。
`debug=True` 和普通外部装配仍沿完整核验及递归解释实现，作为独立数值参考。

<a id="全-diagram-整数连接格式目标待迁移"></a>

## 全 Diagram 整数连接格式（实现完成）

权威格式见[ARCHITECTURE](../../../ARCHITECTURE.md#diagram-的统一整数连接格式目标待迁移)。R 从 `ContractionTerm.subscripts: AxisPattern`
读取输入和输出轴 ID，以整数输出轴、结果映射和原有顺序恢复用户表达式；R 不解析或生成
einsum 方程。输出布局与配方保持不变；受影响 CPU 模块验证通过；GPU/生产规模仍未验收。

## 完整 cfg 输出安排（统一设备执行）

目标见[完整规划与直接执行](../../architecture/correlator-target-architecture.md#完整规划与直接执行)。R 沿用 C 的 `OutputRecipe` 描述每个输出点；收缩编排把配方、坐标映射和缓冲安排接入内存接收与 DatasetWriter 流式输出。X 的 NumPy/CuPy 统一执行在输出边界交付主机结果，R 使用同一配方恢复。全基底、真实 MPI 生产规模仍待验收。

完整 cfg 规划阶段保存输出字段、dtype/shape、坐标顺序、physical-dt 累计规则和缓冲额度。执行阶段逐点组装并交给收缩编排；memory iterator 由消费者逐 cfg 接收后释放数值结果。持久结果按 cfg 写入 HDF5 具名字段，并在数据集根 `dataset.json` 中登记；只有完整 cfg 字段写完才登记该 cfg 完成。输出根目录在执行期绑定，不进入档案。

## C 交付记录

    @dataclass(frozen=True)
    class OutputNode:
        kind: str
        value: object = None
        children: tuple[OutputNode, ...] = ()
        keys: tuple = ()
        shape: tuple[int, ...] = ()

    @dataclass(frozen=True)
    class OutputRecipe:
        root: OutputNode
        term_ids: tuple[str, ...]
        coordinates: CoordinateSpec
        dtype: str | None = None
        intermediate_dtype: str | None = None

OutputNode.kind 使用 diagram_sum、term、constant、constant_array、add、multiply、power、dict、list、tuple、array 等节点。R 保留 C 记录的类型身份，不另定义同名树节点。

## 公开函数与精确签名

结构化请求的配方组合由独立 `results/structured.py` 实现，公开函数为
`compose_output_recipe(root, replacements, term_ids, coordinates)`。
构造期基础结果叶子替换成 native provider 的数值配方，复用不可变子节点，
返回共同 `OutputRecipe`。未知结果 ID 和非法节点明确报错。组合只处理现有
节点类型，不引入第二套执行 IR；装配、保留量核算和档案编码沿用原接口。

### collect_diagrams

    collect_diagrams(value) -> tuple[Diagram, ...]

从 SymPy Add/Mul/Pow 和嵌套 Mapping、ndarray、list、tuple 收集 S.api Diagram，按首次出现顺序去重。数值标量和非 object ndarray 是常量；不支持的叶子报 TypeError；递归容器引用自己时报 ValueError。该函数只检查结构，不展开数值网络。

### build_output_recipe

    build_output_recipe(
        correlator,
        diagram_terms,
        term_ids,
        coordinates,
        *,
        diagram_recipes=None,
    ) -> OutputRecipe

参数含义：

- correlator：原始符号表达式与外层 Python 容器。
- diagram_terms：Diagram 到 TermContribution tuple 的映射；每项必须引用一个已知 term_id。
- term_ids：本次展开的唯一 term_id 顺序。
- coordinates：C CoordinateSpec，决定固定坐标或扫描坐标的恢复 shape。
- diagram_recipes：可选 Diagram 到 OutputNode 子树的映射，用于已明确描述的线性/乘积项。

扫描顺序识别 Diagram 类型只从 lqcdweave.operators.correlator.api 导入 S.api。Add、Mul、整数幂 Pow 映射为纯表达式节点；非整数幂报 TypeError。Mapping 记录键和值顺序；object ndarray 按 C-order 递归记录并保留原 shape。数值 ndarray 被复制成只读常量，数值标量转为 Python 数值。循环容器报 ValueError。

一个 Diagram 可以在多个输出位置复用；diagram_terms 中每个 contraction term 只需一次定义。一次构造内，相同符号子表达式和不可变 SymPy 数值标量复用 recipe node，每个 Diagram 的配方仅核对一次；collect_diagrams 也跳过已经访问的共享子结构。缓存仅在本次调用内存在，循环容器仍报错。数值数组仍按输出位置复制为独立的只读常量；数值装配保持各输出位置，不承诺不同可变结果具有 Python 对象别名共享。

### assemble_correlator

    assemble_correlator(expanded, term_values) -> object

只读取 expanded.output_recipe、expanded.coordinates 以及 TermValue，不访问 expanded 的 adapter、bindings 或输入 loader。它先验证 recipe 与坐标一致、term_id 唯一、坐标 key 唯一且 scan_shape 与 point 数一致，再将 term_values 建成 (term_id, coordinate_key) 映射，要求每个组合恰好出现一次。

表达式求值规则：

1. diagram_sum / term 节点读取原始数值网络结果，并在此处应用各自的 TermContribution.coefficient 一次。
2. add、multiply、整数 power 节点按原符号表达式求值；不执行网络缩并，也不额外再次计权。
3. dict、list、tuple 保持容器类型、键/元素顺序和嵌套；空容器原样返回。
4. object ndarray 保持原 shape。均匀的标量或等 shape 数组子项可以内部 stack 成数值 ndarray；若子项含嵌套容器或 shape 不同，则返回 object ndarray，不自动 stack。
5. 固定整数时间不新增扫描维；time_axes 与 time_pairs 按 CoordinateSpec 请求顺序恢复。标量扫描结果 shape 为 scan_shape；数组值轴放在扫描轴之前；顶层 ndarray 结果再保留原 ndarray shape。
6. 常数标量沿扫描坐标广播；常数 ndarray 的结果 shape 为原 shape 后接 scan_shape。

TermValue 到达顺序不影响输出坐标顺序；坐标 key 决定回填位置。重复项、缺项、未知 term_id、未知 node kind、错误容器 shape 与错误系数类型分别报 ValueError 或 TypeError。

### bind_output_assembly

    bind_output_assembly(recipe, *, debug=False) -> callable

用于反复组装同一个固定输出点。调用约定要求 `recipe` 为 `OutputRecipe`，坐标没有扫描轴且
只有一个点。默认直接消费构造/规划边界交付的正确冻结配方，不重新检查类型或坐标支持范围。
只有 `debug=True` 才在绑定时诊断这些前置条件，并按 `assemble_correlator` 的既有规则核对
term ID、完整节点树与坐标记录；扫描坐标（即使只有一个扫描点）以及多点配方在 debug 报错。

返回的 callable 接收一个有长度并支持整数索引的值序列，顺序必须与 `recipe.term_ids` 一致；
每项是该单点对应 term 的标量或数组值。默认直接运行已绑定的表达式，值数量与顺序由
固定执行表交付；数量复核放在 debug 路径。它不构造 `TermValue`，也不重复遍历配方、构造覆盖集合
或核验键。`reshape`、`transpose` 和 `scatter` 仍执行原操作；绑定路径跳过这些节点重复的运行时
shape/size 断言，依赖冻结配方与对应 term 值顺序。

`debug=True` 保留诊断路径：每次调用检查数量，把有序值包装为单点 `TermValue`，交给
`assemble_correlator` 完整核验并求值。`debug` 必须是 Python `bool`；该运行时选项不进入计划或档案。
普通 `assemble_correlator` 仍是完整外部边界，每次检查所有 TermValue 的唯一性和完整覆盖。

绑定时为可共享的算术节点固定整数槽，调用时只分配本次标量缓存，按槽读取，
省去逐节点的对象 ID 哈希查询。只缓存不可变数值标量；数组结果仍按原有出现次数
计算并保留独立 owner。值序列可只提供长度和整数索引，同一绑定的后续调用可改变值的秩。
缓存不跨调用，矩阵/选择节点仍使用原有的单次调用缓存。

固定输出的细分审查如下；“保留”指实际交付、算术、结构或生命周期操作，
“debug”指正确冻结配方下不改变结果的重复诊断。缓存命中只替代同次调用中
相同不可变标量的重复计算，不能据此跳过数组操作。

| 步骤 | 实际操作 | 省略的影响及处理 |
|---|---|---|
| 1 | 读取本次 `debug` 选项 | 保留；决定是否运行完整参考路径 |
| 2 | 检查 `OutputRecipe` 与坐标记录类型 | debug；正确固定配方不需要重复断言 |
| 3 | 断言没有扫描轴且只有一个坐标点 | debug；固定单点是调用前置条件，不是逐点数据转换 |
| 4 | 遍历所有 term ID 检查格式及重复 | debug；执行表已固定有序 term 交付 |
| 5 | 递归校验全部节点种类、系数、维度及引用 | debug；不在普通绑定重复审核 |
| 6 | 检查全部坐标 key、点数及扫描维一致性 | debug；与步骤 3 一起诊断调用前置条件 |
| 7 | 创建 `term_id → 整数索引` 表 | 保留一次；决定每个叶子读取哪个输入 |
| 8 | 按对象身份查找已绑定节点 | 保留在绑定期；共享子树只生成一次函数 |
| 9 | 依次绑定子节点 | 保留剩余子节点；已按原序算完的乘法开头字面量不创建无用逐叶函数 |
| 10 | 将系数转为既有 Python 数值形式 | 保留一次；否则混合精度及 SymPy 数值规则会改变 |
| 11 | 为共享算术固定整数槽 | 保留一次；调用期不再重复查询节点 ID 字典 |
| 12 | 整体构造 `TermValue` 与单点展开包装 | 普通路径删除；debug 才构造参考输入 |
| 13 | 调用时将有序序列复制为 tuple | 普通路径删除；直接用整数索引消费，debug 为覆盖检查收集 |
| 14 | 调用时核对序列总长度 | debug；固定执行表保证数量与顺序 |
| 15 | 为本次调用分配空标量槽和矩阵/选择缓存 | 保留；不能跨点使用上一次数值或数组 owner |
| 16 | 每个算术节点按 ID 查询字典 | 删除；直接读取步骤 11 绑定的槽 |
| 17 | 缓存命中后复用本次数值标量 | 保留；term/multiply 合并缓存与运算调用，删除二者之间的转调 |
| 18 | 缓存未命中时访问有序输入对应索引 | 保留；这是本次点实际 term 值 |
| 19 | 零维 ndarray 用 `.item()` 恢复标量 | 保留；不能改成 NumPy 标量而改变提升规则 |
| 20 | 对 term 值施加本叶子的系数 | 保留；系数不能遗漏或重复应用 |
| 21 | `diagram_sum` 按原顺序逐项乘系数并由 0 求和 | 保留；不展开或重排浮点和式 |
| 22 | `add` 从 0 调用原 `sum` | 保留；包括 Python 版本既有求和规则 |
| 23 | `multiply` 从 1 按子节点顺序左乘 | 保留算术顺序；开头连续纯常数段只算一次，当前点继续剩余左乘，不合并中间或尾部常数 |
| 24 | `power` 使用当前底数与已存指数 | 保留；错误仍在调用时触发 |
| 25 | 对全为 Python 字面量的算术子树预计算 | 保留原顺序；可固定纯常数，算术异常则留在调用期 |
| 26 | 判断当前算术结果是否为不可变数值标量 | 保留；把数组写入共享缓存会改变 owner 关系 |
| 27 | 仅将该次不可变数值写入整数槽 | 保留；不记住输入数组或历史值 |
| 28 | 每次出现常量数组时创建副本 | 保留；否则修改一个输出会影响其它输出或配方 |
| 29 | 对数组算术的重复出现分别执行 | 保留；结果必须有原来的独立 owner |
| 30 | matrix 节点运行 reshape/transpose/scatter | 保留；其轴和落点决定输出含义 |
| 31 | matrix 节点逐次重复断言已知 shape/size | 普通绑定路径删除；debug 参考检查 |
| 32 | select 按既有索引选择；复用当前选择源 | 保留；不能重新计算出不同源或丢失索引 |
| 33 | 按保存的键与次序新建 dict/list/tuple | 保留；保持前端请求的容器与独立可变结构 |
| 34 | 判断 array 的当前孩子是否均为同形数值 | 保留；当前值可在不同调用间改变秩 |
| 35 | 同形数值 stack 后 reshape | 保留；恢复声明的算符轴和当前数值尾轴 |
| 36 | 嵌套异构值按索引填 object array | 保留；不能默认为均匀数值数组 |
| 37 | 无子值时创建声明 shape 的空数组 | 保留；空结果也必须保持结构 |
| 38 | 返回新容器及当前结果 owner | 保留；不把内部临时缓存交给下次调用 |
| 39 | 释放本次缓存持有的临时值 | 保留；绑定函数只持有结构和常数 |
| 40 | 绑定完成后继续保留递归绑定函数的自引用 | 删除；断开自引用，绑定期工作表立即释放 |

档案读取、格式分派与保存的另外 36 个步骤见
[计划档案 CPU 审查](../plan_archive/plan_archive.md#cpu-加载与固定输出的逐步审查)。

### assemble_workload_outputs

    assemble_workload_outputs(workload, values, backend) -> object

这是 X persisted generic session 使用的 R.api 结果边界。若 C WorkloadSpec 没有 adapter 标记，则原样返回 `tuple(values)`，不复制已由 X 交付的拥有型值。对已通过 C 校验的 `meson-simple-v1`，它按 `metadata.output_recipes` 的顺序将每组 term value 相乘，以显式 backend 转换每项并 stack；该存档 recipe 是扁平 term-index 分组，不是 `OutputRecipe` 的坐标表达式树。term 系数已在执行值中生效，不在此重复应用。CuPy backend 下会同步调用方当前 stream 以完成本次装配；R 不选择设备、不创建或持有 stream，也不负责设备生命周期。

### 同目标 scatter 求和

R 在普通及固定坐标装配中识别同目标形状的 scatter 求和，按原顺序将紧凑块累加到自有结果矩阵。
块可重叠，块内索引仍须唯一；dtype 按原始从 0 开始的逐项数组加法提升，混合 dtype 时仅在提升处复制结果。
不修改借用输入、不跨输出 occurrence 共用可变结果；扫描轴仍在尾部，固定绑定只预存位置索引。
不再为每个分块保留一份完整零矩阵，不改写 OutputRecipe 或引入新节点/档案格式。

元数据预留与实际装配同时调整：计入紧凑子表达式、索引、gather/add 暂存和提升时共存的两个完整 owner。
不同输出 occurrence 仍分别计费。该更改不改变缩并、路径搜索和资源上限；GPU 的既有融合输出程序继续消费相同配方，
CPU 数值与资源验证、元数据兼容均不能代替真实 GPU 验收。

### estimate_output_retention_bytes

    estimate_output_retention_bytes(
        recipe,
        problem,
        batch_output_axes,
        coordinates,
    ) -> int

该 helper 由 E 调用，估算本任务所有 TermValue 与嵌套 recipe 数组 payload 的驻留量。它按 ContractionProblem 的 term shape/dtype、输出 lane 轴、扫描坐标数和 recipe 中的逐元素算术估算临时数组，并有意对嵌套算术临时量采用偏保守计数。term IDs、batch_output_axes 数目或 recipe 引用不相符时报 ValueError/TypeError。

一次估算按不可变 OutputNode 的对象身份复用子树的 shape/dtype/bytes 计算；父节点仍按每次出现累加完整字节数。重复输出不会因共用配方而少计预留，也不缓存数值装配结果。

E 将估算写入 C 的 `DiagramInputs.assembly_host_bytes` 或 `assembly_device_bytes`，按实际 backend 保留已有更大的预留量，并交付到各 task batch。P 只消费这些中立字段，将它们加入 host/device 预算预留，不依赖 R 或重新估算 recipe；依赖方向保持 E → R.api、E/P → C。

估算包含数组 payload bytes 及容器/object-array 的指针槽位，不包含 Python 对象和容器本身的 header、ndarray header；不是进程 RSS 上界，也不代表完整结果必然可放入 host memory。

## 所有权与失败

R 不持有或释放 loader、raw block、device stream 或 contraction 中间量。X 在交付 TermValue 前必须将复用缓冲中的结果复制到稳定存储；R 不保留输入 ExpandedCorrelator 的数组别名，也不选择设备或管理 stream。结果对象由 assemble 调用方持有；PlanBundle 后端装配只完成自身输出后同步传入 CuPy backend 的当前 stream。

失败只发生在 recipe、坐标或结果覆盖检查，以及纯表达式求值；R 不清理 reader 资源，不重试执行，也不补充缺失数据。无法确定表达式项、非整数幂、非法循环容器或坐标映射时明确失败，不静默补零、重排或猜测 alias。

## 验证、限制与兼容

CPU 单元测试为 tests/correlator/results/test_result_structure.py，覆盖多轴与 time_pairs 坐标顺序、dict/list/tuple/object ndarray 嵌套、空容器、常数广播、数组值轴位置、重复 Diagram、系数只应用一次、缺失/重复/额外 TermValue 和循环容器；同时验证 PlanBundle 两组复数乘积 recipe 与 generic owned-array tuple 行为。`tests/correlator/results/test_bound_output_assembly.py` 覆盖单点有序值绑定、多节点共享与系数、矩阵结果操作、正常路径不深审、debug 配方与数量错误，以及扫描点拒绝。`tests/test_contraction_session.py` 验证 archive reload 后两个复数 recipe 的 session 执行、复用 loader buffer 时结果保持稳定、再次执行不搜索新计划，并在 session 建立时拒绝未知保存 adapter。

`assemble_correlator` 的容器和广播实现使用 NumPy，故不声明 CuPy 数组回填已验收；PlanBundle 装配的 CuPy 当前-stream 同步分支也未做 GPU 平台验收。旧 result_structure.py 不再提供导入路径；上述服务由 R.api 公开，OutputNode/OutputRecipe 仍由 C 唯一定义，TermValue 仍复用 contracts.execution。C/S 实现仍各只有一份。


## 基础数值结果与符号入口的依赖

R 的模块导入只消费 C 的唯一 OutputNode/OutputRecipe、既有 TermValue 与
基础数值库；清理无消费者的 dataclass 导入。`collect_diagrams` 和
`build_output_recipe` 内各自在调用时从 S.api 导入真实 Diagram，再加载
SymPy。两个函数的符号处理算法保持原实现，不能只延后其中一处导入。

`assemble_correlator`、`assemble_workload_outputs` 与
`estimate_output_retention_bytes` 对已构造的合法纯记录无需符号库；拥有型
数组、backend 转换/同步、系数和坐标规则沿用上述原约定。缺少可选库时
symbolic 入口沿 S 公共边界给安装指引，其他导入错误原样传播。
新子进程检查在选定源码/隔离安装包路径上模拟缺库并实际组装 C records；
原符号结果测试继续验证真实 Diagram 和 recipe 的同一类型。

## Dynamic Diagram result assembly

`results.api` exports `assemble_simple_diagram_outputs(values, output_recipes, backend) -> array` and `assemble_baryon_diagram_outputs(values, backend) -> array`. Implemented in `results/dynamic_diagrams.py`, these functions consume completed per-term values only; they do not plan, execute, or read getters. Both require an explicit NumPy/CuPy backend. CuPy assembly synchronizes the caller's current stream after producing its result and does not create or retain a stream.

For simple diagrams, each `output_recipes` entry holds term indexes for one output. R multiplies each recipe's values in order and stacks the diagrams in input order. A diagram with no contraction groups returns the multiplicative identity `1.0`; no diagrams returns an empty one-dimensional array. When each recipe contains one value with uniform shape and float64/complex128 dtype, R retains the legacy one-stack neutral-multiply fast path. Other shapes and dtypes use ordered per-recipe multiplication and backend stacking. Dynamic simple results may be non-scalar; the persisted Meson scalar-group rule does not constrain this path.

Baryon outputs stack one scalar for every topology in order. Empty input returns shape `(0,)` with `complex128` dtype, matching the current adapter. R does not reapply coefficients or infer dataset identity from values, shapes, keys, or source declarations. `tests/correlator/test_dynamic_diagrams.py` checks the B→E→existing compiler/executor→R NumPy path against independent `numpy.einsum` scalar and vector references, plus repeated getter-buffer snapshots, multi-group recipes, C topology/term identity, empty inputs, and failure ordering.

## 整行结果选择

共同 OutputNode 增加 `kind="select"` 的 R 解释：value 为非空非负整数 tuple，
恰好一个 child；按 tuple 选择结果的前导轴，时间扫描轴仍在尾部。用于 E 的
compact meson 行向量输出。校验拒绝未知类型和非法结构，保留量估算拒绝越界
及索引扫描轴。数值装配缓存同一个 child 的整行 coordinate 恢复结果，避免
每个元素重复恢复整行；各最终矩阵字段仍创建独立数组。通用配方行为保持。

## 矩阵块组装节点

既有 OutputNode / OutputRecipe 支持 reshape、transpose、scatter、zeros 四种纯结果操作，内部实现
独立于 matrix_output.py。R 不读取输入，不重算缩并，不决定 Wick 或算符共轭。

- reshape：一个 child，shape 指定坐标轴之前的完整目标形状；元素数须相等。用于将两侧
  圈向量扩展为 `[left,1,left_PC,1]` 和 `[1,right,1,right_PC]`，随后由既有 multiply 组合。
- transpose：一个 child，value 是结果轴的完整排列，shape 指定目标形状；末尾扫描轴
  保持原位置。用于将有序物理块的 PC 组合排为 `[left,right,left_PC,right_PC]`。
  返回数组视图，估算保留其完整底层数组。
- scatter：一个 child，value 为各前导算符轴的目标索引元组，shape 为完整目标形状。
  每轴索引须唯一且在范围内；子结果形状须与矩形、其余 PC 轴及末尾扫描轴一致，空轴合法。
- zeros：无 child，value 为数值 dtype 名，shape 为目标形状；用于零 Wick 区域和纯零任务。

assemble_correlator 保留末尾扫描轴，共享的矩阵节点在一次组装内只处理一次。输出保留量
估算覆盖完整目标、scatter 索引及 reshape 最坏拷贝；不作为 RSS 或 GPU 上限。独立专项
覆盖两个坐标、圈外积、PC 轴、不连续/空矩形、零输出、畸形索引及形状拒绝。

## 完整 cfg Dataset 输出

`complete_calculation(..., output_mode="dataset", physical_lt=...)` 在规划时冻结完整字段表、原始
坐标次序和 raw/dt 的计算配方。收缩编排的 `contraction.correlator_output` 组合 `storage.correlator_results.CfgOutputWriter`，
按字段保留各自真实 shape/dtype 写入每 cfg 的 HDF5 文件；公共科学 metadata 与 cfg 登记保存在共享根 `dataset.json`。

默认 CPU 后处理的 dt 累计由 storage writer 调用 TS.api 的 `TimePairAccumulator`，使用当前 cfg 的有序 `physical_times`，按 `(t_sink-t_source) mod physical_lt` 分组求均值；
未覆盖的 dt 由 reader 返回 NaN，计数为零。每端的字段值在完成当前 cfg 写入后释放。写出或计算异常
会中止当前 writer，失败 cfg 不会登记为完成。

`contraction.correlator_output.load_correlator_output(directory, configuration=cfg, path=(), *, dt=False)` 重导出 storage 的同名读取服务，按具名字段恢复原 shape/dtype。dt 坐标、缺失填充及 one-point 字段服从输出配方；storage 将每 cfg 结果保存为 HDF5 具名字段，并由共享根 `dataset.json` 登记。输出布局容量由 P.api.configure_cfg_output_layout 纯计算；R 保留字段数值及容器组装职责。

显式 `postprocess_backend="gpu"` 时，R 保持冻结配方的语义验证与 CPU 数值基准；
X 将固定 `OutputRecipe` 编译为逐字段融合程序，保留表达式顺序、复系数、dtype/中间转换
和字段结构。X 消费设备 term 值并执行数值组装，组合 TS 的有序 dt/one-point 设备归约，
storage 将原始 shape/dtype 字段写入 HDF5；GPU 传输阶段可临时打包，不改变已保存字段格式。
CPU 只保留小型物理 dt 映射、计数与首次覆盖标记；不逐对回传 term 再组装。
`dt_packed` 最终下载，`pairs_and_dt` 仍逐对下载。科学字段定义、身份与 dtype 保持，
不构造完整源×汇数值栈。默认 CPU、GPU 目标要求、预算及数值/实机比较见
[GPU 后处理](gpu_output_postprocessing.md)。

限定 CPU 验证覆盖字段 shape/dtype 与数值保留、raw/dt 坐标恢复、缺失 dt、两 cfg 的数值差异、写出失败时不登记完成
完整 cfg，以及独立 ordinary 数值参考。此结果路径尚未在真实生产作业、MPI 多 worker 或 GPU 上验收。
