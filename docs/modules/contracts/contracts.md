# 共享数据契约与 C：中立 Correlator 契约 L2

日期：2026-10-07。上层基线：[总体架构](../../../ARCHITECTURE.md)、
[Correlator L0/L1](../../architecture/correlator-l0-l1.md) 与 [production L0/L1](../../architecture/production-l0-l1.md)。

<a id="shared-data-contract"></a>

## 共享数据契约：版本、轴、精度、身份、选择和覆盖

本节是 production → storage → reader/binding → contraction 的共同数据语义入口。
它汇总已有公开记录，不引入新的 `ArrayContract`、统一版本号、序列化格式或运行时层。
通用 `contracts.dataset`、时间/空间采样、后端协议和下面的 Correlator C 各有明确职责；
不能把通用数据集登记改成 C 的执行 IR，也不能把 C 的 `TensorSpec` 当作已发布数据。
模块文档只补充各自物理布局、接口和实现限制，通过本节引用共同语义。

### 版本分别属于各自记录

| 已实现的记录 | 唯一定义与当前语义 |
|---|---|
| `DatasetSpec` | [`contracts/dataset.py`](../../../src/lqcdweave/contracts/dataset.py)：严格 `schema: 1`，精确字段为 `schema/kind/configuration/axes/dtype/bases/definition/conventions/parents`；未知版本或额外字段拒绝 |
| 实际产物发布 | [`storage/dataset.py`](../../../src/lqcdweave/storage/dataset.py)：共享 `dataset.json` 登记公共科学 metadata 与各 cfg 的身份/覆盖；每 cfg 数值保存在完整 NPY 或 HDF5 文件 |
| 执行/来源记录 | 执行参数与来源 knownness 按各自科学契约记录，不混入数据位置或文件布局；高级恢复状态不在本节承诺，见[默认存储格式](../storage/data_storage.md) |
| 实际时间记录 | [`TimeSelectionRecord`](../../../src/lqcdweave/contracts/time_selection.py) 文档只有 `Lt/times`，不另加 schema；保存记录不等于产物登记 |
| 收缩计划档案 | 独立的 envelope/payload 版本及显式迁移规则见[计划档案 L2](../plan_archive/plan_archive.md)；不得用其版本判断数据产物是否兼容 |

这些记录不是一条可互换的“数据版本”序列。读取不猜测科学格式、不按包版本自动改写
科学含义；未支持的版本必须走该模块明确提供的迁移或重建入口。

### 具名轴与 dtype

- `DatasetSpec.axes` 是有序的 `{name, coordinates}` 列表；名称非空且唯一，每轴坐标非空、
  已知且按 canonical JSON 唯一，`shape` 从坐标长度得到。格点总尺寸固定为 `[Lx,Ly,Lz,Lt]`，
  不代表实际数组轴顺序。各产物的源/汇、spin/color、模态/点、动量和路径轴须显式声明
- 坐标比较复用 `canonical_json`：`true`、`1`、`1.0` 是不同标签；同内容 tuple/list
  序列化为同一个 JSON 数组。物理域可以更严格，例如时间和实际模态索引必须是整数

Correlator 的纯整数坐标身份直接产生与 canonical JSON 相同的十进制文本，避免为每个
模态/点标签创建 JSON 编码器。布尔、浮点、字符串及整数子类仍沿原编码规则，NumPy
标量先按已有契约转为 Python 标量；身份文本和拒绝边界不变。

内部构造 TensorDescription 时可引用已验证记录的不可变 metadata 子树。只接受
确切 TensorDescription 实例所持有的 transform/semantics/bindings；普通只读代理
仍递归复制并校验，因为它可能引用外部可变字典。该构造证明仅为 InitVar，不进入
结果记录、字段表或存档；默认构造与独立读回继续完整校验，不引入新数据格式。
- `DatasetSpec.dtype` 描述实际数组精度；NPY/HDF5 保存各字段声明的 dtype，不静默转精度。
  外部裸二进制输入的 dtype 另需声明字节序；NPY 头或文件名不能补出缺失科学约定。
  普通旧 NPY 输入便利接口的读取范围见 [preset](../storage/preset.md)；其文件头只提供 shape/dtype，
  不补充缺失的基底、来源或物理定义。
- `TensorAxisMetadata` 描述 reader 的最终逻辑轴和数值 dtype，轴数须等于 rank，
  支持 NumPy 数值 kind `biufc`。`TensorSpec` 只做 shape/NumPy dtype 描述，允许标量；
  它不是 DatasetSpec 的持久 dtype 白名单。二者详细字段见下文
- 收敛容差属于求解定义；存储精度单独记录。相位、自旋基、边界条件、归一化及 Laplace、
  Dirac、elemental 各自规范场版本属于适用的科学约定，不能从 dtype、shape 或同名文件推断

### 三层身份与不变量

`DatasetSpec.definition_id = stable_digest(spec.document)` 使用排序键、紧凑、UTF-8、
禁止非有限数值的 canonical JSON 和 SHA-256；dict 键顺序不影响身份，列表/轴/坐标顺序影响。
`spec.document` 返回副本。定义包括实际父项及约定，不包含执行尝试编号或文件位置。

`data_id` 标识实际登记的数据；不同基底、坐标顺序或来源不能仅因 shape 相同而视为同一输入。
它不是数组内容校验和，也不证明数值或物理正确性。位置和文件格式不属于科学 identity；
公共 metadata 与 cfg 差异由根 `dataset.json` 登记，执行参数另记。

`bases[axis] = {data_id, indices}` 绑定实际基底/点集及有序索引，索引必须等于该轴
坐标。同 seed、Ne/Np、shape 或生成参数不证明实际基底一致。未知 ID 保持 `None`；
允许描述未知不等于允许严格复用。`DatasetRequest` 只在已定义的父项/科学 ID 槽内保存
`{"request": node}`，由已有请求图解析成真实 ID，不能作为实际产物身份使用。

reader 的 `data_identity` 与 `selection_identity` 分别表示来源和选读；后者不替代前者。
`TensorDescription` 的 `bindings` 保存实际绑定，`semantics` 保存逻辑约定。
`TensorSpec` equality 仅比较 shape/dtype，科学兼容须显式核对 description/source。
`axis_bindings` 再将具名轴绑定到实际 ID 与有序标量标签；不把 integer `AxisPattern`
连接编号当作物理轴名或基底身份。精确字段及检查范围见下面 C L2。

### 选择是视图，完成覆盖是存储事实

默认全物理时间 `Nt=Lt`。只有显式启用时间选择才使用 `TimeSelectionRecord(Lt, times)`；
实际时间保留环绕后的槽顺序，不能排序后冒充同一记录。`TimeSelection` 是周期模板，
两者不可互换为“已生产覆盖”。`conventions.time_coordinates` 的 slot mapping 使用
既有 `{domain: "slot", layout, selections}`；absolute 与 cyclic_difference 由调用者声明。
生产求解仍保持物理 Lt；紧凑 reader/收缩用槽，缺失物理端点报错，不能把邻槽当邻物理时间。
具体操作见[共同时间交接](../temporal_sampling/temporal_production.md)。空间规则与实际点坐标同样分开，
见[空间采样 L2](../spatial_sampling/spatial_sampling-api.md)。

通用 `plan_requirement` 只消费 storage 已核验的 observation。默认逐轴严格核对实际坐标
及顺序；显式 `subset_axes` 才允许选读，返回的 selection 保持请求顺序。只有完整登记的
候选可用于公开复用；写入中的临时状态不作为复用候选。`completed/missing` 是
请求首轴坐标中的位置，不是物理坐标值。候选首轴可只覆盖请求的一部分，其余轴须覆盖
全部请求；一次只选择一个完整候选，不拼接不同产物。精确选择、未知身份阻塞与恢复
规则由[产物规划](../dataset_planning/dataset_planning.md#planning-a-requirement)维护。

storage 的 complete 只证明声明布局和覆盖已完成，`scientific_validation` 仍可 unknown；
metadata-only 查询不打开数组，不能充当数组布局验证过的规划 observation。
按块读入、数值计算分块和输出分块是独立能力，单有 selection 不证明全链超内存支持。
reader/producer 的借用输出必须在下一次读取/计算或关闭前消费/复制；精确生命周期由
[EV](../eigenvector/eigenvector.md)、[EL](../elemental/elemental_production.md)、[PR](../perambulator/perambulator.md) 的公开协议规定。

### 共同语义的回归入口

`tests/contracts/test_shared_data_contract.py` 固定版本、dtype、坐标类型及身份；
`tests/storage/test_shared_data_contract.py` 检查发布和精度边界；
`tests/test_data_contract_handoff.py` 检查实际发布→规划→选读的身份与完成覆盖交接。
进一步的恢复、来源、格式与物理数值验收仍由各模块测试负责。
这些检查不新增协议，也不把本机 CPU 结果扩大为 GPU、MPI、共享存储或真实系综验收。

## C：中立 Correlator 契约

本模块只定义数据描述、验证与注入协议，不读取数值数组、不搜索、不执行、不导入业务实现。
已有记录已迁为唯一定义；正式公开入口 contracts.correlator.api 与 contracts.correlator 包级重导出同一记录。行为 State、CostProfile、Operator 分别留在 P、M、O。

`CfgPlan` 的合批成员检查为每个模板建立一次操作索引表；实例共享原合批对象时直接命中，
独立解码但值相等的对象按相同 Action 索引核对完整描述。不逐实例扫描全部模板合批。
未知合批、参数变化和被标记为复用的合批成员仍在构造/加载边界拒绝。
`DenseKernelBatch.stacked_sides` 区分没有物理自由轴的输入侧与真实单例科学轴。
前者使用一个批内轴拼接独立输入；`DenseBatchResult.squeeze_axes` 只能去除对应
结果切片中的单例批内轴。共同审计核对物理轴配对、原 TensorSpec、完整输出 owner
及工作区，复用既有逻辑 Action 与资源轨迹。

## 全 Diagram 整数连接格式（实现已完成；受影响 CPU 模块验证通过）

权威格式见[ARCHITECTURE](../../../ARCHITECTURE.md#diagram-的统一整数连接格式目标待迁移)。
C 维护唯一 `AxisPattern`：`inputs` 是有序的 `tuple[tuple[int, ...], ...]`，`output`
是 `tuple[int, ...]`。输入轴可重复表示对角，多个输入共享整数表示对齐；输出唯一且必须
来自输入。整数非负、数量不限，且只表示连接关系，不表示物理轴名或运算顺序。
`normalized()` 按首次出现重编号；`dimensions(shapes)` 核对操作数、rank 和共享/对角轴尺寸，
`output_shape(shapes)` 按输出顺序返回尺寸。

整数轴记录构造的主机操作逐项如下。外部整数协议仍在 C 的公共边界归一化；
这里省去的是已知 Python 整数的重复转换和空输出必然成立的集合判断，
不把轴的合法性检查移入 debug。

| 步骤 | 操作 | 必要性及处理 |
|---|---|---|
| 1 | 按原序消费各输入轴组 | 保留；生成器只能消费一次，操作数顺序决定连接关系 |
| 2 | 拒绝直接作为整数轴组的 str/bytes | 保留；外部字母标签由另一公共入口解析 |
| 3 | 逐个读取当前轴 ID | 保留；错误后不继续消费后续值 |
| 4 | 对精确 Python int 再调用整数转换协议 | 删除；当前值已经是所需整数 |
| 5 | 对其余值拒绝 Python bool | 保留；bool 不能借整数继承关系成为轴 ID |
| 6 | 对 NumPy 整数、自定义整数和 int 子类调用 index 协议 | 保留；恢复标准 Python int，非法类型仍报错 |
| 7 | 拒绝负整数 | 保留；非负整数是连接格式的前置约定 |
| 8 | 将每个轴组收集为新 tuple | 保留；不能让外部可变容器改变已有记录 |
| 9 | 将有序输入组收集为 tuple | 保留；记录不能持有未消费迭代器 |
| 10 | 按相同规则归一化输出轴 | 保留；输入归一化不证明输出合法 |
| 11 | 拒绝没有任何操作数的记录 | 保留；标量操作数用空轴组表示，仍须有操作数 |
| 12 | 为空输出构造集合并核对唯一性、输入包含关系 | 删除；空输出唯一且恒为输入轴集合的子集 |
| 13 | 为非空输出构造轴集合 | 保留一次；同一个集合用于唯一性和包含关系，删除第二次构造 |
| 14 | 核对非空输出轴没有重复 | 保留；不能将重复输出误作对角输入 |
| 15 | 归并输入轴并核对非空输出来自输入 | 保留；未出现的轴没有可确定的维度 |
| 16 | 将归一化输入和输出写入冻结记录 | 保留；后续规划和执行只消费整数关系 |

`ContractionTerm.subscripts` 构造后恒为 `AxisPattern`；直接传入该构造器的字符串仍要求显式
`->` 输出。Factor 的 `labels`、Action 的 `equation`、BaryonJoinSpec 的自旋/模态槽位以及
固定 MatrixKernelSpec 的 `equation` 也保存整数关系。已有动态/持久化入口中的外部字符串只在
C `parse_axis_pattern` 边界解析一次：支持显式输出和隐式输出（仅出现一次的 ASCII 字母，按
ASCII 顺序），标签限 ASCII 字母且不支持省略号。C 不保存数组，也不另建连接 IR；整数迁移
受影响 CPU 模块验证通过；GPU/生产规模仍未验收。

## 交接流程

既有中立 `contracts.memory` 唯一定义 `MemoryEstimate`、`MemoryLimits`、
`BudgetError`、`BudgetExceededError` 与 `TargetHardwareMemory`。
估算记录保留输入、快照、中间量、输出、组装及工作区字节字段和逻辑峰值；
上限记录区分声明/解析后的 host/device cap 与实际观察值。
这些记录是 P 计划估算与 [runtime.memory](../runtime_memory/runtime_memory.md) 执行预算消费的
中立数值契约；记录本身不估算计划或探测资源。`MemoryEstimate` 没有 host/device
分配域标签，也不会自动拆分、转换或合并两个域；使用方必须在每次核算和上限比较时
明确字节所属的域，需要并列记录时分别使用同一记录类型。

在当前 contraction 计划估算中，P 从输入 shape/dtype 计算每个不同输入的字节数；
启用输入快照核算时，`snapshot_bytes` 再单独计入保留的快照副本；
`intermediate_bytes` 按操作最后使用关系计算同时存活中间量的峰值，`output_bytes`
计入保留的各项输出，`assembly_extra_bytes` 由调用方明确提供，`workspace_bytes`
包含调用方声明的 reserve 和已知的打包/输出临时量。未知后端 scratch 不会从 reserve
自动推导或验证。

`logical_peak_bytes` 只是按调用方模型求和的逻辑量，不是 RSS，也不保证涵盖分配器、
原生库或未建模的后端临时空间；它本身不代表 host/device 可用上限。当前记录和消费接口
不构成覆盖所有 production 阶段与后端的完整估算器或严格完整内存上界。X 的
loader/source/plan 错误由 X.api 公开，不混入中立内存契约。原有
TargetHardwareMemory 校验及字段保持。

```mermaid
flowchart LR
    I["S/O/B 交付描述"] --> C["C 验证轴 / 类型 / 指标 / 身份字段"]
    C --> E["E 全场景网络与输出表达式"]
    E --> P["P 动作与存储计划"]
    P --> X["X 执行项值"]
    X --> R["R 输出恢复"]
```

## 完整张量目录补充

唯一 `TensorSpec(shape, dtype, *, description=None)` 现在可携带 `TensorDescription`；
`ExpandedCorrelator.tensor_specs` 直接公开这一目录，不另外维护一套同名张量记录。

```python
TensorDescription(
    axes=(), data_identity=None, selection_identity=None, role="",
    source_bindings=(), transform=(), semantics={}, bindings={},
)
```

`axes` 是最终逻辑张量的具名轴，提供时必须与 TensorSpec rank 相同且名称唯一。
`role/transform/semantics` 描述相对端点、约定、参数依赖与操作定义等不随组态/坐标改变的含义。
`bindings` 记录实际组态、基底、绝对端点、点池引用和覆盖；未知值保留 None。
描述允许有限数值、字符串、tuple、只读 mapping；拒绝大数组、回调和 live reader。
它不取代 loader，不能通过描述触发读取。

`TensorSpec.axes/data_identity/selection_identity` 是公开查询属性。
为兼容既有数值 ABI，TensorSpec equality 仍只比较 shape/dtype；
科学身份、复用和档案兼容必须显式核对 description/source，不能把 ABI equality 当作数据相同。
中间状态与档案保持同一类；档案白名单已纳入 TensorDescription。

现有 `BaryonJoinSpec` 和 `Action` 增加 keyword-only 字段 `sparse_spin=None`。
非 None 时为不可变 `tuple[(a,b,c,complex), ...]`，描述 B 已完成基底转换与源 adjoint 的精确三腿系数，
不含外层 Operator 系数，不传递 O 私有对象。索引限定整数0..3，严格按三索引字典序排列且唯一；
值必须有限且精确非零，`1e-200j` 等小系数仍保留。记录构造时冻结外部 list，非法结构立即拒绝。
`None` 表示只有原 dense fallback，空 tuple 表示明确全零；两者不可混同。
有此结构的 `BaryonJoinSpec.spin` 必须由 `(4,4,4)` TensorSpec 描述；现有无结构与任意 spin extent 的 ABI 保持。
该字段随 problem/action 的计划与归档保留，basis/source/field_order 的科学语义仍位于现有 TensorDescription。

协议补充：`StateDescriptor` 公开 problem/actions/catalog/results/seconds/done/peak 字段；
`CostModel` 公开物理读取费率及 load/preparation/contract 方法；
`DiagramBindingProvider` 只有 describe，数值能力由 SourceLoader 及可选 batch/measurement/IO 协议注入。
旧 DiagramData.execute_direct 空方法已移除，数值执行统一归 X。

本模块独立测试：`tests/correlator/contracts/test_tensor_description.py`，
检验只读交付、禁止 live 数据、轴 rank/唯一性，以及 ABI equality 不证明科学身份；
`tests/correlator/contracts/test_tensor_axis_bindings.py` 验证实际轴绑定、坐标类型身份、顺序与extent检查。

## 约束与所有权

- 纯 IR 和来源记录不含真实数组；shape/dtype 是描述。
- inputs/expanded 的 legacy loader/adapter 字段只承载执行能力引用，不是可序列化计划数据。P只读其描述字段，A不保存回调。
- tensor_id/term_id/coordinate_key 是范围内引用；物理共享还依赖实际基底、端点与数据身份，不能仅凭同名或shape。
- ContractionProblem要求网络物理系数归OutputRecipe，Term coefficient为1；自旋张量元素作为真实输入使用。
- 记录中 MappingProxyType/tuple 是只读，不允许规划器就地改变上游登记。
- 兼容alias必须指向同一类，不复制IR；计划档案白名单需随迁移核对，旧结构签名应保持。

## 文件、类与字段规格

### `indices.py`

`AxisPattern(inputs, output)` 保存有序输入轴元组和输出轴元组。`parse_axis_pattern` 是外部
方程的公共边界解析器；它接受显式输出或按 ASCII 规则推导隐式输出，并返回 `AxisPattern`。
`axis_ids(values) -> tuple[int, ...]` 在外部边界接受 ASCII 字母并归一为整数 ID 元组；整数
输入也会归一为元组。

### `equations.py`

#### `ParsedTerm`

```python
source: ContractionTerm
equation: AxisPattern
input_labels: tuple[tuple[int, ...], ...]
output_labels: tuple[int, ...]
input_keys: tuple[Hashable, ...]
output_shape: tuple[int, ...]
label_dimensions: Mapping[int, int]
```

- `parse_term(term: ContractionTerm, specs: Mapping[Hashable, TensorSpec], term_index: int)`

`ContractionTerm.subscripts` 接受显式输出的 `str` 或 `AxisPattern`；构造时解析字符串一次并
立即保存为 `AxisPattern`。动态/持久化入口可先通过 `parse_axis_pattern` 解析隐式输出，再将
Pattern 传给 Term。`parse_term` 只验证整数关系及维度；`ParsedTerm.equation`、轴字段和维度键均为
整数关系。`source` 引用整数形式的 `ContractionTerm`，保留输入身份与系数，不保留外部字符串。

### `expanded.py`

#### `ExpandedCorrelator`

```python
correlator: object
inputs: DiagramInputs | None
output_recipe: OutputRecipe
bindings: DiagramBindings
coordinates: CoordinateSpec
batch: BatchSpec
adapter: object = field(compare=False, repr=False, default=None)
identity_token: object = field(compare=False, repr=False, default_factory=object)
```

- `problem(self)`
- `tensor_specs(self)`
- `terms(self)`
- `output_structure(self)`

### `inputs.py`

#### `DiagramBindings`

```python
propagator_map: object = None
vertex_map: object = None
configuration: object = None
usedNe: object = None
usedNp: object = None
groups_independent: bool | None = None
```

- `__post_init__(self)`

#### `TensorAxisMetadata`

```python
axes: tuple[str, ...]
shape: tuple[int, ...]
dtype: object
data_identity: str | None
selection_identity: str
sparse_layout: str = 'dense'
point_shape: tuple[int, int] | None = None
nnz_upper: int | None = None
point_pool_ids: Mapping[str, str] = field(default_factory=dict)
point_pool_ids_by_batch: tuple[Mapping[str, str], ...] | None = None
axis_bindings: Mapping[str, tuple[str, tuple[object, ...]]] = field(default_factory=dict)
```

- `__post_init__(self)`

Reader-owned axes/shape/dtype identity; no numerical tensor is materialized.
`axis_bindings` 的键必须是 metadata 中存在的具名轴；值为 `(nonempty_actual_data_id, ordered_coordinate_tuple)`，坐标数量必须匹配该轴extent。坐标标签保留原值和顺序，字段复制为只读mapping/tuple；规范比较沿用 DatasetSpec 的 JSON 坐标身份，所以 `true`、`1` 与 `1.0` 不混同。空mapping保留未绑定的旧 reader/Array 用法。

公共 `canonical_coordinate_identity(value)` 返回一个已知 JSON 标量标签的规范字符串；接受字符串、bool、int 和有限 float，NumPy 标量先转 Python 标量，不接受 None、complex 或数组。公共 `canonical_axis_binding_identity(axis_bindings)` 返回按轴名排序的 `(axis, data_id, canonical_label_tuple)` 元组；保留每个轴内的坐标顺序。两者复用 `contracts.dataset.canonical_json`，不另建序列化或摘要规则；非法标签类型报 TypeError，非有限数值/空实际 ID 报 ValueError。extent 与轴存在性由 `TensorAxisMetadata` 校验，identity helper 不重新解释 shape。

#### `CoordinatePoint`

```python
key: str
values: tuple[tuple[object, int], ...]
```

#### 符号时间键

`normalize_time_key(value, *, name="time key")` 接受非空字符串或非负整数，拒绝 bool、
浮点、空白字符串及负值。字符串与整数保留不同身份，名称没有 source/sink 等保留语义。
此标量验证用于 Tag/Flavor 等符号构造边界。`describe_coordinates` 的 `time_axes`/`time_pairs`
保留既有任意 hashable key 契约，不转换对象身份；坐标值仍须是非负整数。
扫描轴和坐标保持映射声明顺序；同一个键只定义一个变量。
`time_key_order(value)` 仅为符号图 canonicalization 提供不同类型间的确定顺序，不用于输出排序。

#### `CoordinateSpec`

```python
mode: str
axes: tuple[tuple[object, tuple[int, ...]], ...]
fixed_axes: tuple[object, ...]
scan_axes: tuple[object, ...]
points: tuple[CoordinatePoint, ...]
scan_shape: tuple[int, ...]
```

- `__post_init__(self)`

#### `BatchSpec`

```python
size: int | str = 'auto'
axis: object = None
```

- `__post_init__(self)`

#### `CoordinateBatch`

```python
batch_index: int
axis: object | None
outer_values: tuple[tuple[object, int], ...]
point_indices: tuple[int, ...]
points: tuple[CoordinatePoint, ...]
batch_size: int
```

- `__post_init__(self)`
- `actual_size(self)`
- `point_keys(self)`
- `is_tail(self)`

One ordered, unpadded chunk of scan coordinates.  ``batch_size`` is the resolved chunk size capped to this outer-coordinate group; ``actual_size`` records the real number of points, including a shorter tail. No padding is represented by this object. ``point_indices`` always refer to the original ``CoordinateSpec.points`` order, so grouping by one axis never changes result ordering.

#### `PhysicalSlabBinding`

```python
identity: Hashable
describe_slab: Callable[[tuple[int, ...]], TensorAxisMetadata] = field(compare=False, repr=False)
read_slab: Callable[[tuple[int, ...]], object] = field(compare=False, repr=False)
read_workspace_bytes: Callable[[TensorAxisMetadata], int] | None = field(default=None, compare=False, repr=False)
```

- `__post_init__(self)`

Runtime-only request for a physical slab, shared by logical tensors.

#### `BatchTensorBinding`

```python
logical_role: Hashable
physical_slabs: tuple[PhysicalSlabBinding, ...]
times: tuple[int, ...]
lane_materializers: tuple[Callable[[object], object], ...] = field(compare=False, repr=False)
batch_dependent: bool = True
topology_key: Hashable | None = None
workspace_host_bytes: int = 0
workspace_device_bytes: int = 0
```

- `__post_init__(self)`

Logical tensor materialization separated from its physical slab read.  One fixed-coordinate adapter returns one lane. A batch adapter combines compatible bindings into ordered ``physical_slabs``, ``times`` and ``lane_materializers`` tuples. Several logical bindings may reference the same PhysicalSlabBinding.identity and therefore share one physical read. Callbacks are runtime-only and are never part of a saved contraction plan.

- `group_coordinate_batches(coordinates, batch, *, resolved_size=None)`

- `describe_coordinates(*, time_axes=None, time_pairs=None, batch='auto')`

#### `TermContribution`

```python
term_id: str
coefficient: object = 1
```

- `__post_init__(self)`

#### `DiagramInputs`

```python
problem: object
blocks: object
preparations: object
loader: object
coordinates: CoordinateSpec
batch: BatchSpec
term_ids: tuple[str, ...]
diagram_terms: Mapping = field(default_factory=dict)
diagram_recipes: Mapping = field(default_factory=dict)
tensor_requests: Mapping = field(default_factory=dict)
assembly_host_bytes: int = 0
assembly_device_bytes: int = 0
coordinate_batch: CoordinateBatch | None = None
batch_bindings: Mapping[str, BatchTensorBinding] = field(default_factory=dict)
batch_topology: Hashable | None = None
batch_output_axes: tuple[int | None, ...] = ()
task_batches: tuple['DiagramInputs', ...] = field(default=(), repr=False, compare=False)
batch_candidates: Mapping[int, tuple['DiagramInputs', ...]] = field(default_factory=dict, repr=False, compare=False)
```

- `__post_init__(self)`

Unique tensor catalog, all Terms, physical requests, and explicit output map.

#### `DiagramData`

- `describe(self, correlator, *, bindings: DiagramBindings, coordinates: CoordinateSpec, batch: BatchSpec) -> DiagramInputs`

Structural adapter interface; describe must not call numeric getters.

### `ir.py`

#### `TensorSpec`

```python
shape: tuple[int, ...]
dtype: np.dtype
```

- `__post_init__(self) -> None`

Shape and dtype contract for one named input tensor.  ``dtype`` is normalized with :func:`numpy.dtype`, and ``shape`` is copied into an immutable tuple.  A scalar tensor therefore has ``shape == ()``.

#### `ContractionTerm`

```python
inputs: tuple[Hashable, ...]
subscripts: str | AxisPattern
coefficient: Any = 1
```

- `__post_init__(self) -> None`

One explicit Einstein-summation term. ``subscripts`` accepts a string equation with explicit output or an `AxisPattern`; construction normalizes the field to `AxisPattern`. A term's coefficient is intentionally kept outside operation nodes and is applied only while assembling the execution result.

#### `InputRef`

```python
key: Hashable
```

- `__post_init__(self) -> None`

Reference to an input binding in an operation.

#### `NodeRef`

```python
index: int
```

- `__post_init__(self) -> None`

Reference to an operation result in a compiled plan.

#### `Operation`

```python
index: int
inputs: tuple[Ref, ...]
input_subscripts: tuple[tuple[int, ...], ...]
output_subscript: tuple[int, ...]
output_shape: tuple[int, ...]
output_dtype: np.dtype
```

- `__post_init__(self) -> None`
- `equation: AxisPattern` (property)

A unary or binary explicit-index operation. ``input_subscripts`` and ``output_subscript`` are integer axis IDs; `equation` returns the same relation as an `AxisPattern`. The compiler guarantees that there are one or two dependencies; execution consequently never invokes an unrestricted global contraction.

#### `CompiledTerm`

```python
output: Ref
output_subscript: tuple[int, ...]
coefficient: Any
output_shape: tuple[int, ...]
output_dtype: np.dtype
```

- `__post_init__(self) -> None`

Compiled output reference and coefficient for one source term.

#### `ContractionPlan`

```python
specs: Mapping[Hashable, TensorSpec]
operations: tuple[Operation, ...]
terms: tuple[CompiledTerm, ...]
optimize: Any = 'greedy'
operation_use_counts: tuple[int, ...] = ()
```

- `__post_init__(self) -> None`
- `total_operations(self) -> int`
- `unique_operations(self) -> int`
- `shared_operations(self) -> int`
- `eliminated_operations(self) -> int`
- `summary(self) -> _MappingView`

Immutable shared contraction DAG produced by :func:`compile_plan`.

#### `ExecutionStats`

```python
executed_operations: int
total_operations: int
unique_operations: int
shared_operations: int
peak_live_bytes: int
peak_intermediate_bytes: int
input_bytes: int
output_bytes: int
scratch_bytes: int | None = None
```

- `peak_logical_live_bytes(self) -> int`
- `operation_count(self) -> int`
- `as_dict(self) -> dict[str, Any]`
- `__getitem__(self, key: str) -> Any`

Synchronous logical-memory and operation metrics.  ``peak_live_bytes`` includes plan-owned intermediate arrays and copied output arrays that are still logically live.  User-owned input binding memory and NumPy's allocator or temporary scratch buffers are excluded; ``input_bytes`` reports the former separately for reference.  No timing or RSS measurement is implied by these metrics.

#### `ExecutionResult`

```python
values: tuple[Any, ...]
stats: ExecutionStats
```

- `__post_init__(self) -> None`

Numerical term values and metrics from one plan execution.

### `outputs.py`

#### `OutputNode`

```python
kind: str
value: object = None
children: tuple['OutputNode', ...] = ()
keys: tuple = ()
shape: tuple[int, ...] = ()
```

#### `OutputRecipe`

```python
root: OutputNode
term_ids: tuple[str, ...]
coordinates: CoordinateSpec
```

### `plan.py`

#### `PlannedContraction`

```python
expanded: object
plan: object
search: object
prediction: dict
planning_seconds: float
execution_mode: str = 'planned'
memory_limit: int | None = None
host_memory_limit: int | None = None
plan_identity: int | None = None
profile: object = None
reserved_host_bytes: int = 0
reserved_device_bytes: int = 0
batch_paths: tuple['PlannedContraction', ...] = ()
batch_shape_keys: tuple[object, ...] = ()
```

### `protocols.py`

#### `DiagramBindingProvider`

- `describe(self, correlator, *, bindings: DiagramBindings, coordinates: CoordinateSpec, batch: BatchSpec) -> DiagramInputs`

Metadata-only provider for Diagram input and output descriptions.  ``describe`` may inspect registered reader metadata, but must not invoke numeric block getters, load data, or produce missing inputs. Numeric access stays in the execution capability carried by the returned request object.

#### `CostModel`

- `snapshot_seconds(self, nbytes: int) -> float`
- `preparation_seconds(self, flops, traffic_bytes, *, kernel_calls=0, dtype=None) -> float`
- `contract_seconds(self, flops: int, traffic_bytes: int, *, kind='contract', dtype=None, kernel_calls=1, sparse_records=1) -> float`
- `kernel_call_for_dtype(self, dtype=None) -> float`

CostModel 只覆盖快照、显式准备计算和数值缩并。文件读取、传输与 spill/restore/discard 保留在
执行及资源容量契约中，但不由成本模型估算时间。

#### `StateDescriptor`

```python
problem: ContractionProblem
actions: tuple[Action, ...]
seconds: float
done: bool
peak_bytes: int
peak_host_bytes: int
peak_device_bytes: int
```

Read-only state facts consumed by plan inspection and execution.

#### `SourceLoader`

- `read_block(self, key: str, block) -> object`
- `materialize(self, key: str, source, blocks: Mapping[str, object]) -> object`
- `close_block(self, key: str, block: object) -> None`
- `close_all(self) -> None`

Runtime-only numeric access injected by bindings, never archived.

#### `BatchLoader`

- `activate(self, batch_index: int) -> ContextManager[None]`

Optional task-scoped loader selection for a preplanned compute batch.

#### `MeasurementAwareLoader`

- `set_measurement(self, enabled: bool) -> None`

Optional measurement switch for adapters that expose per-read events.

#### `IOReportingLoader`

- `drain_io_events(self) -> Iterable[Mapping[str, object]]`

Optional adapter telemetry drained at a specific execution action.

### `scene.py`

`normalized_scene_relations(terms, joins=()) -> (normalized_terms, normalized_joins)` 按每个
Term 中轴的首次出现顺序重编号，并同步转换关联 Join 的自旋/模态槽位。它保留输入数据 ID、
系数和输出轴顺序，不修改外围输出配方，也不搜索或读取数组；用于档案与路径重绑时比较
相同连接关系。

#### `SourceSpec`

```python
region: str
region_bytes: int
staging_bytes: int = 0
transfer_bytes: int = 0
host_region_bytes: int = 0
snapshot_bytes: int | None = None
metadata: Any = None
read_bytes: int | None = None
```

- `__post_init__(self)`

A cfg-independent request and its shareable backing region.

#### `ResourceTransition`

一次声明式分配移动的 C 唯一记录。

```python
domain: str
source: str
target: str
nbytes: int
workspace_host_bytes: int = 0
workspace_device_bytes: int = 0
```

domain 取 tensor 或 block；source 取 host、device 或 disk；target 取上述层级之一或 absent。字节字段均为非负整数。该记录只描述元数据，不代表会执行数值 I/O。

#### `Action`

```python
kind: str
tensor: str
operands: tuple[str, ...] = ()
equation: AxisPattern | None = None
seconds: float = 0.0
reused: bool = False
used_np: int = 0
join_id: str = ''
left_axes: tuple[int, int, int] = (-3, -2, -1)
right_axes: tuple[int, int, int] = (0, 1, 2)
same_point: bool = False
transpose_bridge: bool = False
term_index: int = -1
positions: tuple[int, ...] = ()
kernel_path: tuple[tuple[int, ...], ...] = ()
used_np_right: int | None = None
sparse_spin: tuple[tuple[int, int, int, complex], ...] | None = None  # keyword-only
core: object = None  # keyword-only; TensorCoreSpec for a declared logical core
resource: ResourceTransition | None = None  # keyword-only
```

可选的 keyword-only resource 字段保留现有位置参数构造方式。当前可执行路径中它为 None；在执行支持落地前，非 None 的转移会被资源执行门禁拒绝。

原生 `Action` 的 operands 已是原生 tuple、equation 已是原生 `AxisPattern` 或 None，
且 resource、sparse_spin 均为 None 时，构造后不再做原值转换和赋回。它不跳过任何
尚待执行的规范化；列表、外部方程、可选记录及扩展类型继续完整原路径。
core 和其余字段仍原样交付，不新增缓存或前端选项。

| 步骤 | 处理 | 依据 |
|---|---|---|
| 1. 确定是原生 Action | 保留分支条件 | 子类的字段描述器可能有原有赋值行为，不能省略其钩子 |
| 2. 确定 operands 是原生 tuple | 保留分支条件 | 列表、迭代器和 tuple 子类仍可能需要转换 |
| 3. 确定 equation 是原生 AxisPattern 或 None | 保留分支条件 | 外部字符串仍须解析，空字符串仍转为 None |
| 4. 确定 resource 和 sparse_spin 均为 None | 保留分支条件 | 非空资源需核对类型，稀疏系数仍须转换，并核对顺序和有限性 |
| 5. 对已确定的原生 tuple 再调用 tuple | 删除此分支的调用 | 返回必然是同一个对象，没有数据变换 |
| 6. 将已确定的原生 AxisPattern 再交给解析器 | 删除此分支的调用 | 解析器必然直接交回原对象 |
| 7. 将 None 交给稀疏系数规范化 | 删除此分支的调用 | 结果必然仍为 None |
| 8. 将上述未变化值再次赋回字段 | 删除此分支的赋值 | 原生类没有扩展描述器，字段已经持有相同对象 |
| 9. 需要实际转换或有扩展类型时的原构造流程 | 保留 | 可变输入冻结、稀疏复系数、资源约束及原异常仍执行 |

稀疏 bridge 动作携带 COO 两个索引轴的实际 extent：used_np 是第一轴，used_np_right 是第二轴。旧方形动作省略第二个 extent 时按 used_np 解释。

#### `Factor`

```python
tensor: str
labels: tuple[int, ...]
```

#### `PendingTerm`

```python
factors: tuple[Factor, ...]
output: tuple[int, ...]
```

#### `ContractionProblem`

```python
specs: Mapping[str, TensorSpec]
terms: tuple[ContractionTerm, ...]
sources: Mapping[str, SourceSpec]
sparse_bridges: tuple['SparseBridgeSpec', ...] = ()
baryon_joins: tuple['BaryonJoinSpec', ...] = ()
backend: str = 'numpy'
single_read_regions: bool = False
factorized_vertices: bool = False  # keyword-only
```

`factorized_vertices` 声明 Term 已含整体顶点的行打包 C/Φ 因子，只影响 P 的引擎选择与首路线引导，
随问题写入档案（缺少该字段的旧档案不再读取），不改变执行；见[重子行打包因子网络](../expansion/baryon-factor-network.md)。

- `__post_init__(self)`

构造时逐项核对当前输入键与实际形状。没有 `baryon_joins` 时，同一个原生不可变
`AxisPattern` 对象及相同有序形状只建立一次维度核对结果；单次构造最多保留 128 个
此类证明，超额布局仍逐项检查。不同科学输入键和 Term 不合并，系数仍独立核对。
临时 `ParsedTerm` 检查后立即释放，不保存解析目录；新构造不沿用旧证明。
扩展 TensorSpec、Term、轴记录或输入键仍走原逐项路径；形状变化或缺失输入仍调用
原解析器并保留当前 Term/operand 编号。有 join 时保留原有完整解析目录供三腿尺寸核对。

| 步骤 | 处理 | 原因 |
|---|---|---|
| 1. 核对 specs、sources 和物理绑定 | 保留 | 布局证明不包含科学数据身份 |
| 2. 确定是否为原生不可变记录及字符串键 | 保留 | 扩展对象可能具有额外取值行为，沿原解析路径 |
| 3. 为当前 Term 读取实际输入的有序形状 | 保留 | 不能根据同一方程猜测当前数据尺寸 |
| 4. 输入缺失时报告当前项及 operand | 保留 | 回到原解析器，维持错误顺序和信息 |
| 5. 首次出现的轴对象和有序形状核对维度 | 保留 | 首次必须确定 rank、重复指标及跨输入尺寸一致 |
| 6. 已证明同一布局后重建维度字典 | 删除 | 原生不可变轴及相同有序尺寸不能给出不同结果 |
| 7. 保留实际 Term、输入键和系数处理 | 保留 | 不因布局复用而合并计算项或科学输入 |
| 8. 保存全部临时 ParsedTerm | 删除无消费者目录 | 无 baryon join 时只有本次核对需要它 |
| 9. 跨 Problem/档案加载缓存证明 | 不引入 | 证明只对本次实际 specs 与 terms 有效 |
| 10. 有 baryon join 的逐项解析和目录 | 保留 | 后续三腿尺寸核对读取每项实际解析结果 |

#### `SparseBridgeSpec`

```python
left: str
right: str
record: str
used_np: int
nnz_upper: int
left_axes: tuple[int, int, int] = (-3, -2, -1)
right_axes: tuple[int, int, int] = (0, 1, 2)
same_point: bool = False
scene: int | None = None
transpose_bridge: bool = False
part_count: int = 1
reverse_operands: bool = False
used_np_right: int | None = None
```

- `__post_init__(self)`

`used_np` 和 `used_np_right` 描述 materialized sparse record 的有序 COO 轴；缺少 `used_np_right` 的旧调用等价于方形 `(used_np, used_np)`。`point_shape` 返回归一后的二元 extent。`transpose_bridge` 把记录轴映射到当前 Einstein operand endpoints；E 在 source dagger 后记录已交换的实际 shape，不能按数组 shape 推断点身份。旧方形档案继续按缺省第二轴解码。

#### `BaryonJoinSpec`

```python
scene: int
name: str
spin: str
elemental: str
spin_labels: tuple[int, ...]
mode_labels: tuple[int, ...]
sparse_spin: tuple[tuple[int, int, int, complex], ...] | None = None  # keyword-only
core: object = None  # keyword-only; TensorCoreSpec for a declared logical core
```

- `__post_init__(self)`

### `sparse.py`

- `cycle_point_equivalence(equation: AxisPattern, first_bridge, second_bridge)`

- `cycle_point_equalities_covered(equation: AxisPattern, first_bridge, second_bridge)`

- `bridge_point_equivalence(equation: AxisPattern, left_axes, right_axes, *, same_point=False)`

- `bridge_point_equalities_covered(equation: AxisPattern, left_axes, right_axes, *, same_point=False)`

- `bridge_intermediate_labels(equation: AxisPattern, left_axes: tuple[int, int, int], right_axes: tuple[int, int, int], *, same_point: bool=False)`

- `cycle_external_labels(equation: AxisPattern, first_bridge: Any, second_bridge: Any)`

### `storage.py`

#### `StorageBlock`

```python
dataset: str
selection: str
read_bytes: int
host_bytes: int = 0
device_bytes: int = 0
transfer_bytes: int = 0
staging_host_bytes: int = 0
staging_device_bytes: int = 0
read_kind: str = 'dense'
```

- `__post_init__(self)`

One physical read, with separate byte, persistent-allocation, and temporary-allocation bounds. ``device_bytes`` is zero for CPU blocks; CPU storage uses ``host_bytes``. ``selection`` is an explicit adapter-owned, immutable selection identity. The block carries no predicted read time.

#### `SourcePreparation`

```python
blocks: tuple[str, ...]
workspace_host_bytes: int = 0
workspace_device_bytes: int = 0
flops: int = 0
traffic_bytes: int = 0
kernel_calls: int = 0
dtype: str = 'complex128'
```

- `__post_init__(self)`
- `seconds(self, profile: CostProfile) -> float`

Logical source dependencies; materialization itself performs no file IO. Dense results are snapshotted by replay. Sparse results may borrow from these blocks, so their blocks survive until the logical source is released. Workspace excludes the final owned snapshot; it MUST include temporary materialization results when they do not alias a raw block. Explicit preparation calculation is billed through CostProfile.

#### `StorageEvent`

```python
action_index: int
kind: str
block: str
```

#### `MemoryPoint`

```python
action_index: int
phase: str
raw_host_bytes: int
raw_device_bytes: int
logical_host_bytes: int
logical_device_bytes: int
workspace_host_bytes: int
workspace_device_bytes: int
reserved_host_bytes: int = 0
reserved_device_bytes: int = 0
retained_host_bytes: int = 0
retained_device_bytes: int = 0
temporary_disk_bytes: int = 0
```

- `host_bytes(self)`
- `device_bytes(self)`

temporary_disk_bytes 单独记录临时磁盘驻留字节数，不计入 host_bytes 或 device_bytes 属性。

#### `SceneStoragePlan`

```python
state: object
blocks: Mapping[str, StorageBlock]
preparations: Mapping[str, SourcePreparation]
events: tuple[StorageEvent, ...]
memory: tuple[MemoryPoint, ...]
preparation_seconds: float
preparation_costs: Mapping[int, float]
peak_host_bytes: int
peak_device_bytes: int
feasible: bool
memory_limit: int
host_memory_limit: int | None
reserved_host_bytes: int = 0
reserved_device_bytes: int = 0
cost_profile: CostProfile | None = None
retained_before: tuple[str, ...] = ()
retained_after: tuple[str, ...] = ()
retention_planned: bool = False
retained_releases: tuple[str, ...] = ()
peak_disk_bytes: int = 0
disk_limit_bytes: int | None = None
explicit_resources: bool = False
```

- `seconds(self)`
- `with_reserves(self, *, reserved_host_bytes=None, reserved_device_bytes=None)`

peak_disk_bytes 和 disk_limit_bytes 分别描述磁盘峰值及可选上限。`seconds` 汇总 state 的快照/数值计算模型成本与 preparation 计算成本；物理读取、传输和资源移动不计入预测秒数。`explicit_resources` 仍描述存储轨迹，不执行资源移动。

#### `ResourceAllocation`

```python
domain: str
key: str
host_bytes: int = 0
device_bytes: int = 0
disk_bytes: int = 0
```

domain 为 tensor 或 block，key 是非空字符串；每个身份只表示一个所有者分配。借用该分配的逻辑视图通过依赖关系表示，不重复计为拥有者。host_bytes、device_bytes 和 disk_bytes 均为非负整数。

#### `ResourceSnapshot`

```python
allocations: tuple[ResourceAllocation, ...] = ()
borrowed: tuple[tuple[str, tuple[str, ...]], ...] = ()
opened_blocks: frozenset[str] = frozenset()
```

该记录保存某一时点的拥有者分配、借用视图到 block 的依赖关系及已打开 block。构造时会冻结字段，并检查分配身份唯一、借用视图身份唯一、依赖指向存活的 block 所有者，且借用的 tensor 不会再拥有第二份分配。它不执行 I/O 或设备初始化。

这些类型分别在 contracts.correlator 的场景与存储契约中唯一定义，并由 contracts.correlator.api 重导出。contracts.correlator.api.validate_executable_resources(value) 是规划、回放和档案路径共用的公开门禁。普通回放默认只接受计算动作和元数据轨迹；显式提供临时数组 store、正磁盘上限且满足完整单路径 NumPy CPU 条件时，也接受拥有型 tensor 的 `spill_disk`/`restore_host` 和 host `discard`。其它资源动作会在读输入前拒绝；未知计算动作或存储事件会抛出 ValueError。成本模型不为这些资源移动预测耗时。

### `vertices.py`

#### `VertexFactorMetadata`

```python
term_index: int
operator_term: object
logical_sector: str
raw_sector: str
anchor_time: int
bar_time: int
field_time: int
bar_time_unwrapped: int
field_time_unwrapped: int
endpoint_offsets: tuple[tuple[int, int, int, int], tuple[int, int, int, int]]
dagger: bool
current_term: object
current_term_index: int | None
link_origin_time: int | None
tensor: TensorAxisMetadata
point_pool_ids: object
point_pool_metadata: object
usedNe: object
usedNp: object
coefficient_state: str = 'unweighted_tensor; OperatorTerm.coefficient belongs to OutputRecipe'
mode_indices: object = None
coordinate_slot_maps: tuple[tuple[int | None, ...], ...] = ()
endpoint_time_selection: TimeSelectionRecord | None = None
physical_offset_mapping: bool = False
```

- `__post_init__(self)`

Metadata for one unweighted Meson operator term.  This record contains identities, coordinates and tensor metadata only. It never owns an elemental array and ``describe_factors`` must not load one. The Operator coefficient is kept separately for the output recipe.

#### `BaryonFactorMetadata`

```python
owner_id: int
term_index: int
operator_term: object
time: int
tensor: TensorAxisMetadata
usedNe: tuple[int, int, int] | None
dagger: bool
field_order: tuple[int, int, int]
coefficient: complex
```

- `__post_init__(self)`

Header-only description of one unweighted, ordered three-mode vertex.

### `workload.py`

#### `ExecutionBudget`

```python
host_bytes: int | None = None
device_bytes: int | None = None
reserve_fraction: float = DEFAULT_RESERVE_FRACTION
min_reserve_bytes: int = DEFAULT_MIN_RESERVE_BYTES
workspace_bytes: int = 0
host_staging_bytes: int = 0
```

- `__post_init__(self) -> None`
- `device_reserve(self, total_bytes: int) -> int`
- `as_dict(self) -> dict[str, Any]`

Host/device logical-memory limits used during plan selection.  ``reserve_fraction`` and ``min_reserve_bytes`` describe the default device reserve policy: available device memory is reduced by ``max(total * reserve_fraction, min_reserve_bytes)`` when a runtime knows the device's total memory.  Explicit ``device_bytes`` and ``host_bytes`` remain hard upper bounds for execution budgets.

#### `WorkloadSpec`

```python
specs: Mapping[str, TensorSpec]
terms: tuple[ContractionTerm, ...]
input_sources: Mapping[str, Any] = field(default_factory=dict)
convention: str = DEFAULT_CONVENTION
output_groups: tuple[tuple[int, ...], ...] | None = None
metadata: Mapping[str, Any] = field(default_factory=dict)
```

- `__post_init__(self) -> None`
- `output_recipes(self) -> tuple[tuple[int, ...], ...]`
- `as_dict(self) -> dict[str, Any]`

Immutable, configuration-independent contraction workload description.

`metadata.output_recipes` 是 PlanBundle adapter 的有序 term-index 分组，和下文 R 的 `OutputRecipe` 表达式树不是同一格式。构造时 C 检查其索引类型与范围；`output_groups` 独立要求所有 term 恰好分组一次。

    validate_frontend_workload(workload) -> None

这是 C.api 的纯元数据校验入口。没有 adapter 标记的通用 workload 保持兼容；当前已知的 `meson-simple-v1` 必须同时匹配 convention、按 recipe 恰好覆盖每个 term 一次、只包含 scalar term 输出，并声明足够的 `assembly_extra_bytes`。未知 adapter 或不完整 recipe 在执行准备和数值 loader 调用前报错。它只检查 `WorkloadSpec`，不读输入数组。

#### `PlanBatch`

```python
plan: ContractionPlan
term_indices: tuple[int, ...]
memory_estimate: Mapping[str, Any] = field(default_factory=dict)
```

- `__post_init__(self) -> None`

One immutable compiled DAG and the source terms it evaluates.

#### `PlanBundle`

```python
workload: WorkloadSpec
batches: tuple[PlanBatch, ...]
budget: ExecutionBudget = field(default_factory=ExecutionBudget)
metadata: Mapping[str, Any] = field(default_factory=dict)
```

- `__post_init__(self) -> None`
- `signature(self) -> str`
- `performance_fingerprint(self) -> Mapping[str, Any] | None`

A reusable workload plus one or more already-compiled DAG batches.

Plan coefficients use the public `contracts.correlator.api.encode_scalar` and
`decode_scalar` codec. Archive code consumes this shared contract API; it does
not maintain a second private scalar-payload entry point.

## 校验、异常与验收

结构验证复用原实现：TypeError区分非法类型/非JSON来源，ValueError区分shape/标签/扇区/坐标/预算约定错误，KeyError用于网络引用不存在的张量。
这些校验只证明结构完整；数据实际科学身份兼容和覆盖由B/E检查，规划可行性由P检查，真实数值/分配错误由X检查。

当前 resource records 描述轨迹、容量和所有权，不给资源移动计价。`validate_executable_resources` 在 replay 调用 loader/getter 前拒绝不受支持的资源动作；单路径 NumPy CPU owned-array 的 spill-to-disk/restore-to-host 已可通过显式临时数组 store 执行并保存为纯资源描述。档案不保存 store/receipt；GPU 设备迁移和跨批保留状态仍不支持。旧档案字段白名单规则由 [A L2](../plan_archive/plan_archive.md) 说明。

有向电流 raw contract 的 v1 `cache_identity` 复用 `contracts.dataset.stable_digest`；当前字段都是严格校验的 ASCII JSON 值，因此已有身份摘要逐字节不变。`tests/test_directed_current_contract.py` 固定该 v1 摘要并验证数据/边界校验。

验证入口：`tests/test_correlator_module_boundaries.py` 检查契约类型identity、模块依赖与无环；既有IR/compiler、坐标、采样、存储和归档测试继续检验对象行为。

保存的前端元数据边界由 `tests/correlator/contracts/test_workload_frontend.py` 检查，包含通用 workload 兼容、已知 adapter 匹配、term 覆盖、scalar 输出和装配内存预留；`tests/test_contraction_session.py` 另验证存档 PlanBundle 在执行前拒绝未知 adapter。实现位于 `src/lqcdweave/contracts/correlator/workload.py::validate_frontend_workload`；旧登记表保留符号来源记录，但本次不声称逐行恢复了登记表所指的旧 checkout。
GPU 与生产规模须另行验证。

## 矩形介子模式块

`MatrixKernelSpec(kind, shapes, dtypes, partitions=None, row_block=1, mode_block=False)`
仅描述矩阵/圈布局。`mode_block=True` 表示已经选定完整的物理矩形块；不能同时指定
partitions。两个传播子可分别为 `[4,4,O,N]` 与 `[4,4,Q,P]`，两侧无权重顶点须为
`[component,4,4,N,Q]` 和 `[component,4,4,P,O]`。算符系数为 `[operator,component]`。
单圈顶点须对齐传播子的反向模式轴。输出是无 PC 轴的小矩阵或圈向量；PC 配对由
共同 OutputRecipe 完成。未选块模板继续要求一致的方形模式轴。
该描述不选择计算路径、不读取数组，也不扩展 exact/DP 的操作域。

`MatrixGroup(axes, wick_blocks, partitions, partition_mode="radial")` 是共享矩阵
目录中的不可变结果声明。`"radial"` 保留 PC pair 输出；`"memory"` 仅以给定
区间分解完整模式求和，输出只含原算符轴。未知模式及缺少 partitions 的 memory
声明报 ValueError；它不选择路径、分配数组或建立另一套执行记录。绑定后模式的
语义保存在既有 OutputRecipe，因此完整 cfg 档案无需增加新的执行类型。

## 原实现迁移对应

| 旧位置 | 唯一新位置 |
|---|---|
| contraction/ir.py | contracts/correlator/ir.py |
| contraction/diagram_inputs.py | contracts/correlator/inputs.py |
| compiler中的parse校验 | contracts/correlator/equations.py |
| scene_state纯记录 | contracts/correlator/scene.py |
| scene_storage纯记录 | contracts/correlator/storage.py |
| result_structure输出记录 | contracts/correlator/outputs.py |
| diagram_expansion展开记录 | contracts/correlator/expanded.py |
| path_planning计划交付记录 | contracts/correlator/plan.py |
| workload结构记录 | contracts/correlator/workload.py |
| sparse_join/bridge标签与等价关系 | contracts/correlator/sparse.py |

`contraction/ir.py` 与 `contraction/diagram_inputs.py` 的转发入口已删除；IR 与 DiagramInputs/DiagramBindings/坐标及轴记录由 `contracts.correlator.api`
公开，`TensorSpec/ContractionTerm/ContractionPlan` 仍是唯一的同一类对象。
现有 contraction 包根的公开类型导出保留，不复制记录。当前新写档案使用 v8 envelope 并保存
`AxisPattern`/整数元组；普通 `plan-bundle` reader 直接接受版本 5/6/7/8，版本 4 bundle 须显式离线迁移。
`storage-schedule`、`correlator-plan` 和 `complete-cfg-plan` 的直接 reader 仅接受 v8。
`migrate_plan_archive` 显式转换 v4/v5 `correlator-plan` 与 `storage-schedule`，以及 v5 `plan-bundle`；
迁移只转换表示，不升级成本或存储契约，不符合当前契约的档案须重新规划。旧 complete-cfg/matrix 格式无迁移。
常规加载不执行迁移；新档案拒绝字符串连接记录。


## 外部来源 knownness

`contracts.source` 定义显式外部导入来源 envelope 和读取后的来源 knownness。
调用方导入旧 NPY、分片 NPY 或裸二进制输入时，显式指定
`provenance_mode="external_import"`，才会按新契约校验来源字段；默认 `legacy` 保留
任意既有 provenance 的 schema-1 身份与 ID，即使 JSON 恰好长得像新 envelope。
外部来源的五个字段分别声明 known value 或 unknown reason；登记环境不进入科学 identity。
该声明保存显式来源信息，不做签名认证。新数据的默认持久格式见[数据保存说明](../storage/data_storage.md)。

已登记 `input_roles` 和嵌入 `input_views` 中的真实 dataset 引用可以被严格来源策略追溯；
任意字符串、路径、shape、seed 或 `dataset_id` 不会自动变成 dataset ID。schema-1
历史 provenance 始终按 unknown 读取。

## 动态三腿重子记录

`contracts.correlator.api` 唯一导出 frozen 的 `BaryonVertex(spin, elemental)`、`BaryonTerm(source, sink, propagators, source_slots=(0, 1, 2), coefficient=1)`、`BaryonLine(propagator, source_slot, sink_slot)` 和 `BaryonTopology(source_vertex, sink_vertex, lines)`。contraction 的既有公开名称直接重导出同一 C 类型；X 不再承载重子记录或编译便利入口。

Vertex/Term 的输入键须可哈希，三条传播子与 source-slot permutation 在记录构造时规范化；系数为数值标量。Line 使用正整数 propagator ID，槽位属于 0/1/2；Topology 的源汇顶点分别属于 0/1 且不同，三条边按 sink slot 排序，source slots 为双射。非法类型报 TypeError，非法数目或范围报 ValueError。记录仅描述结构，不保存数组、getter、设备或文件路径；邻接矩阵解析与图批次预检归 E。验收入口为 `tests/correlator/contracts/test_baryon_records.py`，数值与端到端验收分别归 X/收缩编排。

## Logical sparse cores

矩阵介子候选另以`MesonWholeCore`描述保留独立输出行的完整逻辑顶点，以`MesonWholeKernel`
描述closed connected/trace的实际数值依赖及固定私有布局。`ContractionProblem.meson_cores`
默认空、`ContractionTerm.kernel`默认None；旧档案使用原字段默认值。virtual core有完整
逻辑spec但没有虚构的SourceSpec/owner，外层Term不引用内部component/phase因子。
`whole-matrix-v1/v2/v3`由中立`VertexKernelDescription`区分局部算术版本；C不保存第二张操作图。
字段含义、准备/计算及验收范围见[介子整体顶点](../execution/meson-whole-vertex.md)。

Sparse bridge output axes may retain either endpoint point coordinate without
equating the two endpoints. `cycle_intermediate_labels(..., needed_labels=...)`
preserves integer point axes needed by residual factors as well as term outputs;
consumed spin/color axes remain closed. `cycle_requires_generic_join` selects
the existing bounded COO join when a cycle exports a point axis or has nonlocal
point equalities. Closed locally covered cycles keep the two-stage kernel.

`CoreComponent(inputs, coefficient=1, sparse_spin=None)` and `TensorCoreSpec(kind, axes, spec, components)` are neutral immutable records exported by C.api. B owns baryon core grouping and registration; E binds its leaf references and ordered propagator axes to the existing `BaryonJoinSpec.core`. `Action.core` carries a selected multi-component recipe. Both fields are keyword-only and default to None. A baryon core declares spin0/1/2 and laplace_vector0/1/2 axes with ordered spin/elemental parts; P2P metadata declares spin/point/color axes at each end and refers to the existing complete sparse record. They define logical tensors without allocating their dense arrays. The search uses virtual core factor IDs, and consuming a core updates the Term expression; no per-node join-completion set is retained. The final plan continues to use the existing Action IR. C validates rank, references and component shapes; P/M predict the actual sparse computation, and X performs numerical work. See [planning](../planning/planning.md#统一-anytime-优化器的模块设计), [binding](../binding/binding.md#registered-baryon-reader-boundary) and [archive](../plan_archive/plan_archive.md).


### 紧凑时间的既有准备记录扩展

`VertexFactorMetadata.coordinate_slot_maps` 为空或三张等长非空表，依次为输出角色槽→elemental anchor slot、bar endpoint slot、field endpoint slot。元素为非负整数或显式缺失 None；有表须附 endpoint_time_selection。physical_offset_mapping 为 bool：同时间分支只核对实际边的完整有序记录，时间偏移分支允许按真实坐标映射到边自己的已声明记录。不存在额外的执行 IR。

`SourceReadRequest` 保留原 `coordinate_arguments=(argument_index, role, offset, period)`；新增可选 `coordinate_tables=(argument_index, role, tuple[slot|None,...])`。同一参数不能同时出现在仿射与查表记录中；role 是非空字符串，参数号须在 arguments 内。at(values) 在已有槽域完成查表，越界报 ValueError，选中的缺失端点报 KeyError。默认全时间仿射路径不变。conventions 的 time_selection_identity 保存 reader 的完整有序映射（包括显式 sampling_selection）；运行前比较该值，不能仅比较 Nt。字段由现有计划编码器保存，旧档案缺少新可选字段时使用空表。
## 零夸克数据契约

胶球沿用已有 TensorAxisMetadata、ContractionTerm 和 SourceReadRequest；
固定标量块 `shape=axes=()`，单输入 AxisPattern 为 `((),) → ()`。
identity preparation 允许 conjugate，不允许空间 permutation 或自旋常量。
elemental 家族的三参数读取表示零腿观测量，四参数读取保留原夸克 sector。
读取约定可包含不可变 `glueball_definition` 文本，用于 Dataset 跨 cfg 的科学定义核对。
此扩展没有增加共享 IR 类型，见[胶球接入](../../architecture/glueball_architecture.md)。
