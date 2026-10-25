# E 图与采样展开：L2

最低计算精度转换在规划前同步更新逻辑TensorSpec和输入快照字节描述；
临时的输出装配估计视图也遵守同一约定，避免float32胶球带复系数升为complex64时
先构造出快照额度不足的中间问题。该过程只处理元数据，不新增读数或数值转换。

E 把符号 correlator、B 顶点 header、reader source metadata、坐标和显式 sampling pool 关系展开为有序 tensor catalog、全部数值 terms、物理输入请求和输出恢复 recipe。E 的工作止于描述，不读取数组、不选缩并路径、不执行 einsum。

孤立零夸克顶点必须绑定 Glueball；它不能绑定真实夸克边。原介子及重子 provider
共用 `expansion.glueball.glueball_term` 登记一个标量输入和单输入恒等 term，
不引入 provider 或 IR 类型。读取仍属 elemental 家族，参数只有 component/momentum/time，
没有伪造的夸克 sector。源汇独立时间角色保留到读请求，即使模板坐标暂时相同也不合并。
时间 slab 及恒等准备沿用原批处理协议；见[胶球接入](../../architecture/glueball_architecture.md)。

收缩编排在完整描述后调用 `expansion.precision.normalize_contraction_precision`：从全部候选/时间批输入及
原始请求 dtype 选择最低浮点分量精度，保留复数类型，并固定逻辑 TensorSpec、准备常量、
`TensorPreparation.compute_dtype` 与 `OutputRecipe.dtype`。原始 read request、存储块和数据身份不变，
准备临时转换的工作区加入原有上界；数值转换由 X 在读取所需块后执行。无新增前端参数。
不可描述的 opaque 转换及特殊 sparse/baryon 的混合输入明确拒绝，不能沿旧混合精度执行。

一次 E 描述是 reader 全局约定的快照：每个实际 reader 的完整 `dilution_identity`
只序列化一次，各 read request 共用其不可变文本，下一次描述重新捕获。构造期索引
不进入数值 loader，不跨调用缓存输入。比较驱动在同一驻留时间对和 R 内也共用
QR 前缀及其 D reader 视图，实际父空间、向量标签、时间池及每图权重保持不变。

同一实际 DilutionSource 的完整文档在本次登记中冻结一次，经 C 的已验证 metadata
证明交给各 TensorDescription；保留全部父标签与顺序，不用摘要替代记录。已冻结
子树的规范材料化键按对象身份复用，仍以完整字段决定相等；相等但独立的来源对象
分别核验，未知身份不合并。文档、键和证明索引仅用于构造期，loader 不持有它们。

S 显式按三夸克交换特征合并的重子图保留表示行上的 `quark_symmetry`。
`BaryonDiagramBinding` 在读取数值前检查交换腿的实际时间、模式尺寸、已声明基底与
有序标签及显式模态选择兼容；不兼容则拒绝，不把完整求和的重数直接用于不同有序块。
数值项和系数仍交给既有 C/P/X/R，声明及验证见[重子交换标签](../operators/baryon_quark_exchange.md)。

矩阵分量的 `quark_exchange` 也随结构化编译的原生顶点键保留，E 不重新推导交换关系。
除逐腿检查外，实际参与约简的同族分量必须具有一致的端点空间、时间、基底身份和
显式模态选择；这些检查只读 header，不读取 payload。矩阵运算、稳定子和独立列选择归 S。

<a id="全-diagram-整数连接格式目标待迁移"></a>

## 全 Diagram 整数连接格式

权威约定见[ARCHITECTURE](../../../ARCHITECTURE.md#diagram-的统一整数连接格式目标待迁移)。E 根据已有图连接和 B 提供的有序腿槽、
具名轴、数据与成员 ID 直接构造 C `AxisPattern`，不重新匹配连接，也不生成方程字符串。
批次轴采用所有现有轴 ID 的最大值加一。S/B 的原始连接和物理轴描述保留；前端不增加
编号或转换操作。此前 CPU 定向检查已完成；所有 GPU 核、DCU/MPI 与生产规模仍待验收。

## 完整任务档案的当前边界（阶段五已接通）

目标见[完整规划与直接执行](../../architecture/correlator-target-architecture.md#完整规划与直接执行)。完整 cfg 可保存并执行通用 Scene Action、已描述的 baryon join 和 sparse bridge，以及有可序列化 preparation recipe 的输入。缺少可保存配方的 opaque 输入仍不支持；coordinate-batched sparse GPU 也明确不支持。下文的 provider 限制是其各自展开范围，不代表共同完整 cfg 执行器只支持 dense NumPy。

完整 cfg 路线在规划阶段完成全坐标来源描述与共享输入登记，交付模板和实例绑定表；执行坐标直接使用已保存绑定，不逐点重建 `ExpandedCorrelator`。当前公开 `calculate`/`complete_calculation` 将支持的 NumPy/CuPy 路径和已描述的特殊操作交给 P 构造 cfg 计划，并保留已选 DenseKernelBatch。低层 `MatrixCatalogBinder.iter_bind` 仍是独立的惰性描述 API，不是档案执行器。

## 显式传播子读取上下文

某些已有数据的反向重建依赖当前主源行，单独的 `(sink,source)` 不能标识读取。
这类 reader 可声明不可变 `read_context_axes=("source",)`；每个角色必须存在于当前
坐标。E 将相应整数值依声明顺序追加到 `describe_block`/`block` 的
`(sink,source,sector,*context)` 参数，保存到原 `SourceReadRequest.arguments`，并用
现有 `coordinate_arguments` 参数化。可选 slab 方法同样接收追加的固定上下文；不提供
slab 时仍可从一次选路补全全部固定坐标并保存完整 cfg 档案。

该声明是读入条件，不是新夸克腿或 Flavor。reader 必须在 metadata 的实际 data/selection
identity 中区分不同重建来源；真实同点原始块可以保留同一实际身份。E 的 header 缓存、
P 的读取身份与生命周期及 X 的普通固定读入都保留上下文，不根据相同端点猜测共享。
声明随 conventions 保存，当前 reader 改变上下文角色时在数值读取前拒绝；已有 reader
没有该声明时仍使用原三参数接口，旧档案不增加或推断上下文。所有坐标仍一次完成
E/P/资源安排，回放不更换 reader 内部的全局“当前源”状态。

## 责任与调用边界

`expand_correlator` 识别 S.api 的结构化请求，分派到独立 `expansion/structured.py`。
它逐输出位置消费 Wick 模板，经 B.api 登记无权重介子成分；相同拓扑、实际
成分、时间和传播子标签的基础 Diagram 只登记一次，再由现有 native provider
生成 tensor/source catalog、ContractionTerm 和物理绑定。独立圈始终保留乘积，
重子仍使用现有核心及 join。PC 模式投影通过 `Meson.with_mode_indices` 描述完整
有序块，输出配方施加对角权重 1、非对角权重 `1/sqrt(2)`；不创建整张投影
SymPy 矩阵或 ModeBlockedDiagram。

返回对象仍为共同 `ExpandedCorrelator`，最终输出使用 R.api 组合的既有
`OutputRecipe`。同一次调用内，相同不可变结构化请求只组装一次输出子树；
驻留估算仍按每次出现计入占用，数值装配仍为各输出字段分配独立数组。
完整输出保留量计入每个候选 batch 的存储预算，批次选择仍走
原规划器。请求系数和源 Operator 共轭只计一次；不推断矩阵 Hermitian 对称性
或真空减除。结构化模式当前支持一或两个线性单强子插入，directed-current
明确拒绝，重子扫描仍受原 provider 的范围限制。CPU 小规模复数、I=0/I=1、
模式投影、扫描、原生重子核心及档案重载对照见 `tests/test_structured_correlator.py`；
这不构成完整 meson55 规模或 GPU 验收。

原生固定坐标展开识别 S.api 的 `ModeBlockedDiagram`，通过 B 的公开
`Meson.with_mode_indices` 缓存各顶点选择，并将相连 field/bar 的实际模式位置
传给 perambulator 的 `source_mode/sink_mode`；source dagger 使用 B 的原始槽映射。
分块末边界须等于共同显式 Ne；真实矩形 shape、选择身份与 actual basis binding
进入既有 tensor/source catalog。扫描 slab 的 describe/read 使用相同选择，
规划不调用数值 getter。无需新建优化器、执行器或存储系统。

固定坐标描述在一次调用内缓存顶点的成分/扇区选项及完整因子 header、传播子的
`reader + source/sink time + sector + selection` 请求、端点匹配和缩并网络模板。
网络模板按完整邻接结构、V/P 状态及 point-label 对齐复用；模式选择、时间或扇区
不同的输入不会混合。每次 `describe` 重新建立缓存，不跨 reader 重载或组态保存，
不缓存数值数组。共有张量仍按完整身份登记，输出系数和各 Term 顺序保持原样。
同一顶点因子只登记一次完整准备配方及批次读取绑定；传播子每次先检查两端绑定，
再按原 header 对象和完整逻辑身份复用登记结果。不同 header 即使声称相同身份也仍
进入原 catalog 一致性检查。缓存只存已有 tensor ID，不持有数值输入或跨调用状态。
`batch_topology` 引用已有不可变 `OutputNode` 配方并进行完整结构比较，不另造配方
字符串或以摘要替代语义；持久化继续使用共同契约的既有编码。

唯一实现位于 src/lqcdweave/contraction/expansion，公开入口为 expansion.api。E 依赖 S 的符号 Diagram、B 的公开 binding API、C 的纯记录/协议和 R 的公开 output recipe API；不导入 P 优化器、X contraction 数值核或 adapters 实现。Native provider 由收缩编排选择，external reader 仍由 adapters 提供。

E 的公开交接函数与默认值：

    contains_structured(value) -> bool

此纯结构查询复用 `expansion/structured.py` 的既有实现，检查 StructuredValue 及
嵌套 dict/list/tuple/object-array，不展开 Wick、不加载 reader、不查询数值；循环容器
报 ValueError。收缩编排在加载 preset 前先用它识别结构化请求，避免把该请求提前交给仅消费
普通符号图的收集器。结构化算符拆分仍由 E 完成，所需 reader 在拆分前通过收缩编排/B 绑定。

    expand_correlator(correlator, *, adapter, bindings=None,
                      time_axes=None, time_pairs=None, batch="auto")
        -> ExpandedCorrelator

    NativeDiagramBinding(**budgets)
    NativeDiagramBinding.describe(correlator, *, bindings, coordinates, batch)

    describe_coordinates(*, time_axes=None, time_pairs=None, batch="auto")
    group_coordinate_batches(coordinates, batch, *, resolved_size=None)

最后两项是 C 的唯一实现，由 E API 重新导出以方便 流程；CoordinateSpec、BatchSpec、DiagramInputs、ExpandedCorrelator 和 TensorDescription 仍由 C 唯一定义。

图内调用关系：

    S.api Diagram
      -> E.expand_correlator
      -> C.DiagramBindings / CoordinateSpec / BatchSpec
      -> NativeDiagramBinding.describe
      -> B.describe_factors + reader.describe_block / describe_slab
      -> C.TensorSpec / TensorDescription / ContractionTerm / DiagramInputs
      -> R.build_output_recipe
      -> C.ExpandedCorrelator
      -> P.path planning
      -> X.reader load / materialize / contraction

## expand_correlator 逐步行为

adapter 对含 Diagram 的 correlator 必须提供 C DiagramBindingProvider.describe。bindings=None 使用空 DiagramBindings；time_axes 和 time_pairs 的互斥、输入顺序、合法时间和 scan shape 交给 C 坐标 helper；默认 batch 为 auto。

1. 经 R.api 收集 correlator 里的唯一 Diagram 并保持首出现顺序。空 Diagram/常量/空容器走空 OutputRecipe，返回 inputs=None，不伪造 TensorSpec 或 ContractionTerm。
2. 对非空符号图调用 adapter.describe(correlator, *, bindings, coordinates, batch)。该阶段只能查 B/reader header 和构建小型记录/回调，不能调用 block、load、生产器或缩并核。
3. 要求返回 C DiagramInputs，且坐标与 batch 和请求一致。非空输入问题必须有 term；diagram_terms 和 diagram_recipes 必须恰好覆盖 unique Diagram 集合一次。
4. 经 R.api 按 Diagram、term_id 构造唯一 OutputRecipe。所有 Wick/Operator/sampling 系数在该表达式里恰好应用一次，数值 ContractionTerm 保持 coefficient=1。
5. 通过 R 输出 reserve 函数计算整个任务的 host/device assembly reserve，按 backend 和 task batches 写回描述。
6. 返回 C ExpandedCorrelator；E 不重新查找或改变 reader data identity。

参数类型错误、provider 返回值类型错误、漏图/多图/漏项、坐标或 batch 不符、非空 Diagram 展开为空 term 都会以 TypeError、ValueError 或 NotImplementedError 立即失败。

## C envelope 和执行期请求分离

ExpandedCorrelator 的 canonical 字段为 correlator、inputs、output_recipe、bindings、coordinates、batch、可选运行期 adapter capability 和 identity_token。problem、tensor_specs、terms、output_structure 是 envelope 上的只读访问属性。

DiagramInputs 包含的域描述有 problem、physical blocks、source preparations、coordinates/batch、term_ids、Diagram→terms/recipes 映射、logical tensor requests、assembly reserves、coordinate batch、batch bindings、batch topology、batch output axes、task_batches 和候选 batch widths。

### 可保存的输入读取与准备描述（阶段一）

`DiagramInputs.read_requests` 是物理 block ID 到 C `SourceReadRequest` 的只读映射。记录包含
`reader_role`、family（当前为 propagator 或 elemental）、已解析的 reader arguments/selection、
`TensorAxisMetadata`、可选的 `prefetch_host_bytes`、别名请求和约定字段。普通 dense
propagator/vertex 及矩阵 binder 已登记这些请求。初始登记不会检查可选后台 slab capability，
`prefetch_host_bytes` 为 `None`。

`SourcePreparation.recipe` 可选引用 C `TensorPreparation`；B 为 dense Meson 提供
`describe_factor_preparation`，记录 spin/profile 常量及 source 变换。E 只描述并登记 recipe、读取请求
及物理身份，不执行准备 recipe。X 的 `execution/preparation.py::PreparedSourceLoader` 消费该 recipe；
`with_backend(backend)` 返回共享物理 reader、使用独立常量缓存的新准备器，不改原 loader。稀疏输入
沿用其已有的准备及读取描述。

读取约定中的动量序列冻结为不可变 tuple，支持 reader 使用 list、tuple 或 NumPy 数组声明；
绑定时核对有序内容。源声明被原地修改不会改变保存的约定，也不会改变 cfg 读取去重的身份键。

E 的 `describe_input_reads(inputs) -> Mapping[str, SourceReadRequest]` 在完整计划交付阶段
查询 loader 是否支持后台读取及其 host 字节上界，并返回带冻结额度的请求表；不支持时额度
保持 `None`。这使读取来源、选择、兼容性 header 与可选预读额度能用纯数据表达；完整 cfg
交付会在此基础上冻结额度，并由保存的安排接入跨坐标/cfg 预读。

`restore_input_loader(requests, readers, preparations, specs, *, configuration=None,
constants=None, reader_aliases=(), debug=False, convention_cache=None)` 按保存的 reader role 接入已打开 reader，并返回执行期 loader；它不展开
correlator、不构图、不读取数值。构造时先对全部请求执行兼容检查，再允许后续读取：核对 reader
约定、cfg、axes、shape、dtype、layout、有序 endpoint labels、端点共享关系、source sharing
及显式 alias readers 的实际共享关系。若 reader 明确声明 `momenta` 或 `momentum_list`，其有序
动量 tuple 会进入保存约定，绑定时须逐项相同；未声明的动量身份保持未知，不从 shape 或名称推断。
实际 cfg 的数据身份可变化，只要保存的共享/别名关系仍兼容。任何不兼容项均在数值读取前拒绝。
可选低层 `convention_cache` 由收缩编排在同次执行内持有，最多 128 项、只含弱 reader 引用与不可变声明；
相同 reader/cfg 的未变化稳定约定只核对一次。每次仍取当前声明，mutable list/ndarray 不因对象
身份相同而跳过；替换声明、cfg、时间选择或 reader 会重新检查。每个请求及 alias 的 header、
axes/shape/dtype/layout、有序选择与实际科学共享身份始终核验，debug 另作准备图深度审计。
完整 cfg payload 会保存纯读取表和隐藏 stack 成员描述；运行期 reader/闭包仍不入档。

阶段二为 `SourceReadRequest` 增加 `coordinate_arguments` 与 `at(values)`。每项坐标参数按
`(argument_index, role, offset, period)` 描述：调用者显式指定坐标角色、偏移和可选周期，
`at(values)` 用给定坐标生成该点的读取参数，并同步绑定别名请求。E 不从相同数值或参数位置
推断不同请求共享输入；完整 cfg 计划会检查各坐标实例的实际读取共享关系是否保持一致。该机制
只参数化读取描述，不改变数值读取或 loader 生命周期。

该读取参数化机制已供 P 的 `build_cfg_plan` 使用，并由收缩编排接入 `calculate`、完整档案加载、连续
预读队列和 DatasetWriter。可序列化的 preparation recipe 与支持的 sparse/baryon 描述可进入完整
cfg 计划；未描述的 opaque 准备无法归档，coordinate-batched sparse GPU 在执行前拒绝。原有 provider
对扫描、lane 批处理和特定拓扑的限制仍按下文各自的支持范围执行。

`DiagramInputs.loader`、`PhysicalSlabBinding.read_slab`、`BatchTensorBinding.lane_materializers`
和 `ExpandedCorrelator.adapter` 仍是运行期闭包/reader capability，不是 scientific identity 或
可归档数据。P 只读取 spec/terms/source/storage/batch 的纯描述，不调用 loader；A 保存完整 cfg
所需的纯输入请求和准备配方，但不保存这些闭包。X 才读取数值、物化 logical tensors、管理借用缓冲释放。

原生 dense-V `BaryonDiagramBinding` 将每个重子顶点交付为两个现有 `TensorSpec`：`baryon-spin` 是 `(4,4,4)` complex128 小 fallback，`baryon-elemental` 是独立 rank-3 mode 块。spin 的精确非零 tuple 从 B 的 `describe_spin_entries` 获取，存入其 `TensorDescription.semantics.entries` 和该场景的 `BaryonJoinSpec.sparse_spin`；basis/source/field_order 保存在 transform。它不是 elemental 产物，data/selection identity 保持 None，不借用文件或空间产物 ID 冒充自旋身份。spin 使用零物理依赖的 `SourcePreparation(blocks=())`，无文件读取，1024 字节逻辑快照；生成 fallback 的 host 临时为1024字节，GPU另计1024字节device临时。Loader 对零依赖源直接物化捕获的 entries，不重做 basis/source 转换。

空间源沿用 reader 原始物理 block identity、selection、mode axis bindings 与读取字节，只在执行期复制及 source 共轭。每个 Baryon 顶点用现有 `BaryonJoinSpec` 关联小自旋源、空间源和原始有序三腿标签；它们由 join 消费，不另作为普通 term factor。Meson/current 和传播子仍是原普通输入，Operator 系数仍由 `OutputRecipe` 应用一次。这样统一 Diagram 的规划与直接执行都能生成已有 `baryon_join` 动作，不先生成 `64 × Ne0 × Ne1 × Ne2` 的 rank-6 顶点。不扩展 P/V 或空间点稀疏支持。

纯重子请求在适用时改用行打包因子网络：每个顶点交付行堆叠的精确自旋常量 `C[r,s0,s1,s2]`（可精确低秩时拆为双夸克矩阵与行因子）与原 Φ，二者都是普通 Term 输入，物理行成为保留输出轴，R 以 `select` 取回；部分行请求精确拆成若干子满积；问题带 `factorized_vertices=True` 且无 join。适用条件、回退与验收见[重子行打包因子网络](baryon-factor-network.md)。上文 join 降阶仍用于混合介子/流、夸克交换、模态分块与只剩单行的请求。

PhysicalSlabBinding 表示一个可共享的实际 slab identity、描述/读取方法及读取 workspace。BatchTensorBinding 表示一个 logical tensor 如何从若干 physical slabs、实际时间和 materializer 生成。一个物理 slab 可供多个 logical tensor；逻辑变换和物理读取不能混成一个 identity。

## TensorDescription 与唯一 TensorSpec catalog

原生稠密展开在一次调用内按完整物化配方统一登记因子：已声明的 data/selection identity、
物理源块、shape/dtype、输出轴、gamma/profile/dagger 变换、配置/时间/模式绑定及准备工作区
全部参与比较。不同 Meson 对象或算符分项编号不阻止等价配方复用；输出系数仍由
OutputRecipe 分别保存。字典比较完整键，不依赖摘要碰撞或形状相同推断数值相等。
数据身份未知时保留原有逻辑出现身份。稀疏记录及 Baryon 核心登记保持既有范围。
实际 slab 身份使用声明的产物身份；身份未知时仅在同一 reader 的调用范围内复用。

每个 B/E provider 用 C 的同一个 TensorSpec(shape,dtype,*,description=TensorDescription|None) 创建最终逻辑输入。不会造平行 TensorCatalog 类型。每个唯一 tensor_id 的 description 现在包含：

- axes：输出给缩并网络的 logical axes，rank 必须与最终 TensorSpec.shape 相同。Meson axes 明确加 spin_field/spin_bar，并在 source dagger 下同步转换空间尾轴名；Baryon 分为三条 ordered spin_slot 的小系数源及保留三条 mode 轴的空间源。
- data_identity / selection_identity：来自 reader header。reader 返回未知 identity 时保持 None，不能用 id(reader)、文件名或 shape 冒充稳定科学身份。
- role：propagator、elemental、baryon-elemental 等输入职责。
- source_bindings：对应的 StorageBlock physical block key，可与其他逻辑 tensor 共享。
- transform / semantics：纯值记录空间轴变换、dagger、spin/gamma/current 配方、采样与相位规则。系数归属 OutputRecipe 明确记录；不包含 reader callback 或大数组。
- bindings：实际 cfg、source/sink/vertex 时间、相对端点偏移、pool ID/设计、Ne/Np selection、phase/boundary、basis/data identity 等当前绑定值和覆盖信息。
- axis_bindings：原始 reader 具名轴到 `(actual dataset ID, ordered selected coordinates)` 的绑定。E 将它保留在TensorDescription中；每条传播子边把 source 端对到 Meson 的逻辑 field、sink 端对到逻辑 bar，并依照所选 sector 的 raw named axes及Meson dagger映射坐标。两侧都声明时必须实际dataset ID与有序坐标逐项一致，描述阶段即拒绝不兼容输入，不读数组。两边都未声明或只有一边声明时关系状态保留为 `unknown`（分别标记both-unbound/one-unbound），不会称为科学身份已验证；这使未绑定旧 Array reader仍可走内存路径。
- 批次 catalog 保留 coordinate_batch 轴、坐标顺序、lane transform/bindings 和涉及的所有物理 source block IDs。

TensorDescription 的 data/selection identity 也保留在 TensorSpec 顶层查询属性中；shape/dtype equality 不等于语义身份相同。Legacy E 在共享 logical/physical tensor identity 时也比较axis_bindings。结构化缩并计划仍可跨数据复用；它只描述形状与运算拓扑，不认证实际端点身份。实际身份保存在description并由E在数值读取前比较。旧自定义 adapter 若没有足够 header metadata，description可以为None或对应字段保持未知；E不猜测补全。

V/V 顶点遵循既有数值空间槽位：非 dagger 原始槽 0 是 field、槽 1 是 bar；source dagger 交换两槽。即使 reader 将两轴命名为 sink_mode/source_mode，也不按其字面时间角色重新解释顶点 field/bar。混合 V/P/P/P 顶点按 logical sector 与 dagger 映射其具名 left/right 轴。

对于 `CurrentTerm.temporal_point_split` 或普通 `GaugePath` 的非零净时间位移，field/bar 仍是逻辑代数端点，不能直接当作物理源/汇时间：E 为 graph source/row 取 descriptor 的 physical bar_time，为 graph sink/column 取 physical field_time。P 的 pool ID 与 D 的物理时间 pool key 使用同一映射，compact 端点也通过对应物理时间表连接；源 dagger 已由 B 对换时间和原始轴。这一共同规则由 C 的 `uses_temporal_endpoint_order` 描述，不转置数值张量；纯空间项保持既有映射。小规模独立四标签参考见 `tests/test_temporal_path_endpoints.py`，完整 weak_field 证据见[慢测审计](../../../tests/reference/weak_field_audit.md)。

当前 SamplingPoolSpec 提供 group_id、spatial_sites、sampled_points、design、compensation 和 observed counts，B TensorAxisMetadata 提供 point-pool ID/per-batch ID。当前接口没有在这些记录中携带完整空间 coordinates 数组或可独立读取的稳定坐标 dataset URI；description.bindings 记录 pool registration reference，不伪造 point coordinates。Baryon/Legacy descriptions保留已知绝对时空端点和逐腿空间/时间 offset。

## 采样池与 exact coefficient

SamplingGroup 是 E 内部 frozen record：

    SamplingGroup(group_id, legs, spatial_sites, sampled_points,
                  compensation="expected", observed_counts=None,
                  sampling_design="simple_random_without_replacement")

legs 是显式 unique string label；组 ID 非空。全集/样本大小是非负整数，Np<=M；非空 legs 要求 M,Np 均正。当前只有固定大小简单无放回抽样。unknown/biased design 报 NotImplementedError。

SamplingScene 的字段为 point_labels、partitions、coefficient。point_labels 把每个 P leg 映射为带 pool 前缀的 symbolic label；partitions 保存各 pool 的 restricted-growth partition；系数为精确 Fraction。不同 partition blocks 是自由求和指标，不保证最后坐标不同，zeta coefficient 会修正所有更细的实际坐标重合情况。

    expand_sampling_scenes(groups, *, groups_independent=None,
                           local_weights=None, point_equalities=None)
        -> tuple[SamplingScene, ...]

每组默认列举所有 labelled partitions 后作 Cartesian product。该内部 builder 可再接收由 elemental 结构 metadata 生成的 `point_equalities`（pool ID 到原始 `legs` 位置对的映射）。只有 expected pool 中、同一实际点池及相同有序点选择下，被 `diagonal` 或 `point_diagonal` 结构证明相等的 P/P 点端才会先合并，再对商化后的 `r_eff` 个点端列举 Bell(`r_eff`) 个分区；输出 partitions 和 point labels 会映射回原始 legs。多个非空池必须 groups_independent=True；未声明报 ValueError，声明相关/联合抽样报 NotImplementedError。池间不建立等式，observed pool 忽略这些结构等式并保留原分区。空组、空输入和所有低模 V/V terms 返回系数为 1 的一个场景。不得由时间、offset、seed、Np、shape 或数值零推断池身份、组独立性或点端对角支持。

例如，一个已证明 P 对角或完全对角的同池 P/P current 将两条点腿从 Bell(2)=2 个场景商化为 Bell(1)=1 个；其未预计权系数是 `W1=M/n`，所以 `n=1` 也可计算。两个同池对角 P/P currents 则把四条点腿的 Bell(4)=15 个场景降为两条商类腿的 Bell(2)=2 个场景，`n=2` 足够。没有显式结构标签的 P/P 仍按全部原始腿分区；不同池不会因数值坐标重合而合并。混有一般项的 vertex 没有结构对角证明，因而不合并。

精确数值实现位于 `contracts/sampling_weights.py`，供生产和图展开共用；`expansion/sampling_coefficients.py` 保留兼容性 re-export：

- labelled_set_partitions(r) 枚举 RGS 分区；不同 labelled pattern 不折叠为整数分拆。
- blocks_of(partition) 返回首次出现顺序的成员位置。
- is_finer(fine, coarse) 是集合分割 refinement。
- expected_weight(M,Np,k)=C(M,k)/C(Np,k)=(M)_k/(Np)_k；k=0 的权重是 1。
- observed_weight(M,Np,k,count) 使用当前 selection 的有序 distinct k-tuple 数作分母；count=0、缺计数和 k>min(M,Np) 都是错误。
- scene_coefficients(r,M,Np,*,weights=None) 用 Fraction 解 zeta 逆变换 Z^T c=w。观测补偿按 observed count 构造 w，不等价于支持 correlated 或 nonuniform sampling。
- SamplingGroup.observed_counts 可按完整 RGS partition 或 block 数 k 查询；超过当前 Np 中可能的 (Np)_k 个有序 tuple 报错。

点池绑定辅助函数：

    describe_point_pool_metadata(factors_by_vertex, states)
        -> (point_pool_by_leg, pool_specs)

    expand_explicit_point_pools(point_pool_by_leg, pool_specs,
                                *, usedNp, groups_independent,
                                local_weights=None, point_equalities=None)
        -> tuple[SamplingScene, ...]

    logical_vertex_tail_shape(description) -> tuple[int, ...]

第一个函数只接受 B header 中显式且一致的 point_pool_id/SamplingPoolSpec，检查所有 raw point axis 的已选 extent；第二个按 pool identity 汇集 vN:field/bar P endpoints、检查 usedNp 不越登记覆盖，再调用 exact scene expansion。logical_vertex_tail_shape 使用 raw-sector 轴顺序，source dagger 时应用对应 transpose，并检查 logical rank/color extents。

## Elemental-local compensation and graph residual

`calculate(..., elemental_weighted=False)` retains the raw-valued estimator and legacy behavior for data without an elemental structure declaration. Proven P-diagonal support may still reduce the scene count with raw values because it is a structural identity, not a weight. When local weighting is enabled, E requires the point-bearing vertex data to declare matching structure metadata and reads the explicit pool IDs and `SamplingPoolSpec` already used by production. It does not infer a pool from coordinates, seed, shape, or a matching `Np`.

Let `M` be the number of sites, `n0` the sampled count used when a local elemental block was produced, and `n` the count selected for the current correlator view (`usedNp`, or the registered count when omitted). For each exact coincidence pattern, the desired graph factor is the expected inverse inclusion `frontend(M,n,k)=(M)_k/(n)_k`, where `k` is the number of distinct points. The vertex metadata describes the local factors already in the stored data, using `n0`; E solves for the residual `Fraction` scene coefficients that turn the sum over the expanded free labels into the desired global factor. This is the correction for a narrower `usedNp` view as well as for multiple currents. The stored sample count must remain the original production count.

If production and consumption use the same `n`, a single V/P or P/V current needs residual coefficient `1`. For one ordinary same-pool P/P current without P-diagonal support, the fully expanded fine scene has coefficient `1` and the coarse same-label scene has coefficient `0`: the diagonal and off-diagonal blocks already carry their respective local factors. A structurally P-diagonal or fully diagonal P/P current instead has one quotient point leg; its local factor is `W1=M/n`, and its same-`n` residual coefficient is `1`. Multiple currents sharing a pool still require the exact graph residual after quotienting; preweighted local factors do not remove their joint correction, and zero residual scenes may be skipped. Multiplying local factors directly would count overlaps incorrectly. Different explicit pools require the existing `groups_independent=True` declaration, and the target graph coefficient factors across those groups only under that declaration. Observed compensation keeps its existing expansion.

The spatial-point local-weight path accepts expected factors from fixed-size uniform sampling. `compensation="observed"`, nonuniform or correlated point designs remain unsupported. D uses an explicit mixed-column inclusion and group/pool equality contract under the same elemental_weighted option; see the next section. `elemental_weighted=False` keeps values raw; proven P-diagonal structure can still change scene expansion, while old or unmarked readers retain their prior expansion.

A complete `plan_archive` fixes the weighting rule, elemental structures, expanded graph and residual coefficients. Replay uses those saved declarations without requiring `elemental_weighted` again; it validates loaded support against the archived declaration before numerical reads. Start a new plan to change the choice.

### D-component inclusion under the same option

Ordinary Meson Diagram expansion preserves explicit V/D/P endpoint kinds and
actual basis bindings. A D leg requires DilutionSource, matching flattened
mode labels and actual propagator endpoints. Pools match actual parent basis,
declared pool UUID and ordered parent group; physical time does not split an
explicitly shared column-selection pool. Each time still binds its own actual
vectors and propagator endpoints. Default NV generation assigns different UUIDs
to independently mixed times; distinct groups and these distinct time pools
within one actual source use its algorithm's independence contract. Repeated
UUIDs across times declare a common mixed-column choice, so equality classes and
H_r span those times. Different pools from different actual D sources require
the existing groups_independent=True. Missing required provenance or contradictory full source
documents for one actual D ID are rejected. Source dagger preserves real endpoints.

For three complete mixed columns with two retained, a common subset across two
times has three exhaustive choices; independent subsets require all nine pairs.
Two local diagonal D/D currents leave two free column labels after quotienting,
so the shared rank-two case is estimable. This does not establish estimability
of four unconstrained D labels at rank two. UUIDs record the actual selection
relationship and must not be renamed to manufacture independence.

D labels identify actual orthogonal mixed columns in the complete rotated parent
basis. An assignment using r distinct columns from one group receives
H_r=(n)_r/(m)_r, with n parent columns and m saved mixed columns. In a local
same-pool D/D matrix the diagonal receives H1 and the off-diagonal H2; independent
groups/pools factor their own H1. These are inclusion coefficients on the mixed
D column axis, not declarations that D values are selected original Laplace modes.
The raw VQ vectors and propagators remain unscaled.

dilution_weights.py delegates to the existing SamplingGroup/expand_sampling_scenes
partition algebra. Different free summation labels may have equal values; exact
Fraction zeta inversion produces free/equality scenes rather than changing graph
connections. For two raw same-pool legs the unrestricted scene has coefficient H2
and the equal-label scene H1-H2. Known complete D/D diagonality first quotients
those labels. With elemental_weighted=True each RHS coefficient H_global_distinct
is divided by each already weighted vertex's H_local_distinct, then zeta-inverted.
This also governs P residuals under the same Boolean. Scalar division by one
first-moment factor per leg would lose the diagonal/off-diagonal distinction.

Estimability requires that the effective independent-label count after proven
quotienting fit the retained rank m. A proven fully diagonal D/D current with two equal labels may therefore
be supported with m=1; an unproven term requiring a distinct label absent from the
sample is rejected. There is no fixed five-order limit. Exact/full-column groups
produce one unrestricted scene with coefficient one at any order without Bell
partition enumeration. Legacy u1 allows first order and rejects shared-pool higher
orders. Complete group selections/reorderings remain supported; partial groups do
not have the current consumption contract. Matrix-specific and mixed baryon/meson
D consumers are not expanded.

Persistent D elemental conventions.dilution_weight_scheme records
component_inclusion-v1. Weighted data missing this internal marker, or declaring
an obsolete scheme, is rejected and must be regenerated from raw values. Bare
ArrayElemental weighted declarations refer to the current inclusion law, not old
uniform squared factors. Raw D with complete provenance can be newly planned;
archives from the previous estimator cannot be replayed because dilution_identity
includes the semantic version. This does not add a user-visible option. See the
[source and user contract](../../../.agents/skills/lqcdweave-elemental/references/dilution.md).

### Dense fully diagonal preparation

The three elemental structure labels are `general`, `diagonal` and `point_diagonal`. Here `diagonal` means equality of the complete V/D mode index or P point-plus-color index; `point_diagonal` keeps the P color matrix general while proving equal point indices. Missing labels, including legacy identity records that carry no structure metadata, mean `general`; no value scan or shape-based guess upgrades a reader. A selected view whose ordered left and right basis/point indices differ loses the declaration before numerical reads.

For ordinary dense Meson Diagram lowering, a verified fully diagonal V/V, D/D or P/P tensor can be compacted to its diagonal coefficients. The preparation still multiplies those coefficients by the requested Gamma matrix and contracts the merged mode or point-plus-color index, retaining every coefficient. P-diagonal support alone does not permit collapsing color. Matrix-specific kernels and mixed Baryon/Meson consumption keep their existing dense layouts. Sparse P/P uses the existing COO bridge and consumes structure metadata there; this does not claim new GPU sparse support.

## 图 lowering records 与 V/D/P 扇区

以下是 E 内部 network records；adapter 兼容路径可继续 re-export，流程 不要求用户创建：

    TensorRequest(kind, tensor_label, sector, subscripts,
                  vertex_index=None, source_vertex=None,
                  sink_vertex=None, endpoint_side=None)
    ContractionNetwork(requests, equation, vertex_indices, propagator_edges)
    StateSamplingVariant(vertex_states, scene)

    expand_state_sampling_variants(vertex_count, current_vertices,
                                   point_pool_by_leg, sampling_pools,
                                   *, groups_independent=None, usedNp=None)

    lower_state_graph(adjacency_matrix, vertex_states, *, point_labels=None)

`expand_state_sampling_variants` 为含 current 的顶点枚举 V/V、V/P、P/V、P/P；普通 Meson 默认仍固定 V/V。调用方可在 B 的 `Meson(..., requested_sectors=(...))` 显式选择普通项的逻辑 V/P 或 P/V 分解，包括登记 temporal point-path。E 先向 B 查询 reader 的 metadata-only `available_sectors()` 声明并验证所请求的 raw 能力，再枚举实际选择的逻辑 sector；source dagger 对换 raw V/P 与 P/V，但不改逻辑请求。`None` 仍保留未声明能力的旧 reader 默认行为。缺失扇区不会补零，也不会由类型或 shape 推断。全图 P endpoints 仍按显式同池关系构造 sampling scenes。`lower_state_graph` 验证 adjacency 方形、state 的左右为 v/d/p、propagator labels 为非负整数，并要求每个 meson-like活跃顶点恰一进一出。先 union fermion edges，再 union 同 sampling point class，因此不连通网络若共享一个 sampled point 会合并为联合 Einstein network。

每条 edge 的 spin/mode/point/color labels 由固定槽位表分配；所有 equation 都显式含 -> 输出指标。propagator tensor sectors 为九类 V/D/P 产品，vertex logical sectors 为九类 v2v/v2d/v2p/d2v/d2d/d2p/p2v/p2d/p2p；V/D 都使用 mode 轴但绑定不同实际基底。旧符号标签与轴方向由 adapter验证，E 不读数组反推轴。点标签缺失、非法 sector、label alphabet/slot 超限或不支持的多夸克 topology 用明确的 ValueError、IndexError、NotImplementedError 失败。

## Provider 实现与签名

### NativeDiagramBinding

NativeDiagramBinding(**budgets) 保存可传给 BatchDiagramBinding 的预算；describe 从已绑定 vertex mapping/callable 解析本次 unique Diagram 的 vertex rows。任何一个 vertex 是 Baryon 时走 BaryonDiagramBinding(sector_scope="dense-v")；否则走 BatchDiagramBinding(LegacyDiagramBinding(), **budgets)。只委托 provider，不读 block/不做 contract。

### LegacyDiagramBinding

    LegacyDiagramBinding.describe(correlator, *, bindings, coordinates, batch)

Legacy provider 使用已有 propagator_map、vertex_map、configuration、Ne/Np 和独立性声明。当前只处理固定时间坐标；coordinates.scan_shape 非空、显式 execution batch 参数或缺 map 时在描述阶段报错。每个 current项按各四类 V/P state、显式 pool 分区和 all graph shared labels展开；每个 network有唯一 ContractionTerm/term_id，operator和采样系数进入输出表达式。SourceSpec/StorageBlock保留物理source和logical view；支持 reader 明确声明的slab reuse。

dense tensor input调用reader.describe_block，不调用reader.block。Sparse P2P从header SparsePointRecordMetadata建立虚拟 SourceSpec/SparseBridgeSpec；当前桥接限CPU NumPy，每个 network至多两个 sparse P2P legs。source record metadata中保留 kind、point_shape、nnz、index/value dtype、bytes、backend、cfg、anchor/两端时间、pool IDs和coefficient owner。矩形 point_shape 的左右 extent 按 COO 索引顺序原样进入 C；source dagger 按实际逻辑方向交换两轴，E 不从相同 shape 推断端点身份。读取返回的数据由 reader adapter按自己的header contract校验，X 再按数值 kernel/bridge contract校验。全局 sampling 的 point-pool 选择仍由显式 bindings/producer 声明，矩形 extent 本身不构成池身份或随机设计证明。

LegacyDiagramBinding 和相关 graph labels/lowering 现在只由 E canonical implementation 提供。旧 `adapters/legacy_diagram_binding.py` 转发入口已删除；reader 格式与外部对象识别仍留在 adapters。

物理描述与登记已从 describe 的闭包提取到本模块内部的
`physical_inputs.PhysicalInputRegistry(bindings)`，普通图和矩阵绑定共同使用。它保存
TensorSpec/SourceSpec、StorageBlock、SourcePreparation、SourceReadRequest、端点核对缓存及
运行期 reader capability。E 的 `PhysicalSourceLoader` 连接描述的物理读取接口；recipe 执行由 X 的
`PreparedSourceLoader` 统一处理，矩阵常量和 stack 同样作为 preparation 描述交给 X。稀疏记录与 slab
后台读取沿用既有描述能力。Registry 不产生 Diagram、Term 或完整计划，不执行数值准备；图与输出构造仍由各 E
构造器负责，无新执行 IR。

### BaryonDiagramBinding

    BaryonDiagramBinding(*, sector_scope: str, coordinate_slabs=False, vertices="auto")
    BaryonDiagramBinding.describe(correlator, *, bindings, coordinates, batch)

scope 必须显式为 dense-v。当前支持固定坐标、dense complex VSV propagators、source/sink 三腿 Baryon VVV vertices，以及中间 Meson/Current V2V vertices。Baryon必须为graph source(3 out,0 in)或sink(0 out,3 in)，slot 0/1/2各出现一次；Meson/Current必须1-in/1-out。scalar/vector/matrix adjacency保留声明的per-leg source/sink slot。每一组合法 operator term组合/非连通network都在ContractionProblem和OutputRecipe中表示；输出系数没有进入spin×elemental网络。

直接调用此低层 `describe` 时只接受固定坐标和默认 batch 描述，不接受扫描。常规 `NativeDiagramBinding` 会在外层用 `BatchDiagramBinding` 组织坐标 slab 与合批，这一限制不能用于否定公开完整任务入口的坐标能力。P/V baryon legs、空间点稀疏 baryon joins、Baryon 全空间 Current 及非零 baryon endpoint_time_offsets 仍不由此低层 dense-v 入口支持。等时reader覆盖不能解释为更广全空间能力。非零三腿时间偏移在B describe阶段明确NotImplementedError。

`BaryonDiagramBinding` 由 `contraction.expansion.api` 公开导出，类实现仍归 E 的 `baryon_diagram.py`；`NativeDiagramBinding` 通过外层批次适配器使用该类；默认打开行因子网络，只有显式 `vertices="joined"` 使用封闭 join。旧 `adapters.baryon_diagram_binding` 转发入口已退役。

### BatchDiagramBinding

    BatchDiagramBinding(fixed_adapter, *, memory_limit=None,
                        host_memory_limit=None, reserved_host_bytes=0)
    BatchDiagramBinding.describe(correlator, *, bindings, coordinates, batch)

Batch provider 对每个 coordinate生成 fixed-lane description，校验各 lane的 term/equation/spec/source/topology兼容。只支持 NumPy dense meson terms；含 sparse bridges、baryon joins、CuPy或lane结构变化时早失败。单个固定 coordinate直接委托给fixed_adapter。

扫描时按命名batch axis分组，并产生不padding的CoordinateBatch。真尾批通过 actual_size和point_indices表示；不同source/outer轴分组不改变请求顺序。Auto宽度候选包括有限的分块宽度，不假定逐个source/sink scalar调用。宽度预测用物理source bytes、lane snapshots、reader read workspace、materialization workspace、全任务TermValue和output recipe reserve。未知预算时保留既有安全候选行为；显式/full batch与有足够预算的auto均可多sink向量化。

dense准备工作区按主机数组载荷估算。对随coordinate lane变化的logical role，loader先分配批次结果，逐lane物化后复制进结果并释放该lane临时值；因此声明的额外workspace为`max(BV, 收缩编排)`，其中B是本批actual_size、V是单lane最终TensorSpec字节数、W是单lane变换workspace。X随后仍按既有所有权规则复制最终快照。lane不变化时继续使用W。该批次暂存只调整存储描述；`kernel_calls`仍按原单lane变换workspace计价，所以借用raw view的传播子保持0，GammaPhi保持1。

原生GammaPhi dagger 的单lane workspace包括输出V和raw空间数据`phi.conj()`副本C；其后`transpose`只产生view，不另计载荷。C使用reader描述的raw shape和dtype计算。批次lane值在写入批次结果后立即释放，避免前一lane的变换临时值与下一lane变换重叠。该估算不包含Python对象、分配器缓存或BLAS内部scratch。

PhysicalSlabBinding的reader.describe_slab给出按实际lane times的物理描述，read_slab只能由执行期loader调用；若多个logical lanes共用identity，则同一个全任务block catalog按physical block key去重。可选的 `read_slab_in_background(times)` 回调仅用于 X 的 host read-ahead；E 将它作为运行期 capability 保存，不读数组或把它写入计划。原生 reader 还必须明确声明 `background_read_safe=True`，并提供 callable 的后台读取方法；未声明安全或没有该方法时保持同步读取。后台回调只能返回 host 数据，不得操作设备或触碰主线程并发使用的借用缓冲。BatchTensorBinding仍逐logical tensor保存各lane times和materializer。

内部 _BatchTaskLoader 状态：catalog、每个batch的materializations、active_batch、物理cache与last_use。activate(index)进入时核对batch已注册，退出时恢复原active index；read_block从关联物理slab读取/复用，materialize执行该logical lane的物化，close_block/close_all清理执行闭包拥有的资源。具备上述 slab 回调的 native loader 可通过 C 的 `BackgroundBlockLoader` 暴露五项 host read capability，X 在计划预算余量内使用单个 IO worker 预取；workspace 和 host 上界仍由 loader 声明。由于完整 task catalog 已绑定每个 slab 的 callback、实际 times 和 block key，后台读取不读取 active_batch；因此该 loader 可显式标记 `prefetch_across_batches=True`，在非 streaming 且 plans 已 join 时跨 coordinate batch 预取。Future 结果仍须等到对应批次 activate 后由主线程 finish，再执行该批 materialization。X 调用loader；E describe不读数组。旧 `adapters/batch_diagram_binding.py` 纯转发入口已退役；`BatchDiagramBinding` 仅由 E 内部 `batching.py` 实现并供 `NativeDiagramBinding` 组合，用户入口是 `expansion.api.NativeDiagramBinding`。

## VSVData reference provider 与坐标

adapters/vsv_diagram.py 的 VSVData 只作为明示CPU reference/source callback provider。Spec callback给shape/dtype后，VSVData按本类明确VSV/V2V契约添加具名spin/mode axes、dataset_identity与时间绑定；用户自定义spec若自带TensorDescription则原样保留。该reference只接受单个固定time map、不做scan/batching；不声明通用reader/格式或GPU能力。

E API re-export C coordinate helpers：

    describe_coordinates(*, time_axes=None, time_pairs=None, batch="auto")
    group_coordinate_batches(coordinates, batch, *, resolved_size=None)

coordinate objects保存实际有序requested values；batch object记录未padding的点索引与尾批真实长度。E不默认做时间稀疏选择，measure scan仍需由调用者/流程给明确time axes；项目默认全时间要求保持不变。

## 系数、输出与错误职责

ContractionTerm的系数固定为1。B的物理factor变换由TensorDescription.transform记录，Operator coefficient与spin network输入分开；SamplingScene.coefficient只在recipe乘一次。非连通network的乘法关系保持OutputNode树，组态响应的result container/coordinate order由R恢复，P只能读完整all-scene term目录。

Reader缺coverage/不兼容身份是B/E错误；bad graph/time/state/label、缺sampling pool/independence/observed counts、term mapping失配是明确ValueError/KeyError；当前provider尚不支持的拓扑/设备/sampling design报NotImplementedError。E不补零、不猜basis、不暗改估计量、不重试数据读取。

## 旧代码复用和范围记录

- diagram expansion envelope迁自原 contraction/diagram_expansion.py；旧转发入口已删除，调用方使用 expansion.api.expand_correlator。ExpandedCorrelator 唯一由 C 定义；output description只依赖 results.api。
- sampling exact Fraction/zeta inversion由生产与图展开共用的 `contracts/sampling_weights.py` 实现；`expansion/sampling_coefficients.py` 保留兼容性 re-export。系数与 `lattice/scene_coefficients.py::scene_coefficients` 对照；SamplingGroup 等 L2 内部记录仍唯一位于 E 的 sampling.py，模块测试直接消费该实现，不新增跨模块公共类型。
- pure V/P graph labels/lowering从旧 `adapters/legacy_diagram_binding.py` 迁入 E；算法来源可追到 `lattice/quark_diagram.py` 中 StateExpandedDiagram 的 axis/subscript lowering。旧 adapter facade 已删除。
- sparse/current H/D provider保留此前已有覆盖的CPU reader/COO bridge约定，不引入新kernel。
- Native/Batch/Baryon provider和batch slab拼接从原 adapters 对应模块迁到唯一E实现；旧转发路径已退役，reader格式与外部对象识别仍留在 adapters。
- expansion/tensor_description.py只组合C TensorDescription，不复制C类型。Operator项完整纯语义由B.api的operator_term_semantics描述；profile仅在B识别的已注入Operator项上生成有限descriptor/hash，不进C数组字段。
- adapters/vsv_diagram.py仍是外部reference source provider，不搬入E。

## 测试和验收范围

E独立CPU测试：
- tests/correlator/expansion/test_coordinate_batches.py：命名/外层coordinate分组、pairs输入顺序、固定点与真tail batch、无padding、非法轴/override；所有数据都用小型coordinates records。
- tests/correlator/expansion/test_sampling_scene_expansion.py：用有限总体穷举复/非对称函数作为独立oracle，覆盖r=2..4 expected精确weight、observed count、multi-pool independence、低模空组、冻结映射和0-inclusion错误；根级 `test_diagonal_sampling_scenes.py` 用精确 Fraction 检查点端商化、scene lift、`n=1/2` 可估计性及局部权重残余。
- tests/correlator/expansion/test_tensor_description_catalog.py：注入ArrayElemental/ArrayPropagator/ArrayBaryonElemental小型metadata readers并patch所有block getter使其一经调用就失败；断言meson/baryon/propagator每个logical spec都有与rank吻合的axes、真实cfg/time、block/source identity、spin-slot顺序与endpoint binding，并拒绝同shape下不同V/V basis身份或坐标顺序。

B/E贯通保留在根 tests：
- test_diagram_expansion.py：四种meson sector label/axis次序、source dagger/coefficient唯一归属、Current真实端点与header-only reads、52共享pool scenes。
- test_batch_diagram_binding.py：two-source multi-sink批次一致性、full-task slab共享、plan-only无IO、运行输出与独立scalar oracle。该CPU小样本joined host peaks为width1=6,800、width2=8,592、width4=13,200、width5=16,016 bytes；budget 7,500选择width1，40,000选择width5，6,000在任何reader read之前MemoryError。固定batch=2仍核对两汇向量化与唯一物理slab读取。
- test_baryon_diagram_frontend.py：三腿source slot排列/adjacency拓扑、source dagger和独立手写CPU oracle、Baryon+Current多顶点端点与限制。
- test_sparse_diagram_binding.py 与 test_sparse_cycle_partitions.py：COO/identity records、source dagger、single/double current、全部15个partition和密集穷举oracle。
- test_calculate_diagrams.py/test_correlator_frontend_pipeline.py/test_correlator_module_boundaries.py：W的四阶段贯通、plan-only/replay、公共依赖DAG与兼容类型身份。
- VSVData/reference readers在test_calculate_diagrams.py与test_correlator_frontend_pipeline.py覆盖；reader格式/生产/档案的归属测试留各自模块。

CPU 测试不能证明 GPU/QUDA/MPI、真实系综或生产规模性能。每个测试只支持其显式CPU样本、尺寸和reader范围，不扩大重子全空间current或generic format支持声明。

## Dense-V baryon endpoint identity metadata

The provider consumes B-owned `BaryonCoreMetadata` groups. It registers all leaf spin/elemental inputs and attaches the returned `TensorCoreSpec` to the existing `BaryonJoinSpec`; group coefficients belong to the core, while output assembly keeps its outer diagram coefficient. Different actual propagator IDs/axis bindings remain different Terms. Existing one-component unweighted joins retain their numerical operation domain. Multi-component cores support fixed/direct and anytime CPU replay; exact/DP explicitly reject this extended domain. A core describes the logical six-axis tensor and its sparse computation recipe, without materializing that tensor.

The existing Baryon provider compares VSV source_mode/sink_mode with the raw Baryon axis selected by field_order[edge slot], or the Meson dagger-dependent field/bar axis. Every propagator description has endpoint_axis_bindings source/sink records with neutral vertex/propagator pairs, axis names, vertex kind/index and Baryon slot. Both-bound matches are verified; known disagreement fails in describe before numerical reads. Missing declarations remain unknown for in-memory compatibility. Persistent planning requires verified actual IDs and ordered coordinates.

For mode-blocked Baryon diagrams, raw-slot selections flow through B metadata to actual VVV elemental and VSV propagator `mode_indices` reads. Source `field_order=(2,1,0)` maps quark lines to propagator fields; it does not reorder the three Baryon mode axes. The existing `BaryonDiagramBinding` and optimizer remain in use. This path is limited to equal-time dense VVV/VSV at a fixed time pair; scans are unsupported. See [Baryon mode-block contract](../operators/baryon_mode_blocks.md).

Logical propagator identity includes endpoint-check identity, retaining all consumed edges while physical blocks remain shared. Physical/logical reuse compares canonical_axis_binding_identity along with shape/dtype. The necessary local identity helpers follow legacy_diagram's endpoint/axis-binding logic; common Tensor metadata/IR stays in C. Native provider selection, equations, field_order and numerical materializers are unchanged. `correlator/expansion/test_baryon_diagram_endpoint_identity.py` covers raw source reversal, equal-shape wrong IDs/reordered labels, mixed Meson mapping, unknown compatibility and sharing conflicts; see [registered production/measurement](../elemental/baryon_production.md).


## Persisted scalar Meson description

`expansion.api.describe_meson_diagrams(diagrams, times, *, vertex_specs, propagator_specs, vertex_sources, propagator_sources) -> WorkloadSpec` builds the existing `meson-simple-v1` contract from diagram structure, scalar integer time labels, `TensorSpec` declarations, and caller-supplied JSON source metadata. The function has no getter argument and performs no numerical value reads. It preserves term order, one ordered recipe per diagram, one indivisible `output_groups` entry per non-empty recipe, and the existing 48-byte-per-recipe assembly reserve.

This persisted frontend currently supports scalar-output VSV diagrams. An omitted `propagator_types` field follows the simple-Diagram VSV convention; explicitly supplied type groups must align with the contraction groups and edges and contain only `VSV`. Non-scalar outputs raise `NotImplementedError`. Boolean/non-integer times and indices are rejected; source/sink/vertex positions must index the supplied `times` sequence. The signature has no lattice time extent, so it cannot verify that an integer label lies within a physical lattice.

`input_sources[*].source` records caller declarations. It is not evidence that a getter reads the named dataset. The old `contraction.structural_frontend` path is removed after the E/B API migration; callers import this function from `lqcdweave.contraction.expansion.api`. No equivalence claim is made for the unavailable legacy registry checkout or untested platforms.

## Dynamic simple and baryon diagram expansion

The dynamic getter-based paths have separate pure E services: `expand_simple_diagrams(diagrams, keys_by_occurrence, specs) -> (terms, output_recipes)`, `extract_baryon_topology(adjacency_matrix) -> BaryonTopology`, `validate_baryon_diagram_batch(diagrams, times, vertex_count, *, coefficients=None) -> (topologies, normalized_times, normalized_coefficients)`, and `expand_baryon_diagrams(topologies, term_input_keys, coefficients) -> tuple[BaryonTerm, ...]`. `expansion.api` exports these functions from `expansion/dynamic_diagrams.py`. They do not call numerical getters, inspect binding arrays, or select a contraction path.

`expand_simple_diagrams` preserves the existing dynamic simple string inputs at its adapter boundary and passes each equation through C `parse_axis_pattern` once; E does not synthesize an equation. Since occurrence keys are returned by B, the service order is B read/snapshot followed by E term construction and C parsing. A later B getter failure can surface before an earlier malformed equation reaches that boundary; callers retain this order. E creates one `ContractionTerm` with an `AxisPattern` per matched group and one recipe per diagram; R multiplies the term outputs in each recipe. Implicit output retains the current opt_einsum convention: labels that occur once become outputs in ASCII order. Labels must be single-character ASCII, and ellipses are unsupported. Dynamic simple diagrams retain array-valued outputs; the persisted `meson-simple-v1` scalar-output constraint does not apply. Group/subscript matching preserves the current adapter's zip traversal.

`extract_baryon_topology` purely parses a two-vertex adjacency matrix into the unique C `BaryonTopology` and `BaryonLine` records. A graph must contain exactly one directed three-line block with bijective source and sink slots. `validate_baryon_diagram_batch` checks the whole batch's two-time/two-vertex arity, scalar integer times, coefficient count and scalar types, diagram metadata, and topology before B reads values. Empty batches accept either empty or standard two-vertex inputs. The current dynamic baryon path supports VSV only; `sampling_times`, `temporal_extent`, `usedNp`, active `vertex_list`, non-empty `leg_offsets`, and non-VSV types fail explicitly. `expand_baryon_diagrams` combines B's source/sink/propagator keys with source-slot order into C `BaryonTerm` records and preserves caller coefficients. Input keys and array shapes do not prove physical dataset identity. The algorithms migrate the currently verified adapter behavior; they do not claim equivalence for unavailable legacy registry checkout behavior or untested platforms.

## 介子矩阵描述接口与优化别名

`expansion.api.expand_meson_matrix(...)`、`describe_meson_matrix(...)` 和
`bind_meson_matrix_catalog(...)` 保留为低层输入描述/绑定接口，不构成独立规划或执行路线。
收缩编排的 `optimize="meson_matrix"` 是统一 `anytime` 搜索入口的兼容别名；它使用相同的搜索预算、
评分、路径审计和存储审计，不固定 K/H 顺序或跳过候选搜索。推荐新调用使用默认的
`optimize="anytime"`。矩阵专用 K/H 数值核仍可作为低层 Action 实现复用，见 X 的执行说明。

E 负责把矩阵的成分、轴、selection 与 preparation recipe 描述成现有 C 记录；不执行
数值准备、不读取数组、不选择路径。扫描、存档和 reader 绑定仍沿用通用收缩编排/P/A/X 边界；
前端不要求新增矩阵 builder 或 runtime 参数。低层目录接口的适用条件见下文。

## 矩阵任务目录

`expansion.api.describe_meson_matrix(correlator, *, bindings,
representative_times=None) -> BoundMatrixCatalog` 位于 `matrix_catalog.py`。
`bindings` 必须是有 vertex_map 的 DiagramBindings；representative_times 可将
符号时间标签绑定到实际槽，用于一次 header 描述，默认按标签本身描述。

该服务直接遍历结构化容器和各算符轴，复用 B 的无权重成分身份，以整数 ID 和
只读的 int64 offsets/component_ids、complex128 coefficients 表登记线性映射。
Wick 按不同味道组合生成矩形子集与模板；分区只登记整数边界。每个请求仅生成
一个 matrix 输出引用，不生成各矩阵元素、模态投影或成分对的输出节点。

当前只描述 unselected dense V/V Meson，借用 reader 的 describe_block，不调用
block/read_slab；稀疏、重子、current 和已选模态视图明确拒绝。系数已包含 S 的
位置权重和 B 的 dagger 变换；模板保留独立 Wick 权重及单圈乘积。时间标签来自
S 的显式 flavor 时间；不从未包含时间的 HadronIrrepRow 打印名称推断时间。
`representative_times` 接受与符号定义相同的整数/非空字符串键，保持声明顺序，值仍是实际非负整数时间。

返回目录中的因子 header 属于当前 cfg/代表时间；component_bindings 只是临时绑定，不能单独保存为
执行档案，也不是已审计可执行计划。该低层 describe API 负责单次 header 描述；完整扫描由收缩编排/P
按统一 `anytime` 路线构造和审计 cfg 计划。`optimize="meson_matrix"` 仅是该搜索入口的兼容别名，
不会选择独立固定矩阵路线。低层核限制见[执行 L2](../execution/execution.md)。

## 共享矩阵目录的固定坐标绑定

E.api `bind_meson_matrix_catalog(described, *, bindings, coordinates=None, row_block=1)`
将 describe_meson_matrix 的 BoundMatrixCatalog 交给既有 ExpandedCorrelator / DiagramInputs。
bindings 必填，使用同一组显式 reader；coordinates 默认使用目录 representative_times 或
原符号时间，row_block 为正整数。支持固定坐标和仅一个点的扫描请求，保留请求的末尾
坐标轴；此低层 API 不扩展成多时间完整计划。不读取数值数组，不调用标量矩阵或模态组合展开器。

每种 Wick 网络只降低一次，登记输入角色；每个成分及有序模式选择在当前坐标只描述一次，直接使用
PhysicalInputRegistry 原有端点、轴、cfg、精度和物理块检查及读取/准备回调。不再构造
代表 Diagram、单侧替换 Diagram 或标量 Term。只接受固定 connected / 单圈公式；
各成分仍须通过两端核对，若改变传播子端点或选择则拒绝；非 NumPy 顶点在读取前拒绝。
单位相位自旋分离在一次绑定内按实际成分与有序模式选择判断一次；已登记的
输入 tensor key 直接复用其完整空间准备身份，避免为每个分量对重复构造、核验和
hash 同一个配方。首次登记仍比较完整 raw blocks、profile 常量、dagger/排列、
compute dtype 与空间形状；等宽块不能据此共享。索引只属于本次绑定，不跨坐标或
cfg 持有输入。数值项、输出节点、sum_barriers 和准备预算不变。

普通 anytime 的单位相位路线将不同实际空间准备保留为唯一 component 轴，
每个固定传播子端点/有序模式选择的 job 只生成一个 Term，不再对两端空间身份做
笛卡尔积。spin 与算符投影系数以小常量包保留；同一空间/自旋槽中的多个原始
贡献使用额外 contribution 轴，不在 E 求和或提前 cast。精确 ±1/±i 乘法保持原
单项系数的统一 dtype 边界，profile、dagger、物理选择和空间准备算术不变。
共享空间只准备一次。mode、spin、component 和 contribution 的求和关闭由
可存档 sum_barriers 耦合；每条 spin/component/contribution 轴都必须与剩余 mode 求和同时关闭，禁止提前窄 spin、Gram 或 component 投影。普通 mode 部分缩并仍保留原 c64 中间量边界；P 可以直接准入
同一个 FactorizedWideReduction，而不用先物化可能超预算的 binary 展开。
非单位相位、非 monomial 及空间标量继续使用原精确 fallback，不按通道名称或阈值丢项。
网络/轴位置/模式选择相同的计算跨组共享。无分区时完整矩阵或圈为一个 ContractionTerm；
有 q 个区间时，每个连通网络登记 q⁴ 个有序物理块 Term，每个单圈登记 q² 个。
算符与成分保持数组轴，Term 数不再乘算符行数或成分对数。
`Meson.with_mode_indices` 的区间和源端 dagger 槽反转传入共同登记表，传播子与顶点轴
是真实块宽；不产生 `[PC,Ne,Ne]` 掩码。DiagramInputs.matrix_kernels 按 Term 顺序保存
C 的 MatrixKernelSpec，物理块标为 mode_block=True、partitions=None。

S/C 请求的 `partition_mode="memory"` 复用这些真实物理块，以单位权重构造
网络的有序块求和，不构造 PC array/transpose 输出。独立圈先分别求和，再按
原 Wick 系数相乘；原矩阵形状、零区域及扫描轴保持。默认 radial 与 memory
可在同一嵌套请求中交付，相同区间与端点的基础 Term 共享，组装配方分别持有。
普通 structured 展开及紧凑 anytime 展开遵循相同语义。结果组装预留使用实际
配方重新估算，仍包含全部 TermValues，不宣称消除了 q⁴/q² 的规划与保留开销。
`tests/test_memory_partition_correlator.py` 对照独立全模态迹与未分区符号结果，
覆盖不等长块、I0/I1、复系数、扫描、混合输出、普通搜索及完整档案直接执行。

准备顶点栈时逐成分写入完整栈；原始块由共同 SourcePreparation 登记，不单独读取。
准备工作区计入完整临时栈、单个成分结果及该成分原工作区，快照仍由 X 持有。算符系数
为只读常量，无物理读取；PC 权重保存在输出配方。精度、实际成分来源及系数定义保留于 TensorDescription。
矩阵系数、目录数组、完整 TermValues 和 R 组装保留量纳入 assembly_host_bytes；Python
对象头/进程基础占用须另预留。这是数组估算，不能作为生产 RSS 上限。

R 的 array/transpose/reshape/scatter/zeros 配方处理 PC 组合、独立圈外积、混合味道矩形及零 Wick 块，系数仅在
系数表和 Wick 配方所属位置计入一次。纯零任务返回 inputs=None、空 Term IDs 和有形状
的 zeros 配方。数值结果保留算符轴、PC 轴和末尾扫描轴。没有自行补共轭或利用矩阵对称性。

CPU 专项已核对 charged/I0/I1、复系数、多段模态分区、混合味道零区域、单源端圈、
共享计算和一个时间对；完整扫描与档案通过独立 cfg 计划路线接入。真实 GPU、MPI 多 worker
和生产规模待验收。

## 通用 anytime 的紧凑矩阵展开

`expansion.api.try_expand_search_matrices(correlator, *, adapter, bindings,
time_axes=None, time_pairs=None, batch="auto")` 提供通用搜索可复用的紧凑矩阵描述。
收缩编排在 `optimize="anytime"` 或其兼容别名 `optimize="meson_matrix"` 时使用统一搜索路线；
前端无需增加参数或用户分组。
输入必须是原生绑定的固定坐标 structured NumPy dense V/V 介子矩阵。
不支持的图、绑定或扫描返回 `None`，由收缩编排保留原通用展开；无效坐标、布局和科学约定
仍正常报错，不能以回退掩盖不兼容输入。

它复用 `describe_meson_matrix` 与 `bind_meson_matrix_catalog`，保留成分栈、算符
系数和真实矩形分块，移除 `DiagramInputs.matrix_kernels` 固定核标记。
每个有序块的连通网络为普通方程
`BAON,DCQP,cADNQ,dCBPO,ic,jd->ij`；单圈为
`BAON,cABNO,ic->i`。每块的整个矩阵/圈登记一个 Term，独立圈乘积、PC/Wick 权重
及结果布局继续使用共同 OutputRecipe。E 不选择缩并顺序、不读取数值。

P 对这些普通方程使用通用 DFS、共享匹配和最终逻辑/物理审计；X 执行已选择的
普通 Action，不调用预制矩阵核。准备、只读常量及完整输出占用仍沿用原契约计量。
减少输出组合的 Python 对象不等于减少完整数值矩阵或成分栈的执行存储。
此因子化扩展了启发式搜索的表示，可能改变共享机会和路径成本；不声称与原标量
搜索域等价。`exact` / `dynamic_programming` 不使用该入口，既有证明范围保持。
CPU 独立标量参考覆盖复系数、charged/I0/I1、多成分、不等宽分区；普通搜索计划
可离线补全并保存为完整 cfg 档案，加载和执行无需再次展开或搜索。该 provider 的扫描自动紧凑化
需要在目标设备上单独验证。

## 可复用的矩阵构造绑定

E.api `MatrixCatalogBinder(described, *, bindings, row_block=1)` 在同一 cfg 的固定
reader 范围内复用成分匹配、网络输入角色及轴关系。算符系数矩阵在首次绑定
时生成，随后共享同一只读数组及定义摘要。模式区间按实际输入选择登记。`bind(*, coordinates=None)`
返回一个固定坐标的 ExpandedCorrelator；`bind_meson_matrix_catalog`
沿用该实现的一次性入口。

`iter_bind(coordinates: CoordinateSpec)` 按 `coordinates.points` 惰性地产生固定坐标
描述，并保留各点原有的 `point.key`。迭代器复用目录、角色绑定与只读常量，不缓存已返回
的 ExpandedCorrelator 或未来 loader；调用方拥有描述及其数值生命周期。

现有 Binder 的各时间对仍重新核对端点/基底/精度与实际准备需求，得到独立 loader 和身份 token。
Binder 不保留读入的源数据、K/H 或数值结果，先前返回的 loader 不被后续绑定改写。
单次 `bind` 内先检查已交付的 monomial job，再按实际 component 和有序 mode selection
复用精确准备配方的 spin 列支持（包括不支持时的 dense fallback）。不按 gamma 名称或
形状分类，也不跨本次 reader/坐标绑定保留这些判断。已登记的传播子边仅在 reader、
符号、sector、实际 source/sink factor 描述对象及时间角色全部相同时复用；这些 factor
已经包含当前 cfg、物理端点、实际基底/选择、source dagger 和 profile。首次登记仍沿
原服务完整核验两端，并保存全部读角色/别名；不同物理描述和新一次绑定仍独立核验。
这些局部索引只省去重复的元数据构造，不缓存数值、不增加档案状态或前端开关。
其内部匹配缓存是 E 的运行期构造状态，不是执行 IR，不进入计划档案；调用者不得改变
其 reader/算符绑定。跨 cfg 需要新绑定对象。`iter_bind` 只提供惰性描述，不执行矩阵、
不交付逐点结果。P `build_cfg_plan` 将已选路径实例化为全坐标 cfg 计划；它不改变这个 Binder 的单点绑定行为。
扫描、跨 cfg 连续预读和 DatasetWriter 输出沿统一完整 cfg 路线交接。所有 GPU 核和完整 production route 尚待验收，MPI
多 worker 和生产规模也仍待验收。

此前直接绑定 CPU 检查的历史记录不作为当前统一搜索/设备路径的集成验收；当前工作区集成测试仍在进行。
该历史直接绑定样例不是新路线的实机验收；未执行 401 cfg 或生产系综。

## 紧凑时间身份

线性算符行可以包含不同物理 mode selections，例如径向正、反序项的对称或反对称组合。矩阵目录按实际选择拆分系数轴，维持每个内核的传播子端点一致，再通过同一输出上的有序 scatter 加和恢复完整线性算符。相同宽度的不同模态不合并；不退回逐矩阵元展开。多个选择共享一次输出分配，不能为每项保留整矩阵临时量。`test_linear_rows_mixing_physical_modes_stay_compact` 核对不等宽径向项、源 dagger、断开项、独立通用收缩及禁止标量展开/提前读取。

E 的既有 SourceReadRequest.conventions 新增 time_selection_identity，保存 `(axis, physical_Lt, ordered_times)` 的不可变有序元组。完整计划重绑先检查这一记录，再读取数值；相同 Nt 不能复用不同选择的绑定。未标记的全时间 reader 保持 None，原有坐标/输入描述、计划格式及数值核不另建实现。

普通直接扫描计划的预测报告仅保存每批坐标数、backend、数值张量 shape/dtype 和项数，保持 strict JSON；包含 `AxisPattern`、输出配方和复系数的内部复用签名保留在 `batch_shape_keys`，不直接写入预测报告。计划本身通过既有类型编码保存这些结构，加载后可不重新搜索而回放。
