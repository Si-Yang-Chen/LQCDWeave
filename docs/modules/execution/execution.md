# X：数值执行与原子操作（L2）

整体矩阵介子候选在`execution/meson_core.py`绑定固定局部算术，保留原unit-spin、
FWR宽累加及dense回退；只缓存几何，不缓存当前数组。准备、常量、H2D凭据与owner沿现有服务，
不新增IO/上传队列或内层计划图。v1/v2档案的局部顺序分别保留，v3保留单空间块的原小Gram算法，实测采用状态见
[介子整体顶点](meson-whole-vertex.md)。
最终v3的CSNS完整任务ABBA与同步单点phase诊断分别记录；CuPy原私有池限额覆盖局部普通分配，
原FWR的有界partial预约继续扣除驻留/常量/输出余量。reserved高水位含缓存，不能直接当活跃峰值。

## 原计划操作游标与资源流水线（2026-10-22）

`execution/plan_cursor.py` 是X内部执行机制，不是新的前端或计划格式。
`plan_operations(operations, belongs=None, ready=None)` 只引用原计划/既有绑定视图；
不复制步骤、不生成依赖图、不选择设备、不读生产数组。无附加条件时直接返回原序列的
iterator，保留原固定回放的逐操作成本。`BoundProgram.run` 已通过 `operations()`
消费这个游标；已有读取、准备、核调用、释放及最终输出快照的顺序不变。

需要分资源推进时，`belongs` 仅筛选计划已经分配的资源，`ready` 只查询当前依赖与额度，
不可阻塞或在失败探测时预占资源。队头未就绪产出内部 `WAITING` 标志，原项不丢失、
后续同资源操作不越过它。`drive_plan_cursors` 逐资源提交已就绪项；全部停滞才调用现有
事件等待服务。它只保存资源到游标的少量引用，不保存另一份操作表。

提交回调负责原子准入/入队；游标吐出不等于完成。驱动返回只表示全部操作已提交，
已有执行owner负责最终等待、输出快照、异常收束及缓冲释放。线程future/device event
和数据owner仍属于本次执行服务。异常停止后关闭游标，不取消外部执行器，也不自动重试。

实际device固定回放和cfg连续队列已接三种资源的执行游标：

| 资源 | 原计划的既有视图 | 就绪与执行 |
|---|---|---|
| IO | ReadAhead按原read事件接受的有界ticket，ticket引用原StorageEvent | 原IO worker读取独立主机块 |
| CPU准备/上传提交 | 同一ticket及固定DeviceSourceLoader阶段 | IO完成后单CPU worker做原pinned打包与上传提交 |
| GPU计算提交 | BoundProgram原调用表及read依赖 | 主机准备完成后才产出read调用，消费时沿原H2D event设置stream依赖 |

IO和CPU生成器只读取已获预算的在途事件视图；等待各自条件时不阻塞其它资源。
计算游标通过ready/WAITING驱动等待既有ticket，不额外搜索依赖或复制全部步骤。
新增CPU worker填补原IO线程同时做打包造成的串行段，沿用同一预读额度；不另外预留一套缓存。
队头顺序、消费者输出、预算释放及device retirement仍按原计划/实际event执行。
CPU worker只承接已有主机打包和上传提交；Gamma×Phi等数值准备仍在原保存的后端执行，
不能由这个并发机制擅自把GPU准备换成CPU或改变数据布局。

CPU普通读者、单个可预取输入以及关闭io_overlap/测量模式保留原短路径，
不启用额外CPU worker与ready分派。所有reader已声明host_resident且计划read_bytes总和
不超过8 MiB时，也不增加CPU准备线程；GPU复用原inline上传方式，不再按4个block截断。
CPU保留原来至多4个block的短路径条件，其余仍用原IO worker。
实际文件输入和较大内存输入仍可分开推进IO与CPU准备，不新增前端开关。
下载/结果消费复用既有output_queue及D2H事件。
关闭先停止IO，再停止CPU接单并等待活动调用，尚未准备的主机块直接交还原owner；不为清理而上传。
游标、在途队列和完成状态不归档，无新前端参数。

`test_resource_pipeline.py` 用真实线程与事件握手验证IO/CPU并行、原event身份、readiness、
额度、错误与关闭；device/cfg协议测试覆盖已有完整回放。真实GPU验证和端到端性能另行记录，
不能把这些线程/模拟设备测试当作GPU验收。
游标和当前完成状态不写入档案。`tests/correlator/execution/test_plan_cursor.py` 另用
真实已有dense计划回放两份变化输入，确认引用原调用表、独立结果及禁止运行中重新绑定。

## 既有数值内核与回放策略

坐标批量的物理slab目录按dataset、selection、read_bytes和read_kind核对读取身份。
CPU/GPU放置改变host/device、传输及staging预算，不改变该身份；同步读取与后台预读
使用同一规则。

重子join现已按结构绑定、当前准备、计算及释放接入原回放和原格式存储档案，
实际边界与验证见[重子join准备与计算](baryon-join-preparation.md)。
新规划已交付中立 `VertexKernelDescription`：P 调用 M 粗计价并保存空的局部
`kernel_path`，X 根据 `ordered-join-v1` 绑定局部顺序；旧档案缺少声明时仍沿原路径。
跨 join 的提前打包和常驻共享仍是未采用的可选优化，不是整体顶点边界或生成器尚未接通。
两合并腿已接分量中间和，单合并腿已接[整体标量归约](baryon-scalar-sum.md)，
都消费当前数组、遵守原 P/C 和批宽。标量核的小 CPU 操作提前保留旧路径；
245186 的 216 项 NumPy/真实 V100 测试及 69 子测试覆盖正式分派，完整项目结果另见
[验收表](../correlator/correlator-integration-acceptance.md)，不由小核门禁推断完整 LL3 已通过。

`FactorizedWideReduction` 是 P 已冻结的三/四输入整数收缩 Action，而非 X 的路径优化。
构造时依原合同验证输入、轴、宽归约域和资源上界，并一次绑定不可变整数几何。
完整有序方程、shape、large operand 顺序、tile 与最终输出 dtype 为键，最多2048项
共享派生元数据；每个 core/Action 仍保留独立对象身份，JSON 字段和旧档案精度语义不变。
`dimensions` 返回新的字典，其余几何为不可变值；不保存数组、reader、配置或设备指针。
两个大输入先写入内部 complex128 Gram partial，小因子在第二阶段宽精度组合，
默认将 owned 输出转为逻辑 complex64。新规划的终端三输入 tiled core 则显式保存
`output_dtype=complex128`，令可能参与断开乘积或跨项相消的内部叶子保留宽精度；
发布字段仍由原输出表达式最终转换为 complex64。准备配方与已有空间 GEMM 的舍入保持不变。
C 保存全部输入轴、shape、两项大输入位置；P 独立验证实际求和约束、单消费者子图、
成本与完整 partial/输出分配 padding，重新审核真实输入的生命周期。没有虚拟展开
数组；输入 TensorSpec 仍统一 complex64，显式宽结果另有自己的 TensorSpec。档案回放直接执行同一 Action；布局缓存只存
内核及 stride 元数据，逐次读取当前输入，不持有数值 owner。NumPy 参考路径
使用显式计费的宽工作区；不支持的规划几何和旧档案继续使用普通收缩。
显式坐标批回放沿同一固定四输入图增加保留 lane，Gram partial 按 lane 分区，
静态小因子继续共享；其轴、成本、工作区和独立回放检查均由 P 同步提升。

新 component 轴路线由 P 在普通整数搜索中直接描述三/四因子宽求和候选，避免
要求巨大 binary 展开先通过内存准入。两个 large 只归约其共同 mode 标签；
剩余 Gram/component/spin 轴按保存的 `tile_elements` 有界分块。小因子独有的
contribution 轴在 finish 内以 complex128 分别求和，再与宽 Gram 组合。
各 tile 的输出留在内部 complex128 accumulator，全部 tile 完成后才 cast 一次。
`output_dtype` 缺省为 None 的旧 core 保持原 cast 语义；修复内部叶子精度需要重新生成计划，
不能靠新执行器默默改变旧档案。宽结果的值身份、坐标提升、输出字节和 allocator padding
均按保存的结果 dtype 核算。四输入 connected core 继续沿原 complex64 结果合同。
P 计入 partial、完整宽 accumulator、输出分配与 allocator padding、tile kernel
次数和 traffic；X 不重新选 tile。坐标 lane 分区独立，负 stride 按当前输入解析。
旧档案缺 `tile_elements` 时恢复 None，沿原四输入、4096 epilogue 上限与原 CUDA 源码。
新 tiled core 允许1..65536的显式宽度；P 已把该宽度保存进身份和预算。
X 仍仅按保存宽度运行，不能因为共享 reduction scratch 尚有空间就扩大它。
较宽 partial 沿原 stream-local scratch 借用/分配与异常释放；结果和跨 tile
complex128 accumulator 仍由当前调用独立拥有，不新增数值缓存。
新 tiled core 的 NumPy 路线在单 tile 容纳完整 Gram 时使用已计费的宽 einsum，
其余几何按同一已选 epilogue tile 计算有界宽 partial，避免 Python 枚举整个因子乘积。
NumPy 另计实际 reduction slices、宽小系数包、坐标向量和 tile 乘积工作区。
旧无 tile core 保留原标量 oracle；CUDA CuPy 为 GPU 实际执行路线，
HIP、完整生产规模和端到端收益须分别验收。

兼容参数 `calculate(..., factorized_gram_tiling=True)` 对应 tiled 输入复用，显式 False
对应 reduction；缺省 None 由下述单一 strategy 入口决定。
此执行选项随 Calculation 传到普通、完整 cfg 和 iterator 回放，不参与科学档案身份；
读档时可重新选择。每个 owner 的 factory scope 隔离配置，执行缓存和完整 core 工厂键
隔离开关值；当前数值和地址不进入配置。内部轴若可严格分成有序、互斥的 left/right 矩阵轴，且 tile 宽度至少
1024，X 可用一个 CTA 同时计算相邻 2×2 个部分和，复用两侧输入读取。保留输出 lane
独立分区；任意 byte strides、负步幅、非整行 tile 和尾块仍按原 partial 地址写入。
每个输出点保持原来的 256 线程、K 分区、complex128 乘积/累加及 block reduction 次序，
编译继续禁止 fmad；finish、跨 tile accumulator、blocks、tile 与全局工作区不变。
共同内部轴、交错矩阵轴、过大的 grid、较小 tile 和旧无 tile 档案保留通用核。
固定的 CTA shared memory 从128增至512字节，不新增设备 owner 或数值缓存。
首次编译可能更贵；阶段加速不代表完整公共任务或完整 cfg 已加速。

`factorized_gram_strategy='gemm'` 是不同的、显式存档的数值实现。P 在已经选定的
普通单坐标路径上独立重放该实现候选，重新计算结果身份、共享关系、释放与物理存储
准入；显式 reduction 路径和预算不变。首版只接受四输入、split blocks=1、无保留 lane、
归约长度256..1024、至少4096个 Gram 点及 tile 至少1024的严格矩阵轴；两个 small 的
内部轴须互斥且完整覆盖 Gram，输出轴也独立。另以输出点数至少128、逻辑复数乘积数
`M*N*K >= 2**23`、打包强度 `M*N*K / (16*(M+N)*K) >= 2` 筛选。后一个量的
单位是逻辑复数乘积/打包字节；这是已测 V100 范围内的保守启发式，不是普遍校准模型。
小输出实测不能摊薄提交成本，正域外保留已选 reduction/whole tile，仍可使用2×2复用。
成本更差或资源不足沿原 reduction，
合同错误继续抛出。路径含其它 core、特殊存储/驻留或显式 kernel batches 时暂不转换。
坐标 lift 新增保留 lane 时由 P 恢复 reduction 并重新计算身份和预算。

X 把当前两个大输入按已证明的 left/reduced、reduced/right 轴次序，在一次固定布局
pack/cast 中写入 complex128 arena 的互不重叠分区，再做一次无额外共轭的 wide GEMM。
固定二维几何和分区字节偏移复用既有 pointer GEMM
绑定，当前设备、stream、handle 与数学模式仍逐调用核对；不兼容的 CuPy 策略保留
普通 matmul，仍写入同一已计费 Gram 分区。普通 GEMM 调用仍默认分配自己的结果。
完整 Gram 交给原宽 finish 方程一次收尾；独立最终输出与整个 arena 同时存活。
P 计入各分区512字节对齐、arena/输出重叠的峰值及 NumPy 完整宽 oracle 的保守额度。
当前设备执行 scope 可借用已准入且同线程、设备、context、stream 的内部归约工作区，
容量不足或身份不符则普通分配同一 arena，不扩大预算。绑定不保留 arena 或数值，
该工作区不跨执行 scope 保存。三次数值调用为一次合并 pack/cast、一次 ZGEMM 和一次
finish；traffic 包含窄输入读取、宽包写入/读取、Gram 写入和全部收尾读写。内部成本
按 complex128 描述，模型仍是 provisional，不能代替同设备完整边界实测。
此实现可改变浮点累计顺序，输入/输出精度与 sum barriers 不变，数值身份显式带策略，
不能与 reduction 的 CSE owner 混用。vendor library/driver scratch 沿既有后端全局额度，
不冒充已计费的数组 owner；验收报告单列池范围。布局缓存仍有界，不保留数字或地址。

公共单一入口为 `factorized_gram_strategy=None|'reduction'|'tiled'|'gemm'`：None 对新 CUDA
任务默认尝试保守 GEMM 候选，对其它新任务为 reduction，对档案跟随保存的数值实现；
tiled 是原算序的2×2执行复用。旧
`factorized_gram_tiling=True/False` 分别映射 tiled/reduction，与冲突的枚举同时出现会报错。
旧档案缺 `gram_strategy` 恢复 reduction；需要 GEMM 时重新离线准备。载入档案不能
更换已计费的数学实现，已选 GEMM 沿保存的 core 执行，不依赖全局配置。
gemm 策略未选中的归约启用已验证的 tiled 执行复用，不兼容几何继续沿通用核。
当前组合选择顺序为 retile → conservative GEMM → exact_support；GEMM 不转换已有
support capability 的 core，exact_support 不转换已选 GEMM。两者不组合内核、不将单项
收益相乘；此优先级不是全域最优声明，重描述必须保留已经选定的数学身份与预算。

恒等 preparation 支持可选复共轭，供零夸克数据源使用；普通和共享绑定路径使用相同
准备规则。带共轭时不能直接借用未变换的上传块。单输入、输入轴等于输出轴的动作
直接交付输入并沿原所有权规则复制，不调用 einsum/GEMM。胶球与夸克 term 的乘法
由原 OutputRecipe 完成，见[胶球接入](../../architecture/glueball_architecture.md)。

执行入口先核对冻结计划的类型、展开身份及基本准入状态，再准备受管输入、
访问执行缓存或激活设备。该检查成本固定，不重走规划或完整资源审计；
完整绑定与资源关系核验仍仅在 `debug=True` 执行。错误计划不得触发输入生产或数值读取。

收缩编排/E 在路径搜索前统一完整任务的计算 dtype：采用输入中最低的浮点分量精度，有复数时保留复数。
物理读取仍验证原始 dtype；`TensorPreparation.compute_dtype` 记录读取后的转换和准备运算精度。
所有逻辑 TensorSpec、快照及准备工作区与新精度一致，`OutputRecipe.dtype` 固定输出系数和装配的精度。
CPU/GPU 都在当前需要的块物化时转换，不在每次缩并内转换，也不整体转换未请求的时间对。
普通执行边界拒绝混合 dtype 的旧计划；需要用统一输入描述重新规划，不能只改运行时数组。
当前统一范围为可描述的稠密准备配方；无法描述转换的 opaque、特殊 sparse/baryon 混合输入明确拒绝。

低精度窄归约在既有工作区允许时，由 `execution/reduction.py` 按缩并标签分块，
用 float64/complex128 计算和累加局部结果，最后转换回计划 dtype。该局部数值实现
不改变输入定义、Action 顺序或输出类型；普通双侧自由指标的矩阵乘法继续使用 BLAS。
绑定阶段逐块核对上界 `8*输入块字节 + 4*宽精度输出字节 + 256 <= 原输入字节 + 原输出字节`，
不满足或分块超过256块时保留既有内核。重复对角标签在每个出现位置使用同一切片，
普通、debug 和固定调用表共用准入和预算规则。CPU/debug 保留上述分块 einsum；
CuPy 普通执行由 `reduction_kernel.py` 直接按当前字节步长加载窄精度数据，在寄存器和
共享内存中用 float64/complex128 求和，再由第二个核归约宽精度 partials 并转换输出。
每个 256 线程块先用 warp shuffle 做寄存器内求和，只将八个 warp 的结果放入共享内存，
一次块同步后完成汇总；两阶段都保留宽精度。块数和分配预算不因这一步改变。
不物化输入的宽精度副本，不另做打包或逐块加法；两个临时分配的总字节仍不超过原
Action 的输入字节加输出字节。负步长、重复对角标签和保留输出轴共用生成的地址规则。
至多64项源码核和每个归约绑定至多64种步长布局只保存代码/几何，不保存本次输入和输出。
共享 producer 的不同批宽可产生多种真实行跨度；扩大有界布局复用减少暖回放中的源码生成、
参数表构建与首次上下文确认。形状、dtype 和当前数组步幅的读取与判定不因复用省略。
两输入绑定直接读取左右输入并核验数量、各自 shape/dtype，再读取各自当前 strides，
省去逐调用的两个生成器遍历；其它输入数保留通用检查。检查器不保存数组或地址，
各入口仍共用同一 raw callable；提交异常时在释放临时 owner 前同步当前 stream。
真实 GPU 编译、精度与性能须另验，CPU 源码解释测试不代表 CUDA 已通过。

两个二维输入、一个共享归约轴及各自独立自由轴的原生归约，使用按 dtype 共享的两份源码；
M/N/K、分块数和有符号字节步长在绑定时交付为 int64 常量，当前输入地址仍逐次读取。
grid.y/grid.z 分别表示列和行，避免在内核中除以动态矩阵尺寸；任一输出维超过 65535 时
保留平铺网格的通用源码。两阶段宽精度求和、分块规则、临时量预算和异常 owner 生命周期
不变；通用对角/多轴路径仍使用原静态几何，不为共享代码扩展支持范围。

多个输出的矩阵归约以固定 2×2 输出小块复用每线程的输入寄存器；每个输出保留原256条
线程累加链、K分段、blocks、宽精度乘加和两层 warp 求和树，四组 warp 部分和共用一次
块同步。奇数边缘只屏蔽无效输出，不让线程越过块同步提前退出；partials 布局、第二阶段
和设备数组工作区字节不变。源码仍按 dtype 共享；单输出沿原矩阵核，原 grid 上限回退保留。
寄存器增加可能影响驻留数，收益需真实设备测量，不能仅按理论读取量宣称加速。

公共 `DeviceExecution` 可额外预留最多8 MiB、512字节对齐的归约 partial 临时区；
准入从有效额度扣除保存峰值、常量和输出保留，并同步减少预读容量。只有未启动的独立
scene 或自有完整 cfg 队列能建立此预留，外部已启动窗口继续逐操作分配。
同一线程、device、CUDA context、stream 及宽精度 dtype 才借用同一个一维缓冲；
RawKernel仅消费其指针，不逐次建立切片或形状视图。结果始终独立分配并按原消费者生命周期保留。
无预算或协议能力时走原分配；CPU和debug沿原参考路径。正常退出在既有同步后释放，
同步或上下文失败则仍以失败抛错，并同时保留未确认完成的数组及其私有内存池所有者，
可能持续到进程结束，不能再次借用；正常关闭清除临时区对内存池的引用，再次准入时重新绑定。
预留/实际峰值、分配及命中/回退计数写入 device report；临时区不跨执行作用域保存。

`scalar_reduction_group.py` 通过 `scalar_group_execution.py` 接入公共固定执行表：内部
`bind_scalar_reduction_group(equation, shapes, dtype, allowances, backend)` 绑定2–32个
同指标、形状、精度及原分块数的独立标量内积，`run(tasks)` 接收各自的左右输入对，返回
不重叠的标量数组元组，小输出沿下述公共规则共享 owner。一次 partial 和一次 finish 提交保持每成员原K遍历与宽精度
求和树；源码按8/16/32成员容量共用固定参数槽，2–8成员源码及ABI保持原样。
四阶输入32路的首核参数为2576字节，finish为272字节，均不超过3072字节。
更高rank由P选择16或8路，X仍核验参数上界；Cartesian矩阵组保持8路。
当前指针和步长更新，不新增打包或结果复制。
仅一次借用所有 partial 所需总字节，按成员偏移隔离；不扩大既有缓存，借用失败时按已声明
总工作区分配。私有池与异步使用的 owner 在异常同步失败后保留。C 的独立分组记录与
内存预算将这块 partial owner 和每个独立标量输出分别按512字节池粒度计入；
allocator rounding、提前标量桶和经验池余量保持各自含义，异常诊断同时列出两种增量。
内存桶证明独立性、输入驻留、同时工作区和提前输出的存活预算；X 不改变 Action、事件、
释放或 retained 的原位置。原单项内核仍完整绑定；只有各成员原本满足公共宽精度 raw
准入，才在首成员一次发布独立输出，后续成员不再提交计算。CPU/debug、不支持的几何、
部分 reused 或原本不能宽精度执行的小归约保持原逐项路线，debug 仍核验资源预算。
measure 将首成员记录为两次原生提交，后续成员记录零计算，聚合值不用于单项内核校准。
`BoundSceneExecution.scalar_groups` 仅列出实际绑定成功的组，与笛卡尔合批分别记录。

cfg 模板复用 remap 标量组的 Action 索引；只要有一个成员 reused，整组回到各成员原位置
执行。提前输出和 partial 的独立内存桶在 reused 的中间 Action 上仍保留，过滤后允许保守
多计。cfg 执行表签名包含分组记录；缓存只保存不可变结构，不保存数值数组或地址。
每个稳定成员组分别保留其有界 stride/ABI 缓存，避免不同组的布局组合互相挤出；
编译源码和原生 kernel 仍全局共享，不按组重复编译。
每次执行读取当前数组，直接 cfg owner 表和延迟结果视图使用相同约定，并继续继承
producer/prepared-side 布局。专项 CPU 测试覆盖数值、事件、输出 owner、回退和预算；
显式 CUDA gate 另核验公共 storage 回放、当前地址/元数据变化及 cfg 直接 owner 表的逐位结果。
CPU 源码解释测试不表示 CUDA 已通过，不新增前端开关。

`reduction_group_runtime.py` 复用标量和矩阵组的当前输入检查、参数绑定、64项步幅
缓存、arena 借用、输出分配和异常 owner 管理。模式、网格、常量和输出 shape 在
绑定时确定，暖调用不重新选择策略。标量旧8路生成源码保持逐字节不变；
新增16/32路只扩参数槽和task选择，不改变每成员数学、网格分块或partial排列。
公共绑定键保留稳定成员身份和容量。32个complex64标量共256字节，
共享输出仍在512字节准入内；任意非空活子集的旧独立owner计费至少512字节，
因此延至最后一个子结果释放也不超过已有ordinary/cfg生命周期包络。
标量组最大partial为32×1024×16=512KiB，原8MiB arena及借用失败分配规则不变。
固定的两输入任务直接检查左右数组，省去内层 zip 和临时步幅列表；当前 shape、dtype、
步幅和地址的检查仍保留，新步幅布局的整数范围核验仍只在缓存未命中时执行。
`matrix_reduction_group.py::bind_matrix_reduction_group(shapes, dtype, allowances, backend)`
是内部数值绑定，接收2–8对已准备好的同形 `(M,K)/(N,K)` 矩阵，
返回每对子任务互不重叠的 `(M,N)` 结果；不负责 packing 或提前结果发布预算。
源码按 dtype 共用，M/N/K/blocks 在运行参数中传入，task 在 CTA 入口解析一次。
绑定已证明 M 和合并 grid.z 不超过65535，令 d=ceil(M/2)，预先计算
magic=ceil(2^32/d)。CTA 用64位无符号乘法及右移32位得到 task，再用减乘得到行余数，
不做动态整数除法或取余；有界 z(d−1)<2^32 保证商严格等于 floor(z/d)，d=1 时仍
保留 magic=2^32。首阶段参数增加8字节至432字节，不增加设备表、工作区或按形状编译。
行坐标、signed stride 和 partial 偏移继续64位，不改地址或归约顺序。复用
生产矩阵核的完整2×2算术主体，保留每输出 K 序列、原 blocks、宽精度树、partial 排列
和最终 cast。各子任务的 blocks 必须一致；超过原或合并网格上限、预算不足、CPU/debug
及不支持的几何返回 None。一次借用完整 partial 总量，超过现有8 MiB arena 时按调用方
已准入的总工作区独立分配，不扩大 arena。CPU 源码/参数解释和独立宽精度对照不代表
CUDA 编译或性能通过；显式 CUDA gate 对照原生产矩阵核的 partial/final 字节，需另行运行。

普通 dense 场景中，CUDA 原生标量/矩阵归约组的全部输出 payload 不超过512字节时，绑定一次确定使用一个
allocation owner，结果仍是互不重叠、可独立修改的数组视图；超过512字节继续逐子任务
独立分配。真实 CuPy 先 slice，再用公开 `view.shape = output_shape` 原位设定形状，
只创建一个子 ndarray；CuPy 该 setter 无法零复制时会抛错。fake/非 CuPy 协议仍用
slice+reshape。保留共同 ndarray.base root，使后续相邻拼接仍能零复制；直接
`ndarray(memptr=...)` 虽能保活内存，却没有该 base 链，故不采用。标量始终是零维数组，
不提取宿主 scalar。每次调用重新分配、
传入当前指针，不缓存数值；只减少输出 empty 次数，不改两次 launch、partial、求和树或预算。
CPU/debug/HIP 与不支持组的既有回退不变。

这个策略依赖现有 CuPy 请求粒度和逐 owner 预算：任意非空存活子集原来至少保留一个
512字节 owner，而整个共享 payload 只需一个512字节请求。因此延至最后一个子结果释放，
在 read、跨坐标 reused 和 output 阶段仍不超过原独立 owner 包络；既有 early-output 桶
继续保留，不抵扣任何费用。普通计划已有逐 owner rounding，cfg 需要全局 rounding v1 或
通过旧档案资源兼容审计；旧低估 cfg 必须先重做资源规划。含 opaque/core/sparse 或显式
资源 Action 的场景保留独立 owner，因为公共绑定没有这类逐 owner 证明；不从 cfg 特例
猜测更广准入。裸内部数值 binder 默认也保留独立 owner，公共执行表仅在上述普通场景
绑定时传入私有策略参数，缓存签名包含此策略。其 allowance
仍只表达原 logical scratch，调用方原有的物理资源准入责任不变。未给总 payload 大于512的
组推广共享 owner，因为其尾部存活可能反而占用更多字节。

共享 owner 创建后即显式保活；第一个或中间 view 创建失败也会同步，若同步再次失败，
owner、已建视图、输入、partials 和私有 pool 一起保存，不依赖异常 traceback。
CPU 来源解释及存活子集 oracle 只验证数值/布局/预算逻辑；新增真实 CUDA owner gate 与
原 partial/final 字节 gate 需在下一次集中测试运行，不据分配次数推断端到端加速。

`cartesian_group_execution.py` 将保存的 `CartesianReductionGroup` 接入同一个 scene / storage /
cfg / `BoundProgram` 执行表；组引用多个完整的原 `DenseKernelBatch`，不展开为独立 scalar 组。
绑定复用每个 child 已有的 `_BatchLayout`，只有原 `precise_product` 已满足 raw 2×2 宽归约、
双侧 stacked、实际 `(M,K)/(N,K)`、共同 dtype 和原 blocks 时才启用；原最小逻辑成员的
workspace allowance 不扩大。每个稳定 child 集合保留独立的 stride/ABI 缓存，编译源码仍全局共享。
CPU/debug/HIP、原宽核不可用、组不支持、任一 child 缺失或部分 reused 都在绑定期整组退回原路，
原 child callable 始终保留；运行中的 shape、打包、分配、launch 或发布错误直接抛出，不重算。

组首按原 side.pack 准备每个 child，保留 prepared producer 布局、相邻 owner 借用与动态回退；
只在实际 shape 不同时 reshape。一次分组提交返回互不重叠的 child cube，再交给原 views/deferred views
发布。普通 mapping 先构造全部结果视图再 update；cfg 继续使用 `_NodeView` 的当前数组引用、
延迟视图与每 child 独立 `_ResultOwner`；总 payload 至多512字节时底层使用上述共同 owner。
从第一次 pack 起持有已准备 owner，
异常先同步；同步失败继续持有异常栈、输入、packing、输出以及私有 pool。首位置以外只省略
数值工作，原 before/after/read/release/retain 位置保持不变。

cfg 在实际 kernel_batches 与 reused 状态确定后按原批发布时间过滤整个保存组。缺失、拆散、
部分复用或任一输入在组首未驻留时保留原 child；不会重新搜索、重组或借用新组的提前输出
让其它组变得可执行。模板签名含组记录，local_plan 仅写入 tables 实际启用的组；模板的额外
packing/partial/提前 cube 内存桶可保守保留，包括跨 reused 与 read 阶段的费用。
measure 在组首归集各 child 的 dense_batch_pack 与一次 contract_kernel，后续成员零数值耗时；
标明实际两次归约提交。packing 按已绑定的合并侧及未覆盖原侧计算提交上界，prepared 侧为零，
每个 raw 侧按最多32源分块；Elementwise 回退沿其自身 ABI 分块。计数不重复包含已合并的原侧，
也不把可能借用 owner 而省掉的 copy 当作实测提交。未知扩展打包器没有可证明上界时明确标为
estimate；普通批只有逻辑描述而没有实际绑定信息的提交计数同样标为 estimate。
这些计数只在 measure 报告中读取，聚合观测不可用于逐 Action 校准。
这些接线的 CPU 数值、事件、owner、动态回退及预算检查不替代完整公共 GPU/cfg 验收，
裸矩阵测试的耗时不能当作完整 DD* 加速。

Cartesian 组的同侧非 prepared child 可在 X 绑定时合并 packing：只接受原 side 为
axis=0/prepend stack、完整内维形状一致且按原 `input_layouts` 映射后各源形状相同的情况；
不按相同 flat K 猜测布局等价。复用现有 `bind_packing/pack_original`，每次提交最多32个
原始输入，较宽侧沿现有固定分块写入共同buffer的不同偏移；每块按最坏 mixed-stride
维数检查3KiB参数上限，不引入新kernel源码。参数超限或不支持时仅该侧保留原
child packing；prepared child继续原producer借用。
可合并的侧一次分配共同buffer，再以连续child视图交给原宽归约，输出cube的非重叠约定不变。
每次读取当前shape/dtype/步幅/地址，错误直接传播；共同buffer在首次pack提交前就登记
到异常owner列表，同步失败时与private pool一起保留，不依赖调用方保留异常traceback。

合并buffer的payload等于原非prepared侧payload之和，512字节取整满足
`round_up(sum(payload)) <= sum(round_up(payload))`，因此不增加C/P已经计入的同时packing
额度，也不调整计划预算。整个合并侧若已连续位于同一个owner仍可零复制；各child独立
owner恰能分别借用时，合并可能新增一次copy。这是待整任务测量的取舍，不做运行时子集
重新规划。普通dense侧没有假定启用GEMM专用的PackedValueCache。CPU源码解释与显式
CUDA gate分别检查数值、提交数、owner范围、32路分块及失败保护；CPU通过不等于GPU提速。

`batch_reduction.py` 对含虚拟自由轴的 float32/complex64 笛卡尔合批保留宽精度累加，
避免合批后退回窄精度 GEMM。GPU 可用同一两阶段核一次计算整批结果；额外 partials
与输出临时量合计不超过最小原成员的输入加输出字节，原合批 packing 和共享结果
仍由 P 按完整存活期计费。成员输出字节从已保存的结果切片范围推导，保留 squeeze 前的
单例轴体积；绑定只需输入规格或输入数组，不读取尚未生成的结果数组。CPU/debug 用有界
K 分块的宽精度矩阵乘参考，空间不足时逐输出求和；极小原成员没有宽精度工作区时保留
原路线。普通双侧真实自由轴批继续用 BLAS。
新规划不将此类窄精度归约批合入 cuBLAS GEMM 组，既有组档案经兼容执行保留各子批的
宽精度规则与原共享结果 owner；异常同步发生在输入、打包和临时 owner 释放之前。
这是已有 `kernel_batching` 选项的内部实现，默认值仍为 False，不新增前端选项。
双侧均为虚拟自由轴的窄精度标量批，原生 GPU 归约可将右侧直接打包为 `(N,K)`，
以 `ik,jk->ij` 使两个输入都沿 K 连续读取；复用现有打包器，不增加转置副本。
这只改变 X 的临时物理布局，原 P 描述、打包/归约工作区字节、输出顺序和共享 owner 不变。
prepared 右侧只含一个输入，移动长度为1的虚拟轴仍借用原 producer 布局。
CPU/debug、原生核不可用或工作区不足时保留完整原布局；旧 GEMM 组也保留原矩阵指针约定。

NumPy 与 CuPy 的已选二元 dense 操作都可在绑定时转换为矩阵乘法，使用原有布局和
工作区上界，不重新选缩并路径。同 dtype 的对角或其它不支持表达式保留参考 einsum；
BLAS 的归约顺序可与参考不同，完整数值比较使用明确的浮点容差。
NumPy 的未绑定逐步参考操作仍沿 einsum，独立参考与固定绑定回放分别验证；
读重叠和异常清理测试在 einsum/BLAS 两种数值入口都注入同一门控或错误。
绑定后的 NumPy einsum 若返回原生且自有存储的 ndarray，直接交付结果；借用视图仍复制。
其它后端的所有权判断沿原实现，不根据 NumPy 标志推断设备数组的存储契约。

公共 `Calculation.run()` 在同一句柄及其结果副本间共享 X 的 `RunResources`，
它与仅持有结构的 `ExecutionBindingCache` 分离。支持原生设备上下文身份时，
复用一个受当前计划设备预算限制的显存池；每次仍建立独立传输票据、stream、
输入、输出和释放事件，返回前完成依赖及清理。异常即清空该池；并发使用同一
owner 时另用临时池，不共享数值状态。设备/上下文或预算改变会替换旧池。
`Calculation.close()` 或上下文管理退出可提前释放空闲池，既有 CPU 结果不变；
后续 `.run()` 可重新建立池。句柄销毁也会回收，不持有历史数值输入。
固定单场景的不可变配方常量可随同一个池复用；最多 128 项、8 MiB，容量还受
当前已准入显存减去原计划峰值的余量限制，按实际分配 extent 而非 ndarray 逻辑字节计费。
该预留从预读可用显存中扣除，不能与上传队列重复使用。公共固定回放将已有的新鲜显存
准入结果传给设备池上限，不为常量缓存再次查询设备。无余量、debug、其它入口或不明
分配 extent 保留原流程。缓存键是保存的常量内容；改变配方不会读到旧值。
设备/上下文/池预算变化、异常或 close 会清空；并发回放使用独立临时池及常量。
动态显存检查仍每次查询驱动，只补计当前 owner 可供本次复用的空闲池字节，
以及可复用或可淘汰的本句柄常量 extent，不把其他句柄、活跃运行或别的设备计入。
CPU 不建立设备池。缓存不持有传播子、elemental、准备结果或当前指针。
单场景不超过四块且总计不超过 8 MiB 的明确主存数组输入，可省去预读 worker；
文件映射及其视图、未知 reader 和跨批次队列保留原流程。
其中 GPU 上传可沿原预读队列在调用线程提前提交：仍先核算主存和显存余量，
逐块预留两域容量；计算流等待各上传事件，锁页暂存到传输完成后才释放。
CPU 小数组继续同步直读，GPU 无额外余量或 reader 不支持预读时也保留直读。
此方式省去线程启动并允许 H2D 与计算重叠，不意味着多条上传流或多个 CPU 核并行。

普通固定单坐标 Calculation 和 CompleteCalculation 共用 R 的 `OutputAssemblyCache`。
同一 handle 的替换副本共享一份不可变输出定义的绑定；更换定义即失效，debug 走原审计，
求值缓存仍只属于本次调用。多坐标普通 Calculation 继续按坐标装配。
GPU Dataset 后处理使用自己的设备程序，不从该 CPU 装配缓存推断收益。

普通 `Calculation.run()` 与重复使用的 `BoundContractionExecution` 自动持有 X 的
`ExecutionBindingCache`。单场景和已选坐标批通过同一个 `_BlockExecution` 缓存调用表；
同一次单场景回放在入口绑定准备器后，块执行器直接复用该运行内绑定；不同计划或后端仍重新绑定。
计划标记使用弱引用，不延长流式窗口的寿命。公共执行缓存还可按准备角色复用原生 identity、constant
和 spin_spatial 叶操作，缓存只捕获不可变配方、数值几何与后端；不捕获 reader、当前数组、设备常量或指针。
同一角色的配方或 shape/dtype 改变替换原槽，不积累历史版本。每次运行独立建立常量与融合程序，
实际数据和布局仍来自当前输入；可变表在运行内替换记录时清理相关常量，debug 走原绑定。
stack、opaque 和扩展描述仍使用原有递归/融合路径，不因此改变准备内核或数值顺序。
普通单场景 GPU 存储回放还可让同精度 identity 准备接管本次 H2D 的自有分配，
省去一次设备复制。X 同时跟踪原始块与逻辑使用者的释放；两边均结束后才记录计算完成事件，
并沿原上传流/预读额度回收。多个逻辑使用者共享同一上传时，等到最后一个结束。
主存 reader 按原事件关闭，不额外延长主存驻留；设备存储始终由原始或逻辑寿命覆盖，
原计划的独立快照上界继续保留。实际 dtype 不同仍转换；CPU、外部设备数组、debug、
跨场景 shared/retained、资源搬移及其它执行入口保留原复制路径。
交接只记录当前运行的数组；执行表只保存角色和释放序号，异常及最终输出均完成交接清理。
键覆盖场景描述及后端对象身份，原生不可变 Action、事件、合批与 retained 描述才可复用。
作用域由完整已选路径限定，换路径/后端替换缓存；不建全局历史缓存，也不缓存当前数值。
缓存根只持有纯 SceneStoragePlan（坐标批为全部场景计划的有序身份），不持有含当前
expanded/loader 的 PlannedContraction。替换运行包装而场景计划不变时可继续复用；
任一所选场景计划或后端变化即替换缓存。释放调用者的绑定后，缓存不会延长 reader/输入数组寿命。
只有已接受不可变绑定才发布缓存根；可变扩展或首次绑定失败不保存空根，避免间接保留运行 owner。
首次绑定检查可复用性，命中后不重复扫描 Action。debug 和可变扩展描述重新绑定。
普通存储回放还在同一绑定作用域复用已成功的静态资源审计：首次完整审计后，额外证明
原生 catalog、Action/轴关系、合批/分组、事件、MemoryPoint、资源限额与保留标记不可变。
更换计划（包括预算或预留量替换）或后端对象失效；debug 每次审计，未提供缓存作用域时
也每次审计。可变字段、扩展记录、catalog 子类、core 与显式临时资源路径保持原审计。
首个作用域失败不发布新缓存根。缓存只保存成功标记，不保存 reader、数组或设备状态；
当前可行性、loader、坐标批 workspace ABI、输入 shape/dtype、设备及动态资源准入仍逐次核验。
准备和物理读取描述仍由本次 loader 绑定处理，不把该证明用于跳过其动态检查。
该机制由 CPU/GPU 的普通 `replay_scene_storage` 与非流式批回放共用，普通
`Calculation.run()` 的对应存储路径同样受益；完整 cfg 的 `_CfgBlockExecution` 原本只在
debug 做此深度资源审计，不能把普通回放的节省外推为完整 cfg 的相同收益。
原生 MatrixKernelSpec 与 TensorCoreSpec（含不可变 CoreComponent）同样可复用，
因此多分量重子 core 不因采用专用数值核而退回逐次绑定；其原计算顺序保持。
底层固定 `replay`、`replay_scene_storage`、非流式 `replay_storage_batches`、
`execute_contraction_path` 可显式传 `execution_cache=ExecutionBindingCache()`；
普通前端无需传参。流式窗口继续使用原有的有界相邻窗口复用机制。
bare replay 仅以原生 state 绑定其无 storage 变体；外部 source execution 若未提供
完整执行表，按其当前合批描述重绑，不能借用同一 state 的 bare 表。

动态 `ContractionSession` 的各入口在编译前通过 B 的 `normalize_array_precision`
使用同一最低精度规则；同 dtype 借用数组，转换只产生本次调用的 owner。
Session 共用上述执行缓存的身份及 debug 规则，复用 IR 的静态依赖计数，CuPy 另复用
`PreparedKernels`。NumPy IR 保留原 einsum 参考算法；显式低层混合 dtype 计划仍拒绝。
NumPy IR 与场景核共用原生数组所有权判断，自有 einsum 结果不再重复复制，借用视图仍快照。
持久 Session 已有 `PreparedBundle`，沿用其固定交接，不再加第二套执行表。

同一 CompleteCalculation 的多次 `.run()` 还自动复用 X 的固定执行表。收缩编排只持有
`CfgExecutionCache` 并传入执行边界，它沿用公共执行缓存的身份/发布规则；缓存只保存一个原生
CfgPlan 对象身份和 backend 对象身份匹配的描述。替换计划或后端重新绑定，debug 每次
重新绑定并保留审计。CPU/GPU 共用该机制；memory、Dataset 和 cfg iterator 经同一交接。
每次调用仍创建独立的 reader、数组驻留、预读队列、输出及清理状态；相同表可供并行
回放读取。缓存不存数值输入/输出、设备句柄或 stream，也不进入计划档案；新加载的 handle
首次运行重新绑定。底层 `execute_cfg_plan`/`iter_execute_cfg_plan` 的高级组合可以传入
`execution_cache=CfgExecutionCache()`；普通前端无需增加参数。

单个二维批次输入在 dtype 和形状匹配且为 C/F 连续矩阵时，直接借用当前视图给 GEMM。
不另作 C 连续复制；输入仍由当前活跃表持有，输出独立。类型提升、高维展开及其它布局
保持原打包和工作区上界。NumPy 二维批次允许 F 布局交给矩阵乘法。

P 可在含 baryon join / sparse bridge 的计划中保存局部 Cartesian 批及 CUDA 独立 GEMM 组。
X 沿相同 Action 表执行，特殊操作保持原位置，组不能越过特殊操作边界；档案回放不再次选批。
无 logical retention 的任务分批可分别携带已选批。此覆盖扩展复用原数值核，不新增前端参数，
不扩大原稀疏 GPU 或设备平台能力；当前 CPU 参考通过不构成真实 CUDA 吞吐证据。

完整 cfg 的普通 GPU 输出也通过私有 owner/位置快照进入同步或异步 D2H，
不先构造每个 Term 的设备切片。下载仍按原 dtype/shape 分组，完成事件持有全部
设备 owner；下载完成后才创建主机数组视图，公开返回的 tuple、标量零维数组、顺序、
重复 lane 和独立主机存储保持不变。自定义设备 transform 仍须声明接受该私有快照，
其它调用及 debug 路径保持原交付；没有新增设备同步或公开开关。

## 增量窗口交接

`X.api.iter_execute_cfg_windows(windows, configuration, loader_factory, *, backend=numpy,
host_memory_limit=None, device_memory_limit=None, prefetch_host_bytes=None,
prefetch_device_bytes=None, debug=False, report=None)` 消费 P 已完成资源检查的 `CfgPlan` 窗口。
`loader_factory(window, cfg)` 返回该窗口 reader 的上下文管理器；输出 `(cfg, index, values)`
使用整个 cfg 的坐标序号。`execute_cfg_windows(..., consume, **options)` 是对应回调入口。
仅在输入窗口流正常结束后发出一次 `(cfg, None, None)`，供上层发布完整产物；失败或提前关闭
不发出完成通知，并清理 reader、预取队列和支持 `close()` 的规划生产者。

窗口复用同一场景执行器和持续读取/上传队列，窗口内沿用准备、合批、驻留和释放事件，窗口间
不保留数值中间量。执行器最多接纳当前与下一个窗口；生产者可另有一个已请求的后续窗口。
提供 `next_ready()` 的生产者尚未完成下一窗口时，当前计算继续，仅在窗口边界等待。
各窗口必须保持同一执行目标、backend、实现版本、输出模式和连续坐标序号。资源额度在接纳
窗口时检查，预取额度只缩小，不因未来窗口增加已承诺的额度。固定执行表按完整不可变模板、复用 mask、合批记录和事件位置跨相邻窗口复用；调用表缓存仅保留当前窗口实际使用的结构，不保留 reader、全局数据 ID 或数组。改变结构必须重绑。
完整模板匹配后，数值核独立于复用 mask 和资源事件表缓存；不同调用表可复用同一方程及 producer 布局的核，合批按完整记录区分。数值缓存限定为该模板的 Action、布局及原合批/子批，不随历史窗口累积；模板或 backend 改变即失效。每次调用仍解析当前数组，当前事件位置和生命周期不从旧调用表借用。
应用窗口复用 mask 时，只派生复用标志或成本实际变化的不可变 Action；不变记录共享原模板，全部不变时共享原状态。复用操作的成本仍清零，模板本身不修改。
完整 cfg 的正常固定执行表直接使用共享数组表中按 P 驻留的复用结果，省去逐个复用 Action 的存在性查询；该 Action 前后的资源事件仍按保存位置执行。普通场景仍从独立 retained cache 恢复，debug 保留解释路径中的驻留检查；全局释放、输出快照和异常清理不变。
冻结 raw 准备调用在布局命中后复用常量参数、标量和启动尺寸的绑定骨架，每次只填入当前输入与新输出。CUDA 支持 CuPy CNumpyArray 时，64 位参数值与持有单元素主机视图的参数容器在每个线程首次使用时准备；后续不逐参数建立 CArray 或 NumPy scalar 包装。CNumpyArray 从视图的数据地址交付参数值，不使用会将 Python 对象本身转成指针的 CPointer 构造器。动态地址在提交返回或抛错后清空，输入和输出不进入缓存；常量 owner 继续随 loader 存活。debug、HIP、缺少此能力及自定义核保留原交付。stride/dtype 布局键、32 路 ABI 上限、舍入规则及原参数字节限制保持不变。

普通 CUDA RawKernel 支持 Function/driver ABI 时，线程同时复用指向 uint64 槽的主机 void** 表，
避免 Function 每次重建参数指针向量。每次仍由 RawKernel.kernel 解析当前设备 Function，
首次使用或 Function 改变时建立线程运行时 context，当前 stream 每次读取；不跨设备缓存旧 Function/stream。
表只持有主机槽地址，没有新的设备指针表或设备工作区。合作组核及缺失该 ABI 时保留 CNumpyArray 提交；
debug 沿原公开调用，正常提交没有同步，异常原样交由既有数值 owner/事件清理处理。

固定 raw 打包、GEMM 组打包及输出快照复制共用上述参数槽服务。布局首次命中时保存每个分块的输入索引、启动尺寸及输出 lane 偏移；热路径从完整当前输入序列读地址，不再逐块建立指针标量与参数 tuple。线程首次调用建立本机 uint64 内存视图，后续填入及清零地址不进入 NumPy 标量转换；视图保活原主机槽，CNumpyArray 视图和 driver 地址表仍指向相同的槽字节。延迟结果引用和连续拼接判断直接读取当前 owner 地址加已证明的切片偏移，不创建数组视图。每线程槽位仅保存本次提交前的动态地址，提交返回或异常后清零；布局缓存仍至多 8 个变体，设备复制、混合 dtype 转换及工作区额度保持不变。debug 与不支持该参数 ABI 的平台沿用原调用。

地址清理也在绑定时固定：至少 8 个动态槽、相邻区间数不超过动态槽数一半时，保存这些区间的零字节，提交返回或抛错后一次清理各区间。较宽且零散的布局保存固定模板的主机字节，通过一次复制恢复常量、标量和零动态槽；短布局仍逐槽清理。字节视图与 uint64 槽及主机参数表共享同一线程的存储；清理后的每个字节与原规则相同，不持有动态 owner 或历史地址，不增加设备同步、扫描或执行统计。参数取得中途失败也沿同一清理路径。

已绑定为数组地址读取的参数槽直接按当前值填写：延迟结果用当前 owner 地址加布局偏移，
NumPy 读当前数组接口，设备数组读当前 data 指针。省去逐输入 getter 和延迟视图方法调用，
同一位置可在后续调用切换数组类型；任意自定义 getter 仍逐项调用。输入获取中途失败同样清零，
不缓存输入类型、地址或 owner，不增加设备工作区和同步。

固定打包侧及 GEMM 组在绑定时整理有序模板输入键。完整 cfg 的 `_NodeView` 提供内部批量
取值：先由固定取值器解析当前实例的全局 ID，再从当前数组表取得引用，避免每个输入
重复进入 Python 单项取值函数。单项、多项、重复输入均保留原顺序和缺失键行为，返回
当前延迟引用，不创建视图或设备工作区。绑定不保留数组表、实例映射、数值数组或地址；
debug、普通映射及只有旧单项内部协议的扩展表继续沿原 `_packing_values` 路径。

冻结模板准备首次绑定时，从当前 loader 的全局配方表建立只读的驻留 ID → raw 输入 ID 索引，后续实例的叶子角色直接查当前输入。回放不再逐叶读取 recipe、判断 spin 种类和提取 inputs；非 spin 项保留原 fallback。索引只含标识、随 loader/backend 独立建立，不保留数值或跨 cfg 的 owner；CPU/debug、可变表和不使用完整模板的路径保持原实现。

准备的调用布局键也减少同次重复读取：输入至少 8 项，首两项、末两项或首末项出现同一数组对象时，本次按对象 ID 复用 dtype/shape/stride 元组，依原 lane 顺序组成完全相同的键。临时表在调用内释放，下一次重新取得元数据，因而同一数组改变步幅、替换输入及不支持 dtype 仍经过原布局/冷路径判断；短、未观察到重复及 debug 调用保持逐项取值。这里只复用当前输入对象的元数据，不依据池化程序中可能过时的 recipe 输入标识选择数值来源。

内置 GPU 数据集后处理通过私有消费者能力交接 owner/位置快照，不提前生成全部 Term
数组视图；普通消费者和公开 tuple 不变。X 首次对冻结布局检查 Term 顺序、规格及 lane
边界，后续沿用有界弱身份 owner schema 缓存。间接参数表直接使用 owner 的当前指针、
lane 偏移和实际步幅；直接参数按需创建并在当前快照内复用视图。既有完成事件保留完整
快照与数值 owner，缓存不持有当前数组或指针。debug 仍走原 tuple 及逐 Term 检查。
CPU 协议及地址/步幅 oracle 可验证此交接；CUDA 数值与性能须分别在实际设备验收。

免复制拼接先核对底层 owner 身份，再检查完整逻辑布局与覆盖。独立分配无需读取各项
shape/stride 或地址即可回到原打包路径；共用 owner 的 shape、dtype、stride、次序及
字节边界仍全部核对。检查属于实现选择的必要证明，不受 debug 开关影响。

固定合批侧在绑定时保存免复制证明使用的 C 元素步长、总元素数、轴映射和单输入排列，
普通计算、完整 cfg 和 GEMM 组共用同一绑定。逐次检查直接读取当前 shape/stride，
不再为虚拟轴与排列各建一份逻辑 shape/stride 元组。字节步长只保留一个 itemsize
对应的纯元数据记录；类型宽度改变时替换，局部调用使用自己的不可变快照，不缓存 dtype
集合、数组或指针。当前 owner/base、dtype、连续性、实际布局、地址、对齐和范围检查不变；
单个未映射输入也必须匹配绑定 shape 才能借用。这是 CPU 与 GPU 共用的主机端优化。

完整 cfg 的原生节点 ID 已由 C 核对为一维 int64 表。X 逐点绑定时直接取得 Python 整数列表，
省去逐元素构造 NumPy 标量再转换为 `int`；列表仅在本次字典构造期间存活。键的次序、
整数值和严格长度匹配保持，debug 与外部记录保留逐元素转换，不缓存跨坐标的 ID。
正常固定回放另外复用一个仅含原生字符串键、值全为 `None` 的字典模板；选择当前计划中
出现次数最多的模板，字典分配最多 8 MiB，不为各时间对保留完整 ID 字典。逐点复制键表后
填入该实例当前解码的 ID，得到独立的原生绑定，后续取值仍走原字典路径。其它模板、超限、
debug 或扩展可变描述保留原构造。相同不可变计划/backend 的 owner 缓存可跨 `.run()` 复用
键表；窗口变化只保留当前键表，相同键序可复用，换序重新建立。并行回放不修改共享键表，
reader、数组、节点 ID、指针和 cfg 值均不进入它；该上限只约束新增主机调度元数据，
原数值工作区、输入校验及预读预算检查保持不变。

GPU 输出源码生成按二元表达式的左右子节点直接推进后序栈，省去逐节点生成器；
子节点与父节点同形时直接传递索引，其余形状仍用原广播。生成语句、变量和参数次序、
每级舍入、常量及资源预留保持相同。这里只减少主机绑定开销，实际数值仍由 GPU 执行。
表达式描述的微型类型探针共用一次 NumPy 错误策略作用域，不逐二元或幂节点切换。
成功和异常退出均恢复调用者的策略与回调，内核工厂在作用域外运行；类型提升、形状和
不支持类型的拒绝保持原行为。Python 3.11/NumPy 1.26 的八轮完整描述与冷绑定对照
中位墙钟为 18.69→16.68 秒；487 阶段源码及 ABI 相同。此计时不含 CUDA 工厂、JIT 或 GPU 执行。

X 绑定 CuPy 已展平的二维矩阵乘时选择公开 `dot`，省去通用 `matmul` gufunc 的广播和形状处理；
带 batch 轴时，仅原生 CuPy GUFunc、两个矩阵具有相同 batch extents/dtype 且
使用默认轴的调用，直接进入该 GUFunc 自身的底层 `matmul` 例程，保留其精度策略、
当前 stream 和输出分配，原例程继续处理转置、负步幅及零步幅视图。每次核对当前布局；debug、batch 广播、dtype 变化、
能力缺失或公开入口被包装/替换时仍使用公开 `matmul`。NumPy 与协议后端保持原路。
float64/complex128、默认计算与 cuBLAS math 模式下，批矩阵的 M=N=1 且
K≥4096 时复用现有两阶段归约；生成源码使用 double/complex<double> 输入、乘积、
累加与输出，不经过窄类型。partial 与结果总字节数不超过原 Action 的输入加输出
工作区；直接消费原始张量的步幅，不先做矩阵打包，因此不重复消耗该工作区。
当前 stream、输入与输出所有权仍由既有归约绑定管理。
其余带 batch 轴、K≥4096 且 M、N 都大于 1 的 CuPy 矩阵乘不交给 strided-batched GEMM：
它不拆分长归约，V100 上 complex64 的 (6,6,10⁶)×(6,10⁶,4) 需 104 ms，逐矩阵 GEMM 只需 3.5 ms。
批数 ≤16 且（批数 ≤4、K≥10⁵ 或 M·N≥16384）时逐矩阵调用 `dot`；否则当批数×(32×32 输出瓦片数) < 256
且 K 在 [64, K/64] 内有因子 s 时，把 K 拆成 s 段做批矩阵乘再沿段求和；输出瓦片足够时保留原批矩阵乘。

完整 cfg 的普通 Cartesian GPU 批及固定 GEMM 组可向私有数组表交付延迟结果切片：绑定时证明切片的
形状、步幅、squeeze 和字节边界，运行时只保留当前 owner 与冻结描述。数值消费者
首次需要数组时创建视图并替换该条引用。发布时在当前数值 owner 内读取一次固定存储的
基地址；输出快照与固定 raw 打包只加保存偏移，避免为仅需地址的复制逐项创建 CuPy 视图。
每次新结果都创建独立 owner；地址只随该结果存活，不进入模板、核缓存或持久计划。
打包仍使用已证明的 shape/stride，并按共同
owner、完整步幅、地址次序及字节范围判断能否只借用最终连续视图；不满足时使用
原有打包工作区。参数持有本次引用，核变体不缓存数值 owner 或指针。原来的独立输出副本、全局释放和
合批 cube/指针表所有权不变。debug、measure、CPU 和普通数组表仍创建原有视图。
地址/步幅 CPU oracle 与实际 CUDA 数值验收分别报告；本地协议测试不能代替设备验证。

这是显式增量接口；已保存完整档案仍直接回放，不隐式启动 P。普通 `calculate` 与完整 cfg
前端尚未自动切换到窗口流。

准备配方中的相同不可变 `ArrayConstant` 在一个 loader/backend 生命周期内共享一次数组转换和设备上传；每个准备 ID 保留自己的配方与结果生命周期。现有缓存也先按常量描述去重再转换，换 backend 或 cfg loader 时建立独立缓存。固定准备在 loader 生命周期内绑定数值调用。适用的 CuPy float32/float64/complex64/complex128 spin/profile 与兼容嵌套 stack 使用固定融合核：每个线程按原顺序作三次有类型舍入的乘法、逐 profile complex128 累加，再应用 dagger/轴排列、spin 乘法及每级 stack cast；不代数重排，不重复计算16次 profile 或另分配 projected 矩阵。RawKernel 指针路径每次最多32个 spin leaf，形状和 byte stride 固定在生成代码中，每个数组参数占8字节，另有两个 int64 和一个输出指针；legacy ElementwiseKernel 路径仍最多8个 leaf，并保留保守3072字节 CArray 参数上限。每绑定最多8个核变体。raw kernel 逐次接收当前数组指针，不上传指针表。CPU、自定义后端及不支持的 dtype/profile 广播保留固定广播乘法与逐成员切片路径；非连续输出回到 legacy 分块与参数限制。仅 X 自有 spin_spatial/stack 的独立分配可直接接管；identity、共享常量和 opaque 回调仍快照。可变配方/规格被替换时失效相关绑定，不缓存 raw 数值结果。

日期：2026-10-08。职责按 [L0/L1](../../architecture/correlator-l0-l1.md) 与 [H05-H08](../../architecture/correlator-target-architecture.md)；本文件记录现有实现及 CPU 证据，不把目标平台或生产规模视为已验收。

固定 raw 准备核在绑定时按常量数组身份合并重复参数。同一 spin/profile owner 在一个
分块内只交付一次；相等但独立的数组仍分别交付。较宽非 debug 调用发现重复的当前输入对象时，
raw 输入指针也可在该分块内共享槽；每个叶子的逻辑视图和数值表达式仍独立。
首次出现位置模式加入调用缓存键，每次调用重新解析，不把当前 owner 或地址存进缓存。
debug、短调用及未发现重复对象的调用继续使用独立输入槽。此项不改变 P 的物理身份或设备额度。
每个叶子保留原名和原 shape/byte-stride 的索引代理，代理可指向共同常量参数；原数值
表达式、舍入与累加次序保持。参数别名模式纳入变体键，冷调用与热调用使用同一绑定
交接表；不捕获当前输入或结果。最多32个 raw leaf、8个 legacy leaf，按未去重参数
数量保守限制3072字节；缓存、工作区和结果所有权不扩大。CPU协议和实际GPU数值
验证分别记录，尚未证明端到端收益或GPU利用率目标。

raw 源码按完全相同的叶子 dtype、形状、步幅、profile、conjugate、排列及逐级 cast
描述共享算术体：每个线程先按原 lane 选当前视图，再执行一份原表达式并写原偏移。
相同索引布局的代理共享 C++ 类型；每个叶子的参数位置及逻辑视图仍保留。不同舍入或
布局不合并，无可共享项时保持原算术源码。此项由默认固定准备入口自动使用，不新增
前端选项、设备指针表、数组、同步或工作区，不复用历史输入及地址；缓存容量及 ABI
额度保持不变。首次编译和稳态性能分别用同设备输入验证，不将源码缩小等同于端到端加速。

RawKernel 在一个已选 stack 的输出内按最多32个独立叶子分块，减少较大 stack 的启动次数。
叶内的 profile 累加、类型舍入、轴排列和写出偏移保持原表达式；输出仍一次分配。
启动数缩减和实际设备的 warm 时间分别验收；较宽分块不代表所有规模都会更快。

## 全 Diagram 整数连接格式（当前工作区整合验证进行中）

权威格式见[ARCHITECTURE](../../../ARCHITECTURE.md#diagram-的统一整数连接格式目标待迁移)。X 的 dense、baryon 和 sparse 核路线已使用
`AxisPattern`/整数元组。NumPy 使用交错数组/整数参数，
现有 GPU matmul 路线直接消费 Pattern。需要字符串表达式的实际后端才在调用处生成局部标签，
不将方程存入计划。此前 CPU 定向检查已完成；所有 GPU 核、DCU/MPI 与生产规模仍待验收。

## 完整 cfg 执行（阶段五；统一设备路径已接通，Ne64 V100 三组同输入完整 cfg 对照及冻结数组只读复核完成；更广覆盖待验收）

目标见[完整规划与直接执行](../../architecture/correlator-target-architecture.md#完整规划与直接执行)。阶段三增加 `execute_cfg_plan`，阶段五由收缩编排/A 接入完整档案、迭代配置和流式输出。完整 cfg 按保存的通用 Action、可序列化准备配方、已选 DenseKernelBatch 与资源事件执行；已描述的 baryon join 和 sparse bridge 使用共同 X replay。NumPy 与 CuPy 共用设备入口，coordinate-batched sparse GPU 与缺少可保存 preparation recipe 的 opaque 输入仍不支持。所有 GPU 核、DCU/HIP、MPI、多 worker 节点资源分配和生产规模仍待验收。

完整档案路线消费已保存的存储/预读事件与额度，用一个持续队列跨时间对和连续 cfg 推进，并沿用既有场景回放和数组生命周期管理。执行只按已保存事件及当前额度前进，不重新规划或预测成本。X 的 `replay_storage_batches` 窗口模式仅消费预先 join 和计入已知 sparse workspace 的计划；低层即时便利准备由收缩编排完成。

## 边界与公开依赖

多成分 `Action.core` 的 baryon_join 按 C 核心配方调用既有自旋稀疏/稠密内核，然后以公共结果精度复制、加权并累加；不生成稠密 rank-six 核心。共同计划同时计入各次计算、复制、权重/累加及保守工作区。单成分无权动作保留原执行路径。NumPy CPU 已与独立 einsum 和图级 source-adjoint 参考比较；2026-10-07 历史快照的 V100 定向执行测试通过，但该核心没有独立的 Ne64 性能验收，不能据此声明所有 GPU baryon 核已验收。

规范实现位于 src/lqcdweave/contraction/execution/，原子操作位于其 operations/ 子包。contraction.execution.api 是 X 的公开入口；旧 direct_contraction 转发入口已删除，直接执行由 X.api.execute_direct_contraction 提供；scene_runtime、scene_storage_runtime、baryon 和 sparse_bridge 的旧转发路径已删除。固定计划回放及重子数值核由 X.api 提供；重子编译并执行便利入口归收缩编排，contraction 包级调用保持不变；稀疏 kernel 内部测试直接核验实际 operations 实现，类型/label helpers 由 C.api 提供。无调用的 `contraction/compiler.py`、`contraction/executor.py`、`contraction/lowering.py`、`contraction/execution/compiler.py` 与 `contraction/operations/` forwarding facades 已删除，数值实现保留在规划/执行 canonical API。执行只读取 C 中的 ExpandedCorrelator、PlannedContraction、Action、SceneStoragePlan、StorageBlock、SourcePreparation、TermValue 及报告类型；计划合法性调用 planning.api.validate_execution_binding，测量调用 performance.api.ExecutionMeasurement，数值数据只由注入的 C SourceLoader 读取。执行不会调用路径搜索，也不会实例化 adapter 的 execute_direct 方法。泛型运行调度仍由 runtime.scheduler.execute_request 提供；收缩编排构建 BoundContractionExecution 后把它注入该调度器。

共享文件队列 `runtime.dispatch.Dispatch` 是运行时任务分配服务，不参与 X 的动作调度。`AtomicOpen` 始终在文件第 0 字节加锁，append 打开时会恢复原有写入位置；flush/fsync/seek/unlock 失败时仍尝试回到锁定字节、解锁并关闭文件。无 body 错误时传播首个清理错误，已有 body 错误优先传播。它不回滚已取出的队列行。`tests/runtime/test_dispatch.py` 验证单 rank 队列行为及文件锁清理失败路径，不构成真实 MPI 或共享集群文件系统验收。

输入准备的私有调用缓存逐次读取当前输入 shape/strides，并将四种受支持 dtype
映射到不含 metadata 的标准 dtype；缓存键不逐叶生成 dtype 字符串，也不保留
metadata 中可能引用的数组。不同 dtype、步幅及目标布局继续分开绑定；未知或
非本机字节序类型不能命中受支持调用，沿既有冷路径返回，不写入调用缓存。
内核源代码描述、每级舍入、工作区和持久格式保持原有语义。
模板准备中的每个 spin_spatial 叶按 C 的单输入契约直接解析当前 raw block。
省去逐叶的单元素生成器；普通路径逐次解析当前 gid/raw id，保留重复叶、顺序、作用域
检查及不支持配方的回退；不持有跨实例的数组或指针。
原生数字节点绑定的单叶角色也使用已有数字驻留索引，避免把当前整数 ID 转成字符串
后再查表。单输入仍是一元 tuple；外部映射及数字索引未命中时沿原文本解析，
保留首次消费时的缺失 ID/key 诊断。准备的数值程序及物理布局分派不变。
单叶直接读取唯一叶角色，不在绑定期建立位置提取器，也不在消费时构造单元素
raw 键元组或查询多叶展开表；只构造准备程序实际接收的一元输入 tuple。
固定调用表同时记录真正执行的非复用 load 角色。模板准备可为高复用变体绑定
有序独立叶及各角色的位置（至少64次叶引用，引用数至少为独立叶数的四倍），
每模板最多缓存8个变体。每次实例只建立当前 raw ID 元组，数值仍在消费时从
当前 raw 表读取；重复叶、原次序、必要数值规格检查和资源事件保持。
变体中的每个角色同时绑定位置提取器、一元 tuple 标志和适用的重复叶展开器。
消费时先按固定位置取得本实例的 raw 键，再按序从当前 raw 表读取，不逐角色创建
输入生成器或重新查询重复叶表。单输入仍交付一元 tuple；多输入的读取与首次
缺失键异常顺序保持。提取器只保存整数位置，不保存 raw 数组、地址或实例 ID。
单个准备角色的重复叶也在绑定时记录独立位置和原序展开器：至少8次引用且
引用数至少为独立叶数两倍时，每次消费各独立叶的当前输入一次，再展开原序。
该规则同时适用于模板逐角色解析和实例 ID 绑定，只依赖实际本地叶身份，
不依赖共享数值程序内首次绑定的配方 ID；缓存不持有当前数组或指针。
作用域在实例绑定边界检查一次；缺失或不支持的 ID 使整个实例绑定回退，
由原路径在首次实际消费时报告错误。低复用、CPU、不支持后端、可变表和
debug 不启用此绑定；不添加前端参数、数值缓存或设备同步。

### 输入准备的主机步骤审查

下表只审查准备和调用之间的主机工作；实际 spin/profile 乘法、dagger、cast、
stack 顺序及结果分配仍由原准备程序执行。

| 步骤 | 操作 | 必要性及处理 |
|---|---|---|
| 1 | 读取本次 loader/backend、冻结状态及 debug 选项 | 保留；决定是否可使用冻结模板程序 |
| 2 | 在同一 loader 内查找模板身份 | 保留一次；原生不可变模板可直接命中，协议对象仍核对完整描述 |
| 3 | 遍历模板的准备闭包，展开 spin/stack 叶角色 | 首次绑定保留；决定实际乘法、叶次序和嵌套 cast |
| 4 | 每实例重新遍历相同闭包 | 删除；模板程序保存角色关系，每 loader 最多16个模板 |
| 5 | 按完整数值描述共享准备程序 | 保留；相同程序可以服务不同 raw 来源，不能因此合并叶身份 |
| 6 | 建立驻留 ID 到 raw 输入 ID 的只读索引 | 每 loader 一次；消费时不再重复解析同一配方 |
| 7 | 从执行表取得真正执行的非复用 load 角色 | 保留；未消费的缺失角色不得提前报错或触发绑定 |
| 8 | 核对模板绑定的 loader 作用域 | 在实例绑定边界保留；不能误用另一 loader/backend 的程序 |
| 9 | 按实际 load 顺序查询准备变体 | 保留；缓存最多8个变体，只含角色及位置 |
| 10 | 统计叶引用和独立叶数，决定是否提前绑定实例 ID | 每变体一次；低复用仍沿既有惰性路径 |
| 11 | 为独立叶固定整数位置 | 每变体一次；它们指向本实例的 raw 键元组，不指向数组 |
| 12 | 依原阈值选择可折叠的显式重复叶 | 每角色一次；不按共享程序或恰好相同的 raw 值去重 |
| 13 | 绑定有序位置提取器、一元标志和原序展开器 | 每角色一次；避免每次创建生成器和重复叶查询 |
| 14 | 读取当前节点映射，生成本实例 raw 键元组 | 每实例保留；不能沿用前一个时间对的 ID |
| 15 | 对未知、缺失或非 spin raw ID 提前执行数值读取 | 不执行；绑定失败退回原惰性路径，在实际消费时报告 |
| 16 | 从当前角色表取得准备程序及提取规则 | 每次消费保留；未知角色返回既有 fallback |
| 17 | 按固定位置从本实例元组取得 raw 键 | 保留；无数组读取，不创建逐位置 Python 生成器 |
| 18 | 从当前 raw 表按键顺序取得实际数组 | 保留；替换块后必须使用新值，首个缺失键仍立即报错 |
| 19 | 将单输入包装为一元 tuple | 保留；不能把数组本身当作输入序列 |
| 20 | 展开已折叠叶的原有重复位置 | 保留；只少读显式重复叶，实际乘法和 stack lane 不减少 |
| 21 | 交付当前输入到原准备程序 | 保留；不缓存本次数值、owner 或指针 |
| 22 | 读取当前 shape、dtype、strides 和目标布局 | 保留；实际数组布局可变，固定指针核不能套用旧步幅 |
| 23 | 重复检查冻结内部规格和依赖闭包 | debug；可变外部表仍执行失效处理，实际布局分派保留 |
| 24 | 分配本次结果并执行原数值操作 | 保留；每级舍入、所有权和输出落点仍由既有程序决定 |
| 25 | 离开实例后在模板缓存中保存当前数组 | 删除；缓存仅保留描述和程序，实例只持有当前 raw 键 |

## DenseKernelBatch 执行

CUDA 的 `DenseGemmGroup` 在共同 `kernel_batches` 中引用 3–21 个同 dtype/MKN 的
既有 Cartesian 批。P 只选择首位置输入已驻留、无组内依赖且资源可行的批；不提前读输入。
X 一次绑定子布局，用一个设备指针表准备核和一次 cuBLAS batched GEMM 执行。
同组同一输入侧的多输入 raw 打包共用一个缓冲，每次核最多32个原数组指针；较宽
输入在绑定时分块，切点可在子批内部，直接写该子批的原有偏移。各子批的
拼接轴、排列、singleton 轴、实际 byte strides 和 dtype 转换仍按原布局执行。
目标 cube 占用等于各子批该侧原预留之和，不增加指针表或额外中间数组。若任一
子批可以直接复用连续 owner，或含 prepared 输入、debug、超出 ABI/类型能力，
保持原子批打包。缓存只保留布局/代码，每绑定最多8个 stride 变体。
正常 CUDA 执行直接把整组打包缓冲交给指针准备核，后者按已绑定的矩阵字节跨度
生成各 lane 地址，省去逐 child 的输入 view 与 reshape。左右侧独立选择完整缓冲
或原子批连续缓冲。正常路径无需将这些缓冲 reshape 为二维，矩阵尺寸已绑定到
cuBLAS 调用；debug 保留二维转换和逐矩阵审计。免复制回退、prepared 及测量标签保持。指针核只绑定可用的
侧组合与常量跨度，逐次接收当前数组；不缓存数值、扩大工作区或增加正常同步。
输出 cube 和固定 512 字节指针表由同一个 owner 持有，直至最后结果 view 释放；
两者每次共同分配一个字节缓冲，指针表占前 512 字节，cube 使用其后区域的 C 连续视图。
前缀同时保证 uint64 指针表与四种结果 dtype 对齐，实际字节等于原 cube 加指针表额度；
每子批使用三个 uint64 地址，64个槽可容纳21个子批；C/P/X共用此容量，表大小保持512字节。
输出直接借用该分配，不另建 UnownedMemory 包装，不缓存或复用跨次结果。
打包缓冲同时计入工作区。正常路径不加全设备同步，异常提交后完成清理。
cuBLAS 的固定维度、操作标志及 alpha/beta 地址在绑定时整理；调用程序同时持有
这两个主机标量，防止地址脱离其 owner。无计时回调时直接执行三阶段，不创建
临时 partial 或计时函数；显式测量保持 dense_batch_pack、pointer_setup 和
contract_kernel 三个边界。工作区仍在本次调用中分配并保持至异常完成事件之后。
当组内 child 的保存 `output_shape` 相同且非空时，绑定阶段固定结果 view callable；执行时对完整 cube 只 reshape 一次，再按已保存的 bounds 和 squeeze 直接建立各 child view。异形或空输出布局仍沿用逐 lane 路径。
cfg 完整未复用时保留组；部分复用时保留仍需计算的原 Cartesian 子批。
CPU/HIP 不自动选择这条 CUDA 实现；NumPy 低层仅用于独立数值对照。
成本状态为 `estimated-from-dense-v1`，不宣称专门校准或端到端提速；档案恢复选择，不重分组。

CuPy 的固定合批布局绑定为数组指针复制核，每次 RawKernel 最多32个输入，含目标指针共264字节；实际 byte strides 固定到生成代码，每个绑定最多8个布局变体，跨绑定代码缓存最多64项。复杂布局按实际 byte strides 读取，混合 dtype 在写入最终打包缓冲时转换。正常非 debug 的支持型 RawKernel 路径把保存的 permutation 与批内 singleton 轴编码为逻辑 shape/byte-stride 映射，直接传当前原数组指针，避免为每个输入创建 transpose/expand view。缺少 RawKernel、不支持类型或非连续目标缓冲沿用既有逻辑 view 与 ElementwiseKernel 复制路径；该 legacy 路径仍最多8个输入且参数不超过保守的3072字节。没有额外指针表、数值缓存或新的工作区；异常高 rank/不支持类型仍用既有后端路径。准备程序在每个 loader 内按常量身份、形状、类型、cast 和 dagger/排列共享（结构索引最多512项），每程序最多8个实际布局调用变体；只复用程序，输入逐次解析。冻结配方表的重复依赖扫描仅 debug 开启，可变外部表仍保留失效处理。输出程序用最多16个模板弱引用身份缓存（协议测试对象不支持弱引用时保留有界强引用），遇到新程序清空旧身份缓存，关闭时释放；深层结构比较不再逐时间对执行。

等长、同 dtype/步长且拼接维为 singleton 的 stack 打包共享固定32输入指针源码；
另传两个 int64 表示当前跨度和块偏移，连同输出指针共280字节，不上传设备指针表。
grid.y 选择当前输入，未用指针槽不访问；批宽及32路块偏移不再改变源码。
地址表达式在生成时去掉单位维，并合并满足连续步长关系的相邻维，使不同秩的连续输入
复用同一固定步长读取。其余形状/步长仍是编译期常量，不增加动态整数除法；
非同质布局沿原通用打包路径，当前数组地址、复制次数和设备分配预算保持原约定。

同 shape/dtype 的堆叠若各输入步长不同，固定32指针槽并以有符号 host 参数交付每个有效维度
的步长；shape、坐标除数及类型仍为编译期常量，span/offset沿原调用绑定。四个有效维度的
参数合计1304字节。布局缓存未命中时先去掉长度1的坐标，再仅合并所有输入都满足
`outer_stride == inner_extent * inner_stride` 的相邻维；此关系按有符号字节步长检查，
因此连续的负步长和全零广播步长也可合并，任一输入不满足就保留该维边界。
合并后两个／三个有效维度的参数分别为792／1048字节；坐标除数继续为编译期常量，
输出拼接跨度和偏移保持原定义。合并后若仍超过3072字节，保留原通用源码。
同 stride 快路径继续优先；不新增设备参数表、数组工作区或当前输入缓存。
相同合并形状的不同步长组合可复用源码；合并模式不同会增加少量代码变体。
减少的参数和地址运算不直接代表 GPU 加速，仍需冷/暖实测权衡。

raw 打包的布局查找使用当前输入的完整原始 byte strides；保存的 permutation 与
singleton 轴仅在缓存未命中时映射为逻辑步幅。普通逻辑 view 和原数组指针调用在
同一有界缓存中区分模式，避免相同步幅数值误用不同轴映射。组打包同样只在首次
遇到原始布局时排列步幅；新数组仍逐次交接，负步幅、零步幅与布局变化继续分开绑定。
每绑定最多8个变体、代码缓存最多64项；不增加工作区或缓存数值数组、owner 与指针。

单个未置换、未插入批轴的 raw 输入若 shape 和 dtype 等于绑定值且已 C 连续，
`_BatchSide.pack` 直接交付当前 owner，不再构造逻辑 shape/stride 或推导目标步幅。
这些物理布局边界始终检查；非连续、类型提升、prepared、置换和插入轴保留原路径，
debug 保留完整布局解释。它不改变打包额度、输入生命周期或结果，也不缓存数组。

普通 `anytime` 计划在完整固定 DAG 存储排程和审计后，可能带有 `SceneStoragePlan.kernel_batches`。
P 可将有多个自由尺寸变化的完整 Cartesian 网格按其它自由尺寸拆成不重叠子网格，
各侧仅留最后一个变化轴供现有拼接布局使用；每个子网格独立通过原成本和资源准入。
这些仍是既有批注，X 不划分网格或改选实现。
`replay_scene_storage` 消费这些已保存批注：到达批次首个 Action 时执行一次 `operations/dense_batch.py::contract_dense_batch`，
并在后续成员 Action 处复用已发布的结果。批注只索引原有 paid binary Actions，不构成第二套计算 IR；
`replay(state)` 没有 storage plan，因此仍逐个执行逻辑二元 Action。
`stacked_sides` 将保存的单例轴纳入逻辑 shape，再沿该轴打包，仍使用同一个
矩阵核。结果按保存的 `squeeze_axes` 去除批内轴，保留逻辑 shape/dtype；这些结果
仍是完整输出 owner 的 view，释放与资源记账沿用原有规则。

NumPy 执行先按保存的轴排列准备批输入。每侧有多个输入时沿自由轴调用一次 `concatenate`；单输入侧直接借用转置视图。
仅当左右各有一个自由轴且两侧连续时使用 GEMM，否则使用 `einsum(optimize=False)`。
CuPy 计划保存 `packing_dtype`，两侧输入都以 C 连续、结果 dtype 的布局进入 GEMM。多输入侧先分配一个目标
缓冲，再用已绑定复制核完成拼接和 dtype 提升；不支持固定核时回退 `concatenate(..., out=destination)`；单输入侧复制/置换到目标布局。
若某侧由 `prepared_sides` 标记，则它的 producer 已按消费者要求的轴顺序生成连续的结果 dtype 输出，消费者直接
借用所需 view，保持 producer 原路径。借用前核对连续性和 dtype，失败时记录张量名、实际 shape/strides 与预期 dtype。
完整 cfg 的输出排列取自完整模板，而非当前实例留下的子批集合；部分复用跳过合批也不能
撤销生产者的布局承诺。P 已将该布局纳入全局 owner 复用身份，同布局继续共享，不同布局分别计费。
其它输入仅在当前原数组的 dtype、root owner、指针次序、映射后的实际 shape/strides、完整范围与边界
证明已是连续拼接时复用最终视图；独立 owner、负步幅、间隙、重叠或类型提升仍走原打包路径。
二元 GEMM 与批核的形状、排列、fallback 整数轴、结果切片一次绑定，恒等 transpose/reshape 不再调用。
普通内置 dense Action 在固定执行表中直接进入同一存储调用，不再经 `_numerical` 只转交
`context[0]`；调用仍从本次 context 的当前数组映射解析输入，不捕获数值数组。
不保证独立所有权的结果仍复制。自定义 operation 保留原转交边界，dense batch 和 sparse/baryon
也保留各自路径。每次从当前映射解析输入，
不捕获数值数组。有序输入键在绑定期整理为提取器，逐次从当前映射读取，
不再为每个操作创建输入生成器；重复输入仍按原序访问，单输入仍交付一元 tuple。
完整 cfg 的原生共享数组视图在每次回放开始时提供当前全局数组表和节点绑定，
二元 dense 调用直接查这两张表，省去每个操作的两次读取和一次写入的 Python 映射转发。
延迟批结果仍在首次实际消费时展开，重复输入、当前值替换、输出快照及资源事件顺序保持一致。
扩展视图或映射保留其原有读写协议；此优化同时用于 CPU/GPU，不缓存数值或跨运行的数组表。
`debug`/`measure` 及注入的自定义 operation 继续走原有操作调用。
同一模板和 backend 内，普通 dense 的完整整数连接、有序输入 shape/dtype 和 producer
输出排列相同时，只绑定一次数值程序。每个 Action 仍单独绑定自己的输入/结果键并
逐次读取当前数组；不合并计算、数据来源、结果或生命周期。输入的物理身份核验
继续在原边界执行。带 core 的 Action 保留逐 Action 绑定，合批仍按完整 batch 记录绑定。
缓存只含程序和不可变描述，数量不超过该模板的 Action 与保存 batch 数；模板或 backend
改变时仍沿原有失效规则，不跨 cfg 留存输入、结果、地址或工作区。

### 绑定程序的逐步审查

以下从 `_bind_cfg_tables` 拆到 `bind_scene_execution` 和 `bind_program`。
“保留一次”表示在相应模板、复用状态或事件变体首次出现时完成；每个时间对的
实际数组仍在执行时读取。缓存查找用于省去重复绑定，不替代数据或资源操作。

| 步骤 | 操作 | 必要性及处理 |
|---|---|---|
| 1 | 取得本次缓存并核对 backend 身份 | 保留边界核对；数值程序捕获 backend，不能交给另一个 backend |
| 2 | 为本窗口建立状态表、执行表、模板索引和实例引用 | 保留；只发布本窗口会使用的变体，避免累计历史窗口 |
| 3 | 按保存顺序读取各坐标实例及模板编号 | 保留；复用状态和事件属于具体实例 |
| 4 | 首次遇到模板时取得完整 keys、state 和 batch 描述 | 保留一次；这些字段共同决定程序语义，模板编号不足以证明相等 |
| 5 | 与上一窗口的模板作完整相等比较 | 每模板一次；兼容独立解码得到的等值对象，不能只比较 shape |
| 6 | 对每个实例再次深度比较同一模板 | 删除；本窗口模板索引复用步骤 5 的结论 |
| 7 | 交接相等模板已有的数值程序缓存 | 保留；其中没有 reader、数组、地址或工作区 |
| 8 | 以保存的 reuse mask 查找状态变体 | 保留；不同复用状态执行的 Action 不同 |
| 9 | 新状态解码一次 mask，并按 Action 顺序应用 reused/seconds | 保留一次；同一原生模板有至少三个未缓存模式时，只提取一次 reused/非零 seconds 标志，再按差异位置替换；debug 和扩展字段沿原逐项路径 |
| 10 | 对未改变的 Action 或完整 state 再造副本 | 删除；沿用原不可变对象，只替换实际变化的记录；原生多模式不逐模式用 Python 扫描全部 Action |
| 11 | 读取当前 batch 成员和事件的位置、种类 | 绑定时保留；实例查找键中的原生 GEMM 组成员只在本次绑定展开一次，debug/扩展组逐次取属性；原生实例回放直接取已定表引用 |
| 12 | 按状态、完整 batch 与事件模式复用已有执行表 | 保留；事件表只含位置和种类，实际块编号由本次执行提供 |
| 13 | 对已有执行表重新绑定全部调用 | 删除；同一变体直接交付原表 |
| 14 | 新变体建立 batch 首项表和成员表 | 保留一次；批核只在首项执行，其余成员不得重复计算 |
| 15 | 从 prepared sides 取得 producer 输出排列 | 保留一次；相同逻辑 shape 不代表相同物理排列 |
| 16 | 筛出未复用、未包含在 batch 内的 contract Action | 保留；已有步骤 9 的完整原生动作证明时，从当前 mask 取得未复用序号并按原顺序绑定，省去再扫已复用动作；序号只在本次绑定存活，debug/扩展字段完整扫描 |
| 17 | 为同一共享驻留变体再次扫描不会产生调用的 Action | 普通原生路径删除；复用核绑定时收集的操作位置，并按原顺序补回全部事件及首项推进 |
| 18 | 读取普通 dense 的完整整数轴连接 | 保留一次；不凭节点编号、轴数或归一化后猜测等价 |
| 19 | 按输入顺序读取完整 shape 和 dtype | 保留一次；交换输入、改变尺寸或精度都可能改变程序 |
| 20 | 把 producer 输出排列加入普通 dense 缓存键 | 保留；不能把不同排列的输出交给同一绑定程序 |
| 21 | 按 Action 编号分别绑定相同的普通 dense 数值程序 | 删除；步骤 17–19 相等时共享程序，每个 Action 仍绑定自己的输入/结果键 |
| 22 | 对 core Action 仍使用 Action 编号和输出排列 | 保留原路径；不能把 core 的额外科学描述省去 |
| 23 | 程序未命中时绑定缩并和布局，并写入模板缓存 | 保留；只保存程序和描述，不捕获当前数组 |
| 24 | 将程序关联到当前 Action 编号 | 保留；共享程序不合并 Action、计算次数或结果落点 |
| 25 | 按完整 batch 记录绑定或复用批程序 | 保留首次/等值描述查找；本次同一模板内，字段已证明不可变的同一原生 batch 直接交接已取得程序，省去重复完整哈希；debug/扩展字段沿原查找 |
| 26 | 按保存顺序绑定 load、contract、reuse、move、release 调用 | 保留；不可为了减少主机调用跨越数据依赖和生命周期 |
| 27 | 内置普通 dense 再套一层只取 `context[0]` 的转调 | 删除；直接绑定同一存储调用，自定义 operation 保留原交接 |
| 28 | 绑定首项推进及 Action 前后的读入、释放事件 | 保留；无首项读取时也要推进连续队列，末尾事件在输出快照后执行 |
| 29 | 绑定有序输出键和适用的固定输出布局 | 保留一次；结果顺序和独立快照必须与保存计划一致 |
| 30 | 每个时间对重新解释整份调用表 | 普通路径删除；debug/measure 保留独立解释入口 |
| 31 | 独立重放资源审计和重新验证固定内部布局 | debug；正常流程消费已经审定的保存计划，实际输入边界核验仍执行 |

已绑定 dense 存储的主机步骤进一步拆分如下。它消费已经选定的 Action 和 kernel，
不改变缩并本身，也不在内部交接增加默认校验。

上述位置复用只适用于无 retained 交接、原生 Action 及冻结整数事件的共享驻留程序。
debug、扩展 Action、可变事件及普通 retained 恢复仍扫描原表；末尾事件仍在输出快照后执行。
临时位置表仅在本次绑定内存活，不进入档案或形成另一份执行排程。

| 步骤 | 操作 | 必要性及处理 |
|---|---|---|
| 1 | 从 Action 读取有序输入键和结果键 | 保留一次；键决定实际数据及结果落点 |
| 2 | 绑定单输入的一元 tuple 或多输入提取器 | 保留一次；kernel 接收的输入 tuple 结构必须一致 |
| 3 | 每次重新创建输入生成器并逐次恢复生成器帧 | 删除；绑定提取器直接按同一顺序访问 |
| 4 | 从本次映射读取每个输入 | 保留；不得复用上一点的数值或实例映射 |
| 5 | 重复输入键按其出现位置再次读取 | 保留；不改成集合，也不改变映射访问及异常顺序 |
| 6 | 将本次有序输入 tuple 交给已绑定 kernel | 保留；这是实际数值调用的交接 |
| 7 | 逐点重新整理已绑定的方程和布局 | 普通路径不执行；debug/measure 沿原有数值操作入口 |
| 8 | 执行已选 kernel 并接收结果及独立存储标志 | 保留；数值和所有权决定后续交付 |
| 9 | 对不保证独立存储的结果创建快照 | 保留；否则输出可能随借用输入或可复用缓冲改变 |
| 10 | 对 kernel 已保证独立存储的结果再次复制 | 删除；直接交付该结果对象 |
| 11 | 数值计算及必要快照成功后写入结果键 | 保留；失败不能发布不完整结果 |
| 12 | `_numerical` 提取 `context[0]` 后再转交内置 dense 存储 | 删除；直接进入同一存储调用，不增加前端参数 |
| 13 | 为普通调用构造计时 partial 和空计时闭包 | 删除；自定义 operation 保留原调用边界，合批及 sparse/baryon 保留各自路径 |

已绑定合批在没有计时回调时直接调用两侧打包和缩并，不逐批构造空计时闭包；
有回调时保持原 `dense_batch_pack` / `contract_kernel` 边界和结果发布原子性。
已保存的 `squeeze_axes` 可在一次索引中用单例 bounds 的起点消去对应轴；零维
NumPy 结果保留原数组 view 路径。`PackedOutputs` 的非标量 lane 直接用整数索引，
避免先取长度一的切片再 reshape，仍由独立 snapshot owner 支撑完整 tuple 结果。
绑定的固定输出布局一次保存 shape/dtype 分组。CuPy raw 复制每块最多128个原指针，
加上一个 int64 输出偏移和一个目标指针共1040字节；分块同时服从原3072字节参数预算。
legacy Elementwise 复制仍最多16路并沿原CArray参数预算，不增加设备指针表或缓冲。
CuPy 支持的
类型用 RawKernel 按当前输入的 byte strides 做精确元素复制，可处理标量、空数组、
负步幅和转置视图；每个绑定布局最多保留 8 种 stride 变体，进程级代码/工厂缓存最多
64 项。每次回放仍从当前结果映射解析数组，不缓存数组、指针或设备状态，也不增加
工作区或同步。内部固定结果的重复 shape/dtype 校验只在 debug 开启时执行；直接调用
`bind_output_layout` 默认仍校验。debug、float16、不支持类型或缺少 RawKernel 时沿用
原 ElementwiseKernel/逐 lane 复制路径。此实现预期减少受支持 CuPy 输出快照中的
ElementwiseKernel 调用与逐元素索引工作；批宽的设备收益需单独测量，CPU协议不能证明GPU加速。
关闭 layout 校验与 execution debug 的内部 CuPy 快照使用仍为 tuple 的私有
`PackedOutputs` 子类，携带当前 `BoundOutputLayout` 的只读来源；当前快照持有该布局，输出程序缓存仅保存弱身份。绑定输出程序首次
接收该来源时仍完整检查 term 数量、backend、shape 与 dtype，并保存分组 owner schema；
同一 layout 身份的后续调用改核验 owner 数量及每个 owner 的 backend、`(lanes, *shape)`
和 dtype。身份缓存最多 16 项，只含弱 layout 身份及纯 schema，不持有窗口、数组或指针。
外部 tuple/`PackedOutputs`、NumPy、debug 与 identity miss 保持逐 term 检查；序列化回到
普通 `PackedOutputs`，不把程序或布局身份写入计划。
不保留跨次数组缓存；已保存 packing workspace 和模型成本保持保守上界。producer 必须是已有的同 dtype 普通二元求和 Action，且 C 的计划验证会排除批 owner 输入和
冲突布局。producer 的逻辑 Action 及结果轴保持不变，没有新增计算操作。

一次批核产生完整 C 连续输出，再把各结果切片作为 view 一次性发布；最后一个 view 释放前，完整输出 owner 仍计入
资源峰值。规划将输出 owner 和未准备侧的打包缓冲分别计入 NumPy host 或 CuPy device 轨迹。失败时不会发布一部分结果。
测量在批首成员记录实际合批工作，其余 paid 成员的物理核计数为零。批计时包含 pack 与缩并，报告标记
`calibration_eligible=False`，不把它作为普通单核成本拟合样本。

批注保存在固定 `storage-schedule` 档案中，加载会验证结构和资源轨迹，并按当前 blocks/preparations 重绑，不重新搜索或合批。
cfg 模板保留 `SceneStoragePlan.kernel_batches` 批注，作为已选执行描述的一部分；加载和回放不重新合批。
带批注的计划要求执行 backend 与计划 backend 相同，因此 NumPy 计划不能交给 CuPy，反之亦然。
旧专用 `matrix_device_execution` scope 已废弃；低层矩阵 kernel 可继续作为计划中的操作实现，但不构成独立设备执行路线。

### 输出快照交接的主机步骤审查

固定输出分组一次绑定可序列化的键提取器；普通映射按原序读取当前值，
支持延迟输入的内部复制通过既有批量驻留协议读取当前 owner。
批量读取缺失时退回原逐项查询，保持首个缺失 local key 或驻留 owner 的错误。
外部旧协议保留原逐项读取；debug 仍使用原数组视图和完整布局核验。
提取器只持有输出键，不持有数组、实例映射、指针或设备状态。

| 步骤 | 操作 | 必要性及处理 |
|---|---|---|
| 1 | 从保存布局取得输出分组及键顺序 | 保留；分组决定复制形状、类型及输出位置 |
| 2 | 为每组绑定键提取器和单输入标志 | 每布局一次；普通路径不逐点构造输入生成器 |
| 3 | 读取本次 debug 及公开布局校验选项 | 保留；决定参考路径与外部输入检查 |
| 4 | 从当前映射按序读取有序值 | 保留；不能复用前一点的数组或驻留 ID |
| 5 | 对内部驻留映射逐键调用 raw getter | 正常命中时删除；使用已有批量协议查询本实例的当前 ID 与 owner |
| 6 | 在缓存中保存本实例的 ID、输入值或指针 | 不执行；布局只保存不变的键提取规则 |
| 7 | 保留重复输出键和原顺序 | 保留；不能将输出序列改成集合 |
| 8 | 将单输出包装为一元 tuple，空分组交付空 tuple | 保留；不能把标量或数组当作有序输出序列 |
| 9 | 批量查询缺失时再次按原顺序逐项查询 | 仅错误回退保留；较早缺失的 owner 不能被较晚缺失的 local key 遮盖 |
| 10 | 重复核对冻结内部结果 shape/dtype | debug；公开布局默认仍逐项核对 |
| 11 | 分配本次独立快照 owner | 保留；源结果释放或修改不能改变已交付结果 |
| 12 | 执行既有复制程序或逐 lane 回退 | 保留；实际 byte strides、dtype 和资源额度仍按原程序处理 |
| 13 | 将本次 owners 交给原 tuple 或延迟快照 | 保留；位置表及 provenance 仍引用同一保存布局 |
| 14 | 在布局对象中保留本次输出 owner | 不执行；owner 随当前快照生命周期释放 |

## 跨边界 reader 协议

完整 cfg 的冻结 preparations/specs 在 `SavedPhysicalLoader` 恢复时保持原引用。
X 合并两张冻结表时保留全部组件并冻结结果，显式表优先；任一输入表可变时仍使用
普通合并字典和既有失效处理。该接线仅复用描述，不跨窗口或设备 scope 共享数值常量。
冻结的 CuPy spin/stack 配方另按 kind、完整常量、shape/dtype、dagger/排列、cfg 和
有序成员程序复用已有 `FusedSpinPreparation`。每个配方 ID 保存当前 raw 输入 ID 元组；
运行时解析当前数组，不重新构造成员操作或逐 ID 常量表。结构索引最多 512 项，
每程序沿用 8 个布局变体；debug、可变表和不支持的结构走原有准备及失效路径。
冻结结构程序在支持 `RawKernel` 的 CuPy 后端可使用固定 stride 的指针参数，保持
每次至多 32 个 leaf 与原数值表达式、类型舍入、3 KiB 参数预算和既有输出分配；
legacy CArray 路径仍至多 8 个 leaf。
代码缓存最多 64 项，只缓存代码和工厂；布局变体不持有输入数组、结果或指针。
异常布局或缺少该 ABI 时沿用 ElementwiseKernel，CPU/debug 路径不变。
调用缓存的查找与保存使用同一份 raw 输入 dtype/shape/stride、目标布局与 ABI 键，
和生成核使用的 raw/常量数组签名分别命名。相同布局的新输入直接启动已绑定调用，
不重算叶元数据；实际输入和结果 owner 均不进入缓存，debug 仍重走描述检查。
已绑定 raw proxy 的 byte strides 已包含 recipe 的轴排列，命中时传入原数组指针，
省去逐输入 transpose 视图；legacy ABI 保留逻辑视图。异常 backend 若在命中后分配
非连续输出，则用该已分配 owner 回退 legacy 绑定，不把 raw 核用于错误输出布局。

完整 `CfgTemplate` 携带冻结的 preparation closure/spec 时，CuPy prepared loader
在自身 backend scope 内按 template 对象身份和完整不可变描述，为每个 local prep role
绑定一次同一套 spin/stack 程序及有序 leaf local keys。每个实例仍从自己的 local-to-global
binding 和当前 frozen preparation recipe 解析 raw block ID，再把当前数组交给程序；不按
每个 cfg node ID 建立程序，也不保留数值输入。该模板绑定缓存最多 16 项并强持有描述身份，
已有 loader 内完整语义程序缓存仍负责跨结构等价 role 复用。缺少完整冻结描述、CPU、debug、
可变表或不支持的准备类型均保留原 materialize 路径。
X 从已接受的 `IntArrayRecord` 创建当前整数 node binding 时，以私有类型标记该交接；
至少四叶的模板角色预绑定有序索引提取器，直接经过只读整数 node-to-raw 索引提取本次输入。
它保留原顺序、重复叶及首次消费错误，不缓存当前数组或指针，也不在每次准备中把整数 ID
转成字符串。单叶原生角色直接以唯一 local key 查询同一数字索引，不创建索引提取器；
外部映射、非规范数值名称、二至三叶角色和 debug 沿原解析；高复用实例绑定仍优先。

    SourceLoader.read_block(block_id: str, block: StorageBlock) -> object
    SourceLoader.materialize(
        tensor_id: str, source: SourceSpec, blocks: Mapping[str, object]
    ) -> object
    SourceLoader.close_block(block_id: str, value: object) -> None
    SourceLoader.close_all() -> None

close_block / close_all 只释放该 loader 为本次执行持有的块或句柄；它们不关闭调用方持有的 reader。materialize 只能使用本次显式传入的块；返回视图可以借用这些块。可选扩展协议为 BatchLoader.activate(batch_index: int) -> ContextManager[None]、MeasurementAwareLoader.set_measurement(enabled: bool) -> None 和 IOReportingLoader.drain_io_events() -> Iterable[Mapping[str, object]]。它们不改变核心 reader 协议。

C 的正式 SourceLoader Protocol 包含以上四个方法。当前 X 的兼容校验只要求 `read_block` 与 `materialize`，并把 `close_block` / `close_all` 作为可选钩子；这允许 loader 返回已拥有的数据，并在返回前关闭临时读取资源。`NpySceneBlockLoader` 复制所选 NPY 数据，在 `read_block` 返回前关闭 mmap，属于这一类。该兼容行为不豁免为本次执行跨调用保留资源的 loader：若 loader 自身为该执行保留缓存数据、文件句柄或其它任务级资源，就应通过适用的关闭钩子释放；调用方持有的 reader 仍不属于钩子释放对象。例如 `_BatchTaskLoader` 在 `close_block` 和 `close_all` 中清理自身的任务缓存。回放进入执行后，在正常或异常退出时会尝试可用的 loader 级 `close_all`；多批回放还会逐一尝试剩余 raw block 的 `close_block`。单计划的 `close_block` 回调若失败，当前释放循环会提前结束，随后仍会调用 `close_all`，因此不承诺该路径为每个剩余块调用一次 `close_block`。已有读/核异常优先于清理异常；无主异常时第一个清理异常传播给调用方。`tests/correlator/execution/test_loader_cleanup.py` 通过公开存储回放入口验证这些清理边界。

## X 的入口

### 固定调用序列

P/A 已保存的 Action、合批、复用标记和 StorageEvent 是唯一执行安排。
X 的 `bind_scene_execution(..., events=..., retained=...)` 在执行设备上一次绑定
`BoundSceneExecution`：将操作类型分派、合批成员折叠及无事件生命周期入口消除。
`bound_program.BoundProgram` 只持有数值函数、计划常量和输出键，不持有运行数组、
reader 或流；不是另一个可保存 IR。完整 cfg 按模板、复用 mask、合批成员和事件位置
共享绑定，实际 block ID 和输入仍由当前实例提供。跨 cfg 不共享数组。

`debug=False, measure=False` 顺序调用绑定函数；只在首操作及物理读取处推进预取，
只在保存的释放点执行释放，合批成员不再经过诊断检查。输入 shape/dtype、稀疏边界、
资源额度、事件顺序及异常清理仍保留。`debug=True` 使用逐 Action 路径作额外核验；
`measure=True` 独立保留完整 Action/phase 测量，并按需读取已存的逐操作成本。
正常执行不再整理测量成本表。绑定本身不初始化目标之外的设备，不重排、搜索或预测。

连续有界队列中的读取及上传仍与当前计算重叠；已计入预读额度的上传在首次消费时
由 compute stream 等待对应 event，主线程不在正常 ready 路径同步该 event。CuPy Event 用实际 `done` 属性查询完成；兼容事件类的 `query()`，只有两者都缺失的旧协议才退回阻塞完成。接口按类型有界绑定，不捕获事件或输入owner。已消费但尚未完成
H2D 的 ticket 保留 pinned staging 和预读额度，完成查询确认后才回收；额度压力时仅等待
最早 ticket；已消费的延后 ticket 元数据另有32项上限，达到时也只等待最早项。
整个 host/device 预读预留都保留到完成，不提前归还其中一域。关闭时先排空上传再释放 owner；
输入释放仍让上传流等待末次计算 marker，防止池复用过早。未获延后 staging 额度的直接读取、
finish_prefetch 返回新 host 的重上传、缺少完成查询能力的兼容后端保持安全同步。
数值准备沿已保存 backend/current stream 执行，不引入异构后端选择。结果资源队列见下节；
真实 GPU 吞吐与生产规模内存仍需针对本次源码重新测量。

### 按资源消费结果任务

`execute_cfg_plan` / `execute_cfg_windows` 的回调路线，以及 `iter_execute_cfg_plan` 的直接迭代
和惰性多 cfg memory 前端，共用同一双槽结果队列：IO 使用原有有界
ReadAhead 队列；已保存的 GPU 准备/收缩按原 compute stream 提交；CPU 结果队列按坐标顺序
消费完成下载的快照、组装/累加或写出。它不创建第二套图，也不在 CPU/GPU 之间抢占或重新选择核。

普通固定 GPU 程序在绑定时按相同 shape/dtype 分组结果，每组分配一个连续 owner。适用 CuPy
打包融合核每次最多复制16个结果，并按保守3072字节参数上限进一步分块；参数携带 CArray
指针/stride，不上传临时指针表，不分配中间快照。支持 strided、scalar 与 empty 来源；
每组只下载一次，dtype 不提升，原 term 次序与 axes 保留。CPU 交付仍是普通 tuple，
各数组是独立 snapshot owner 的不重叠 view；后续计算、输入释放和下一次输出均不能覆盖它们。

仅普通非 debug 路线、backend 支持异步 ndarray.get，且保存的 host/device 预读余量及实际 worker 上限都容纳
`2 × 最大单坐标下载 owner 的逐个 512 B 对齐字节数` 时，才预留该额度并相应收缩 IO 预读额度。
普通固定回放按唯一 key 的相同 shape/dtype 分组计 owner；不会把每个标量 view 当成独立分配。
为沿用同一队列额度接口，host 也保守预留这个 device 数量；这不表示 pinned host 使用同样对齐。
P 已在本坐标输出阶段计入当前快照，X 另预留完整两槽；这部分保守重叠不代表实际第三次分配。
原生不可变计划的 owner 描述在队列作用域内只保留一个当前计划，统计字典不作为 owner 证明；
每次配置仍重新检查当前 host/device 上限和剩余预读额度。换窗口替换描述，扩展可变计划重新描述。
两个结果任务上限同时覆盖排队项和执行项；生产前背压，下载流等待 compute event，CPU worker
只等待对应下载 event。无需逐 Action 全设备同步。额度不足、NumPy 或 debug 路线保持同步，
不改变计划或自动超额。直接 `iter_execute_cfg_plan` 最多提前生产两个尚未交付的坐标，
暂停迭代就形成背压；不会因慢消费者继续产生完整未来 cfg。已经交给用户并由用户长期保留的结果仍由调用者负责。

回调可能在唯一 CPU 结果线程调用，顺序保持不变；完成通知在所有结果成功后调用，cfg/window
reader 关闭之前先排空队列。回调、读入、核或传输失败会取消未启动消费、等待在途工作并保持
主异常，未成功的 cfg 不发布完成。直接迭代器不把尚未完成的传输 ticket 暴露给用户。
`iter_execute_cfg_plan(..., transform=None)` 可选纯 CPU `transform(cfg, index, host_values)`：
仅对实际坐标调用，返回值成为该坐标交付的 values；它可能在唯一结果线程执行，不能访问
reader 或 CUDA 状态。无 transform 时仍交付原 tuple；cfg 完成仍为 `(cfg, None, None)`。
惰性多 cfg memory 前端通过这个交接在后台组装单点，对外依旧按顺序返回完整 `(cfg, result)`。
提前停止必须 `close()`，或用 `contextlib.closing`。原始 `iter_execute_cfg_windows` 逐点迭代
仍同步交付；其既有回调 facade 继续自动重叠，不另外叠加队列。

报告含 `output_queue_tasks`、`output_queue_peak_tasks`、`output_queue_wait_seconds`、
`output_buffer_host_bytes` / `output_buffer_device_bytes`；device 子报告含实际
`output_download_calls`、`peak_pending_outputs` 和 `output_download_wait_seconds`。
这些是调用次数、保守预留字节额度和主机墙钟，不证明设备利用率或加速。

同一次操作可以复用已经接纳的确切不可变计划绑定；更换模板/实例、可变描述、
公开 Python 构造和独立档案加载仍验证，输出布局/预算变化也继续检查。Meson 保存/打开
分别完成自己的语义核验与一次通用公开档案核验，不跨独立调用跳过外部校验。
整数 planner 的完整表达式/碰撞诊断归 debug；执行动作重建、完成/成本及资源可行性始终保留。

已选 GEMM 组的指针设置核复用 `raw_arguments` 的固定参数槽和 driver/CNumpyArray ABI。
每个打包 mask 在冷绑定时确定矩阵、当前指针表及新结果的参数位置；调用后包括异常路径
都清空动态地址。正常执行仍使用当前设备 Function 和当前 stream，不持有数值输入或历史
指针，不增加设备工作区或同步。debug、自定义核与缺少该 ABI 的后端保持原 RawKernel 参数交付。

### 路径执行

`FactorizedWideReduction.tile_elements` 非空时，布局绑定同样复用 `raw_arguments`：
每种 byte strides 最多保存完整块和尾块各一对绑定。单 tile 的起点/计数固定；多 tile
的起点通过 `scalar_positions` 指定的 signed-int64 参数槽填写，计数仍为模板常量。
各调用线程独立保存主机参数表，动态地址与起点在提交前写入；CNumpyArray 回退在成功
或抛错后恢复动态槽，driver 路线沿现有覆盖后提交的约定。绑定不保存数值数组 owner。
没有按 tile 数预建 wrapper，没有新增设备分配、同步、前端选项或档案字段。
旧 `tile_elements=None`、三/四输入角色、负 strides、保留坐标 lane、complex128 跨块
累加与原有 workspace 准入保持原语义；不支持该 ABI 的核沿普通 RawKernel 参数交付。

正常 native Action 的 thin-wide 数值工厂在既有模板 `kernel_cache` 中按完整
`FactorizedWideReduction` 值及 backend 身份共享；方程的整数轴关系、输出顺序、有序
shapes、large operand 角色、dtype 和 tile 字段均参与身份，不按原始 tensor 名字合并。
每个 Action 在缓存查询前重新核验其方程、wide 标记、全部输入规格及输出 shape/dtype；
非空 producer 输出布局仍拒绝。不同数学 core、backend 和独立模板不自动共享。
debug 与扩展 Action 保留原逐动作路线。

CUDA 工厂只保存 core、后端和解析出的标量/shape/dtype；`blocks`、partial owner 大小、输出
shape、tile 宽度/数量和 workspace 等在绑定时冻结，不在每次调用或每个 tile 递归求值。
每次输入 shape/dtype 校验、实际 stride 读取、当前地址绑定和新输出分配仍执行。
共享布局缓存不超过64项；并发调用即使淘汰布局，已取出的局部调用仍有效，动态主机槽
继续由既有 thread-local ABI 隔离。工厂不持有数值输入、部分和、输出或历史 cfg owner。
这项变化不改 CUDA 源码、编译选项、求和次序、设备预算或档案格式；首次工厂绑定、
首次布局/JIT 和稳定期 `.run()` 应分别计时，不把某一模块成本当成全部启动时间。

    @dataclass(frozen=True)
    class BoundContractionExecution:
        expanded: ExpandedCorrelator
        path: PlannedContraction
        debug: bool = False

        def execute(self, *, backend: object = numpy, measure: bool = False,
                    io_overlap: bool = True) -> PathExecution: ...

    execute_contraction_path(
        expanded: ExpandedCorrelator,
        path: PlannedContraction,
        *,
        backend: object = numpy,
        measure: bool = False,
        io_overlap: bool = True,
        debug: bool = False,
    ) -> PathExecution

`debug=False` 是执行入口默认值。普通模式保留必要的 descriptor identity、backend、可行性、loader 和输出覆盖检查；完整 `validate_execution_binding` 深度重放只在 `debug=True` 时执行。`Calculation.run(..., debug=None)` 和 `CompleteCalculation.run(..., debug=None)` 使用本次 Calculation 的临时 debug 选项，显式传值只覆盖本次调用；debug 不写入计划档案。它只控制重复深度审计，不改变测量、资源依赖、cleanup 或设备同步。

入口按 `execution_mode` 分派。empty 只返回空 PathExecution；direct 调用固定顺序 direct 核；planned 按已审计的 SceneStoragePlan 回放。非空计划要求完整状态和匹配的展开身份。执行期不补搜索、改动作、换预算或自动重试。单计划扫描只接受一个 CoordinateBatch；带多批的 PlannedContraction 使用 P 已经全局连接物理块生命周期的 batch_paths。direct task batches 同样走共享块回放，并用 loader.activate(index) 选择对应子批；每个结果按 batch_output_axes 拆 lane、复制到稳定值，再按 CoordinateSpec 的点 key 验证恰好覆盖一次。

内部 `_validate_term_values(expanded, term_values)` 的完整 term×coordinate 覆盖、重复与缺项审计仅在 `debug=True` 执行；正常执行沿已验证的固定计划交付，不重复枚举全部预期坐标。外部输入 shape/dtype、COO/坐标错误、浮点/复数输出 NaN/Inf、当前资源额度和可变 reader 绑定仍在所属公开边界检查。每次 execution binding 仅接纳同一个计划一次；direct 批次边界检查执行一次，内部 retained-after 集合相等核验也归 debug。

### 预算审计的 direct 路径

    execute_direct_contraction(
        expanded: ExpandedCorrelator,
        path: PlannedContraction,
        *,
        backend: object = numpy,
        measure: bool = False,
    ) -> DirectExecution

P 必须先通过 plan_direct_execution(...) 为固定的 bridge-first、其余固定顺序动作构造完整 SceneStoragePlan；缺 preparation/block 或预算不可行均在第一次 loader 调用之前失败。direct executor 消费 path.plan.state.actions，不再用无限 budget 重建状态。BaryonJoinSpec 同样由 P 构造固定动作；X 以 `no_search=True` 执行，稀疏核不调用路径工具，不适用时逐给定 `kernel_path` 调用已有 dense 数值核（GPU 适用矩阵乘，参考 `einsum(optimize=False)`），不把五输入退化为一次全指标枚举。单批 direct 一次读取计划需要的物理块和逻辑源，直到输出快照完成后才释放其 owner。多批 direct 则使用 join 后的事件表，跨 child 复用 raw block 到最后需求点；direct child 的保留逻辑张量为空。

运行实测超过目标时报告偏差但继续；真实分配 OOM 与数值核异常原样传播。不支持项会显式报错：没有 CoordinateBatch 的 direct 扫描，以及依赖 device-native COO 的稀疏操作。没有稠密化点记录或跳过预算的 fallback。

## 原子操作签名与保证

CPU generic/retained-point 稀疏分支在同一次调用内复用已经检查并规范化的 lane，
不再沿内部 join 重复扫描同一索引边界；独立公开 join 调用仍完成输入校验。
这是重复扫描减少的结构性保证，不代表已经实测 GPU 加速。

GPU 原子核可以在保持操作契约时独立替换实现：dense 二元操作复用
`lower_binary_matmul` 的轴配方，适用且含共享求和指标时执行矩阵乘；外积、逐元素乘、
trace 等参考路线通过有界纯方程缓存跳过降低准备，不缓存数组或设备。不支持的方程及 CPU 路线
保持原 `einsum(optimize=False)`。这不重新搜索缩并顺序，planned 的结果快照与
direct 的 `copy_result=False` 保持原语义。GPU 矩阵乘结果独立于全部输入时，
`result_snapshot` 阶段直接接管新分配结果，报告复制字节数为0；结果可为持有新 product
存储的视图，后续操作不会覆盖其数据。普通 einsum 与 CPU 路线仍复制快照。
默认 dense 工作区预留输入总字节加
结果字节，另计输出存储，覆盖两输入布局打包和结果快照；CuPy 内存池观测不包含
CUDA/cuBLAS 原生工作区。正式存储计划重放动作的默认工作区来审计峰值。

已绑定的非公共批轴 CUDA GEMM 和 Cartesian 批可用当前指针直接调用同一个 cuBLAS
矩阵核，省去二维数组 header。绑定阶段固定原轴尺寸、C/F 展平步幅、leading dimensions
及两份主机标量常数；每次只接受当前 shape/dtype/stride 匹配的布局，不保存输入或设备状态。
结果按已保存的自由轴 shape 独立分配，输出轴排列仍保持。原张量不能直接展平时，CUDA
可复用已有 raw 打包服务，按保存的排列和当前原轴步幅直接写入二维矩阵，省去输入
transpose/reshape header。只有需要打包的一侧分配原额度内的副本，另一侧按已证明的
C/F 步幅直接交付当前 owner；含单例轴、缺少原指针打包能力或异常 shape/dtype 沿原转换。
每个打包绑定仍至多缓存 8 个步幅变体，不缓存数值 owner 或指针。打包后的同一 GEMM
保留原输出布局；若当前精度模式不允许私有 GEMM，则把已打包矩阵交给公开矩阵乘，
不再重复打包。没有新准备依赖、融合核、持久数组或工作区。debug、公共批轴、CPU及HIP
沿原数组实现；CuPy 的非默认 compute type 或 cuBLAS math mode 保持其精度策略。
异常后才为所有权清理同步，正常调用由 CuPy GEMM wrapper 绑定当前流。

Baryon join 的 256 项有界缓存只保存 equation、shape、dtype 和显式 kernel_path
对应的表达式，不绑定常量数组、设备或 stream。首次表达式构造计入核阶段，
结果快照和 reused 短路保持。外部后端接收全 NumPy 输入时沿原固定路径调用，
避免表达式转换缓存持有数值数组。

`Action.sparse_spin` 可显式交付不可变 `(i,j,k,complex_weight)` 非零项；`None` 沿原稠密表达式，空 tuple 为全零自旋。B 已完成自旋基转换及源端 adjoint，X 不重复这些变换，也不把算符系数乘入此核。逻辑输入保持三阶 G、三阶稠密 Phi 和三条传播子，不物化 G×Phi。`operations/baryon_sparse_spin.py` 按第三自旋及第二自旋分组，先混合小的第一传播子自旋切片，再用 GEMM 完成三个模态求和，最后一轮按第一开放腿分块累加。重新结合求和可能改变浮点舍入，精度不改变。

稀疏路线要求五输入、三个独立的自旋/模态腿、G 的三个自旋 extent 均为4、各开放指标互异、所有数组为同一 complex64 或 complex128。精确非零项不超过6个时放行，包括 DR H 中间行的6个 `(j,k)` 对及4个不同 k。更大支持仍要求不同 `(j,k)` 对至多4个、不同 k 至多2个；超出此范围沿原 dense fallback，不按权重阈值剪枝。当前 DR G1/H 宇称配方的所有行都满足自旋支持条件；旧 DP 选块转成 DR 的16/24项保留原回退。组数增加仅增加顺序执行的 GEMM/累加次数，各组临时量逐组释放，不扩大既有 scratch 上界；性能收益取决于设备、规模和实际执行分支，已测范围及未采用的扩大支持方案见[重子内核正式分支复核](baryon-join-preparation.md#当前操作内的生命周期)。支持指标重命名、输入腿/轴及输出轴的排列和非连续布局；共享开放指标、trace、保留本地自旋或模态、混合精度及其它结构沿原显式 `kernel_path` fallback。另一256项有界 recipe 缓存只持有 shape、dtype、轴及给定 path，不持有数值、设备或 stream；无路径搜索。

令三个模态 extent 为 a/b/c，三条开放腿维数乘积为 F0/F1/F2，H=F0×b×c、A=F0×c×F1、q=max(1,F0//4)。稀疏核保守 scratch 上界为 max(2A+H,A+2H,A+H+2aF0,A+q×F1×(c+F2)) 个数值。纯指标消去重放给定 binary path 得到其最大中间量；仅当该 scratch 不超过原路径最大中间量时启用。输入布局打包仍计入原输入总字节预留，显式打包避免 batched GEMM 隐式广播第二传播子。Ne64 普通四自旋开放端的 scratch 为72/144MiB（complex64/128），完整输出为128/256MiB；CUDA/cuBLAS 原生工作区不包含在数组上界内。scratch 在结果交付前释放；CPU 复制快照，GPU 稀疏核接管独立结果存储（见下文）；reused 短路保持。CPU 独立显式求和专项见 `test_baryon_sparse_spin.py`，真实 V100 的已测范围见[重子内核文档](baryon-join-preparation.md)；三独立腿、两合并腿与单合并腿的结果分别记录，不互相替代。

当第三自旋组超过2个时，可将相邻两组的第二模态缩并结果直接写入同一 packed
accumulator 的列块，把第三腿传播子沿模态轴拼接，以一次 final GEMM 合并两组贡献。
此路线不保留额外的逐组 accumulator，减少大输出的重复累加。其 scratch 上界为
max(3A+2H+2aF0,2A+q×F1×F2+2cF2)，只有不超过给定路径最大中间量时才启用；
否则保持逐组稀疏计算，仍不因六项支持结构退回稠密。recipe 的 `pair_groups` 仅为纯
shape/path 标志，不包含数组或算符值。Ne64 普通开放端的该上界约112.25/224.5MiB；
GPU 稀疏自旋核的返回值只持有本次新分配的 result 存储，可直接接管为稳定结果；
`result_snapshot` 阶段仍记录，复制字节为0。即使输出为 reshape/transpose 视图，也不借用
输入或可复用缓冲，后续调用不会覆盖它。CPU 和所有稠密 fallback 保持显式 owned copy。
输入布局副本和 CUDA 库私有工作区的计数范围与上述规则相同，计划预算保持保守。

空间核心 Phi 在每次稀疏操作开始时按 `(a,c,b)` 连续打包一次并复用，使第一 GEMM 的
新结果可直接 reshape 为 `(F0*c,b)` 供第二 GEMM 使用；取消每个 `(j,k)` 分组的
大中间量 `(F0,b,c)` 转置复制。矩阵乘仍为二维，不引入广播传播子副本；Phi 打包计入
既有输入布局预留，scratch 使用上述未降低的保守上界。矩形模态、负步长输入、
指标及输出排列通过独立 NumPy 完整求和专项，不缓存数值数据。

单 P2P bridge 保留点坐标的 CuPy 路线将逐 COO 行的累加替换为一次逐贡献 GPU
核调用，复用现有贡献、点索引与输出数组。Ns=4 且 Nc 为正时进一步融合 gather、
spin/color 贡献与累加；Nc 作为运行时参数交付，循环按实际 Nc 计算，CuPy CArray
多维坐标读取沿各数组实际步长寻址，避免将展平指标按运行时 extent 反解。
两端、gamma、颜色块可各自为 float32/float64/complex64/complex128，不分配端点
gather、整数组精度转换副本或贡献数组；加载值在核内转为 complex<double> 后计算，
保留原 complex128 输出和实虚部 double 原子累加。低精度/混合输入的重结合及更高
中间运算精度可能改变舍入，不保证与参考逐位相同。仅复用已有核构造器，Nc 不进入
自定义缓存键，类型编译由 CuPy 自身管理。
支持任意 gamma，不假定其属于某个自旋基的单项置换。其它输入类型、Ns 或 HIP 保持参考路线。
重复行仍逐行贡献，复数128输出的实部
与虚部分别累加；并行求和顺序可能改变，按精度容差验证，不保证逐位相同。
CPU、闭合点输出及跨 current join 沿原实现。对应专项为
`test_dense_atomic_kernel.py`、`test_baryon_atomic.py` 和
`test_sparse_retained_accumulation.py`。

| 操作 | 精确签名 | 作用和保证 |
|---|---|---|
| Dense contraction | contract_dense(action, arrays, backend, *, timed, device=False, copy_result=True) | 执行已计划的 einsum/显式 kernel path；输出写入执行数组表；不会自行搜索 contraction path。 |
| Baryon join | contract_baryon_join(action, arrays, backend, *, timed, device=False, no_search=False) | 消费显式 Action.sparse_spin 进行局部非零自旋缩并；不适用时保留 kernel_path 表达式。no_search=True 的 fallback 逐给定路径执行已有 dense 数值核；重用动作不重新算。 |
| Sparse bridge action | contract_sparse_bridge(state, action, arrays, backend, *, timed, device=False) | 用 C SparseBridgeSpec 对齐端点并调用现有 P2P bridge/cycle 数值核。输入 records 保持 sparse，不建 dense point-pair 张量。 |
| Sparse source validation | validate_sparse_bridge_parts(problem, key, value) | 检查 part 三元组、record shape/dtype/index，并核对 C part_count、两个 COO point extent、nnz_upper；不重排或合并重复 COO 行。 |
| Raw block read | read_block(loader, block_id, block, *, timed) | 仅读计划列出的块。 |
| Logical source read | load_source(loader, tensor, source, regions, *, timed) | 兼容显式 load_region/view reader；核心格式绑定由 loader 完成。 |
| Retained source | load_retained_source(source_execution, tensor, arrays, *, timed) | 复用仍在存储计划驻留范围内的逻辑张量。 |
| Materialize | materialize_source(source_execution, tensor, source, *, timed, device=False)；materialize_from_blocks(loader, tensor, source, blocks, *, timed, device=False) | 只以计划已读取的 raw blocks 构造逻辑输入。 |
| Transfer | copy_to_backend(value, backend, *, phase, timed, device=False) | 计时并复制到指定后端。 |
| Output snapshot/lane | copy_output(key, arrays, backend, *, timed, device=False)；select_output_lane(value, backend, lane, axis, *, timed, device=False) | 返回稳定的拥有型 TermValue，不让输出借用之后可能复用的中间缓冲。 |
| Release | release_tensor(action, arrays, regions, sources, *, timed) | 仅移除计划指定的逻辑张量与已无消费者的 region view。 |

_BlockExecution 维护真实 raw block / retained array 表；C/P 的 ContractionState 只表示规划状态。当 materialize 返回借用 raw block 的稀疏记录时，物理块仍由全局 storage event 保持到最后一个逻辑消费者结束。设备上有待完成操作时，在 close_block 前同步当前 stream；异常清理也先完成该 fence。任何清理异常都不会替换原始读/核错误；无原始错误时清理异常正常传播。

该内部状态对象的精确构造与方法为：_BlockExecution(plan, loader, backend, shared=None, retained=None, sources=None, temporary_scope=None, io_overlap=True, measure=False, task_read_ahead=None, action_offset=0)、begin_action(index, timed)、materialize(key, source)、source(key)、end_action(index, timed)、has_retained(key)、retained_value(key)、retains(key)、retain_tensor(key, value)、model_features(key)、action_seconds(index)、close()。plan 是 C SceneStoragePlan；shared 是批执行期间共享的 (live_blocks, opened_block_ids)；retained 是按 P 计划跨批保留的逻辑数组；sources 是物理 SourceSpec 描述映射，默认取 plan.state.problem.sources。方法只实现事件回放与所有权管理，不更改计划。

多批路径内部入口为 _execute_task_batch_plans(expanded, path, *, backend, measure, io_overlap=True)。它按 P 交付的 batch_paths 顺序校验 child token/plan identity，构造与计划对齐的 batch_context；direct 模式将 backend 包成 _NoSearchBackend，使 P2P kernel 的逐项 einsum 关闭内部 path search。consume(index, report) 按每批 CoordinateBatch 和 batch_output_axes 拆分 TermValue，并复制到稳定数组。最后 X 再对 parent ExpandedCorrelator 做完整坐标覆盖检查。

## 共享块回放与存储策略

    replay_scene_storage(
        plan: SceneStoragePlan,
        loader: SourceLoader,
        *,
        backend: object = numpy,
        measure: bool = False,
        source_descriptions: Mapping[str, SourceSpec] | None = None,
        io_overlap: bool = True,
    )

    replay_storage_batches(
        plans, loader: SourceLoader, consume,
        *, backend=numpy, measure=False, prejoined=True, streaming=False,
        batch_context=None, source_descriptions=None, io_overlap=True, lookahead=2,
    )

X 的 replay_scene_storage 拒绝需要先前 batch 保留态的计划，并在任何 IO 前检查 plan.feasible、backend 和已保存的 sparse workspace。X 的 replay_storage_batches 只接受 prejoined=True（默认），显式 False 在读取计划/输入前拒绝；不调用 P 的 join 或补预算服务。每一批在 IO 前核对 feasible、backend 和 retained_before；每批完成后核对 retained_after。streaming=True 只允许 prejoined=True，后续批次若失败，先前 consume 的输出由调用方负责恢复。consume 必须在返回前将结果复制到稳定存储，不能把设备临时结果保留到下一批。

`lqcdweave.contraction.replay_scene_storage/replay_storage_batches` 保留原便利签名，并指向 `contraction.replay`。收缩编排的批入口仍默认 `prejoined=False`：先调用 P.api 的 `join_storage_batches`，再对最终计划调用 `prepare_coordinate_batch_workspace`，然后调用上述 X 服务。`prejoined=True` 跳过 join；streaming 只允许该值并惰性完成每个传入描述的准备。收缩编排的 batch_context 仍收到原始已 join 描述，X 消费准备后的描述；普通已完整计划的准备是幂等且保留对象身份。

`lookahead` 必须是正整数，默认最多保留两个计划描述。对 `streaming=True, prejoined=True`，
只有 loader 明确启用任务级 `prefetch_across_batches=True` 并实现五个后台读取方法时，X
才会在有限窗口内跨批预读；`source_descriptions` 在该路径中惰性消费，并检查每个计划
恰有一个来源描述。窗口容量按窗口内各计划的最小 host 余量准入。该能力只读入 host 数据，
未来批的 activate、finish 和 materialize 仍由 owner 线程按序执行。普通 loader 逐计划推进，
不跨批预读；非 streaming 的既有行为保持不变。

计划释放最后 raw owner 前，执行器同步设备 stream；只有 storage event 请求释放的 block 才会关闭。完成动作后不默认丢弃计划仍需保留的张量；evict/reload 服从计划的 retained_* 与 read/release 事件。任务结束时临时数组随本次执行作用域退出，已复制给调用方的 TermValue 继续有效。

当前可执行的临时资源子集限于完整、可行、单路径的 NumPy CPU `SceneStoragePlan`，只支持 owned host tensor 的 `spill_disk` / `restore_host`。调用方须把 `TemporaryNpyArrayStore` 显式传给 `NpySceneBlockLoader(temporary_array_store=...)`；scope、receipt 与 scratch 路径只在本次执行内存在，成功和失败退出都会清理。显式正 `disk_limit_bytes` 是必需项；X 在读块前用 store 的 `storage_bytes(shape, dtype)` 计算同时存活的 scratch 文件字节上界，超限即拒绝。没有 store、没有正 cap、GPU/device moves、保留态 batch 与 direct 路径均在任何输入读取前拒绝。

临时数组是独立的运行时能力例外：X 从显式 loader/store 读取 `host_reserve_bytes`，将其另加到本次局部执行描述的 host reserve，再进行准入；原始传入计划不变。该额度依赖注入的 scratch-store，不能由 shape/稀疏核 ABI 推导，不属于已移至 P 的 coordinate-batch 准备。本例外不声明所有运行时资源都已离线冻结。

恢复在同一已核验文件描述符上，将 payload 直接读入按 NPY C/F 顺序分配的唯一 owned 数组；
保留 dtype/字节序与 shape，不依赖 `np.load` 返回值的所有权布局，也不增加整数组暂存复制。
零字节数组不读取 payload；分段短读持续补齐，提前 EOF 则拒绝恢复。原有预算和 alias 拒绝保持。

P 的 `MemoryPoint.temporary_disk_bytes` 和 `SceneStoragePlan.peak_disk_bytes` 仍按数组 payload 字节计。NPY store 的上界另含文件头；测量模式的 `ReplayReport.execution_profile["temporary_file_peak_bytes"]` 记录这个由 store 声明的文件字节上界，不是文件系统占用测量。上界不含文件系统簇、目录和其它文件系统 metadata；调用方可在声明 cap 时自行留出余量。NPY store 只接收连续数值数组。存档只保存固定计划记录，不保存 store、scope、路径或 receipt；加载后执行仍须由调用方注入 runtime store。

### 有界后台读取

`replay_scene_storage` 和 `replay_storage_batches` 的 `io_overlap=True` 默认值允许 X 在主线程执行当前计划的计算时，由一个 IO worker 按原 storage-event 顺序读取后续 raw block。同一个持续 worker 只提交一次，待消费票共用一个完成条件，不为每个读取创建 Future/Condition；CPU 或无设备输入的待消费票上限128。GPU 按当前窗口最大的 host/device 输入上界和两域容量选择 128–1024 的票上限，实际仍逐票检查并保留两域字节额度，已消费且上传未完成的票上限32。零字节输入同样受数量上限约束。每次最多64票成批加入同一个 worker 队列并通知一次；早退或异常仍提交此前已接纳的票，保持首错和清理可完成。关闭先取消未开始的票并等待正在读取的票，再清理 owner。每个未消费条目都保留 loader 声明的完整 host 上界（返回数据和读取工作区），直到主线程消费且异步上传完成。CPU 下 cap 为 `memory_limit` 和 `host_memory_limit` 中存在的较小上限；GPU 上传预取分别受 host/device 余量约束，直到完成事件确认后才归还两域额度。若没有 host cap，队列最多预留所有可预取输入所声明的有限上界。X 只在执行期单独保留一个标量队列 capacity，不复制或改写 P 计划及其 MemoryPoint 轨迹；capacity 不超过每个相关计划峰值到对应 cap 的最小余量，因此受 cap 约束时 `计划峰值 + 队列预留 <= cap`。它只改变读取开始时间，不改变计划中的计算顺序、读取顺序或释放位置。完整 cfg 的轻量报告记录票上限和实际待消费高水位，不把它们作为吞吐证明。

普通单计划按自身已知的 read events 预读。非 streaming 的 `replay_storage_batches` 保持既有
整项任务 FIFO 行为：plans 已 join 且 loader 明确声明 `prefetch_across_batches=True` 时，
可在当前批计算期间读取后续批的 host 数据；未来批的 `finish_prefetch` 和 materialize 仍等到
其在主线程 activate 后执行。streaming 路径则只将最多 `lookahead` 个计划组成一窗，用现有
`ReadAhead` 跨该窗预读，并用窗内最小 host 余量核定容量。窗口结束时先 join/关闭读取队列，
再释放窗口描述并准备下一窗；窗口边界不提前读取下一窗的数据。因此这个 `replay_storage_batches`
窗口模式不是连续滚动预读，也不跨独立 cfg 或任务提前读取。新的参数化 cfg 执行入口见下节。
来源描述过少或过多都会报错。

可选的 `contracts.correlator.api.BackgroundBlockLoader` capability 有五个方法：`can_prefetch_block(key, block) -> bool`、`prefetch_host_bytes(key, block) -> int`、`prefetch_block(key, block)`、`finish_prefetch(key, block, value)` 和 `close_prefetch(key, value)`。前两个只描述资格与内存上界；`prefetch_block` 只在 worker 上做 host read；主线程上的 `finish_prefetch` 将结果转成与普通 `read_block` 相同的值，并负责 GPU transfer；未消费结果由主线程 `close_prefetch`。GPU 操作仍在 owner thread 上提交并按现有 stream fence 释放。loader 必须保证后台读取不会触碰并发使用的设备或借用缓冲，且 reader 会在 worker join 后才关闭。

loader 没有这五项 capability、下一项 read 不支持后台读取、队列剩余空间不足以容纳下一项，
或 `io_overlap=False` 时，相关读取走原同步路径；队列不会跳过 FIFO 中的条目去读更后面的块。
`measure=True` 始终同步执行，以保留校准阶段的独立读/算观测。写出和 spill/restore 仍是同步
路径。`CostProfile` 只预测明确的准备计算、快照和数值核成本；读取、传输及资源移动时间不计价。
`plan.seconds` 不代表串行或重叠执行的实际 wall time，实际 IO 耗时仍保留在 replay 观测报告中。
前端与档案行为不变；对上述 replay 接口，后台队列与 future
只存在于一次 replay 调用内，保存的计划不含队列、数组或线程。参数化 cfg 入口的队列生命周期
见下节。

### 参数化 cfg 计划执行（阶段三接口；阶段五已接入）

X.api 新增入口：

    execute_cfg_plan(
        plan, configurations, loader_factory, consume,
        *, host_memory_limit=None, device_memory_limit=None,
        prefetch_host_bytes=None, prefetch_device_bytes=None, debug=False,
    )

`plan` 是 C 的参数化 `CfgPlan`。`configurations` 是惰性迭代器，可由现有 `Dispatch` 提供；
X 不把它展开成列表。`loader_factory(cfg)` 必须返回上下文管理器，进入后得到该 cfg 的
`SourceLoader`。执行调用线程（owner thread）是唯一消费迭代器并创建/关闭 loader 上下文的线程；
后台线程只执行已登记的读取回调。队列分别按 host 和 device 额度准入，不把设备字节并入 host 预算。

`consume(cfg, point_index, values)` 接收每个计划坐标的结果。一个 cfg 的坐标全部消费并且
数组/物理块已释放后，X 再调用 `consume(cfg, None, None)` 标记该 cfg 完整。收缩编排可用这个标记决定
何时发布完整产物；X 本身不写 DatasetWriter，也不接管输出发布。

执行将每个 `CfgInstance` 的节点映射和 `reused` 位表接到既有 scene replay 视图，按保存的
Action、读取事件和释放点运行。
逐动作的 `tensor_releases` 在对应动作完成后删除全局 owner；跨坐标共享值以全局最后使用为准，
输出则在快照后按 `releases` 删除。编译表的缓存键包含释放位置模式，实际全局 ID 留在当前
实例中；普通与 debug 回放使用同一释放表。CPU/GPU 共用该规则。
每个 worker 按不同的已保存 `reused` 位表编译一次回放描述，
报告 `descriptor_preparation_seconds`、实际新建 state 数 `compiled_template_variants`、
新建调用表数 `compiled_execution_tables` 与表命中数 `execution_table_cache_hits`；
`execution_table_plan_reuses` 为 1 时表示跨调用直接复用了同一完整计划的表，省去本次
`_bind_cfg_tables` 和逐 Action 绑定，新建 state/表数为零；绑定墙钟仍单独计入报告。执行时仅构造当前实例的
节点 ID 映射，不为全部坐标保留映射字典。这一步只解码保存的 Action 标志，不搜索、重演路径或
预测成本。它不调用 P 的路径/存储规划，也不调用 M 的成本预测。报告中的
`source_materialization_calls` 和 `source_materialization_call_seconds` 累计源准备调用次数及主机调用墙钟。
GPU 上该时间包含提交及调用内部等待，不等于准备核的设备耗时；`output_download_seconds` 记录下载
调用墙钟：同步路线包含等待，队列路线只含提交。下载事件等待另在 device 子报告中记录。
它属于 `numerical_replay_seconds` 的一部分；这些轻量统计不增加同步，不进入档案或科学身份。

同一原生模板在本次绑定有至少三个尚未缓存的复用模式时，X 只遍历一次 Action，
提取原 `reused` 和 `seconds != 0` 两张布尔元数据表；每个保存 mask 只在差异位置
替换记录。未改变的 Action 和完整未变的 state 继续共用原对象；复用项仍把耗时置零，
解除复用项保留原耗时，包括 NaN、Inf 和负零。临时标志表只活到本次绑定返回，
不进入窗口缓存或档案。少于三个新模式、debug、扩展 Action/字段及非原生描述沿原
逐项处理；已有窗口状态命中时不提取标志。此步骤只处理主机执行元数据，不移动数值输入或缩并。

构造各实例的执行表查找键时，相同原生 `DenseGemmGroup` 的有序 Action 编号只展开、
排序一次。临时字典以对象身份索引，并同时持有组描述与编号 tuple，避免完整 batch
内容哈希或身份复用；绑定返回即释放，不跨窗口累计。C 已冻结原生组及其原生 child
的编号。debug 和扩展组仍逐次读取属性，普通 Cartesian batch 仍直接读取已保存编号。
查找键的值及顺序、完整 batch 的表缓存键和数值程序均保持原语义。

同一次 cfg 表绑定、同一模板内，X 还以临时身份表保存已取得的原生 batch 程序引用。
首次遇到描述仍经过完整 batch 值键查找/绑定；另一次窗口独立解码的等值描述也先走
原值键。仅原生 `DenseKernelBatch`/`DenseGemmGroup`、原生 operand/result 字段及原生
标量字段已保证不可变时，后续状态变体可直接交接同一个程序，避免再次哈希全部成员。
身份表同时持有描述和程序，按模板分开，绑定返回即释放。debug、扩展记录/字段保持
完整值键路径；原程序缓存键、数值布局、当前输入解析和跨窗口失效规则不变。

`execution_seconds` 是包含 consume 回调的实测墙钟，`numerical_replay_seconds` 单独记录场景回放；
它们与计划中的模型成本分开。

读取队列在首个 Action 边界启动，此后在实际消费物理读取事件后推进；后台读取完成并不提前归还额度。
纯计算 Action 不重复推进队列，空事件查询不插入字典。读取顺序、额度准入和异常清理保持原有语义。

X 为一次 `execute_cfg_plan` 调用创建一个有界 `ReadAhead` 队列；后续 cfg 的已保存读取事件
通过 `ReadAhead.extend` 追加到同一队列，并使用同一个单线程 IO 池跨时间对和 cfg 边界继续工作。
每个 cfg 的实际来源使用独立键前缀，避免队列中的输入相互覆盖。正在读取和已完成但尚未被主线程
消费的数据都保留完整预读字节额度，直到消费或清理。只有已保存读取字节上界且 loader 声明可
后台读取的块会进入队列，其余块沿同步读取路径执行。若当前额度放不下队列中的下一项，后续读取
和 cfg 领取都暂停；X 不跳过事件或更改执行顺序。

启动时分别核对保存的 `plan.peak_host_bytes` / `plan.peak_device_bytes` 与 worker 上限，
并将 host、device 预读额度分别限制在 `plan.prefetch_host_bytes` / `plan.prefetch_device_bytes`、
对应上限扣除计划峰值后的余量及可选调用额度中的最小值。若任一 worker 上限低于对应保存的
计算包络，X 在领取 cfg 前报 `MemoryError`。运行时只会收缩已保存额度，不会重排路径或重算存储计划。
任一额度为零时仍逐项惰性领取并同步处理相应资源。

若回调、读取或计算失败，X 先停止并排空/取消结果消费及 join 读取线程，再清理当前及已领取 cfg 的 loader、数组和未消费
读取结果，并向调用者传播错误。`Dispatch` 已领取的行沿用其原有语义；X 不添加 ACK/租约，
也不调用 `Dispatch.clear()`。调用者须在所有 worker 结束后统一清理共享分发文件。

报告另含实际预读额度、队列峰值、reserved-byte 高水位、完成读取数、后台读取耗时与等待耗时。
后台读取、数值回放和 consume 有重叠，不应将这些阶段时间相加。分配到 worker 的 host 上限若低于
已存 envelope，执行在领取 cfg 前拒绝；启动时预读容量可收缩为零，但不会重排已存事件。

持续读取的票在共用条件锁内登记实际等待者；worker 发布该票的值或首错时，仅在该票
有等待者时发出完成通知。其它票完成及消费者取用已经完成的票均不需要唤醒。
关闭仍通知全部条件等待者，取消尚未开始的票并 join 正在读取的 worker；资源回收、
设备完成事件和失败传播保持原有边界。这不添加轮询、逐操作同步或额外执行参数。

`tests/correlator/execution/test_cfg_runtime.py` 覆盖跨 cfg 读取与当前核的事件握手、零预读额度、输入隔离及失败清理。
此前阶段记录的 CPU 合成检查不等于当前整合验收；当前工作区集成测试仍在进行，不能据此声明本次接口整合通过。

## 测量与数值能力

X 从 performance.api.ExecutionMeasurement 获得阶段测量上下文。每个 read、materialize、transfer、dense/sparse contraction、output copy、release 都在独立 phase 边界内计时；设备计时按 M 实现同步或 stream 事件。measure=False 不创建 collector，也不查询设备内存。

测量报告区分：execution_profile.sampled_peak_bytes 是 M 的定时采样；ReplayReport.peak_device_bytes 是 CuPy pool reservation 的动作边界观测，peak_host_rss_bytes 是可用时的进程 RSS 动作边界观测。它们是 checkpoint observation，不是经硬件高水位计认证的完整过程峰值。报告另写入计划峰值、observed_minus_planned_*_bytes 和 observed_storage_budget_exceeded；越过目标只记录、不主动中止。

phase 名称按原子边界区分 read_block、materialize、input_transfer、snapshot、contract_kernel、baryon_join_kernel、sparse_bridge_kernel、output_copy、release_block 和 resource_stream_sync。ActionTiming.observed_seconds 是该动作各 phase 的和；父子计时关系由 PhaseTiming.parent_action_index 表示，profile 拟合不把父动作和子 phase 重复相加。direct fallback 标记 calibration_eligible=False，因为固定顺序的 provisional 预测不作为校准样本。

## 旧泛型 compiled executor

旧 C ContractionPlan evaluator 仍由 X 执行，和新 Diagram 的 PlannedContraction 路径分开：

    @dataclass(frozen=True, slots=True)
    class PreparedKernels:
        plan: ContractionPlan
        operations: tuple[tuple[object, object], ...]

    prepare_plan_kernels(plan: ContractionPlan) -> PreparedKernels
    execute_plan(
        plan: ContractionPlan,
        bindings: Mapping[object, object],
        *,
        backend: Any = None,
        kernels: PreparedKernels | None = None,
    ) -> ExecutionResult

prepare_plan_kernels 只按 C plan 的 shape/dtype 创建表达式或 batched-matmul lowering，不绑定 arrays、设备或 stream。execute_plan 对 bindings 验证 key、shape、dtype、后端和单一 CuPy device；输入数组由调用方拥有，执行同步返回自有的中间量与结果，CuPy 当前 stream 在返回前同步。传入的 kernels 必须属于同一个 ContractionPlan。底层 MatmulLowering 为不可变轴 recipe，lower_binary_matmul(equation, operand_shapes, operand_dtypes) 不支持该结构时返回 None，由 executor 走现有 einsum 路径。

数值核复用旧实现：direct_contraction.py 固定顺序核、scene_runtime.py 共享状态 replay、scene_storage_runtime.py raw block 生命周期、旧 operations/*.py 原子操作、sparse_bridge.py P2P bridge/cycle、operations/sparse_join.py 有界点坐标 join、operations/baryon_join.py baryon kernel；generic path compilation 已归 P，generic executor 与 lowering kernel 归 X。

复用来源及必要 helper 闭包：execution/direct.py 以旧 direct_contraction.py 数值 load/materialize/action/output 实现为起点，移除其无限 budget `_build_fixed_state`，改消费 P 已审计的 state；execution/scene_runtime.py 保留旧 scene_runtime 的 `_model_features` 与 replay，CostProfile 更新/拟合移至 M；execution/storage_runtime.py 保留旧 scene_storage_runtime 的块事件生命周期并补 stream fence、cleanup 原错保留、source override 和偏差记录；execution/operations/* 从旧 contraction/operations/* 迁入，dense、baryon 与 bounded COO 数值核没有另写替代算法。`execution/scene_runtime._nbytes(spec)` 按 shape × dtype.itemsize 计算，来源是旧 planning.state `_nbytes` 纯 shape helper 的等价小闭包，避免 X 导入 P 私有函数。`execution/operations/sparse_bridge._operand_order(spec)` 保留旧 planning.state `_bridge_in_operand_order` 对 reverse_operands 的轴交换/转置规则，来源是 C SparseBridgeSpec 公共字段；P 和 X 都需执行这一小段 descriptor conversion，X 不再依赖 P 私有函数。Sparse equality/endpoint 描述与常量统一从 C.api 导入。

P2P kernel 边界的精确默认值：`sparse_pairs(record, used_np, nc=3, *, used_np_right=None)` 接受已有 identity/COO/matched 记录并返回 point-index rows 和 color blocks；`used_np_right=None` 保持旧方形调用。`contract_p2p_bridge(left, right, parts, *, used_np, used_np_right=None, backend=numpy, left_axes=(-3,-2,-1), right_axes=(0,1,2), same_point=False, transpose_bridge=False, equation: AxisPattern | str | None=None)`；`contract_p2p_cycle(left, right, first_parts, second_parts, *, first_bridge, second_bridge, equation: AxisPattern | str, backend=numpy)`。跨 current 的 `contract_sparse_p2p_join(left, right, first_parts, second_parts, *, first_bridge, second_bridge, equation: AxisPattern | str, point_equivalence, output_labels: tuple[int, ...] | str, backend=numpy, join_index_block_size=64)` 和单 bridge的 `contract_sparse_p2p_bridge_join(left, right, parts, *, used_np, used_np_right=None, left_axes, right_axes, same_point, transpose_bridge, equation: AxisPattern | str, point_equivalence, output_labels: tuple[int, ...] | str, backend=numpy)` 不创建 dense P2P tensor。字符串仅可作为直接调用的外部输入，由 C `parse_axis_pattern` 归一后按整数轴执行；join index block size 必须为正整数，当前跨 current join 明确拒绝非 NumPy record/backend。

支持边界：通用 dense einsum 使用现有 NumPy/CuPy backend；identity 仅可用于方形两端，COO/matched CPU records 的 P2P bridge/cycle 支持独立左右 extent并保留重复行独立求和。每个索引列分别按其extent验证和规范化；source-dagger 的已交换记录 extent 由 C `point_shape` 和 `transpose_bridge` 一起映射到 operand axes。复杂等价关系用 `contract_sparse_p2p_join` 按 `join_index_block_size` 分块扫描第二个 COO，默认 64 行，工作集不构造 nnz_first × nnz_second 数组。该跨 current join 当前只支持 NumPy COO record，不支持 device-native record；此时显式 NotImplementedError。没有把稀疏顶点稠密化来假装 bounded memory。实际 record 先经 adapter/loader 按格式 metadata 验证，X 再核对数值 shape、dtype、坐标与 C descriptor 上界。此项矩形支持仅由 CPU 数值测试覆盖，不构成 GPU/设备端 COO 验收。

### Coordinate-batched sparse bridge workspace

普通 CPU bridge 支持显式保留左、右或两个独立点轴，包含矩形点集、重复 COO 行、
source dagger 与空支持；保留坐标不会施加额外同点约束。双流环保留点轴时走已有
bounded COO join，闭合且局部等式已覆盖的环沿原两阶段核。规划和共享索引保留后续
因子仍需要的点轴。坐标合批的保留点路径直接生成最终轴序的结果，该结果由 P 计费，
不再添加不存在的 raw 投影缓冲；P 与 X 的独立 workspace 公式同步核验。
这是后端稀疏轴契约的能力，不提供点分辨算符/数据集的新前端。新增行为只经过 CPU 验收。

标记为 `p2p-coordinate-batch` 的 P/P sparse source 由 X 的真实稀疏 batch kernel 执行。lane 轴固定在输入第 0 轴；每个 COO lane/part 保留独立行归属，重复 COO 行仍逐行贡献。仅支持 NumPy，GPU/其它 backend 在 source read 前明确拒绝。低层无计划 `replay` 不接受该标记，因为它没有 P 的预算轨迹；P 的存储/资源审计在交付前冻结已知 workspace，完整 cfg 构建也在生成模板前完成该准备。direct、storage replay 和完整 cfg 启动只比较实际 kernel ABI 所需额度是否不超过已保存 sparse action 的 workspace；不足即在 loader 读取前抛出 MemoryError，不补额度或替换 sparse 计划。非 streaming 批处理会先预检全部已选计划，任何计划不可行时整批都不会开始读输入；显式 streaming 仍按既有逐批语义执行，后续失败可能发生在先前结果已被消费之后。已复用的 sparse action 不为未执行 kernel 另计 scratch。

P 负责 raw region、normalized source snapshot、逻辑端点/结果 owner 与已知核工作区；由 X 核 ABI 描述并在 P 独立估算的额外数组包络包括 endpoint reshape copy、按 COO 行收集的两侧数据、packed lane/left/right/gamma/coefficient/value arrays、贡献数组，以及当 kernel raw 结果与目标轴序不同时的额外投影结果。若启用 same-point 或方程绑定相同 point labels 的过滤，定义 `C=min(64, part_count × sum(lane_nnz_upper))`、`F=min(64, max(lane_nnz_upper))`；C、F 都非零时按 `C+2F-2` 行的完整过滤索引/值基数组加 `2F` 个布尔 mask 计费。原因是 X 的逐行 view 可在 flush 后继续持有整个 advanced-indexed COO chunk；只按 pending 行数或单个 64 行 chunk 计费会漏掉被保留的 base。空支持取零。该界限是已知 NumPy 数组的保守执行包络，不是 RSS、native/einsum 内部工作区或 GPU 显存上限；调用方若需覆盖 opaque workspace，仍须提供独立 reserve。该已知额度保存在 MemoryPoint.workspace_host_bytes，不占用调用方 opaque reserved_host_bytes，也不进入科学身份或结果定义。P 采用与原运行前修正相同的全点统一增量，逐点总额、峰值和准入判定保持一致；再次准备不重复计费。

内核输出分配前的索引范围 mask 受对应 raw COO row 数量约束；归一化索引 mask 受 E 拥有的 int64 pair 数约束；gamma 有限性 mask 受 E 的每 part complex128 gamma snapshot 约束。这些检查临时已由 P/E 的输入快照和 staging 轨迹覆盖，X 不再重复计入执行 reserve。该覆盖仅适用于通过 marked C descriptor、逐 part 上界检查及 E normalizer 验证交付的输入。

`execution.api.audit_execution_memory(path, *, backend=numpy)` 可在 流程 创建输出或调用 loader 之前，对 P 返回的完整 `PlannedContraction` 做只读 X 内存审计。它逐个处理 `batch_paths` 中已经 join 的计划（普通单计划处理一次），独立核对 X ABI，并返回 immutable mapping：已保存的最大 host/device peak、整体 feasibility 和逐批记录。兼容字段 `x_workspace_correction_host_bytes` 现在表示缺失额度（完整计划为 0）；`corrected_peak_*` 保留旧字段名但返回未修改的保存峰值。缺失 workspace 即 feasible=False，即使上限尚有余量也不在 X 自动补全。它不搜索、执行、激活 loader 或读取 getter，也不修改输入 path/plan；P/收缩编排已记录的 opaque reserve保持原样且只计一次。空路径保留 `DiagramInputs.assembly_host_bytes/device_bytes` 与 path 上显式 host/device reserve 的峰值。标记的坐标批次目前只支持 NumPy；GPU 路径显式拒绝。该接口报告已知模型和声明式 reserve，不保证 RSS/native scratch 上限；调用方应在产物创建前对 `feasible=False` 作拒绝处理。直接执行和 storage replay仍会独立重复准入检查。

三腿 baryon 的用户入口仍由 contraction 包级提供，归属分别为 C 记录、P 编译和收缩编排便利组装；
收缩编排返回共同 Calculation，value/旧 values 保留 term tuple，report/旧 stats 保留统计，X 内部
ExecutionResult 不变。共同约定见[返回说明](../correlator/calculation.md)：

    BaryonVertex(spin, elemental)
    BaryonTerm(source, sink, propagators, source_slots=(0, 1, 2), coefficient=1)
    compile_baryon_terms(specs, terms, *, optimize="greedy") -> ContractionPlan
    execute_baryon_terms(bindings, terms, *, backend=None, plan_cache=None,
                         optimize="greedy") -> Calculation

这是调用方显式请求的泛型编译/执行入口，不是 calculate 的 direct fallback；收缩编排.contraction_session.execute_baryon_terms 组合 P.api 编译/缓存与 X.execute_plan。X 不重导出编译器或便利组装。符号/插入系数仍由调用方按旧 API 约定处理，不把该 helper 当作新的三腿 Diagram binding。

### Persisted generic sessions

`ContractionSession.from_plan` belongs to frontend in `contraction/contraction_session.py`; it coordinates C/P/X/R and does not discover archive storage. Callers inject the archive load and validation functions from A:

```python
from lqcdweave.contraction.contraction_session import ContractionSession
from lqcdweave.storage.contraction_plans.api import load_plan, validate_bundle

session = ContractionSession.from_plan(
    "plan.json",
    backend="numpy",
    load_plan=load_plan,
    validate_bundle=validate_bundle,
)
result = session.execute_cfg(cfg, loader)
```

Archive paths require a callable `load_plan`; all inputs require a callable `validate_bundle`. The session loads and validates once, prepares its kernels once, and executes each cfg without compiling or searching again. Callback, validation, runtime-fingerprint, and execution errors propagate to the caller. `tests/test_contraction_session.py` uses a real PlanBundle archive to verify callback counts, repeated cfg execution, no plan search, and validation-error propagation.

Memory records and budget exceptions are uniquely defined in `contracts.memory`.
P.api exposes pure `estimate_plan_memory` and `resolve_budget_limits`; the latter
receives an explicit CPU/GPU target and host/device observations without probing
the machine. `runtime.memory` normalizes the backend and obtains fresh resource
observations for startup and execution guards; its `fresh_execution_limits`
includes reusable pool blocks once and preserves separate driver/pool fields.
X does not query driver/pool free capacity directly. X owns saved-plan loader, source
identity and plan-mismatch exceptions in `execution.errors`, exported through X.api.

The finite saved-workload scheduler is exposed through P.api as
`list_plan_candidates` and `plan_workload`. It samples whole, half-width and
single-output-group batches with greedy path planning, counts retained outputs
across the complete workload, and keeps output groups indivisible. An explicit
`target="cpu"` or `target="gpu"` selects resource accounting; `target=None`
preserves inference from the declared device cap without observing a backend.
Its former `joint` path policy is retired; passing `optimize="joint"` raises
`ValueError`. This finite candidate policy does not prove globally optimal batching.

The dynamic session methods are `tensors`, `baryon_terms`, `baryon_diagrams`, and
`diagrams`; persisted sessions reject dynamic replanning. `diagrams` retains the
scalar-time getter graph behavior through B/E/P/X/R public services. It uses
the session backend explicitly; omitting the backend selects NumPy. The former `scenes` and `evaluate` methods had no implementation or
consumers and have been removed. Symbolic point/current calculations use
`contraction.correlator.calculate` and its returned `Calculation.run`, described
in [frontend L2](../correlator/correlator.md). Their supported adapters and coordinate coverage apply;
this change does not restore the legacy prepare/bind object or all multitime
branches, and does not add a second frontend inside the execution policy.

## 异常职责

- 非法类型/参数在执行入口或 C record validator 报 TypeError / ValueError。
- 外来 plan、预算/绑定不一致、coordinate 覆盖不完整报 ValueError；不可行审计计划在 IO 前报 MemoryError。
- 不支持的 batch/kernel/backend 组合报 NotImplementedError；不会静默稠密化稀疏输入。
- Loader 与数值内核原始异常原样传播。stream fence / close_block / close_all 的清理错误不覆盖原异常。
- 实际运行超过 memory target 不抛规划错误；报告观测差值。真实分配 OOM 照常传播。

## 旧路径与验证

旧 contraction.execution.py 的类型执行入口迁为 contraction/execution/domain.py 并由 package api.py 提供；旧同名文件已移除，from lqcdweave.contraction.execution import ... 仍由 package API 提供。direct、scene replay、storage replay、atomic operations、sparse bridge 和旧 generic executor 各只有一份 X 实现。无调用的旧 executor/lowering/compiler/operations facades 已删除。纯路径编译唯一属于 planning/，X.api 不保留 compiler alias；contraction 包级编译入口直接指向 P。

未引用 load_legacy_p2p_parts reader helper 因全仓无调用删除；绑定的 reader 仍通过 SourceLoader 显式注入。

相关行为测试：tests/test_direct_contraction.py（direct 参考、异常、OOM、Term×coordinate 覆盖、稀疏桥接）；tests/test_direct_batch_execution.py（P plan → runtime.scheduler.execute_request → X 共享 raw block 生命周期与坐标回填，属于 P/X 集成）；tests/test_direct_measurement.py（测量边界与清理）；tests/test_execution_profiling.py、tests/test_scene_storage.py、tests/test_scene_storage_batches.py、tests/test_logical_retention.py、tests/test_sparse_cycle_diagonal.py、tests/test_sparse_cycle_partitions.py（P/M/archive 交接）、tests/correlator/execution/test_sparse_cycle_partitions.py（稀疏 join 独立稠密参考）、tests/correlator/execution/test_rectangular_sparse_bridge.py（矩形桥独立稠密参考）、tests/correlator/execution/test_contraction_engine.py 和 tests/correlator/execution/test_baryon_kernel.py。runtime.dispatch 文件队列和锁清理由 tests/runtime/test_dispatch.py 覆盖。CPU 测试验证 NumPy 路径；未运行真实 GPU、QUDA、MPI 或 cluster。

### Saved-bundle execution services

`execution.api` exports `prepare_bundle_execution(bundle, *, backend=None)` and `execute_bundle(bundle, cfg, loader, backend=None, *, prepared=None)`. Their implementation is `execution/scheduled.py`; the retired root `contraction/scheduled.py` is not a supported import. Preparation validates compiled batches and prepares kernels once. Execution binds fresh loader values, checks their declared shape/dtype and source identity, enforces the declared budget, executes batches, and returns term values and execution statistics. It does not search or assemble Diagram products; R assembles final outputs and frontend owns Session policy. Loader staging allocations remain the loader's responsibility. `tests/correlator/execution/test_saved_bundle.py` exercises the execution boundary, while real archive and repeated-configuration checks live in frontend's Session tests.

The baryon records are the unique C classes. The contraction facade points `compile_baryon_terms` to P and `execute_baryon_terms` to frontend.contraction_session. frontend observes binding shape/dtype, borrows arrays, delegates structural compilation/cache to P, then calls X.execute_plan. X cannot trigger compilation on a cache miss. frontend does not snapshot getter buffers or retain arrays in the structural cache; dynamic Diagram snapshots belong to B.

## 矩阵 K/H 数值核与计划接入

独立 `matrix_kernels.py` 通过 X.api 提供两个 NumPy 数值入口：

- `contract_meson_matrix_pc(p0, p1, left, right, left_weights, right_weights,
  *, partitions=None, row_block=1)`：传播子 `[4,4,Ne,Ne]`、无权重顶点
  `[component,4,4,Ne,Ne]`、线性系数 `[operator,component]`。返回
  `[left_operator,right_operator,left_PC,right_PC]`；无分区时省略两个单例 PC 轴。
- `contract_meson_loop_pc(propagator, vertices, weights, *, partitions=None)`：
  返回 `[operator,PC]`，无分区时返回 `[operator]`。

两个纯核还接受 `mode_block=True`，表示输入已选为单个实际矩形块；partitions 须为 None，
各传播子与顶点的模式轴按真实端点宽度对齐，返回无 PC 轴的小矩阵或圈向量。
这些纯核是低层 NumPy 数值入口，不代表独立的矩阵执行路线。矩阵任务通过统一 `anytime`
搜索入口规划；设备执行边界与验收状态见下方 `device_execution` 说明。

每个有序输入块的 P0×V / P1×收缩编排各计算一次，再应用两侧线性系数表，沿实际矩形
块点积并累加 PC 结果。右侧加权乘积保留，左侧按显式 row_block 点积。它不读文件，
不调用路径搜索，不补 Wick 负号，也不增加源端共轭。源/汇顶点与复系数必须已经
按 B 约定准备。四段区间形成两个长度 10 的 PC 轴，非对角两种取向各乘 1/sqrt(2)。

二元操作复用 X 原有 lower_binary_matmul / execute_matmul；遍历参考已迁移的
`contraction.meson_matrix.contract_meson_vertices_blocked`，将三因子中间量和前缀
累计改成 P 固定行流程的共享 K/H 与不相交 PC 累加。底层矩阵核要求统一输入 dtype，
不再自动提升混合输入；二元打包、原始/加权乘积及输出由 P 的矩阵模板估算和独立审计计量。
输出独立持有，输入不修改。纯核本身不读文件或管理生命周期；预算受控执行由既有计划和资源审计约束，
低层核本身不扩展公开规划范围。

CPU 专项以独立四因子 NumPy 缩并和显式投影顶点为参考，覆盖复数、统一精度、
不同算符轴长度、四段不等宽分区、单圈、行块宽及乘积复用。`optimize="meson_matrix"` 是统一
`anytime` 搜索的兼容别名；低层矩阵核不构成独立 planner。相关记录见[矩阵架构](../../architecture/meson_matrix_architecture.md)。

## 低层 MatrixKernelSpec Action 兼容回放

既有 `contract_dense` 对 Action.core 为 MatrixKernelSpec 的动作调用共享 K/H / 圈
数值核。它核对公式、输入形状/精度及显式 PC 系数表，返回独立输出；未知 dense core
明确拒绝，避免忽略其计算含义。仍沿用 load、release、contract_kernel、result_snapshot、
输出拷贝和原始块读/释放交接，不增加 loader、存储格式或执行 IR。

低层 `replay_scene_storage` 可消费 P 的 `plan_matrix_actions` 结果；执行模型特征来自 P.api
`estimate_matrix_action`，包含真实分块模板的算术调用和投影核对估计，不按完整八因子
公式计价。CPU 之外的后端在 scene replay 动作循环之前拒绝；P 同样在 IO 前拒绝。
NumPy 参考与低层 replay 的历史覆盖不代表当前接口集成通过。这里描述的是
`replay_scene_storage` / `replay_storage_batches` 低层 API；它们仍按原窗口边界关闭队列。
完整 cfg 路线由 `execute_cfg_plan` 消费已保存安排并交付结果。GPU/DCU 统一设备执行仍在集成测试，
完整 cfg 路线由 `execute_cfg_plan` 消费已保存安排并交付结果。GPU/DCU 统一设备执行仍在集成测试；该低层兼容 Action 尚无独立的全操作/性能验收。MPI 多 worker 与生产规模也仍待验收。


## 统一 CPU / CuPy 设备执行

X 的 `device_execution(backend, *, device_memory_limit)` 是统一设备 scope。NumPy 后端通过空 scope 保持 CPU 执行；CuPy 后端在 CUDA 或 HIP runtime 下使用同一执行与资源所有权接口。CuPy 读取可通过独立 allocator、非默认 stream 和事件完成设备上传；消费端只等待对应输入事件。本次执行的局部池、上传和 cleanup 属于 X，不属于 E 的 preparation recipe；进程级设备选择、初始化和资源观察仍由 runtime/adapters 管理，X 不自主探测容量或注册全局设备。

CuPy HIP 导出的 `driver.ctxGetCurrent` 是未支持的入口，不能按函数存在就调用。
HIP 每次启动直接观察硬件指纹，并使用当次执行独有的设备池和普通归约分配；
不启用依赖 context 身份的跨次池复用或归约工作区复用。额度、stream 同步及关闭清理
沿用原路径，CUDA 仍以真实 context 隔离复用，真实驱动错误继续向上传递。
协议测试覆盖 HIP 的未支持查询不得调用、硬件变化拒绝及当次资源释放；实际 HIP
数值验证与完整生产结果另记，不由模拟测试推断。

cfg 计划分别保存 `peak_host_bytes`、`peak_device_bytes`、`prefetch_host_bytes` 和 `prefetch_device_bytes`；X 在执行时按对应 worker 额度分别收缩两类预读，不改变计划顺序。`SceneStoragePlan.kernel_batches` 是已选的低层执行批注；统一设备 scope 不会在加载或运行时重新搜索、合批或预测。

原始设备输入关闭后，若已绑定的 ReadAhead 尚有设备额度，X 保留该输入所有者直到计算事件完成，
而不将整条上传流排到计算之后。未消费上传与这类暂留输入共享同一 `device_capacity`；
暂留元数据最多 32 项，额度或槽位不足时使用原有上传流事件栅栏，不增加缓冲额度。
队列接纳后通过非阻塞事件查询回收，只有资源压力或关闭才等待对应事件；
异常期间 DeviceExecution 继续持有未回收所有者，直到设备 scope 清理。
设备 scope 一次绑定 CuPy 依赖事件工厂，上传完成、输入退休、下载前置及下载完成
事件使用 `disable_timing=True`；这些事件只用于 query/wait/synchronize，不交付耗时。
校准及显式测量仍使用各自的计时事件，输出等待统计仍按主机墙钟记录。
自定义或兼容后端的无参数工厂保持原调用方式，不在每次上传时探测接口或重试创建。
debug、零设备预读、没有完成查询的后端及未绑定队列的读取沿用原有释放路径。
`peak_prefetch_device_bytes` 包含共享额度内暂留输入；`peak_retired_input_bytes` 和
`retired_input_count` 另记该部分峰值和累计接纳数。CPU、计划及前端接口保持原样。

设备池额度是执行资源限制，不代替 P 的 host/device 双峰值与预读额度审计，也不涵盖 runtime/context 或库内原生分配。所有 CUDA/HIP 操作与生产规模仍待验收。低层矩阵 K/H 核保留供既有 Action 使用，但不建立独立的 matrix-device 执行路线。

窗口执行在每个窗口（含首个与后续窗口）接入 reader/预取前，沿用完整 cfg 执行的 coordinate-batched sparse workspace 只读校验；保存的工作区不足或后端不支持时拒绝，不在 X 中补规划或读取数据。


原生完整 cfg 的执行表在绑定时按实例顺序保留引用；回放直接取当前时间对对应的表，
不再重新展开、排序 GEMM 组成员及构造事件查找键。引用仅覆盖当前计划/窗口，
不保存数值或改变已选变体；debug、扩展记录和可变事件列表仍使用原查找路径。

## 显式 GPU 输出后处理

完整 cfg 与窗口执行的可选 `device_transform` 在仍处于当前 device scope 时消费设备
term 快照，替代默认先下载 term 的结果交接；`None` 保持已有 CPU 后处理路线。
收缩编排只在 Dataset 显式 `postprocess_backend="gpu"` 时连接它，不更改普通返回值默认。

收缩编排交付正确的冻结单点配方；R 定义 CPU 求值语义，深度重复核验放在 debug；X 不导入 R。
`bind_device_output_assembly(recipe, problem, backend, *, debug=False)` 消费已验证的 C 记录，
把数值字段编译为融合 CuPy 程序；大字段拆为最多 8,192 节点的有序段，
临时缓冲按最后消费者复用并显式预算。shape 操作用索引表达式
实现，term 求和/乘法按保存的顺序和 dtype 转换，避免逐节点创建数值数组。
常量数组仅绑定时上传；超出直接核参数额度时，每次调用上传共享的小型 pointer/stride
表，表与主机 staging 保持存活到设备消费完成，不回传 term 数值。
调用构造间接地址表时，只取一次当前 owner snapshot 的固定位置表；每个 owner 的地址和
步长仍按本次调用读取，标量条目不建立空的步长循环，不增加跨调用的指针缓存。
相容窗口只保留当前模板程序，不无限缓存历史计划。`describe_device_output_fields`、`estimate_device_output_bytes`
、`estimate_device_output_metadata_bytes` 和 `describe_device_output_program` 只计算纯 metadata，
不初始化 CUDA。CPU 生成核体/协议验证不能证明设备编译或实际加速。

`describe_device_output_reservation(recipe, problem)` 在一次离线遍历中返回同一份
`fields`、`device_bytes` 与 `metadata_bytes`。收缩编排的输出预算同时需要这三项时使用此入口，
不分别重建三次描述，也不绑定随后丢弃的 CPU 求值器。原有单项接口仍可独立使用；
共同描述只在本次调用内存活，不缓存配方、数值或设备 owner。

构建输出程序时，NumPy 版本决定的标量提升策略只在模块加载时判断一次；每个运算节点仍
在建立时按原有 probe 推导并记录 dtype，后续阶段切分和源码发射复用该固定节点的类型；
probe 仍参与原标量提升判断，在旧 NumPy 下保留值相关提升的拒绝检查。每段依赖只遍历一次，
同时登记参数顺序、间接地址表和临时量最后消费者；随后按该最后使用表安排 scratch。
不保留全部分段参数表的第二份副本，不改变分段、内核源码、算术顺序或资源预留。
原始 OutputNode 的身份查重表只在输出树编译期间使用，共享节点仍只编译一次。
输出树构建离开该作用域时，无论正常返回或异常，均释放此表；编译后的表达式 DAG、
字段和阶段继续持有所需节点，原异常及调用者 NumPy 错误策略保留。
原生标量和 NumPy 数组的 dtype 校验复用节点已捕获的类型，省去重复转换；不透明 probe
仍沿原转换和校验顺序，保留回调次数及失败边界。原生节点在没有子节点或仅有一、两个
原生子节点时直接合并已有工作量计数，其它子节点容器沿原遍历。依赖遍历只为有子节点的对象登记访问身份，
叶依赖仍按来源键去重，保留参数首次出现顺序及共享分支的处理。
两个标量运算的输出形状直接取 `()`，省去通用广播形状推导；任一侧含数组轴时仍调用
原 NumPy 广播检查，保留空轴、不同秩及不兼容形状的行为，dtype 与 probe 运算不变。
生成同一条二元运算语句时，只读取一次节点 dtype，并供两侧转换和结果变量共用；
两侧显式转换共用一次已确定的 C 类型文本，不重复规范化 dtype；变量定义直接追加到
当前源码列表，保留原有声明顺序、转换和舍入代码。
常量、输入和二元算术的声明直接追加完整文本，复用已确定的类型及变量编号，
省去辅助调用及中间字符串，其余节点保留原生成入口；常量缓存与转换回调保持原样。
零维间接输入直接使用原 0LL 地址，非零秩仍沿原坐标公式；只有二元分支创建待完成帧。
左右求值次序、变量编号、显式转换、逐级舍入及完整生成文本保持一致。
类型只属于本次构建的表达式，不引入跨调用的 probe、数值或设备缓存，原有转换与舍入代码保持一致。
私有表达式用固定字段保存 kind、shape、probe、children、value 及构造时计算的 work/dtype，
省去每个原生节点的实例字典。五项构造参数、repr、身份比较/哈希、弱引用和普通子类的
扩展属性保持原行为；弱引用基类兼容 Python 3.10。此改动只调整主机描述对象的存储，
不改变数值运算、生成源码、参数顺序、阶段或设备资源预留。
浮点字面量仍先转换为 Python float；有限值以一次原生有限性判断进入原十六进制格式化，
非有限值沿原 NaN 检查区分 NaN 与无穷，保留符号零及原有生成文本。
支持类型的整数 0/1 文本在模块加载时固定；其它原生常量的格式化文本只在当前段的发射器
内复用，最多保存 128 项。浮点和复数键使用十六进制表示，保留实部、虚部各自的符号零；
不透明对象及数值子类仍逐次沿原转换路径。此处复用编译文本，不保存数组、设备 owner 或
跨段动态数值；不改变变量编号、生成源码、算术顺序、分段和设备预留。

输出内核的编译策略由 X 私有维护：C++17 支持精确的十六进制浮点系数，CUDA 使用
`--fmad=false`，HIP 使用 `-ffp-contract=off`，均保留原配方的中间舍入语义。
不增加前端参数，不在运行失败后试探切换编译模式或回退 CPU。HIP 选项分流经过协议
测试，不代表 DCU 数值或性能验收。最多缓存 16 个生成核对象，键包含工厂、完整参数
与代码、编译策略及实现版本，不含 cfg、term 数值或指针；CuPy 继续管理各设备和真实
数组签名的二进制缓存。相容窗口/新 cfg 可复用代码，但不跨 cfg 保留数值所有者。
内部代码标识使用 v2 命名空间：原生参数声明、输出声明、核体和 preamble 分别按
UTF-8/surrogatepass 编码，并加 8 字节小端长度；选项及版本的 repr 另加长度后写入
SHA256，省去整段核体的转义文本。旧版本及非原生扩展元数据仍用原 repr 编码。
格式升级会更换内核名称，使已有二进制缓存首次调用时重新编译；算术核体、编译选项、
输入输出声明、阶段及资源预留不变。此版本由 X 内部维护，不增加用户参数。
CuPy 在首次实际签名调用时 JIT，绑定不伪称完成编译，也不通过虚构输入预热。
后续调用不扫描缓存或增加诊断同步；编译异常仍在普通模式报告，并仅在失败时将
选项、编译器错误与生成源码写入系统临时目录的独立日志，原异常链保留。日志写出
失败不会掩盖原编译错误。额外数值检查沿用 debug，正常路径不执行合成自测。

`DeviceOutputProcessor` 在既有 scope 内保存当前/前一项有界所有者，通过 stream event
维持 term、融合字段与累加的生命周期；取消、出错或关闭先排空仍使用的设备操作再释放。
收缩编排注入 TS `DeviceTimePairAccumulator`，其物理 dt/首次槽语义不在 X 重复实现。
`dt_packed` 的完成均值、尾轴排列、NaN 和字节打包均在设备完成，最后只下载最终
dt/one-point packed records。`pairs_and_dt` 每点仍下载原始 packed record，经既有
输出队列写出；最终 dt 逐块下载，不能为减少传输改变布局。

X 的 device report 中 `downloaded_bytes/output_download_calls` 跟踪输出 D2H；
预算预留、池观测与整设备峰值是不同指标。P/收缩编排在分配前预留两域全 cfg 输出常驻与最终缓冲，
队列只能使用剩余额度，不扩大目标硬件 cap。storage 只接收已拥有的主机记录，
既有数值编码、身份与失败不发布保持。支持边界、完整调用和实机验收见
[GPU 输出后处理](../results/gpu_output_postprocessing.md)。

GPU 输出后处理的间接 pointer/stride 表上传计入 device 报告的 `output_metadata_upload_bytes`
和 `output_metadata_upload_calls`，与最终数值 D2H 的 `downloaded_bytes/output_download_calls` 分开；
这些计数不是核耗时或显存峰值测量。

实际设备装配调用、成功核提交及绑定数分别记录为 `output_postprocess_calls`、
`output_postprocess_stage_launches`、`output_postprocess_bindings`。
`output_postprocess_max_scratch_bytes` 是最大逻辑 scratch 预留，并非实测峰值。
仅 GPU 后处理报告 `output_postprocess_queue_modes`；它按 cfg/window 配置记录实际选择的
`overlapped_downloads`、`synchronous_budget`、`synchronous_debug`、
`synchronous_transfer_protocol` 或 `no_pair_downloads`，保留既有等待/任务计数。

CuPy 的同 dtype 参考二元缩并在固定绑定时可省去重复 einsum 解析：仅唯一指标、非单位/空维、受支持浮点/复数类型且有共享收缩的单侧求和场景，复用 sum(dtype=result) 与已绑定 GEMM。收缩标签按参考核顺序排列；额外准备 owner 和双侧打包上界必须不超过 P 通用 input-bytes + output-bytes 工作区，否则保持参考路径。结果独立拥有存储；CPU 沿既有实现执行。


### cfg 输出分配粒度与旧资源计划

GPU 输出装配的离线描述提供每个 field、常量、共享 scratch slot 和指针表的独立 owner 字节，
保留原 `device_bytes` 的逻辑 payload 口径。收缩编排对这些 owner 分别应用设备分配粒度；
TS 的 `describe_device_accumulator_allocations` 描述各 field 的驻留累加值与最终输出 owner，
原 `estimate_device_accumulator_bytes` 仍返回二者的逻辑总量。CPU 默认路径不增加 GPU 费用。
最终打包 owner 也单独计费；这些是已知数组请求的上界，不能替代 allocator 碎片或不透明库工作区预算。

内置 GPU Dataset 后处理先把不同精度的 field 装入一个 int64-word owner，队列使用这个最终下载
布局；`dt_packed` 在逐点阶段无下载槽。只有 `output_bytes` 总量的外部 device_transform 无法证明
独立 owner 数量，保留同步下载，不猜测它的分配布局或更改回调结果。

缺少 cfg 分配粒度版本的旧 CuPy 计划，在直接执行、首窗口和后续窗口的资源边界核验既存 envelope。
若原有余量足够则继续；否则在首次进入设备作用域前、后续窗口读入前要求沿已选缩并路径重做资源规划，无需重新搜索路径。
新版本原生计划沿已验证的资源交接执行；下载队列仍保留当前预算检查和同步回退。

原生 cfg 每点的 tensor final-use 表直接保留只读 int64 `(Action, global ID)` 字节视图，
不再把每行展开为 Python 整数列表再建立字典。公共回放的 `end_action` 按递增 Action
执行，绑定程序即使跳过 reused 数值核仍保留释放边界；末尾输出快照之后才处理终端事件。
紧凑表只缓存上次索引范围，并保留重复/逆序诊断读取的原映射语义，不持有数值数组或设备指针。
缺失/重复 global ID 仍在原释放时刻报错，storage 关闭失败不会提前删除 tensor owner。
debug、扩展执行对象/数组映射、非原生整数格式或未排序表继续用原字典实现；无前端新选项、
不改变 GC 配置、不为多个点缓存一组展开的 Python 释放表。此主机端改进适用于 CPU 和 GPU。

原生 dense 坐标批处理保留逐 lane 的 identity/spin_spatial 纯准备配方：`TensorPreparation.raw_index` 在共享物理 slab 上先选 lane，再按任务 dtype 转换、dagger 和自旋乘法；动态输出使用既有 stack 配方，静态输入仍复用同一 slab。带 lane 选择的配方走描述化回退，不进入尚未支持该索引的融合核或整块上传借用优化。真正不透明的回调仍不能隐式转换精度。


### 普通 dense 介子中的精确 monomial spin

当实际常量的非零值还精确属于 `±1/±i` 时，E 可把它保留为小 spin 常量，
并按完整 raw 请求、所选端点、profile 有序常量、dagger、compute/store dtype 共享
纯空间准备。`TensorPreparation.spatial_only` 只改变写出布局；profile 的逐项
乘加、共轭、转置和后续 cast 顺序不合并。单个空间标量与非 unit-phase 常量沿原路径。
P 仍选择普通二元缩并，不固定 Gamma 与传播子的执行顺序。

`ContractionTerm.sum_barriers` 是可存档的数值约束：对窄精度，消去一组 trigger
的最后标签时，必须同时消去其 dependent 标签。E 用实际空间与 spin 标签声明，
P 的整数候选与独立固定回放共同检查；没有此字段的旧档案默认无约束。
最终求和 Action 保存 `wide_accumulation=True`，X 在原 input+output workspace
额度内执行已有两阶段宽累加，只将最终输出 cast 回任务 dtype；不会生成窄 Gram。
普通未标记矩阵乘法保留原选定实现。共享减少的 GEMM 与新增小 spin 数据准备均须
从公共选定 Action 和包含准备、pack、归约的同设备计时验证。

默认 structured matrix 搜索展开按实际有限 4×4 spin 常量的精确非零支持分组：
每行每列恰一非零时，相同四元 spin permutation 的 components 保持同组，
不展开算符输出行或逐 component pair。每顶点最多四组；两顶点最多十六种组组合。
不满足该结构的整组保留原 dense 表达式，不从 gamma 名称猜测支持或按阈值丢项。

E 将每顶点的两个 spin 标签经实际排列合并，源张量仅保存四个非零 spin×spatial 块；
相邻传播子以纯 `identity.index_permutations` 准备对应 spin 轴。普通整数表达式保留
component coefficient maps，P 自然将保留的 spin 标签视为 batch 轴，X 继续使用既有
matmul 与归约服务。首次顶点×传播子模式的复乘加从 64 N³ 降为 16 N³；
输入四块替代十六块，输出大小不变。首次编译、额外 gather/pack、分组后的规划及完整
任务收益仍须实测，不把运算量比值当作端到端加速。显式固定 K/H 专家模板不切换此表示。

纯 `TensorPreparation.spin_columns` 校验实际有限常量的完整精确支持，profile、dagger、
空间排列、计算精度和逐级 stack cast 顺序保持原表达式。`identity.index_permutations`
每次从当前原始块生成独立 owned 输入，不保留历史值/地址或借用大 slab；该配方不能
走原 identity 上传借用捷径。两项进入配方与档案身份，旧档案缺字段时恢复空描述。
P 保守保留原 profile 工作区，另给 gather 明确的临时量及 allocation padding；
prepared snapshot 按实际压缩 shape 计费。PC 仍先绑定真实有序 mode 子区间、在小结果
上按原系数组合，完整 cfg 与档案继续保存/回放普通 Action 和同一纯准备配方。
