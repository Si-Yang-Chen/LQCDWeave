# 既有重子 join 的准备与计算

## 新规划的中立声明

编排层取得 X 纯 VertexKernelDescription，写入 C join；新 P 仅调用 M 的 estimate_vertex_kernel
返回粗工作量/traffic/工作区，不调用 _prepared_join_path、_direct_join_path 或 _join_path_cost，
Action.kernel_path 为空。M 固定形状投影不产可执行路径，不发现 spin 支持，不试跑数组。
X 在绑定 ordered-join-v1 时确定本地固定次序；实际分配沿用执行器预算、allocator 和余量。
绑定不重做 M 的粗估，不把估算工作区当作可信上限，不增加每次执行的形状描述构造。
多个逻辑端点可位于同一 operand，共享指标/自迹使用一般固定回退；稀疏专核仍遵守原拓扑、精度与工作区条件。

旧档案缺声明时保留原 path，新增字段默认 None，不升级旧数值策略。全局 core 仍一个 Action，
旧档案的分量拷贝/复系数/累加顺序保持不变；新声明的合并腿优化见下方分量中间和。
test_whole_vertex_integration.py 验证禁用三种 P 路径函数的新规划、
存档、源槽位次序、单输入多个端点、共享指标及执行不重复计价。
真实 GPU 小核对照与完整 native 回放分别见下文；不把探测结果当作完整任务验收。

下文为7fad7f40已合入的旧档案兼容说明。

本页记录已经接入实际执行入口的兼容改动，与整体顶点的实验性能策略分开。
`calculate` 用法、保存的 Action、存档格式、数值派发和工作区合同保持原样。

## 复用与修改

| 部分 | 本次处理 |
|---|---|
| 物理定义、Wick/味道、源端伴随及颜色 epsilon | 复用原绑定与科学约定，不在内核重定义 |
| 完整顶点 core 与内部有序分量和 | 复用原一个 baryon join Action，不把分量或非零 spin 拆成全局动作 |
| IO、快照、精度/设备处理、上传及完成凭据 | 复用现有 reader/preparation/device/lifecycle 服务，不新增队列 |
| 原 sparse-spin 专核 | 把原 Phi 打包提取为 `prepare_sparse_spin_elemental`，计算允许消费当前已准备包 |
| 原 fixed-path fallback | 复用同一保存路径、二元操作和表达式执行 |
| X join 入口 | 接入结构绑定、当前 prepare、compute 和 `finally close` |
| P 内部路径选择 | 新声明只消费 M 粗成本；旧档案缺声明时继续消费原 path |
| 算术路线/存储排期/资源与依赖 | 消费原计划，不重新选设备、数值策略或全局次序 |

`operations/baryon_kernel.py` 的不可变结构绑定只保存方程、shape/dtype、原有 path、
自旋标量元数据及派生整数配方；至多缓存256项，不含生产数组或指针。
它在绑定时解析原 Action 已有的 sparse 兼容条件，compute 不再次判断支持或改换路径。
该兼容派发与上文新声明并存：缺少声明的旧档案仍保留原有路径。

## 当前操作内的生命周期

prepare 消费原输入服务已经交付的当前 spin/Phi：稀疏路线对当前 Phi 做原来的 transpose/contiguous/reshape，
一般路线绑定保存路径的表达式。它不自己打开 reader、不上传、不选设备，
也不在其它全局操作之前长期持有额外包。实际腿或已连接的上游中间量仍在 compute 时消费。

```mermaid
flowchart LR
  I[原计划交付当前 spin 与 Phi] --> P[当前 join 的 prepare]
  P --> C[同一 join 的 compute]
  L[实际上游腿与原依赖完成] --> C
  C --> S[原结果快照与 core 分量累加]
  C --> R[finally 清空当前准备引用]
```

准备和计算都使用现有调用的 backend/stream；GPU 提交不等于完成，
输入与结果 owner 的安全复用仍由原执行依赖和设备生命周期规则管理。
本次没有通用 CPU 准备 worker，没有自动提前所有准备，也没有复制一张资源操作图。

`contract_sparse_spin` 的新增可选 `prepared_elemental` 只接收当前操作的包。
不传时仍执行原打包步骤，兼容原调用；本次不引入另一个工作区/策略参数。
每次调用的准备对象不缓存，返回或异常都会清空 spin/Phi/pack/表达式引用。
多分量仍在原 Action 内按原序加权、快照与求和，当前结果不写进结构缓存。
完整core直接调用共用的分量计算函数，不再逐分量复制Action、构造局部数组字典并递归进入join。
这只减少Python包装开销；原结果快照、core_copy、系数乘法和累加顺序均保留。

## 测量和验证范围

实际执行报告新增 `baryon_prepare`，计算仍叫 `baryon_join_kernel`。
之后保留 `result_snapshot` 和原 `core_copy/core_weight/core_accumulate` 相位。
测量模式会同步阶段，不能把阶段和当成异步流水线吞吐；本次不以实验微基准宣称端到端加速。

`tests/correlator/execution/test_baryon_kernel_preparation.py` 的 CPU 测试覆盖：

- 实际 join 入口的 prepare/compute 顺序，计算期间禁止重新选择 sparse 配方或重新打包。
- 数值异常后的当前 owner 清理，结构缓存不保留数组。
- native 重子图经 `save_storage_schedule/load_storage_schedule` 保存完整已选存储计划，再实际固定回放。
- 单/多分量、两种源腿排列、测量/非测量入口，变化当前数值后按原计划重新读取；回放封住 P 与 opt_einsum 选路。
- 独立显式颜色 epsilon、源伴随和有序三腿求和参考；原有复系数、微小非零、非连续和 fallback 科学参考。

该验收针对 CPU 固定坐标和原格式存储计划。它不代表 native opaque reader 已实现跨 cfg
档案重新绑定、重子时间合批、GPU、MPI 或生产尺度验收。

## CPU reader 到 GPU 的中立输入准备

原 NumPy Baryon 绑定的 opaque Phi 回调不能消费上传后的 CuPy raw block。中立准备由 B 描述原 spin 常量及 identity/compute_dtype/source-conjugate Phi 配方，
E 写入原准备表，X 使用既有 PreparedSourceLoader 和上传事件。科学定义仍取同一个 B 的
`describe_spin_entries`；颜色 epsilon 不重算，field_order 与各腿指标不变。
同一次图展开内，B 对相同顶点、算符项、旋量基、源共轭约定和端点映射只描述一次静态 spin entries；
后续 spin 张量、core 描述与 join 共用该小常量。这个复用不持有 Phi、传播子或跨任务输入。
相同传播子 reader、物理时间对和模态选择的 `vsv` header 也在同次展开内只描述一次；
每条图边仍独立核对端点和绑定，数值 block 仍按原计划读取。

identity 按任务声明的计算精度先转换，再 source 共轭并产生独立 owner。
避免 opaque 输出之后再重复
snapshot。验收要求 raw 后续修改不影响已准备结果，并对照旧 B 物化及独立三腿参考。
旧opaque档案不被改写成identity；原回调仍保留，档案的准备签名匹配规则继续执行。
新描述不增加存档字段或用户开关。

## 支持与精度

### 共享坐标的 sparse-spin 小核

`sparse_spin_recipe` 识别 Phi 或多条腿首轴上保留的同一个坐标 lane；spin 仍为静态
`(4,4,4)`。绑定仅存形状、轴排列和批宽。Phi 在当前 join 的 prepare 阶段打包为
`(B,I,KJ)`；静态 Phi 沿用二维 `(I,KJ)`。计算时三条腿各按原有源槽位配方整理，
静态腿只加单例首轴供矩阵乘法广播，不复制成 B 份。

按第三个自旋指标分组后，第一步 `(B,F0,I) @ (B或静态,I,KJ)`，第二步
`(B,F0K,J) @ (B或静态,J,F1)`，第三步按原 free0 chunk 计算
`(B,chunk×F1,K) @ (B或静态,K,F2)`。同组复系数按原顺序累加，最终结果只有一份
lane 轴；没有对 lane 做 Python 逐坐标循环，也不生成稠密 spin×Phi。没有共享 lane 的
原标量路线保持原样。只有一条腿带 lane 且 Phi 静态时，原路线把该轴视作 free 轴，
仍能给出相同数值。独立腿等旧路线不支持的拓扑继续走已存的 fixed path。

### 原生重子的完整坐标任务

E 将整体顶点隐藏的 spin/Phi 输入纳入物理依赖闭包，通过现有 reader slab 读取多时间块；
固定 spin 和固定源端 Phi 共享，动态输入使用共同的保留坐标轴。P 复用选定的缩并顺序，
按同一内核及分量装配调用估算批量成本；没有在坐标提升时搜索内核内部路径。
X 复用上述批量核，W 仍用 `ContractionTask.plan()/run()/save_plan()/load_plan()`。

当前坐标表示要求动态 Phi 的 lane 在其场景传播子指标中也有来源；不支持只有隐藏 Phi
变化而全部场景传播子静态的低层输入。受管 VVV 自动补产及任意重子流不由上述用例保证。

### 声明并固定输入精度

`calculate` 和 `ContractionTask` 接受 `dtype="complex64"` 等声明，在规划前统一
整体 core、spin 常量、Phi、传播子、谱权重与输出系数的计算类型；raw reader 元数据
仍描述原文件。执行按需读取，各块先转换再共轭、计权，已有输出宽累加策略保持不变。
计划及回放固定该类型，不根据后续读到的数据重新推断。

多分量 core 的每个组合系数也在进入正常处理前按声明类型舍入，避免只有 spin 和数据块
转换而系数仍保留原精度。

### 规划组件复用固定系数

共享计算分类后的局部组件只改变 join 的 scene 编号。原生冻结 join 通过浅拷贝复用
已经校验的 spin/core 描述；不再因重编号逐项解析稀疏系数。父问题保持不变，新问题
仍检查作用域及张量关系，扩展子类保留构造流程。该改动不缓存数值数据、不改变存档格式。
整数规划器建立局部 scene 到 join 的索引后，每个 scene 只构造自己的 join 上下文。
索引依原父问题的 join 顺序填充，保留同一 scene 内的原有顺序；组件 helper 的一般
入口仍可直接从完整父问题筛选。完整顶点 core 的隐含输入和 sparse bridge 作用域
继续由原组件构造逻辑处理。索引仅存在于本次规划器构造期间，不保存物理数组。

## 合并腿的分量中间和

整体顶点为多个 gamma×空间分量之和时，每个分量都与相同的两个合并腿 L、Q 相连。
内核先保留原顺序算出较小的中间量 H，
按原分量顺序加权累加 H，最后只与公共 Q 缩并一次。各分量的空间输入可以完全不同。

\[
H_r(x,\gamma,k)=\sum_{\alpha,\beta,i,j}
 \Gamma_r(\alpha,\beta,\gamma)\Phi_r(i,j,k)L(x,\beta,j,\alpha,i),
\qquad
C(x,y)=\sum_{\gamma,k}\left(\sum_r w_rH_r(x,\gamma,k)\right)Q(y,\gamma,k).
\]

这里 x、y 分别打包左腿的两个自由轴和右腿的四个自由轴；可另保留共同时间批轴。
保存的全局 Action、外部读集、依赖、结果形状、存储排期及批宽均不变。
浮点乘法与加法的位置改变，不能声称逐位等价。

| 阶段 | 后端行为 |
|---|---|
| 准备 | 消费已经交付的当前 spin/Phi/腿；按形状和轴标签绑定不可变前缀/尾部表达式，不读新数据、不搜索路径 |
| 计算 | 每个分量先缩并 spin 与左腿，再与 Phi 缩并；小中间量加权后按分量顺序相加，最后只做一次公共尾缩并 |
| 释放 | `finally close` 清除当前输入引用，返回结果仍走独立快照；缓存不保留数值输入 |

`operations/baryon_component_sum.py` 实现该局部内核。仅 `ordered-join-v1`、多分量、
同声明复精度、两个合并腿的正式表达式路线参与；支持无批次和共同前导批轴，Phi 可静态或随批次变化。
只在两个小累加缓冲不大于一个最终结果时使用。旧档案、显式逐步路线、单分量、独立三腿、
单腿标量及其它连接仍沿用既有实现；没有新前端参数或全局规划选项。

### 批内二元缩并复用矩阵执行

共同时间批轴使通用表达式的某些二元步骤不使用 BLAS。现沿用相同的三个固定二元步骤，
在准备阶段复用 X 既有 `lower_binary_matmul` / `bind_matmul`，将第二步和公共尾步绑定为矩阵执行。
没有再搜索顺序、拆批或改变外部排期。单坐标仍使用上一节的表达式实现。
CPU 所有模式长度都为 1 时保留上一节的表达式，小工作量不增加矩阵绑定路径。

### 单合并腿的整体标量归约接线

`contract_baryon_join` 在正式 whole core、单合并腿分支尝试
`prepare_scalar_sum`，具体数值方案与支持边界见[独立内核文档](baryon-scalar-sum.md)。
准备对象返回 `None` 时继续原分量路径；可处理时按准备、计算、关闭顺序执行，
将独立结果 owner 直接交付。计时使用 `baryon_scalar_prepare`、`baryon_scalar_kernel`、
`result_snapshot`；最后一步无额外复制，登记零复制字节。
reused 操作和显式 `no_search=True` 路线不进入该准备；两腿中间和及三独立腿分支不变。
CPU 任一模式轴小于 8 时保留原路径。join 先用已声明 core shape 排除小 CPU 操作，
准备函数再按实际腿的轴尺寸判定；阈值共用 `CPU_MIN_MODES`，不是两套独立规则。
