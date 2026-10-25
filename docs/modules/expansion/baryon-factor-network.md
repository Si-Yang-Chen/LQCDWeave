# 重子行打包因子网络

本文是 E/P 的 L2 补充，记录 dense-V 重子请求（含介子/流、胶球混合）的“行轴 + 可部分缩并顶点”降阶。
模块职责沿用[展开](expansion.md)与[规划](../planning/planning.md)；旧的整体 join 降阶见[重子内核准备](../execution/baryon-join-preparation.md)。

## 前端开关

`calculate(..., baryon_vertices=...)`（`ContractionTask` 同名声明）选择新计划中整体重子顶点的降阶：

| 取值 | 顶点 | 自旋常量交给 P 的形式 |
|---|---|---|
| `"auto"`（默认，`None` 同此） | 打开 | 分别规划三种候选并保留 CostProfile 预测耗时最小者：按族规则（拆分能压缩非零元，即秩一/折叠或秩 k 小于非零位置数 n，用 `pair`，否则用 `nonzero`；各族可不同）、统一 `pair`、统一 `nonzero`；选择记入 `baryon_vertex_selection`，候选形式已相同则不重复规划；实际开销取决于需要比较的候选和搜索范围 |
| `"pair"` | 打开 | 两部分 `Σ_k A[k,sa,sb]·B[r,sj,k]`，与参考程序的"双夸克矩阵 × 自由夸克"同型；任何秩都拆，不回退稠密 C |
| `"nonzero"` | 打开 | 全部非零元 `Σ_n W[r,n]·E0[n,s0]·E1[n,s1]·E2[n,s2]`，E 为单热选择矩阵 |
| `"dense"` | 打开 | 保留整个 `(R,4,4,4)` 常量 C；供对照，实测从不最快 |
| `"joined"` | 不打开（仅显式声明） | 原整体 join 内核（稀疏自旋专核、合并腿中间和）；供对照 |

强制取值（`pair`/`nonzero`/`dense`）对每个行族都生效，没有效率回退；`"auto"` 的选择规则与依据见
下文"按结构的最优降阶"。除显式 `"joined"` 外，每个请求都打开，没有结构回退（见"适用范围"）。
`"nonzero"` 不依赖 Γ 形状：n 取各行非零位置的并集，三个自旋槽各是一个独立输入，
P 可见 C 的全部稀疏性；代价是 n 轴同时连接四个输入，规划和执行的操作更碎。
完整档案保存所选降阶，`plan_archive` 回放只接受缺省 `None`；显式 `diagram_adapter` 自带降阶，不能同时传本开关。
`BaryonDiagramBinding(vertices=...)` 是同一选择的后端形式，供对照测试与后端开发使用。

## 表示

一个重子顶点写成其各部分之和 `V[r; s, n] = Σ_p C_p[r, s0, s1, s2] · Φ_p[n0, n1, n2]`：

- `Φ_p` 是原有 `baryon-elemental` 空间源，读取、共轭与 identity 不变；
- 一个部分是同一腿绑定（同一 `describe_core_factors` 分组）且共用同一有序 Φ（同一分量/动量/时间/选择/场序）
  的全部自旋分量，E 把其加权自旋部分精确求和成一个 `(4,4,4)` 常量；物理系数在常量内，Diagram 外层系数为 1；
  一行可有多个部分（如位移/导数算符的不同空间路径，或不同腿绑定）；
- 各部分（除 C 外的全部数据）相同的物理行组成一个保留的输出行轴 `operator_row`，每部分的 C 形状为 `(R,4,4,4)`，
  用零依赖 `TensorPreparation("constant")` 交付，不读文件；部分按规范顺序排列，单行族同样打开。

介子/流顶点不打包行：每个描述因子（或同端点的顶点和）是一个部分，其标量系数在 R 中相乘；
孤立胶球照旧是单输入数据项。每个连通分量对每组行族、每种"各顶点取一个部分"的组合生成一个普通
`ContractionTerm`：输入为传播子、介子张量、各重子顶点该部分的 C 与 Φ，输出为各重子顶点的行轴。
R 用 `select` 节点按行号取回每个 Diagram 的值，同一分量的部分组合用 `add`，跨分量用 `multiply`。
Wick 图、味道权重、源伴随与输出系数不变。

`factor_row_coefficients` 进一步检查 C 能否在某个自旋槽处精确低秩分解
`C[r,s0,s1,s2] = Σ_k A[k,sa,sb] · B[r,sj,k]`（常见于"双夸克矩阵 × 行所在夸克自旋"的重子算符；
重构误差不超过最大元的 1e-13）：

- 优先秩一（无 `k` 轴）。B 每行至多一个非零时（如行即该夸克自旋，或源伴随后的单项矩阵），
  term 只带 4×4 的 A，该槽的输出轴直接是 `sj` 自旋轴；R 的 `select` 取 `sj = spins[r]` 并乘常数 `factors[r]`；
  否则 A 与 `(R,4)` 的 B 都作为普通输入，输出仍是行轴；
- 没有秩一槽时，取三个槽中最低的精确秩 `k`（主元 Gram-Schmidt 选 C 的独立行作 A，最小二乘求 B 并核对重构），
  `k` 作为 `factor_rank` 收缩轴：`A` 为 `(k,4,4)`、`B` 为 `(R,4,k)`；
- `factor_row_coefficients` 的缺省 `bounded=True` 只认因子总元数 `k·(16+4R)` 少于稠密 C 的 `64R` 的拆分；
  `"auto"` 与强制 `"pair"` 都用不设限的拆分，任何秩都拆。P 仍可先把 A、B 乘回 C。
- 秩 k 不超过非零位置数 n（C 的槽矩阵只有 n 个非零列）。k = n 时拆分没有压缩任何非零元：
  每个非零元各占一个 4×4 配对矩阵，此时 `nonzero` 的单热选择更紧凑。

稠密 C 同时耦合三个自旋与行轴，相邻收缩都要多带这些轴；拆分后与原始参考实现的 Γ 矩阵收缩同阶。
问题带 `ContractionProblem.factorized_vertices=True`，不含 `baryon_joins`。

## 适用范围

`plan_factor_network` 只读元数据，对每个 dense-V 请求都给出打开的分组，没有结构回退；只有显式
`baryon_vertices="joined"`（或档案、显式 adapter 选了 joined）才走整体 join。此前回退的输入现在这样打开：

- 介子/流混合图：介子顶点按其描述因子各成一部分、不打包行，与重子的 C、Φ 一起进入普通 term；
  混合 `usedNe` 元组仍要求三腿相等；带 D elemental 扇区的介子在两种降阶下都报未实现；
- 胶球混合：孤立胶球 Diagram 仍是单输入数据项，与重子分量相乘；
- `quark_exchange` 多重态标签：交换矩阵已由符号层约化进系数，E 只做与 join 相同的端点空间校验
  （交换腿的基底、时间与模态选择一致，同一多重态各分量读同一端点空间）；
- `ModeBlockedDiagram`：每个块先按其模态选择取出顶点，作为不同 Φ 选择的行实例打包；
  端点仍须等于显式 `usedNe`，含介子的分块在两种降阶下都报未实现；
- 一行多个 Φ 或多个核心分组：每个部分一个 C，各顶点取一个部分的每种组合一个 term。

所有行族都只有一行的请求（如只要对角元）照常打开：实测单行打开比整体 join 快 3–10 倍（见下文）。

顶点 dagger 与图方向不一致不是适用范围问题，而是输入错误：`gen_correlator` 的行与推荐绑定
（`source=row.dagger`）总是一致，只有手写 `vertex_map` 填反 `source` 时才出现。
三种降阶都在读入数据前报同一错误；打开路径自行检查，不经过整体 join 的检查。

请求的行组合不是完整乘积时，`_product_cover` 把它精确划分为若干互不重叠的子满积
（以最小未覆盖组合为种子，逐槽贪心扩展），每个子满积一个打包 term，不计算未请求的行组合。
划分后只剩单行族（如只要对角元）时同样打开。

坐标批处理把 C 视为跨 lane 不变的常量角色，Φ/传播子按原 slab 绑定。

## 规划

`factorized_vertices` 问题与 join 问题一样进入轻量整数路径搜索（非 debug、非 verify、单进程时）。
首条路线由 `PairPlanGuide` 引导：每个场景按当前 TermRecord 用 opt_einsum DP（超过 14 个操作数时用 greedy）
求自身的最省 flop 树，按 `(pattern, 形状)` 记忆；每步选择“在最多场景树上是叶对合并”的共享操作，
平手取较低模型秒数。该路线和其它路线一样经逻辑审计与物理存储排程后才成为 incumbent，
之后原有的局部替代搜索照常进行。引导只改变首条路线的顺序，不改变成本模型、存储或执行。

X 不变：C 与 Φ 是普通 dense 输入。

## 档案

计划档案按问题编码，保存所选降阶并按其固定回放。`factorized_vertices` 是档案的必需字段：
本字段引入前保存的档案（含介子、胶球档案）不再读取，需用当前源码重新规划保存。

## 验收

- `tests/correlator/expansion/test_baryon_factor_network.py`：2×2 行满积（含一行两个自旋分量）对独立标量 oracle
  与 `vertices="joined"`，覆盖三种源槽置换、`none`/`anytime`；双夸克行（单项折叠与一般 B 两条路径）
  同样对照 oracle 与 join；默认行例覆盖秩 2 拆分（含时间扫描与档案回放）；
  `factor_row_coefficients` 单测覆盖秩 3、稠密回退与单项折叠；部分行请求拆为 2×2 与单格两个 term，
  对照 oracle 与 join；只剩对角单行的请求在各打开降阶下都打开并对照 oracle。预制 `baryon_diquark_operators` 的 Cγ5 源（秩一）
  与 Cγ5/Cγ4γ5 混合汇（秩二）、Cγᵢ 的 H 行（两行秩二、四行秩三）全程无稠密 C，同样对照 oracle 与 join。
- `tests/correlator/expansion/test_baryon_factor_network_scope.py`：每个此前回退的结构（介子流、胶球、
  交换多重态、模态分块、一行两个 Φ、一行两个核心分组）与单行、时间扫描各一例，在全部五种降阶下
  断言除 joined 外都打开，并对照不依赖行打包的参考、两两比较。
- `tests/correlator/expansion/test_baryon_row_structures.py`：结构见 `baryon_row_structures.py`
  （秩 1–12 与折叠、稀疏/稠密 pair 矩阵、无低秩与 k = n 的随机 C、目录与双夸克配方、Dirac-Pauli 数据、
  混合族、单行、每行两个 Φ），每种在 auto/pair/nonzero/dense 下核对生成的自旋输入形式并对照 joined，
  并断言 auto 的按族规则与所选候选的预测耗时最小；常规层跑五个小请求，
  其余在临时层 `temporary/tmp_baryon_row_structures.py`。uuu 全 Wick 和中 local-G1g、Cγ5、Cγ4γ5
  及其组合精确相消（自旋对反对称，如 uu 标量双夸克），测试断言其值为零，计时仍有效。
- 现有重子前端、坐标任务与执行测试默认经过新降阶。
