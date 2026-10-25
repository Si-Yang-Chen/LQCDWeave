# Correlator 完整计算流程

本页描述当前跨模块流程。模块职责见 [L0/L1](correlator-l0-l1.md)，交付字段见[交接契约](correlator-target-architecture.md)；具体参数、算法和限制由[模块文档](../modules/README.md)维护。计算用法从[correlator skill](../../.agents/skills/lqcdweave-correlator/SKILL.md)进入。

## 声明、规划与回放

物理算符经 `gen_correlator` 生成关联函数。常规入口为 `calculate`；需要显式生命周期时使用 `ContractionTask`。任务构造保存声明和 reader 句柄，不读取或复制数值输入。`plan()` 安装完整计划，重复调用复用已选计划；`save_plan()` 保存，`load_plan()` 绑定当前输入，`run()` 返回新的 `Calculation`。

```mermaid
flowchart LR
    S[物理算符与 Wick] --> E[输入绑定与展开]
    E --> P[计算路径与存储规划]
    P --> A[保存完整计划]
    A --> X[绑定当前输入并回放]
    X --> R[按原输出配方装配]
```

构造与执行参数不能混用；运行不能改变已声明的坐标或物理定义。任务不能并发执行，惰性结果流在耗尽或关闭前占用任务；关闭释放当前资源，不使已交付数组失效。详细行为见[生命周期](../modules/correlator/calculation.md)，实现见 [ContractionTask](../../src/lqcdweave/contraction/contraction_task.py)。

## 输入与产物

绑定检查实际基底、端点、具名有序坐标、精度和统计来源。形状相同不证明科学兼容。规划使用描述与读取请求；运行读取当前数值，不从档案恢复历史输入数组。

受管 elemental 的复用、补产和保存通过同一个 correlator 入口组织，支持范围见[自动准备](../modules/correlator/managed_elemental.md)。其它基础输入按各生产模块的契约提供。共同存储与发布行为归[存储模块](../modules/storage/README.md)，不另建项目级运行器。

## 顶点表示与内核边界

物理顶点的含义由算符和绑定定义；交给规划器的数值表示按实际路线选择，不能把某种内核的封闭边界推广为所有重子的约束。

| 路线 | 规划与执行边界 |
|---|---|
| 整体介子矩阵 | 外层消费完整 connected/trace 描述及粗成本；分量、phase 与 FWR 算术留在固定局部内核 |
| 默认重子行因子网络 | 展开层保留行轴，交付自旋因子和空间因子，允许规划器部分缩并并消去内部指标 |
| 显式 `baryon_vertices="joined"` | 保留封闭重子 join，由已选局部内核消费实际输入 |
| 胶球 | 已有数据作为零连接顶点接入，沿原结果配方装配 |

重子的 `"auto"` 比较按族规则、统一 `pair`、统一 `nonzero` 的不同候选，跳过无法得到可行计划的候选，在可行者中选择预测耗时最小的计划；不保证实际运行总是最快，也不保证全局最优。实现见 [calculate](../../src/lqcdweave/contraction/correlator.py)，科学定义、降阶和档案约束见[重子因子网络](../modules/expansion/baryon-factor-network.md)。

介子局部算法与版本回放见[整体介子](../modules/execution/meson-whole-vertex.md)，重子封闭内核见[准备与计算](../modules/execution/baryon-join-preparation.md)。不同求和或加权顺序可能改变浮点舍入，按对应精度与独立参考验证，不能假定逐位相同。

## 计算路径、存储与坐标

P 选择数学操作及复用关系，存储阶段按已选计算依赖安排读取、驻留与释放；坐标合批须使用对应路线实际支持的结构。元数据成本模型用于选择和预算估计，不代替目标设备的分配峰值或实测吞吐。

启发式搜索在预算内寻找可执行方案。候选评分可以近似，轴连接、系数、输出及输入身份必须保持正确。静态描述、共享分类、局部更新和选中动作交付由[共享链接](../modules/planning/planning-shared-links.md)及[规划模块](../modules/planning/planning.md)维护。

完整档案保存选定的操作、坐标与资源安排，后续绑定当前兼容输入并回放。兼容条件和版本拒绝规则见[计划档案](../modules/plan_archive/plan_archive.md)。旧档案不能在加载时猜测缺失语义或静默更改数值顺序；具体版本是否可读以档案实现为准。

## 执行、完成与所有权

X 消费已交付计划，读取、准备、上传、计算和输出沿现有服务推进。提交操作不等于设备完成；原 future/event 和 owner 生命周期决定输入何时可消费、何时可释放。游标与当前完成状态不作为另一份执行计划保存。

结构缓存不持有历史数值输入，返回结果须有稳定的所有权。异常沿原服务清理；未成功的输出不能发布为完成。输出装配归 R，执行与结果的详细边界分别见[执行模块](../modules/execution/execution.md)和[结果模块](../modules/results/results.md)。

## 验证边界

独立数值参考用于核验物理输出；固定回放测试封住重新展开和规划入口，并改变当前输入核对结果、坐标覆盖与释放行为。CPU、真实设备、完整任务和性能验证分别提供证据，不能相互替代。测试归属见[测试索引](../../tests/README.md)，跨模块验收方法见[集成验收](../modules/correlator/correlator-integration-acceptance.md)。
