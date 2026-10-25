# 统一关联函数流程的完整项目验收

本页维护验收范围与证据边界；单次作业状态和性能过程不作为当前实现契约。
用法见[任务生命周期](calculation.md)，模块职责见[统一架构](../../architecture/correlator-workflow.md)。

## 能力及证据边界

| 范围 | 核验要求 | 契约 |
|---|---|---|
| 任务入口 | 构造只声明、规划幂等，回放不展开或重新规划 | [编排](correlator.md) |
| 整体顶点 | 独立数值参考、源汇及腿身份、精度、变化输入 | [介子](../execution/meson-whole-vertex.md)、[重子](../execution/baryon-join-preparation.md) |
| 存储与生命周期 | 固定计算语义、两域预算、共享 owner 和最后使用 | [规划](../planning/planning.md)、[cfg 绑定](../planning/cfg-binding.md) |
| 固定档案 | 完整内核与输入兼容、共享身份、损坏拒绝及无重规划回放 | [档案](../plan_archive/plan_archive.md) |
| 执行与输出 | 依赖与异步完成、借用有效期、完整字段与发布范围 | [执行](../execution/execution.md)、[结果](../results/results.md) |

## 完整计算范围

完整验收固定科学输入、声明精度、全部坐标/字段及所选计划，使用独立公式参考。
CPU、真实 GPU、MPI 和生产规模分别报告，模块门禁不能代替完整项目验收。
首次及暖运行、计划构造/加载和数值执行分别计时；模型资源不等于实际 RSS 或显存峰值。

LL3 行打包因子网络的已记录科学范围与参考对照见[对应模块](../expansion/baryon-factor-network.md)。
其 complex64 误差按该页定义记录；未确立的完整接受阈值不得自行放宽或宣布通过。
旧固定计划的计时、旧 CPU 超时和单顶点比例不作为新路线的性能结论。
轻/粲介子与胶球的独立输入及对照范围见[共享规划与项目对照](../planning/planning-shared-links.md)。
任何后续实现改动仍须重新核验受影响范围，既有通过结果不自动覆盖新版本。
