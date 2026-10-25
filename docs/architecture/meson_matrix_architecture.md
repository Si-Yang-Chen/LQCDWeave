# 介子矩阵与完整计划

`optimize="meson_matrix"` 是 `anytime` 的兼容别名，不提供独立矩阵规划器。
支持的结构化整体介子路线由 E 交付中立 whole core，P 描述完整 connected/trace 调用及
存储安排，不搜索内核内部自旋/空间因子。低层 `MatrixKernelSpec` 与矩阵核继续服务
既有兼容 Action，不能据此推断公共入口采用旧固定 K/H 路线。

<a id="diagram-统一格式对矩阵路线的要求目标待迁移"></a>
## 矩阵路线的整数连接格式

跨模块采用 C 的 `AxisPattern` 与整数连接关系；成员、物理输入和共享结果分别保留身份。
只有实际调用字符串 einsum 后端时才转换局部标签，档案保存整数描述。

## 规划、执行与结果

- P 在保存前确定模板、坐标实例、输入请求、生命周期、资源事件和输出布局。
- 加载核对当前 reader 的科学身份、轴、shape、dtype 与覆盖；不重新展开、搜索或选择批宽。
- X 消费已保存顺序与资源安排，按额度推进读取和计算；R 按保存配方组装结果。
- 内存与 Dataset 输出消费同一结果定义；完整 cfg 发布后才登记对应覆盖。
- 计划可行性、CPU 数值、真实 GPU 和生产规模分别验证；模型内存不是实测峰值。

具体契约以[展开](../modules/expansion/expansion.md)、[规划](../modules/planning/planning.md)、
[执行](../modules/execution/execution.md)、[结果](../modules/results/results.md)和[档案](../modules/plan_archive/plan_archive.md)为准。
公开参数见[优化别名说明](../modules/execution/meson_matrix_execution.md)，任务生命周期见[Calculation](../modules/correlator/calculation.md)。
