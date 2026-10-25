# 多点：双守恒流四点与带动量重子弱跃迁

这是保留的固定课题示例，用于解释测量定义和独立数值对照；其中参数不是通用默认，定义绑定 helper 不代表跨 cfg 复用计划。固定生产按[通用回放约定](../../../.agents/skills/lqcdweave-correlator/references/plans.md#联合任务与一次规划反复回放)组织。

目标或投影未定时按[物理量选择](../../../.agents/skills/lqcdweave-correlator/references/measurement-choice.md)核对候选；已明确任务直接沿用。下列三点/四点是固定例子，不限定其它插入数量、动量或模块组合方式。

按[多点测量接线](../../../.agents/skills/lqcdweave-correlator/references/multipoint.md)构建一个嵌套 dict：

```python
{
    "four_point": {"J4c_J4u": ...},
    "baryon_weak": {"V4_weak": ..., "A_parallel_weak": ...},
}
```

本例将整个 dict 传给 calculate 并保持结果结构。其它多点可按[测量契约](../../../.agents/skills/lqcdweave-correlator/references/measurement.md)和[公开模块](../../../.agents/skills/lqcdweave-correlator/references/interfaces.md)组织；联合生命周期与共享成本由实际计划核对，独立调用不自动获得联合复用。

## 双守恒流四点

D=bar(u)c，源汇动量都是(0,0,1)。两个插入分别为J4_c=bar(c)J4 c、J4_u=bar(u)J4 u，插入动量为零，采用同一明确Wilson-r约定。两者均用ConservedVectorOperator，不用局域V4冒充其中一个。

四点时间顺序(sink,current1,current2,source)，例子(12,4,8,0)。所有真实t±1/空间端点和边界处理必须匹配，锚点时间不能代替全部端点。保持插入顺序；同时间或重合点另确认接触项与数据支持。

这是两个物理味道不同的守恒流，不是给同一流任意改名。相同物理流重复插入在当前前端仍有计数缺口，任务要求这种四点时应明确需先修复，不能用本例代替。

## 带动量的重子弱跃迁

- 初态：中子，Flavor(dud)，p_i=(0,0,1)，moving-frame G1，parity=None。
- 末态：质子，Flavor(udu)，p_f=(0,0,0)，G1、静止宇称+1。
- 荷变流：Flavor(ud)=bar(u)Gamma d，V4及沿q的空间轴矢量A_parallel分别保留。
- 明确采用q_insert=p_i-p_f=(0,0,1)，联合投影p_f+q_insert=p_i。这个q约定与数据傅里叶符号一起确认，不混用另一种transfer定义。
- 质子与流投影到总动量p_i的G1，中子留在投影外，保留两行。流V4用A1，A_parallel用A2。

重子三点时间顺序(sink,current,source)，例子(12,6,0)。两个通道提供裸Euclidean强子矩阵元，不包含轻子、GF/Vud、Z匹配或衰变率。

上述三点/四点的符号对象组成同一嵌套请求，交给统一 `calculate`。
已有输入与算符的接线见[绑定参考](../../../.agents/skills/lqcdweave-correlator/references/binding.md)，
兼容的重复任务按[共同计划](../../../.agents/skills/lqcdweave-correlator/references/plans.md)固定回放。

数据绑定用同一Insertion→Operator→Meson/Baryon/Current流程。elementals含D、J4_c、J4_u、proton、neutron、V4_weak、A_parallel_weak；momentum编号须匹配模板的0与(0,0,1)声明。传播子映射覆盖q与c及所需local数据，非简并时分别提供实际味道。

确认可参考[多点清单](../../../.agents/skills/README.md#execution-scope)：保留各外态和流的实际腿映射，共同值说明一次，差异单列；仅实际点端披露点集/Np，随机取样再补权重。混合重子与四点dict的数值支持仍须实数数据核对，符号构造通过不等于任意组合已验收。
