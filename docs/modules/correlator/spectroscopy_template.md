# 谱学：固定味道量子数的moving frame

这是保留的固定课题方案，用于解释测量定义和独立数值对照；其中参数不是通用默认。固定生产按[通用回放约定](../../../.agents/skills/lqcdweave-correlator/references/plans.md#联合任务与一次规划反复回放)组织。

目标或投影未定时按[物理量选择](../../../.agents/skills/lqcdweave-correlator/references/measurement-choice.md)核对候选；已明确任务直接沿用。下列例子的固定量子数、动量和调用方式不是所有谱学任务的默认或限制。

本例将两个测量放在同一 dict；物理声明见[谱学测量参考](../../../.agents/skills/lqcdweave-correlator/references/spectroscopy.md)。兼容 cfg 的完整任务在循环外规划一次并保存，后续仅绑定当前 readers 固定回放，不逐 cfg 重建计划。其它任务可按[公开模块](../../../.agents/skills/lqcdweave-correlator/references/interfaces.md)组合。

## 固定例子

| dict测量 | 味道量子数 | 空间量子数 |
|---|---|---|
| light/rho_I1_Iz0_Cminus_P001 | I=1，I3=0，(uu-dd)/sqrt(2)，完整C=-1 | P=(0,0,1)，E行0 |
| charm/chi_c1_DDbarstar_I0_Cplus_P001 | I=0，I3=0，味道C变换因子=-1；局域PI/RHO配方的完整C=+1 | 同一P，E行0；chi_c1与D*Dbar/Dbar*D构成2×2矩阵 |

动量单位为2pi/L对应的整数。Dstar用p=(0,1,1)、B1，D用p=(0,-1,0)、A2，联合投影总P=(0,0,1)的E行0。模板显式parity=None：固定moving-frame子空间不把静止系宇称标签直接套来。

这里为零净charm的D Dbar*相关散射通道，不是净charm=2的DD*扇区；后者不能直接赋同样的C本征标签。名字、味结构和粒子/反粒子身份必须明确。

### 固定味道相位

Flavor字符串按“反夸克、夸克”排列。沿用原有相位约定：

```python
charmed = (Flavor("dc"), Flavor("uc"))
anticharmed = (Flavor("cu"), -Flavor("cd"))
F_DDbar = charmed[0]*anticharmed[1] - charmed[1]*anticharmed[0]
F_DbarD = -anticharmed[0]*charmed[1] + anticharmed[1]*charmed[0]
F = (F_DDbar - F_DbarD)/2
```

这给出I=0、I3=0和C对味道转置的因子-1。它不是完整算符C=-1：还要乘具体Gamma/路径的C因子，本例PI为+1、RHO为-1，得到完整C=+1，与chi_c1配方一致。改变Gamma、空间路径或相位时重新核对，不能只复制任务键的Cplus。

按上文味道与投影定义调用 `gen_correlator`，将两个测量放入共同字典，
交给 `calculate`。实际数据绑定见[绑定参考](../../../.agents/skills/lqcdweave-correlator/references/binding.md)，
生产读取已保存的共同计划，见[回放约定](../../../.agents/skills/lqcdweave-correlator/references/plans.md)。

每个 cfg 的 light、charm 和 elemental readers 通过已有数据格式接口打开，并在当前绑定与 `run` 完成前保持有效。`prepare_spectrum` 只构造符号定义和 `OperatorPath`；返回的 `compute` 默认 `execute=False`，每次调用都根据传入 readers 建立该 cfg 的计划。`stored_momenta` 是已有数据声明的动量编号→字符串映射，须覆盖投影后的完整动量轨道，而不只是两条代表动量。不能按文件索引猜实际动量。`elementals` 按 rho/chi_c1/Dstar/D 提供已有读取器。

degenerate=True只合并u/d，charm保持独立数据。用户确认local-charm开关后，在完整dict上用remove_disconneted_diagram(...,[r"S^c_\mathrm{local}"])删除含目标线的整张图；不删local light或所有断开图，也不把整个charm通道清零。这是估计量选择，不能为加速自动开启。

逐腿选择见[基底与截断](../../../.agents/skills/lqcdweave-correlator/references/vertex-legs.md)。核对 rho、chi_c1、Dstar、D 源/汇的实际 light/charm 基底与有效值；原项目模板拒绝差异逐腿配置，该历史限制不代表全部公开模块能力。

提交前使用[谱学确认](../../../.agents/skills/README.md#execution-scope)。C因子/同位旋、moving frame、完整投影轨道及过滤开关是必核对项。

截断核对同时检查u/d同位旋伙伴与C配对项的实际基底/截断关系；仅固定Flavor系数不能证明任意逐腿截断后仍保持I/C量子数。
