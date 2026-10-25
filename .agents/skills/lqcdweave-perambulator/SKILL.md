---
name: lqcdweave-perambulator
description: "按给定作用量、源时间及实际 V/D 向量、原始 P 或投影点 H 生产 LQCD 传播子；H 沿用 P 编码。"
---

# LQCDweave · 传播子生产

任务路由、执行范围和保存约定见[技能共同约定](../README.md)。多 cfg 分配、循环前共同准备与完成核验见 [Dispatch skill](../lqcdweave-dispatch/SKILL.md)。本 skill 负责当前 cfg 的传播子计算和科学身份。

## 常规 VSV

先核对实际规范场、Dirac 作用量与边界、V 基底身份和有序截断、物理源时间、容差及输出精度。生成器构造时只绑定规范场和后端；每次 `calc` 显式传入实际源与汇，返回顺序与汇列表一致。

```python
from lqcdweave.production.perambulator.api import (
    PerambulatorGenerator, VectorBasis,
)
from lqcdweave.storage.dataset import DatasetWriter

generator = PerambulatorGenerator(
    lattice, gauge_reader, mass=mass, tol=tol, maxiter=maxiter,
    backend=solver_backend,
)
generator.load(configuration)

basis = VectorBasis(basis_reader, count=Ne)
with DatasetWriter(output_directory, output_spec, provenance=provenance) as writer:
    for slot, source_time in enumerate(source_times):  # 物理时间，默认 range(Lt)
        [vsv] = generator.calc(source_time, src=basis, sink=[basis])
        writer.write(slot, vsv.values)
        del vsv
    data_id = writer.publish()
```

本例的 `output_spec` 使用源时间行布局，源/汇轴必须与实际 basis、`source_times`
和生成器的汇时间顺序一致，dtype 与结果一致；spec 构造见[通用生产说明](../../../docs/architecture/generic_production.md#perambulator-入口)。
计算与保存由脚本明确组合，不再调用库内的基础产物“计算并写出”封装。
使用六类 pair 布局时改用[配套 pair 范例](references/dilution.md)，不能把整条源时间行交给 pair writer。

`VectorBasis` 优先使用显式或句柄实际 endpoint；没有 endpoint 时使用 `dilution_source` 或 `vector_kind`，普通 Laplace 读取器默认为 V。已登记为 D 但缺少来源的句柄仍是 D，来源保持未知，不补造身份；自定义 D 句柄须提供真实 D endpoint。`PointBasis` 表示 P 点集，高模投影通过该对象的 `highmode_basis` 指定。源和各汇可使用不同句柄、截断及端点身份。结果持有自己的数组，`.source`、`.sink` 是实际端点记录。已 `load` 组态后，`generator.endpoint(basis, source=False)` 可解析实际端点供存储规划，不执行求解。

每次计算使用 `calc(t_src, *, src, sink, write=None)`。`sink` 是有序非空列表，可混合 V/D 向量汇、点汇和 `None`；返回列表逐项对应。`None` 汇保留完整未缩并解场，`sink=None` 是 `[None]` 的简写。完整接口、结果轴、反向转换和写出回调见 [PR L2](../../../docs/modules/perambulator/perambulator.md)。

## 按任务展开参考

- 含 D 向量、九类 V/P/D 产品、六类 pair 保存或九类读取视图：先按需读[稀释向量与 pair IO](references/dilution.md)。D 的生产由 [Noisevector skill](../lqcdweave-noisevector/SKILL.md) 负责；传播子不缩放 D，也不乘估计量权重。
- 需要 PSV、VSP/DSP、PSP 或同次多汇共享求解：读[点端传播子](references/point-flow.md)。只有实际使用点端才引入点集。
- 需要流式基底、资源块大小、原始解场复投影或同步块写出：读[模态与内存选项](references/modes.md)。
- 需要 low/medium/high 分区或投影点 H：读[完整三分区交接](../../../docs/modules/perambulator/three_sector_production.md)及[H 定义与路线](../../../docs/modules/perambulator/point_highmode.md#h-contract)。高模只投影完整 low+medium 父空间，不以 D 列代替。
- 测量含流：核对[流的传播子需求](references/current.md)。
- 读取并复用已发布传播子：读[已有传播子的增量复用](../../../docs/modules/perambulator/perambulator_reuse.md)，先核对作用量、实际端点和完整科学定义。pair 覆盖查询只提供信息，计算范围由脚本显式指定，不自动补缺求解。
- 仅当设备目标尚无档案时，读[目标档案与内存估算](../../../docs/architecture/generic_production.md#unregistered-target)。资源估算不代表硬件内存上限。

稀疏时间只在用户明确需要时启用：从同 cfg 读取选择记录，将 `source_selection` / `sink_selection` 传给构造器；源参数仍用物理时间，输出按汇记录顺序紧凑排列。输入句柄须声明 compact 时间映射，不能用 Nt 推断 Lt 或槽序。详见[时间工作流](../../../docs/modules/temporal_sampling/temporal_production.md)。

后端必须显式传入；生成器不识别路径、QIO 或原生 QUDA 对象，也不自动发布、恢复或补算输入。保存、读回与任务脚本约定见[通用生产说明](../../../docs/architecture/generic_production.md#perambulator-入口)和[数据保存说明](../../../docs/modules/storage/data_storage.md)。
