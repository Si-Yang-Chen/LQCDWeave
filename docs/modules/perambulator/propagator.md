# 传播子生产

PR 数值入口为 `PerambulatorGenerator`。构造时显式提供规范场、Dirac 作用量参数和后端；每次计算再绑定实际源及有序汇列表。详细构造器、结果轴、Gamma5 反向、点端高模和 pair IO 见[PR L2](perambulator.md)。

## 常规 VSV

```python
from lqcdweave.production.perambulator.api import (
    PerambulatorGenerator, VectorBasis,
)

generator = PerambulatorGenerator(
    lattice, gauge_reader, mass=mass, tol=tol, maxiter=maxiter,
    backend=solver_backend,
)
generator.load(configuration)
basis = VectorBasis(basis_reader, count=Ne)
for t_src in source_times:  # 物理时间，默认 range(Lt)
    [vsv] = generator.calc(
        t_src,
        src=basis,
        sink=[basis],
    )
    save_source_result(t_src, vsv.values)
```

`VectorBasis` 优先使用显式或句柄实际 endpoint，其次使用 `dilution_source` 或 `vector_kind`；普通 Laplace 读取器默认为 V。已登记为 D 但缺少来源的句柄仍是 D，来源保持未知，不补造身份；自定义 D 句柄须提供实际 D endpoint。`PointBasis` 表示 P 点集，可在对象上通过 `highmode_basis` 绑定完整 low+medium 高模父空间。已 `load` 组态后，`generator.endpoint(basis, source=False)` 可解析实际端点供存储规划，不执行求解。

`calc(t_src, *, src, sink, write=None)` 返回一个与 sink 列表顺序相同的结果列表。不同汇可拥有不同 handle、截断和 endpoint；一次源求解会复用到所有汇。`sink=None` 是保留完整未缩并解场的 `[None]` 简写。结果对象拥有 `.values` 数组，并记录实际 `.source` 与 `.sink`；使用同步 write 回调时结果不保留完整投影数组。

## 点端、时间和存储

PSV/PSD 使用向量源和点汇；VSP/DSP 使用点源和向量汇；PSP 使用点源和点汇。将所需汇放进同一个 `sink` 列表，可共享一次相同源求解。H=Qη 仍按 P 编码，在对应 `PointBasis` 设置 `highmode_basis`；投影扣除使用完整 low+medium 父空间，不使用 D D† 代替。

默认源汇时间覆盖完整物理 `Lt`。明确稀疏化时，将独立的 `source_selection` / `sink_selection` 传入构造器；`t_src` 仍是物理时间，输出第一轴按汇记录顺序为紧凑 `Nt_sink` 槽。输入句柄必须声明 compact 时间映射，PR 不根据 Nt 猜测 Lt 或坐标。见[时间工作流](../temporal_sampling/temporal_production.md)。

传播子不包含时间差重建、空间采样权重或随机估计量权重。通用 writer 和 pair IO 的产品身份、保存约定、Gamma5 时间交换、coverage/resume 见[PR L2](perambulator.md)；新文件默认格式见[数据保存说明](../storage/data_storage.md)。实际 V/D dilution 还需[稀释向量说明](../../../.agents/skills/lqcdweave-perambulator/references/dilution.md)。
