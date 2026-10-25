# 按需生产重子 elemental

`OnDemandBaryonElemental` 实现公开的重子 VVV reader 接口，在请求时计算一个指定的分量、动量、物理时间和三腿模式选择。它复用 `BaryonElementalGenerator`，可直接交给现有 `Baryon(reader, operator)`；`Baryon` 原有的源端共轭、自旋基变换和 `calculate` 工作流保持不变。

```python
from lqcdweave.adapters import OnDemandBaryonElemental
from lqcdweave.contracts.elemental import BaryonComponent
from lqcdweave.operators.single_hadron import Insertion, Operator
from lqcdweave.contraction.binding.api import Baryon

component = BaryonComponent.paths()
operator_table = [{
    "component": component,
    "paths": ((), (), ()),
    "coefficient": 1,
}]
insertion = Insertion.local_spin("H", "g", {0: (0, 0, 0)}, component)
operator = Operator("H_row0", [insertion[0](0, 0, 0)], [1])

with OnDemandBaryonElemental(
    gauge_field, eigenvectors, [(0, 0, 0)], operator_table, cfg="cfg001",
) as reader:
    vertex = Baryon(reader, operator).load("cfg001")
    factor = next(vertex.factors(0))
    print(factor.spatial.shape)
```

`operator_table` 与 `BaryonElementalGenerator` 使用相同表格格式，但这里每个 component 必须是 `BaryonComponent`，路径只能包含空间方向。输出块的轴为 `(mode0, mode1, mode2)`。

## 输入和参数

```text
OnDemandBaryonElemental(
    gauge_field, eigenvectors, momenta, operator_table, *,
    lattice_size=None, mode_counts=None, basis_bindings=None, cfg=None,
    array_module=np, phase_convention="anchor", path_boundary="periodic",
    ne_block_size=16, spatial_block_size=256,
    cache_bytes=64 * 1024**2, transport_cache_bytes=64 * 1024**2,
)
```

| 输入 | 约定 |
|---|---|
| `gauge_field` | 数组布局 `(Lt,Lz,Ly,Lx,4,3,3)`；也可提供 `read_time(cfg,t) -> (Lz,Ly,Lx,4,3,3)`，或 `load(cfg) ->` 完整规范场数组。若是 handle，须给 `lattice_size=(Lx,Ly,Lz,Lt)`。同时有数组外形和读取方法的对象按 handle 使用。 |
| `eigenvectors` | 一个共享数组/`read_modes` handle，或按腿顺序给出三个数组/handle。数组布局 `(Lt,Ne,Lz,Ly,Lx,3)`。handle 提供 `read_modes(cfg,t,start,stop) -> (mode,Lz,Ly,Lx,3)`；handle 都须显式给 `mode_counts`。只有规范场是 handle 时才必须显式提供 `lattice_size`。 |
| `momenta`、`operator_table` | 动量是按输入顺序排列的 XYZ 整数三元组；算符表沿用公开重子生成器的 `component/path(s)/coefficient` 记录格式。 |
| `basis_bindings` | 可选的三项序列，每项为 `(actual_basis_id, ordered_labels)`；未知身份传 `None`。类不会根据 shape 推断基底身份。默认全未知。 |
| `cfg` | 此 reader 固定绑定的组态。调用 `load(cfg)` 时 key 必须相同；切换组态要新建 reader。 |
| `array_module` | 默认 `numpy`，也可注入其它数组模块。GPU 数值验收仍待完成。 |
| `phase_convention`, `path_boundary` | 默认分别为 `anchor` 和 `periodic`；相位也可选 `none`，空间路径边界也可选 `open`。相位必须与 `Baryon` 的设置一致。 |
| `ne_block_size`, `spatial_block_size` | 生产器每腿模式块和空间块大小，默认分别为 16 和 256。`ne_block_size` 可用一个整数或三个正整数。 |
| `cache_bytes`, `transport_cache_bytes` | `cache_bytes` 限制 adapter 保留的单个最终结果，默认 64 MiB；较大的结果不缓存。`transport_cache_bytes` 控制一次底层计算的路径输运缓存，默认 64 MiB。 |

## 读取和限制

`describe_block(component, momentum, t, "vvv", ...)` 只返回 shape、dtype、选择和已知基底标签，不读取规范场/本征向量，也不生产 elemental。`block(...)` 同步计算并返回拥有的 complex128 数组。`usedNe` 选择每腿有序前缀（整数或三元组）；需要任意有序选择时传 `mode_indices={"mode0": ..., "mode1": ..., "mode2": ...}`。两者不能同时指定。NumPy 返回数组标为只读；CuPy 不提供同等的写保护标记，调用方均应把结果视为只读。

输入由调用方持有，计算期间须保持稳定。若原输入内容有变化，先调用 `clear_cache()`；`load(cfg)` 会清除缓存，但不能改绑到另一个组态。`close()` 只释放 adapter 的结果缓存，不关闭调用方的 field handles。

此接口只生产请求的 VVV 空间块，不自动执行 stout smear 或 SU(3) 投影；调用方须提供所需规范场版本。它不写入或登记持久产物，也不为结果分配科学数据身份。小模式选择直接限制底层生成器读取及输出，不会先生成完整 `Ne^3` elemental。总体内存仍不受 `cache_bytes` 约束：调用方数组、生成器工作区、所选输出和其它缓存均占用内存；`load(cfg)` 规范场 handle 可能先装入完整时间数组。小格点 NumPy CPU 独立 epsilon/Fourier 参考的最大绝对差为 `8e-15`，与公开投影生产路径的对照差最大为 `3e-14`；GPU、真实系综和生产规模仍待验证。

## 验证范围

新增的 9 项 CPU 测试覆盖合成复数场、独立颜色行列式与正号 Fourier 相位、复系数规范核、实际基底标签、借用输入缓冲、缓存失效、源端伴随和交换腿槽、全部 64 个两段径向块及整体恢复，以及无数据读取的规划。另有 19 项既有模块边界、重子顶点和径向分块测试通过。规范核测试使用 `(Lx,Ly,Lz,Lt)=(3,2,1,2)`，三腿模态数为 `(3,4,2)`；关联函数测试使用 `(3,1,1,2)` 和每腿 3 模态。这些合成场验证接口及约定，不代表真实系综结果。

完整 CPU discovery 使用 `2390272` 基线及叠加本次实现的隔离快照：分别运行 1547 和 1556 项，两次均有 3 个失败、9 个错误和 24 个平台条件跳过；失败名单完全一致，本次没有新增失败。全仓回归尚未通过，条件跳过也不计为 GPU 验证。
