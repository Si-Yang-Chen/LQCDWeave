# 模态端与点端 elemental

用户明确需要点端数据时读取，包括采样流和已有确定点集的数据准备。核对两端实际点集、顺序、时间覆盖、路径和动量，不自动引入随机采样。

唯一采样入口 `sample_spatial_points(SpatialSamplingDesign(...), seed=..., selection=...)` 返回 `SpatialPointSet`，其 `coordinates` 为 `(time,point,xyz)` 只读 `int64`。已有坐标按 `SpatialPointSet(lattice_size=lattice, times=tuple(range(Lt)), coordinates=coords)` 构造；`SpatialPointHandle` 用于读取或适配，不负责采样。

已有点集通过初始化参数 points 提供，数组布局 `(time,point,xyz)` 或 SpatialPointSet；同一点集按两端真实时间使用。前端没有 usedNp 或独立 points_right 参数；需要前缀时在输入适配处保留实际选择与身份，核对 reader 能力。独立左右点集属于另一种交接需求，不能虚构前端参数。按数据需求选择：

| 所需数据 | 公开blocks选择 |
|---|---|
| 模态到模态 | v2v |
| 模态到采样点 | v2p |
| 采样点到模态 | p2v |
| 采样点到采样点 | p2p |

```python
generator = ElementalGenerator(gauge, vectors, momenta, operator_table,
    points=points, blocks=("v2v", "v2p", "p2v", "p2p"))
blocks = generator.calc_path_blocks(t)
```

这行展示可选项，只计算任务所需组合。构造时接入点集再 load；左点取 anchor time，右点取路径真实 field endpoint。V/V、V/P、P/V 需要实际基底；纯 P/P 可令 eigenvectors=None。

V/V 布局 `(component,momentum,Ne,Ne)`；V/P 为 `(component,momentum,Ne,Np_right,3)`；P/V 为 `(component,momentum,Np_left,3,Ne)`；P/P 为按 component/momentum 索引的 SparsePointBlock，保存 rows、columns、values、point_shape。copy() 保存独立稀疏块，to_dense(array_module=...) 是额外稠密分配的明确选择。

初始化可传 `write(component_index,momentum_index,sector,slices,value)` 同步保存；P/P 的 slices=()。路径、相位、边界、后端及模态块沿用[介子选项](meson.md)。独立左右点截断不等于独立左右模态截断。

same/independent采样和同池点重合的权重不同，参数共用或独立也必须正确声明。空间生成器不负责最终估计量权重；计算入口按采样需求展开，不让用户手写分区或重复乘权。端点不受支持时明确缺口，不能用锚点数据替代。

需要独立左右点集时可核对现有登记点端入口及公开数值适配边界；它的端点选择能力不能反推为表格前端已支持。保存与消费者按[保存与持久交接](production.md)接线，保留 current schema 和真实端点；现有特定时间单链支持不证明任意跨时间流已接通。重子前端仍只有 VVV。

<a id="h-elemental"></a>
## H 传播子配套的 elemental

H=Q eta 的[投影定义](../../../../docs/modules/perambulator/point_highmode.md#h-contract)属于传播子。Elemental
仍用同一实际 P 坐标、顺序、点池和物理时间，生产原始空间核；不把 H 当作新的正交 V/D
向量输入，不新增 `h2v/h2h`，不在已有 H 传播子旁再次施加 Q。含 D 时沿用
`vector_bases` 和[九扇区接线](dilution.md)。

`elemental_weighted` 继续统一控制 D 与点采样的局部计权，correlator 补整图余权；投影
父空间的大小不替代点的包含概率。改变 D 每组抽取列数，需要相应实际标签和统计系数，
但不改变固定父空间 Q。改变 P 子集保留实际原点标签与池身份，不按子集另造独立池。
结构标签描述实际算符的原始空间核；不能因为 H 是高模就宣称 H 向量正交或核完全对角。

普通 Diagram 的 D 选择仅支持完整组保留/重排；组内减少混合列需要显式联合列选择估计量，不能直接裁数组后沿用原 m 和来源。专用扫描实现的支持不等于普通 reader 支持，见[D 来源契约](../../lqcdweave-elemental/references/dilution.md#同池混合列包含系数与范围)。
