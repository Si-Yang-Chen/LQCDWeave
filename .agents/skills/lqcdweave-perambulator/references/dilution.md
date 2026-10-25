# V/D 稀释向量与 pair 存储

这是[常规 VSV 路线](../SKILL.md)之外的高级内容。实际 D 向量由独立 [Noisevector skill](../../lqcdweave-noisevector/SKILL.md) 生产和保存。PR 把 D 作为空间颜色向量消费，不生成混合系数、不重组父模式、不缩放或归一化 D、不乘估计量权重，也不平均随机实现。

## 绑定实际 V/D 端点

`VectorBasis(handle, count=None, endpoint=None)` 表示一个实际有序 V 或 D 向量集合。优先使用显式或句柄实际 `endpoint.kind`，没有 endpoint 时再看 `dilution_source` 或 `vector_kind`；普通 Laplace reader 默认为 V。登记为 D 但缺少 dilution 来源时仍按 D 处理，来源保持 unknown，不补造。自定义 D handle 必须携带其实际 D endpoint。句柄提供的 `.Ne` 在 D 情况下表示可用向量数 `Nd`。构造器不绑定源汇，也不选择产品；每次计算通过源对象与有序汇列表明确身份。

例如，D 源同时投影到 V 汇和点汇，分别得到 VSD 与 PSD，共享同次 D 源求解：

```python
from lqcdweave.production.perambulator.api import (
    VectorBasis, PointBasis,
)

vsd, psd = generator.calc(
    source_time,
    src=VectorBasis(dilution_reader, count=Nd),
    sink=[
        VectorBasis(v_sink_reader, count=Nv),
        PointBasis(point_sink_reader, count=Np_sink),
    ],
)
```

每项是 `PropagatorResult`，`.product` 按实际 endpoint 显示产品，`.values` 拥有该汇的完整投影数组，`.source` / `.sink` 保留实际端点。不同 V/D 表示组合可用不同源和汇对象；九类产品为：

| 汇端 / 源端 | V | P | D |
|---|---|---|---|
| V | VSV | VSP | VSD |
| P | PSV | PSP | PSD |
| D | DSV | DSP | DSD |

## 保留 D 身份和原始幅度

持久化时，每个 D endpoint 必须记录实际 `data_id`、有序向量标签及 `vector_labels`、`mode_groups`、`parent_basis_id`、`realization_id`、`noise_pool_id` 和 `normalization="raw"`。`NoisevectorGenerator.describe()` 提供父基底、分组、标签和原始幅度约定；`data_id`、实现 ID 与 pool ID 须来自实际生产流程。seed 或数组 shape 不能代替这些身份。

由 D handle 自动得到的端点应从所生成结果的 `.source` 或 `.sink` 读取。自定义 handle 可在 `VectorBasis(handle, endpoint=d_endpoint)` 绑定真实 endpoint。若句柄提供 `PropagatorEndpoint`，生成器会保留有序身份；选择 count 前缀时会同步截取 D `vector_labels`。存储必须比较完整 endpoint 记录，不能把不同随机实现或每时间 pool 合并成猜测出的公共 ID。

```python
from lqcdweave.production.perambulator.api import PropagatorEndpoint

noise_info = noise_generator.describe()
d_endpoint = PropagatorEndpoint(
    "D", actual_d_data_id, range(Nd),
    metadata={
        "vector_labels": [list(label) for label in noise_info["vector_labels"]],
        "mode_groups": [list(group) for group in noise_info["mode_groups"]],
        "parent_basis_id": noise_info["parent_basis_id"],
        "realization_id": actual_realization_id,
        "noise_pool_id": actual_noise_pool_id,
        "normalization": "raw",
    },
)
d_source = VectorBasis(custom_d_handle, endpoint=d_endpoint)
```

## 高模父空间

low/medium/high 投影通过 `PointBasis(..., highmode_basis=high)` 输入，`high` 由 `HighmodeBasis(parent_handle, ModePartition(parent_id, low_modes, medium_groups))` 构造。它只在 P 端投影完整 low+medium 父空间，不使用 `D D†`，也不增加 medium 父模式的 Dirac RHS。父句柄须声明 `basis_orthonormal=True`；D 端须携带完整且实际的 `DilutionSource`。源汇父空间不同就分别在源、汇 `PointBasis` 绑定实际 highmode basis。

投影后 P endpoint 的 `high_mode_projection` metadata 是科学身份的一部分，pair storage 与 Gamma5 reversal 会保留；不能用原始点 endpoint 替代。elemental 仍使用原始点坐标和 raw spatial kernels，权重由既有 `elemental_weighted` 约定处理。已有完整父传播子时，`project_highmode_block` 可从实际 overlap 与 sink/source/cross auxiliary blocks 派生 PSV/VSP/PSD/DSP/PSP，并检查物理时间对和父空间身份。

开启 `stream_eigenvectors=True` 且父句柄支持 `read_modes` 时，每个 high endpoint 最多保留一个完整父空间物理时间片；其它句柄 eager 加载。这个片段独立于普通 V/D sink cache 限额，资源规划还须计入 RHS/解场、输出、反向写出暂存和求解器工作区。投影点 H 仍采用 P schema；HSV/HSD 必须保留全有向时间覆盖，只有相同实际端点的 HSH 可使用自反产品时间三角。见[H 路线](../../../../docs/modules/perambulator/point_highmode.md#h-contract)和[存储约定](../../../../docs/modules/storage/data_storage.md#h-storage)。

## 六类保存代表与九类读取视图

由 `calc` 显式指定源和完整汇列表，同源多汇共享求解；随后将各结果的 `at(t_sink)` 交给对应的 `PropagatorWriter` 保存。pair 写入不调用生成器，也不根据缺失覆盖自动安排求解。

`propagator_pair_spec` 为单个保存产品声明显式物理时间覆盖。首轴为 `time_pair`，每个坐标为 `(sink_time, source_time)`。固定保存产品是 `VSV`、`PSV`、`PSP`、`VSD`、`PSD`、`DSD`；六类代表可通过 Gamma5 关系提供九类读取视图。

```python
from lqcdweave.production.perambulator.api import (
    PropagatorWriter, propagator_pair_spec,
)
from lqcdweave.storage.dataset import DatasetWriter

time_pairs = [(t_sink, t_source)
              for t_source in source_times for t_sink in sink_times]
spec = propagator_pair_spec(
    "VSD", configuration=configuration, sink=v_sink_endpoint,
    source=d_endpoint, time_pairs=time_pairs, gauge_id=gauge_id,
    dirac=dirac_definition, gamma5=gamma5, spin_basis=spin_basis,
    gamma5_hermitian=True,
)
with DatasetWriter(output_dir, spec, provenance=provenance) as dataset_writer:
    pair_writer = PropagatorWriter(dataset_writer)
    for source_time in source_times:
        [vsd] = generator.calc(
            source_time, src=d_source, sink=[VectorBasis(v_sink_reader)],
        )
        for sink_time in vsd.sink_times:
            pair_writer.write(vsd.at(sink_time))
    data_id = dataset_writer.publish()
```

`PropagatorResult.at(t_sink)` 返回带实际时间和端点的 `PropagatorBlock`；writer 同步消费块。`PropagatorWriter` 也会将 VSP、DSV、DSP 转存到反向保存代表 PSV、VSD、PSD，并交换时间与端点。`PropagatorReader(readers).read(product, sink_time, source_time, *, sink, source)` 返回直接或 Gamma5 派生的 block；例如 DSV 会寻找反向时间对上端点精确交换的 VSD。spec 要求显式 Gamma5 矩阵和 Gamma5-Hermitian 作用量；缺失输入不启动求解器。reader/writer 生命周期、发布和 resume 由调用方管理。

关系 `VSP(t_sink,t_source) = gamma5 PSV(t_source,t_sink)† gamma5` 需要同时交换物理时间和真实 V/P endpoint，复共轭并交换完整自旋、模态、点/颜色端点轴，再在两自旋轴作用 Gamma5。相同规则覆盖 DSP/PSD 与 DSV/VSD；VSV、DSD 和 PSP 在反向时间对自反。高模 P 必须交换同一完整父空间投影身份；raw P 不能替代投影点 endpoint。

生成器绑定 Gamma5 信息后，每个 `PropagatorResult` 还可直接调用 `.reverse(t_sink)`。省略时间返回源结果中每个物理汇时间对应的反向 block 列表；这些 block 的源时间互不相同，不构成一条反向源时间行。仅当后端声明 `gamma5` 与 `gamma5_hermitian=True` 时可用；QudaBackend / DenseWilsonBackend 内置该声明，普通自定义后端可在构造器显式传入两项。

`coverage(product, *, sink, source)` 与 `missing(product, time_pairs, *, sink, source)` 只检查 metadata。缺失覆盖应写入分离的新 pair dataset，再与旧 reader 合并，不能覆写已发布数据。`source_time_blocks(...)` 只把已有 dense 源时间行拆为借用的单时间对视图，不避免完整输出分配。六类 pair layout 可用 `production.perambulator.pair_planning.plan_indexed_perambulator_pairs(index, spec)` 做只读登记覆盖预览；查询结果不触发补算。生产范围由调用方显式指定，通过 `calc` 计算，再用 `PropagatorWriter.write` 保存已有结果。调用方负责发布和通用存储恢复。旧 indexed planner 仅预览四种 V/P 产品的 `source_time` 行布局，不能与 pair layout 混用。

普通 Meson Diagram 可通过 `adapters.propagator.PropagatorPairReader` 消费实际 pair 产品。多组实际 V/D 端点通过 `endpoints={"vsd": (v_endpoint, d_endpoint), ...}` 显式绑定。有向测量权重属于 elemental 局部补偿及图级分量纳入残差，传播子不乘权重。V/D 消费端约定见[elemental dilution 说明](../../lqcdweave-elemental/references/dilution.md)；独立全空间逆矩阵的 CPU 参考见[perambulator dilution 回归](../../../../tests/production/perambulator/test_dilution.py)，覆盖产品和时间/端点交换。GPU、QUDA、MPI 与生产规模仍按各自任务单独验收。
