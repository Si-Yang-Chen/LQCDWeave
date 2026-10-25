# 时间选择：从生产到结果保存

默认使用全部物理时间片，不搜索或随机选择子集。只有显式提供 selection 时启用稀疏时间。
本页按普通用法、显式稀疏、持久化与高级限制逐层展开；各产品的精确接口仍以其 L2 为准。

## 1. 最少参数：保持全时间

现有生产器、preset 数据源与 calculate 的默认调用保持全时间。
不传 selection，不把物理 lattice/Lt 改为较小 Nt，也不用给普通收缩内核增加 physicalLt。

- [EV 生产](../eigenvector/laplace_eigvector.md)：规范场到 Laplace 基底和本征值
- [空间点集](../spatial_sampling/spatial_sampling.md)：只在点端测量中需要
- [介子 elemental](../elemental/elemental.md)与[重子 elemental](../elemental/baryon_production.md)
- [传播子](../perambulator/propagator.md)：VSV、PSV、VSP、PSP
- [数据导入](../storage/preset.md)、[收缩](../correlator/correlator_frontend.md)、[结果保存](../storage/correlator_results.md)

## 2. 显式稀疏：每个 cfg 保存并复用一份实际顺序

共同参数和模板搜索可以放在 cfg 循环外；模板与实际 cfg 记录不同。

```python
from lqcdweave.sampling.temporal.api import (
    select_t, exponential_pair_q, realize_time_selection,
)
from lqcdweave.storage.time_selection import save_time_selection, load_time_selection

score = exponential_pair_q(ell=ell, differences=desired_differences)
template = select_t(Lt=Lt, Nt=Nt, key=score)
if template is None:
    raise RuntimeError("没有满足要求的时间方案")

# 仅首次创建此 cfg 的选择文件时运行；后续直接执行下面的 load。
selection = realize_time_selection(template, cfg=cfg)
save_time_selection(selection_file, selection)
selection = load_time_selection(selection_file, Lt=Lt)
```

已有文件直接 load，不重新搜索、不再次平移、不覆盖。`realize_time_selection` 与
`select_t(..., cfg=cfg)` 使用相同 cfg→PCG64 周期平移，前者不重复方案搜索。
选择 NPY 仅保存一维有序整数 times，因此读取显式绑定物理 Lt；正式数据集登记同时保存
物理格点、完整有序 selection 和槽布局。文件路径与相同 Nt 都不证明科学身份相同。
选择 NPY 若因写入中断或损坏而读不回，读取明确失败；不会自动重新抽样或覆盖。
后续计算应继续使用原始实际记录，不能用新的平移接管旧槽位。

例如物理 Lt=128、Nt=40：规范场和 Dirac 求解场始终长128；选中基底、点集、elemental
和源汇投影的时间轴长40。slot i 对应 selection.times[i]，跨边界的顺序不会排序。

## 3. 生产边界使用物理坐标，消费边界使用槽

- EV 的 selection/iter_slices 仅求解所选物理片，基底与本征值保持同槽模式顺序
- 空间采样接受显式 selection，保存 Nt 行；固定 seed 保持原全时间随机流，因此未选片的随机抽样仍被消费，不宣称减少 RNG 工作
- 同片介子/重子 elemental 接受 selection；实际基底/点输入可通过显式记录映射，额外基底覆盖与锚点选择分别声明
- PR 接受 source_selection/sink_selection。源时间调用仍是物理时间；每个源求解保持完整 Dirac 格点，但只投影/提取选中的物理汇片，输出按 sink_selection 顺序
- 未带时间记录的生产输入句柄继续表示完整规范物理顺序。压缩输入必须显式附带映射，不能按 shape 猜测。流式 read_modes/read_points 的时间参数仍是物理坐标，由适配器映射到存储槽
- 面向收缩的 Array/preset/Dataset reader 用纯槽索引，执行长度为 Nt；实际 Lt 和有序选择只保留在边界元数据中。calculate 的普通同片计算不需要用户传物理 Lt

源与汇可以使用不同记录和数量。传播子实际端点基底/点池身份仍分别绑定；同一个槽号在不同
记录中不一定表示同一物理片。绑定阶段检查同一时间端点上的输入兼容性；计划身份包含完整
有序选择，重排、不同平移或不同物理 Lt 不能因为 Nt 相同而复用旧绑定。

## 4. 结果默认保留源槽×绝对汇槽

新结果优先使用 source slot × absolute sink slot；两条独立扫描轴直接保存，不先平均。

```python
from lqcdweave.sampling.temporal.api import reconstruct_two_point_from_coordinates

# data 的时间轴0/1分别是源槽和绝对汇槽；其余算符/动量/组态轴保持。
result = reconstruct_two_point_from_coordinates(data, selection)
values, counts = result.values, result.counts
```

物理差值是 `(sink_times[k] - source_times[i]) % physicalLt`。按每个实际 dt 的贡献数归一，
不把 Nt 槽差当物理 dt，不恢复未计算的逐源数据。缺失差值默认报错，可显式请求 NaN。

已有“源槽×循环差槽”结果必须明确 `time_layout="cyclic_difference"`；其汇槽为
`(i+j) % Nt`。该 j 不等于物理 dt。旧 `reconstruct_two_point(...selected_times=...)`
仅保留这一明确的兼容语义。不要凭 Nt×Nt 的 shape 自动判断布局。

`save_correlator` 默认保持原容器、算符轴和时间坐标。只有显式开启 `average_tsrc` 才在保存
边界平均；槽域必须声明 `coordinate_domain="slot"` 和对应 selection，物理域必须声明
`coordinate_domain="physical"` 和物理 Lt/边界。平均后的文件记录计数和布局，不能重新恢复
原始逐源时间轴。具体签名与不对称端点用法见[结果保存 L2](../storage/correlator_results.md)。

## 5. 持久化、导入与恢复

通用 `DatasetSpec/DatasetWriter/DatasetReader` 仍是唯一持久契约与读写机制。
时间槽保持现有轴名，坐标为 `range(Nt)`；`conventions.time_coordinates` 记录
`domain="slot"`、`layout="absolute"` 和各轴的 `{Lt, times}`。原 `configuration.lattice`
保留物理格点，实际父基底、点池、规范场、Dirac 参数与非时间轴不变。

builder 的 selection 参数生成上述契约；高级自定义产品可用
`bind_time_selections(document, {"time": selection})`，前提是输入 document 的该轴已经
按 selection.times 有序声明。该函数只转换契约，不移动数值数组；禁止把旧槽随意改绑新选择。
`dataset_time_selection(spec, axis)` 读取映射，`dataset_physical_times` 返回实际物理顺序。

完成覆盖按 writer 实际完成的首轴槽登记。发生中断时须使用同一选择和相同输入；可恢复范围
由当前 writer/runtime 接口和验收决定，不将旧布局的 resume/publish/回执测试视为新格式保证。
不可把目录存在、进程成功、describe 或零填充视为完成。
裸 NPY 不含完整科学元数据；导入时显式附带选择和时间布局，未知基底/点池身份仍保持未知。

## 6. 跨时间端点与资源边界

跨时间路径先从物理锚点 t 按路径计算实际端点 t±offset，再映射到实际输入的槽。
若端点不在锚点 selection，可给输入基底/点集提供额外端点覆盖；不存在时明确报错。
不能拿相邻槽替代物理 t+1，也不能补零或猜测相同基底。开放边界的原几何排除规则与
“输入缺失”是两种情况，不能互相代替。

- 稀疏 EV 减少求解片数；稀疏 PR 减少所选源求解次数和所选汇投影，单次 Dirac 仍覆盖完整 Lt
- Nt 输出不表示规范场、求解工作区、常驻输入、共轭副本和缓存都缩为 Nt；分别按实际路线计量
- 内存估算沿用已有资源记录，包含完整父产物和求解场；不是 RSS/本机库/CUDA 分配硬上限
- 小规模 CPU 独立参考与模拟 GPU 接口测试不代表真实 GPU/QUDA/MPI、物理系综或生产规模验收
- 重建给出选中配对的时间平均估计，不保证单 cfg 等于完整全源平均，不插值缺失数据

精确搜索/映射见[时间选择](select_t.md)、[TS L2](temporal_sampling.md)；
每个生产器的可用数据形状、缓冲所有权、平台及跨时间支持仍按对应模块 L2 核对。

### 旧专用接口的明确边界

压缩跨时间 Current 使用原生 Meson/Current + DatasetElemental 的物理端点查表绑定；
旧 `load_registered_directed_current` 与其专用 pair/chain 数值入口依赖连续物理 ±1 索引，
只接受完整物理时间，显式拒绝 compact slot 数据。不能修改 Lt 或补零后绕过此检查。
旧固定两轴 persistent Diagram helper 的 compact 路径要求共同有序选择；不同源/汇
记录或跨时间额外端点通过原生 calculate/保存计划路径处理。

稀疏 P/P COO 的直接计算/扫描使用现有稀疏描述；当前 complete-cfg 档案要求可描述的准备配方，
COO 路线尚未具备该配方，会在构建/保存完整计划前明确拒绝。V/V 与具备 dense 准备配方的
混合 V/P 路线保存非仿射端点表并直接重放；不能把这一验收推广到尚不支持的稀疏档案。

紧凑数据的 SpectralProfile 加权项目前明确拒绝：profile 数组虽然有时间轴，但尚未声明其
物理/槽选择记录，不能从数组长度或与 reader 相同的 Nt 推断对应关系。完整物理时间的
profile 和不使用 SpectralProfile 的模式分块保持原支持。

同片三腿 Baryon 可与具有物理时间偏移的 Meson/Current 共同进入原生图；端点通过已有
B/E 绑定映射到各条实际传播子轴，保持 source dagger 与自旋/模式契约。Baryon 自身的
非零时间位移腿仍不在空间 Baryon 契约内。该 provider 尚无 complete-cfg 准备配方，
保存完整计划时保留既有明确拒绝，不用 meson-only 档案验收替代。
