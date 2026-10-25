# 重子 elemental 的表格前端

常规测量入口与介子相同：`contraction.correlator.Elemental, calculate`。用 `baryon_preset(version="v1").select(...)` 选择空间配方，`Elemental(gauge, basis, table, momentum_dict, directory=...)` 自动识别三腿目录；basis 可为一个共享 V reader 或三个有序 V reader。绑定 `Baryon` 后直接 calculate，后端复用已保存结果、补缺及恢复。同一顶点三腿等时；多个源汇时间对可以由同一完整计划固定合批和尾批，VVV 不涉及随机采样计权。NumPy/V100 的局域 VVV 小规模 complex128 档案加载后首次补产、复用和当前传播子回放已验证，不能外推为生产规模性能验收。见[自动准备](../../../../docs/modules/correlator/managed_elemental.md)和[版本化目录](../../../../docs/modules/operators/elemental_catalogs.md)。

以下为内部数值接口：`lqcdweave.production.elemental.api.BaryonElementalGenerator`。

自旋不进入 elemental 生产。完整 64 行自旋配方通过 `baryon_spin_operators(version="v1")` 选择（双夸克低秩配方用 `baryon_diquark_operators`），配合本页空间 component 交给 `recipe.insertion` 或 `Insertion.from_irreps`；见[自旋目录](../../../../docs/modules/operators/baryon_spin_catalog.md)。
组合交换投影 `insertion.project_exchange(...)` 会给出覆盖被交换腿位置的 `result.operator_table`，普通测量将它交给 `Elemental(...)`，由后端自动补产；相关组合系数已在算符自旋项中，空间表不重复计权。见[组合交换投影](../../../../docs/modules/operators/baryon_exchange_projection.md)。

## 默认与可选时间选择

默认完整时间，不需要时间参数。明确稀疏化时传 `selection=load_time_selection(..., Lt=physical_Lt)`，直接数组按该记录紧凑存储；`iter_slices()` 返回真实 t 与借用结果，`calc_path_blocks(t)` 仍接收物理时间。`Lt` 不变，`Nt` 为输出槽数。

额外输入覆盖或不同槽序使用 `eigenvector_selection`（重子可为三腿记录）；介子点数组可用 `point_selection`，SpatialPointSet 保持自身 times。句柄使用 time_selection，缺省为完整按序输入。跨时间路径先求真实 t±offset 再查槽，缺片报错；不会重算输入、按槽邻接或补零。

## 初始化参数

- 必填：`gauge_field, eigenvectors, operator_table` 和动量表；使用 `momentum_dict=` 或旧的 `momenta` 有序列表，二者只能选一个。
- `momentum_dict` 可直接与 correlator 的 `Insertion` / `InsertionGaugeLink` 共用；字典值的插入顺序决定动量槽，key 不作为槽号。`generator.momenta` 保存实际使用的有序 XYZ tuple。
- eigenvectors 默认一个实际基底供三腿共用；显式三元组则分别使用三腿基底。
- `Ne=None`：继承各实际基底模态数；整数共用截断，三元组分别截断。
- `array_module=np`，`ne_block_size=16`（整数或三元组），`spatial_block_size=256`。
- `phase_convention="anchor"`：当前内核接受 anchor/none；`path_boundary="periodic"`：periodic/open。前端虽转交相位选项，midpoint 仍会被内核拒绝。
- `cache_bytes=64*1024**2`：运输缓存；`stream_eigenvectors=False` 为默认值。启用流式读取时，基底句柄必须同时提供正整数 `Ne` 与 `read_modes(key,t,start,stop)`，供内核识别模态数并分块读取。
- `vector_scratch_limit_bytes=None`：流式模态暂存限制，要求 stream_eigenvectors=True；不代表所有内存上限。
- `flavors=("q","q","q")`：兼容腿标签；物理味道属于算符层。
- `write=None`：同步消费三腿输出块，由脚本保存；启用时返回 None。
- `lattice_size=None`：数组推断尺寸；惰性句柄须明确提供尺寸。

## 表与唯一计算方法

operator_table 每条记录包含 component、path、coefficient；path 是三条有序空间路径的三元组。也支持 id/terms（项中 path 或 paths）的嵌套格式。相同 component 的**相关三腿路径三元组**按复系数相加，不是分别对三腿求和后相乘。距离平均在构表时展开成三腿路径三元组及权重，计算方法不接 distances 参数。

```python
generator = BaryonElementalGenerator(gauge, vectors,
    momentum_dict=momentum_dict, operator_table=operator_table,
    Ne=Ne, ne_block_size=16)
for t in times:
    values = generator.calc_path_blocks(t)["vvv"]
    save(values)
```

- `calc_path_blocks(t)` 是唯一计算入口，仅支持 VVV；没有 blocks、points 或 calc_derivative。
- 结果 `(component,momentum,N0,N1,N2)`；保留腿顺序、相关路径、复系数，没有额外体积因子。
- 数组自动加载；句柄使用 load(configuration)。规范场变换支持 project_SU3、stout_smear。
- write 签名 `(component,momentum,"vvv",three_slices,value)`，借用块须同步消费。
- 模态和空间块、缓存、流式输入和输出块分别核对；GPU、真实规模及持久身份须另行验证。

数组输入下，非流式基底形成有序前缀快照；流式读取同步消费模态块。距离配方在表中展开，例如每腿非零直线距离的六方向平均系数为 1/6，再组成相关三元组；实际权重仍由所选配方决定。

保存按[保存与持久交接](production.md)选择。`plan_indexed_baryon_elementals` 可规划 raw_paths/projected/distance 产品；生产脚本按选定格式组合计算器与通用 writer。核对 `production/elemental/frontend.py`、`baryon.py` 与 `tests/production/elemental/` 的对应测试。
