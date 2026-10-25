# 测量中的 elemental 自动准备

算符构造允许自定义路径、复系数和组合。算符确定以后，用户绑定实际输入并选择后端、计权及输出等已有选项；不编写 prepare、producer、loader 或保存回调。缺少内部交接能力时应补齐后端，不能要求用户另写流程绕过。

介子、空间流和重子 VVV 共用 `contraction.elemental_preparation.Elemental` 声明输入。它接受已打开的登记数据 reader；输入 reader 在测量结束前保持有效。已有完整 elemental reader 仍可直接绑定。

```python
from lqcdweave.contraction.binding.api import Meson
from lqcdweave.contraction.correlator import Elemental, calculate

elemental = Elemental(
    gauge, {"V": low_basis, "D": dilution_vectors}, operator_table, momentum_dict,
    points=points, directory=shared_elemental_directory,
    stout=(20, 0.12), phase_convention="anchor",
)
vertices = {
    "source": Meson(elemental, source_operator, source=True, phase_convention="anchor"),
    "sink": Meson(elemental, sink_operator, phase_convention="anchor"),
}
result = calculate(
    correlator, vertex_map=vertices, propagator_map=propagators,
    configuration=cfg, time_axes={0: source_times, 1: sink_times},
    elemental_weighted=True,
)
```

只提供 V 时，`bases` 可以直接是低模 reader；没有点端则省略 `points`。可用扇区由实际 V/D/P 输入确定。算符与图决定请求的分量、动量、扇区和物理时间。`operator_table` 和 `momentum_dict` 来自算符构造的配套定义，不重新手写系数。计权只在 `calculate` 设置，原始声明不变；D/P 系数沿用共同数值实现。基础输入、stout 和相位属于科学定义，必须明确。

## 后端行为

1. 用元数据描述需求，`prediction["input_preparation"]` 给出所需时间、已有覆盖与缺项准备内存；明显超预算在规划阶段拒绝。`execute=False` 不读取规范场或基底数值，不创建输出目录，不生产。
2. 执行前汇总真实读取请求，包括坐标合批后的时间和源端 dagger 对应的扇区。
3. 调用既有 Dataset 规划与执行服务，检查科学身份与完成覆盖。兼容结果直接复用；已有部分时间可以复制，生产器只处理缺失时间。
   同一 reader 已接受的发布产物按原科学家族索引及实际时间绑定复用，避免完整 cfg 统一准备后各模板重复查询登记；新时间、计权或配方等科学定义变化仍重新核验。关闭 reader 时释放这些绑定。数值读取仍走标准 Dataset view，并返回独立数组。
4. 缺项生产只读取一个物理时间片，执行已声明的空间 stout，并调用 ElementalGenerator。同一规范场及 stout 配方的准备片也作为标准 Dataset 产物共享，不因另一个分量、扇区或消费者重复涂抹。分量、动量和扇区按请求确定；保留完整登记模态轴，以便不同缩并截断复用。
5. 标准 DatasetWriter 预分配并逐时间写入完整 cfg NPY，发布和索引成功后开始缩并。异常留下标准续算状态，未发布数据不交给消费者。

共享目录使用现有 AtomicOpen 按 cfg 协调消费者，首次或新增覆盖的持久检查在锁内进行；索引按科学配方分组，避免每个准备阶段扫描整个 campaign。已接受的绑定无需再次查询登记。不同 cfg 可并行，同 cfg 缺项生产顺序执行。没有第二套缓存格式、完成台账或用户回调。规范场、基底、点集、路径系数、动量、stout、相位、计权与结构声明都参与身份。

生产使用本次执行的数组后端。准备阶段单独检查一个空间片的保守逻辑内存估算，超额报错；它不是 RSS 或不透明 BLAS 工作区的严格上界，准备耗时尚未加入缩并预测成本。

## 当前范围与验收

支持已登记全时间规范场、V/D 基底、可选点集；九扇区空间介子/流；普通 CPU 坐标合批、固定坐标、完整 cfg 计划和档案回放。D 需要实际 dilution 来源，点预权需要登记的采样设计。只提供预制 reader 而没有基础输入时不能补产。

重子选择 `baryon_preset(version="v1").select(...)`，同一 `Elemental` 根据目录的三腿配方使用既有重子内核；bases 接受一个共享 V reader 或三个有序 V reader。三腿保留各自实际基底身份和模态顺序，不取最小 Ne。同一重子顶点的三腿仍等时，任务可以声明多个源汇时间对，由完整计划固定坐标批次及尾批。它与介子共享缺项补产、Dataset 保存、续算和 stout 准备。跨时间路径及 D/P 重子尚未支持。NumPy 数值、覆盖复用与恢复已测试；真实 V100 已覆盖下述小规模档案回放链，DCU、MPI 和自动补产的生产规模尚未验收。

独立空间矩阵参考覆盖九扇区、D/P 计权、乱序输入时间和模式标签；另测只规划零生产、跨消费者复用、缺项扩展、中断恢复、stout 身份、预算拒绝、普通缩并和完整计划回放。见 `tests/test_managed_elemental.py`。

重子另以独立颜色行列式验证三类目录配方、三腿独立基底、源汇收缩、MS/MA 交换矩阵和全部 S2 槽位；自动准备测试见 `tests/test_managed_baryon_elemental.py`。

完整计划回放与自动补产的测试见 [test_managed_baryon_plan.py](../../../tests/workflows/test_managed_baryon_plan.py)。

介子自动准备 CPU 回归见[test_managed_elemental.py](../../../tests/test_managed_elemental.py)。
重子自动准备 CPU 回归见[test_managed_baryon_elemental.py](../../../tests/test_managed_baryon_elemental.py)。
