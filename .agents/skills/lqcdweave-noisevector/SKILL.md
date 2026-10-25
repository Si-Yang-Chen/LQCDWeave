---
name: lqcdweave-noisevector
description: "从已有实际 Laplace 基底按显式 dilution 分组生成 QR 混合向量，编写生成、保存和读回脚本；支持 medium 起点及非连续分组，不负责求解传播子或自动选择重要性方案。"
---

# LQCDweave · Dilution 向量生产

共同任务路由、前端边界、保存/身份及执行范围统一见[技能共同约定](../README.md)。只读取当前任务的条件参考。

父分组、low/medium 范围和混合列数由当前测量定义明确给出，不从课题配方或既有生产配置继承。固定父分组不等于固定混合列数；实际列数仍须满足所测相关函数的统计支持，不能自动推断每组一列。

使用独立 `production.noisevector.api.NoisevectorGenerator`。已有实际 D 数据直接读取，
不因需要传播子而重新抽样。已有父基底交给向量 reader；需要计算 Laplace 基底时才使用
[EV skill](../lqcdweave-eigenvector/SKILL.md)。

## 常规调用

```python
from lqcdweave.production.noisevector.api import NoisevectorGenerator

generator = NoisevectorGenerator(
    basis_reader, dilution=(parent_counts, vector_counts), mode_offset=N_low,
)
generator.load(configuration)
d_vectors = generator.calc(seed=seed)
description = generator.describe()
```

生成和保存分开调用。按下文保存契约从实际 `description`、父基底和 cfg 构造
`output_spec` 后，脚本同步写入结果，不调用基础产物生产 runner：

```python
from lqcdweave.storage.dataset import DatasetWriter

with DatasetWriter(output_directory, output_spec, provenance=provenance) as writer:
    for slot, row in enumerate(d_vectors):
        writer.write(slot, row)
    data_id = writer.publish()
```

本例为 NumPy；设备输出先显式同步转到主机。`output_spec` 必须保存同次生成的
分组、实际列标签及每时间 pool 身份，不能只保存数值 shape。

parent_counts 给出每组父模态数 n，vector_counts 给出每组实际混合向量数 m，要求
`1 <= m <= n`；单个 m 可以重复用于所有组。默认从 mode_offset 开始按顺序连续分组。
已有重要性平衡分组时显式传 `mode_groups=((...), (...))`，组内顺序保持，各组不重叠，
不能同时给非零 mode_offset。NV 只消费确定的分组，不把连续分组称为论文式重要性优化。
`mode_offset/mode_groups` 指输入轴的位置；输入声明 `mode_labels` 时，describe 中的
`mode_groups` 转为该实际父基底的科学标签，供 high partition 核对。重排或非零起点的已保存
父基底不能把位置号当成其原有标签；未声明标签的旧句柄沿用位置标签。

load 输入为 `(Nt,Ne,Lz,Ly,Lx,3)`，输出为 owned complex128
`(Nt,Nd,Lz,Ly,Lx,3)`，`Nd=sum(m)`。输出顺序由 description 中的
`vector_labels=(block,vector_in_block)` 指定。load 保存输入快照；后续 calc 不覆盖已返回数组。
每次 calc 生成一个 realization，每个时间槽独立混合并取得不同 pool UUID；两端复用已有数据时保留实际池身份。
默认完整时间；compact 输入保留原句柄 time_selection 和实际槽顺序，不重新选时间。
句柄的 `time_selection` 使用共享时间记录，`Nt` 必须等于输入首轴长度；不匹配时 load 拒绝输入，不能沿用上次计算。

## 归一化与消费

默认 `mixing="qr"`：m<n 使用复高斯 QR 的前 m 列，包括 m=1；m=n 直接复制父向量。
输出保留原始振幅，`E[D D†]=(m/n) V V†`。正交父基底的 QR 输出组内正交归一。
不额外单位化、不自动乘 sqrt(n/m)，不在向量生产时施加估计量权重。

`basis_orthogonal` 仅从输入 reader 的 `basis_orthogonal is True` 继承：load 开始将它重置为
false，读取失败后仍为 false，缺少 parent 声明也按未知处理。它表示向量之间正交，不等于
单位范数；例如 legacy 单向量组范数平方为 n，仍可与其它不重叠组保持正交。NV 不扫描数组
重算 Gram 矩阵，也不因 QR 分支自动猜测 parent 已正交。

只有复现旧单向量相位规则时设置 `mixing="legacy"`；其 m=1<n 使用 U(1) 相位，
协方差为 V V†，范数平方为 n。其余分支与默认一致。记录实际分支及归一化，
不能因为都叫 dilution 就混用权重。旧 dilution_pair_weights 入口描述直接抽取 Laplace 父模态；当前 QR 消费以实际混合列标签为单位。两者都区分相等和不同标签的包含系数，但不互换基底身份或调用旧 is_blending 入口。QR 的混合列包含系数、相等分区与整图余权统一见[EL 接线](../lqcdweave-elemental/references/dilution.md#同池混合列包含系数与范围)。

## 保存与读回

保存实际 D 数组，连同 cfg/格点、物理时间记录、父基底实际 data_id、有序父模式分组、
n/m、向量标签、算法/版本、分支和 raw 协方差因子。seed 记录在 provenance，
不替代实际 D 身份；每次 calc 为各实际时间槽生成新的、彼此不同的 time_pools UUID，保存本次 describe 到 provenance.generator；公共 DatasetWriter 发布后由实际 data_id 绑定 DilutionSource，未知来源仍保持未知。describe 的 complete 只表示本次计算完成。高级导入若实际跨时间共用同一混合列子集，须如实保留共享 UUID；消费按这个列选择池联合计权，不因物理时间不同而拆池。不能改写默认独立生成的 UUID 来虚构共享选择，或改写共享 UUID 来虚构独立性。详见[同池混合列说明](../lqcdweave-elemental/references/dilution.md#同池混合列包含系数与范围)。

使用已有 DatasetSpec、DatasetWriter/Reader 和 DatasetFieldHandle；不新建台账。
保存与读回约定见[数据保存说明](../../../docs/modules/storage/data_storage.md)，实际 D 来源的持久化核验见
[dilution reader 回归](../../../tests/adapters/test_dilution_readers.py)。
`explicit_basis` 是已有通用数组契约，并以 `representation="dilution"` 区分 D，
不是新的 D 专用存储 schema，也不声明已完成传播子六类存储。

发布 D 时，将 `describe()["basis_orthogonal"]` 写入该 Dataset 的
`conventions.basis_orthogonal`，这样 `DatasetFieldHandle` 可在读回时传递已证明的属性。
父基底若用 `normalization="orthonormal"` 声明，也会由 `DatasetFieldHandle` 识别。
未声明的裸 D/旧数据保持 unknown；shape、identity-record 类型或数组值都不能补出正交证明。

compact 时间用已有 time_selection 契约登记；describe 中的记录须转为其 `.document`
再用于 JSON，不能直接序列化 Python 记录。CuPy 输出保存到主机时显式 cp.asnumpy，
不在每个数值步骤往返设备。裸 NPY 只含数值和数组头，来源需另存，不能从 shape 恢复。

## 模块交接与当前范围

PR 接收实际 D，V/D 复用同一向量数值处理，并支持九类 V/P/D 直接产品、六类 pair 保存和九类读取视图；D 的端点与原始来源身份须由调用方绑定。具体用法见 [PR dilution 高级说明](../lqcdweave-perambulator/references/dilution.md)。NV 不求 Dirac 解、不做 Gamma5 变换。
EL 通过 vector_bases 显式计算九个 V/D/P 扇区，D handle 携带 DilutionSource；生产和 correlator 共用 elemental_weighted 管理局部混合列包含系数及整图相等分区余权。见[EL 接线与支持范围](../lqcdweave-elemental/references/dilution.md)。保持 dilution=None、is_blending=False，不通过旧父模态选模入口给 QR 数据计权；已有QR数据由实际混合列来源和当前 component_inclusion-v1 语义消费。
投影点称为 H，其补空间仍由完整 low+medium Laplace 集合定义，不由低秩 D D† 定义。固定父分组内改变实际 D 列数或选择子集不改变 H 的 Q；保留实际列标签及池身份供统计计权。PR 从实际 D handle 的 dilution_source 自动形成端点；完整生产、保存和整图消费见[V/D/H 三分区闭环](../../../docs/modules/perambulator/three_sector_production.md)，H 不由 NV 作为新的 QR 数据生产。

精确接口、异常、内存与测试见 [NV L2](../../../docs/modules/noisevector/noisevector.md)。
已验证 NumPy，小规模合成数据不证明 GPU/MPI、性能或真实生产规模。
多 cfg 的任务分配需要时按 [Dispatch skill](../lqcdweave-dispatch/SKILL.md)组织，
本 skill 不初始化设备、提交集群任务或恢复旧 highmode/full_noisev 分支。
