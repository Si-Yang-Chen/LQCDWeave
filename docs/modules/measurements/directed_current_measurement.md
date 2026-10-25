# 已登记电流数据上的双电流计算

`contraction.directed_current_measurement` 提供只读计划和共享行写出函数；不提供固定的持久化双电流执行入口。调用者根据所选算符、时间窗口和输出契约，组合已登记的 current/current/VSV readers、现有数值核和通用存储服务。

## 只读规划与行写出

`plan_indexed_directed_current_pair` 校验实际输入 ID、完整物理时间、实际有序基底、动量视图、两套 current 配方和端点，不执行数值核或创建输出。`plan_indexed_directed_current_chain` 对单链输入做对应的元数据规划。规划结果仅说明现有数据是否满足请求。

通用写出函数分别为 `write_directed_current_pair_rows(first, second, vsv, writer)` 与 `write_directed_current_chain_rows(current, incoming_vsv, outgoing_vsv, writer)`。它们消费调用者打开的 readers/views，同步写 `writer.missing`，不关闭输入，也不单独登记完整 cfg。调用者持有读写生命周期、定义输出轴和槽坐标，并按当前支持的接口发布数据。

## 科学定义与数据身份

terms 是 `CurrentTerm` 或对应的有序 mapping。每套配方绑定一个实际已存动量索引，按坐标保存实际 XYZ 值。保留复系数、归一化、重复项和共同端点偏移；单链还需确保方向、位移、bar/field 偏移、link origin 与 dagger 一致。无链局域项、两跳项和不兼容端点必须拒绝。

current 输入是含原始方向路径的 `elemental_vv`，采用 `midpoint` 相位和未缩放归一化；VSV 具有绝对 source/sink time、自旋和完整时间契约。相关输入绑定同一实际有序基底，但规范场身份分别记录，不代表自动等价或严格守恒。`usedNe` 是三者一致的明确前缀视图，不推断共同最小尺寸。

periodic 逐项对端点取模。open 边界下若请求锚点的任一项越界，应在输出创建前拒绝；调用者可以显式提供合法时间坐标，不能把未定义值补零或借此启用稀疏时间估计量。时间顺序、边界、配方和实际 input IDs 由调用者的输出契约完整记录。

## 数值与存储

pair 数值核为 `contraction.directed_current.contract_directed_current_pair_v2v`，先计算各配对迹，再乘各自复系数和 normalization。结果不隐式添加 Wick 负号、共轭、实部投影、源平均或 Ward identity 声明。generic `DatasetSpec`/`DatasetWriter` 可以承载所选持久结果；模块不另建 receipt、resume 或自动来源流程。

测试覆盖 pair/chain 核的独立矩阵 oracle、复系数及实际端点；存储发布和恢复语义由通用 storage/runtime 测试负责。详见[模块契约](directed_current_measurement-api.md)。
