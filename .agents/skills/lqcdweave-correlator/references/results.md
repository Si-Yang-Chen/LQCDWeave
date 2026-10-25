# 读关联函数结果

各收缩编排数值入口统一返回 Calculation，`.run()` 返回新 Calculation，`.report` 始终为映射；
惰性 cfg、Dataset 记录及旧数组/统计字段兼容见[共同约定](../../../../docs/modules/correlator/calculation.md)。

result.value与输入correlator容器对应，保留dict、ndarray、list、tuple及嵌套；系数、多缩并项的求和已应用。用户不再手动回填、求和或reshape。

输出结构、term 索引与系数由结果层按已选 recipe 组装，用户不写回填或另做缩并；
内部绑定及有界装配细节见[结果 L2](../../../../docs/modules/results/results.md)。

新规划的 float32/complex64 任务在输出表达式内部自动保持 float64/complex128，乘积与求和
完成后才转回最终字段精度；无需新增选项。输入、准备和大矩阵乘法保持任务精度。
内部精度与临时内存额度进入完整档案，CPU/GPU 固定回放一致；旧档案继续执行原来保存的
逐节点精度，精度约定见[结果 L2](../../../../docs/modules/results/results.md)。

完整 cfg Dataset 的 `postprocess_backend="gpu"` 是显式高级选择；默认 `"cpu"` 不变。GPU 组装、dt/one-point 累加和打包沿用原字段精度、计数和首次值规则；`dt_packed` 最终才下载打包数据，`pairs_and_dt` 仍保存并下载每个时间对。memory 输出暂不支持 GPU 后处理，不静默回退。调用、预算和数值比较见[GPU 后处理](../../../../docs/modules/results/gpu_output_postprocessing.md)。
输出布局随坐标声明确定。普通源×汇扫描矩阵，即使两轴都是 Nt，也不等于源槽×循环差槽；selection 重建默认接受绝对汇槽；旧循环差槽须显式声明 time_layout="cyclic_difference"。涉及时间稀疏化时先核对[共同布局与当前接线限制](../../../../docs/modules/temporal_sampling/temporal_production.md)，符合布局后按同 cfg 已保存的记录调用[两点重建](../../../../docs/modules/results/two_point.md)。该入口不重建任意多点函数，也不会补齐未计算的逐源数据。

谱学矩阵保留所声明的单粒子、双粒子及交叉矩阵元，时间扫描后按声明追加源汇轴。
只有属于同一守恒量扇区的算符才能组成一个矩阵；非守恒的变换标签不自动用于拆分矩阵。

多流及弱流关联函数按声明分别保存流分量，保持每个动量、流种类及具名输出的身份。

扫描轴按声明顺序追加。只规划时value为None。报告说明通道、动量、时间序列及用户指定的平均/归一化；不能把逐源矩阵当已平均结果。裸强子矩阵元不等同物理衰变率。

## 数值保存和导入

`contraction.correlator.save_correlator(directory, result, configuration=cfg)` 默认保留值容器、算符轴、dtype 和声明的真实扫描坐标。
`load_correlator(directory, configuration=cfg)` 返回 CorrelatorResult，用 `.value` 取回相同数值结构，并保留坐标、来源和归约记录。
原始数值须显式提供共享尾轴 `axes/coordinates`，异构字段可按 tuple 路径给 `field_axes/field_coordinates`。
不把未知算符标签或科学身份从 shape 推导出来。

保存时可显式 `average_tsrc=True, source_axis=..., sink_axis=..., physical_lt=...,
boundary='periodic', coordinate_domain='physical'`。先按真实物理 dt 对齐再按实际贡献数平均；
只保存指定时间差时给 `dt_coordinates=...`（物理 dt，可重排），缺失检查只针对这些输出时间差，
归约记录仍保存完整 Lt 的贡献计数；
额外时间变量和未知域拒绝，缺失 dt 默认报错，显式 `missing='nan'` 才保存 NaN+零计数。
已相对布局改给 `relative_axis`，要求各源共用真实 dt 坐标。紧凑数据显式 `coordinate_domain="slot", selection=...`，默认 source_axis×sink_axis；旧循环差槽用 relative_axis 与 `time_layout="cyclic_difference"`。不能因两轴大小都是 Nt 而当作物理 source/dt。导入已平均数据不恢复 tsrc，也不能再平均一次。
详见[数值结果存储](../../../../docs/modules/storage/correlator_results.md)与[计划及结果保存/读回的 CPU 回归](../../../../tests/test_correlator_frontend_roundtrip.py)。

关联函数结果按 cfg 保存为 HDF5 具名字段，字段保持真实 shape/dtype；共享根 `dataset.json` 登记公共 metadata 与 cfg 身份。GPU 后处理可在设备传输中临时打包，不改变已保存字段。读取时按 `configuration=cfg` 指定 cfg；数据位置与未知科学来源不会从目录名推断。完整规则见[默认数据格式](../../../../docs/modules/storage/data_storage.md)。

完整cfg写盘由storage按字段轴分块、尾随扫描轴逐点写入；不能把基础数据前导时间轴的分块假设用于关联矩阵，造成逐元素HDF5小块。脚本不接管分块或改写字段布局。

维护已有 Dataset 后处理时，同一 cfg 的多个具名字段共用一次打开的公共 DatasetReader，
复用已验证的登记身份，各字段仍按登记 shape/dtype 读取；不要逐字段反复打开并解析整套元数据。
此规则不改变常规计算入口及保存策略，也不要求用户增加读回步骤。

多任务分别保存不同 cfg 时，storage 增量更新共享登记，保留旧 cfg 的压缩差异；不用在调用脚本中重建整个 `dataset.json`。公共信息缩小时只调整受影响部分，读取指定 cfg 只恢复该记录；已有 v1 登记无需迁移。

## 分块矩阵的有界装配

使用保存的未计权逐腿贡献重新装配，不从关联函数总和逆推出缺失的子集贡献。
改变 D 父区间时不能直接重分组旧 QR 逐腿标量；须以实际父基底和完整输入重新构造两条腿及交叉项。

同形状的 scatter 求和由结果层按原有次序累加紧凑块，共用一个完整结果缓冲；
允许块间重叠，保留 dtype 提升、尾随时间轴和独立输出所有权。不为每个径向块
创建完整零矩阵。规划使用同一装配策略的紧凑块、索引和暂存上界；不能仅调低
预留值而不修改实际装配，也不能通过逐通道前端循环绕开联合计划。
