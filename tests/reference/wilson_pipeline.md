# Wilson、Laplace 与关联函数的显式组合

本仓库保留 Wilson/Laplace 的数值后端、只读生产规划、低层 kernel 与 generic storage API；不提供覆盖规范场、基底、本征值、传播子、elemental 和测量的固定生产脚本。调用者根据实际任务选择输入、产品、边界与输出，再组合相应计算接口。小格点 CPU 组合参考见[Wilson 回归](../test_wilson_pipeline.py)及测试拥有的[Laplace](../fixtures/laplace_script.py)、[传播子](../fixtures/perambulator_script.py)、[elemental](../fixtures/elemental_script.py)接线，不定义项目级 runner 或默认研究方案。

## 可用规划

- `plan_wilson_reference(gauge, output, configuration=cfg)` 从共享登记选择指定 cfg 的规范场，预览 Wilson 参考输入与阶段需求；登记中只有一个 cfg 时可省略 `configuration`。
- `plan_indexed_wilson_reference` 只读规划已登记规范场对应的基底、本征值及参考数据需求。
- `plan_indexed_wilson_propagators` 只读规划已登记点集/基底的所选 VSV、PSV、VSP、PSP 产品。
- `plan_indexed_perambulators`、`plan_indexed_point_elementals` 和 generic `DatasetWriter` 可组合生产单独选定的数据产品。

规划不运行数值 kernel、不写输出或作项目 receipt，也不承诺后续任务级恢复。缺失父数据时，调用方自行决定下一步的生产策略；相同 shape 或模态数量不等于实际基底相同。

## 小格点组合的基本阶段

一条由调用者组织的参考计算可以按所需内容依次执行：

1. 读取实际 gauge dataset，按用户指定参数计算并发布 Laplace eigenvectors/eigenvalues。
2. 使用实际 gauge、basis、Dirac 参数和所需 endpoint inputs 计算选定 Wilson sectors。
3. 按指定 paths/momenta 计算 elemental products。
4. 将所选 readers 绑定到 Meson/Diagram，调用 `calculate` 或适用的数值核。
5. 若需要持久结果，调用者按输出轴和 recipe 建立 `DatasetSpec`，使用通用 `DatasetWriter` 写入并发布。

每阶段的路径、坐标、边界、实际父 ID、精度、来源和完成覆盖应由任务脚本明示。shared row writers 只同步写调用方提供 writer 的缺失首轴块，不拥有输入 reader 或发布行为。需要其它阶段顺序、独立基底、不同 point pools 或不同存储布局时，按实际选择调整组合。

## 作用量与约定

CPU `DenseWilsonBackend` 实现各向同性、无 clover 的 Wilson 算符，格距和 Wilson 参数取 `a=r=1`：

\[
D_{xy}=(m+4)\delta_{xy}-\frac12\sum_{\mu=1}^{4}
[(1-\gamma_\mu)U_\mu(x)\delta_{x+\hat\mu,y}
+(1+\gamma_\mu)U_\mu^\dagger(x-\hat\mu)\delta_{x-\hat\mu,y}].
\]

采用项目的 Euclidean DeGrand–Rossi 矩阵，质量为格点单位裸质量。该质量归一化 D 与 hopping 形式 `M=I-κH` 的关系为 `D=M/(2κ)`、`κ=1/[2(m+4)]`；求逆结果不额外乘 κ。空间为周期边界，`t_boundary=±1` 仅作用在跨时间边界的 hopping 上。后端拒绝各向异性、非零 clover 和 multigrid。

Laplace 使用空间周期正 Laplacian、无 smearing；向量逐时间求解并排序，实际 basis ID 绑定其有序 mode labels。VSV 使用绝对时间、DeGrand–Rossi 自旋和未缩放约定。正向与反向夸克线都需由明确传播子输入提供；关联函数保留绝对时间对，不作隐式源平均或 gamma5 推断。

## 验证边界

测试对 Wilson matrix、自由场动量公式、gamma5 关系、规范协变、Laplace 低模/残差、向量投影和独立全空间迹做 CPU 小格点核对。它们不等于物理系综统计、GPU/MPI 支持、生产规模资源验收或 QUDA 归一化交叉验证。通用 storage/runtime 的恢复保证按对应 dataset 生命周期测试核实；示例中的单脚本组合不构成新的执行框架。

完整随附 `4³×8 weak_field` 的 EV/NV/分段空间点→九类 PR/EL→介子、VVV 重子和跨时间流闭环，另列为[显式慢测审计](weak_field_audit.md)。它重新指定无 clover 的 Wilson 问题并使用独立算子组装和物理空间参考，普通开发回归不运行；历史归档传播子的未记录生产参数仍不能据此恢复。
