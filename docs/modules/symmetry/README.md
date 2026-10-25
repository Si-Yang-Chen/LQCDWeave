# 格点对称性

`lqcdweave.symmetry` 负责格点群、不可约表示、旋转、限制表示及子空间投影。它是本软件的基础模块，供算符构造和耦合使用，不负责物理测量声明、数据读取、计算规划或执行调度，也不反向依赖这些流程。

## 接口与约定

公开入口为 `lqcdweave.symmetry`，与 `lqcdweave.symmetry.api` 导出的对象相同。

- `Group`、`Irrep`、`oh_group`、`little_group`、`double_cover`：群与表示。
- `irrep_matrices`、`canonical_label`、`reference_frame`、`compose`、`inverse`：规范表示矩阵、标签、参考动量与群操作。
- `wigner_rotation`、`stabilizer`、`row_transfer`、`project_row`、`restrict_irrep`：旋转、稳定子群、表示行转移与限制。
- `subduce`、`isotypic_subspace`、`row_subduction`：表示分解及子空间投影；`genIrrepOhD`、`genLittleGroupIrrep` 提供精确表示构造。

具名立方群采用 `CONVENTION` 标识的规范基底，表示行、拷贝顺序和相位属于计算约定，不能随意用等价基底替换。`row_subduction` 的输入表示必须使用相同群元素顺序；结果包含投影矩阵、正交基底、表示行和重数。数值与精确符号模式的限制以函数参数及实现为准，不支持的动量方向或表示不能外推为任意群的完整支持。

符号功能依赖 NumPy、SymPy 和 lazy-object-proxy，随 `.[correlator]` 或 `.[cpu]` 安装。模块不在包顶层自动加载，基础数据及结果读取不要求导入符号功能。

## 验证

群乘法、表示幺正性与投影正交性回归见[对称性测试](../../../tests/symmetry/test_symmetry.py)，算符消费行为见[算符测试](../../../tests/correlator/operators/)。
