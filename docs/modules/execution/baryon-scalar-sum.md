# Whole 重子单合并腿标量归约

`baryon_scalar_sum.prepare_scalar_sum(action, arrays, backend, dtype)` 是执行层的内部实现。
它消费已有 `ordered-join-v1` Action，支持一个已合并 rank6 腿到标量，
以及带共同 leading coordinate lane 的对应向量输出，不改变 P、C、路径或批宽。
算符、DR→DP、源端共轭与分量系数仍来自原绑定；内核不增加共轭或归一化。

## 接线与生命周期

调用者只在正式 `no_search=False`、非 reused 的 whole baryon 分支调用此准备函数。
返回 `None` 时继续原有执行路径；返回对象则在 `try/finally` 内调用 `compute()`、`close()`。
`compute()` 返回与任何输入都不共享数值 owner 的数组，精度已是传入的 dtype，
不需要额外 `copy_to_backend` 或再次施加分量系数。关闭准备对象不改变已返回结果。
关闭可重复执行，关闭后再次 compute 明确报错。

准备对象只保留本次操作的 spin、Phi、腿与当前 GPU descriptor，close 后释放这些引用。
结构缓存只保存轴关系、shape、support 索引；编译缓存只保存 dtype、分量数、设备对应的代码。
不缓存历史输入值或地址。GPU descriptor 每次准备都从当前数组重建，
内核读取当前 spin 的实际数值，不把 support 元数据中的历史系数当作当前 spin。
同一已准备对象再次 compute 也读取当时的输入值；不同当前输入需要重新 prepare。

## 数学与支持边界

精确 sparse metadata 声明支持索引的并集，内核计算
`sum_{i,j,k,a,b,c} leg[i,a,j,b,k,c] * sum_r coefficient[r]*spin[r][i,j,k]*Phi[r][a,b,c]`。
不同分量的 Phi 可以不同，复系数只施加一次，support 内的 spin 值可以改变或成为精确零。
若稀疏结构改变，声明也须同步改变；执行不会扫描大输入来重新发现稀疏结构。

目前支持一至三个分量、complex64/complex128 同精度的本地 NumPy 或 CuPy 数组，
三个 spin 轴完整大小为4，模式轴可为矩形，正/负 stride 与腿的轴置换均保留。
lane 可来自合并腿、动态 Phi 或两者，静态 Phi/静态腿按 lane 广播；动态宽度必须一致。
未知实现、legacy 描述、缺少 exact support、混合精度、GPU 外来主机数组及其它端口结构返回 None。
这些情况不触发新的搜索，也不静默改精度或复制输入来扩大适用范围。

CPU 只在三个实际模式轴均至少为8时接管，尺寸来自合并腿的实际轴对应关系，
不把 coordinate lane 当作模式轴。较小尺寸在建立 support、数值描述符和准备对象前返回 None，
继续原收缩路径。GPU 保留已测 Ne1/2 的支持。这是执行层的保守选择，不新增用户选项。
正式 join 在调用 prepare 之前，先用已声明模式 shape 作廉价排除，避免小任务仍支付准备探测成本；
prepare 仍独立核验实际腿尺寸。两处共用内部 `CPU_MIN_MODES=8`，不维护两份阈值。

CPU 按 lane 与联合 support 求和，只产生当前模式尺寸的暂存量。
GPU 按腿的实际模式 stride 选择遍历顺序，用两级确定性 reduction，使用两个 kernel、无 atomics。
complex64 累加保持 float/complex64，complex128 保持 double/complex128。
不同分量的求和重排会改变浮点舍入；这里没有定义完整真实 LL3 的新 c64 验收阈值。

## 验证与计时口径

专门测试对独立 complex128 稠密求和覆盖 G1g/u 的 row0/1、源/汇、不同 Phi、复系数、
矩形模式、非连续和负 stride、轴置换、静态/动态/混合 lane、两次当前输入、
准备后 spin 值变化、零腿、独立结果 owner、关闭释放与不搜索。
小 c128 沿用 `rtol=2e-11, atol=2e-12`；小 c64 沿用既有隔离 baryon 单元范围，
不将它当成完整 Ne100 LL3 的验收标准。

自然单操作计时须包含每次 join 分派、prepare、当前 descriptor 上传、输出分配、
compute、close 和设备同步；输入创建/上传在两边共同边界之外，首次编译或模块加载另记。
排除 lowering 的隔离原型数据不能直接代表该自然计时，更不能代表完整 P 的加速。
完整任务的科学范围与验收边界见[集成验收](../correlator/correlator-integration-acceptance.md#完整计算范围)。
