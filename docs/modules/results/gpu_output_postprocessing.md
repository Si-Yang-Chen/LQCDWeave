# 可选 GPU 输出后处理

只有完整 cfg 的输出组装、dt 累加或打包成为瓶颈时，再显式选择
`postprocess_backend="gpu"`。`complete_calculation` 与
`run_cfg_window_calculation` 的默认值仍为 `"cpu"`；普通 `calculate` 调用、
输出布局、精度和科学定义不因这项优化改变。缩并的 `backend` 与输出后处理是两项独立选择：
同一个 CuPy 缩并任务也可以继续在 CPU 上后处理。

此入口要求已针对 GPU 选择的计划与兼容 CuPy 运行时；它不会把 CPU 计划变成 GPU 计划，
目前只支持 Dataset 输出，memory 输出明确拒绝；不支持的配方或预算也不静默改为 CPU。先用目标名称或
`ExecutionTarget` 声明实际硬件与额度，再完成输出计划。只需要普通计算时不增加此开关。

## 完整计划与显式窗口

`selected` 是已用 `calculate(..., execute=False, target_hardware=target, ...)`
选择的单坐标 GPU 路径；`coordinates` 是本次完整有序坐标请求。`binding` 是已有
`CfgSources`，含兼容的 reader 以及明确的有序 `physical_times/physical_lt`。
这里的 `time_roles` 按汇、源顺序声明；角色名没有内置的物理含义。

```python
from lqcdweave.contraction.correlator import (
    complete_calculation, save_calculation_plan, load_calculation_plan,
)

complete = complete_calculation(
    selected, coordinates=coordinates, readers=binding,
    configuration=cfg, output_mode="dataset", output=output_root,
    physical_lt=Lt, time_roles=("sink", "source"),
    output_layout="dt_packed", postprocess_backend="gpu",
)
save_calculation_plan(plan_file, complete)
restored = load_calculation_plan(
    plan_file, configuration=cfg, readers=binding, output=output_root,
)
result = restored.run(backend=cupy)
```

完整档案在 `plan.statistics` 保存 `postprocess_backend`、`postprocess_device_bytes` 和
`postprocess_host_bytes`；不把这些执行选择放入科学 `output_definition` 或改变 Dataset schema。
加载重放沿用保存的选择，并在读取数值前检查当前实现的 dtype 和必要保留量仍适合该档案；
不足则要求从原 selected 模板重新完成输出计划，不静默扩大额度。执行不重新选路、
改布局或推断另一套预算；要比较 CPU/GPU 后处理，应从同一已选缩并模板分别完成计划。
没有该新字段的兼容旧计划保持 CPU 语义。输出目录仍由加载调用提供，reader 与设备数组
不进入档案。

不需要保存完整坐标排程、而需要按窗口增量规划一个 cfg 时：

```python
from lqcdweave.contraction.correlator_window_run import run_cfg_window_calculation

result = run_cfg_window_calculation(
    selected, coordinates=coordinates, readers=binding,
    configuration=cfg, output=output_root, backend=cupy,
    physical_lt=Lt, time_roles=("sink", "source"),
    output_layout="dt_packed", postprocess_backend="gpu", window_size=window_size,
)
```

窗口入口在启动规划生产者前预留整个 cfg 输出的主机和设备额度，不能把跨窗口累加器只算作
一个窗口的临时量。整个 cfg 成功后才更新共享根登记；中途失败仍由既有 writer/执行作用域清理。
此入口当前一次运行一个 cfg，其返回值不是可保存的 `CompleteCalculation` 档案。

## 数值、布局与数据移动边界

编译兼容性由内部执行实现处理，调用方无需设置语言标准或编译器选项。
首次使用某种内核和数组布局时可能支付 JIT 编译开销，后续相容窗口和 cfg 复用缓存。
离线规划不访问设备或编译；绑定生成程序与首次完成设备编译是两个阶段。
若编译失败，错误信息给出独立诊断日志的位置，正常执行不会额外进行合成自测。
CUDA/HIP 的实现要求与验收边界见 [X 的内部编译策略](../execution/execution.md#显式-gpu-输出后处理)。

- 固定 `OutputRecipe` 在执行开始时编译为逐字段融合程序；大字段拆为有界有序段，临时缓冲按最后使用复用并预留额度；逐点数值组装、复系数和表达式运算
  在设备上完成。保留配方规定的项顺序、dtype 和必要的中间转换，不修改 Wick、dagger、
  模态分区、归一化或估计量
- 每个实际物理 dt 保持原 dtype 求和缓冲，依请求次序累加；one-point 字段只保存各时间槽
  首次出现的值，并要求声明的 one-point 槽全部覆盖；缺失槽不能套用 dt 的 NaN 策略。
  CPU 只维护小型坐标映射、配对计数和覆盖标记，不收集完整源×汇数值张量
- GPU 完成按实际计数除法、缺失 NaN 与尾轴排列；如传输需要，可以临时打包。
  HDF5 持久层保存字段真实 shape/dtype。
- `dt_packed` 只在完成时下载所选 dt 和 one-point 的最终打包数据；不会逐对下载 term 或
  完整字段。默认仍只保存显式布局指定的非零 dt，缺失 dt 保留 NaN 与零计数
- `pairs_and_dt` 仍保存每个请求的原始时间对；因此仍需逐对下载其打包记录，再写出 dt 结果。
  不能为减少下载而悄悄丢掉逐对输出或把它替换成 `dt_packed`
- storage 将各字段的真实 shape/dtype 写入每 cfg 的 HDF5，并以共享 `dataset.json` 登记字段定义、
  科学身份和 measured counts。读回继续使用
  `contraction.correlator_output.load_correlator_output`

这不构造常驻的 40×40 时间对输出张量，也不意味着输出零成本。dt 缓冲、one-point 缓冲、
设备打包缓冲、最终主机下载缓冲及在途输出必须在分配前计入对应预算；输入、缩并工作区、
CUDA 上下文和内存池仍占空间。`target_hardware` 的声明额度是上限权威，运行时观测只能
收紧，不能因卡的市场型号扩大额度。不能假设 V100 都可用 32 GiB，也不能保证剩余显存；
例如已有任务声明的 20,615,843,020 bytes cap 与约 18.47 GiB 观测值是不同证据，均不能
外推为其它任务的可用空间。有关资源模型边界见 [P](../planning/planning.md) 与
[执行](../execution/execution.md)。

GPU 固定字段程序沿用 R 定义的配方语义，由 X 编译和执行；TS 提供设备时间归约，
收缩编排负责组合，storage 只消费准备好的主机记录。支持的数值节点包括 term/diagram_sum、
常量、add/multiply、整数 power、规则数值 array、reshape/transpose/select/scatter 和 zeros；
dict/list/tuple 保留字段结构。object/ragged 数组、非原生字节序及未支持 dtype 会拒绝。
超过直接核参数额度的字段使用每次调用共享的小型 pointer/stride 表上传，仍只传设备数据地址
与布局，不回传数值；该元数据传输也应计入性能报告。为约束生成源/JIT 规模，每个生成段最多 8,192 个展开表达式节点；更大字段按原运算顺序
拆段，保留各步 dtype 与舍入，并在分配前预算可复用 scratch。该段大小不是实测 CUDA 极限。
`describe_device_output_program` 可离线报告每字段节点、段数、生成源大小、scratch 与元数据字节；
总任务 term 数不等于单字段节点数，也不能单凭成功预检宣称实际 CUDA 编译/执行已验收。输出字段支持原生 bool/整数、float32/64、complex64/128，
dt 均值字段必须是支持的浮点/复数。0-D 整数 term 的 Python 任意精度算术与
NumPy 1.x 中值相关、无法静态确定的窄类型提升会明确拒绝；整数字面字段仍保留原字节。若设备推导的 shape/dtype 与已保存 CPU 布局不同，
预留阶段拒绝，不偷偷改精度。支持与数值验证仍以本次测试和真实设备结果为准。

## 合成数据与回归入口

[后处理测试夹具](../../../tests/fixtures/output_postprocessing.py)提供固定 seed 的合成 Ne=2、Lt=4 输入与五个时间对。
[后处理回归](../../../tests/test_gpu_output_postprocessing.py)核对 dt 计数、缺失值、one-point 首次值、dtype 和持久字段，
并区分 NumPy 协议替身与真实 CuPy 用例；真实设备用例的运行条件见[测试说明](../../../tests/README.md)。

这些是接口与数值参考，不是物理系综或性能基准。输入 complex64/128 的测试按完整任务最低精度规则
核对最终输出；新窄任务的输出表达式按已存 `intermediate_dtype` 保持宽中间算术，CPU/GPU
均在字段末尾转换。输入、自旋准备及大矩阵乘法保持任务精度，dt 累加沿最终字段规则。
独立坐标循环核验 dt 求和和计数，
不构成独立 Wilson 反演或一般物理作用量证明。生产调用仍使用上文的 `calculate`、固定计划及 Dataset 入口，
不依赖本机项目脚本的命令行选项。

## 同设备验收与性能比较

1. 在同一授权分配中，固定代码版本、输入文件/hash、cfg、Ne/分区、dtype、实际时间选择、
   有序时间对、dt 覆盖、输出字段与布局、目标硬件、资源 cap 和线程设置。两边都使用相同
   CuPy 缩并 backend，只改变 `postprocess_backend`；CPU-only 与 GPU 整体比较不能隔离后处理收益
2. 先跑相关 CPU 数值/协议测试与 [GPU tier](../../../tests/README.md#测试层级与结果分类) 的真实 CuPy
   测试。CPU 模拟协议和生成核体对照不能证明 CUDA/HIP 编译、设备生命周期或速度；
   没有设备/依赖的 skip 必须报告未运行。不要在登录节点运行 GPU 基准
3. 保存并重载两个 complete 计划；另核对显式窗口与完整计划结果一致。比较每个字段的 shape、
   dtype、物理坐标、counts、NaN 掩码、first-seen one-point 和科学定义；覆盖 complex64/128、
   复系数、混合字段、缺失 dt、零输出、两个布局及异常未发布。所支持范围由实际测试决定
4. 同时与保留的 CPU 数值路径及任务已有独立科学参考比较，报告最大绝对误差、最大相对误差
   及接近零值的处理方式；容差按 dtype/问题规模预先固定。位模式相同与容差内相同分开报告，
   不因融合或不同设备浮点实现自动要求跨后端逐位一致，也不放宽既有科学验收
5. 热身另记并同步设备计时边界，重复同样的执行与写出，报告端到端 wall time、设备执行、
   后处理/打包、D2H 次数与字节、落盘时间及窗口/队列等待的实际可测分项。重叠阶段不能简单
   相加；JIT 编译、规划、保存/加载和 CPU 参考是否包含在计时内须写明
6. 对照最终 dt/one-point payload 字节核验 D2H 计数：`dt_packed` 应只有最终记录下载，
   `pairs_and_dt` 应保留逐对记录下载。输入 H2D、其它运行时拷贝与输出 D2H 分开统计；
   不用“少了若干 Python 调用”代替传输证据。当前 X 的 `report["device"]` 中
   `downloaded_bytes` 与 `output_download_calls` 记录输出下载；需要间接核参数表时，
   `output_metadata_upload_bytes/output_metadata_upload_calls` 单独记录该小型布局表的上传；顶层的
   `postprocess_backend/postprocess_device_bytes/postprocess_host_bytes` 记录 GPU 选择与规划保留量，
   不作为实测内存峰值。`output_postprocess_calls`、`output_postprocess_stage_launches`、`output_postprocess_bindings`
   记录实际装配调用、成功核提交及绑定次数；`output_postprocess_queue_modes` 记录重叠下载、
   预算/诊断/传输协议回退及无需逐对下载的实际选择。缺少某项报告时明确写未测量，不能自动记为零
7. 同报 P 的两域预测峰值、观测的进程 RSS、GPU pool used/reserved、整设备/non-pool 观测与
   采样方法。预算预测不是实测高水位，离散采样也不保证捕获所有瞬态峰值。比较更少的 D2H
   是否值得额外设备驻留；结果可能变快、变慢或因预算不足拒绝，不能预先承诺加速

本页不提供真实 GPU 性能数据。旧 V100 验收只证明当时的代码与样本，不能作为
本次后处理实现已通过或更快的证据。测试职责见 [R](results.md)、
[storage](../storage/correlator_results.md) 及 [测试说明](../../../tests/README.md)。
