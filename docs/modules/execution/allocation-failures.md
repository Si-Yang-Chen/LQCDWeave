# 设备分配计费与失败证据

CuPy默认池按单个allocation owner向上取整到512字节。X的设备预读声明使用
C的`device_allocation_bytes`，与P的纯元数据计费一致；读队列不能按较小逻辑payload
申报多个实际分配。已释放但计算仍在使用的上传值进入退休队列时，优先按
`value.data.mem.size`的真实owner范围计费，避免view的`nbytes`缩小了其仍占用的内存。
没有真实owner描述的后端沿相同512字节规则给出保守范围。

`allocation_failure.py`只在设备`MemoryError`时生成证据：失败操作、池的
used/total/free/free-block计数、逻辑值和raw值的payload及可见owner、上传与退休ticket、
预读预算/占用、已有设备统计和该操作的计划memory points。绑定执行表在调用失败后、
存储和读队列清理前附加这些记录；一般存储回放也保留异常取证兜底。数值操作内的临时
Python frame可能已经展开退出，因此记录不是精确的失败分配瞬间快照。

记录附于原异常的`lqcdweave_allocation_failure`属性，并在支持`add_note`时
写入traceback。它不捕获OOM后缩批、不改精度、不切换设备、不重试、不清池，也不为取证
同步设备。正常执行不逐操作查询allocator或生成这些统计；原异常和清理协议保持。

池的`total_bytes`包含缓存，不能解释成仍活跃的数值payload。CuPy13.2在达到池上限时
已经尝试回收空闲块和GC；被分拆且尚未完全合并的块可能仍不可归还，stream arena也影响
复用。因此512字节计费修复不等于证明消除了所有碎片风险；应结合失败记录中的used、
total、free和实际owner检查差额，不凭一次总分配数字推断内存泄漏。

CuPy实现依据：[MemoryPool文档](https://docs.cupy.dev/en/v13.2.0/reference/generated/cupy.cuda.MemoryPool.html)、
[13.2.0 memory.pyx](https://github.com/cupy/cupy/blob/v13.2.0/cupy/cuda/memory.pyx)。
CPU协议验证见`tests/correlator/execution/test_allocation_failure.py`，不代表真实GPU容量验收。

## 固定坐标回放的池余量

C的`device_allocator_headroom`声明经验余量`min(1 GiB, cap - floor(cap * 0.8))`。
1 GiB是库的经验默认额度；具体测量及校准证据保存在调用项目中，不写入通用执行契约。
小预算的上限直接复用硬件预算已有的20%默认
留白，128 MiB任务因此仍有数值空间。常量`DEFAULT_PLANNING_FRACTION`统一定义在C，
硬件adapter继续使用并重导出。统计明确表示此默认没有对当前设备运行时校准；此额度没有
数学上覆盖任意碎片、任意设备和任意计算图的保证。

普通单坐标选择的`.run(time_axes=..., batch="auto")`自动把此额度传入P固定坐标排期。
它与调用者`reserved_device_bytes`相加，进入每个memory point、批宽选择和统计；pool
硬上限保持不变。例如28 GiB pool为数值模型留下27 GiB，余下1 GiB不能用于预读、
常量缓存或持久工作区。单点结果再次绑定新坐标会替换旧自动额度，不重复累加。
CPU自动额度为零。这里不改变已有Dataset窗口、归档或其它规划入口的资源政策。

既有硬件预算的80%默认在未指定硬额度时按设备可用量留白；显式pool cap会覆盖该默认，
而且pool内部缓存仍受cap限制，因此它不能代替上述pool内部额度。统计保留政策、来源、
硬cap、数值cap及实际每批memory point；不在OOM后重试或缩批。
