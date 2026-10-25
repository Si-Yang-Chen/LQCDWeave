# 可复用本地验收

`tools/acceptance.py` 管理同一个稳定快照的冻结、wheel/安装目录核对、测试结果和精确本地提交。它不推送、不安装依赖、不提交集群作业，也不修改工作约定。运行前按本次改动确定专项、数值参考、文档检查及目标平台要求；脚本不替代这些判断。

冻结覆盖 src、tests、scripts、benchmarks、tools、CI 输入，以及 pyproject、README、LICENSE 构建输入；包括数据资源，排除 Git、Python 缓存和生成的 egg-info。运行环境记录解释器、标准库/扩展库、显式依赖目录及相关环境变量的实际字节。其余纯文档需另行检查内容、链接和受影响的文档门禁；CPU 证据不替代 GPU/MPI 或性能验收。

所有快照、构建副本、wheel、隔离安装目录和日志默认写入仓库外的 `../acceptance-evidence/<phase>/`。每个新稳定快照使用一个独立 phase；已冻结快照用 `check` 核对，不覆盖旧记录。以下命令在仓库根目录执行，依赖目录须替换为实际验证环境；解释器也应使用本轮已确定的版本。

```powershell
python tools/acceptance.py freeze source-to-receipt --dependency-root ../parent-review-deps --retired-report ../parent-review-wheel-persistent-diagram-sink-batch.json
python tools/acceptance.py build source-to-receipt
python tools/acceptance.py check source-to-receipt
```

`build` 保留独立源码副本，使用现有构建依赖构建 wheel，不安装运行依赖。它核对所有 package 源码和资源、退役路径、隔离安装字节及 Python 3.10 语法；语法检查不是该版本的数值运行证明。

完成专项和相关文档检查后，在两个独立终端/工具会话并行执行同一冻结快照的完整入口：

```powershell
python tools/acceptance.py source source-to-receipt
python tools/acceptance.py installed source-to-receipt
```

源码入口是完整 unittest discovery；安装入口是完整 pytest，禁止源码回退，最终检查所有已加载包模块的来源。安装完整入口要求 `PYTEST_ADDOPTS` 和 `PYTEST_PLUGINS` 为空，避免环境中的参数或外部插件筛掉测试；插件及相关测试环境变量进入冻结记录。退出码、日志、运行器和快照摘要一起保存。成功结果的代码、测试输入、环境、日志及安装包仍一致时再次调用直接复用证据；wheel 复用核对整个压缩包摘要及全部 package 文件，不重复长测试。

运行中断会保留 unfinished 标记，防止重复启动和覆盖日志。标记本身不证明进程仍运行；先检查实际进程/工具句柄，继续等待活跃任务。失败或不兼容结果保留，修复后建立新快照。不要因为一次观察超时而重新启动测试。

```powershell
python tools/acceptance.py audit source-to-receipt
python tools/acceptance.py commit source-to-receipt --file src/lqcdweave/contraction/meson_measurement.py --file docs/modules/measurements/meson_measurement.md --message "Connect indexed Meson requests to shared dataset lifecycle"
```

提交必须列出本次全部相关文件（含删除路径），且现有暂存区为空；目录范围在暂存前拒绝，路径按字面值处理；示例清单不能直接当作实际改动范围。脚本确认源码与安装测试证据有效，按明确路径暂存、检查范围和差异，再核对提交字节及删除结果，仅允许 Git 的 CRLF→LF 正规化。已有无关改动保留；AGENTS.md、SKILL.md 不由这个验收入口提交。文档与未运行平台的限制仍须如实交付。
