# 公开接口与文档检查

普通用户从对应任务的[使用入口](../docs/README.md)开始；本页说明公开边界和维护检查，不要求生产脚本经过额外包装层。

## 哪些名称是公开接口

- 各模块的 `api.py` 和其 `__all__` 定义跨模块公开边界。完整名称列在[机器可读清单](public_api.json)，包含生产、取样、算符、Correlator contracts、绑定、展开、规划、执行、结果、性能和计划档案
- 已有包级短入口列为 `facade`，如 `lqcdweave.production.PerambulatorGenerator`。同模块短入口必须保持与 `api.py` 对象相同；短入口可以只公开其中一部分
- 常用叶模块 `lqcdweave.preset` 与 `lqcdweave.contraction.correlator` 的显式 `__all__` 单独列为 `frontend`；包括 `load` / `save`、`calculate`、结果保存和计划读回。`calculate` 必须保持与 `calc_diagram_eval` 的对象身份相同。这个名单经过明确选择，不把其它实现模块的内部 `__all__` 自动升级为用户接口
- 文档明确给出的其它叶模块入口仍按该页契约使用，例如 `lqcdweave.storage.time_selection.load_time_selection`。文档中的真实导入和可静态确定的调用参数另有检查；清单没有把这些模块的所有属性升级为公开接口
- 以下内容不因为可以导入就成为用户 API：下划线名称、模块偶然导入的依赖、具体实现文件，以及 `contraction.execution.operations` / `contraction.planning.optimizers` 的内部辅助导出表。Elemental 的 `ElementalKernel` / `BaryonElementalKernel` 是既有基础设施之间的显式数值交接，普通用户使用 `ElementalGenerator` / `BaryonElementalGenerator`

清单是代码审查门禁，不是所有版本永远保持签名不变的承诺。接口语义、必填参数、默认值、布局、所有权和适用范围仍由[模块 L2](../docs/modules/README.md)及对应使用页定义；科学身份不能仅通过导出列表检验。

## 用法与 CPU 回归入口

安装[CPU 开发/测试依赖](../docs/installation.md)后，按对应 skill 组织实际任务；以下列出仓库维护的数值或持久化回归入口。`tests/test_documentation.py` 只做文档链接、导入和调用签名检查，不执行这些数值测试，也不依赖本机 `scripts/` 下的计算项目。

| 任务 | 稳定入口 | 检查范围 |
|---|---|---|
| 时间选择与重建 | [选择回归](../tests/sampling/temporal/test_select_t.py)、[重建回归](../tests/sampling/temporal/test_slot_reconstruction.py) | 时间选择、物理配对、槽布局与贡献计数 |
| 空间取样 | [空间点回归](../tests/sampling/spatial/test_spatial_sampling.py) | 点集设计、实际坐标与抽样约定 |
| Laplace 基底 | [本征矢回归](../tests/production/eigenvector/test_laplace_eigvector.py) | 小格点与独立稠密谱 |
| Elemental | [生产回归](../tests/production/elemental/test_elemental.py) | 显式格点/规范链参考与各端点扇区 |
| meson55 预制表与绑定 | [使用说明](../docs/modules/operators/meson55.md)、[目录匹配回归](../tests/test_elemental_preset_matching.py) | 配方版本、真实有序分量与目录匹配 |
| 传播子 | [生产回归](../tests/production/perambulator/test_propagator.py) | 小格点 Dirac 求解及源汇投影 |
| Correlator | [计划/结果往返](../tests/test_correlator_frontend_roundtrip.py) | 固定/完整计划读回、坐标与结果保存 |
| 完整 cfg 输出 | [后处理回归](../tests/test_gpu_output_postprocessing.py) | CPU/协议替身与可选真实 CuPy 用例分别标明；独立 dt/counts/NaN/one-point 参考 |
| 共享读取 | [共享数据回归](../tests/storage/test_shared_data_contract.py) | 物理读取描述、共享准备及独立字段结果 |
| 通用恢复 | [增量登记回归](../tests/storage/test_registration_incremental.py) | 已完成覆盖与增量登记；恢复契约见[项目恢复](../docs/modules/dataset_planning/dataset_planning.md#project-resume) |

这些入口的具体断言和运行方式以测试源码及[测试说明](../tests/README.md)为准。小规模 CPU 或协议替身不证明真实系综、生产规模内存/性能、GPU、QUDA、MPI；有真实设备用例的文件须区分实际执行与 skip。

## 维护命令与边界

在仓库根目录执行：

```sh
python tools/check_public_api.py
python tools/check_docs.py
python -m unittest discover -s tests -p 'test_public_api.py' -v
python -m unittest discover -s tests -p 'test_documentation.py' -v
```

以上测试也由完整 `python -m unittest discover -s tests -v` 发现。

- 导出检查核对清单、遗漏/新增模块、缺失名称、重复或私有 `__all__` 项，以及短入口、用户别名与正式接口的对象身份
- 文档检查扫描 README、架构、AGENTS，以及 `docs/`、测试、基准、CI、约束说明和 `.agents/skills/` 的渐进参考；本机不跟踪的 `scripts/` 计算项目不进入该门禁。普通 Markdown 本地链接与标题锚点会校验，代码中的伪链接忽略。HTTP 等外链和跳出源码树的历史证据不联网核查
- Python 围栏及 skill 自带的可复用脚本检查真实导入；对直接导入的函数、类及其可解析属性调用用 `inspect.signature(...).bind(...)` 核对参数。此步骤不执行文档片段，不检查数值、类型、动态实例方法、跨片段变量或带 `...` / `*args` / `**kwargs` 的模板调用
- API 签名说明（例如裸 `*`、类型标注及 `->`）用 `text` 围栏；`python` 围栏必须可以解析。依赖用户数据、真实设备或外部服务的模板不宣称可独立运行；数值执行由对应测试负责

有意修改公开导出时，先同步实现、调用方与使用说明，再运行 `python tools/check_public_api.py --write-inventory`，逐项审查清单差异，并运行相关测试及完整 CPU 回归。不要为掩盖未知接口变化而直接更新清单；平台限制按对应模块要求另行验证。
