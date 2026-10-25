# 本软件

面向大规模格点 QCD计算，提供从规范组态到关联函数的完整流程，并支持基础数据在不同课题间复用。

总体架构见[ARCHITECTURE.md](ARCHITECTURE.md)，协作约定见[AGENTS.md](AGENTS.md)，安装步骤见[docs/installation.md](docs/installation.md)，各任务用法见[.agents/skills/](.agents/skills/)。

## 来源与许可证

本软件以 [IHEP-LQCD/EasyDistillation](https://github.com/IHEP-LQCD/EasyDistillation) 为直接基础进行重构，重新组织了模块边界、数据契约和计算流程，添加了额外的计算组件。原始上游及各贡献者的工作构成本软件的基础；本仓库的接口和支持范围以当前代码及文档为准，不承诺与上游接口兼容。

本软件代码及随附文档采用 [MIT 许可证](LICENSE)，保留原始上游的 `Copyright (c) 2023 EasyDistillation developers` 声明。MIT 允许使用、修改和再分发，包括商业用途；分发软件副本或实质性部分时须保留版权和许可声明。软件按原样提供，不作担保，完整条款以许可证原文为准。

格点对称性由本软件的 [symmetry 模块](docs/modules/symmetry/README.md)提供，统一适用本软件的 MIT 许可证。外部依赖遵循各自许可证，本软件许可证不替代其条款。

科研成果使用本软件时，建议注明本仓库及所用版本或提交，并致谢原始上游 IHEP-LQCD/EasyDistillation 及相关贡献者；所用算法和外部软件按实际使用情况引用。这是科研引用建议，不是 MIT 许可证之外的使用条件。数值结果的适用性应结合具体输入、精度和运行环境核验，测试通过不代表任意物理任务均已验证。

