# 额外慢测清单

这些用例只在明确运行慢测时收集。文件名不使用普通 `test*.py` 模式，
所以 `unittest discover`、默认 pytest 和 `tools/test_tiers.py --tier all`
均不执行这条流程。已有科学参考测试的分类不变。

```sh
python tools/test_tiers.py --tier slow --list
python tools/test_tiers.py --tier slow --json /external/weak-field-audit.json
```

需要安装项目的 Laplace/correlator extras、SciPy；固定输入在
`tests/data/weak_field`。JSON 保留每个用例的数值核验次数、最大绝对误差和相对
L2 误差；失败时也保留已完成的核验。缺依赖记为未完成，不等于通过。
测试数据全部写入系统临时目录，结束时清理，报告路径由调用者指定。

| 慢测文件 | 规模与内容 | 独立参考 |
|---|---|---|
| `weak_field_pipeline.py` | 完整 `4³×8`，EV→NV/SS→九类 PR/EL→介子/重子/跨时间流→保存与读回 | 格点 Laplace 稠密对角化、独立稀疏 Wilson/平移 stencil、逐点规范链、显式概率与矩阵迹 |

具体参数、各阶段覆盖、结果和剩余缺口见
[weak_field 科学审计](../reference/weak_field_audit.md)。这条 CPU 流程不证明
GPU、QUDA、MPI、历史 clover 作用量或生产规模性能已经验收。
