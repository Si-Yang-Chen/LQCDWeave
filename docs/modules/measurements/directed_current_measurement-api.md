# Directed current 流程：规划与 shared writers

该模块连接已登记数据契约与 canonical directed-current 数值核，提供元数据规划和同步行写出。它不拥有固定测量运行器、持久恢复状态或输出目录策略。

## 公开接口

```text
plan_indexed_directed_current_pair(index, first_id, second_id, vsv_id,
    first_terms, second_terms, *, first_momentum=0, second_momentum=0,
    usedNe=None, first_times=None, second_times=None,
    data_ids=None, source_policy="allow_unknown")

plan_indexed_directed_current_chain(index, current_id, incoming_vsv_id,
    outgoing_vsv_id, terms, *, momentum=0, usedNe=None,
    source_times=None, sink_times=None, anchor_times=None,
    data_ids=None, source_policy="allow_unknown")

write_directed_current_pair_rows(first, second, vsv, writer)
write_directed_current_chain_rows(current, incoming_vsv, outgoing_vsv, writer)
```

Planner 会通过 `index.open` 打开实际输入以构造请求，并对 `index.observations()` 做完整观察校验；该过程只读，但 storage 可能读取数据块做完整性验证。随后它验证请求的数据覆盖、轴、端点、基底、动量及算符配方，不执行数值收缩、不写输出。`data_ids` 按请求节点限制可选的实际候选；请求本身固定的输入 ID 仍保持固定。省略时按通用 DatasetIndex 规划策略解析候选。来源策略验证图中实际输入的来源声明，不创建项目 receipt。

writer 接收调用者已经打开的 reader/view 与由调用者创建的 `DatasetWriter`。它只消费缺失行，不关闭 borrowed inputs、不发布 dataset。输出 Spec 应由调用者按实际输出时间轴、输入 ID、配方、坐标与精度定义；调用者负责完整覆盖判断、发布、索引登记和恢复行为。

## 保留的科学边界

pair 调用原 `contract_directed_current_pair_v2v`；chain writer 沿用现有单链数值核。配方按输入顺序保留复系数和 normalization；输出不额外乘 Wick 符号、不取实部或执行源平均。periodic/open 路径约定分别按每一项的真实端点核验。

common storage/runtime 提供通用身份、写入、发布和生命周期机制。directed-current 模块没有固定 row schema runner、receipt 编排或自动 resume helper。更复杂的持久格式应在调用方按需求以 DatasetSpec、DatasetWriter 与数值接口组合，并独立验证。
