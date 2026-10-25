# 数据集兼容性与依赖规划

`lqcdweave.planning.dataset` 根据科学契约和已登记数据的观察结果，判断数据能否复用、哪些首轴位置缺失，以及多个生产阶段的依赖顺序。它不从目录名或文件名发现科学身份，也不决定数值算法。默认 NPY/HDF5 布局见[数据保存格式](../storage/data_storage.md)，基本读写边界见[Dataset 生命周期](../storage/dataset_lifecycle.md)。

<a id="planning-a-requirement"></a>
## 规划一个数据请求

```python
from lqcdweave.planning.dataset import plan_requirement

plan = plan_requirement(
    requested_spec,
    index.observations(),
    subset_axes=("time", "mode"),
)
```

`requested_spec` 是 `DatasetSpec`。观察记录描述实际 `data_id`、科学契约、来源和已完成覆盖。只有兼容且覆盖充足的数据集能复用。结果字段如下：

| 字段 | 含义 |
|---|---|
| `action` | `reuse`、`compute` 或 `blocked` |
| `data_id` | 一个被选中的完整数据集身份，或 `None` |
| `completed` | 请求首轴坐标中已有的槽位置 |
| `missing` | 仍需生产的请求首轴槽位置 |
| `selection` | 其它轴上按请求顺序排列的实际坐标 |
| `reasons` | 候选排除、覆盖和确定性选择说明 |

兼容性比较包括 schema、kind、系综/cfg/格点、dtype、definition、conventions、父身份、轴名称与顺序、基底轴及实际 `data_id`。`DatasetSpec` 保证基底绑定坐标与相应轴的有序标签一致。Ne、shape、seed、求解参数或同名文件都不能替代实际基底身份。求解精度及物理约定仍由科学定义明确记录。

坐标按 JSON 值比较，而不是按数组长度或存储位置比较。默认所有轴坐标和顺序必须相同。`subset_axes` 显式允许从候选数据集中读取坐标子集：

- 非首轴必须覆盖请求的所有坐标；选择保留请求顺序。
- 首轴允许已有部分覆盖；结果用请求位置报告 `completed` 和 `missing`，即使候选文件中的坐标顺序不同。
- 规划只选择一个候选，不拼接多个数据集补齐一次请求。候选先按覆盖位置数排序，再按 `data_id` 稳定排序。

`data_id=...` 可将查询限制到一个已知身份。已知身份冲突、配置不符或未声明的科学定义不会由 planner 猜测。严格科学匹配所需身份未知时，结果为 `blocked`；provenance 保留供调用方检查，但不替代科学兼容字段。

## 规划依赖图

```python
from lqcdweave.contracts.dataset import DatasetRequest
from lqcdweave.planning.dataset import plan_dataset_graph

requests = {
    "basis": DatasetRequest(basis_template),
    "vsv": DatasetRequest({
        **vsv_template,
        "bases": {
            **vsv_template["bases"],
            "mode": {"data_id": {"request": "basis"},
                     "indices": vsv_template["axes"][1]["coordinates"]},
        },
        "parents": [{"request": "basis"}],
    }, subset_axes=("time",)),
}
graph = plan_dataset_graph(requests, index.observations())
```

`DatasetRequest` 保存未解析模板及去重后的逻辑依赖。逻辑引用只能出现在声明的父项、基底身份及科学输入身份槽；轴坐标、dtype、配置和普通约定值不能包含逻辑引用。`resolve(actual_ids)` 只有在所有父请求都绑定实际 `data_id` 后才返回可发布的 `DatasetSpec`。

`plan_dataset_graph` 检查缺失节点和循环，返回稳定拓扑顺序、各阶段决策和已解析的父绑定。阶段只有在父数据集完整且兼容时才得到实际 `spec` 和父 ID。部分首轴覆盖可以用于本阶段补算，但不能作为下游的完整父数据。可用 `data_ids={"node": "actual-id"}` 或请求自身的 `data_id` 固定输入；相互冲突的固定身份会报错。

Planner 只决定科学兼容、复用范围和依赖关系；它不调度数值算法、合并来源或改变 `DatasetSpec` 中的实际坐标。

## 执行显式 producer

```python
from lqcdweave.runtime.datasets import run_dataset_requests

def produce_vsv(spec, inputs, writer):
    basis = inputs["basis"]
    for slot in writer.missing:
        time = spec.coordinates(spec.axis_names[0])[slot]
        writer.write(slot, calculate_row(time, basis))

data_ids = run_dataset_requests(
    requests, index,
    outputs={"basis": "run/basis", "vsv": "run/vsv"},
    provenance={"basis": basis_provenance, "vsv": vsv_provenance},
    producers={"basis": produce_basis, "vsv": produce_vsv},
    resume=True,
)
```

调用方显式提供输出目录、provenance 和 producer。producer 接收解析后的 spec、只读依赖视图和 writer；借用输入须在返回前消费或复制。writer 按 `DatasetSpec` 的首轴槽写数据，默认落成每 cfg 一个完整 NPY；多字段数据使用 HDF5。runtime 在阶段完成后发布数据集并返回实际 ID。具体计算仍由 production/contraction 模块或调用方选择。

依赖图不是一次全局原子交易。已完成的父阶段可以在后续阶段失败前发布；重试时仍按科学定义及实际父身份规划。生产期间的恢复状态和公开文件布局分离，细节以当前 Dataset writer/runtime 验收为准。

## 坐标视图

```python
from lqcdweave.runtime.datasets import open_dataset_view

with open_dataset_view(
    index, requested_spec, subset_axes=("time",), data_id=actual_data_id,
) as view:
    row = view.read(time=requested_spec.coordinates("time")[0])
```

视图代表已有数据集上的一个科学兼容选择，不创建新的持久数据集，也不改变父 ID。首轴必须显式绑定，非首轴选择仅能使用请求视图中的有序坐标。数据由底层 reader 提供；需要跨关闭后使用时，调用方应复制数组。

<a id="project-resume"></a>
## 项目回执

项目回执可将一组逻辑请求、解析后的定义、依赖和 `data_id` 记录在一起。`plan_project_dataset_requests` 与 `run_project_dataset_requests` 限定候选查询范围；项目序列接口用于显式组织多个阶段。回执辅助复用已有输入，不是数值结果校验凭据或第二份科学定义。序列恢复和项目 retention 的高级细节不在本页承诺；使用前需核对当前接口和验收范围。

## Diagram 输入规划

`planning.diagram_measurement` 可检查 Diagram 所需 VSV/PSV/VSP/PSP、elemental 和点集数据的配置、端点身份、坐标与时间覆盖。它只做科学契约规划，不读取完整数值、不展开或执行收缩。详见[Diagram 测量](../measurements/diagram_measurement.md)。

## 相关验证

科学兼容和依赖图用例位于 `tests/planning/test_dataset_planning.py`、`tests/planning/test_dataset_graph.py`、`tests/test_dataset_graph_runtime.py`、`tests/test_dataset_流程.py`。默认存储布局与读写行为由 storage 和 流程 测试覆盖；格式和恢复细节以当前验收为准。
