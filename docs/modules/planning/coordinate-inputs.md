# 已选路径的坐标输入描述

`contraction/planning/coordinate_inputs.py` 为固定标量路径补充坐标输入描述，
不重新展开算符、不读取数组，也不调用计算路径搜索。收缩编排提供 E 已保存的读取请求、
完整 preparation closure、TensorSpec 和本次真实坐标；P 使用与完整 cfg 绑定相同的
请求替换、别名身份与准备依赖顺序规则。

`bind_coordinate_inputs(template_scene_plan, requests, preparations, specs,
coordinate_points, *, read_metadata=None, reader_aliases=(), debug=False)` 返回
`CoordinateInputBinding`。`source_names` 按原坐标顺序保存每个原始 source key
对应的实际标量准备 ID；`requests/blocks/preparations/specs` 保存按真实读取和
完整配方身份去重后的纯描述。相同 shape 不能证明输入共享；坐标查表、实际元数据及
物理别名检查仍在数值读取前完成。未知 reader 身份沿用显式 `reader_aliases` 契约。

`describe_coordinate_batch(binding, group_indices, *, lane_axis, key_prefix="")`
返回 `CoordinateBatchInputs`。`inputs` 是既有 `DiagramInputs`，其 loader 为 None；
`preparation_closure/specs/requests` 交给 E 的 `restore_input_loader` 恢复能力。
不同批的顶层 source 可由 `key_prefix` 区分，标量成员和物理读取 ID 保持全任务共享。
`source_map` 供固定路径提升使用。内部生成的 `inputs.term_ids` 是位置占位符，收缩编排必须
替换为原任务的 term IDs；原输出配方及其复系数不变。

同一批内，实际准备 ID 完全相同的输入保持原 rank；只有实际 ID 不同的输入增加一个
共同的 leading lane 轴，并用现有 `TensorPreparation("stack")` 组合标量成员。
涉及该轴的 Term 保留该轴到输出，其它 Term 的 `batch_output_axes` 为 None，输出恢复时
按所请求坐标广播。宽度为 1 或全部为重复坐标的批可以没有新增轴。批尾不填充。
原有 stack 配方作为标量成员保留，因此算符内部的分量轴与坐标轴不会混淆。

顶层 preparations 严格覆盖本批的 logical sources；完整闭包单独保存，避免把内部
成员重复计为同时驻留的场景输入。准备工作区保守计入一个成员临时值和其已有工作区，
按 NumPy/CuPy 目标归入 host/device，精度沿用已经选择的计算精度。调用方随后必须使用
既有 `plan_scene_storage`、批内核 lowering 和全任务 storage join 完成资源审计，
不能以本输入描述代替中间值、打包或跨批驻留预算。

当前仅接受有 TensorSpec、原始读取和纯配方的 dense 输入；opaque callback、sparse
record、baryon join 在读取前明确拒绝。本模块不新增数值内核；CPU 复数矩形模式参考、
40 个汇时刻、非整除批尾、共享源、原有 stack、源端共轭和原生 storage/replay 的验证见
`tests/correlator/planning/test_coordinate_inputs.py`。CuPy 目标在此仅完成离线描述验证，
真实 GPU 验收由公共执行路径测试承担。
