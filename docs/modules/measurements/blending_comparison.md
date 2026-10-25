# blending 与 localized_blending 的共同缩并脚本

两路测量均使用当前 correlator 入口，输入的实际低模、抽样池与统计身份须匹配。

## 同一高模抽样维度的两路比较

比较只保留 blending 和全颜色 localized_blending，每点保留三个颜色。
R列D与R/3个P点的高模抽样分量数相同，低模、自旋、算符与归一化相同。
R须为3的倍数，并满足D列数和P点容量。一般同池双插入图需足够的不同点。

两路测量各自绑定当前输入，并使用相同的 `calculate` 和 `elemental_weighted` 约定。
具体生成器与绑定见[correlator skill](../../../.agents/skills/lqcdweave-correlator/SKILL.md)；
旧项目 helper 不属于可导入的库接口。

历史单色数值不能当作全颜色数值使用。
两路各自保存实际统计来源与`comparison_dimensions`，随机实现的两路数值无需相等。

输入的共同实际有序低模、点容量、D 父维度及 P 的投影集合必须匹配；改变缩并参数不能改变已保存传播子的投影定义。

blending 通过 `adapters.dilution_prefix` 选择已有单组 Haar QR 向量的前 R 列，父模式
集合保持完整，实际列实现及派生视图身份保留。新的 m=R 进入同分量/异分量概率；
已预计权的局部 D elemental 按新旧概率比重计权，继续使用同一个 `elemental_weighted`
选项。该接口只用于单独的前缀估计量，不支持与原父组样本在同一图中联合缩并。
不能只设置 `usedNe={'D': R}` 后保留原父组的统计元数据。非 QR、未知实际来源、
多个 D 分块、非完整补空间会明确拒绝；预计权的 D 前缀暂不支持时间位移算符。旧 GaussianDIS 名称本身不能
证明 Haar QR 条件；实际 reader 的来源与归一化须按旧输入说明核验。

共享/独立池关系改变时必须使用相应兼容模板。

| 路线 | elemental 扇区 | 传播子读取视图 |
|---|---|---|
| blending，单分块 D | V→V、V→D、D→V、D→D | VSV、VSD、DSV、DSD |
| localized_blending | V→V、V→P、P→V、P→P | VSV、VSP、PSV、PSP |

源端由 `Meson(..., source=True)` 处理 dagger。双向读取可使用同一个已经具有正确
时间约定和 Gamma5 视图的传播子 reader。实际端点身份、父组大小、混合列数、
抽样池和归一化仍从输入数据传入；统一计权不等于忽略这些参数。

两路测量各自绑定当前输入，并使用相同的 `calculate` 和 `elemental_weighted` 约定。
具体生成器与绑定见[correlator skill](../../../.agents/skills/lqcdweave-correlator/SKILL.md)；
旧项目 helper 不属于可导入的库接口。

## 抽样组独立性与时间轴

`groups_independent=True` 只用于真实 D/P 采样池设计能够保证不同非空组相互独立的情形。
是否独立由实际抽样来源决定，不能因为两端使用不同时间就推断独立。若两端共享同一池，
reader 元数据中应保留相同的池 ID，并保持 `groups_independent=None`；不能为不同时间伪造
不同 ID。图中存在多个非空组时，只有确认真实独立后才能传 `True`。

`time_axes=None` 时，两条路线都扫描 `{0: range(Lt), 1: range(Lt)}`。小范围实测可传
`time_axes={0: (0,), 1: (12,)}`；两条路线会使用同一组所选绝对时间轴，返回结果仍保留
绝对源时间 `0` 和绝对汇时间 `12`，不会自动改成时间差或把坐标重排。完成验证后，
再按相同物理坐标规则重建时间差并平均。

这里沿用[介子绑定说明](../../../.agents/skills/lqcdweave-correlator/references/binding.md)中的一个连通 Wick 图口径。
两条路线必须使用同一物理味道/Wick 系数；完整物理关联函数按测量配方补齐。
返回值保留绝对源时间与绝对汇时间，随后按同一规则转换时间差并平均源时间，
见[两点重建](../results/two_point.md)。两条路线完成相同目标的每 cfg 结果后，才进行共同 cfg 的
配对重采样、关联函数及误差比较；缩并耗时和实际求解成本分别记录。

## 实际输入要求

直接比较完整局域 pion 两点函数时，先用零动量，再用最小非零动量；两者均使用
同一作用量、规范场、源时间集合、空间归一化和完整目标空间。
blending 的单个 D 父组表示 V 之外的完整补空间，不能用当前仅含 101–428 模的
medium-D 替代。localized 的点端也须对应相同目标空间，而不是换成纯低模目标。

完整补空间 D 已有历史数据，不需要因本次比较重新生成。昆山根目录为
`/public/home/huzhch_p/work/randomV/output/beta6.20_mu-0.2770_ms-0.2400_L24x72/`
下的 `randV_eig100_rand400_StoutSmear_20_0.12_GaussianDIS`；例如 cfg10000 的
`randomVector_t000_cfg10000_Neig100_Nrv400.pkl` 已在只读盘点中确认存在。
其400个实际随机向量对应旧100模之外的完整补空间，单组父维度为41,372。
配套500×500传播子按前100列V、后400列D拆为VSV、VSD、DSV、DSD，具体布局、
生成来源、时间约定和作用量证据见[旧100/400输入](../perambulator/perambulator_reuse.md)。

这里需要完成的是旧数据 reader 与共同缩并入口的接线，而非补产完整 D。
pion elemental 按实际 V/D 向量与所选动量读取或生成；不能拿旧RHO elemental替代。
完整D的父空间维度也不能误写成保留的400列或新方案的328个medium父模式。
已有64点 PSV/PSP 须读取其真实定义，不能按文件名假定已经减除428维；
比较两条路线时，应匹配它们共同的旧100模和作用量。此次没有提交额外传播子生产任务。
