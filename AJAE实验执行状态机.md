# AJAE 细粒度实验执行状态机

> 依据：`AJAE新主线方案.md`。本文件把主线方案中的不可变约束、四个 Decision Gates、B0–B5 对照、正常运动安全、对象尺度诊断、开发纪律与一次性真实 OOD 验证，拆成可顺序执行的细粒度实验节点。

## 0. 使用规则

1. **一次只执行当前已解锁实验。** 后继实验只有在当前实验 PASS 后才能解锁，除非本节点明确给出 FAIL 分支。
2. **一个实验只回答一个局部问题。** 不允许把失败解释成“整体 Gate 失败后再内部探索”；需要的诊断节点已经预先写在状态机中。
3. **主线方案没有给出数值阈值的地方，不允许事后决定。** 这些阈值必须在该指标首次正式运行前，基于科学容忍度和数据规模写入协议并冻结。
4. **19 条公开真实异常和 51 条隐藏测试严格后置。** E97–E98 完成前不得访问 19；Gate 4 通过前不得使用 51。
5. **任何会改变科学方法的修改都会使其下游实验失效。** 文中 FAIL 路径会明确指出需要回滚到哪个最早节点。
6. **PASS 不等于“效果很好”。** 对机械/资格实验，PASS 只表示该局部事实已被验证；对 B1/B3/B4 等科学实验，PASS 才对应相应科学主张。
7. **所有实验必须保存最小证据包：** command、resolved config、git commit、seed（若有）、输入数据 identity/hash、输出 artifact、日志、判定脚本与最终 PASS/FAIL。

## 1. 实验执行架构图

下图中每个实验序号都是一个独立节点。实线为主 PASS 路径；虚线为关键 FAIL 回退路径。节点详情中的“FAIL→”比图中的虚线更精确，应以节点详情为最终准则。


```mermaid
flowchart TB
  subgraph P0["Phase 0｜协议、数据纪律与官方 STU 运行资格"]
    E00["E00 工作区与协议快照冻结"]
    E01["E01 公开 19 条真实异常访问保护"]
    E02["E02 51 条隐藏测试访问保护"]
    E03["E03 官方 STU 依赖导入"]
    E04["E04 官方 STU checkpoint 实例化"]
    E05["E05 STU 单真实帧前向"]
    E06["E06 STU 冻结不变量"]
    E07["E07 缓存身份与跨世界隔离"]
    E00 --> E01
    E01 --> E02
    E02 --> E03
    E03 --> E04
    E04 --> E05
    E05 --> E06
    E06 --> E07
  end
  subgraph P1["Phase 1｜规范 OS1-128 射线身份"]
    E08["E08 槽位总数与空槽规律"]
    E09["E09-v2 128 beam row 身份恢复"]
    E10["E10-v1 azimuth column 连续性"]
    E11["E11 slot→ray 跨帧方向稳定性"]
    E12["E12 多回波重排风险"]
    E13["E13 raw→ray→raw 点数往返"]
    E14["E14 raw→ray→raw 几何往返"]
    E15["E15 多序列射线资格确认"]
    E08 --> E09
    E09 --> E10
    E10 --> E11
    E11 --> E12
    E12 --> E13
    E13 --> E14
    E14 --> E15
  end
  subgraph P2["Phase 2｜程序化几何、正常控制与放置"]
    E16["E16 primitive 数值有限与有界"]
    E17["E17 单 primitive 射线求交"]
    E18["E18 CSG 与连续形变稳定性"]
    E19["E19 单连通实体拒绝"]
    E20["E20 形状/尺度/轴比/材质解耦"]
    E21["E21 局部支撑平面估计"]
    E22["E22 悬空与埋地检查"]
    E23["E23 已观测正常几何碰撞"]
    E24["E24 插入实体相互碰撞"]
    E25["E25 正常控制语义放置"]
    E26["E26 完整世界规格确定性"]
    E16 --> E17
    E17 --> E18
    E18 --> E19
    E19 --> E20
    E20 --> E21
    E21 --> E22
    E22 --> E23
    E23 --> E24
    E24 --> E25
    E25 --> E26
  end
  subgraph P3["Phase 3｜第一回波反事实渲染机械链"]
    E27["E27 normal-control 几何命中"]
    E28["E28 anomaly-proxy 几何命中"]
    E29["E29 回波概率非退化"]
    E30["E30 normal-control 有效回波"]
    E31["E31 anomaly-proxy 有效回波"]
    E32["E32 插入物遮挡背景"]
    E33["E33 正常前景遮挡插入物"]
    E34["E34 空射线新增与拒绝"]
    E35["E35 强度支持范围"]
    E36["E36 normal/proxy 共用渲染路径"]
    E37["E37 重叠窗口共享帧一致性"]
    E27 --> E28
    E28 --> E29
    E29 --> E30
    E30 --> E31
    E31 --> E32
    E32 --> E33
    E33 --> E34
    E34 --> E35
    E35 --> E36
    E36 --> E37
  end
  subgraph P4["Phase 4｜Gate 1：传感器一致性与反作弊"]
    E38["E38 per-beam 回波率一致性"]
    E39["E39 per-range 回波率一致性"]
    E40["E40 beam×range 强度分布"]
    E41["E41 empty→valid 比例"]
    E42["E42 单实体可见点数分布"]
    E43["E43 连续帧可见点数变化"]
    E44["E44 遮挡率分布"]
    E45["E45 三方严格匹配审计集"]
    E46["E46 真实正常 vs 渲染正常来源分类"]
    E47["E47 来源指纹归因消融"]
    E48["E48 normal-control vs anomaly-proxy 难度分类"]
    E49["E49 Gate 1 正式裁决"]
    E38 --> E39
    E39 --> E40
    E40 --> E41
    E41 --> E42
    E42 --> E43
    E43 --> E44
    E44 --> E45
    E45 --> E46
    E46 --> E47
    E47 --> E48
    E48 --> E49
  end
  subgraph P5["Phase 5｜冻结 STU 点接口与五帧坐标"]
    E50["E50 128D STU 高层特征接口"]
    E51["E51 稀疏体素→原始点逆映射"]
    E52["E52 共享体素下的原始点身份"]
    E53["E53 官方 query assignment"]
    E54["E54 19D 语义证据与可靠性"]
    E55["E55 AJAE 真实输入张量"]
    E56["E56 中心坐标对齐"]
    E50 --> E51
    E51 --> E52
    E52 --> E53
    E53 --> E54
    E54 --> E55
    E55 --> E56
  end
  subgraph P6["Phase 6｜固定 201 开发试验台与评价器"]
    E57["E57 24 条 in-generator 开发世界冻结"]
    E58["E58 6 条 held-out 诊断世界冻结"]
    E59["E59 开发世界 N_vis / O / d 覆盖"]
    E60["E60 开发世界 V=1..5 覆盖"]
    E61["E61 pure-normal 与 moving-normal 开发子集"]
    E62["E62 自研 evaluator 与官方 evaluator 一致性"]
    E63["E63 开发决策规则冻结"]
    E57 --> E58
    E58 --> E59
    E59 --> E60
    E60 --> E61
    E61 --> E62
    E62 --> E63
  end
  subgraph P7["Phase 7｜AJAE 模型机械单元资格"]
    E64["E64 时间身份体素隔离"]
    E65["E65 mean-max 池化数值"]
    E66["E66 按时间差分层邻域"]
    E67["E67 空跨帧分支与 gate"]
    E68["E68 同帧残差生存路径"]
    E69["E69 同帧 3-NN 上采样"]
    E70["E70 平衡 BCE 空类别安全"]
    E71["E71 概率融合公式单元测试"]
    E64 --> E65
    E65 --> E66
    E66 --> E67
    E67 --> E68
    E68 --> E69
    E69 --> E70
    E70 --> E71
  end
  subgraph P8["Phase 8｜Gate 2：异常代理监督是否有效"]
    E72["E72 B0 冻结 STU 单帧参考"]
    E73["E73 B1 单帧 smoke train"]
    E74["E74 B1 三独立训练种子"]
    E75["E75 B1 vs B0 代理监督效应"]
    E76["E76 B1 正常安全"]
    E77["E77 Gate 2 正式裁决"]
    E72 --> E73
    E73 --> E74
    E74 --> E75
    E75 --> E76
    E76 --> E77
  end
  subgraph P9["Phase 9｜Gate 3：跨帧信息是否提供可识别增益"]
    E78["E78 B2 无跨帧五帧对照"]
    E79["E79 B3 五帧 smoke train"]
    E80["E80 B3 三独立训练种子"]
    E81["E81 B3 vs B1"]
    E82["E82 B3 vs B2"]
    E83["E83 五帧正常运动安全"]
    E84["E84 Gate 3 正式裁决"]
    E78 --> E79
    E79 --> E80
    E80 --> E81
    E81 --> E82
    E82 --> E83
    E83 --> E84
  end
  subgraph P10["Phase 10｜时间位置校准与 B4 融合"]
    E85["E85 q 位置分数诊断"]
    E86["E86 真实重叠点身份与 m_p 覆盖"]
    E87["E87 B4 融合评估"]
    E88["E88 B4 vs B3"]
    E85 --> E86
    E86 --> E87
    E87 --> E88
  end
  subgraph P11["Phase 11｜机制、安全、对象尺度与因果消融"]
    E89["E89 实体内部得分方差"]
    E90["E90 异常边界泄漏"]
    E91["E91 V=1..5 可见性趋势"]
    E92["E92 B5 因果窗口正确性"]
    E93["E93 B5 因果性能"]
    E94["E94 计算成本与输入公平性"]
    E89 --> E90
    E90 --> E91
    E91 --> E92
    E92 --> E93
    E93 --> E94
  end
  subgraph P12["Phase 12｜方法冻结"]
    E95["E95 最终模型选择规则执行"]
    E96["E96 阈值与 DBSCAN 冻结"]
    E97["E97 AJAE Method Freeze Manifest v1"]
    E98["E98 冻结完整性演练"]
    E95 --> E96
    E96 --> E97
    E97 --> E98
  end
  subgraph P13["Phase 13｜一次性真实 OOD 确认与最终测试"]
    E99["E99 19 条真实 OOD 锁定推理"]
    E100["E100 真实 OOD 官方点级指标"]
    E101["E101 真实 OOD 对象级指标"]
    E102["E102 真实正常运动安全"]
    E103["E103 Gate 4 迁移裁决"]
    E104["E104 51 条隐藏测试最终提交"]
    E99 --> E100
    E100 --> E101
    E101 --> E102
    E102 --> E103
    E103 --> E104
  end
  E07 --> E08
  E15 --> E16
  E26 --> E27
  E37 --> E38
  E49 --> E50
  E56 --> E57
  E63 --> E64
  E71 --> E72
  E77 --> E78
  E84 --> E85
  E88 --> E89
  E94 --> E95
  E98 --> E99
  E46 -. "FAIL:定位来源指纹" .-> E47
  E47 -. "修复传感器因素" .-> E38
  E48 -. "FAIL:重做 hard proxy" .-> E20
  E49 -. "Gate1 FAIL" .-> E08
  E75 -. "Gate2 FAIL" .-> E38
  E76 -. "安全 FAIL" .-> E38
  E81 -. "B3≤B1" .-> E64
  E82 -. "B3≤B2" .-> E64
  E83 -. "运动安全 FAIL" .-> E64
  E85 -. "q偏置" .-> E85
  E90 -. "边界泄漏 FAIL" .-> E64
  E98 -. "冻结完整性 FAIL" .-> E97
  E103 -. "Gate4 FAIL:研究周期停止" .-> STOP["STOP 当前研究周期结束"]
  E104 --> DONE["AJAE COMPLETE"]
```


## 2. 总体阶段与科学主张对应

- **Phase 0–4：Gate 1** —— 证明规范射线与第一回波反事实 renderer 足够可信，且不会留下明显来源捷径。
- **Phase 5–7：训练接口资格** —— 证明冻结 STU 点接口、五帧坐标、开发世界、评价器和 AJAE 机械结构按方案工作。
- **Phase 8：Gate 2** —— 证明 anomaly-proxy supervision 本身在新背景有效，即 B1 相对 B0 有增益且正常安全。
- **Phase 9：Gate 3** —— 证明跨帧信息提供可识别增益，即 B3>B1 且 B3>B2，并通过 moving-normal safety。
- **Phase 10–11：融合与机制** —— 判断 B4 是否有额外价值、时空共识机制是否有证据、因果版本代价如何。
- **Phase 12：Method Freeze** —— 冻结所有会影响结果的内容。
- **Phase 13：Gate 4 与隐藏测试** —— 一次性确认 proxy→real OOD transfer，之后才允许 51 hidden test。


# Phase 0｜协议、数据纪律与官方 STU 运行资格

## E00｜工作区与协议快照冻结

**目的 / 唯一问题**

建立后续所有实验的唯一可追溯起点。

**建模 / 实施**

记录 Git commit/branch/dirty state、protocol/dev 配置 hash、STU checkpoint hash、renderer/generator 版本、Python/PyTorch/CUDA 环境；生成实验注册表，不运行训练。

**PASS 条件**

所有后续实验都能唯一绑定同一快照，且关键协议文件进入版本控制。

**FAIL 条件**

仍有无法追溯的未提交核心文件、配置来源不明或产物无法绑定版本。

**状态转移**

- PASS → **E01**
- FAIL → **整理并提交/冻结当前协议快照后重跑 E00。**


## E01｜公开 19 条真实异常访问保护

**目的 / 唯一问题**

确保确认集在方法冻结前不会被任何旁路读取。

**建模 / 实施**

全仓库搜索 public/val/19 相关 loader；对所有读取标签/结果的入口加 freeze guard 或物理权限隔离；仅做访问保护测试，不读取确认标签。

**PASS 条件**

冻结前任何入口均不能读取 19 条标签/结果，且访问会被显式拒绝并记录。

**FAIL 条件**

存在绕过正式 evaluator 的标签读取路径。

**状态转移**

- PASS → **E02**
- FAIL → **封闭旁路后重跑 E01。**


## E02｜51 条隐藏测试访问保护

**目的 / 唯一问题**

保证最终测试在 Gate 4 之前不可被开发流程触碰。

**建模 / 实施**

检查 hidden/test/51 的 loader、路径、脚本和环境变量；建立只在最终提交阶段开放的保护。

**PASS 条件**

开发态无法读取或生成 51 条隐藏测试相关结果。

**FAIL 条件**

存在开发路径可访问 hidden test。

**状态转移**

- PASS → **E03**
- FAIL → **封闭访问后重跑 E02。**


## E03｜官方 STU 依赖导入

**目的 / 唯一问题**

确认工作区具备真实调用官方 STU 的最小环境。

**建模 / 实施**

只执行官方 STU import 链，验证 Hydra、OmegaConf、MinkowskiEngine、PyTorch3D 及官方模块版本兼容；不加载数据、不训练。

**PASS 条件**

官方模块完整导入，无缺失依赖或 ABI 冲突。

**FAIL 条件**

任一依赖缺失、版本不兼容或导入失败。

**状态转移**

- PASS → **E04**
- FAIL → **只修环境/依赖，再重跑 E03。**


## E04｜官方 STU checkpoint 实例化

**目的 / 唯一问题**

确认指定官方权重与当前代码接口兼容。

**建模 / 实施**

按主线指定的官方配置和 checkpoint 构造 STU，记录权重 hash 和缺失/多余 key。

**PASS 条件**

模型完整构造，权重加载符合官方预期，无未解释 key mismatch。

**FAIL 条件**

checkpoint/config/API 不兼容。

**状态转移**

- PASS → **E05**
- FAIL → **修 checkpoint/config 绑定后重跑 E04。**


## E05｜STU 单真实帧前向

**目的 / 唯一问题**

证明不是 toy tensor，而是真实 206 帧可以走通官方前向。

**建模 / 实施**

只取 206 的一个真实帧，运行官方 STU forward；检查所有关键输出 finite、shape 合法。

**PASS 条件**

真实帧前向成功，输出无 NaN/Inf。

**FAIL 条件**

真实输入无法前向或输出异常。

**状态转移**

- PASS → **E06**
- FAIL → **修输入适配/官方接口后重跑 E05。**


## E06｜STU 冻结不变量

**目的 / 唯一问题**

确认 STU 在 AJAE 中真正冻结，而不只是口头或 optimizer 排除。

**建模 / 实施**

构造一次非正式 smoke backward：检查 requires_grad=False、optimizer 参数与 STU 参数集合不相交、STU grad 为空；比较一次 optimizer step 前后 parameter 与 buffer hash，并保持 eval 模式。

**PASS 条件**

参数、buffer、梯度和模式状态全部不变。

**FAIL 条件**

任一 STU 参数可训练、产生梯度或状态发生变化。

**状态转移**

- PASS → **E07**
- FAIL → **修冻结/eval/optimizer 构造后重跑 E06。**


## E07｜缓存身份与跨世界隔离

**目的 / 唯一问题**

防止同一 frame_id 在不同反事实世界中错误复用渲染或 STU 特征。

**建模 / 实施**

将缓存身份至少绑定 world identity、frame identity、renderer/generator version、STU identity；构造两个 world 同 frame 的反例并比较 cached/uncached 输出。

**PASS 条件**

跨世界不会误命中；cached 与 uncached 逐点一致。

**FAIL 条件**

cache key 只含 frame_id、跨世界污染或缓存前后不一致。

**状态转移**

- PASS → **E08**
- FAIL → **修复 cache identity 后重跑 E07。**



# Phase 1｜规范 OS1-128 射线身份

## E08｜槽位总数与空槽规律

**目的 / 唯一问题**

确认原始文件的 slot 结构是否稳定到足以继续物理射线审计。

**建模 / 实施**

跨多帧统计总 slot 数、有效 slot 数、空槽模式和异常帧。

**PASS 条件**

结构稳定或异常模式有明确可处理规则。

**FAIL 条件**

slot 结构无稳定规律。

**状态转移**

- PASS → **E09-v2**
- FAIL → **先解析原始数据编码/slot 语义，再重跑 E08。**


## E09-v1｜128 beam / elevation 恢复（历史失败，禁止改写）

**目的 / 唯一问题**

原协议同时要求恢复垂直 beam row 身份，并要求每个 row 的跨帧中位仰角近似不变。

**建模 / 实施**

在 train/206 的全部 449 帧上，将固定 131072 个槽位按 128 行、每行 1024 列解释，统计每行真实回波的仰角中位数。预注册条件包括每帧恰好 128 行、每行有回波、全局相邻行间隔至少为 $0.15^\circ$，以及每行相对参考值的跨帧中位仰角偏差不超过 $0.10^\circ$。

**永久结果**

**E09-v1: FAIL。** 128 行始终存在且均有真实回波，所有帧的行中位仰角严格保持顺序，没有 row crossing 或 permutation；最小同帧相邻行间隔为 $0.215376^\circ$。但是，跨帧行中位仰角最大偏差为 $0.348507^\circ$，超过预注册的 $0.10^\circ$ 条件。同槽方向残差的 0.99 分位数为 $0.611656^\circ$，最大值为 $1.703179^\circ$。

该 FAIL 必须永久保留，不得因后续协议修订改写成 PASS。

**缺陷判定**

事后语义审查发现，$0.10^\circ$ 的跨帧行中位仰角条件测量的是物理方向不变性，而不是有序 beam row 身份能否恢复。它与 E11 的科学问题发生构念重叠，因此 E09-v1 被标记为 **specification defect**。该标记只说明原测量定义错误，不撤销原始 FAIL，也不把已观察结果用于放宽 E11。


## E09 协议修订｜拆分 row identity 与 physical direction

协议版本化修订如下：

1. E09-v2 只验证 128 个有序 beam row 的拓扑和身份能否逐帧确定性恢复。
2. E11 独立验证同一个规范 ray/slot 的单位物理方向是否跨帧稳定，以及能否安全建立 $\rho_f(r)$。
3. E09-v1 的 $0.10^\circ$ 跨帧固定仰角条件从 E09-v2 删除；不允许将该删除解释为 E11 已通过。
4. E09-v1 的全部输入、预注册判据、结果和 FAIL 结论继续保留为历史证据。


## E09-v2｜128 beam row 身份恢复

**目的 / 唯一问题**

是否能在每一帧中稳定恢复同一套 128 个有序 beam row 身份？

**观测对象**

只检查 row topology / identity，不判断单个 ray 或整行的跨帧绝对物理方向是否稳定。

**建模 / 实施**

对 train/206 的全部 449 个审计帧使用固定、确定性的恢复规则：将 131072 个原始槽位解释为 128 个候选 row、每行 1024 个候选 column；空槽仍使用 E08 冻结的 XYZ 全零规则；在每帧内按各 row 有效回波的仰角中位数从高到低赋予 row ID 0–127。对同一输入从原始扫描文件独立执行两次，并比较完整 row-ID 数组及规范化摘要哈希。

**重跑前冻结的 PASS 条件**

1. 每个审计帧均恢复恰好 128 个 row，每个 row 恰好对应 1024 个候选槽位。
2. 每个 row 在每个审计帧中至少有 512 个真实有效回波，即至少覆盖候选槽位的 50%；不得以空行或填充值构造 row。
3. 每帧 128 个 row 的中位仰角严格递减，候选 row 的恢复排序在所有帧中均为同一个 0–127 恒等排列，不发生 row crossing 或 permutation。
4. “相邻 row 不发生身份重叠到无法唯一排序”具体定义为：每帧任意相邻 row 的有效回波仰角中位数间隔均不小于 $0.10^\circ$。这等价于相邻中位位置各自保留 $\pm0.05^\circ$ 的非重叠排序带。$0.10^\circ$ 沿用 E09-v1 正式运行前已经存在的角度分辨容差，并基于 OS1-128 约 $0.35^\circ$ 的平均垂直行间距冻结，不依据 E09-v2 的新结果选择。
5. 两次独立执行必须产生逐元素完全相同的 row-ID 数组、每帧排序、支持计数、相邻间隔摘要和规范化摘要哈希。
6. 不要求 $|\tilde\theta_{b,f}-\tilde\theta_{b,\mathrm{ref}}|<0.10^\circ$，也不设置任何等价的跨帧固定物理仰角条件。

**正式结果（协议修订提交后执行）**

**E09-v2: PASS。** 449/449 帧均恢复 128 个 row，所有帧的候选 row 排序均为同一个 0–127 恒等排列，没有 crossing 或 permutation。每行每帧最少有 645 个真实有效回波；最小同帧相邻 row 中位仰角间隔为 $0.215376^\circ$。两次从原始文件独立执行得到完全相同的 row IDs、支持计数、中位数、相邻间隔、摘要和 SHA-256 `966baf6e2ea0cf86c9bbe9ee42834cfc8dbc228803188ea1ddd5e16d23b1161d`。

该 PASS 只证实有序 beam row 身份可以稳定恢复。它不证实任何 row、column、slot 或规范 ray 的跨帧物理方向不变；该问题仍由 E11 独立裁决。

**FAIL 条件**

任一帧的 row 数或候选槽位数错误；任一 row 的真实回波支持低于冻结下限；发生 row crossing、permutation；任一同帧相邻 row 的中位仰角间隔低于 $0.10^\circ$；或重复执行不能精确复现 row IDs 与摘要。

**状态转移**

- PASS → **E10-v1**
- FAIL → **停止解锁 E10，检查 row 恢复规则或原始槽位拓扑；任何新定义必须再次版本化修订并在重跑前冻结。**


## E10-v1｜azimuth column 连续性（历史失败，等待协议决定）

**目的 / 唯一问题**

验证方位角列可以形成稳定扫描序列。

**建模 / 实施**

对每个审计帧，将同一候选 column 中所有有效 row 的 XY 单位方向取圆周均值，得到 1024 个 column 方位代表值。分别检验文件列序的顺时针和逆时针假设，并以相邻代表值的模 $360^\circ$ 正向增量检查完整闭合周期。对同一输入从原始扫描文件独立执行两次。E10 只检查列拓扑、循环次序和可重复恢复，不要求第 $a$ 列在不同帧具有相同绝对方位相位；后者属于 E11。

**正式运行前冻结的 PASS 条件**

1. 每帧恰好恢复 1024 个 candidate columns，每列恰好包含 128 个候选 row。
2. 每列每帧至少有 64 个真实有效回波，即 row 支持率至少为 50%，且圆周均值有限、非退化。
3. 对每帧的两个方向假设，必须恰有一个方向使包含末列回到首列在内的全部 1024 个循环增量落在 $[0.10^\circ,0.60^\circ]$。OS1-128 的名义列步长为 $360^\circ/1024=0.3515625^\circ$；冻结区间排除重复列、逆序和接近漏掉整列的跳变，同时不要求逐列物理方向完全固定。
4. 449 帧选择的循环方向必须一致，原始 candidate column 的循环排列均为同一 0–1023 次序，不发生列 permutation 或内部跳转；首尾连接只作为正常周期边界。
5. 两次独立执行必须产生逐元素完全相同的列方向、支持计数、循环增量、排序摘要和规范化摘要哈希。
6. 不比较各帧第 0 列或任意固定列的绝对方位角，也不以跨帧 azimuth phase 漂移判定 E10；这些量只在 E11 按其独立冻结的判据裁决。

**正式结果**

**E10-v1: FAIL。** 每列每帧最少只有 16 个真实回波，低于冻结下限 64；仅 46/449 帧存在唯一且全部循环增量均落入 $[0.10^\circ,0.60^\circ]$ 的方向；正式跨 row 圆周均值估计器共有 4040 个增量越界。两次独立执行结果完全一致，SHA-256 均为 `3fdb998866a8858b157555feb7673de598408afd132442de59bcce33600328ac`，因此失败不是非确定性造成的。

失败后的定位诊断不改变判决：3331/4040 个越界发生在相邻两列支持均不少于 64 时，排除了“只有极稀疏列才失败”的解释；对同一 beam row 内 55682452 对相邻有效列进行检查时，全部步长落在 $[0.291824^\circ,0.406113^\circ]$，没有逆序或越界。这支持“跨 row 圆周均值混合了 beam 特定方位偏置，并随可见性组成变化产生伪跳变”的解释，但该诊断不能把 E10-v1 改写为 PASS。

如要继续，必须先决定并版本化预注册 E10-v2：直接测量每个已恢复 beam row 内的相邻列连续性，或使用显式校正 beam 方位偏置的列估计器。E11 在此之前保持锁定。

**FAIL 条件**

列数或支持不足；循环方向有歧义或跨帧翻转；任一循环增量超出冻结区间；发生列 permutation、内部跳转；或重复执行不能精确复现。

**状态转移**

- PASS → **E11**
- FAIL → **E11 保持锁定；先确定 E10-v2 的观测对象与恢复算法，版本化预注册后才能重跑。E10-v1 的 FAIL 永久保留。**


## E11｜slot→ray 跨帧方向稳定性

**目的 / 唯一问题**

对已经恢复的规范身份 $r=(b,a)$，判断同一个候选 ray 在不同帧中的单位方向是否足够稳定，从而能否安全建立 $\rho_f(r)$。

**建模 / 实施**

对同一候选 slot/ray 跨帧比较单位方向，并检查 azimuth phase、beam/column reconstruction、deskew 和坐标变换是否改变其物理含义。阈值必须在正式运行前独立冻结；E09-v1 已观察到的 0.99 分位数 $0.611656^\circ$ 和最大值 $1.703179^\circ$ 只能作为提前暴露的风险，不得用来修改 E11 判据。

**PASS 条件**

方向误差在 E11 独立冻结的容差内，且固定 slot→ray 映射的物理含义成立。

**FAIL 条件**

同一 slot 跨帧对应不同物理方向，azimuth phase 漂移，或必须逐帧重建 beam/column 才能得到物理 ray。

**状态转移**

- PASS → **E12**
- FAIL → **禁止直接用 slot 作为 ray；显式重建 beam/azimuth mapping，再重跑 E11。**


## E12｜多回波重排风险

**目的 / 唯一问题**

排除多回波导致 slot/ray 身份被动态重排。

**建模 / 实施**

检查原始格式与多回波字段；搜索同一 ray 的回波排序变化和重复射线。

**PASS 条件**

不存在影响第一回波身份的重排，或已显式提取第一回波并固定规则。

**FAIL 条件**

存在未处理的 reorder。

**状态转移**

- PASS → **E13**
- FAIL → **先处理 multi-return identity，再重跑 E12。**


## E13｜raw→ray→raw 点数往返

**目的 / 唯一问题**

验证规范射线网格不会增删原始有效回波。

**建模 / 实施**

跨大量帧执行 raw point→canonical ray grid→raw recovery，只比较有效点数和 slot/ray 对应。

**PASS 条件**

有效点计数完全恢复。

**FAIL 条件**

存在丢点、重复点或身份冲突。

**状态转移**

- PASS → **E14**
- FAIL → **修映射后重跑 E13。**


## E14｜raw→ray→raw 几何往返

**目的 / 唯一问题**

验证 range 与方向在规范化往返中不失真。

**建模 / 实施**

比较原始与恢复后的 range、单位方向和 xyz；容差在首次正式审计前冻结。

**PASS 条件**

最大/分位误差均在冻结容差内。

**FAIL 条件**

range/方向/坐标有系统偏差。

**状态转移**

- PASS → **E15**
- FAIL → **修 ray calibration/round-trip 后重跑 E14。**


## E15｜多序列射线资格确认

**目的 / 唯一问题**

确认 E08–E14 不是只在单条序列偶然成立。

**建模 / 实施**

在允许使用的多序列、多帧上重复核心 ray 审计并生成独立报告。

**PASS 条件**

跨序列稳定，建立固定 ρ_f(r) 或等价映射。

**FAIL 条件**

某些序列不满足统一规则。

**状态转移**

- PASS → **E16**
- FAIL → **回到对应失败的 E08–E14 修复后重新做 E15。**



# Phase 2｜程序化几何、正常控制与放置

## E16｜primitive 数值有限与有界

**目的 / 唯一问题**

保证异常代理基础几何不会产生 NaN/Inf 或无界形状。

**建模 / 实施**

大量采样 superquadric primitive，检查参数、implicit/SDF 值、bounding region。

**PASS 条件**

全部有限且有界；失败率在预先冻结的生成器容忍范围内。

**FAIL 条件**

存在数值爆炸、无界或异常高拒绝率。

**状态转移**

- PASS → **E17**
- FAIL → **修 primitive 参数化后重跑 E16。**


## E17｜单 primitive 射线求交

**目的 / 唯一问题**

验证最基础几何交点是可信的。

**建模 / 实施**

构造解析或高精度可验证的射线-primitive 场景，比较最近正交点距离。

**PASS 条件**

交点误差在冻结数值容差内，miss/hit 判定正确。

**FAIL 条件**

交点距离、法向或 hit/miss 错误。

**状态转移**

- PASS → **E18**
- FAIL → **修 intersection 后重跑 E17。**


## E18｜CSG 与连续形变稳定性

**目的 / 唯一问题**

验证 union/difference/intersection、bend/twist/taper/低频形变仍支持稳定求交。

**建模 / 实施**

对每种机制分别做定向样例和随机 stress test，记录 invalid hit、NaN、求交失败。

**PASS 条件**

所有启用机制在冻结失败率标准内。

**FAIL 条件**

某机制不稳定。

**状态转移**

- PASS → **E19**
- FAIL → **修复该机制；若无法稳定则在协议中禁用并重新冻结生成器，再重跑 E18。**


## E19｜单连通实体拒绝

**目的 / 唯一问题**

保证一个生成器实体不会偷偷变成多个互不相连物体。

**建模 / 实施**

故意构造多组件 CSG，验证连通性检查和 reject 路径。

**PASS 条件**

多组件被稳定拒绝，合法单组件不误拒。

**FAIL 条件**

多组件可进入正式世界。

**状态转移**

- PASS → **E20**
- FAIL → **修 connected-component validation 后重跑 E19。**


## E20｜形状/尺度/轴比/材质解耦

**目的 / 唯一问题**

防止异常标签被固定尺度、材质或复杂度组合泄漏。

**建模 / 实施**

统计采样器联合分布，检验 overall scale、axis ratio、complexity、material、placement 的独立/近独立采样设计。

**PASS 条件**

不存在标签确定性的固定组合；覆盖小/中/大、块状/扁平/细长/不对称。

**FAIL 条件**

某些几何机制与材质/尺度固定绑定。

**状态转移**

- PASS → **E21**
- FAIL → **修 sampling factorization 后重跑 E20。**


## E21｜局部支撑平面估计

**目的 / 唯一问题**

确保插入对象建立在可信的地面几何上。

**建模 / 实施**

在 road/parking/sidewalk 等正常地面区域估计 Π_g=(n_g,b_g)，对人工可验证区域检查法向、残差。

**PASS 条件**

法向与平面残差满足冻结标准。

**FAIL 条件**

支撑面估计不稳或方向错误。

**状态转移**

- PASS → **E22**
- FAIL → **修 plane fitting/候选区域后重跑 E21。**


## E22｜悬空与埋地检查

**目的 / 唯一问题**

保证对象与支撑面接触合理。

**建模 / 实施**

随机放置实体，测量最低点/接触带相对支撑面的距离与穿透比例。

**PASS 条件**

无明显悬空和大面积埋地。

**FAIL 条件**

存在系统悬空或穿透。

**状态转移**

- PASS → **E23**
- FAIL → **修 vertical alignment/contact rule 后重跑 E22。**


## E23｜已观测正常几何碰撞

**目的 / 唯一问题**

防止插入实体与可观测非地面正常表面明显穿插。

**建模 / 实施**

对候选实体与已观测非地面点/表面做碰撞近似，保存拒绝样例。

**PASS 条件**

超过容差的穿插被拒绝。

**FAIL 条件**

明显穿插仍可通过。

**状态转移**

- PASS → **E24**
- FAIL → **修 collision rule 后重跑 E23。**


## E24｜插入实体相互碰撞

**目的 / 唯一问题**

防止 normal-control/proxy 互相大面积穿插。

**建模 / 实施**

多实体世界中计算 pairwise overlap/距离，验证拒绝逻辑。

**PASS 条件**

明显互穿被拒绝。

**FAIL 条件**

互穿仍进入世界。

**状态转移**

- PASS → **E25**
- FAIL → **修 pairwise placement 后重跑 E24。**


## E25｜正常控制语义放置

**目的 / 唯一问题**

避免 normal-control 因不合理位置反而成为异常。

**建模 / 实施**

按类别检查 car/truck/person/bicycle 等模板的允许 surface 和姿态。

**PASS 条件**

放置符合方案规定的基本正常语义。

**FAIL 条件**

正常实体经常出现在明显不合理 surface/姿态。

**状态转移**

- PASS → **E26**
- FAIL → **修类别 placement constraints 后重跑 E25。**


## E26｜完整世界规格确定性

**目的 / 唯一问题**

落实“先确定完整反事实世界，再切窗口”。

**建模 / 实施**

固定 Ω={entities,geometry,position,material,seed}，重复序列级构造；比较 world spec hash 和实体参数。

**PASS 条件**

同一 world spec 可完全复现；窗口生成不重新采样实体。

**FAIL 条件**

窗口级随机重采样或 world spec 不可复现。

**状态转移**

- PASS → **E27**
- FAIL → **修 world-spec/seed 管理后重跑 E26。**



# Phase 3｜第一回波反事实渲染机械链

## E27｜normal-control 几何命中

**目的 / 唯一问题**

确认正常控制在当前 ray grid 和 placement 下真的能被射线击中。

**建模 / 实施**

暂时只观察几何交点，不让随机 return rejection 混入；统计 d_insert<∞。

**PASS 条件**

不同距离/尺度下均有合理几何命中。

**FAIL 条件**

控制对象几乎无几何命中。

**状态转移**

- PASS → **E28**
- FAIL → **回到 E21–E25 或模板几何修正，再重跑 E27。**


## E28｜anomaly-proxy 几何命中

**目的 / 唯一问题**

定位 proxy 无回波是否首先源于几何/放置。

**建模 / 实施**

与 E27 相同，只看 anomaly proxy 的几何 hit。

**PASS 条件**

不同形状/距离/尺度均产生合理 hit。

**FAIL 条件**

proxy 几何命中为零或极端稀少。

**状态转移**

- PASS → **E29**
- FAIL → **回到 E16–E24 定位几何/放置问题，再重跑 E28。**


## E29｜回波概率非退化

**目的 / 唯一问题**

确认 P(return|b,d,μ,ρ) 不会系统性把合法 inserted hit 全部拒绝。

**建模 / 实施**

对 E27/E28 的几何 hit 统计条件 return probability 分布，按 beam/range/incidence/material 分层。

**PASS 条件**

概率不是整体塌缩到 0/1，且与 206 校准支持一致。

**FAIL 条件**

某条件区间系统塌缩导致代理/控制无法返回。

**状态转移**

- PASS → **E30**
- FAIL → **修 return calibration/material modulation 后重跑 E29。**


## E30｜normal-control 有效回波

**目的 / 唯一问题**

验证正常控制通过完整 return model 后能产生真实可见点。

**建模 / 实施**

完整执行几何命中→return sampling→最近回波候选，统计有效 control returns。

**PASS 条件**

覆盖主要距离/遮挡条件且回波非退化。

**FAIL 条件**

control returns 为 0 或高度退化。

**状态转移**

- PASS → **E31**
- FAIL → **依据 E27/E29 结果修相应环节后重跑 E30。**


## E31｜anomaly-proxy 有效回波

**目的 / 唯一问题**

验证异常代理真正成为可监督 LiDAR 点。

**建模 / 实施**

完整执行与 normal-control 完全相同的回波流程，统计 proxy returns。

**PASS 条件**

proxy 在覆盖的尺度/距离条件下稳定产生非零有效回波。

**FAIL 条件**

proxy returns=0 或仅极少偶然回波。

**状态转移**

- PASS → **E32**
- FAIL → **若 E28 PASS 而 E31 FAIL，回 E29；若 E28 FAIL，回 E28。**


## E32｜插入物遮挡背景

**目的 / 唯一问题**

验证 inserted return 更近时替换原背景回波。

**建模 / 实施**

构造 sensor→inserted→background 的定向场景。

**PASS 条件**

inserted 有效回波胜出，原背景回波从新世界观测中消失。

**FAIL 条件**

背景仍保留或两者同时错误存在。

**状态转移**

- PASS → **E33**
- FAIL → **修 nearest-return competition 后重跑 E32。**


## E33｜正常前景遮挡插入物

**目的 / 唯一问题**

验证 original foreground 更近时 inserted object 被遮挡。

**建模 / 实施**

构造 sensor→normal foreground→inserted 的定向场景。

**PASS 条件**

前景保留，后方插入物不产生最终点。

**FAIL 条件**

后方 inserted 错误穿透前景。

**状态转移**

- PASS → **E34**
- FAIL → **修遮挡/距离竞争后重跑 E33。**


## E34｜空射线新增与拒绝

**目的 / 唯一问题**

验证空射线既能新增合法回波，也能在 return rejection 时保持空。

**建模 / 实施**

分别构造 empty+valid inserted return 与 empty+geometry hit but rejected 两种场景。

**PASS 条件**

前者新增点，后者仍为空。

**FAIL 条件**

空射线逻辑与预期不符。

**状态转移**

- PASS → **E35**
- FAIL → **修 empty-ray/return acceptance 后重跑 E34。**


## E35｜强度支持范围

**目的 / 唯一问题**

保证 synthetic intensity 落在 206 的实际支持内。

**建模 / 实施**

对 control/proxy 的有效回波统计强度 min/max/quantiles，核对裁剪和噪声。

**PASS 条件**

无越界、NaN，且不是大面积卡死在边界。

**FAIL 条件**

越界或严重边界饱和。

**状态转移**

- PASS → **E36**
- FAIL → **修 intensity sampling/clipping 后重跑 E35。**


## E36｜normal/proxy 共用渲染路径

**目的 / 唯一问题**

确认标签不会触发不同 ray、遮挡、return、intensity 或 slot recovery 分支。

**建模 / 实施**

对代码路径和运行 trace 做差分审计，除 geometry/label 外其余传感器流程必须相同。

**PASS 条件**

不存在 anomaly-only 传感器处理。

**FAIL 条件**

存在标签专用噪声、稀疏化、强度或回波分支。

**状态转移**

- PASS → **E37**
- FAIL → **合并到共享 renderer 后重跑 E36。**


## E37｜重叠窗口共享帧一致性

**目的 / 唯一问题**

验证同一 world/frame 被不同五帧窗口请求时逐槽完全一致。

**建模 / 实施**

随机至少多个 world，每个取多个重叠窗口，对共享帧做 hash/逐槽比较。

**PASS 条件**

同一 world/frame 结果完全一致。

**FAIL 条件**

共享帧随窗口变化。

**状态转移**

- PASS → **E38**
- FAIL → **修 world determinism/cache 后回 E26/E07，再重跑 E37。**



# Phase 4｜Gate 1：传感器一致性与反作弊

## E38｜per-beam 回波率一致性

**目的 / 唯一问题**

检验反事实渲染在 beam 条件下是否接近 206 的基本传感器统计。

**建模 / 实施**

分别统计 real normal、rendered normal-control、proxy 的 per-beam return rate；PASS 容差需首次正式运行前冻结。

**PASS 条件**

normal-control 与真实正常在冻结标准内；proxy 无传感器退化。

**FAIL 条件**

存在系统 beam 指纹。

**状态转移**

- PASS → **E39**
- FAIL → **修 return calibration 后重跑 E38。**


## E39｜per-range 回波率一致性

**目的 / 唯一问题**

检验距离条件下的回波率是否存在来源指纹。

**建模 / 实施**

按固定 range bins 比较三类来源。

**PASS 条件**

normal-control 与真实正常满足冻结标准；proxy 具合理覆盖。

**FAIL 条件**

某些 range bins 系统偏离。

**状态转移**

- PASS → **E40**
- FAIL → **修 range-conditioned return model 后重跑 E39。**


## E40｜beam×range 强度分布

**目的 / 唯一问题**

替代“只看均值”的不足，验证完整强度统计。

**建模 / 实施**

比较 median、quantiles、spread/ECDF 等条件分布，不仅报告 mean。

**PASS 条件**

rendered normal-control 与 real normal 在冻结标准内，proxy 也落在合理支持。

**FAIL 条件**

强度分布可明显暴露来源。

**状态转移**

- PASS → **E41**
- FAIL → **修 intensity calibration/material modulation 后重跑 E40。**


## E41｜empty→valid 比例

**目的 / 唯一问题**

确认插入导致的新回波比例不过度异常。

**建模 / 实施**

统计原空 ray 因 control/proxy 变有效的比例，按距离/beam 分层。

**PASS 条件**

比例在预先冻结的合理范围且 control/proxy 可比较。

**FAIL 条件**

新增回波率极端或标签强相关。

**状态转移**

- PASS → **E42**
- FAIL → **修 placement/return 规则后重跑 E41。**


## E42｜单实体可见点数分布

**目的 / 唯一问题**

防止标签被 N_vis 单独推断。

**建模 / 实施**

比较 control/proxy 的每帧可见点数，并与真实正常实例规模作参考。

**PASS 条件**

两类覆盖重叠且不形成标签决定性捷径。

**FAIL 条件**

proxy/control 可见点数完全分离。

**状态转移**

- PASS → **E43**
- FAIL → **修尺度/距离/遮挡匹配后重跑 E42。**


## E43｜连续帧可见点数变化

**目的 / 唯一问题**

验证静止插入实体的时序观测变化符合 LiDAR 几何而非随机闪烁。

**建模 / 实施**

对固定实体计算 consecutive-frame N_vis variation。

**PASS 条件**

变化率稳定、无无因跳变。

**FAIL 条件**

频繁随机出现/消失。

**状态转移**

- PASS → **E44**
- FAIL → **修 deterministic return/world rendering 后重跑 E43。**


## E44｜遮挡率分布

**目的 / 唯一问题**

避免 proxy 和 normal-control 的遮挡程度成为标签捷径。

**建模 / 实施**

比较两类 inserted entities 的 occlusion ratio 分布。

**PASS 条件**

分布具有充分重叠并满足冻结匹配标准。

**FAIL 条件**

遮挡分布高度可分。

**状态转移**

- PASS → **E45**
- FAIL → **修 placement/matching 后重跑 E44。**


## E45｜三方严格匹配审计集

**目的 / 唯一问题**

在来源分类前先消除 distance、beam、surface、N_vis、occlusion 等显式混杂。

**建模 / 实施**

构造 real normal / rendered normal-control / rendered anomaly-proxy 的匹配样本；匹配规则在查看 E46 结果前冻结。

**PASS 条件**

三方样本满足预定匹配平衡标准。

**FAIL 条件**

匹配后仍有明显协变量失衡。

**状态转移**

- PASS → **E46**
- FAIL → **改进匹配策略后重跑 E45。**


## E46｜真实正常 vs 渲染正常来源分类

**目的 / 唯一问题**

检验 renderer 是否留下低层来源指纹。

**建模 / 实施**

训练低容量分类器，输入仅 x,y,z,intensity,beam,range,local density；按 frame/sequence 分组划分，避免泄漏。

**PASS 条件**

分类器不再能“轻易接近饱和”区分来源；具体接受区间必须在本次正式运行前冻结。

**FAIL 条件**

仍表现出强来源可分性。

**状态转移**

- PASS → **E48**
- FAIL → **进入 E47 做指纹归因；不得训练 AJAE。**


## E47｜来源指纹归因消融

**目的 / 唯一问题**

在 E46 FAIL 时定位究竟是哪一类低层特征泄漏。

**建模 / 实施**

分别运行 coordinate-only、intensity-only、beam/range-only、density-only 等低容量分类器。

**PASS 条件**

定位主要泄漏来源并形成单一修复目标。

**FAIL 条件**

无法定位或多源同时泄漏。

**状态转移**

- PASS → **E46**
- FAIL → **修对应的 E38–E44 环节后重新 E45→E46。**


## E48｜normal-control vs anomaly-proxy 难度分类

**目的 / 唯一问题**

判断 proxy 是否夸张到低容量单帧模型即可接近饱和。

**建模 / 实施**

在固定 201 synthetic worlds 上训练低容量单帧分类器，使用与 Gate1 规定一致的低层输入。

**PASS 条件**

proxy 具有可学几何差异，但不被极低级统计轻易近饱和分开；判据需预先冻结。

**FAIL 条件**

任务过易，时空模型贡献将不可识别。

**状态转移**

- PASS → **E49**
- FAIL → **增加 near-normal-boundary/hard proxy 并回到 E20、E42–E48 重新验证。**


## E49｜Gate 1 正式裁决

**目的 / 唯一问题**

只在 E08–E48 通过后判断 renderer 是否有资格生成 AJAE 训练监督。

**建模 / 实施**

汇总 ray、机械正确性、sensor distribution、source leakage、proxy difficulty 的冻结产物。

**PASS 条件**

全部前置实验 PASS。

**FAIL 条件**

任一关键前置未过。

**状态转移**

- PASS → **E50**
- FAIL → **回到最早 FAIL 节点；B0/B1 及之后实验保持锁定。**



# Phase 5｜冻结 STU 点接口与五帧坐标

## E50｜128D STU 高层特征接口

**目的 / 唯一问题**

确认实际使用 all_features[-1]→point_features_head 的 128D 特征。

**建模 / 实施**

在真实帧 hook/记录官方路径与 shape。

**PASS 条件**

真实输出维度 128，来源层与主线一致。

**FAIL 条件**

取错层、维度错误或仅 toy tensor。

**状态转移**

- PASS → **E51**
- FAIL → **修 STU adapter 后重跑 E50。**


## E51｜稀疏体素→原始点逆映射

**目的 / 唯一问题**

确认 π_f(p) 能覆盖每个原始有效回波点。

**建模 / 实施**

在真实帧核对 inverse map 完整率、索引范围、重复体素情况。

**PASS 条件**

所有有效 raw points 都有合法体素映射。

**FAIL 条件**

存在 unmapped/错位点。

**状态转移**

- PASS → **E52**
- FAIL → **修 inverse map 后重跑 E51。**


## E52｜共享体素下的原始点身份

**目的 / 唯一问题**

验证多个 raw points 共享 STU voxel feature 时仍保留独立坐标、intensity、ray identity 和最终预测位置。

**建模 / 实施**

构造/查找真实 shared-voxel case 做逐点检查。

**PASS 条件**

共享 feature 不导致 raw point identity 合并。

**FAIL 条件**

原始点身份丢失。

**状态转移**

- PASS → **E53**
- FAIL → **修 raw-point interface 后重跑 E52。**


## E53｜官方 query assignment

**目的 / 唯一问题**

确认语义证据采用 softmax query class、sigmoid mask 和 q*(v) 最强官方分配逻辑。

**建模 / 实施**

用真实 STU query/mask 输出复算 q*，检查 19 normal classes 和确定性 tie-break。

**PASS 条件**

实现与公式一致，tie 可复现。

**FAIL 条件**

仍使用所有 query 未控制求和或分配不一致。

**状态转移**

- PASS → **E54**
- FAIL → **修 assigned_stu_evidence 后重跑 E53。**


## E54｜19D 语义证据与可靠性

**目的 / 唯一问题**

确认 e_normal、r_assign、r_noobj 的维度与数值定义正确。

**建模 / 实施**

逐体素/逐 raw point 对照手算或独立实现。

**PASS 条件**

19D evidence、assignment reliability、no-object probability 全部一致。

**FAIL 条件**

任一量定义/广播错误。

**状态转移**

- PASS → **E55**
- FAIL → **修证据计算后重跑 E54。**


## E55｜AJAE 真实输入张量

**目的 / 唯一问题**

确认每个 raw point 实际接收 128+19+1+1+1，再加中心坐标和 q 编码。

**建模 / 实施**

记录真实五帧输入 tensor schema、shape、字段统计，检查没有 query token/entropy/energy/MSP。

**PASS 条件**

字段与主线完全一致。

**FAIL 条件**

多/少输入字段或顺序/维度错误。

**状态转移**

- PASS → **E56**
- FAIL → **修 input adapter 后重跑 E55。**


## E56｜中心坐标对齐

**目的 / 唯一问题**

确认 x^(t)=T_{S_t←W}T_{W←S_k}x 的方向正确，且 moving object 不被实例级对齐掉。

**建模 / 实施**

先做方向可判别的 synthetic rigid transform，再在真实静态背景检查跨帧重合误差并观察 moving object 仍保留位移。

**PASS 条件**

中心帧近似恒等、静态背景对齐、运动目标保留相对运动。

**FAIL 条件**

矩阵方向错、静态背景不重合或运动被错误抹平。

**状态转移**

- PASS → **E57**
- FAIL → **修 pose transform 后重跑 E56。**



# Phase 6｜固定 201 开发试验台与评价器

## E57｜24 条 in-generator 开发世界冻结

**目的 / 唯一问题**

建立唯一用于模型/超参选择的 201 synthetic development worlds。

**建模 / 实施**

生成 world spec manifest、hash、seed、背景序列和实体规格；24 条都同时含 normal-control 与 proxy。

**PASS 条件**

24 条身份固定、可复现、进入版本控制。

**FAIL 条件**

数量、内容或身份仍会变化。

**状态转移**

- PASS → **E58**
- FAIL → **重新定义并冻结后重跑 E57。**


## E58｜6 条 held-out 诊断世界冻结

**目的 / 唯一问题**

建立只做机制诊断、绝不参与选择的 held-out worlds。

**建模 / 实施**

固定 6 条使用训练时完全未见的程序化机制，并在代码层阻止其进入 selection。

**PASS 条件**

held-out mechanism 与训练 generator 隔离，使用规则写入协议。

**FAIL 条件**

训练阶段可能采样到 held-out mechanism 或结果可参与选择。

**状态转移**

- PASS → **E59**
- FAIL → **修 generator split/selection guard 后重跑 E58。**


## E59｜开发世界 N_vis / O / d 覆盖

**目的 / 唯一问题**

保证 24 条世界覆盖可见点数、遮挡和距离难度，而不是单一简单条件。

**建模 / 实施**

按预先冻结 bins 统计三维覆盖；不足则只重新定义 201 synthetic worlds，不训练模型。

**PASS 条件**

各难度层达到冻结的最小覆盖标准。

**FAIL 条件**

某些关键层级缺失。

**状态转移**

- PASS → **E60**
- FAIL → **补充/替换固定世界后重跑 E59。**


## E60｜开发世界 V=1..5 覆盖

**目的 / 唯一问题**

保证后续能检验多帧可见证据与性能关系。

**建模 / 实施**

统计每个 entity 的 V；分别核对 V=1,2,3,4,5。

**PASS 条件**

五个层级均达到预先冻结的最小样本标准。

**FAIL 条件**

任一层级缺失或近乎空。

**状态转移**

- PASS → **E61**
- FAIL → **重新放置/选择固定世界后重跑 E60。**


## E61｜pure-normal 与 moving-normal 开发子集

**目的 / 唯一问题**

建立正常泛化与运动安全检查的固定数据。

**建模 / 实施**

冻结 pure-normal 201 和 moving car/person/bicycle 等 diagnostic subset；这些标签不进入模型输入。

**PASS 条件**

子集身份、定义和 hash 固定。

**FAIL 条件**

子集定义可变或 moving label 泄漏到训练输入。

**状态转移**

- PASS → **E62**
- FAIL → **修 subset/loader 后重跑 E61。**


## E62｜自研 evaluator 与官方 evaluator 一致性

**目的 / 唯一问题**

确保 AP/AUROC/FPR95、2.5–50m、ignore、每帧异常点<5规则一致。

**建模 / 实施**

对同一固定 prediction 文件同时跑自研和 STU 官方 evaluator，比较逐帧/pooled 结果。

**PASS 条件**

指标在冻结数值容差内一致。

**FAIL 条件**

定义、过滤或累计逻辑不一致。

**状态转移**

- PASS → **E63**
- FAIL → **修 evaluator 后重跑 E62。**


## E63｜开发决策规则冻结

**目的 / 唯一问题**

防止看到 B1/B3 结果后再改 checkpoint、PASS 阈值或安全判据。

**建模 / 实施**

在首次正式 B0/B1 前冻结：checkpoint selection、B1>B0 判定、normal safety 容忍、三 seed 稳定性规则；具体数值若主线未给出必须现在预注册。

**PASS 条件**

规则已写入协议并 hash 固定。

**FAIL 条件**

任一关键判据仍是 null/事后决定。

**状态转移**

- PASS → **E64**
- FAIL → **补齐预注册判据后重跑 E63。**



# Phase 7｜AJAE 模型机械单元资格

## E64｜时间身份体素隔离

**目的 / 唯一问题**

验证 voxel key 含 q，不同时间位置不会在 pooling 时直接合并。

**建模 / 实施**

构造相同 xyz、不同 q 的点并逐层检查 voxel identity。

**PASS 条件**

L1/L2/L3 均保持独立 temporal identity。

**FAIL 条件**

不同 q 被合并。

**状态转移**

- PASS → **E65**
- FAIL → **修 voxel key 后重跑 E64。**


## E65｜mean-max 池化数值

**目的 / 唯一问题**

验证每个体素确实同时使用 mean 与 max 并学习融合。

**建模 / 实施**

用可手算小样本核对 mean、max、concat、linear 输入。

**PASS 条件**

数值与定义一致。

**FAIL 条件**

退化为 mean-only/max-only 或聚合错误。

**状态转移**

- PASS → **E66**
- FAIL → **修 VoxelPool 后重跑 E65。**


## E66｜按时间差分层邻域

**目的 / 唯一问题**

验证 δ∈{-2,-1,0,+1,+2} 各自独立 radius/K，不共同竞争 global K。

**建模 / 实施**

构造同帧邻居很多、跨帧邻居稀少的反例；逐 δ 检查候选、radius cutoff、K 上限。

**PASS 条件**

无跨 δ 抢占、无远点补 K、空候选允许为空。

**FAIL 条件**

任一分支被其他时间点挤占或补远点。

**状态转移**

- PASS → **E67**
- FAIL → **修 neighborhood builder 后重跑 E66。**


## E67｜空跨帧分支与 gate

**目的 / 唯一问题**

验证空 temporal neighborhood 时 m=0 且 g=0。

**建模 / 实施**

构造无跨帧邻居样例，直接读取 message/gate/output。

**PASS 条件**

message=0、gate=0、输出 finite。

**FAIL 条件**

空分支仍产生非零跨帧贡献。

**状态转移**

- PASS → **E68**
- FAIL → **修 gate/empty handling 后重跑 E67。**


## E68｜同帧残差生存路径

**目的 / 唯一问题**

保证即使所有跨帧 gate 关闭，模型仍保留 same-frame evidence。

**建模 / 实施**

将所有 cross-frame branch 人工置空，比较 block 输出是否包含 δ=0 分支和 residual。

**PASS 条件**

同帧路径始终存在且不受跨帧 gate 抑制。

**FAIL 条件**

同帧信息被 gate 一起关闭。

**状态转移**

- PASS → **E69**
- FAIL → **修 residual block 后重跑 E68。**


## E69｜同帧 3-NN 上采样

**目的 / 唯一问题**

验证 decoder 只在同一 q 搜索三个父节点。

**建模 / 实施**

构造其他 q 的 coarse node 更近的反例，检查选中的 parent IDs 和 inverse-distance 权重。

**PASS 条件**

只选 same-frame parent，权重归一化且 finite。

**FAIL 条件**

跨 q parent 被选中或权重异常。

**状态转移**

- PASS → **E70**
- FAIL → **修 decoder 后重跑 E69。**


## E70｜平衡 BCE 空类别安全

**目的 / 唯一问题**

确认 zero-positive 与 zero-negative 窗口都能合法训练。

**建模 / 实施**

分别构造纯负、纯正、正负都有的 logits/labels，核对 balanced BCE 手算结果。

**PASS 条件**

三种情况 finite，正负都有时各占 1/2；<5 anomaly 不影响训练窗口。

**FAIL 条件**

NaN、权重错误或误用官方<5评价规则。

**状态转移**

- PASS → **E71**
- FAIL → **修 loss 后重跑 E70。**


## E71｜概率融合公式单元测试

**目的 / 唯一问题**

冻结 B4 使用 mean(sigmoid(logit)) 而不是 sigmoid(mean(logit))。

**建模 / 实施**

给定固定 logits 手工计算两种公式并与实现比较。

**PASS 条件**

实现严格等于概率等权平均，且无 q/center 权重。

**FAIL 条件**

平均了 logits 或存在隐含权重。

**状态转移**

- PASS → **E72**
- FAIL → **修 fusion 后重跑 E71。**



# Phase 8｜Gate 2：异常代理监督是否有效

## E72｜B0 冻结 STU 单帧参考

**目的 / 唯一问题**

建立不训练 AJAE 时的官方单帧基线。

**建模 / 实施**

在固定 201 开发域生成 B0 prediction，并由官方 evaluator 计算 AP/AUROC/FPR95；保存逐世界结果。

**PASS 条件**

产物可追溯、官方复算完成。

**FAIL 条件**

预测/评价链不完整。

**状态转移**

- PASS → **E73**
- FAIL → **修 B0/evaluator 后重跑 E72。**


## E73｜B1 单帧 smoke train

**目的 / 唯一问题**

在正式三 seed 前只验证单帧代理训练数值链可工作。

**建模 / 实施**

用很小且预先限定的 206 世界预算运行 B1；检查 loss、grad、标签计数、STU 冻结、无 NaN。

**PASS 条件**

训练数值稳定且只更新 AJAE 新参数。

**FAIL 条件**

训练崩溃、标签异常、STU 被更新。

**状态转移**

- PASS → **E74**
- FAIL → **修对应训练机械问题后重跑 E73。**


## E74｜B1 三独立训练种子

**目的 / 唯一问题**

获得代理监督单帧模型的正式开发结果。

**建模 / 实施**

按完全相同超参/selection rule、独立初始化与206世界流训练3 seed；使用同一固定201。

**PASS 条件**

三 seed 均完成、产物齐全、无协议偏差。

**FAIL 条件**

某 seed 异常终止或配置不一致。

**状态转移**

- PASS → **E75**
- FAIL → **修机械问题后仅重跑无效 seed；若协议变更则全部重跑 E74。**


## E75｜B1 vs B0 代理监督效应

**目的 / 唯一问题**

回答“代理监督本身是否在新背景有效”。

**建模 / 实施**

在24 fixed worlds 上做逐世界与三 seed 比较，主指标按 E63 预注册规则判断 B1>B0。

**PASS 条件**

满足预注册的 B1 优于 B0 条件。

**FAIL 条件**

不优于 B0 或提升不稳定。

**状态转移**

- PASS → **E76**
- FAIL → **若 FAIL，进入 proxy/normal-control 重新设计，随后必须从 E38–E75 重新资格化；不得做 B2/B3。**


## E76｜B1 正常安全

**目的 / 唯一问题**

确认 B1 的提升不是靠提高正常点整体分数。

**建模 / 实施**

在 pure-normal 201、rendered normal-control、moving-normal subset 上比较 B0/B1 的平均分与误报。

**PASS 条件**

不超过 E63 冻结的安全恶化界限。

**FAIL 条件**

正常误报明显恶化。

**状态转移**

- PASS → **E77**
- FAIL → **若 FAIL，调整 proxy/control/容量并回 E38；不得进入五帧。**


## E77｜Gate 2 正式裁决

**目的 / 唯一问题**

只在 E72–E76 都满足时宣称“异常代理监督有效”。

**建模 / 实施**

汇总三 seed、逐世界和正常安全结果。

**PASS 条件**

B1>B0 且 pure-normal/control/moving-normal 安全通过。

**FAIL 条件**

任一条件不满足。

**状态转移**

- PASS → **E78**
- FAIL → **回到最早失败节点；五帧实验继续锁定。**



# Phase 9｜Gate 3：跨帧信息是否提供可识别增益

## E78｜B2 无跨帧五帧对照

**目的 / 唯一问题**

隔离参数量、多尺度、共享结构带来的提升。

**建模 / 实施**

使用与完整模型相同的五帧结构，屏蔽所有 δ≠0 边，五帧仍监督，评价只取 q=0；按3 seed正式训练。

**PASS 条件**

三 seed 完成且 cross-frame contribution 运行 trace 为零。

**FAIL 条件**

B2 退化成 B1、跨帧边未完全关闭或产物不完整。

**状态转移**

- PASS → **E79**
- FAIL → **修 B2 条件后重跑 E78。**


## E79｜B3 五帧 smoke train

**目的 / 唯一问题**

在大规模三 seed 前验证 centered five-frame 完整模型数值稳定。

**建模 / 实施**

小预算运行 cross-frame attention+gates，检查每个 δ 实际有邻居/空分支、gate 分布和显存。

**PASS 条件**

训练 finite，五帧全部监督，center 只作坐标规范。

**FAIL 条件**

数值崩溃、只监督中心帧或时间分支未被使用。

**状态转移**

- PASS → **E80**
- FAIL → **修模型机械问题并回 E64–E79 对应节点。**


## E80｜B3 三独立训练种子

**目的 / 唯一问题**

获得正式 centered five-frame 模型。

**建模 / 实施**

与 B1/B2 相同 201、selection rule；3 独立初始化/206世界流。

**PASS 条件**

三 seed 均完整可追溯。

**FAIL 条件**

配置不一致、某 seed 无效。

**状态转移**

- PASS → **E81**
- FAIL → **修后按协议重跑受影响 seed；若规则变化则全部重跑 E80。**


## E81｜B3 vs B1

**目的 / 唯一问题**

回答五帧模型是否优于单帧代理模型。

**建模 / 实施**

按预注册规则做三 seed + 24 worlds 配对比较。

**PASS 条件**

稳定满足 B3>B1。

**FAIL 条件**

B3≤B1 或不稳定。

**状态转移**

- PASS → **E82**
- FAIL → **Gate3 temporal claim 失败；定位 temporal design 后若修改模型，回 E64 并重跑 E78–E81。**


## E82｜B3 vs B2

**目的 / 唯一问题**

排除“只是模型更大/多尺度更强”的解释。

**建模 / 实施**

在完全相同开发域比较 B3 与 B2。

**PASS 条件**

稳定满足 B3>B2。

**FAIL 条件**

B3≤B2。

**状态转移**

- PASS → **E83**
- FAIL → **cross-frame 主张失败；若修改 temporal mechanism，回 E64 并重跑 E78–E82。**


## E83｜五帧正常运动安全

**目的 / 唯一问题**

确认 temporal improvement 不以 moving normal 误报为代价。

**建模 / 实施**

比较 B1/B3 在 moving-normal 平均分、FPR，以及 static-normal 差异。

**PASS 条件**

不超过 E63 冻结的安全界限。

**FAIL 条件**

moving normal 明显恶化。

**状态转移**

- PASS → **E84**
- FAIL → **只允许诊断/修改 temporal neighborhood/gate；修改后回 E64、E78–E83。**


## E84｜Gate 3 正式裁决

**目的 / 唯一问题**

决定是否支持 AJAE 的核心跨帧主张。

**建模 / 实施**

汇总 B1/B2/B3 三 seed、逐世界和安全性。

**PASS 条件**

B3>B1、B3>B2 且 moving-normal safety PASS。

**FAIL 条件**

任一条件失败。

**状态转移**

- PASS → **E85**
- FAIL → **若 FAIL，不进入融合贡献主张；回到对应失败节点。**



# Phase 10｜时间位置校准与 B4 融合

## E85｜q 位置分数诊断

**目的 / 唯一问题**

检查 q=-2,-1,0,+1,+2 的打分尺度是否足以支持等权融合。

**建模 / 实施**

在固定201上分别报告每个 q 的 AP、正常均值、异常均值、分数分布；不使用 19。

**PASS 条件**

不同 q 不存在违反 E63 预注册标准的系统尺度偏置。

**FAIL 条件**

位置偏置明显。

**状态转移**

- PASS → **E86**
- FAIL → **只在201上拟合/修正位置校准规则，冻结后重跑 E85。**


## E86｜真实重叠点身份与 m_p 覆盖

**目的 / 唯一问题**

确认同一 p=(f,r) 能跨窗口正确聚合 1–5 个预测。

**建模 / 实施**

在真实 centered windows 上核对 PointId、q(w)、m_p 分布和序列边缘。

**PASS 条件**

身份一致，1≤m_p≤5，无 padding/镜像/重复帧。

**FAIL 条件**

点身份错配或 m_p 非法。

**状态转移**

- PASS → **E87**
- FAIL → **修 point identity/window traversal 后回 E37/E71 并重跑 E86。**


## E87｜B4 融合评估

**目的 / 唯一问题**

只测“同一 B3 checkpoint + 多窗口概率平均”的额外价值。

**建模 / 实施**

禁止重新训练 B4；复用 B3 checkpoint，执行 E71 冻结公式。

**PASS 条件**

B4 结果完整、可与 B3 一一配对。

**FAIL 条件**

B4 使用不同 checkpoint、不同模型或融合实现不一致。

**状态转移**

- PASS → **E88**
- FAIL → **修融合路径后重跑 E87。**


## E88｜B4 vs B3

**目的 / 唯一问题**

判断 overlapping-window consensus 是否构成独立贡献。

**建模 / 实施**

按预注册规则比较 B4 与 B3 的逐世界/三 seed结果。

**PASS 条件**

B4>B3：支持融合贡献。

**FAIL 条件**

B4≤B3：不支持融合贡献，但不否定 B3 temporal claim。

**状态转移**

- PASS → **E89**
- FAIL → **PASS→E89；FAIL→记录“fusion unsupported”，最终模型可保留 B3，仍进入 E89。**



# Phase 11｜机制、安全、对象尺度与因果消融

## E89｜实体内部得分方差

**目的 / 唯一问题**

检验多帧/多尺度是否减少同一异常实体内部碎片化。

**建模 / 实施**

用 generator ID 仅作诊断，比较 B1/B3/(B4) 的 Var_{p∈O_m}(S_p)，按 N_vis 分层。

**PASS 条件**

若 B3/B4 更低，可支持“内部一致性改善”机制；否则只记录无该证据。

**FAIL 条件**

结果无改善不构成方法整体失败。

**状态转移**

- PASS → **E90**
- FAIL → **无论结果进入 E90；禁止把 object ID 送入模型。**


## E90｜异常边界泄漏

**目的 / 唯一问题**

检验多尺度聚合是否把异常高分扩散到道路或邻近正常对象。

**建模 / 实施**

分别测异常表面、邻近 road、邻近正常对象分数，比较 B1/B3/B4。

**PASS 条件**

背景泄漏不超过冻结安全标准；否则机制安全失败。

**FAIL 条件**

明显扩散到正常背景。

**状态转移**

- PASS → **E91**
- FAIL → **若 FAIL，只能调整多尺度/邻域并回 E64、E78 起重新开发；若 PASS→E91。**


## E91｜V=1..5 可见性趋势

**目的 / 唯一问题**

检验“多帧可见证据增加→性能改善”的机制解释。

**建模 / 实施**

在 E60 冻结的五个 V 层级分别报告性能并检验趋势。

**PASS 条件**

存在与预注册机制判据一致的改善趋势：可写时空共识解释。

**FAIL 条件**

无趋势：不得写“对象尺度时空共识”，但不自动否定主性能。

**状态转移**

- PASS → **E92**
- FAIL → **无论结果进入 E92。**


## E92｜B5 因果窗口正确性

**目的 / 唯一问题**

确认在线消融严格只使用 [t-4,t] 且只输出当前帧。

**建模 / 实施**

检查实际 frame IDs、q 编码、无 future-frame access、当前帧输出。

**PASS 条件**

无未来帧泄漏，配置独立于 B3。

**FAIL 条件**

访问未来帧或与 centered 模型混淆。

**状态转移**

- PASS → **E93**
- FAIL → **修 causal loader 后重跑 E92。**


## E93｜B5 因果性能

**目的 / 唯一问题**

量化去掉未来帧后的性能代价。

**建模 / 实施**

按固定201和同一 evaluator 训练/评价 B5，并与 B3/B4 比较。

**PASS 条件**

产物完整即可；本实验不设必须优于谁。

**FAIL 条件**

运行链不完整。

**状态转移**

- PASS → **E94**
- FAIL → **修 B5 机械问题后重跑 E93。**


## E94｜计算成本与输入公平性

**目的 / 唯一问题**

为论文报告 B1/B3/B4/B5 的真实代价并明确额外时间输入。

**建模 / 实施**

固定 GPU、batch、cache 条件，测 latency、VRAM、throughput、STU cache 命中；注明 B3/B4 使用未来帧。

**PASS 条件**

四条件均有可复现成本报告且比较口径明确。

**FAIL 条件**

缺条件、设备/批次不一致或隐瞒未来帧输入。

**状态转移**

- PASS → **E95**
- FAIL → **补齐成本测量后重跑 E94。**



# Phase 12｜方法冻结

## E95｜最终模型选择规则执行

**目的 / 唯一问题**

在不访问 19 的前提下，按预注册规则选择最终 B3 或 B4 以及 checkpoint。

**建模 / 实施**

只使用 201 development worlds；held-out 6 条不参与选择。

**PASS 条件**

选择结果由 E63 规则唯一决定。

**FAIL 条件**

需要人工看结果临时决定或使用 held-out/19 信息。

**状态转移**

- PASS → **E96**
- FAIL → **修选择规则并回 E63；必要时重新相关开发实验。**


## E96｜阈值与 DBSCAN 冻结

**目的 / 唯一问题**

只在 201 上固定点阈值 τ 与逐帧 3D DBSCAN 参数。

**建模 / 实施**

使用固定 selection procedure，不看 19。

**PASS 条件**

τ/DBSCAN 唯一确定并写入 freeze manifest。

**FAIL 条件**

参数仍待 19 结果后调整。

**状态转移**

- PASS → **E97**
- FAIL → **重新在 201 完成开发并重跑 E96。**


## E97｜AJAE Method Freeze Manifest v1

**目的 / 唯一问题**

在第一次访问 19 前冻结所有会影响结果的内容。

**建模 / 实施**

记录 generator、normal-control、renderer、STU interface、architecture、loss、hyperparams、checkpoint rule、final checkpoint、fusion/q-calibration、τ、DBSCAN、evaluator、代码/配置 hash。

**PASS 条件**

manifest 完整、可机器验证、只读保存。

**FAIL 条件**

存在未冻结字段。

**状态转移**

- PASS → **E98**
- FAIL → **补齐字段后重跑 E97；此时仍禁止访问 19。**


## E98｜冻结完整性演练

**目的 / 唯一问题**

确认 freeze manifest 能阻止之后的模型/协议变化和 19 标签旁路。

**建模 / 实施**

模拟更改关键文件/参数和尝试访问 public19，验证 guard 拒绝。

**PASS 条件**

任何影响结果的变化都会使 manifest invalid；19 只有 manifest valid 时可解锁。

**FAIL 条件**

guard 可绕过或 manifest 不敏感。

**状态转移**

- PASS → **E99**
- FAIL → **修 freeze guard 后重跑 E98。**



# Phase 13｜一次性真实 OOD 确认与最终测试

## E99｜19 条真实 OOD 锁定推理

**目的 / 唯一问题**

第一次也是当前研究周期唯一一次打开公开真实异常确认集。

**建模 / 实施**

在 E97/E98 完全冻结后，一次性对全部 19 条序列生成预测；不得先看部分序列再停下调方法。

**PASS 条件**

19 条推理完整、checkpoint/config/hash 与 manifest 一致。

**FAIL 条件**

推理期间发生方法变化、只跑部分后调参或 lineage 不完整。

**状态转移**

- PASS → **E100**
- FAIL → **若协议被破坏则确认集完整性失效；不得把后续结果称 untouched confirmation。**


## E100｜真实 OOD 官方点级指标

**目的 / 唯一问题**

检验 proxy supervision 和 temporal model 是否迁移到真实 STU OOD。

**建模 / 实施**

由官方 evaluator 计算 AP/AUROC/FPR95，并逐序列报告 B0/B1/final AJAE。

**PASS 条件**

结果完整；是否支持迁移由 E103 统一裁决。

**FAIL 条件**

评价链错误或预测无法官方读取。

**状态转移**

- PASS → **E101**
- FAIL → **只允许修 evaluator/I-O bug；若修复会改变模型/方法，则确认周期失效。**


## E101｜真实 OOD 对象级指标

**目的 / 唯一问题**

评估融合分数经冻结 τ 和逐帧 DBSCAN 后的对象检测质量。

**建模 / 实施**

报告 RecallQ/SQ/RQ/UQ/PQ/TP/FP/FN；不跨帧 tracking。

**PASS 条件**

使用冻结参数、结果可复现。

**FAIL 条件**

阈值/DBSCAN 被事后修改或跨帧 tracking 介入。

**状态转移**

- PASS → **E102**
- FAIL → **若只是 evaluator bug 可修后复算；若需改方法则确认周期结束。**


## E102｜真实正常运动安全

**目的 / 唯一问题**

确认真实确认阶段没有以正常运动误报换取 OOD 提升。

**建模 / 实施**

按冻结 moving-normal 定义报告 final AJAE 与 B0/B1 的安全差异。

**PASS 条件**

满足冻结安全界限。

**FAIL 条件**

明显恶化。

**状态转移**

- PASS → **E103**
- FAIL → **无论结果进入 E103 统一判定。**


## E103｜Gate 4 迁移裁决

**目的 / 唯一问题**

最终回答“合成异常代理是否迁移到真实 OOD”。

**建模 / 实施**

结合 E100–E102，按冻结规则判断相对 B0/B1 的改善与 normal-motion safety。

**PASS 条件**

改善成立且安全通过：proxy→real OOD transfer supported。

**FAIL 条件**

未改善或安全失败：当前迁移假设失败，本研究周期停止。

**状态转移**

- PASS → **E104**
- FAIL → **PASS→E104；FAIL→停止，禁止继续用同一 19 条调方法。**


## E104｜51 条隐藏测试最终提交

**目的 / 唯一问题**

只在 Gate 4 PASS 后执行最终官方测试。

**建模 / 实施**

使用 E97 冻结的完全同一方法与 checkpoint 提交隐藏测试。

**PASS 条件**

完成官方提交并保存最终结果/提交信息。

**FAIL 条件**

任何想根据 hidden 结果再调方法的行为。

**状态转移**

- PASS → **AJAE 完成**
- FAIL → **若需新研究周期，必须重新定义新的独立确认纪律。**



# 3. 四个 Decision Gate 的最终形式

## Gate 1：renderer 是否有资格生成训练监督

只有 **E08–E49** 全部按各自冻结规则通过，才允许把 renderer/generator 视为训练数据生成机制。失败时必须回到最早失败节点；任何 B0/B1/B3 的高分都不能反向证明 Gate 1 合格。

## Gate 2：异常代理监督是否有效

必须由 **B1 相对 B0 的固定 201 开发结果 + pure-normal / normal-control / moving-normal 安全**共同判定。失败说明 proxy supervision 本身不足，不能进入五帧主张。

## Gate 3：跨帧信息是否提供可识别增益

必须同时满足：

$$
B3>B1
$$

以及

$$
B3>B2
$$

并通过正常运动点安全。只有这三者共同成立，才支持 AJAE 的核心 temporal claim。

## Gate 4：proxy 是否迁移到真实 OOD

在 E97–E98 完全冻结后，只允许一次性打开 19 条公开真实异常。只有真实 OOD 相对 B0/B1 改善且正常运动安全通过，才支持：

$$
\text{synthetic anomaly proxy}\rightarrow\text{real OOD transfer}
$$

若 Gate 4 失败，当前研究周期停止；不能继续使用同一 19 条数据调方法并仍称其为 untouched confirmation set。

# 4. 关键不可变约束

整个状态机执行期间，以下规则始终优先于任何单个实验：

$$
\boxed{\text{先确定完整反事实世界，再切五帧窗口}}
$$

$$
\boxed{\text{normal-control 与 anomaly-proxy 共用同一传感器重渲染流程}}
$$

$$
\boxed{\text{五帧共享参数并同等监督，中心帧只规定坐标系}}
$$

$$
\boxed{\text{最终学习输出始终是原始 LiDAR 回波点级异常概率}}
$$

$$
\boxed{\text{STU 全程冻结；206 只更新 AJAE 新增参数}}
$$

$$
\boxed{\text{centered five-frame 为离线主设置；causal five-frame 仅为在线消融}}
$$

# 5. 执行记录模板

每完成一个节点，建议在单独的实验记录文件中追加：

```text
Experiment ID:
Date:
Git commit:
Protocol hash:
Data identity:
Seed:
Command:
Resolved config:
Artifacts:
Primary observation:
PASS/FAIL:
Reason:
Unlocked next experiment:
Invalidated downstream experiments:
Notes:
```

# 6. 如何使用这份状态机推进 AJAE

从 **E00** 开始。不要因为某个更后面的模块“代码已经写好了”就跳过前面的资格实验。任何实验失败时，只执行该节点规定的 FAIL 分支；修复后回到指定节点重新验证。这样每次失败都会把不确定性限制在一个很小的局部，而不是重新打开整个课题。

理想主路径为：

$$
E00\rightarrow E01\rightarrow\cdots\rightarrow E49
\rightarrow E50\rightarrow\cdots\rightarrow E77
\rightarrow E78\rightarrow\cdots\rightarrow E84
\rightarrow E85\rightarrow\cdots\rightarrow E98
\rightarrow E99\rightarrow\cdots\rightarrow E104
$$

最终 **E104** 完成，才表示当前定义下的 AJAE 从数据生成、模型、开发验证、机制诊断、真实 OOD 确认到隐藏测试全部闭环。
