# AJAE 主线方案

> 当前权威基线：本仓库`main`与本文记录的全部历史证据。E25-new、E26-v2、E38–E44刷新、E45B-v2、E48、E49、E50–E58与E61–E71已经正式PASS，E59/E60描述性聚合已经完成，Gate 1、Phase 5和Phase 7已经关闭；E45A全分支已经终止且不构成Gate 1条件，当前正式节点为E72。旧提交`44fd6d13798e826b2cac8371de26a7d17707dadc`只保留为E22-v2时期的历史基线，不再代表当前工作区状态。

---

# 0. 统一符号与不可变约束

为避免“点、射线、文件槽位、体素和窗口”混用，统一采用以下符号。

- $f$：原始序列中的帧编号。
- $r=(b,a)$：校准后的规范射线身份，其中 $b$ 为 OS1-128 垂直光束编号，$a$ 为方位角列编号。
- $\rho_f(r)$：第 $f$ 帧中规范射线 $r$ 到原始文件槽位的映射；若该射线无有效回波，则对应空槽。
- $p=(f,r)$：原始或重渲染后一个 LiDAR 量测点的稳定身份。
- $q\in\{-2,-1,0,+1,+2\}$：点在某个中心对称五帧窗口中的相对时间位置。
- $v$：STU 经过 0.05 m 量化后形成的 Minkowski 稀疏体素，不等同于原始 LiDAR 点。
- $\pi_f(p)$：STU 的逆映射，把第 $f$ 帧原始回波点 $p$ 映射到对应稀疏体素。
- $z_{p,q}^{(w)}$：点 $p$ 在窗口 $w$ 中、位于相对位置 $q$ 时的窗口级异常逻辑值。
- $s_{p,q}^{(w)}=\sigma(z_{p,q}^{(w)})$：对应的窗口级异常概率。
- $S_p$：点 $p$ 在所有有效窗口中的异常概率等权平均。

AJAE 遵守以下不可变约束：

$$
\boxed{\text{先确定完整反事实世界，再从中切五帧窗口}}
$$

$$
\boxed{\text{正常渲染对照和异常代理必须使用同一套传感器重渲染流程}}
$$

$$
\boxed{\text{五帧共享参数并接受同等监督；中心帧只用于规定坐标系}}
$$

$$
\boxed{\text{中间只学习多尺度局部时空上下文，不预测对象身份}}
$$

$$
\boxed{\text{最终学习输出始终是原始 LiDAR 回波点级异常概率}}
$$

$$
\boxed{\text{STU 全程冻结；206 只更新 AJAE 新增参数}}
$$

$$
\boxed{\text{中心对称五帧是离线主设置，因果五帧单独作为在线消融}}
$$


## 0.1 研究执行纪律：整段预设计，逐节点执行

AJAE 后续不再采用“完成一个小节点、再临时设计下一个节点”的推进方式。正式执行采用：

$$
\boxed{\text{整段协议预设计与冻结}\;\rightarrow\;\text{按依赖顺序逐节点执行}}
$$

一个阶段开始前，必须一次性完成该阶段全部节点的：

- 输入接口、数据范围与身份规则；
- 样本规模、随机流、最大重试次数和资源上限；
- 直接对应科学构念的主指标与 PASS/FAIL 条件；
- 描述性统计、实现回归与不参与裁决的辅助量；
- 可观察性不足、实现错误、科学失败三类分支；
- 下游接口和失效边界。

后继节点仍只能在前置节点 PASS 后正式执行，但不得再因为“下一节点尚未写阈值”而中断。阶段协议分为两层：

1. **设计冻结**：冻结科学问题、样本规则、指标、阈值和失败分支；
2. **执行冻结**：在代码实现完成后补充 runner、源码、配置和输入产物身份。

执行冻结只绑定实现身份，不允许重新解释设计冻结中的科学构念。

## 0.2 证据层级与阻断原则

所有后续量分为三层。

### A. 硬科学或语义证据

包括：数据泄漏、标签正确性、射线身份、最近回波语义、normal-control/proxy 共用渲染路径、来源泄漏、B1 相对 B0、B3 相对 B1/B2、正常运动安全和真实 OOD 迁移。这些可以阻断主线。

### B. 机械正确性证据

包括：数值有限、坐标变换、双射、确定性、缓存身份、连续几何交点、碰撞拒绝和 world-spec 不重采样。只要机械语义错误就必须修复，但不把实现修复包装成新的科学贡献。

### C. 描述性与机制证据

包括：普通因素相关、接触带离散点数、理想收敛速率、可视化印象、内部得分方差、可见性趋势和运行时分位数。它们默认只报告，不得因一个辅助代理量未达到理想形态而无限派生资格支线。

新的硬门槛必须回答：

> **若该指标失败，AJAE 的标签语义、物理反事实、核心对照或最终科学结论是否会实质失真？**

若答案是否，则该指标不得成为阻断门。

## 0.3 失败分类与预定义处理

每次 FAIL 必须先归入以下一种，而不是立即改阈值：

- `implementation_defect`：实现没有执行冻结语义；修复后按同一构念版本化重跑；
- `sample_or_observability_defect`：指定数据中对象不可观察或样本定义不可实现；只能启用事前冻结的可观察分支；
- `qualification_specification_defect`：门槛测量的不是目标构念；历史 FAIL 永久保留，修订后的构念必须另行执行；
- `scientific_failure`：实现和测量均合格，但核心假设不成立；停止相应主张；
- `descriptive_deviation`：不影响硬结论，记录后继续。

只有以下情况要求真正停下重新裁决：

1. 新事实改变了后续实验的观测对象或标签语义；
2. 需要修改 generator、renderer、模型结构、损失、数据角色或主评价；
3. 前置硬科学条件失败；
4. 预注册分支均无法执行。

一般的拒绝率、分层差异、少数数值尾部和描述性异常，不再自动阻断主线。

---

# 1. 核心科学问题、主张边界与停止条件

## 1.1 核心科学问题

AJAE 研究的问题不是“能否把更多模块接在 STU 后面”，而是：

> **在渲染器匹配的正常/异常反事实世界中，冻结单帧正常表征上的五帧时空上下文，能否在不增加正常运动点误报的条件下，提高真实 STU OOD 点的异常排序质量？**

这里有四个必须分别识别的因素：

1. 程序化异常代理监督本身是否有用；
2. 多尺度点模型是否有用；
3. 跨帧信息是否提供单帧之外的增益；
4. 同一点接受多个重叠窗口评价后，融合是否进一步有用。

任何最终提升都必须通过对照实验区分这四个来源。

## 1.2 异常代理不等于已经证明的真实 OOD

程序化物体没有现实类别标签，但这并不能证明它必然处于 STU 正常分布之外。某些随机形状可能接近杆、交通标志或车辆部件；另一些极端形状又可能远比真实 STU 异常容易识别。

因此，程序化生成物统一称为：

$$
\boxed{\text{合成异常代理}}
$$

其科学作用是提供训练监督，并检验这种监督能否迁移到真实 STU OOD 物体。

只有在方法冻结后，19 条真实异常验证序列上的结果才可以支持“异常代理能够迁移到真实 OOD”的结论。

## 1.3 预先冻结的停止条件

以下任一情况出现时，都不能继续把当前路线解释为成立：

- 在控制必要观测条件后的rendered normal-control与anomaly-proxy比较中，低容量模型仅凭低层生成统计即可近乎饱和地区分标签来源，说明存在直接标签捷径；
- 单帧代理训练模型在 201 上不能优于冻结 STU 的单帧参考，说明异常代理监督本身无效；
- 完整五帧模型不能稳定优于单帧模型和禁用跨帧边的五帧模型，说明跨帧主张不成立；
- 合成 201 上提升，但冻结后在 19 条真实异常序列上没有改善，说明当前代理到真实 OOD 的迁移假设失败。

若最后一项失败，不能看着真实验证结果继续修改方法后，仍把同一批 19 条序列称为“未触碰确认集”。任何修改都属于新的研究周期。

---

# 2. 反事实训练世界：真实正常、渲染正常对照与异常代理

## 2.1 三种监督来源

训练世界必须同时包含三种点源：

| 点的来源 | 点级标签 |
|---|---:|
| 真实正常序列经过规范射线往返后的正常回波 | 正常 |
| 使用同一渲染器插入的正常类别实体回波 | 正常 |
| 使用同一渲染器插入的程序化异常代理回波 | 异常 |

这样打破：

$$
\text{“由渲染器产生”}\Longleftrightarrow\text{“异常”}
$$

这一致命混杂。

所有反事实帧均先经过相同的规范射线网格重建。正常渲染对照和异常代理使用完全相同的：

- 射线集合；
- 遮挡规则；
- 回波产生模型；
- 强度模型；
- 空射线新增回波规则；
- 点身份与输出恢复规则。

模型不能仅凭“这个点看起来像合成点”完成任务。

## 2.2 正常渲染对照

正常渲染对照从 206 的正常实例标签中提取实体模板。第一版优先使用具有明确实例身份的正常类别：

- car；
- bicycle；
- motorcycle；
- truck；
- other-vehicle；
- person；
- bicyclist；
- motorcyclist。

模板只用于生成器内部，不向 AJAE 提供实例 ID。

每个模板可以进行受限的：

- 刚体旋转；
- 平移；
- 正常类别合理范围内的尺度变化；
- 材质重新采样。

放置位置必须符合该正常类别的基本语义，例如车辆位于 road/parking，行人或骑行者位于 sidewalk/road 的合理区域。

正常对照与异常代理应尽量在距离、可见点数、遮挡程度和放置表面上配对，使标签不能由位置和可见性单独决定。

## 2.3 程序化异常代理的几何

异常代理使用类别无关的程序化隐式几何，不直接使用椅子、箱子、垃圾桶等带现实语义类别的外部 CAD 库。

单个基础形状使用超二次曲面等连续可控 primitive：

$$
z_{\mathrm{primitive}}
=
(
\alpha_1,\alpha_2,\alpha_3,\epsilon_1,\epsilon_2
)
$$

其中三轴尺度控制大小，指数控制圆润、方正、柱状、扁平和尖锐程度。

一个异常代理由 $1$ 至 $5$ 个 primitive 构造成连通 union：

$$
G=P_1\cup P_2\cup\cdots\cup P_n
$$

生成从一个连通母体开始。每新增一个 primitive，都必须给出位于该 primitive 与已有实体严格内部的共同见证点；这些见证边形成一棵连通生成树。因此训练主生成器不再采样任意顺序的 difference 或 intersection，也不需要在生成后用昂贵的一般 CSG 判定器猜测对象是否已经分裂。

并加入：

- 弯曲；
- 扭转；
- 渐缩；
- 平滑低频表面形变。

最终几何必须：

- 有界；
- 闭合；
- 数值有限；
- 至少形成一个连通实体；
- 支持稳定射线求交。

每个带表面扰动的 primitive 必须先取得严格径向星形证书；bend、twist 和 taper 只允许使用已证明为连续双射的参数域。由此，基础 primitive 连通、union 交叠图连通、全局形变保持连通，共同构成训练异常代理的连续单实体保证。

difference 与 intersection 的通用隐式几何实现仅保留用于历史实验、求交回归和非训练诊断，不再属于正式训练异常代理分布。这个限制不声称任意 CSG 没有价值，而是把第一版 AJAE 的几何监督限制在能够构造性证明为单个实体的程序化形状族内。

## 2.4 形状、尺度和材质分布

异常代理的总体尺度和轴比例分开采样：

$$
s\sim p_s,\qquad
(r_x,r_y,r_z)\sim p_r
$$

$$
(l,w,h)=s(r_x,r_y,r_z)
$$

训练分布同时覆盖：

- 稀疏小目标；
- 中等可见目标；
- 点数较多的清晰目标；
- 块状、扁平、细长和不对称结构。

几何复杂度、尺度、放置位置和材质尽量独立采样，避免形成固定组合捷径。

训练阶段至少混合单超二次曲面、强制重叠的多 primitive union 和连续形变；开发诊断中保留至少一种训练时完全没有使用的程序化形状机制。

这些 held-out 形状只用于诊断，不参与模型选择。

## 2.5 世界级实体、合格支撑池与放置流水线

每个插入实体在完整反事实世界中定义为：

$$
O_m=(G_m,P_m,\rho_m,ID_m,\ell_m)
$$

其中：

- $G_m$：程序化几何或 normal-control 模板；
- $P_m$：冻结的世界位姿；
- $\rho_m$：材质；
- $ID_m$：生成器内部身份；
- $\ell_m\in\{\text{normal-control},\text{anomaly-proxy}\}$：生成标签。

第一版插入实体默认静止：

$$
P_m(t)=P_m.
$$

运动不能成为异常标签来源。

### 2.5.1 支撑位置的唯一权威入口

train/206 的正式放置位置只允许来自 E21-v4 已资格的：

$$
\mathcal P_{\mathrm{support}}^{206}.
$$

该池使用实际可观察的：

- road = 40；
- sidewalk = 48；
- other-ground = 49。

parking = 44 与 semantic 60 在 train/206 中没有真实回波，因此当前训练主张不包含它们。不得恢复“从任意 ground return 先尝试再拟合”的旧入口。

train/201 的开发世界必须以同一冻结算法独立建立 support pool；只对实际可观察并取得平面资格的类别作开发主张，不把 206 的池条目或标签结果迁移到 201。

### 2.5.2 放置顺序

每个候选实体必须依次通过：

$$
\boxed{
\text{qualified support patch}
\rightarrow
\text{continuous grounding}
\rightarrow
\text{observed-normal collision rejection}
\rightarrow
\text{inserted-pair collision rejection}
\rightarrow
\text{semantic support policy}
}
$$

其中：

1. 对象局部 $+z$ 轴与支撑法向 $n_g$ 对齐；
2. 沿 $n_g$ 使用连续几何最低支撑值落地；
3. 只对实际观测到的非地面正常回波检查明显穿透；
4. 多实体世界按冻结顺序放置，后放对象不得与已接受对象明显互穿；
5. normal-control 还必须满足类别—支撑面规则。

E22-v2 已冻结的落地边界继续为：

$$
|d_{\min}|\le0.01\ \mathrm m,
$$

以及深入支撑平面超过 $0.02$ m 的表面比例不超过 $2\%$。离散接触带点数只作描述，不再定义是否接触。

### 2.5.3 normal-control 的基本语义策略

train/206 中：

- vehicle-like 模板：car、bicycle、motorcycle、truck、other-vehicle，只放在 qualified road；
- person/rider-like 模板：person、bicyclist、motorcyclist，放在 qualified road 或 sidewalk；
- other-ground 可用于 anomaly-proxy，但不作为 normal-control 的默认正常语义支撑面；
- parking 只有在某个未来允许序列中真实可观察、独立建立合格池并单独取得资格后才可启用。

这里验证的是基本正常语义，不试图重建完整交通规则。

### 2.5.4 碰撞主张边界

由于原始 LiDAR 不能恢复隐藏表面，AJAE 只承诺：

> 插入实体不与已观测正常几何发生超过冻结容差的明显穿插，并且未发现插入实体之间的明显互穿。

不得写成“证明与真实场景全部隐藏几何无碰撞”。AABB 或包围球只能用于 broad phase，正式裁决必须落在连续 SDF 或已资格的凸几何上。

### 2.5.5 身份与证据记录

`WorldSpec`保存最终不可变世界；独立`WorldGenerationReport`保存最终成功world attempt的身份及其中每个已接收实体的支撑proposal记录，包括：

- support-patch identity；
- shape/template identity；
- material、yaw 和随机流身份；
- placement proposal index；
- ground、observed-normal 与 pair-collision 裁决；
- 该成功attempt内、各已接收实体在最终接受前经历的全部支撑proposal拒绝原因。

此前未完成world attempt不会进入这个最终报告；正式资格另以`world_attempt`报告成功前使用的attempt序号。上述字段只用于复现和诊断，不进入AJAE输入。

## 2.6 每条世界的实体组成与生成顺序

一条反事实序列可以属于四种冻结 world type：

- `pure_normal`：不插入实体；
- `control_only`：只插入 normal-control；
- `anomaly_only`：只插入 anomaly-proxy；
- `mixed`：同时包含两者。

异常代理数量允许：

$$
M_A\in\{0,\ldots,9\},
$$

以 $1$ 至 $3$ 个为主，少量世界包含更多异常。normal-control 数量 $M_N$ 采用同一非退化数量域。

完整世界先固定所有实体，再进行任何逐帧渲染。实体生成顺序、support pool采样、几何、模板、材质、位姿、碰撞重试与对象ID均由独立随机流决定。在同一个world attempt内，某个实体的候选失败只推进该实体的冻结支撑proposal流，不移动或重写此前已接受实体；若该实体耗尽固定proposal上限，或者全部实体完成后的最终完整世界复核失效，则该world attempt整体失败，并按冻结attempt seed进入下一次完整尝试。

同一个背景区域在不同世界中可以：

- 放 normal-control；
- 放 anomaly-proxy；
- 不放实体。

normal-control 与 anomaly-proxy 在候选池阶段应尽量共享或匹配：

- 支撑面语义；
- 距离；
- 尺度；
- 可见点数；
- 遮挡程度；
- 材质量化层。

这里的“候选池阶段”不属于E25-new的生成接受条件。E25-new只负责合法、可见且coverage-oriented的normal-control生成。当前硬审计只在E45B-v2/E48中为rendered normal-control与anomaly-proxy控制必要的可比较性并检查标签捷径；real-normal与control的可见点数、遮挡、密度和自然频率差异不再构成硬门，可由E46按需诊断。


---

# 3. OS1-128 射线级反事实近似

## 3.1 射线身份审计是前置条件

在正式渲染前，必须确认原始文件槽位是否稳定对应 OS1-128 的 beam/azimuth 射线。

审计至少验证：

1. 各帧槽位总数和空槽规律；
2. 每个槽位的 elevation 分布；
3. 128 beam 的排列周期；
4. 方位角列的连续性；
5. 同一槽位跨帧方向是否稳定；
6. 是否存在多回波重排；
7. range-image 往返能否恢复原始点数、方向和距离。

内部规范射线身份统一定义为：

$$
r=(b,a)
$$

若原始槽位审计通过，可直接建立固定映射 $\rho_f(r)$；若不通过，则必须先重建 beam/azimuth 索引，不能继续把文件槽位当作物理射线。

该审计未通过前，不能开始正式异常世界生成。

## 3.2 规范射线网格

每一帧使用同一套已校准的 OS1-128 射线集合：

$$
\mathcal R_t=\{r_{t,1},\ldots,r_{t,R}\}
$$

每条射线由：

- beam ID $b$；
- azimuth column $a$；
- 传感器原点；
- 单位方向；

唯一确定。

原始点投回对应射线，得到：

$$
d_{\mathrm{normal}}(r)
$$

原始空射线定义为：

$$
d_{\mathrm{normal}}(r)=+\infty
$$

程序化实体或正常控制与该射线的最近几何交点为：

$$
d_{\mathrm{insert}}(r)
$$

## 3.3 回波产生和最近回波竞争

几何交点不自动等于有效回波。

对候选交点，先按照：

$$
P(\mathrm{return}\mid b,d,\mu,\rho)
$$

决定是否产生回波，其中：

- $b$：beam ID；
- $d$：距离；
- $\mu=|\hat n^\top\hat r|$：入射余弦；
- $\rho$：材质状态。

只有产生回波时，插入实体才参与最近回波竞争：

$$
d_{\mathrm{final}}(r)
=
\min
\left(
d_{\mathrm{normal}}(r),
d_{\mathrm{insert}}(r)
\right)
$$

因此：

- 插入实体更近且产生回波时，替换原正常回波；
- 正常前景更近时，插入实体被遮挡；
- 原本空射线可以因插入实体产生新回波；
- 几何命中但回波模型拒绝时，不产生点。

## 3.4 强度模型

在 206 上估计：

$$
P(\mathrm{return}\mid b,d,\mu)
$$

和：

$$
p(I\mid \mathrm{return},b,d,\mu)
$$

再由对象材质 $\rho$ 做平滑调制：

$$
I=g(I_0;\rho)+\epsilon
$$

其中：

$$
I_0\sim p(I\mid \mathrm{return},b,d,\mu)
$$

强度裁剪到 206 的实际支持范围。

正常渲染对照与异常代理使用同一回波和强度模型。材质参数与几何标签独立，且显式包含与邻近正常表面强度高度相似的异常代理。

## 3.5 物理主张边界

第一版渲染器的准确名称是：

$$
\boxed{\text{基于刚性单帧的 OS1-128 第一回波反事实近似}}
$$

它处理：

- 射线方向；
- 最近回波；
- 双向遮挡；
- 距离；
- beam；
- 入射角；
- 材质；
- 强度；
- 空射线新增回波。

它不声称完整模拟：

- 扫描周期内运动畸变；
- beam divergence；
- 有限光斑；
- 多回波；
- 多径反射；
- 所有电子噪声；
- 未观测隐藏表面。

因此不得写成“与真实采集完全等价”。

## 3.6 完整世界、不可变规格与确定性逐帧渲染

设正常序列为：

$$
\mathcal S^N=(X_1,\ldots,X_T).
$$

完整反事实世界先一次性固定：

$$
\Omega^{(r)}=
\{\text{background identity},\text{world seed},O_1,\ldots,O_M,\text{renderer identity}\}.
$$

`WorldSpec` 采用规范 JSON 序列化并形成唯一 hash。任何窗口遍历、进程数、缓存状态或请求顺序都不得改变 $\Omega^{(r)}$。

随后按帧确定性渲染：

$$
\tilde X_t^{(r)}
=
\operatorname{Render}(X_t,\Omega^{(r)},T_{\mathrm{sensor}}(t)).
$$

同一个：

$$
(\text{world hash},\ f,\ \text{renderer identity})
$$

必须逐槽产生完全相同的：

- XYZI；
- packed label；
- inserted/occluded/unchanged masks；
- internal object ID；
- visible point identity。

“完整世界”是逻辑上的不可变对象，不要求一次将全部帧物化到磁盘或内存。缓存只能复用由上述完整 key 唯一决定的帧，不得把 frame ID 当成跨世界缓存身份，也不得在窗口 loader 内重新采样实体、材质或 return RNG。


---

# 4. Gate 1：渲染器机械资格与标签捷径审计

Gate 1 分为三层，不再为每个描述性分布差异设置孤立硬门。

## 4.1 机械语义层

E27–E37 必须证明：

- normal-control 与 anomaly-proxy 都能被规范射线正确击中；
- return sampling、最近回波竞争、空射线新增和强度生成按同一实现执行；
- 插入物和原始背景的双向遮挡正确；
- label 只在统一传感器流程完成后决定监督语义；
- 同一 world/frame 跨窗口、进程和缓存逐槽一致。

这一层使用解析 fixture 和确定性反例，要求零语义错误。它不要求实际数据分布已经相同。

## 4.2 传感器统计与支持覆盖层

E38–E44 对 real normal、rendered normal-control 和 anomaly-proxy 统一报告：

- per-beam 回波率；
- per-range 回波率；
- beam×range 条件强度；
- empty→valid；
- 单实体 $N_{vis}$；
- 连续帧可见点数变化；
- 遮挡率。

这些节点检查数值、守恒、定义和支持覆盖。普通分布差异不单独构成 Gate 1 FAIL。特别是 real-normal 与 coverage-oriented normal-control 的距离、遮挡、密度和场景组成差异，不等同于 renderer artifact。

## 4.3 Rendered normal-control 与 anomaly-proxy 资格

E45B-v2 从 201 的固定候选银行中构造：

$$
\text{rendered normal-control}
\leftrightarrow
\text{anomaly-proxy}
$$

的匹配实体—帧单元。两者都由 renderer 生成但标签不同，因此这是标签捷径审计的直接比较对象。匹配在查看 E48 结果前冻结，并控制：

- support semantic；
- 距离；
- beam 条件；
- $\log(1+N_{vis})$；
- 遮挡率；
- 局部点密度。

若确定性候选银行不足，只能按事前冻结的容量阶梯扩大 E45B-v2 独立银行；不得根据 E48 结果改变匹配规则。

real-normal 与 normal-control 的 E45A 严格匹配、targeted controls、共同重叠加权、transport feasibility、ESS 优化、更大银行、新 matching、新 weighting 和 target-conditioned generator 均已停止，不再是 Gate 1 前置条件。历史结果永久保留，用于说明自然观测分布与 coverage-oriented 插入分布不同，尤其是场景遮挡不同。

## 4.4 Rendered 标签捷径检查

E48 使用 grouped cross-validation 的低容量模型区分：

$$
\text{rendered normal-control}
\quad\text{vs}\quad
\text{anomaly-proxy}.
$$

输入只允许：

- $x,y,z$；
- intensity；
- beam；
- range；
- local density。

如果极低容量模型仅凭低层特征就近乎饱和地区分两种 rendered 标签来源，说明 proxy 路径存在直接标签相关捷径，Gate 1 必须阻断并返回 proxy 路径。E48 不要求两类完全不可分，因为 anomaly-proxy 的几何差异本来就是监督内容；它只排除任务被低层生成统计近乎直接解决。

E48 PASS 只说明 proxy 任务没有退化成显然的低层捷径。真正的代理监督价值仍由：

$$
B1>B0
$$

裁决。

## 4.5 Real-normal 与 rendered normal-control 来源诊断

E46 保留为可选、非阻断诊断。它可以在事前冻结的合理共同支持区域中报告 real-normal 与 rendered normal-control 的 AUC、平衡准确率、不确定性和特征归因，但不再要求 E45A PASS、严格孪生匹配或 ESS≥256，也不进入 Gate 1 合取条件。

E46 可分只说明来源身份与低层特征相关：

$$
S\rightarrow X.
$$

它不能单独推出该特征足以预测异常标签，因为 rendered 数据内部同时包含标签为 0 的 normal-control 与标签为 1 的 anomaly-proxy。只有 E48、正常安全、$B1>B0$ 和最终 real-OOD transfer 能直接裁决标签捷径是否损害 AJAE。E47 相应降为可选来源差异归因，不再自动触发 renderer 回修。

## 4.6 Gate 1 的统一边界

Gate 1 的硬门只包含兼容的 canonical-ray 与 renderer 机械证据、E38–E44 的直接语义/守恒/支持检查、E45B-v2 control/proxy 资格和 E48 rendered 标签捷径检查。E45A 全分支、E45-V1、E46 和 E47 均不进入合取条件。Gate 1 PASS 后进入 E50–E71，再由 $B1>B0$、正常安全、$B3>B1/B2$ 和最终 real-OOD transfer 回答 AJAE 真正的有效性与迁移问题。


---

# 5. STU：冻结的单帧正常世界编码器

## 5.1 定位和冻结

五帧全部使用同一套官方 STU 权重：

$$
F_k=E_{\mathrm{STU}}(X_k;\theta_{\mathrm{STU}})
$$

并满足：

$$
\boxed{\theta_{\mathrm{STU}}\text{ 全程冻结}}
$$

AJAE 只训练输入投影、分层时空点模型和点级异常头。

异常代理和正常渲染对照均先在原始点云世界中生成，再经过 STU；不在特征空间直接注入异常。

## 5.2 128 维高层稀疏特征

STU 官方模型先执行：

```python
all_features = self.backbone(x)
point_features = self.point_features_head(all_features[-1])
```

`point_features_head` 输出：

$$
f_v^{\mathrm{STU}}\in\mathbb R^{128}
$$

其中 $v$ 是 STU 稀疏体素。

通过官方逆映射：

$$
v=\pi_f(p)
$$

恢复到原始回波点：

$$
f_p^{\mathrm{STU}}
=
f_{\pi_f(p)}^{\mathrm{STU}}
$$

多个原始点落入同一 0.05 m 体素时可以共享 STU 特征，但仍保留各自的：

- 原始坐标；
- intensity；
- 规范射线身份；
- 最终点级预测。

## 5.3 与官方分配一致的正常语义证据

STU 输出：

$$
L^{\mathrm{query}}\in\mathbb R^{Q\times20}
$$

以及：

$$
M\in\mathbb R^{N_v\times Q}
$$

定义：

$$
P_{qc}
=
\operatorname{softmax}(L^{\mathrm{query}})_{qc}
$$

$$
A_{vq}
=
\sigma(M_{vq})
$$

对每个稀疏体素，选择官方置信逻辑下最强的 query：

$$
q^*(v)
=
\arg\max_q
\left[
A_{vq}
\max_{c\le19}P_{qc}
\right]
$$

若并列，固定选择编号最小的 query，保证确定性。

正常语义证据定义为：

$$
e_v^{\mathrm{normal}}
=
A_{vq^*}P_{q^*,:19}
\in\mathbb R^{19}
$$

同时保留两个可靠性量：

$$
r_v^{\mathrm{assign}}
=
A_{vq^*}\max_{c\le19}P_{q^*c}
$$

$$
r_v^{\mathrm{noobj}}
=
P_{q^*,20}
$$

再通过逆映射广播到原始点。

第一版不使用所有 query 的未经归一化求和，因为该量会随 query 重叠数量变化，也不符合 STU 官方点级分配逻辑。

## 5.4 AJAE 点级输入

每个原始回波点的输入为：

$$
v_p=
[
f_p^{\mathrm{STU}},
e_p^{\mathrm{normal}},
r_p^{\mathrm{assign}},
r_p^{\mathrm{noobj}},
I_p
]
$$

再加入：

- 中心时刻坐标系中的三维位置；
- 相对时间编码 $q$。

统一投影为：

$$
u_p
=
\phi(v_p)
+
e_x(x_p)
+
e_t(q_p)
$$

第一版不直接输入 STU Query token，也不额外手工输入 entropy、energy、MSP 等统计量。

---

# 6. 中心对称五帧局部世界

## 6.1 离线主设置

第一版主窗口为：

$$
W_t=[X_{t-2},X_{t-1},X_t,X_{t+1},X_{t+2}]
$$

这是离线序列异常分割设置，允许使用未来两帧。

论文必须明确：

> AJAE 主模型不是在线实时方法。

因果窗口：

$$
[t-4,t]
$$

作为单独消融，报告性能和延迟差异。

## 6.2 中心时刻坐标系

设：

$$
T_{W\leftarrow S_k}
$$

为第 $k$ 帧 LiDAR 坐标到世界坐标的刚体变换。

点统一到中心时刻 $t$ 的 LiDAR 坐标系：

$$
x_{k,i}^{(t)}
=
T_{S_t\leftarrow W}
T_{W\leftarrow S_k}
x_{k,i}^{S_k}
$$

中心帧只提供坐标规范，不具有预测优先级：

- 五帧共享参数；
- 五帧都作为查询点；
- 五帧全部接受监督；
- 最终融合不额外偏向 $q=0$。

## 6.3 正常运动物体

ego-motion 对齐只能对齐静态世界。正常车辆、行人和骑行者在统一坐标系中会形成位移轨迹。

因此 AJAE 必须：

1. 始终保留同帧残差路径；
2. 允许跨帧证据被门控拒绝；
3. 单独报告正常运动点的误报；
4. 不把“世界坐标中不稳定”直接定义为异常。

原始 moving semantic 标签只用于正常安全诊断，不进入模型输入。

---

# 7. AJAE 主体：四级分层时空点模型

## 7.1 固定第一版结构

第一版主模型固定为：

$$
\boxed{
\text{四级金字塔}
+
\text{mean-max 体素池化}
+
\text{按时间分层的局部注意力}
+
\text{同帧 3-NN 上采样}
+
\text{高分辨率跳连}
}
$$

不同时实现 attention pooling、不同层数和多个 decoder 版本后再择优。

## 7.2 每帧独立体素化

五帧已对齐到同一坐标系，但降采样仍按时间位置独立执行。

体素键包含：

$$
(q,\lfloor x/v_l\rfloor)
$$

不同 $q$ 的点绝不在池化阶段直接合并。

四级结构为：

- L0：原始点层；
- L1：细尺度体素层；
- L2：对象部件/中尺度层；
- L3：局部场景尺度层。

## 7.3 mean-max 池化

每个体素同时计算：

$$
h_v^{\mathrm{mean}}
=
\operatorname{mean}_{i\in v}h_i
$$

$$
h_v^{\mathrm{max}}
=
\operatorname{max}_{i\in v}h_i
$$

再融合：

$$
h_v
=
\operatorname{Linear}
[
h_v^{\mathrm{mean}},h_v^{\mathrm{max}}
]
$$

这样既保留总体统计，也减少少量强异常响应被纯均值冲淡的风险。

## 7.4 按时间分层的邻域

不能让五帧所有点共同竞争一个 $K_l$，否则同帧邻居可能占满邻域，模型实际读取不到跨帧证据。

对节点 $i$，按相对时间差分别构造：

$$
\mathcal N_{l,\delta}(i)
=
\operatorname{KNN}_{K_{l,\delta}}
\left\{
j:
q_j-q_i=\delta,
\|x_i-x_j\|<r_{l,\delta}
\right\}
$$

其中：

$$
\delta\in\{-2,-1,0,+1,+2\}
$$

每个时间差拥有独立的半径和最大邻居数；候选不足时允许为空，不引入远距离点凑数。

注意力共享主要投影参数，时间差通过相对时间编码进入：

$$
A_{ij}
=
\frac{Q_iK_j^\top}{\sqrt d}
+
\psi(\Delta x_{ij},\delta)
$$

## 7.5 同帧残差与跨帧拒绝

同帧分支始终保留：

$$
m_{i,0}
=
\operatorname{Attn}
(h_i,\mathcal N_{l,0}(i))
$$

每个跨帧分支得到：

$$
m_{i,\delta},
\qquad \delta\ne0
$$

并使用轻量门控：

$$
g_{i,\delta}
=
\sigma
\left(
G_l[
h_i,m_{i,\delta},\delta
]
\right)
$$

若该时间差邻域为空，则：

$$
m_{i,\delta}=0,\qquad g_{i,\delta}=0
$$

层更新为：

$$
h_i'
=
h_i
+
F_l(m_{i,0})
+
\sum_{\delta\ne0}
g_{i,\delta}F_l(m_{i,\delta})
$$

这样正常运动物体可以保留同帧证据，并拒绝不可靠的跨帧邻域；模型不被强迫把所有空间邻近历史点都当作同一对象。

## 7.6 感受野随层级扩大

满足：

$$
r_{0,\delta}<r_{1,\delta}<r_{2,\delta}<r_{3,\delta}
$$

使上下文从局部表面逐步扩展到对象部件、实体尺度和局部场景尺度。

“对象尺度”在当前阶段是待验证的机制解释，不是由网络结构自动保证的事实。

## 7.7 同帧 3-NN 上采样

从粗层回到细层时，只在相同时间位置中寻找三个最近父节点：

$$
\hat h_i^{(l)}
=
\sum_{j\in\mathcal N_3^{\mathrm{same\ frame}}(i)}
w_{ij}h_j^{(l+1)}
$$

$$
w_{ij}
=
\frac{1/(d_{ij}+\varepsilon)}
{\sum_m1/(d_{im}+\varepsilon)}
$$

再与高分辨率跳连融合：

$$
h_i^{out}
=
\phi[
h_i^{skip},\hat h_i^{(l)}
]
$$

跨帧信息已经写入 coarse feature；上采样不再执行时间匹配。

---

# 8. 点级监督和主损失

## 8.1 标签定义

重渲染后真正可见的点分为：

### 真实或规范往返后的正常回波

$$
y_p=0
$$

### 渲染正常对照回波

$$
y_p=0
$$

### 异常代理回波

$$
y_p=1
$$

### 原来存在但被遮挡掉的回波

不再是新世界中的观测点，不参与损失。

### raw semantic 0

保持 ignore，除非该规范射线被新插入实体产生的有效回波替换。

## 8.2 五帧全部监督

五帧中所有可见点都输出逻辑值并参与监督。

没有“只监督中心帧”或“只监督最后一帧”的设计。

## 8.3 官方距离域

主异常损失只在：

$$
\boxed{2.5\text{ m}\le d\le50\text{ m}}
$$

的可观测点上计算。

距离域外点可以作为上下文，但不贡献主分类损失。

## 8.4 对空类别安全的平衡 BCE

定义：

$$
\mathcal P_+=\{p:y_p=1,\ 2.5\le d_p\le50\}
$$

$$
\mathcal P_-=\{p:y_p=0,\ 2.5\le d_p\le50\}
$$

两类都存在时：

$$
L_{\mathrm{anom}}
=
\frac12
\operatorname{mean}_{p\in\mathcal P_+}
\operatorname{BCE}(z_p,1)
+
\frac12
\operatorname{mean}_{p\in\mathcal P_-}
\operatorname{BCE}(z_p,0)
$$

若当前窗口没有异常点，则：

$$
L_{\mathrm{anom}}
=
\operatorname{mean}_{p\in\mathcal P_-}
\operatorname{BCE}(z_p,0)
$$

若极端情况下只有正样本，则只计算正样本项。

训练不因异常点少于 5 个而丢弃窗口；“少于 5 个异常点不计入指标”只属于 STU 官方评价规则。

## 8.5 第一版总损失

第一版只使用：

$$
\boxed{L=L_{\mathrm{anom}}}
$$

明确删除：

- 点身份 EMA memory；
- $L_{\mathrm{cf}}$；
- 显式点级平滑；
- 对象 ID 损失；
- 跟踪损失；
- 运动回归；
- 重建损失；
- 复杂对比学习。

防止记住 206 主要依靠：

- 206/201 背景分离；
- 相对坐标；
- 大量反事实世界；
- 正常渲染对照；
- 纯正常 201 安全检查；
- 受控模型容量。

---

# 9. 重叠窗口和最终点分数

## 9.1 稳定点身份

最终量测点身份为：

$$
p=(f,r)
$$

其中 $r=(b,a)$ 是规范射线身份。

原始文件槽位只用于输入/输出恢复，不作为未经审计的物理身份假设。

同一帧进入不同窗口时，$p$ 不变，因此可以收集最多五个窗口级预测。

## 9.2 概率等权平均

模型先输出逻辑值：

$$
z_{p,q}^{(w)}
$$

转为概率：

$$
s_{p,q}^{(w)}
=
\sigma(z_{p,q}^{(w)})
$$

最终：

$$
\boxed{
S_p
=
\frac1{m_p}
\sum_{w\ni p}
s_{p,q(w)}^{(w)}
}
$$

其中：

$$
1\le m_p\le5
$$

第一版明确平均概率，不平均逻辑值，也不把它解释为独立证据的贝叶斯融合。

序列边界只使用实际存在的完整五帧窗口，不 padding、不重复帧、不镜像。

## 9.3 时间位置校准诊断

在 201 上分别报告点处于：

$$
q=-2,-1,0,+1,+2
$$

时的：

- AP；
- 正常点平均分；
- 异常点平均分；
- 分数尺度。

若不同 $q$ 的分数分布明显不一致，等权平均的前提不成立，必须先修正位置偏置，不能直接依赖融合提高结果。

---

# 10. 识别时间贡献的最小对照矩阵

以下对照是主研究设计的一部分，不是可选附录。

| 条件 | 设置 | 回答的问题 |
|---|---|---|
| B0 | 冻结 STU 的单帧 MaxLogit/官方可复现单帧分数 | 外部单帧参考 |
| B1 | 单帧 STU 点接口 + 同一正常对照/异常代理训练 + 点级模型 | 代理监督本身是否有效 |
| B2 | 与完整模型相同的五帧结构，但屏蔽全部 $\delta\ne0$ 跨帧边；最终只取 $q=0$ 预测 | 参数量、多尺度和共享结构是否解释提升 |
| B3 | 完整跨帧注意力；每帧只取以该帧为中心的 $q=0$ 预测 | 单个五帧上下文是否优于单帧 |
| B4 | B3 的完整模型 + 同一点五个窗口概率平均 | 多窗口共识是否进一步有效 |
| B5 | 纯因果五帧 $[t-4,t]$，只输出当前帧 | 在线限制下的性能代价 |

核心主张成立至少需要：

$$
\boxed{B3>B1}
$$

并且：

$$
\boxed{B3>B2}
$$

只有：

$$
B4>B3
$$

才支持“跨重叠窗口融合有额外价值”。

B5 用于说明未来帧和离线设置的实际贡献，不能把中心对称五帧结果直接包装成实时在线能力。

---

# 11. 正常运动安全与“对象尺度”诊断

## 11.1 正常运动点安全

利用 206/201 中 moving car、moving person、moving bicycle 等原始语义，单独报告：

- 正常运动点平均异常分数；
- 正常运动点误报率；
- B1 与 B3/B4 的差异；
- 静态正常点与运动正常点的差异。

若五帧模型提升异常代理表现，却显著恶化正常运动点，当前主模型不能通过。

## 11.2 对象尺度上下文诊断

生成器内部的对象 ID 只用于诊断，不进入模型。

至少报告：

### 实体内部得分方差

$$
\operatorname{Var}_{p\in O_m}(S_p)
$$

检查同一异常代理内部是否仍然高度碎片化。

### 异常边界泄漏

比较异常代理表面点与其附近正常背景点的分数，检查多尺度聚合是否把异常高分扩散到道路或邻近正常物体。

### 五帧可见性分层

$$
V_{m,W}
=
\sum_{t\in W}
\mathbf1[N_{m,t}^{vis}>0]
$$

按 $V=1,\ldots,5$ 分层，检查多帧可见证据增加时性能是否改善。

若性能不随多帧证据增加，不能把结果解释为“对象尺度时空共识”。

---

# 12. 数据角色和开发纪律

## 12.1 206：AJAE 新增参数的唯一训练来源

206 用于：

- 真实正常规范世界；
- 正常渲染对照世界；
- 异常代理世界；
- AJAE 新增参数更新；
- 回波和强度模型校准。

冻结 STU 的权重继承其官方训练来源，但 AJAE 不修改这些权重。

## 12.2 201：唯一开发来源

201 不参与梯度更新，用于：

- 纯正常背景泛化；
- 固定正常渲染对照；
- 固定异常代理世界；
- 模型和超参数选择；
- 阈值与 DBSCAN 开发；
- 正常运动安全；
- 渲染器泄漏诊断。

## 12.3 固定开发世界与一次性候选银行

固定 201 开发试验台不再由人工反复补世界，而由一次性候选银行和确定性覆盖选择构造。

### 候选银行

使用与训练完全相同的：

- 201 support-pool 算法；
- E22–E25 放置规则；
- renderer；
- schema 7 generator；
- normal-control 模板与语义策略。

候选银行容量按预先冻结的阶梯扩大，直到满足冻结覆盖或达到最大容量；选择只读取生成身份和预定义难度量，不读取 B0/B1/B3 模型分数。

### 24 条 in-generator development worlds

从候选银行确定性选择 24 条 mixed worlds，用于：

- 模型与超参数选择；
- checkpoint 排序；
- Gate 2/3 的逐世界配对比较；
- 阈值与 DBSCAN 开发。

选择必须覆盖 $N_{vis}$、遮挡 $O$、距离 $d$ 与五帧可见性 $V=1,\ldots,5$ 的冻结边际层。

### 6 条 held-out diagnostic worlds

使用训练时完全未见的程序化机制，身份固定，只做机制诊断，不参与：

- checkpoint 选择；
- 阈值选择；
- PASS/FAIL 阈值调整；
- 方法修改后的优劣选择。

### 纯正常开发域

pure-normal 201 和 moving-normal 子集独立保留，不计入 30 条 synthetic worlds。它们用于正常安全，不被异常世界的样本量淹没。

## 12.4 19 条真实异常验证

19 条公开真实异常序列只在以下内容全部冻结后使用：

- 生成器；
- 正常控制；
- STU 接口；
- 模型结构；
- 损失；
- 超参数；
- checkpoint 选择规则；
- 融合方式；
- 阈值和 DBSCAN。

这 19 条序列是一次性确认实验。

若真实结果失败，当前迁移假设判定失败；不能继续用这批数据调方法后仍将其称为确认集。

## 12.5 51 条隐藏测试

只在真实公开验证支持主张之后进行最终官方测试。

---

# 13. 训练组织、资源边界与预注册预算

## 13.1 动态完整世界

训练持续采样新的 206 世界规格：

$$
\Omega^{(1)},\Omega^{(2)},\ldots
$$

每个世界先固定全部实体、位置、材质和随机流，再按完整合法中心窗口遍历。相邻窗口共享四帧，因此窗口数量不能解释成独立世界数量。

训练世界只能由 E26 已资格的 world builder 产生；不得在 dataloader 内使用另一套 placement 或 collision 路径。

## 13.2 世界类型混合

训练流必须包含：

- pure-normal；
- control-only；
- mixed；
- anomaly-only。

具体比例、每 seed 最大世界数、评价间隔和 early-stopping patience 在 E63 一次冻结，并对 B1/B2/B3 保持一致。不得为某个模型单独增加训练预算后再声称结构优越。

## 13.3 有界滚动缓存

采用：

$$
\boxed{\text{按时间块确定性渲染}+\text{有限帧 LRU 缓存}}
$$

缓存 key 至少包含：

- world hash；
- frame identity；
- renderer/generator identity；
- STU identity；
- 输入接口 schema。

当前块结束后释放不再需要的点级接口。正式大规模任务启动前必须检查宿主磁盘的实际剩余空间，并以峰值占用而不是最终文件大小做预算。

## 13.4 批次与优化

第一版：

$$
\text{micro-batch}=1\text{ 个完整窗口}
$$

使用梯度累积形成有效 batch。不得为提高 batch size 随机删除大量原始点，尤其不能让稀疏小异常优先消失。

B1 smoke 只验证训练机械；正式 B1/B2/B3 均使用同一冻结优化器、学习率、权重衰减、训练世界上限、评价节奏和 checkpoint 规则。

## 13.5 训练种子与公平预算

正式开发固定至少：

$$
\boxed{3\text{ 个独立训练种子}}
$$

每个 seed 具有：

- 独立初始化；
- 独立 206 世界流；
- 相同 201 开发世界；
- 相同训练预算；
- 相同 checkpoint 选择规则。

机械崩溃且协议不变时只重跑无效 seed；任何训练规则变化都使该条件的三个 seed 全部失效。


---

# 14. 评价、统计与公平比较

## 14.1 STU 官方点级指标

严格使用：

- AP；
- AUROC；
- FPR95。

有效距离为：

$$
\boxed{2.5\text{ m}\le d\le50\text{ m}}
$$

经过距离和 ignore 过滤后，一帧异常点少于 5 个时，该帧不加入官方点级指标累计。

正式结果必须再由 STU 官方 evaluator 读取预测文件复算。

## 14.2 对象级评价

对每个原始帧，使用已融合的 $S_p$：

$$
S_p
\rightarrow
\mathbf1[S_p>\tau]
\rightarrow
\text{逐帧 3D DBSCAN}
\rightarrow
\text{异常实例 ID}
$$

不跨帧跟踪。

报告：

- RecallQ；
- SQ；
- RQ；
- UQ；
- PQ；
- TP/FP/FN。

阈值和 DBSCAN 参数只在 201 开发阶段固定。

## 14.3 分层不确定性与统计单位

正式开发报告三层不确定性：

1. 三个训练 seed 的独立结果；
2. 24 条固定开发世界的配对差异；
3. 19 条真实异常序列的逐序列差异。

不得把数百万点当作独立重复。核心比较使用：

- 训练 seed；
- world；
- sequence；

作为抽样单位，采用事前冻结的分层/层级 bootstrap。点级 pooled AP/AUROC/FPR95 仍按官方定义报告，但置信与主张不能只依赖点级样本量。

对 B1、B3 和 B4，正式 superiority 需要同时满足：

- 冻结的实用效应下限；
- 配对层级 bootstrap 的置信条件；
- 至少 2/3 训练 seed 方向一致；
- 正常安全条件。

机制量和相关性只作解释，不替代主指标。

## 14.4 计算成本和输入公平性

论文必须报告：

- 单帧 B1 的推理成本；
- 中心对称 B3/B4 的窗口延迟；
- 因果 B5 的延迟；
- 显存；
- 吞吐量；
- STU 前端是否缓存。

与 NDP、REL、LIDO 等单帧方法比较时，应明确 AJAE 使用额外时间输入和未来帧，不能暗示输入预算相同。

---

# 15. 四个决策门与一次冻结的科学裁决

## 门 1：规范射线和 renderer 是否可信

必须通过：

- ray/slot 与单发布回波接口；
- 几何命中、return、强度、遮挡、空射线和窗口一致性；
- E45B-v2 rendered normal-control vs anomaly-proxy 可比较资格；
- E48 rendered 标签低层捷径审计。

E45A全分支和real normal vs rendered normal-control来源分类不进入硬门；E46只作为可选诊断。硬门失败时只回到被直接证据否定的renderer机械或proxy路径。

## 门 2：异常代理监督是否有效

B1 相对 B0 必须满足 E63 冻结的：

- AP 实用增益；
- world/seed 层级置信条件；
- pure-normal、normal-control 和 moving-normal 安全。

失败说明当前代理监督不足，不能进入五帧主张。

## 门 3：跨帧信息是否提供可识别增益

必须同时满足：

$$
B3>B1,
$$

$$
B3>B2,
$$

并通过正常运动与异常边界安全。B4 只在相对 B3 达到单独的冻结增益时构成额外贡献；B4 失败不否定 B3，可回退为 B3 最终模型。

## 门 4：proxy 是否迁移到真实 OOD

方法完全冻结后，一次性打开 19 条真实异常。最终模型必须相对 B1/B0 达到冻结的真实 OOD 增益，并通过正常运动安全。

Gate 4 FAIL 表示当前研究周期的 proxy→real OOD 假设失败。不能继续使用同一 19 条数据调方法后仍称其为 untouched confirmation。

## 15.1 阶段式协议冻结

后续正式协议按以下批次一次性完成设计冻结：

1. E23–E26：placement/world builder；
2. E27–E37：renderer 机械链；
3. E38–E49：Gate 1；
4. E50–E71：STU、开发试验台与模型机械链；
5. E72–E84：B0/B1/B2/B3 与 Gate 2/3；
6. E85–E94：位置校准、B4、机制、安全和成本；
7. E95–E104：方法冻结、一次性确认与隐藏测试。

批次内仍按依赖顺序执行，但除非出现改变构念的新事实，不再逐节点重新发明样本和阈值。


---

# 16. 当前完整主线

$$
\boxed{
\begin{aligned}
&\text{STU 正常序列 206}\\
&\downarrow\\
&\text{经自标定验证的 OS1-128 规范射线与单发布回波接口}\\
&\downarrow\\
&\text{schema 7 合成异常代理 + 206 normal-control 模板}\\
&\downarrow\\
&\text{qualified support pool}\\
&\downarrow\\
&\text{连续落地、已观测正常几何拒绝、实体间碰撞拒绝、control 语义策略}\\
&\downarrow\\
&\text{不可变 WorldSpec 与逐帧确定性 renderer}\\
&\downarrow\\
&\text{同一 renderer 生成 normal-control 与 anomaly-proxy}\\
&\downarrow\\
&\text{201 control/proxy 资格与 rendered 标签捷径 Gate 1}\\
&\downarrow\\
&\text{五帧共享冻结 STU}\\
&\downarrow\\
&\text{128D 点特征 + 19D 官方分配一致语义证据}\\
&\downarrow\\
&\text{中心时刻坐标对齐}\\
&\downarrow\\
&\text{四级时间身份体素金字塔与按时间差局部注意力}\\
&\downarrow\\
&\text{所有可见点窗口级异常概率}\\
&\downarrow\\
&\text{B0/B1：代理监督价值}\\
&\downarrow\\
&\text{B2/B3：跨帧可识别增益与运动安全}\\
&\downarrow\\
&\text{B4：可选的重叠窗口概率融合}\\
&\downarrow\\
&\text{201 方法选择、阈值与 DBSCAN 冻结}\\
&\downarrow\\
&\text{19 条真实 OOD 一次性确认}\\
&\downarrow\\
&\text{Gate 4 PASS 后提交 51 条隐藏测试}
\end{aligned}
}
$$


---

# 17. 已明确废弃的路线

以下内容不属于当前第一版 AJAE：

- 当前帧 + 四帧历史的主从结构；
- 只预测中心帧或最后一帧；
- 显式对象适配器；
- 对象槽；
- 对象 ID；
- Hungarian 匹配；
- 跨帧对象跟踪；
- 对象级异常分类头；
- 对象分数再回投到点；
- 点身份 EMA memory；
- $L_{\mathrm{cf}}$；
- 显式点级平滑损失；
- 未经 query 分配控制的语义 evidence 求和；
- 五帧共同竞争单一 $K$ 邻域；
- 把中心对称五帧描述为实时在线方法；
- 把异常代理直接称为已经证明的真实 OOD。

---

# 18. 当前状态（当前工作区：E48与E49正式PASS，Gate 1关闭，E50为当前节点）

截至当前工作区权威提交，已经完成：

- E00–E15：规范射线身份与单发布回波接口；
- E16–E20：schema 7 几何、求交、连续尺寸、构造性连通、覆盖与简单捷径资格；
- E21-v4：train/206 qualified support-patch pool；
- E22-v2：连续落地与明显埋地资格；
- E23：已观测非地面正常几何的明显深穿透拒绝资格；
- 历史 E23–旧 E26：统一的 support-pool-only placement/world-builder 接口证据；其中旧 E25、旧 E26只适用于已经失效的旧normal-control分布。

E24 已完成两遍正式运行但未通过：512 个固定世界中 504 个构造完成，8 个世界因固定对象不满足 E22 连续最低支撑差值条件而耗尽 128 次支撑提议。已完成世界的最终实体对明显深穿透为 0，但该子集结果不能满足 512/512 的 E24 门槛。

E24-v2 保留 E24 的世界身份、实体数、全部碰撞规则和支撑提议，只把固定shape identity修订为固定确定性shape proposal stream。每个实体先拒绝不满足E22逐对象条件的shape，再将首个合格shape送入原E23与pair placement；每实体最多64个shape proposals，每个合格shape最多128个placement proposals。该修订不改变E24历史FAIL，也不把失败归因给pair-collision detector。

E24-v2 已按该合同完成两遍正式运行并通过：512/512个原世界全部构造完成，8个E22-invalid shape在support抽样前被拒绝，最终E22/E23 violation、明显实体对互穿、硬错误和两类耗尽均为0。E24历史FAIL继续保留。

历史 E25 已在旧随机放置control分布下完成两遍修后正式运行并通过：train/206实际可观察模板为car、truck、other-vehicle和person各64个；1,024/1,024个normal-control全部完成，类别—支撑违规、缩放错误、姿态错误、E22/E23验证错误、多实体fixture错误、硬错误和放置耗尽均为0。首次运行中6个模板因确定性表面射线错误地假定局部原点位于凸包内部而耗尽提议，该事件已归类为实现缺陷；修复只改变普通模板的表面采样射线内部点，没有改变E25科学判据。该结果不能替代E25-new。

历史 E26 已在旧normal-control分布下完成两遍24进程正式运行并通过：四类固定world各64个，256/256个world全部构造完成；world/report规范往返、E22–E25验证、类别支撑、姿态、材质、最终实体对、窗口遍历、缓存请求身份、单进程manifest重建、权威路径审计、硬错误和耗尽计数均为0，两遍逐元素一致。该结果永久保留，但不能资格E25-new后的正式生产world builder；当前Phase 2必须由E26-v2重新关闭。

E27 已完成两遍24进程正式运行并通过：256个真实normal-control凸包模板覆盖4个active类别、全部128束×2列槽位和2.5–50 m目标距离；target hit、解析miss、法向外向性和object ID错误均为0，最近距离、表面残差和法向单位长度误差均低于冻结容差。

E28-v1 已完成两遍24进程正式运行但未通过，历史FAIL保留并分类为 `protocol implementation defect`。v1 runner在几何求交后错误地继续执行return probability、随机接受和nearest competition；唯一表面失败项seed 2,800,127的原始 `ShapeSpec.intersect` 实际返回26.156862691941157 m，距离参考误差为 $5.316\times10^{-8}$ m。最终无穷距离来自固定随机数0.9999961987049697大于调制后回波概率0.9999864437684041，不是几何漏检。

E28-v2 已完成两遍24进程正式运行并通过：完整继承v1的256个fixture、seed、射线、姿态、距离、独立reference和容差，只把裁决接口改为直接读取 `ShapeSpec.intersect`。target hit、65,280条反向miss和法向外向性错误均为0；最近距离、表面残差和法向单位长度最大误差分别为 $1.226\times10^{-7}$ m、$1.171\times10^{-7}$ m和 $3.331\times10^{-16}$，两遍数组逐元素一致。E28关闭，E29解锁。

E29 已完成两遍24进程正式运行并通过：完整覆盖2,304个beam×range×incidence校准单元及55,296个固定身份决策；基础与材质调制概率、fallback、稳定身份均匀数和accepted mask相对独立reference的错误均为0，`p=0`、`p=1`及中间概率接受/拒绝分支全部覆盖，两遍数组逐元素一致。E29关闭，E30解锁。

E30 已完成两遍24进程正式运行并通过：原256个E27 normal-control fixture各展开24个固定frame身份，共6,144次裁决；accepted mask与E29独立reference零差异，6,137个接受项的点、强度和正常语义载荷全部有效，7个拒绝项均未生成回波载荷，两遍数组逐元素一致。E30关闭，E31解锁。

E31 已完成两遍24进程正式运行并通过：原256个E28-v2 schema 7 fixture各展开24个固定frame身份，共6,144次裁决；accepted mask与E29独立reference零差异，6,137个接受项均形成有限semantic-2点、强度和正确internal object ID，7个拒绝项均未生成载荷，两遍数组逐元素一致。E31关闭，E32解锁。

E32 已完成正式两遍运行并通过：小于tie和等于tie的fixture均保留native，2倍tie的fixture由inserted替换；遮挡mask、单回波、距离、标签和对象身份错误均为0。E32关闭，E33解锁。

E33 已完成正式两遍运行并通过：三个native前景fixture均保留原前景，距离、标签、mask、对象身份和单回波错误均为0。E33关闭，E34解锁。

E34 已完成正式两遍运行并通过：空slot的geometry接受、geometry拒绝和无geometry三类occupancy依次为true、false、false，原空slot intensity payload未制造占用，全部标签、身份和mask错误为0。E34关闭，E35解锁。

E35 已完成两遍24进程正式运行并通过：55,296次强度生成与独立reference最大误差为0，全部强度finite且位于train/206冻结支持，无未定义单元；边界clipping比例完整记录。E35关闭，E36解锁。

E36 已正式执行但未通过，分类为 `protocol design conflict`。三个传感器中间函数的静态label分支数为0；但冻结要求的“相同geometry/material/pose只改label”paired fixture与权威 `ObjectSpec` 合同冲突：normal-control必须使用 `NormalTemplateShape`，anomaly-proxy不得使用该类型，因此两个方向的仅label替换均被构造验证拒绝，paired trace无法执行。E37保持锁定。

E36-v2 已冻结且不修改 `ObjectSpec`：比较下移到几何后的传感器接口层，两个虚拟标签不作为被测函数参数，只在全部传感器中间量完成后写入最终semantic与mask。固定55,296个输入将比较概率、随机流、接受、强度、竞争距离和occupancy，并重新审计生产调用路径。E36-v1 FAIL继续保留。

E36-v2 已完成两遍24进程正式运行并通过：55,296个固定输入的全部标签写入前传感器中间数组逐元素一致，传感器函数与competition前label读取均为0，唯一差异严格限于最终semantic和两类mask bookkeeping。E36关闭，E37解锁；E36-v1 FAIL继续保留。

E37已按状态机冻结正式执行：从E26四类世界各固定32个，以中心帧100和101形成两个重叠五帧窗口；使用正式 `render_frame`、`FrameCache` 与 `FrameCacheKey` 比较串行/24进程、正序/逆序/随机顺序和cached/uncached结果，并单独审计跨world cache身份与window identity是否进入回波随机流。实现已经通过46项完整回归，正式资格结果尚未产生。

E37已正式通过：128个固定世界的9类slot对齐输出在串行/24进程、正序/逆序/随机顺序和cached/uncached路径之间零摘要差异；重复窗口请求逐bit错误0，跨world cache误命中0，window identity进入正式渲染与回波随机流的读取数0。E37关闭，E38解锁。

以下E38–E44、E45A和E45B结果均使用旧normal-control分布，只作为历史证据保留。E25-new形成新版正式control分布后，当时路线要求刷新E38–E44并执行E45A-new和E45B-v2；后续总览止损裁决已退休E45A，但旧E45B PASS仍不能满足当前E48前置资格，必须执行E45B-v2。

E38-v2已冻结为新版control分布下的统一Gate 1候选银行与单次共享渲染。train/201帧4–681的既有E21-v4支撑池不重建，但读取前必须精确核对SHA-256 `fc3646fbc145cdc29d2cf203835a3e0018bacbc6eaf714e091d21f7b93bfaf50`与帧、算法身份。runner从train/206流式重提取四类各64个、共256个未缩放规范模板，核对模板库SHA-256 `de5dfd765ac7d4fe4bb4644c40ecafdd80cdc31a3d0b6fc4fccd8e84a9fd906b`，并核对E25-new正式产物`runs/ajae/e25_new_normal_control.npz`的SHA-256 `30fc7d1ecd60d005cb18c60ac81b1c7335e2121fcd3f1da5f440b5387a747b19`。该E25-new产物只用于身份核对，不作为已缩放模板源。

E38-v2完整继承256个paired seeds `3800000–3800255`、real-normal单位选择、共享五帧身份、每world最多48个确定attempt与既有`attempt_seed=bank_seed+1,000,003a`。这个三来源共享候选银行固定为恰好256个单位，不设统一三方容量阶梯。每个attempt只调用一次`default_rng(attempt_seed+1).integers(0,256)`有放回抽取规范模板索引i，该control的指定距离层固定为i对5取模。runner先对完整201支撑池生成该模板与距离层的E25-new全局top-128流，再只保留support frame在real center±2内的原顺序子序列；不重排，不用全局第129项之后的行补齐。control与proxy复用该行序列，但各自位于独立反事实world。

normal-control继续执行E21支撑身份、E22连续grounding和E23已观测正常几何碰撞；只有完整正式renderer复核得到至少一个最终可见control回波，且其中位official range属于预分配距离层时才能接受。anomaly-proxy的schema 7、shape proposal、放置、材质、姿态和渲染规则保持不变。新候选银行保存三来源各自的support semantic，以及control模板索引、指定与最终距离层、最终观测和完整随机身份。新schema为`gate1-candidate-bank-v2`，旧v1候选银行必须拒绝，不得混用旧trace。

E38-v2的逐beam opportunity、return count/rate、entity-frame聚类、2,000次multinomial bootstrap、`SeedSequence([3801,2000])`、2.5%/97.5%区间与有限、守恒、三来源非零回波PASS判据均不变。单次权威渲染同时保存E39–E44需要的完整原始trace；E39及后续节点只读共享trace各自裁决，不重新计算几何或渲染。E38-v2使用24个进程、每进程数值库单线程，只正式运行一次，不自动重试。新候选银行与E38产物分别为`runs/ajae/gate1_candidate_bank_v2_256.npz`与`runs/ajae/e38_v2_per_beam_return.npz`。若后续两两共同支持不足，E45A-new与E45B-v2只能各自使用独立审计银行按512→1024→2048扩容；候选选择只读取冻结匹配协变量，不读取E46或E48输出。

E38-v2已经正式PASS。新版候选银行完成256/256个paired seeds，错误、合同错误与seed身份错误均为0；三来源各1,280个entity-frame groups，总opportunity分别为826,836、391,049和299,242，总return分别为598,736、385,263和295,250。计数守恒错误、共享trace合同错误和非有限错误均为0。候选银行与共享trace分别用时52.242514秒和44.695235秒；共享trace科学数组哈希为`30bc585de77e730570a942356d127858153350ac672bc6d39887b84381b770b1`，正式产物SHA-256为`914b185ae31d5509fa286208c26bb4271460d289a02ec398eaee715b7eeb7c9a`。按所有者决定只运行一遍，不声明逐元素复现。E38关闭，E39刷新解锁；来源泄漏仍未在E38裁决。

历史E38-v1及其统一候选银行当时冻结为：train/201帧4–681使用E21-v4同一支撑区域算法；首级候选银行使用256个paired seeds，每个seed分别生成独立正常对照世界与异常代理世界，并绑定同一真实正常实体五帧单位。E38-v1保存三来源逐entity-frame、逐beam的opportunity、return count/rate和2,000次cluster bootstrap区间。

历史E38-v1已正式通过：201支撑池包含1,193,969个合格区域；256/256候选seed完成并覆盖183个中心帧。三来源各1,280个entity-frame groups的逐beam opportunity、return count/rate和cluster bootstrap区间全部有限且计数守恒，两遍24进程逐元素一致。该PASS只适用于旧normal-control分布；逐beam来源差异没有在本节点转化为来源泄漏结论。

E39-v2刷新已冻结并由E38-v2 PASS解锁。它只读取SHA-256为`914b185ae31d5509fa286208c26bb4271460d289a02ec398eaee715b7eeb7c9a`的E38-v2共享trace，直接聚合五个距离箱的三来源opportunity、return count/rate和非零return entity-frame groups，不重新读取STU数据、构造世界、计算几何或渲染。正式命令为`python -m src.render qualify-e39-v2 --e38-artifact runs/ajae/e38_v2_per_beam_return.npz --output runs/ajae/e39_v2_per_range_return.npz`；只运行一次，不声明第二遍复现。

E39-v2已经正式PASS。三来源前四距离层均有非零return entity-frame group，覆盖错误、计数守恒错误和非有限错误均为0；real-normal在40–50米仍无观测，normal-control与anomaly-proxy在该层分别有127和119个非零组，均按冻结规则只报告。只读聚合用时0.000295秒，科学数组哈希为`348670e8aa9a8677f600aea55b825723d57d3246b64b3c83dc49bc3c64c29a1a`，产物SHA-256为`e7cea1574638db2f7e41799fe3855519ea57a47e9f6adc04f1a5a37e8aa526e0`。E39关闭，E40刷新解锁。

E40-v2刷新已冻结。它只读取E39-v2共享trace和冻结传感器标定，按source、beam与range bin一次稳定分组后计算条件五分位数、两两ECDF距离与生成来源clipping；不重新渲染，也不重复逐cell扫描整个回波数组。正式命令为`python -m src.render qualify-e40-v2 --e39-artifact runs/ajae/e39_v2_per_range_return.npz --calibration runs/ajae/calibration.pt --output runs/ajae/e40_v2_beam_range_intensity.npz`，只运行一次。

E40-v2已经正式PASS。1,279,249条强度记录的身份、E39-v2计数回算、有限性和生成来源冻结支持越界错误均为0，两类生成来源上下界clipping计数也均为0。单次统计用时0.142458秒，科学数组哈希为`240d204151f6bc9b913997a4809bcef45598384403907b3a9aedf7a17a681349`，产物SHA-256为`e197a309e20003411e760c3236316f2ca763947029bbbef52813fcb214ee6dc5`。E40关闭，E41刷新解锁。

E41-v2刷新已冻结。它只读取E39-v2共享trace中的两类生成来源native-empty、geometry、accepted和final-new整数计数，直接审计关系链与必要分支覆盖，不重新渲染；正式命令为`python -m src.render qualify-e41-v2 --e39-artifact runs/ajae/e39_v2_per_range_return.npz --output runs/ajae/e41_v2_empty_to_valid.npz`，只运行一次。

E41-v2已经正式PASS。两类生成来源的空槽机会、几何命中、回波接受和最终新增关系链违规均为0，normal-control与anomaly-proxy分别实际覆盖46和38个回波概率拒绝，最终新增均非零。单次统计用时0.017559秒，科学数组哈希为`a43887141dd8fb02dfe0a5291926acfba99a1e2d70487b5015f25faa7e2c5fd0`，产物SHA-256为`ac72fb803300c603fe081ec150da0f1e8cabefc778f14f6bb4e015becd71115c`。E41关闭，E42刷新解锁。

E42-v2刷新已冻结。它只读取E39-v2共享trace，沿用既有四个正可见层、五个距离层、两种support semantic及全部计数和覆盖判据，不重新渲染；正式命令为`python -m src.render qualify-e42-v2 --e39-artifact runs/ajae/e39_v2_per_range_return.npz --output runs/ajae/e42_v2_nvis_strata.npz`，只运行一次。

E42-v2已经正式PASS。normal-control与anomaly-proxy均覆盖全部四个正可见层，三来源共有14个support semantic×range bin×$N_{vis}$共同非空层；定义、计数、覆盖和初步匹配可行性错误均为0。单次统计用时0.000793秒，科学数组哈希为`149786c043dbbd438d9d5681aca42f4fc411d9190ffc066b963ded05ccad281f`，产物SHA-256为`af9dd78d1011fa566b5128a33584a8b23796b5f3d23252ca8a7b1823d95b9e84`。E42关闭，E43刷新解锁。

E43-v2刷新已冻结。它复用已通过的E37窗口身份与重复请求证据，只读取E39-v2新版五帧visible-return trace计算一次$N_{vis}$变化、V分层和出现/消失统计；不重新渲染，也不要求E39-v2第二遍。正式命令为`python -m src.render qualify-e43-v2 --e37-artifact runs/ajae/e37_world_frame_consistency.npz --e39-artifact runs/ajae/e39_v2_per_range_return.npz --output runs/ajae/e43_v2_temporal_visibility.npz`。

E43-v2已经正式PASS。window身份、重复请求、变化率有限性和定义错误均为0；三来源五帧可见计数、相邻变化率及出现/消失均已完整保存，出现和消失只作真实帧几何变化描述。单次统计用时0.000526秒，科学数组哈希为`e8f2fc993b95333db5cf2d79b2429bcccac0c9b0338610c3f9ac335c69bb580e`，产物SHA-256为`59d2e834b5b31770349faac591beb22067d87d4dbe1796b67ce857cb2aaf77a3`。E43关闭，E44刷新解锁。

E44-v2刷新已冻结。它只读取E39-v2共享trace，沿用既有遮挡率定义、三层边界、零分母无效策略及support×range×遮挡共同支持判据，不重新渲染；正式命令为`python -m src.render qualify-e44-v2 --e39-artifact runs/ajae/e39_v2_per_range_return.npz --output runs/ajae/e44_v2_occlusion_strata.npz`，只运行一次。

E44-v2已经正式PASS。normal-control与anomaly-proxy均覆盖全部三个遮挡层，三来源共有12个support semantic×range bin×遮挡共同非空层；定义、计数、覆盖和初步匹配错误均为0，17个proxy零分母单元按冻结规则显式保留为无效。单次统计用时0.000673秒，科学数组哈希为`cc874669d7e61732e894f1c9993fa97ac10a2a649f6111465ad34d618c1c4e03`，产物SHA-256为`49880d3b48024a20fe1c2a3155424daf29e8690407dd56437b894097ce464695`。E44关闭，E45A-new与E45B-v2解锁。

E45A-new已经按独立512→1,024→2,048银行阶梯完整执行并正式FAIL，分类为`insufficient_pairwise_common_support`。三级最大匹配分别为14、30和63对；最终只覆盖29个real侧中心帧，四个2.5–40米距离层计数为[1,46,16,0]，未达到1,024对、100帧和四层非空门槛。最终315条合法边的caliper错误与重复使用均为0，五项SMD为[0.083301,0.024199,0.007985,1.778776,0.044545]，最大值为遮挡SMD 1.778776。正式科学数组哈希为`6fa5f901574f5a621633d60bda50037fcb261a136caa4e2f1ae0beada02d1426`，产物SHA-256为`acad2f28c4f2cb47314206671bbfebbdc89004a81cd1c403fc33af15c5dfda21`。该结果没有执行E46来源分类，也不裁决renderer失败；E46当时保持锁定。状态机在该时间点停在E45A-new正式FAIL等待新决策，E45B-v2尚未启动，且没有自动扩容、重试、放宽caliper或修改生成分布。

课题负责人随后作出过渡决策：E45A-new正式FAIL永久保留，其设计层归因为`qualification specification defect`。当时的主线不扩充候选银行，不修改E25-new、E26-v2、renderer、sensor calibration或E38–E44，也暂不执行E45B-v2。E45A一度改为E45A-overlap，只读现有2,048容量unit cache，在support semantic×range bin×45° azimuth sector×occlusion stratum双方共同非空cell内，对原五项协变量执行确定性共同重叠加权。PASS事前要求real/control各自ESS至少256、real侧至少100个center frames、五项加权SMD和加权KS均不超过0.10，并要求两次完整计算逐元素一致。该过渡设计当时规定E45A-overlap PASS后才解锁E46；E46的模型、特征、分组交叉验证、bootstrap和0.65判据保持不变，只把point权重从$1/n_{unit}$改为$w_{unit}/n_{unit}$。

E45A-overlap已经正式执行并FAIL，分类为`scientific_failure: insufficient_effective_overlap`。现有cache中有43个双方共同exact cells，保留1,491个real-normal与6,081个normal-control单位，分别覆盖297与338个center frames。两次完整加权逐元素一致；五项加权SMD最大为$5.513\times10^{-7}$，五项加权KS最大为0.060849，均通过0.10门槛。real/control ESS分别为207.526586与232.335050，均低于事前冻结的256，因此E46不解锁。该FAIL说明名义共同支持和frame覆盖充足、协变量可以被强平衡，但达到该平衡的有效总体过薄；它仍不构成renderer source fingerprint证据。正式产物为`runs/ajae/e45a_overlap_weights.npz`，SHA-256为`90f60e2432975dc8aa0aea6c5fc1e90b463b0318d9b61add362ccb30227bf1a6`，科学数组哈希为`e54eb12599b3887e2a73a5e22200dc277aa47b1a1839a0ebb8a3acd4f0ac3bfb`。

课题负责人随后从AJAE总览作出止损裁决：E45A系列已经把来源风险过度具体化为natural real-normal与coverage-oriented inserted control的观测分布近似等价，而遮挡、距离、密度、support和azimuth差异主要混合了placement与scene composition，并非纯renderer属性。E45A-new与E45A-overlap的正式FAIL和全部历史证据永久保留，但不再是Gate 1硬门，也不要求修改E25-new。停止E45A-D1、transport feasibility、ESS/sampling optimization、E25 distribution-v2、更大control bank、新caliper、新matching、新weighting和target-conditioned generator。E46降为可选非阻断来源诊断；其可分性不能单独推出异常标签捷径。Gate 1硬门由已完成的renderer机械/传感器证据、E45B-v2 control/proxy资格和E48标签捷径审计组成。该裁决当时把正式顺序收敛为E45B-v2、E48、E49；三者现均已PASS，Gate 1已经关闭，当前进入E50–E71并尽快回答$B1>B0$。

E45B-v2已按提交`eaedefb38e7e3f0eca7c02970d5ad4e3d1f181a5`冻结的单次正式身份执行并PASS。容量512得到469对、123个control侧中心帧和四层计数[82,119,137,131]，因配对数低于1,024而按冻结阶梯继续；容量1,024得到1,347对、248个control侧中心帧和四层计数[348,287,370,342]。最终8,296条合法边覆盖40个精确分层，caliper错误与重复使用均为0，五项SMD为[0.020004,0.061885,0.007586,0.010555,0.014482]，最大值0.061885低于0.10。独立只读复核确认种子、输入身份、候选银行与单位缓存科学哈希、来源对[1,2]、三项精确分层、五项caliper、无复用、帧与距离覆盖、SMD和最终科学数组哈希全部一致。正式产物为`runs/ajae/e45b_v2_control_proxy_pairs.npz`，SHA-256为`19ecbc843cc5325e3f12497c50e5855388f0f5caa581179f6fd6639613a8ecfd`，科学数组哈希为`735df664e6ea2f54cac7f3d0c9a9778b17f035259cf716686063f30b5c31eaca`。该PASS只建立E48所需的rendered control/proxy可比较总体，不裁决低层标签捷径；E48现已解锁。

E48在任何模型结果产生前完成版本化拆分规则补全。E45B-v2的1,347对只共享294个center-frame身份；若同时把pair两端放在同一fold并禁止任何center frame跨fold，传递闭包会形成一个包含1,334对的巨型连通分量，无法形成有效五折。补全规则固定以`E48-center-v1:{frame_id}`的SHA-256前8字节小端整数对5取模决定frame fold。对测试折$k$，仅当pair两端frame fold都为$k$时进入test；两端都不为$k$时进入train；其余pair在该折排除。五折test/train/excluded对数固定为[61,46,91,74,97]/[913,1004,800,861,832]/[373,297,456,412,418]，共369个不重复OOF pairs。该规则不读取特征、标签表现或模型输出，不修改E45B匹配。bootstrap相应冻结为2,000次matched-pair cluster resampling，`SeedSequence([4800,2000])`；每次抽中pair时同时带入control/proxy两端。E48的两个模型、七项低层输入、64点上限、unit等总权重和近饱和FAIL门槛均不变，不执行任何feature ablation或attribution。

E48随后从提交`29f152fb4cfd871b74c24da17316945ea6c40fc0`完成唯一一次正式运行并PASS。逻辑回归的AUC与平衡准确率为0.513600和0.508942，对应95%下界为0.497434和0.495379；深度3决策树为0.505187和0.500505，对应下界为0.486033和0.487522。两者都远低于同时要求AUC下界至少0.95、平衡准确率下界至少0.90的FAIL条件。只读复核逐元素重建了折身份、369个OOF pairs、点估计和2,000次matched-pair bootstrap；未运行特征消融或归因。正式产物`runs/ajae/e48_low_level_shortcut.npz`的SHA-256为`b55ad1c7fecf030f4f3f22c5ba4423f1cfeaae46a4d763565ab22c97ad6206ce`。

E49没有新增统计实验。合取裁决逐项核对E27、E28-v2、E29–E35、E36-v2、E37、E38-v2–E44-v2、E45B-v2和E48共20个当前产物，文件均存在且SHA-256与权威记录一致，所有对应当前节点均为PASS；E45A全分支、E45-V1、E46和E47按冻结规则排除。因此E49正式PASS，Gate 1关闭，E50解锁。该结论只说明渲染机械资格和直接低层标签捷径门已通过，不说明代理监督已经有效；后续仍须由E50–E71和$B1>B0$回答。

E50的第一次执行实现错误地把E53–E54才涉及的查询分类、掩码与派生证据纳入“完整编码哈希”，超出了E50只验证`point_features_head` 128维接口的原协议。该错误实现产生的文件SHA-256为`8fa2425bf7645390592d2709fb15dba86c9c08780518afc59ee4ed88c33515f4`，不作为正式证据，也不构成E50 FAIL。直接修正原实现后，E50从提交`1368c94d0514c532208b765bb0faea6cbd9a4b97`完成有效正式运行并PASS。32帧共3,835,507个真实回波全部得到有限的128维逐点特征，覆盖3,438,697个稀疏体素，形状、有限值、梯度、身份和重复性错误均为0；两遍的32个逐帧特征哈希全部完全一致。独立只读复核重新计算了产物科学数组哈希与全部汇总量。正式产物为`runs/ajae/e50_stu_features.npz`，SHA-256为`2c2d8507df0f9e4c9984118e59c6d65a8f13835590fee5b51bed02c282c5671a`，科学数组哈希为`c698f3b53d6a38f579f45fdfb4f06023f3b433d65a9f61e87c386dc2ac9090f2`。E50只资格128维特征接口，不资格后续查询证据；E51现已解锁。

E51从提交`867abd01071ae18e28d8aa2623363564de97d6d2`完成唯一一次正式运行并PASS。它继承E50的32帧身份和通过产物，使用不依赖MinkowskiEngine的NumPy实现独立计算`floor(x/0.05)`体素坐标、首次出现的稀疏行顺序和逐点逆索引，再分别核对MinkowskiEngine直接量化结果与冻结编码器实际逆映射。在4,194,304个文件槽位中，3,835,507个真实回波全部恢复到合法稀疏行，358,797个零槽位全部排除；独立坐标、唯一索引、直接逆映射、编码器逆映射、槽位、范围和恢复错误均为0，两遍32个映射哈希全部一致。只读复核重新计算了全部汇总量与科学数组哈希。正式产物为`runs/ajae/e51_inverse_mapping.npz`，SHA-256为`bca33539ea2c3cb9d815cc4586d98fc356f134d40351e63cbb8d2e1c256ccafa`，科学数组哈希为`02e22e4cc87f5f2bde0f84712fe624930ef2a6a8f52c7b0752fdc1c800f4fee2`。E52现已解锁。

E52从提交`2d13e4a398e8d7d8f34558aff5c59c70c4b7fdb5`完成唯一一次正式运行并PASS。32帧中共有284,441个多点共享稀疏体素，覆盖681,251个原始点；共享点的128维稀疏特征完全一致，但源帧、原始槽位、已校准规范射线、XYZ、强度、原始标签和共享身份碰撞错误均为0。四点解析反例保留四个最终输出位置，两遍输出完全一致，固定点置换后的输出也逐元素同步置换。只读复核重新计算了汇总量、置换关系与科学数组哈希。正式产物为`runs/ajae/e52_raw_identity.npz`，SHA-256为`2e519c358133cb03fbbbafed82062906eceec071279da0149b2e6a1eac1c9a69`，科学数组哈希为`2e8d2a67071b383606cdee1017406d7142d127b9a5fb6915a66ac92964249330`。E53现已解锁。

E53从提交`50361d6e17d6da6783d9e028c7dd580b35346be8`完成唯一一次正式运行并PASS。实现只为审计暴露编码器实际选择的查询编号，不改变查询分配、证据公式或任何AJAE输入。4个身份固定CPU进程对32帧各运行两遍，共覆盖3,438,697个稀疏体素；每帧活跃查询数为30–49。独立从官方`pred_logits`和`pred_masks`重算后，查询身份、19维证据、分配可靠性、无对象可靠性、最小编号并列规则和两遍复现错误全部为0。只读复核重新计算了汇总量、哈希一致性与科学数组哈希。正式产物为`runs/ajae/e53_query_assignment.npz`，SHA-256为`e39511b76aec4c90b6d77d22b9d5f89d57184873ddc495677c8e786ffb476a03`，科学数组哈希为`4d079db8fd7470298333dca366eaed1c5bc552bb4e435b40d35bb87708e38145`。E54现已解锁。

E54从提交`0676cf41634bd6ec3cc0b8a4f732131b287b8c9f`完成唯一一次正式运行并PASS。它继承E53的32帧、确定性CPU路径、逐帧随机身份和4进程×6线程布局，在官方float32张量上另行写出冻结公式，再分别核对实际体素输出与实际逆映射逐点输出。3,438,697个稀疏体素和3,835,507个真实回波的六类最大绝对误差均为0，`1e-7`超限、广播、有限值、梯度和两遍复现错误全部为0。只读复核重新计算了汇总量、哈希一致性与科学数组哈希。正式产物为`runs/ajae/e54_evidence_reliability.npz`，SHA-256为`67187b039bdafbea0d8f728a017daea043c2fdb6f7a6c7754da3998fa6173dac`，科学数组哈希为`53fd0985ec912a879c53d159acf73dda9e61843b5cd9c3f9a9524df4d3ccc651`。E55现已解锁。

E55从提交`64fd3fd3138576cbf463b40b08d0c1fb9a57c28d`完成唯一一次正式运行并PASS。train/206中心14与train/201中心16两个真实五帧窗口分别含625,129和514,296个真实点。实际基础内容严格为128维STU特征、19维正常证据、分配可靠性、无对象可靠性和强度组成的150维，五个时间位均非空；实际内容层、坐标层和时间嵌入钩子全部逐元素对齐。模型签名未包含查询编号/令牌、熵、能量、MSP、实例编号、移动标签、生成器家族及生成诊断字段。字段、模式、签名和两遍复现错误均为0。只读复核重新计算了点数、五时位计数、哈希一致性与科学数组哈希。正式产物为`runs/ajae/e55_actual_input.npz`，SHA-256为`13d367fa0f7f0ed86ba6de24fc535df44e4ea90ab6f38989dec4ea4d6e35aaf8`，科学数组哈希为`68cdfb42f5c8a533d19c4d92302fa4372e46d56030392229b81217be43bf533a`。E56现已解锁。

E56从提交`5568529bfdb4ae8770d260aed8ad2dfb4986d151`完成唯一一次正式运行并PASS。全部32个真实五帧窗口共比较12,601,562个非中心静态点，窗口中位数的中位数由0.115999米降至0.043199米，窗口Q95的中位数由0.587486米降至0.250396米。解析刚体夹具误差为0；95条匹配移动轨迹仍保留运动，最大位移7.492775米。矩阵方向、帧身份、非有限值、改善条件、运动保留和两遍复现错误均为0，独立只读复算通过全部门槛。正式产物为`runs/ajae/e56_coordinate_alignment.npz`，SHA-256为`2314f65af4db7bb7df79d319d07a14e11fc81bb9d3121df4399ddbffd7d41702`，科学数组哈希为`670404303bf03120f62edb19d84b1f98fd9279e540612cd3934cde9b6debf04e`。E57现已解锁。

E57尚未执行时，Phase 6完成一次结果前协议瘦身并版本化为E57-v2。E57仍冻结24个不读取模型结果的train/201混合开发世界；选择器只使用control/proxy各自的`Nvis`、遮挡率、距离和五帧可见数，以秩归一化后的确定性最大最小距离选择保留适度条件跨度，不优化精确分箱配额。硬条件仅包括24个合法且中心帧可评价的混合世界、每类至少12个世界存在`V>=2`实体、完整的world/entity/frame/canonical-ray身份以及train/201无梯度隔离。原E59的距离、`Nvis`、遮挡率分箱和E60的`V=1,...,5`分层全部保留为完整描述统计，但不再具有独立FAIL或阻断B1的权限。E58只检查六个held-out torus世界的身份与隔离；E61、E62、E63的安全集、官方评价器等价性和结果前训练/统计规则冻结职责不变。E64–E71作为一个机械资格包执行，夹具通过后不派生新的资格支线。

E57-v2从提交`c65b946451df17ebe2a32cf56f7b57bf7d85c3d6`完成一次正式运行并PASS。1,024个冻结候选中有113个满足合法、五帧可见和中心帧可评价条件；固定选择器产生24个世界且复现错误为0。control/proxy分别有23/24个世界存在`V>=2`实体，均超过每类12个世界的冻结下限；中心帧有效异常点与正常点的最小值分别为6和60,235。独立只读复核重新解析并重渲染全部24个世界，另行重算秩空间最大最小选择，逐项复现描述量、点数、身份和哈希。正式产物为`runs/ajae/e57_development_worlds.npz`，SHA-256为`b14efc1aad86ac67b5bf7c8631f02b2e68664e071b747b7b210d5f7a30f5d123`，科学数组哈希为`590c467da2dec0a161688f2587dc1c37cea2b0f42f326b9918fd6dc9df81f6ec`。E58现为正式节点。

E58正式执行前的无产物核验发现并修正了两项实现缺陷。旧通用表面取样器以包围球射线指向几何中心，隐含了几何相对中心为星形的假设；torus存在中心孔，因而会被误报为表面射线缺失。唯一权威取样器已对`HeldOutTorusShape`改用解析双角参数化表面点，测试确认点到torus有符号距离在`1e-12`内为0。旧替换实现还把派生的torus几何种子写入`WorldSpec.seed`，从而同时改变了control和proxy的逐槽传感器随机流。修正后，torus种子只决定几何，替换世界严格保留源世界的`seed`与`source_sequence_id`；control、材质、对象身份、朝向和支撑接触不变，proxy只改变形状及用于保持接地的平移。测试确认两个传感器通道的逐槽随机数与源世界完全一致，同时替换世界具有独立的世界和缓存身份。这些都是首次正式运行前的实现修正，不构成E58结果或协议修订。

修正后的E58完成一次正式运行并PASS。24个确定性torus替换候选中20个满足资格条件，固定身份哈希规则选出6个世界；资格、选择复现、语义身份、逐槽传感器随机流、缓存身份和训练采样器隔离错误均为0，中心帧最少包含34个有效异常点和59,995个有效正常点。独立只读复核重算科学数组哈希和选择索引，重建并重渲染全部6个世界，逐项复现源身份、两个传感器通道的随机流、五帧诊断与中心点数。正式产物为`runs/ajae/e58_held_out_worlds.npz`，SHA-256为`cde17c339b5307de5f21c9ceeb9b207ad26a12026e2fe741f6926e1af8a8110b`，科学数组哈希为`125bb629f8449b8fb85a5de98ff29ef5c0f3c18b01a8b8eca94ede86afdd9969`。

E59/E60随后通过一次共享的只读聚合完成，直接使用E57保存的五帧描述量，没有重渲染或重选世界。control/proxy的距离分箱分别为`[6,2,7,9]`/`[5,4,8,7]`，可见点数分箱为`[6,2,9,7]`/`[3,9,7,5]`，遮挡率分箱为`[22,1,0,1]`/`[19,3,1,1]`，$V=1,...,5$分层为`[1,0,0,0,23]`/`[0,0,0,0,24]`。独立复核从E57保存的诊断JSON重新构造全部24×2个记录并逐项复现。E59/E60是描述性完成态，没有科学FAIL权限；当前正式节点为E61。

E61在任何模型分数出现前完成版本化协议补全。独立静态正常安全集使用train/201帧4–681中距离位于2.5–50米、语义非0、非2且不属于252–259的全部48,828,507个点。运动正常安全诊断使用train/206帧0–448中语义为252–259的全部13,011个有效范围点；由于206参与模型训练，该证据只能解释为标签盲的训练域运动子群安全约束，不能声称未见运动泛化。运动与静态对照仅按对应静态语义类别和四个距离层进行确定性无放回匹配，每格两侧取候选数较小值；全部运动点始终进入正式运动误报率，匹配覆盖与两组差异只用于描述运动效应，不能单独FAIL E61。匹配不读取强度、遮挡、密度、STU特征或任何模型分数，也不要求同帧。

E61随后完成一次正式双构建并PASS。它冻结48,828,507个静态正常点、13,011个运动正常点和6,756对确定性运动/静态解释对；两次构建逐元素一致，身份、计数、预测访问、标签输入和复现错误均为0。独立复核直接从原始帧重建全部位掩码、语义×距离候选格、SHA-256排序和配对，完全复现正式产物。匹配覆盖率为51.9253%，只限制运动效应解释，不影响全量运动安全约束。正式产物为`runs/ajae/e61_safety_identities.npz`，SHA-256为`8d3e08e0512dc70a75d2279cfb4515bc960bbfda4f35a872c4a76e9dad69d0e0`，科学数组哈希为`5227e6a6e6c807200373bf64ae947c7eb2634f09131277620e2c12fd91a85e31`。E61结束后进入E62；E62现已PASS，当前正式节点为E63。

E62执行前审查发现现有协议只规定了自定义评价器与官方评价器的比较指标、样本筛选语义和数值容差，却没有冻结解析夹具的具体数组，也没有定义“一份固定真实预测”的数据来源、帧或世界身份、模型检查点和产物哈希。官方评价器源码已经定位并固定到`stu_dataset`提交`8f0f09c2ca4bf7b665e0ae5919b4092ddae140a2`，脚本SHA-256为`ed0330f80fbd3cd4cefafed33d6c747c51f2de521ef191e2868eb24f84b9ce61`；当前仓库没有可直接继承的冻结预测产物。train/201和train/206不能提供含语义2异常点的真实数据正例，公共19序列仍须封存。因此E62尚未执行，也没有FAIL；当前需要先由研究负责人版本化补全证据身份，不能由执行代码自行选择预测来源。

研究负责人随后批准E62-v2最小协议补全。E62只验证AJAE自定义评价器与官方STU评价器的样本选择和指标数值是否等价，不验证模型性能或真实异常泛化。旧的“一份固定真实预测”要求被删除，正式输入改为一套冻结解析夹具和一套10帧×96点的冻结非符号构造数值夹具；两套夹具都不得读取公共19序列、隐藏51序列或任何真实异常数据。解析夹具覆盖float32距离边界、语义0忽略、筛选后4/5异常点帧门槛、全相同与重复分数，以及TPR恰为0.95和首次严格超过0.95的阈值行为。任何E62差异只能判为评价器或比较入口实现缺陷，并在同一冻结夹具上修正重跑，不能解释为AJAE、数据或Gate 2失败。

E62-v2夹具随后在未调用任一评价器的条件下完成冻结。解析套件包含4个独立案例、5帧和90点，构造数值套件包含10帧和960点；其中首个数值帧在距离与忽略筛选后只有4个异常点，其余9帧均达到官方接受门槛。冻结产物`runs/ajae/e62_evaluator_fixtures.npz`为19,270字节，SHA-256为`b7f2a267aebdf6b092ba65a8edb2bd280aab22fddc02b792be6d306318ccb712`，科学数组哈希为`fa77d594151a6cdb2f69a9f7f26965c493b223e4970a6736e6428a60a1de78ca`。后续比较入口只能读取该产物，不得重新生成或改变数组。

E62-v2正式执行并PASS。五个冻结案例中，官方与自定义路径共同接受13帧、跳过2帧，最终纳入864个有效点，其中119个异常点、745个正常点。接受/跳过帧身份、入选点身份、三类计数、汇总标签和分数均完全相等，AP、AUROC、FPR95和官方暴露阈值的最大绝对差为0。独立只读复算再次复现全部逐案例结果和科学数组哈希。该结论只建立评价器等价性，不包含模型性能或真实异常泛化证据；当前正式节点推进到E63。

E63-v2已经正式PASS。固定开发评价使用E57冻结的中心目标帧，并对所有B0–B5统一取完整`[-4,+2]`源帧域；身份清单保留23个世界，只排除`world_id=5`，因为中心帧6缺少源帧2和3。检查点按宏平均逐世界AP选择，差异小于`0.001`时依次使用更低开发集FPR95、更低纯正常交叉拟合误报率和更早检查点破平局。安全折由`E63-safety-crossfit-v1`对原始24个不可变世界身份哈希排序后固定为12/12，与共同域取交集后为11/12。5,000次层级配对bootstrap使用NumPy PCG64种子`63002026`，每次有放回抽取3个训练种子和24个世界，并对比较模型共用同一实现抽样。正式双构建与独立只读复算逐元素一致，身份与复现错误均为0；产物`runs/ajae/e63_training_freeze.npz`的SHA-256为`5dbf99eaa59a05a83774e42beb6b8d7a95cf9309ebd42ab7870604a20d410dd9`，科学数组哈希为`e0df86313f27524fba9ed1d2bc563d94def568d36925c184f13e41a72540d207`。Gate 2、Gate 3、可选B4贡献、Gate 4和0.03绝对安全恶化界均已机器化，B4已从Gate 3删除。整个过程未读取模型结果、六个E58留出世界、19条公开真实异常序列或51条隐藏序列；当前节点推进到E64。

E39已正式通过：三来源在2.5–10、10–20、20–30和30–40米均有非零return entity-frame覆盖，逐实体帧计数守恒错误0、非有限值0，两遍逐元素一致；三来源在40–50米均无观测，按冻结规则仅报告。共享trace已保存1,656,861条逐返回强度及E40–E44所需计数。E39关闭，E40解锁。

E40已正式通过：1,656,861条强度记录的身份、分箱、有限性与E39计数回算错误均为0；两类生成来源的206冻结支持越界和上下界clipping均为0，两遍统计逐元素一致。E40关闭，E41解锁；条件分布差异继续只作描述。

E41已正式通过：两类生成来源的空槽机会、几何命中、回波接受和最终新增关系链错误均为0，均实际覆盖最终新增与回波概率拒绝分支，两遍统计逐元素一致。E41关闭，E42解锁；来源间比例差异继续只作描述。

E42已正式通过：每来源1,280个entity-frame的定义和计数守恒；normal-control与anomaly-proxy均覆盖四个正可见层，三来源共有12个共同非空的support semantic×range bin×$N_{vis}$层，两遍逐元素一致。E42关闭，E43解锁；完整匹配仍由E45裁决。

E43已正式通过：跨窗口身份和重复渲染错误均为0，三来源的相邻帧$N_{vis}$变化率、$V=0,\ldots,5$分层及计数全部有限且守恒，两遍统计逐元素一致。少量出现与消失按冻结规则只作真实帧几何变化描述。E43关闭，E44解锁。

E44已正式通过：normal-control与anomaly-proxy均覆盖三个冻结遮挡层，三来源共有9个共同非空的support semantic×range bin×遮挡层；4个零分母单元以显式无效掩码保留，其余定义和计数全部有效且守恒。E44关闭，E45解锁；完整匹配由E45裁决。

E45正式FAIL，分类为`scientific_candidate_domain_failure`。E38冻结定义下完整train/201 real-normal候选宇宙包含1,635个实体和8,175个entity-frame，五个距离箱的严格上界为[2,492,4,141,1,457,85,0]；相对最低要求[128,128,128,128,32]，30–40米短缺43，40–50米短缺32。容量银行只是该完整宇宙的子集，不能修复必要覆盖，因此扩展银行、完整匹配与E46均未执行。

E45-v1正式FAIL永久保留；后续设计归因修订为`qualification specification defect`。E45-v2在不改变真实对象候选、train/201来源和全部严格匹配条件的情况下完整运行至2,048容量，正式FAIL，分类为`insufficient_three_source_common_support`。最终只有58个triplets、34个real侧center frames，四个可观察距离层计数为[0,51,7,0]，最大pairwise SMD为1.509987；caliper错误、重复使用和复现错误均为0。E46保持锁定。

E45-v2正式FAIL永久保留；后续设计层归因修订为三方审计设计失败。E46所需的real-normal与normal-control、E48所需的normal-control与anomaly-proxy当时拆为E45A和E45B两个独立匹配集。两者完整复用2,048容量冻结单位缓存、train/201来源、real-normal定义、2.5–40米域、精确匹配条件和五项caliper；使用完整合法边上的确定性最大基数二分匹配，并在最大基数固定后最小化归一化协变量平方差。历史E45A正式FAIL：完整合法图778条边，最大匹配135对、73个real侧center frames、四个距离层计数[11,107,17,0]，最大SMD为1.000399；caliper错误、重复和复现错误均为0，E46在当时保持锁定。历史E45B正式PASS：完整合法图29,156条边，最大匹配3,624对、357个normal-control侧center frames、四个距离层计数[1,133,1,877,563,51]，最大SMD为0.031652；caliper错误、重复和复现错误均为0。该前置资格只适用于旧control分布；当前E25-new分布已经由E45B-v2正式PASS资格化，E48现已解锁。

E45A-v2作为Gate 1审计专用定向control银行完整执行至每目标64个proposal的冻结上限，正式FAIL，分类为`targeted_control_common_support_failure`。各阶梯合格control数为[13,36,83,170,325]，最大匹配数为[13,36,80,148,212]；最终只有212对、90个real侧center frames，四个距离层计数为[1,139,72,0]。五项SMD为[0.099364,0.159312,0.064798,0.882238,0.021068]，最大值为遮挡SMD 0.882238。caliper错误、重复使用、硬错误均为0，两遍匹配逐元素一致。该审计银行没有修改E26、renderer或正式normal-control训练分布，也没有使用E46分类器结果。正式产物科学数组哈希为 `00aed2338732f9a9233547cae52c1c3087df6cfb5294da664a73a7b33a0c6192`，文件大小756,236字节，SHA-256为 `290747b6c01ec9d2af152e8688f51cc9c966690cb5c165279265a51fc30e0405`。其历史停止条件先由E25-new合同取代，后来整个E45A分支又被总览止损裁决永久终止；E46现为可选非阻断诊断。

用户已据此授权修改正式normal-control位置生成。E25-v2按train/206真实正常观测引导的位置proposal与拒绝采样完成单次正式执行并FAIL：254/256完成，两个other-vehicle模板耗尽128个目标proposal；接受目标只覆盖27个真实实例且40–50米层为0。正式产物永久保留。结束后的独立只读审核进一步确认，4,827个目标中1,865个参考支撑距真实实例二维凸包超过0.5米，最大26.110105米；person的234个目标全部超过0.5米，40–50米层117个目标中111个超过0.5米。该缺陷来自`_real_instance_support_row`在没有近邻支撑时执行无最大距离限制的最近支撑回退，并已传播到106/254个完成control所引用的环境目标。因目标的support semantic和后续support proposal顺序受错误参考支撑影响，当前E25-v2不能裁决正式normal-control构念是否可行。256个真实模板、缩放、类别、姿态、材质、传感器与schema 7 proxy均未改变；旧E25、E26和E45B结果仍只对旧control分布有效。E26-v2、E46和E48保持锁定，后续目标库精确支撑关联定义等待新的课题负责人决策。

课题负责人随后版本化E25-v3：类别合法支撑语义保持不变；目标帧优先，其后严格按$f-1,f+1,f-2,f+2$扩展；所有候选使用支撑锚点到目标帧真实实例世界XY凸包的二维欧氏距离；禁止半空间0.5米代理与无上限最近邻回退。最终距离边界采用有限的尺寸感知形式，但唯一$D_{xy}$定义和$\alpha$尚未冻结。为此完成了一次纯只读train/206诊断，没有运行生成器、读取train/201或改变caliper。4,827个保留目标在五帧并集中均有合法语义支撑；统一0.5米只覆盖2,962个，分类别为car 2,568/3,888、truck 304/571、other-vehicle 90/134、person 0/234。person最近距离minimum/median/$Q_{0.95}$/maximum为1.197818/1.581225/4.593703/11.937300米，其可见XY凸包直径median/$Q_{0.95}$/maximum为0.530211/0.721785/1.230407米。全体最近距离$Q_{0.75}$/$Q_{0.90}$/$Q_{0.95}$/$Q_{0.99}$/maximum为2.114124/5.550483/10.245840/17.295375/26.110105米。诊断同时逐目标保存凸包直径、轴对齐包围框对角线和最长跨度，以及三者代入候选公式时的最小$\alpha$，但没有选择其中任何一种。产物为`runs/ajae/e25_v3_support_observability.npz`，大小574,508字节，SHA-256为`3d68b829f644540d6ca0392b6dac6b2a907c153b2f6b646c3c783ed9d4f40014`，科学数组哈希为`fcf62d86f7be1e90392135c37e3d3dd6e6d3d3db5e792aee48cdd0b29cd51947`。

课题负责人根据该诊断放弃$D_{xy}+\alpha$距离门，不再选择这两个量。当前E25-v3改为只读支撑平面相容性诊断：仍按目标帧、$f-1,f+1,f-2,f+2$搜索类别合法的E21-v4 patch，帧内按精确锚点—凸包距离与冻结哈希排序，但距离只决定顺序和描述，不作为门槛。每个候选必须在目标XY凸包上满足E21小/大尺度预测高度最大差不超过0.08米，并使目标真实可见点中低于中央平面超过0.02米的比例不超过0.02。两项界限分别直接继承E21与E22，不根据本次结果调整。最低可见间隙、可见高度范围、平面坡度、锚点距离及其相对E21中心半径均只报告。诊断不运行生成器、不读train/201、不改E45A caliper；结果用于判断四类、五个总体距离层和三个总体遮挡层是否仍有目标，E26-v2、E46与E48继续锁定。

该只读诊断已在运行前提交`5b1b0f4`下用24进程完成，用时3.293789秒，不形成PASS/FAIL。两项相容性条件保留4,825/4,827个目标：car 3,886/3,888、truck 571/571、other-vehicle 134/134、person 234/234；五个总体距离层为[866,2115,1147,580,117]，三个总体遮挡层为[1182,3387,256]。121,360次实际候选评估严格分解为55,892次外推稳定性拒绝、60,643次可见几何拒绝和4,825次接受，所有接受项逐项满足冻结的0.08米与0.02/0.02界限。

结果同时否定了“仅靠这两项即可确定可信局部支撑平面”的充分性。接受项中1,042个所选锚点距目标凸包超过5米，461个超过10米，47个超过20米，最大67.849019米；person距离中位数为3.378346米、最大20.628190米。341个接受项的最低可见点到预测平面的正间隙还大于物体全部可见点沿法向的高度范围，最大最低可见点正间隙为5.282527米。现有条件能排除多尺度外推不一致和明显切入可见物体的平面，但不能裁决局部空间有效域或过大的向下外推。目标库因此没有冻结或重建，生成器没有运行，train/201没有读取。

课题负责人随后最终冻结E25-v3可信局部支撑定义，不再设计$D_{xy}$、$\alpha$或经验距离常数。E21-v4对每张patch已经在最大半径$1.25R(d)$内完成局部验证，其中$R(d)=\operatorname{clip}(d/20,1,3)$米；因此只有锚点到目标世界XY凸包的精确二维欧氏距离不超过该patch的$1.25R(d)$，该patch才进入前述外推稳定与可见切入裁决。三项条件分别限定E21已验证的局部空间范围、多尺度预测一致性和真实可见物体几何相容性。

下一项正式工作已经限定为对原4,827个train/206目标执行一次只读资格，不运行生成器或读取train/201。覆盖PASS条件直接继承E25-v2：四个active类别、五个距离层和三个遮挡层均非空，保留目标至少覆盖100帧和32个真实实例。PASS后才重建E25-v3目标库并重跑normal-control；FAIL时保留$1.25R(d)$不变，并把缺失覆盖记录为现有E21-v4可观测局部地面的数据边界。

正式只读资格已经在运行前提交`84c3655`下完成并PASS。4,827个目标保留3,267个，覆盖423帧和37个真实实例；四类计数为car 2,830、truck 333、other-vehicle 100、person 4，五个总体距离层为[861,1532,552,302,20]，三个总体遮挡层为[787,2363,117]。1,560个拒绝目标中1,509个没有任何patch落入其自身E21局部有效范围，14个没有外推稳定的局部patch，37个稳定局部patch均明显切入可见物体。

接受项的最大锚点—凸包距离为2.288462米，最大距离/中心半径为1.247676，逐项满足$1.25R(d)$。person只保留4/234，40–50米总体只保留20/117；该分布边界不改变运行前总体覆盖PASS，但限制后续主张，不能写成每个类别都覆盖完整距离和遮挡范围。E25-v3可信局部支撑定义已经关闭，当前进入3,267目标的正式target bank重建与normal-control重跑。

E25-v3目标库已在运行前提交`be0a8f7`下完成确定性重建并通过独立复核。新库按原顺序保留3,267个`compatible=true`身份，17个非支撑字段全部逐元素继承；两个支撑字段与资格产物逐元素一致，E21-v4行范围、语义、类别规则和帧偏移错误均为0。产物`runs/ajae/e25_v3_real_targets.npz`大小6,800,894字节，SHA-256为`0ae2f4926f1cb8a71b04af3d43d3d1d9feb17bb36fed821bab42b531cae3a360`，科学数组哈希为`16d75e67995bd216e3f802a6a32a19b0faecc9bfa841c6a680b01f13f6a8cf44`。正式normal-control继续使用原256个模板、seed 2,500,000–2,500,255、128×128提议上限、缩放/姿态/材质流、E22–E24、renderer和E45A五项caliper；runner只读取冻结目标库，不再包含旧无上限最近邻提取路径。下一步执行一次12进程正式资格。

E25-v3 normal-control正式资格随后在提交`a97a6c7`上执行一次并FAIL，分类为`local_support_conditioned_control_generation_failure`。256个固定fixture完成208个、耗尽48个，硬错误与完成对象的条件错误均为0；按模板类别分别完成car 64/64、truck 63/64、other-vehicle 62/64、person 19/64。48个耗尽中45个属于person；E25-v3目标库只保留4个person目标，且全部属于10–20米、低遮挡层，每个失败person fixture完整评估这4个目标对应的512个支撑proposal后仍未接受。

208个接受对象只覆盖90个中心帧、26个真实语义—实例身份，五个距离层为[7,163,27,11,0]，三个遮挡层为[37,169,2]。因此256/256完成、零耗尽、256模板、至少100帧、至少32个真实身份及五距离层非空均未达到；40–50米没有接受对象。该层20个目标中只有6个具有非空支撑流，且其确定性目标名次全部落在冻结的前128项目标前缀之后，所以正式运行没有实际尝试远距放置。821,370次实际支撑proposal严格分解为608,950次放置拒绝、212,212次条件拒绝和208次接受。正式产物`runs/ajae/e25_v3_normal_control.npz`大小524,004字节，SHA-256为`e31766c22ded4dcdf312540847944cb70a124c80b36af799f350734b0fb7aa98`，科学数组哈希为`b8d04778024e2c6b858b1361c395b2763a1e0b5655ab777c08578b798c81ed12`；独立复算与元数据一致。

E25-v3目标资格PASS与目标库PASS继续成立，但不能代替normal-control生成资格。本次FAIL不能单独推出renderer失败、E21-v4支撑资格失败、person全类几何不可放或normal-control整体构念不可行；当前只有一次正式运行证据。按照运行前冻结的FAIL路线，没有自动重试，没有改变$1.25R(d)$、128×128上限或E45A caliper，也没有进入E26-v2或train/201审计。该历史分支当时停在E25-v3等待课题负责人决策；后续决策已由下文E25-new合同给出。

课题负责人现已作出新决策并结束E25-v3的真实目标逐对象条件复制路线。E25-v2与E25-v3 normal-control FAIL永久保留；E25-new恢复生成器与审计的职责分离。E25-new只要求256个规范train/206模板各生成一个合法且可见的正常对照，并通过固定索引循环覆盖官方五个距离层；E45A的median beam、可见回波数、遮挡和局部密度caliper全部移回train/201审计阶段，不再作为生成接受条件。

E25-new的fixture index $i=0,\ldots,255$与规范模板索引一一对应，指定距离层为$i\bmod5$，总配额固定为$[52,51,51,51,51]$。最终距离身份只使用`render_frame`后全部可见normal-control回波的official range中位数。模板不得替换或重复，指定层不得回退；每fixture最多128个支撑proposal。类别支撑语义、0.9–1.1缩放、类别姿态、材质、E21–E24、传感器概率、强度与renderer保持不变。train/201、真实目标、E45A匹配结果和E46输出均不进入生成。

E25-new的唯一硬门是256/256完成、模板身份唯一、每项最终距离层正确、至少一个可见回波、E21–E24及类别语义合法、缩放/姿态/材质/renderer合同正确、hard error与指定距离层耗尽为0。方位角、遮挡层和$N_{vis}$只报告，不设置最低数量。正式运行固定24进程、每进程数值库单线程，只执行一次且不自动重试。运行前权威命令为`python -m src.render qualify-e25-new-normal-control --data-root /home/jasongao/Data/STU --support-pool runs/ajae/e21_v4_support_pool.npz --calibration runs/ajae/calibration.pt --output runs/ajae/e25_new_normal_control.npz --processes 24`。

E25-new已经在冻结实现提交`e9ee028f48ca43d5191e37373a23722cfeabec66`上完成唯一一次正式运行并PASS，墙钟时间27.831488263秒。256/256个fixture全部成功，256个规范模板各使用一次，四个类别各64个；预分配距离层和最终可见回波中位official range层均严格为$[52,51,51,51,51]$。全部对象至少有一个可见回波，$N_{vis}$的最小值、中位数、均值、95%分位数和最大值为1、53、269.41015625、1472和4927。408个实际支撑proposal严格分解为256次接受、119次物理放置拒绝、0次无可见回波拒绝和33次距离层拒绝；支撑、类别语义、缩放、姿态、E22、E23、材质、renderer、最终距离、可见性、proposal记账、hard error和耗尽错误均为0。

描述性八方位计数为$[36,43,15,40,52,9,10,51]$，三个遮挡层计数为$[204,50,2]$。正式产物`runs/ajae/e25_new_normal_control.npz`大小580,668字节，SHA-256为`30fc7d1ecd60d005cb18c60ac81b1c7335e2121fcd3f1da5f440b5387a747b19`，科学数组哈希为`4625b8e01be6ba73d41af96e56a530d361c7ecfe5cd9f5c89a0daec64d9fa31a`。结束后的独立只读复核逐项重算fixture身份、模板唯一性、距离层、支撑池身份、对象和放置记录规范往返、材质、E22连续grounding、E23深穿透、遮挡、方位、proposal守恒和科学哈希，全部一致；该复核没有重采对象，不形成第二遍正式运行结论。

E26-v2现已冻结为E25-new选择器的唯一生产接入资格。它完整继承历史E26的256个world seeds、四类world各64个、1–9实体数量流、48次完整world attempt、entity seed、标签顺序、schema 7 shape流、E21–E24、不可变`WorldSpec`、规范world/report/request身份和窗口/cache审计。normal-control仍按既有`template_seed=entity_seed+1`有放回抽取规范模板，不额外消费随机数；抽中模板索引$i$后，指定距离层固定为$i\bmod5$，支撑proposal直接复用E25-new对应模板的合法语义、指定锚点层和固定128项流。缩放、材质和姿态seed继续为`entity_seed+2`、`entity_seed+11`和`entity_seed+31`。

每个control先在当前部分世界的自身支撑帧使用正式world seed和实际object ID执行传感器接受、最近回波竞争与完整renderer复核；全部实体完成后，再在最终完整世界的同一支撑帧复核至少一个自身可见回波和中位official range层。后放实体使已有control失去资格时，整个确定性world attempt失败并进入既有下一attempt，不移动旧实体、不换模板或距离层。总体五层数量只报告，不设置均匀配额；真实目标和E45A五项caliper不进入E26-v2。

生产与资格runner都只调用修改后的唯一`sample_training_world`。实现先核对E25-new规范模板库和冻结传感器标定身份，在fork前预计算全部E25-new支撑流，以原始slot ID在保守角域中调用同一多实体回波竞争，并对任何接受候选保留完整131,072-slot renderer复核；传感器或接口`RenderError`直接形成hard error，不再被当作正常world attempt耗尽。资格runner从world seed与成功attempt序号独立复算数量、标签顺序、逐实体seed及两类对象的随机流，并逐对象复算E21记录、E22 grounding、E23 collision和最终E24 pair。真实train/206冒烟世界2,600,128在attempt 0完成，包含2个control和1个proxy，两项control的指定与最终层均为$[1,0,0,1,0]$，当时已有的control审计错误与hard error均为0；该冒烟不写产物、不计入正式资格，新增的完整独立审计仍以正式运行为准。

E26-v2正式运行固定24个fork worker、每个数值库单线程，只执行一次且不自动重试。命令为`python -m src.render qualify-e26-v2 --data-root /home/jasongao/Data/STU --support-pool runs/ajae/e21_v4_support_pool.npz --calibration runs/ajae/calibration.pt --output runs/ajae/e26_v2_world_builder.npz --processes 24`。PASS要求256/256世界完成，四类身份正确，全部E21–E24、control可见性与距离身份、支撑流和随机流复核、规范往返、pair、遍历、manifest、权威路径、hard error和耗尽计数均为0。

E26-v2已经在冻结实现提交`38079213a0801bf3a279414a8b120bfd24e1cd1b`上完成唯一一次正式运行并PASS。24个fork worker各使用一个数值库线程，GPU未参与，墙钟时间为187.63917079399107秒。256/256个world全部完成，world seed精确覆盖2,600,000–2,600,255，pure-normal、control-only、mixed和anomaly-only各64个；254个world在attempt 0完成，seed 2,600,139与2,600,066分别在attempt 1和2完成，48-attempt耗尽为0。

全部world共含605个实体，其中normal-control 307个、anomaly-proxy 298个。307个control的指定与最终距离层逐项相同，总计数均为$[52,72,55,71,57]$，每个control至少有一个最终可见回波；$N_{vis}$最小值1、中位数45、均值287.09771986970685、最大值5,577。全部E21–E24、control可见性与距离、两类随机流、支撑流、规范往返、pair、遍历、manifest、权威路径、hard error和耗尽错误均为0。

正式产物`runs/ajae/e26_v2_world_builder.npz`大小1,033,953字节，SHA-256为`2653f705d2e890d99cda732a7a00387b5621cd05abb9c4681c7a9f284c34363c`，科学数组哈希为`5766cda5820eb3281c0f9e13c64d2746ffdc120ce4543f32fa6c2c71cf1d4f97`。独立只读复核重算科学哈希、world/report规范JSON、world hash、request manifest、control observation和proposal守恒，所有比较错误均为0；它没有重采world，不构成第二遍正式运行。当前64个anomaly-only世界与历史E26产物的world/report、165个proxy对象、shape proposal和支撑proposal内容逐项完全相同，只有绑定当前renderer源码身份的request manifest发生预期变化。

因此E26-v2只建立新版选择器接入唯一生产world builder后的完整世界可采样性、实体合法性、不可变身份及control最终可见/距离身份；它不建立real/control共同支持、来源不可区分性或真实正常距离分布。新版normal-control分布下的Phase 2由此关闭，E27–E37机械资格继续保留；其后E38–E44刷新已经正式PASS。

这里的距离循环是用于构造反作弊正常对照的覆盖导向采样，不是对真实正常场景距离分布的估计。每个45度方位扇区的总体与分类别计数、最大扇区计数和占比，以及遮挡层与$N_{vis}$分布均只作描述。

当前可成立的局部结论是：

> **schema 7 能合法、确定且高效地产生覆盖冻结几何区域的合成异常代理；这些对象可以从 train/206 的合格支撑池采样，以冻结连续落地规则达到 99% 接触/埋地资格，并能由同一权威放置接口拒绝与 train/206 实际观测非地面回波发生超过 5 cm 深穿透的位置。**

> **E25-new的覆盖导向normal-control选择器已经接入唯一生产world builder；在冻结的256个world身份和48次完整world attempt上，E26-v2完成256/256个world，全部E21–E24、control最终可见性、距离身份、随机流、规范身份和有限耗尽检查均通过。**

当前尚未成立：

- renderer 来源泄漏 Gate 1；
- B1 代理监督有效性；
- B3 五帧时间增益；
- 真实 OOD 迁移。

当前执行节点为：

$$
\boxed{E38\text{--}E44\ \mathrm{PASS}\rightarrow E45B\text{-v2 PASS}\rightarrow E48\ \mathrm{PASS}\rightarrow E49\ \mathrm{PASS}\rightarrow E50\text{--}E58\ \mathrm{PASS}\rightarrow E59/E60\ \mathrm{COMPLETE}\rightarrow E61\text{--}E71\ \mathrm{PASS}\rightarrow E72\ \mathrm{CURRENT}}
$$

E23与E24-v2已按冻结设计通过；Gate 1已完成：

$$
E38\text{--}E44\ \mathrm{PASS}\rightarrow E45B\text{-v2 PASS}\rightarrow E48\ \mathrm{PASS}\rightarrow E49\ \mathrm{PASS}
$$

E27–E37的纯机械资格继续保留；E36-v1、E45-v1、E45-v2、E45A、E45A-v2、E45A-new、E45A-overlap、E25-v2和E25-v3 normal-control FAIL均永久保留。旧E45B已PASS，但只资格旧normal-control分布。$D_{xy}+\alpha$路线、E25-v3逐对象五维条件复制路线和全部E45A后续演化均已终止。E25-new、E26-v2、E38–E44刷新、E45B-v2、E48、E49、E50–E58与E61–E71已经PASS，E59/E60描述性聚合已经完成。E46已降为非阻断诊断；当前正式节点是E72。

因此当前整体判断仍是：

$$
\boxed{Gate\ 1\ \mathrm{PASS}\rightarrow E50\text{--}E58\ \mathrm{PASS}\rightarrow E59/E60\ \mathrm{COMPLETE}\rightarrow E61\text{--}E62\ \mathrm{PASS}\rightarrow E63\text{--}E71\rightarrow B1>B0}
$$

不能写成“AJAE 方法已经验证”或“剩余只需训练”。


---

# 19. 一句话定义 AJAE

> **AJAE 在 STU 正常序列上，从已资格支撑池构造不可变反事实世界，以同一 OS1-128 单发布回波渲染器生成 normal-control 与类别无关 anomaly-proxy，借助冻结 STU 的单帧正常表征，在中心对称五帧中通过保留时间身份的四级点金字塔和按时间分层的局部注意力学习可拒绝的跨帧上下文，最终对每个规范 frame/ray 回波融合窗口级异常概率，并以一次性真实异常序列确认代理监督是否迁移到真实 OOD。**
