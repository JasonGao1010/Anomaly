# 单帧 STU 异常分割研究协议

当前研究对象是一张 STU 激光雷达扫描中的逐点异常。输入为当前扫描的坐标和原始回波强度，输出为同一扫描实际回波的异常分数，分数越高表示越异常。

当前实现提供单帧数据读取、预测存储、评价和几何诊断。主干网络、检测头、训练样本生成方式和训练损失尚未确定；后续应先完成残余失误诊断图谱，再根据真实证据决定方法设计。当前没有正式模型成绩，也没有对边界、稀疏或低矮条件作出保留、合并或放弃的实验裁决。

机器可读定义在 [spec.json](spec.json)，读取和必要语义检查由 [src/protocol.py](../src/protocol.py) 完成。当前格式为 `stu-single-frame`，格式版本为 1。

## 工作区结构

同类内容集中保存，以下命令均从仓库根目录执行。

| 目录 | 内容 |
| --- | --- |
| `src/` | 单帧数据、评价、几何统计和可选渲染代码 |
| `tests/` | 对应的数据身份、评价和几何检查 |
| `protocol/` | 本说明与机器可读定义 |
| `assets/` | 可复用的射线几何标定数据 |
| `results/` | 当前结果；同一次几何统计的明细与表格共用一个目录 |
| `vendor/stu/` | 保持原样的官方评价器及许可证 |

几何统计的明细位于 `results/profile/records/`，表格和统计说明位于 `results/profile/tables/`。只有表格和说明纳入版本控制；明细、预测和其他运行结果保持本地保存。

## 数据范围

开发诊断使用 STU 公开验证集全部 19 条异常序列：125、137、138、139、140、141、142、143、144、145、146、147、148、149、150、151、152、153、169。每条序列的全部原始帧都可以产生单帧预测。

现有正常数据源为 `train/201` 的 682 帧及 `train/206` 的 449 帧。这只是现有数据清点，不规定新方法的训练分配。隐藏测试集不在当前开发范围内，当前数据入口不会开放它。原始数据保留在仓库之外，不随代码提交。

公开验证集已经用于开发分析，后续在此集合观察的规律和模型差异均应保留开发数据的适用边界。几何条件与失误的统计差异不能单独证明因果机制。

目录结构为：

```text
STU/
  train/{201,206}/
  val/<sequence>/
    velodyne/<frame>.bin
    labels/<frame>.label
    calib.txt
    poses.txt
```

`spec.json` 保留已记录的原始压缩包摘要，供需要核对数据来源时使用。

## 单帧读取和模型输入

[STUSequence](../src/scene.py) 通过 `source_frame(frame_id)` 返回一个 `SourceFrame`。索引和迭代同样返回原始单帧。

`SourceFrame.xyzi` 保存完整原始文件槽位，形状为槽位数乘 4，数据类型为 `float32`。`real_slots` 按原始顺序列出坐标不全为零的实际回波槽位。零坐标占位槽继续保存在源数据中，进入模型时通过 `real_slots` 排除。强度保持原值，不能预设它严格位于 0 到 1 之间。

模型可读的基本数组是：

```python
x = frame.xyzi[frame.real_slots]
```

语义标签、实例标签以及依赖标签计算的几何量仅用于监督或诊断，不能作为推理输入。用 `label_mode="forbidden"` 读取源数据时，读取器不打开标签文件。

源对象保留官方坐标及特征的原有计算能力，供数据核对和外部方法适配使用：`coordinates` 按发布位姿计算，`features` 包括原始强度及相对扫描坐标均值的距离。它们是辅助数据表达，不改变当前模型以单帧传感器坐标和强度为输入的任务定义。官方评价距离始终来自 `xyzi[:, :3]`，不能用世界坐标范数代替。

发布标签的低 16 位是语义，高 16 位是实例。原始打包标签完整保留。当前源对象和渲染工具保留正常语义类别转换能力，但尚未据此指定新模型的监督方式。

读取示例：

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m src.scene \
  --data-root /home/jasongao/Data/STU --partition val \
  --sequence 125 --frame 0 --labels required
```

## 预测身份和存储

[src/data.py](../src/data.py) 中的 `FramePrediction` 使用以下字段：

```python
FramePrediction(
    partition,
    sequence_id,
    frame_id,
    source_slot,
    anomaly_score,
)
```

每个分数对应一个明确的原始文件槽位。保存和加载时都必须验证：

- 分区、序列号和帧号与传入的 `SourceFrame` 一致。
- 槽位无重复，且其集合恰好等于 `SourceFrame.real_slots`。
- 分数是有限的 `float32`；生产端显式决定精度转换，存储端不静默量化。
- 分数可以为负，也可以超过 1；评价使用原始数值的排序。

生产端可以采用任意点行顺序，但必须同时保存对应槽位。`restore(source)` 按槽位放回完整原始顺序，并将零坐标槽的分数填为 0。依赖数组行号而忽略槽位身份的结果不能进入评价。

`save(path, source)` 不覆盖已有预测；`load(path, source)` 不仅检查压缩包字段，也重新核对真实源扫描。当前存储格式是 `stu-frame-prediction`，不接受其他任务格式作为当前结果。

统一目录为：

```text
<predictions>/val/<sequence>/<frame:06d>.npz
```

各模型或消融实验分别使用自己的预测根目录。当前尚未导入可靠的单帧模型预测。

## 官方评价和误差归属

[src/evaluate.py](../src/evaluate.py) 保留 [STU 官方评价器](../vendor/stu/compute_point_level_ood.py) 的点集和阈值定义：

- 点到当前传感器的距离满足 `2.5 <= r <= 50` 米。
- 语义 0 忽略，语义 2 为异常，其余有效语义为正常。
- 距离过滤后异常点少于 5 个的帧整体不进入正式指标。
- 所有合格帧的全部有效点合并计算指标，不平均逐帧或逐序列指标。

平均精确率 AP、排序区分指标 AUROC、以及高召回工作点的正常误报比例 FPR95 以百分数报告。FPR95 使用官方保留的 ROC 阈值节点中召回率严格大于 95% 的第一个节点。相同分数的点必须整体进入或离开检测集合。

低误报工作点使用完整官方点集上实际误报比例不超过 1% 时的最高召回率；召回率相同时选择最早达到该召回率的阈值。如果没有异常点能在该条件下检出，阈值记录为空，含义是拒绝全部点。子组统一使用这两个全局工作点。

指标实现保留精确的单精度排序，并采用磁盘临时数组就地排序和分块累计；不把分数量化到直方图。官方评价器保留为独立核对依据。

`APAttribution` 记录每个异常分数对应的全局精确率及所需误报比例。对异常点 `i`，全局同分组全部纳入后的精确率记为 `P_i`，异常点总数记为 `N_A`。其失分为 `(1-P_i)/N_A`，全部异常点的失分之和应为 `1-AP/100`。任意互斥分组可以据此汇总对全局 AP 的失分贡献。

所需误报比例按分数大于或等于该异常点分数的正常点数量除以全局正常点数量计算。它与 AP 失分都来自同一个全局排序。当前模块提供这些通用计算，尚未实现图谱的条件匹配、序列重采样及贡献裁决。

完整开发集评价示例：

```bash
PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  .venv/bin/python -m src.evaluate --data-root /home/jasongao/Data/STU \
  --predictions results/predictions/<model> --output results/evaluation/<model>
```

可用 `--sequence` 运行明确标记的开发子集。完整评价要求所选序列的每个原始帧均有符合身份约定的预测。临时排序文件使用结束后即释放。大体积评价必须根据当时 Windows E 盘空间保留至少 10 GB 余量。

## 单帧真实几何画像

[src/profile.py](../src/profile.py) 保留单帧统计算法，包括异常点数、距离、强度、最近正常点距离、正常邻域、同实例最近邻、可见实例尺寸、协方差形态和地面关系。几何量在当前扫描中计算，源点身份贯穿统计过程。

画像描述全部原始帧的实际回波，范围比官方指标点集更广。原始异常点数和官方范围内异常点数分别保存；符合官方帧门槛的计数另行汇总。忽略点和范围外点的描述性统计不能混入正式 AP 失分结论。

实例几何与地面估计保留原有可靠性规则。无可靠地面拟合时，高度保持缺失并报告原因。异常点实例标识为零时，不假定这些点属于同一个物体；点级和帧级异常统计继续保留它们。

[src/profile_report.py](../src/profile_report.py) 根据实际生成的逐帧记录、逐实例记录和分布累计结果导出 [统计表格](../results/profile/tables)。分位数来自完整加权经验分布，不平均各序列分位数。表格目录保存：

```text
frames.csv
instances.csv
sequences.csv
totals.csv
count_distance.csv
continuous.csv
categorical.csv
bins.csv
reliability.csv
index.csv
method.md
```

CSV 数值列与实现字段直接对应，空值表示缺失或不适用。CSV 本身不保存字体。详细统计定义由生成源写入 `method.md`，修改定义时应修改生成源并重新生成结果。

完整运行示例：

```bash
PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  .venv/bin/python -m src.profile --data-root /home/jasongao/Data/STU \
  --output results/profile --workers 12
```

进程数必须依据当时 CPU、内存和其他任务实测状态确定。`--output` 同时决定明细子目录 `records/` 和表格子目录 `tables/`。少量帧检查可使用 `--sequence` 和 `--limit`，并指定独立的 `--output`，避免将子集结果覆盖到完整画像位置。

已有完整明细时，在上述命令中加上 `--tables-only` 可重新汇总并生成表格，无需重新读取原始扫描。表格中的 `index.csv` 使用相对于表格目录的路径指向明细汇总文件。

## 可选渲染工具

[src/render.py](../src/render.py) 保留连续几何、形状族、激光射线、物体放置、地面支持、传感器标定和真实遮挡竞争。`render_frame(source, world, ray_grid, sensor)` 的基本渲染对象是一个 `SourceFrame`；`render_frames` 只是逐个调用单帧渲染的流式工具。

[assets/rays.npz](../assets/rays.npz) 保存独立的射线几何标定数据。`calibrated_ray_grid` 读取该文件，计算束方向、束原点以及原始槽位到射线的对应关系。当前未预置地面支持库或传感器回波规律产物；新训练生成方案应在残余失误诊断之后另行确定。

当前渲染工具不作为真实画像或官方评价的依赖。合成测试用例仅验证实现和物理遮挡约定，不能作为真实数据上的科学结论。

## 必要验证与当前边界

关键检查对应数据和评价的实际风险：原始槽位错配、标签错配、零坐标槽处理、距离边界、少于 5 个异常点的帧过滤、全局同分排序、严格高召回阈值、AP 失分加总、地面估计缺失和前景遮挡。

```bash
PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  .venv/bin/python -m pytest -q -p no:cacheprovider
```

接口检查使用人工构造的分数时，只能支持存储、排序和评价实现的正确性，不产生模型性能证据。

下一步需要建立残余失误诊断图谱所需的局部采样尺度和归一化边界深度，接入可靠单帧模型的真实预测，执行控制比较与序列层面的不确定性分析。模型结构及训练目标应由这些结果决定。
