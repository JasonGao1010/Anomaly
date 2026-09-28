# SERVE

本工作区实现 `preview.tex` 中的研究方案：利用完整扫描的外观支持，以及排除目标角度单元后预测的类别条件距离密度，共同进行正常语义分割与未知目标检测。

当前 `preview.tex` 正文没有声明结果是假设性的；但仓库已提交版本 `5a7668f` 的源文件首行将新增实证结果标为假设性预览，同版本的图生成代码也明确把硬编码数值标为未测量的示例。目前没有找到支持论文全部数字的对应实验记录，因此这些数字不作为已复现结果。`figures/preview.py` 只有显式指定 `--illustrative` 才重建示例图。正式结果必须来自下面的训练、评价及三种子汇总。论文已有的未提交编辑不由实现过程覆盖。

## 文件与数据

- `src/data.py`、`src/nuscenes.py`：原始数据、19 类允许标签集合、补充标注、正常划分与原始点身份。
- `src/normal.py`：角度单元、独立单元编码、排除目标后的类别条件 Student-t 混合密度、预测损失与诊断。
- `src/model.py`：LitePT-S、外观点特征、类别支持、全部训练变体和固定权重读出。
- `src/train.py`：两阶段完整训练、正常开发集选权重、断点续训和可选的正常参考分数变换。
- `src/evaluate.py`：官方指标、逐点独立复算、配对检测与语义纠错、实例覆盖、原始点位导出、运行时间和三种子汇总。
- `assets/normal.json`：助手视觉检查的补充正常源标签，尚无独立人工复核；建筑补充标注也保留这一来源说明。`assets/nuscenes.pth`：官方预训练权重；`assets/val.json`：真实 STU 验证清单。
- `results/data/background/`：完整 nuScenes 源训练、开发清单和补充标注。清单中的历史格式名称只用于读取原始数据，不代表旧方法仍在使用。
- `vendor/`：实际使用的 LitePT、STU 官方评价代码、论文模板及许可证；`literature.csv`、`paper.bib` 保留本课题文献依据。

旧实现的已有实验记录完整保存在 `results/previous/evidence/`，其中路径和配置保留原始来源。它们没有被新代码重算，不能填入新方法的结果表。此目录下原有未提交文件仍只保存在本地。新实验使用独立的 `results/train/<variant>-<seed>/`，新模型会拒绝旧实现的权重身份。

本轮按用户要求暂停并替换了初始训练配置。该轮完整记录保存在本地 `results/previous/baseline/`，对应代码提交 `c9b4ad8`；种子 206 的外观、联合和分离目标模型正常 mIoU 分别为 46.44%、51.71%、45.86%，标准分类器保留源阶段第 21,250 次更新的可恢复状态。它们只用于比较旧配置，不能与新配置合并计算三种子均值。当前实现身份为 `SERVE-2`，所有模型重新从官方预训练骨干开始。

## 环境和数据准备

使用 Python 3.13，当前可用解释器为 `.venv/bin/python`；包版本在 `requirements.txt` 中。PyTorch、FlashAttention、torch-scatter 和稀疏卷积扩展必须使用相容的 CUDA 二进制包。当前环境已验证 PyTorch 2.12.0+cu130。nuScenes 这里只使用固定划分表，另以 `pip install --no-deps nuscenes-devkit==1.2.0` 安装，避免其完整开发包的旧 NumPy 约束替换本项目环境。

原始数据默认位于 `/home/jasongao/Data/Nuscenes` 和 `/home/jasongao/Data/STU`。训练使用 nuScenes 的 28,130 个扫描，源开发集为 6,019 个扫描；STU 206 的 449 个扫描用于适应，201 的 682 个扫描用于正常选择与诊断。训练和开发均拒绝目标异常标签。所有非空返回保留为输入，监督限定在 2.5–50 米的可靠正常标签点。

现有清单可以直接读取。需要从原始数据重新生成时，输出应使用空目录或不存在的文件：

```bash
.venv/bin/python -m src.data nuscenes --output results/data/source --normal-annotations assets/normal.json
.venv/bin/python -m src.data val --output assets/val.json
```

使用新源清单训练时增加 `--source-directory results/data/source`。正常 STU 根目录可用 `--stu-root` 指定。原始文件与补充标注会核对点身份和实际字节，训练与开发的场景、日志、扫描和标签文件必须互斥。

## 训练

完整运行九个模型变体、三个种子以及验证、配对分析、正常纠错、计时与汇总：

```bash
.venv/bin/python -u -m src.train --suite --output results/train
```

同一命令可以继续未完成的实验矩阵；`progress.json` 记录当前实际执行任务。各任务串行使用 GPU。最终 `summary.json` 汇总论文所需的实测统计，`figures/results.pdf` 使用真实三种子数据。用户已确认没有隐藏测试数据或官方测试服务，因此测试指标明确保持缺失，验证集实验完成不能等同于论文全部结果复现完成。

每个种子先训练外观、联合和分离目标模型，三者齐备后立即执行配对异常评价、正常决策比较和计时，再训练其余对照及消融。新增结果因此能尽早回答方法是否改善异常排序，而无需先等待该种子的全部九种训练。

配对评价保留三个独立模型的逐点记录；三个固定权重读出的临时数组在完整统计后释放，保留其指标、检查点和原始点身份以供重新生成。这样避免三种子的重复数组占满 E 盘。模型、数据、损失、随机流、样本和指标不因节省空间而改变。

单个训练任务也可以单独执行：

```bash
.venv/bin/python -m src.train --variant joint --seed 206 --output results/train/joint-206
```

种子为 `206`、`307`、`409`。下面每种训练变体均使用同一正常数据、公共模块初始化规则、逐扫描顺序、旋转增强、查询身份和更新预算。

| 参数 | 对应论文模型 |
|---|---|
| `joint` | SERVE |
| `semantic` | 独立训练的外观模型 |
| `separate` | 外观分类与距离预测分别训练，推理联合使用 |
| `standard` | 标准线性分类头，共享权重输出最大概率和能量分数 |
| `cssr` | 类别条件特征重构及一阶、二阶正常激活支持 |
| `target_available` | 允许预测器读取目标单元 |
| `single_component` | 单个 Student-t 分量 |
| `no_nll` | 移除预测负对数似然 |
| `no_compactness` | 移除外观紧致性损失 |

源训练遍历两次，目标训练遍历八次，每四个目标扫描插入一次源重放，有效批量为两帧。完整预算分别为 28,130 和 2,244 次优化器更新。查询上限为 4,096：按各个不同的允许标签集合等额抽样，不超过配额的小集合全部保留，再从未选可靠点中均匀补足。辅助外观分类及正常语义评价仍覆盖全部可靠标签点。重放的一半位置优先覆盖目标训练中缺少的停车区域和骑自行车者标签。每次访问编号进入旋转与查询随机流，同一帧同一轮重复重放获得新的旋转和查询；同种子不同模型保持逐访问一致。

优化器为 AdamW，权重衰减 0.005、数值稳定项 1e-6。源阶段骨干、外观及共享类别向量、距离预测分支的峰值学习率分别为 5e-5、8e-4、4e-4；目标阶段分别为 1.5e-5、2e-4、1e-4。学习率在前 5% 更新线性增加，随后按余弦衰减到峰值的 5%。共享类别向量仍使用外观参数组的学习率，因为两个分支都依赖它。各阶段重新建立优化器，逐帧累积两帧梯度。训练与评价均使用 FP32。

骨干的点序列化顺序仅在训练时打乱，评价固定顺序，使同一权重的不同读出及配对评价可重复。距离分支的尺度初始偏置从 −2 改为 −3，初始尺度约为 0.0496；该分支的坐标除数从 50 改为 25，物理距离、对数距离、方向偏移、外观分支和骨干输入均保持原定义。Student-t 分布、三个分量、尺度下限 0.001、目标排除机制、损失权重和数据标签保持论文设定。

上述抽样、随机流、学习率及初始化是论文未唯一指定部分的一组联合改进候选，尚未证实能够提高指标，也不能通过这次合并实验辨认每一项的独立贡献。网络中间宽度、激活及归一化顺序、体素聚合等仍采用当前代码的明确选择；两层、三个注意力头已有论文方法图依据。训练配置保存具体优化参数和实现选择，不把它们称为已恢复的原始实验设置。

每 2,000 次源更新和 225 次目标更新，以及各阶段末尾，分别在 50 个源开发场景的中央帧和 68 个均匀选择的目标开发帧上评价。先比较正常 mIoU，再用正常目标函数打破完全相同的 mIoU。完成全部源更新后，用选出的源权重初始化目标阶段；完成全部目标更新后，使用选出的目标权重生成 `model.pt`。不会用异常检测分数选择权重。

中断后在原命令中添加 `--resume`，从 `last.pt` 恢复权重、优化器、随机状态和下一批身份。`--match-run results/train/semantic-206` 可核对已完成的同种子对照。`--calibrate` 对最终分数拟合论文附录所述可选的单调正常参考变换，默认直接输出论文原始分数。

训练输出包括 `config.json`、`stages.json`、`normal201.json`、`training.jsonl`、选定权重和续训状态。CSSR 仅在最终权重选定后，用正常训练点拟合激活参考，不使用开发标签拟合该参考。每次训练都重新检查 CPU、内存、GPU 和 Windows E 盘空间，磁盘保留至少 10 GB。

## 评价与结果

```bash
.venv/bin/python -m src.evaluate validate --checkpoint results/train/joint-206/model.pt --output results/train/joint-206/val.json --record-points
.venv/bin/python -m src.evaluate compare --semantic results/train/semantic-206/model.pt --joint results/train/joint-206/model.pt --separate results/train/separate-206/model.pt --manifest assets/val.json --output results/train/paired-206 --fixed-readouts --normal-fpr .001 .005 .01 .02 .05
.venv/bin/python -m src.evaluate compare-normal --semantic results/train/semantic-206/model.pt --joint results/train/joint-206/model.pt --output results/train/paired-206/normal.json --fixed-readouts
```

单模型评价通过 `--readout appearance`、`--readout common_density` 或 `--readout independent_minima` 指定固定权重消融；标准分类器使用 `--readout softmax` 或 `--readout energy`。公共密度的语义标签与外观读出相同，独立最小值仅改变未知分数。

官方评价先应用逐点距离和标签掩码，再保留至少五个有效异常点的扫描。AP、AUROC 和 FPR95 使用完整合格总体的点分数。配对比较在同一个正常点总体上按指定误报预算确定各模型阈值，并报告离散分数下的实际误报率。高置信未知子集固定由独立外观模型定义。实例覆盖按扫描内实例计算，至少一半有效点检出才算覆盖，小实例仍保留。

`--record-points` 保存分数、原始点槽位和独立的 `*_records.json`。`src.evaluate.recompute_metrics(score_path, manifest)` 可依据这些记录独立复算，并检查原始点分数与指标总体是否相符。完整验证和三模型配对会写入数 GB，运行前会重新检查 Windows E 盘剩余空间。预测诊断明确记录其查询总体，报告真正的混合分布分位数和逐查询均值。

三个种子分别完成后汇总，不把不同模型或不同总体混在一起：

```bash
.venv/bin/python -m src.evaluate aggregate --results results/train/joint-206/val.json results/train/joint-307/val.json results/train/joint-409/val.json --output results/train/joint.json
.venv/bin/python figures/preview.py --comparisons results/train/paired-206/comparison.json results/train/paired-307/comparison.json results/train/paired-409/comparison.json
.venv/bin/python -m src.evaluate benchmark --checkpoint results/train/joint-206/model.pt --manifest assets/val.json --output results/train/joint-206/runtime.json
```

汇总使用三个种子的算术平均和样本标准差。空的高置信子集或对象大小组保持未定义，不用零代替，也不缩减到部分种子求平均。运行时间在 50 个预热扫描后测量全部 8,659 个验证扫描，包含准备、传输、模型计算和原始点输出构造，排除磁盘读取和全局指标计算；异常指标仍使用其中 1,960 个合格扫描。

本地尚无隐藏测试数据。取得数据后，`src.data test` 可以生成无标签清单，`src.evaluate export --split test --manifest ...` 可以按原始点顺序导出分数。没有真实测试标签或官方返回结果时，不生成测试性能数字。

## 验证

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/python -m pytest -q
```

测试覆盖目标单元排除、密度与梯度、类别允许集、固定读出、CSSR 统计、数据身份、阈值与指标、完整预算及断点续训等关键语义。小规模真实扫描检查验证代码可执行；正式性能仍需完整训练和评价。
