# AJAE 数据协议复现说明

当前活动协议是 schema 34。科学定义见 `PROTOCOL.md`，机器可检查合同见 `protocol.json`。

当前正式数据池已生成并通过完整窗口检查，协议状态为 `frozen`：206 训练池包含 128 个单代理世界、3080 个窗口；201 合成验证池包含 92 个单代理世界、2360 个窗口；原始完整 201 另有 678 个在线窗口。完整检查记录见 `artifacts/data/qualification.json`。这说明数据实现通过本协议检查，不代表模型效果已经得到验证。

训练池和合成验证池的 5440 个真值着色 PLY 已全部导出至 `artifacts/ply/`，共 49,311,794,379 字节（45.93 GiB）。220 个正式世界在末帧均有可见异常回波；训练池仍有 20 个、合成验证池仍有 10 个全窗口无异常回波的样本，均按规则保留。末帧物理可见不等于进入官方 2.5–50 米评价范围，资格记录分别保存两种计数。

`vendor/stu/` 保存官方点级评价源码和许可证。为直接导入评价类，只将原始数据命令行工具的 `utils.common` 导入移至 `main` 内，点筛选和指标公式不变；模型训练和最终评价的完成情况应以真实运行结果为准。

## 原始数据

STU 原始数据不随仓库分发。先按 `protocol.json` 的 `data.official_archive_sha256` 核对官方压缩包，再整理为：

```text
STU/
├── train/201/{velodyne,labels,poses.txt,calib.txt}
├── train/206/{velodyne,labels,poses.txt,calib.txt}
├── val/<sequence>/{velodyne,labels,poses.txt,calib.txt}
└── test/<sequence>/{velodyne,poses.txt,calib.txt}
```

数据池生成与资格检查只读取 `train/206` 和 `train/201`。真实异常数据的常规实验仍未开放；用户另行授权的 `val/125` 第 0–18 帧有限目视检查见下文和 `PROTOCOL.md`。隐藏 `test` 序列未读取。

## 代码检查

数据链路使用 NumPy、SciPy 和 PyTorch（读取传感器标定文件），检查使用 pytest 和 Ruff。安装所需依赖后，在仓库根目录运行：

```bash
python src/protocol.py
python -m pytest -q
ruff check src tests
```

协议摘要应显示 schema 34、206 训练源、201 完整验证源、3080 个合成训练窗口和 2360 个合成验证窗口。正式池尚未完成时，状态必须保持 `qualification_pending`，`training_allowed` 必须为 `false`。

## 正式训练数据入口

仓库已实现下文的固定八窗口学习诊断，尚未启动完整数据池训练。正式训练数据入口已经实际执行冻结清单核验，不依赖调用者先手动运行检查命令：

```python
from pathlib import Path
from src.data import FrozenWindowDataset, WindowPartition
from src.protocol import load_protocol

protocol = load_protocol()
data_root = Path("/absolute/path/to/STU")
train = FrozenWindowDataset(data_root, protocol, pool_name="train")
validation = FrozenWindowDataset(data_root, protocol, pool_name="validation")
normal_validation = WindowPartition(validation.source_sequence, 4, 681)
```

每个构造函数在返回数据前核验原资格文件、两份清单及全部 220 个片段文件，任一文件缺失、哈希不符、身份或边界不符都会拒绝加载。数据集按索引返回完整 `SceneWindow`，窗口数量分别为 3080 和 2360；正常 201 保留全部 682 帧，在线窗口仍为 678 个。只有 `train.gradient_updates_allowed` 为真，验证数据不允许用于参数更新。

以下只读命令调用同一个训练数据构造函数，并实际加载所选池的首末窗口；它不启动训练：

```bash
python src/data.py check --pool train --data-root /absolute/path/to/STU
python src/data.py check --pool validation --data-root /absolute/path/to/STU
```

预测写入使用 `PredictionBatch.from_window(window, scores).save(path, window=window)`；读取使用 `PredictionBatch.load(path, window=window, expected_sha256=file_sha256)`。`scores` 必须已恢复为原输入窗口的全部点顺序；保存、读取均核对实际窗口，已有文件不覆盖。

## 五帧异常分割模型

`src/model.py` 实现唯一的首版模型：九通道联合体素输入、官方语义分割 LitePT-S 编码器与轻量解码器、`81→32→1` 逐点异常头，隐藏层使用 GELU。共有 12,729,089 个可训练参数，其中九通道骨干 12,726,432 个、异常头 2,657 个。模型全部随机初始化，不下载或加载外部权重，没有独立预训练阶段。

`vendor/litept/` 仅引入 [LitePT 官方仓库](https://github.com/prs-eth/LitePT/tree/f0cc7692b81518124b96c856e79346fd19f40bec) 提交 `f0cc7692b81518124b96c856e79346fd19f40bec` 的独立 `litept/`、`libs/pointrope/` 实现及 MIT 许可证。骨干只把 PointROPE 的导入改为包内相对路径；结构参数保留官方语义分割小模型默认值，输入通道从 4 改为 9，解码输出为 72 维。未引入官方数据加载器、训练框架、演示数据或检查点。未安装 PointROPE 专用扩展时，使用官方提供的纯 PyTorch 实现。

输入直接来自冻结的 `SceneWindow`，不重新配准，不裁点，不重新渲染。体素边长默认 `0.05` 米，以当前 LiDAR 原点为网格原点，按照 `floor(xyz / d)` 分组；整数网格再减去最小格号，以满足稀疏卷积的非负索引要求，物理坐标不平移。分格除法和均值累加使用双精度，网络输入保存为单精度。每个体素的九通道依次为坐标均值、原始强度均值和五次扫描命中标记。命中标记仅表示该次扫描在体素中存在返回，不代表可见性或异常判断。五帧共用一个场景编号，不把扫描来源拆成不同批次。

每个原始点按逆映射取得 72 维体素特征，再拼接三维体素内均值偏移除以体素边长、该点原始强度、五维相对扫描来源独热编码，得到 81 维输入。标签、标签有效性、绝对帧号、槽号、序列编号和代理编号不进入特征。同一体素内的正常、异常和忽略标签保持逐点独立，不生成体素标签。

模型的 `forward(window)` 返回原窗口顺序的 `M` 个未经过 sigmoid 的值，用于训练；`predict(window)` 返回已有的 `PredictionBatch`，保存全部点身份和 sigmoid 异常分数。分数尚未经概率校准。训练时只用 `labels.anomaly_target != -1` 排除官方忽略点，不使用 `current_mask` 限制训练监督。固定样本重复学习时，可将 `joint_voxelize` 的结果通过 `forward(window, inputs=...)` 复用；输入绑定到该窗口的只读点对象和体素大小，仍走同一个骨干与逐点预测实现。

```python
from pathlib import Path
from src.model import AJAE

window = train[0]  # train 来自上面的 FrozenWindowDataset，构造时已核验冻结池。
model = AJAE(voxel_size=0.05).cuda().eval()
prediction = model.predict(window)
prediction.save(Path("runs/window_000004.npz"), window=window)
online_scores = prediction.anomaly_score[prediction.online_mask]
```

先保存全部点预测，再提取当前帧在线分数；不同窗口中同一点的其他预测不参与在线分数融合。上述随机模型示例只验证接口，不产生有意义的异常检测结果。体素边长属于模型超参数，不改变 schema 34 的数据协议。

本机模型环境使用 `.venv`，复用已有 Python 3.13、PyTorch 2.12.0 / CUDA 13.0，编译器为 CUDA 13.2；不复制另一份 PyTorch。新增依赖安装在项目环境内，不修改基础环境。安装命令如下，其他机器必须选择与其 PyTorch、Python 和显卡匹配的扩展版本：

```bash
python -m venv --system-site-packages .venv
.venv/bin/python -m pip install --no-cache-dir \
  spconv-cu126==2.3.8 addict==2.4.0 colorhash==2.1.0 \
  timm==1.0.27 einops==0.8.2 ninja==1.13.0
.venv/bin/python -m pip install --no-cache-dir --no-deps \
  torch-scatter==2.1.2+pt212cu130 \
  -f https://data.pyg.org/whl/torch-2.12.0+cu130.html
MAX_JOBS=3 NVCC_THREADS=1 FLASH_ATTN_CUDA_ARCHS=120 \
  FLASH_ATTENTION_FORCE_BUILD=TRUE \
  .venv/bin/python -m pip install --no-cache-dir --no-deps \
  --no-build-isolation flash-attn==2.8.3.post1
```

FlashAttention 针对本机 `sm_120` 编译，保留官方骨干使用的注意力实现。编译前仍需检查 E 盘物理剩余空间；本机使用三个编译任务，避免编译器内存叠加挤占交换空间。PointROPE 当前使用官方 PyTorch 实现，不执行上游仅指定 `sm_90` 的专用扩展安装命令。

模型权重保持单精度，训练可由调用者启用自动混合精度。实际检查发现 `spconv 2.3.8` 的推理分支跳过了训练分支的权重自动类型转换，半精度特征与单精度权重混用会报错。因此 `AJAE.forward` 在 `eval()` 模式下关闭外层自动混合精度，按单精度调用骨干与预测头；官方注意力内部的半精度实现保持不变。该处理只限定执行精度，不修改官方网络结构，也不改变冻结坐标、点数或标签。

下列检查不启动优化器或长训练，不保存模型检查点。普通检查覆盖独立体素计算、标签隔离、官方骨干前向与反向、全点预测保存和读取；显式设置数据路径后，另以完整冻结窗口进行显存与全点输出检查，训练 206 只做一次反向，合成与正常 201 均仅前向：

```bash
PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  .venv/bin/python -m pytest -q -p no:cacheprovider
AJAE_STU_ROOT=/absolute/path/to/STU PYTHONDONTWRITEBYTECODE=1 \
  OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  .venv/bin/python -m pytest -q -s -p no:cacheprovider \
  tests/test_model.py -k complete_frozen
```

本机已实际执行上述检查：普通测试 38 项通过，默认跳过的完整数据检查另行执行并通过；Ruff 和差异格式检查通过。所有输入特征的独立复算、混合标签逐点保留、有效历史点与当前点梯度、忽略点零损失梯度、全点预测保存与回读均通过对应实现检查。以下三个窗口的当前帧均为 4，没有裁点或改变体素边长：

| 输入视图 | 全窗口点数 | 联合体素数 | 当前帧点数 | 执行内容 | 峰值已分配／已预留显存（GiB） |
| --- | ---: | ---: | ---: | --- | ---: |
| 206 合成训练 | 617,966 | 396,386 | 124,140 | 前向与反向 | 5.33／5.92 |
| 201 合成验证 | 1,163,987 | 390,734 | 103,929 | 仅前向 | 1.56／3.05 |
| 201 原始正常 | 1,163,892 | 391,106 | 103,915 | 仅前向 | 1.56／3.05 |

显存由 PyTorch 统计，不包括桌面和驱动占用，也不是整个数据池或正式优化器训练的显存上界。训练检查使用半精度自动混合精度及损失缩放；验证使用上面说明的推理精度路径。修复后该次检查的单窗耗时分别为 6.93、1.53、0.96 秒，包含输入整理及对应梯度检查；首次训练窗口检查耗时为 28.45 秒，不能以其中任一单次耗时直接推算稳定训练吞吐。上述模型适配检查未执行优化器更新、模型选择或 STU 性能评价，只支持模型接口与所测窗口的可执行性。

## 固定八窗口学习诊断

`src/train.py` 只回答模型能否学习见过的合成异常，不承担正式验证或完整训练。它通过现有冻结入口读取 206 第 0 条合成序列，在片段 0、2、4、6、8、10、12、15 各取末尾五帧窗口，当前帧固定为 27、83、139、195、251、307、363、448。八个窗口互不重叠，完整保留所有可见点、有效标签和忽略点；不使用 201 观测、真实异常序列或隐藏测试数据。冻结入口仍核验两份池清单与文件身份，不因此把验证数据送入模型。

模型和异常头均随机初始化，种子 23。每轮独立打乱八个窗口且无放回访问，共 25 轮、200 个计划步骤，每步一个完整窗口，无梯度累积或增强。AdamW 固定学习率 `3e-4`、权重衰减 `1e-2`、`betas=(0.9,0.999)`、`eps=1e-8`；全模型梯度范数上限为 1。沿用半精度自动混合精度，损失缩放初值 128。先解除缩放并检查梯度，再裁剪、更新参数；实际调用优化器才计作成功更新。连续三次梯度溢出、显存不足或 30 分钟保护时限会停止并保存已有状态，不追加步骤或修改配置。

损失在五帧全部有效点上计算：正常点和异常点各自求二分类 logits 损失均值，两类各占一半；仅存在一类时使用该类完整均值。忽略点不参与分母和损失，但保留输入观测。该配方仅是本次学习诊断的起点。

检查时刻为 0、40、80、120、160、200 步。每个窗口每次检查均以 `23 + window_index` 重置 Python、NumPy、PyTorch CPU 和 CUDA 随机状态，前向后恢复训练随机状态；使用 `eval()` 和无梯度推理，并确认 BatchNorm 运行统计未改变。同一检查点连续前向两次，记录实际分数、损失和 AP 差异，不要求 GPU 跨运行逐位一致。八个联合体素输入只整理一次，复用前以首个真实完整窗口与原整理路径比较，容差为 `atol=1e-6, rtol=1e-5`。

全窗口分别记录正常类、异常类和总损失。当前帧 AP 直接沿用官方筛选：忽略语义 0，语义 2 为异常，其余非零语义为正常，只保留闭区间 2.5–50 米内的点；有效异常点少于 5 时记录不可评价。逐窗 AP 以百分数表示，汇总时对可评价窗口求均值和中位数，不把重复前向当作独立样本。分数分布同时明确记录全窗口有效点和当前帧官方评价点两种范围，各含正常点中位数、正常点第 95 百分位和异常点中位数。

```bash
PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  .venv/bin/python -u -m src.train \
  --data-root /absolute/path/to/STU --output runs/learn
```

已有输出目录会被拒绝，不覆盖此前证据。`runs/learn/metrics.jsonl` 记录固定配置、完整访问顺序、源数据身份、逐步更新与各检查点结果；`predictions/step_*/` 保存两次前向的全部点分数及身份，历史四帧不丢弃。`initial.pt` 保存初始状态，`final.pt` 同时保存模型、优化器、损失缩放器、随机状态、计划／成功步骤计数及后续访问位置。检查点只加载自行生成的可信文件；含 NumPy 随机状态时使用 `torch.load(..., weights_only=False)`。

本机已完成这一次固定配方实验：八个窗口共 5,009,637 个可见点，200 个计划步骤全部成功，每窗恰好访问 25 次，溢出跳步为 0，损失缩放保持 128。第一次更新使骨干 12,726,431 个参数元素、异常头全部 2,657 个参数元素发生变化。最终状态中全部 203 个优化器参数状态的步数均为 200。

以下全部是**训练子集上的拟合诊断**。损失是固定推理模式下八个全窗口的均值，AP 是八个当前帧的逐窗结果汇总；不是 201 验证或真实 STU 测试成绩。

| 计划步骤 | 正常类平均损失 | 异常类平均损失 | 总损失 | 当前帧 AP 均值（%） | 当前帧 AP 中位数（%） |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 0.663256 | 0.729808 | 0.696532 | 0.8807 | 0.6749 |
| 40 | 0.343440 | 0.908632 | 0.626036 | 20.1637 | 9.0452 |
| 80 | 0.117764 | 1.713136 | 0.915450 | 22.4118 | 4.6910 |
| 120 | 0.041174 | 1.292886 | 0.667030 | 67.1465 | 82.4132 |
| 160 | 0.015084 | 1.151241 | 0.583163 | 75.4649 | 84.3480 |
| 200 | 0.008283 | 0.247967 | 0.128125 | 97.5719 | 99.5007 |

| 当前帧 | 初始 AP（%） | 最终 AP（%） | 最终正常分数中位数 | 最终正常分数第 95 百分位 | 最终异常分数中位数 |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 27 | 0.2154 | 99.3455 | 0.003608 | 0.031262 | 0.990420 |
| 83 | 1.1019 | 96.7668 | 0.003268 | 0.033467 | 0.965638 |
| 139 | 0.1714 | 100.0000 | 0.001612 | 0.028037 | 0.960714 |
| 195 | 0.1385 | 99.6560 | 0.002172 | 0.032558 | 0.982854 |
| 251 | 0.2777 | 86.3942 | 0.000860 | 0.024190 | 0.553563 |
| 307 | 1.4358 | 98.5806 | 0.002122 | 0.029901 | 0.994625 |
| 363 | 1.0721 | 99.9792 | 0.004852 | 0.036479 | 0.992729 |
| 448 | 2.6326 | 99.8532 | 0.002060 | 0.029372 | 0.993624 |

表中分数使用当前帧官方评价点范围。八个窗口均具备评价资格且 AP 均改善，正常类损失均下降，异常类损失有七个窗口下降。251 是例外：全窗口异常类损失由 0.737089 升至 1.118747，最终当前帧异常分数中位数为 0.553563，明显低于其余七窗的 0.960714–0.994625。因此结果支持当前模型和全窗口监督能够学习这些已见合成样本，但不能称所有窗口已经充分拟合。中间检查的异常类损失先升后降，也不能表述为训练与推理损失始终同步下降；尚未隔离归一化统计、内部随机机制或其他因素的贡献。

六个检查点共 48 对重复前向，实测逐点分数最大绝对差、AP 差和损失差均为 0；这是本次环境的实测结果，不是跨设备逐位复现承诺。全部 96 份预测已通过完整窗口身份回读，覆盖 60,115,644 条点分数记录；调用官方完整指标计算程序重算的 AP 与日志之差为 0，两种点范围的分位数也与保存预测一致。另在独立进程中分别恢复 `initial.pt` 和 `final.pt`，各重跑全部八窗，分数、AP 和总损失与对应保存结果的最大绝对差也均为 0，没有执行额外优化器更新。

从初始状态保存到最终状态保存共 117.03 秒，不含冻结数据载入和一次性输入整理；其中 200 个训练步骤合计 51.02 秒，首次 5.56 秒，其余步骤平均 0.228 秒。完整训练步骤包含优化器状态和八窗输入复用，其峰值已分配／已预留显存为 6.38／7.23 GiB。全部本次产物共 501,076,127 字节（约 477.9 MiB），不据此推算完整池的显存上界或训练时间。当前代码测试 45 项通过，未启用涉及 201 的完整数据测试；没有执行 201 模型选择、真实异常评价或额外训练。

本次证据不支持泛化能力、五帧优于单帧或超过 STU 基线的结论；也没有证据要求更换骨干、重抽异常世界或重建数据池。下一项能改变研究判断的是在独立固定验证数据上检验泛化，而非继续把已见八窗的拟合分数当成正式性能。

## 运行时几何输入

`artifacts/calibration.pt` 由 206 全部 449 帧建立，射线参数来源为 `artifacts/calibration_source.npz`。206 支撑面池 `artifacts/training_206_support_pool.npz` 覆盖 0–448，锚点范围为 2–446。各文件由 `protocol.json` 中的哈希绑定。

201 的活动支撑面池必须覆盖完整 0–681，锚点范围为 2–679。首次建立时执行：

```bash
python src/prepare.py support-validation \
  --data-root /absolute/path/to/STU --processes <按实测资源确定>
```

程序写出 `artifacts/validation_201_support_pool.npz` 并打印文件 SHA-256，应与 `protocol.json` 对应字段一致。该文件覆盖 0–681，包含 1,210,186 个合格支撑记录和 640 个实际产生合格记录的锚点帧。

201 帧 0–3 的文件槽数分别为 393,216、393,216、291,328 和 262,144，它们包含协议文档记录的精确重复槽。复现时不得先行删除或去重；读取和渲染程序会校验 `xyzi`、标签和冻结的多对一射线布局。

## 先导生成

正式生成前，只选择一个训练片段写入 `runs/`，测量耗时、内存和稀疏文件大小：

```bash
python src/data.py generate --pool train \
  --data-root /absolute/path/to/STU \
  --sequence-indices 0 --segment-indices 0 \
  --output-directory runs/pilot
```

先导必须确认固定根种子可重复、未命中异常的原始槽位逐位不变、异常标签与新回波同步、片段内每帧只渲染一次、窗口引用相同的渲染帧。先导结果不能加入正式清单，检查结束后应从 `runs/pilot` 删除。

每段必须严格包含一个异常代理物体。生成端、冻结片段读取端和资格检查都执行这一要求。

物体放置采用 `PROTOCOL.md` 中的尾部优先规则。资格结果记录末帧可见性和物理回退情况，不因窗口没有异常回波而删除窗口。

## 正式池生成

正式生成前必须重新检查 CPU、内存、交换空间、GPU 占用、其他实验进程和 Windows E 盘实际剩余空间。若预计峰值新增超过 1 GiB，应从 Windows 侧执行：

```powershell
Get-Volume -DriveLetter E | Select-Object Size,SizeRemaining
```

生成期间须保留 E 盘总容量 5% 与 10 GiB 中较大的安全余量。当前实现按片段保存稀疏变化，不复制未变化的官方点，也不为重叠窗口重复保存点数组。

确认资源安全后，生成完整池。本轮实测资源允许使用 6 个进程，复现时应重新确定适用的进程数：

下面的生成与建立清单命令用于首次建池阶段。当前池已冻结，不再对正式路径运行这些命令；读取既有池使用上面的训练数据入口。若需复现原始生成文件，应使用原资格证据记录的生成实现，不能把当前消费接口的源码哈希替换进原清单。

```bash
python src/data.py generate --pool train \
  --data-root /absolute/path/to/STU --workers 6
python src/data.py manifest --pool train

python src/data.py generate --pool validation \
  --data-root /absolute/path/to/STU --workers 6
python src/data.py manifest --pool validation
```

中断后只能使用 `--resume` 复用已经存在且身份完全一致的正式片段。它不会更换根种子，也不会覆盖内容冲突的文件。训练池应包含 128 个片段世界和 3080 个窗口；合成验证池应包含 92 个片段世界和 2360 个窗口。

## 冻结完成条件

生成完整池后执行：

```bash
python src/qualify.py --data-root /absolute/path/to/STU
```

当前状态为 `frozen`，该命令默认只读复核，不重写 `artifacts/data/qualification.json`。如需另外保存结果，指定尚不存在的路径，例如 `--output runs/qualification.json`；指向冻结资格文件或任意已有文件时都会拒绝执行。复核重新检查已有世界；同种子确定性检查只在内存重放既定首段，不重抽种子、不挑选替代世界，也不写回数据池。两份冻结清单同样禁止覆盖。

需要逐窗口人工核查时，可在同一次检查中导出完整点云：

```bash
python src/qualify.py --data-root /absolute/path/to/STU \
  --ply-directory artifacts/ply
```

该选项导出训练池的 3080 个窗口和合成验证池的 2360 个窗口，不导出原始正常 201。路径为 `artifacts/ply/{train,validation}/sequence_000/window_000004.ply` 等，文件名中的帧号为当前帧。每个二进制 PLY 保留五帧全部可见点和当前帧 LiDAR 坐标；灰色 `(160,160,160)` 为正常，红色 `(255,0,0)` 为异常，蓝色 `(0,128,255)` 为官方忽略。颜色只用于核查，不能作为模型特征。

完整 PLY 约占 45.93 GiB，导出前必须按实际点数和 Windows E 盘剩余空间重新确认安全余量，运行期间持续监测。已有且内容完全一致的文件可复用，冲突文件不会被覆盖。PLY 不纳入 Git，也不作为模型输入；正式数据始终来自冻结的稀疏片段文件。

另有用户授权的真实异常目视样本：`artifacts/ply/real/125/window_000004.ply` 至 `window_000018.ply`，共 15 个文件、141,149,315 字节（134.61 MiB）。选择协议列出的第一条真实序列 `val/125`，取最早连续 15 个当前帧均有可见异常的窗口，仅需读取第 0–18 帧点云和标签。导出复用 `STUSequence`、`WindowPartition(sequence, 4, 18)` 和 `src/qualify.py` 的 `_write_window_ply`，未修改数据读取或导出实现。坐标约定和配色与合成窗口相同，未下采样；每窗有 120–126 个异常点。15 个 PLY 均已回读并逐行核对坐标与官方真值颜色，未运行模型、计算性能指标或修改任何合成数据。

模型无关资格程序必须进一步核对所有正式片段、重建帧、窗口和原始 201 的 678 个在线输出。只有以下三个文件全部生成且通过独立检查后，才允许把其 SHA-256 写入协议：

```text
artifacts/data/train_manifest.json
artifacts/data/validation_manifest.json
artifacts/data/qualification.json
```

随后才可把协议状态改为 `frozen`，并同步设置 `data_pool_frozen=true`、`training_allowed=true`、`validation_tuning_allowed=true`。`real_anomaly_access_allowed` 仍保持 `false`，直到模型和选择规则另外完成冻结。
