# AJAE 数据协议复现说明

当前活动协议是 schema 34。科学定义见 `PROTOCOL.md`，机器可检查合同见 `protocol.json`。

当前正式数据池已生成并通过完整窗口检查，协议状态为 `frozen`：206 训练池包含 128 个单代理世界、3080 个窗口；201 合成验证池包含 92 个单代理世界、2360 个窗口；原始完整 201 另有 678 个在线窗口。完整检查记录见 `artifacts/data/qualification.json`。这说明数据实现通过本协议检查，不代表模型效果已经得到验证。

训练池和合成验证池的 5440 个真值着色 PLY 已全部导出至 `artifacts/ply/`，共 49,311,794,379 字节（45.93 GiB）。220 个正式世界在末帧均有可见异常回波；训练池仍有 20 个、合成验证池仍有 10 个全窗口无异常回波的样本，均按规则保留。末帧物理可见不等于进入官方 2.5–50 米评价范围，资格记录分别保存两种计数。

`vendor/stu/` 保存官方点级评价源码和许可证，供核对官方语义使用；模型训练和最终评价的完成情况应以真实运行结果为准。

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
pytest -q
ruff check src tests
```

协议摘要应显示 schema 34、206 训练源、201 完整验证源、3080 个合成训练窗口和 2360 个合成验证窗口。正式池尚未完成时，状态必须保持 `qualification_pending`，`training_allowed` 必须为 `false`。

## 正式训练数据入口

仓库目前尚未实现模型训练循环。正式训练数据入口已经实际执行冻结清单核验，不依赖调用者先手动运行检查命令：

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
