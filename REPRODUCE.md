# AJAE 数据协议复现说明

当前活动路线是 schema 34“数据与五帧监督协议 v2”。科学定义见 `AJAE数据与五帧监督协议v2.md`，机器可检查合同见 `protocol.json`。旧 schema 33 协议快照位于 `history/schema33/protocol.json`，只用于解释保留的 F0/F1 产物；旧路线文档、F2/F3 入口和旧冻结 STU 模型入口均已删除。

## 原始数据

STU 原始数据不随仓库分发。先按 `protocol.json` 的 `data.official_archive_sha256` 核对官方压缩包，再整理为：

```text
STU/
├── train/201/{velodyne,labels,poses.txt,calib.txt}
├── train/206/{velodyne,labels,poses.txt,calib.txt}
├── val/<sequence>/{velodyne,labels,poses.txt,calib.txt}
└── test/<sequence>/{velodyne,poses.txt,calib.txt}
```

活动协议只在数据冻结阶段读取 `train/206` 和 `train/201`。19 条真实异常 `val` 序列与隐藏 `test` 序列仍保持封存。

## 代码检查

安装依赖后，在仓库根目录运行：

```bash
python src/protocol.py
pytest -q
ruff check src tests
```

协议摘要应显示 schema 34、206 训练源、201 完整验证源、3080 个合成训练窗口和 2360 个合成验证窗口。正式池尚未完成时，状态必须保持 `qualification_pending`，`training_allowed` 必须为 `false`。

## 运行时几何输入

现有 `artifacts/calibration.pt` 由 206 全部 449 帧建立，并由协议中的文件哈希绑定。206 支撑面池 `artifacts/training_206_support_pool.npz` 已覆盖 0–448，锚点范围为 2–446，可以直接继承。

201 的活动支撑面池必须覆盖完整 0–681，锚点范围为 2–679。首次建立时执行：

```bash
python src/prepare.py support-validation \
  --data-root /absolute/path/to/STU --processes <按实测资源确定>
```

程序写出 `artifacts/validation_201_support_pool.npz` 并打印文件 SHA-256。当前冻结候选文件的 SHA-256 为 `1414387f046a674a115138a3e4f525e76aa4c5b006a609d6cc6727f1f2ead99f`，覆盖 0–681，包含 1,210,186 个合格支撑记录和 640 个实际产生合格记录的锚点帧。旧 4–553 开发区支撑池已删除。

201 帧 0–3 的文件槽数分别为 393,216、393,216、291,328 和 262,144，它们包含协议文档记录的精确重复槽。复现时不得先行删除或去重；读取和渲染程序会校验 `xyzi`、标签和冻结的多对一射线布局。

## 先导生成

正式生成前，只选择一个训练片段写入 `runs/`，测量耗时、内存和稀疏文件大小：

```bash
python src/data.py generate --pool train_v1 \
  --data-root /absolute/path/to/STU \
  --sequence-indices 0 --segment-indices 0 \
  --output-directory runs/ajae/data_v2_pilot
```

先导必须确认固定根种子可重复、未命中异常的原始槽位逐位不变、异常标签与新回波同步、片段内每帧只渲染一次、窗口引用相同的渲染帧。先导结果不能加入正式清单，检查结束后应从 `runs/ajae/data_v2_pilot` 删除。

## 正式池生成

正式生成前必须重新检查 CPU、内存、交换空间、GPU 占用、其他实验进程和 Windows E 盘实际剩余空间。若预计峰值新增超过 1 GiB，应从 Windows 侧执行：

```powershell
Get-Volume -DriveLetter E | Select-Object Size,SizeRemaining
```

生成期间须保留 E 盘总容量 5% 与 10 GiB 中较大的安全余量。当前实现按片段保存稀疏变化，不复制未变化的官方点，也不为重叠窗口重复保存点数组。

确认资源安全后，生成完整池：

```bash
python src/data.py generate --pool train_v1 \
  --data-root /absolute/path/to/STU --workers 3
python src/data.py manifest --pool train_v1

python src/data.py generate --pool validation_v1 \
  --data-root /absolute/path/to/STU --workers 3
python src/data.py manifest --pool validation_v1
```

中断后只能使用 `--resume` 复用已经存在且身份完全一致的正式片段。它不会更换根种子，也不会覆盖内容冲突的文件。训练池应包含 128 个片段世界和 3080 个窗口；合成验证池应包含 92 个片段世界和 2360 个窗口。

## 冻结完成条件

模型无关资格程序必须进一步核对所有正式片段、重建帧、窗口和原始 201 的 678 个在线输出。只有以下三个文件全部生成且通过独立检查后，才允许把其 SHA-256 写入协议：

```text
artifacts/data_v2/train_manifest.json
artifacts/data_v2/validation_manifest.json
artifacts/data_v2/qualification.json
```

随后才可把协议状态改为 `frozen`，并同步设置 `data_pool_frozen=true`、`training_allowed=true`、`validation_tuning_allowed=true`。`real_anomaly_access_allowed` 仍保持 `false`，直到模型和选择规则另外完成冻结。

## 历史证据边界

`artifacts/f0_qualification.json` 的 SHA-256 为 `b78bce85ff336469583611d2a756ee8070590f9dd6cf9d6ec9537ce08fccf6ef`；它只说明旧路线的官方 STU 输入与执行等价。`artifacts/f1_geometry.json` 的 SHA-256 为 `add308d89ac130cc59401685fc1295cf4d621652f8f2d885e32c74a7e9394b3b`；它只说明旧开发区间 546 个窗口的几何稠密性。这两个结果不能证明 schema 34 数据池已经生成、标签正确或训练有效。
