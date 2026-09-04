# AJAE 代码与输入复现说明

AJAE 直接使用的项目代码、第三方源码、冻结权重、射线几何源和运行时输入均位于当前工作区。代码不再读取任何兄弟工作区。Python、CUDA、编译器和已安装软件包仍属于宿主环境，不复制进代码仓库。

## 工作区内的第三方代码

第三方源码身份如下：

- `vendor/stu/Mask4Former3D`：STU 官方提交 `8f0f09c2ca4bf7b665e0ae5919b4092ddae140a2`。AJAE 未修改其模型源码，许可证见 `vendor/stu/LICENSE`。
- `vendor/stu/compute_point_level_ood.py`：同一提交的官方点级评价程序，文件哈希由 `protocol.json` 绑定。
- `vendor/pytorch3d`：PyTorch3D 提交 `f34104cf6ebefacd7b7e07955ee7aaa823e616ac` 的最小运行子集。它只编译 AJAE 使用的最远点采样算子，并包含 CUDA 13 构建兼容修改；BSD 许可证和第三方许可证均保留。
- `vendor/MinkowskiEngine`：MinkowskiEngine 提交 `02fc608bea4c0549b0a7b00ca1bf15dee4a0b228`，包含 Python 3.13、PyTorch 2.12 和 CUDA 13 所需的构建兼容修改；MIT 许可证保留。

`src/model.py` 默认只把当前工作区的 `vendor/stu/Mask4Former3D` 加入模块搜索路径。MinkowskiEngine 和 PyTorch3D 的二进制扩展由宿主 Python 环境加载，但与其对应的源码已经保存在上述 `vendor/` 目录，不再依赖其他工作区。

权重和大型 NPZ 运行时输入使用 Git LFS。克隆后应运行 `git lfs pull`，并用 `python -m src.qualify` 核验权重、STU 源码和协议身份。

五帧实验不使用 STU 数据集中的多扫描时间编号；五帧回波对齐后全部作为同一时刻的稠密空间观测送入冻结 STU，因此研究对象是空间观测密度，而不是带时间标识的时序建模。

## 准备数据与运行时输入

STU 原始数据约 24 GiB，不随代码仓库再分发。下载官方压缩包后，应先按 `protocol.json` 的 `data.official_archive_sha256` 校验 `train.zip`、`val.zip` 和 `test.zip`，再整理为以下结构：

```text
STU/
├── train/201/{velodyne,labels,poses.txt,calib.txt}
├── train/206/{velodyne,labels,poses.txt,calib.txt}
├── val/<sequence>/{velodyne,labels,poses.txt,calib.txt}
└── test/<sequence>/{velodyne,poses.txt,calib.txt}
```

仓库已保存并由协议绑定以下运行时输入：

- `artifacts/e11_d4b_calibration.npz`：生成正式射线网格所需的小型 E11 几何源；
- `artifacts/calibration.pt`：从正式射线网格和 `train/206` 全部 449 帧重新估计的传感器统计；
- `artifacts/development_201_support_pool.npz`：F1–F3 开发池，只使用 `train/201` 的 4–553 帧，支撑面估计锚点帧为 6–551；
- `artifacts/training_206_support_pool.npz`：若进入 F4 时使用的训练池，使用 `train/206` 的 0–448 帧，支撑面估计锚点帧为 2–446。

若需要从官方原始数据重建并逐字节核对三个运行时产物，执行：

```bash
python -m src.prepare all --data-root /absolute/path/to/STU --processes 24
```

也可以分别选择 `calibration`、`support-development` 或 `support-training`。生成器按锚点帧顺序合并并写出确定性 NPZ，进程数只影响运行时间。生成结束后会自动比较 `protocol.json` 中的 SHA-256；不一致会直接报错。

## 宿主环境边界

以下内容仍由宿主机提供：Linux x86-64 内核、Python 环境、兼容 CUDA 的 NVIDIA 驱动和 GPU、CUDA 工具链、编译器、Python 软件包，以及用户自行下载的官方 STU 数据。这些是环境或数据条件，不是其他工作区的代码依赖。

当前已实际核验的参考环境为 Python 3.13.12、PyTorch 2.12.0（CUDA 13.0 构建）、CUDA 工具链 13.2.78、GCC/G++ 14.3.0 和 NVIDIA GeForce RTX 5080 Laptop GPU。MinkowskiEngine 0.5.4 与最小 PyTorch3D 0.7.6 均已从本工作区 `vendor/` 源码成功构建并执行 CUDA 算子。不同 CUDA、编译器和 PyTorch 组合可能需要重新编译这两个扩展。

当前参考环境中的 CUDA/MinkowskiEngine 路径未通过同一输入重复前向检查：逐点 MaxLogit 和类别会变化；CPU 路径重复前向完全一致，并已在 `train/201` 的帧 8、198、387 上通过官方 `sweep=1` 路径与 AJAE 编码器的端到端等价检查，三帧 MaxLogit 最大绝对误差均为 0，逆映射和类别均逐点相同。因此 schema 33 暂时只授权 CPU 执行 F2/F3。重新授权 GPU 前必须先修复当前重复性问题，并在相同真实帧上重新通过重复前向和官方等价检查；不能仅凭 CUDA 算子可运行就用于正式实验。
