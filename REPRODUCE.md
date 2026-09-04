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

五帧 STU 输入与官方 `LidarDataset(sweep=5)` 使用相同的时间升序文件槽、官方世界坐标、全扫描中心距离和 5 厘米体素化。模型实际使用的两个输入通道仍是强度和距离；官方整理路径产生的扫描时间编号只存在于原始坐标附加列，当前 `Mask4Former3D` 的位置编码不读取该列。因此研究对象是空间观测密度，而不是带时间标识的时序建模。

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

`artifacts/f0_qualification.json` 保存正式 F0 证据：固定真实单帧与五帧窗口的官方输入、体素、逆映射、MaxLogit、类别和 CPU 重复性比较。重新核验时执行：

```bash
python -m src.qualify --data-root /absolute/path/to/STU --device cpu \
  --output artifacts/f0_qualification.json
```

真实 F0 和正式 F1--F3 都拒绝从存在已跟踪文件修改的工作区生成结果；`runs/` 中未跟踪的实验输出不影响该检查。

当前保存的 F0 证据由干净提交 `804f92998409bec2d5ded3f92beaa5c86b95c08e` 生成，科学合同身份为 `5330228fcc93c400b0154ea7f57c53cf11b90d2f355be2bdcc478941c584a353`。单帧 8、198、387 以及窗口 4--8、194--198、383--387 的全部输入、体素和逆映射均一致；官方与 AJAE、AJAE 两次重复以及当前帧视图的 MaxLogit 最大绝对误差均为 0，类别不一致数均为 0。该证据只确认 F0 输入与执行语义，不是 F1 几何结果或性能证据。

若需要从官方原始数据重建并逐字节核对三个运行时产物，执行：

```bash
python -m src.prepare all --data-root /absolute/path/to/STU --processes 24
```

也可以分别选择 `calibration`、`support-development` 或 `support-training`。生成器按锚点帧顺序合并并写出确定性 NPZ，进程数只影响运行时间。生成结束后会自动比较 `protocol.json` 中的 SHA-256；不一致会直接报错。

## 宿主环境边界

以下内容仍由宿主机提供：Linux x86-64 内核、Python 环境、兼容 CUDA 的 NVIDIA 驱动和 GPU、CUDA 工具链、编译器、Python 软件包，以及用户自行下载的官方 STU 数据。这些是环境或数据条件，不是其他工作区的代码依赖。

当前已实际核验的参考环境为 Python 3.13.12、PyTorch 2.12.0（CUDA 13.0 构建）、CUDA 工具链 13.2.78、GCC/G++ 14.3.0 和 NVIDIA GeForce RTX 5080 Laptop GPU。MinkowskiEngine 0.5.4 与最小 PyTorch3D 0.7.6 均已从本工作区 `vendor/` 源码成功构建并执行 CUDA 算子。不同 CUDA、编译器和 PyTorch 组合可能需要重新编译这两个扩展。

当前参考环境中的 CUDA/MinkowskiEngine 路径未通过同一输入重复前向检查：逐点 MaxLogit 和类别会变化；因此当前执行状态暂时只授权 CPU 执行 F2/F3。重新授权 GPU 前必须先修复当前重复性问题，并在相同真实帧和窗口上重新通过重复前向与官方等价检查；不能仅凭 CUDA 算子可运行就用于正式实验。

协议身份分为两层。`contract_identity` 只绑定 F0--F3 不可变的科学问题、数据、输入、渲染、STU 和评价规则；阶段、完成声明及 CPU/GPU 授权属于可变执行状态，不进入该身份。`protocol_file_sha256` 则绑定当次运行使用的完整 `protocol.json`，因此执行状态变化仍可逐次追踪。
