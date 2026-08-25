# AJAE 工作区落实情况全量自查

Audit date: 2026-08-25
Git commit: `8e61059a91e7dca4d72bdd0d593b450af3add1fd`
Git branch: `main`
Workspace root: `/home/jasongao/Study/AJAE`
Dirty working tree: yes
CUDA / PyTorch environment: Python 3.13.12；PyTorch 2.12.0+cu130；CUDA available=true；GPU=NVIDIA GeForce RTX 5080 Laptop GPU（16,303 MiB）；MinkowskiEngine 0.5.4；Hydra/OmegaConf/PyTorch3D 未安装
Auditor: Codex（只读工作区审计；未训练、未改协议、未访问 19/51 确认集）

## 总览

统计口径：652 个编号检查项，加 13 个 U 运行检查和 6 个 STATUS 核验，共 671 项；Decision Gates 另列，不重复计数。

P0 critical violations: 2 个明确 FAIL（A04-01、A04-07）；45 个 P0 条目中另有 29 个尚未达到 PASS。
P1 protocol violations: 17 个 P0 之外的明确 FAIL，涉及公开标签旁路、201 世界覆盖/冻结、Gate1 来源泄漏、传感器审计、官方 STU 运行、坐标测试、V 分层、缓存身份和配置 生成链路。
P2 missing diagnostics: 至少 63 个直接诊断/评价条目为 NOT RUN，另有相关诊断因 Gate1 处于 BLOCKED。
PASS: 95
PARTIAL: 326
FAIL: 20
NOT RUN: 118
BLOCKED: 106
N/A: 6

## 审计范围与证据纪律

- 当前权威路线按工作区中的 schema 30 `protocol.json`、`dev.json`、`src/` 和 `AJAE新主线方案.md` 判断；Git HEAD 本身不能复现当前路线，因为多个核心文件未跟踪或未提交。
- 代码存在、toy test 通过和静态结构正确只记作实现证据；没有正式运行、可追溯产物或官方复算时不判 PASS。
- 本次只执行只读、非训练、非确认集检查；没有写入 run state，没有读取 19/51 标签或结果。唯一新增文件是本审计报告。
- 当前 `runs/ajae` 只有 `calibration.pt`（10,024,791 bytes，SHA256 `44f0c1779bd8a50589d4b56417b23773bc85a51bf0f7c12da0a670d2ffc8f39e`），没有 B0–B5 正式产物。

## 本次真实执行记录

1. `PYTHONDONTWRITEBYTECODE=1 python -m pytest -q -p no:cacheprovider`：25 passed in 6.99s。
2. `ruff check --no-cache src test_ajae.py`：All checks passed。
3. `git diff --check`：通过。
4. `PYTHONDONTWRITEBYTECODE=1 python src/protocol.py --development`：schema 30；24+6 worlds；`definitions_only_unvalidated`；Gate1 pending；validated=false。
5. `PYTHONDONTWRITEBYTECODE=1 python src/train.py --condition B1 --data-root /home/jasongao/Data/STU --max-worlds 5 --device cpu`：退出码 1；因 Gate1、fixed-201 evaluator/scope、checkpoint rule、decision criteria 和 maximum_worlds 未冻结而拒绝，且明确未写 run state。
6. 真实 train/206 frame 0 官方 STU 前向只读尝试：`FrozenSTUPointEncoder.from_protocol(...)` 在构造阶段因 `ModuleNotFoundError: No module named hydra` 失败；没有反向、更新或输出产物。
7. `PYTHONDONTWRITEBYTECODE=1 python - <<...>>` 环境探测：MinkowskiEngine=True，hydra=False，omegaconf=False，pytorch3d=False，PyTorch 2.12.0+cu130，CUDA available=true。
8. `src/scene.py --check-all` 对真实 train/201 B3 窗口只读检查：682 源帧，合法中心帧 6–679，共 674 个完整中心窗口。

## 关键产物身份

| 对象 | 状态/散列 |
| --- | --- |
| `protocol.json` | schema 30；SHA256 `6661c4d8602bf962a9f3b10cd5c89dea79fa89f37ea16f0f748682fe7eb03ee3` |
| `dev.json` | untracked；definitions-only；SHA256 `b811dc3a94bbea87acfa9a4caeeb181cbc2eb44b02dbeb806e7ec95f21e0c4cf` |
| `runs/ajae/calibration.pt` | ignored；SHA256 `44f0c1779bd8a50589d4b56417b23773bc85a51bf0f7c12da0a670d2ffc8f39e` |
| `weights/59p6pq_ens1.ckpt` | SHA256 `743b10d39c4076d98533bf1e84d389ad2703016904d31146e48919618b07b67a` |
| `weights/59p6pq_ens1.model_state.pt` | SHA256 `bd62c2ace0fd13911e2ba81f4969ca6633e73ec5270ffc0b1bd61840b05f924d` |

## 逐项审计

## A00：完整世界先于五帧窗口

### [A00-01] 状态: PARTIAL

要求:
检查反事实实体、位置、材质、随机种子是否首先作为完整世界规格 $\Omega$ 固定。

发现:
`WorldSpec` 在窗口遍历前保存实体几何、位置、材质与种子，动态世界训练代码也按世界后按合法中心帧遍历；但真实的“三个世界×五个重叠窗口”共享帧逐槽检查没有运行，现有确定性证据仅覆盖控制世界三帧和小夹具。

证据:
- src/render.py:1356-1441
- src/train.py:1549-1658
- test_ajae.py:535-545

判断:
只能证明静态实现、局部运行或缺少 生成链路 的产物，尚不能证明可追溯正式路径完整采用；按审计规则不得升级为 PASS。

需要修改:
转为 PASS 所需条件：按固定世界规格运行 A00-06/U01 的重叠窗口逐槽一致性检查，并保存命令、提交、seed 和结果。

### [A00-02] 状态: PARTIAL

要求:
确认不是每取一个五帧窗口就重新随机生成异常。

发现:
`WorldSpec` 在窗口遍历前保存实体几何、位置、材质与种子，动态世界训练代码也按世界后按合法中心帧遍历；但真实的“三个世界×五个重叠窗口”共享帧逐槽检查没有运行，现有确定性证据仅覆盖控制世界三帧和小夹具。

证据:
- src/render.py:1356-1441
- src/train.py:1549-1658
- test_ajae.py:535-545

判断:
只能证明静态实现、局部运行或缺少 生成链路 的产物，尚不能证明可追溯正式路径完整采用；按审计规则不得升级为 PASS。

需要修改:
转为 PASS 所需条件：按固定世界规格运行 A00-06/U01 的重叠窗口逐槽一致性检查，并保存命令、提交、seed 和结果。

### [A00-03] 状态: PARTIAL

要求:
确认同一世界、同一帧进入不同重叠窗口时，渲染结果逐点完全一致。

发现:
`WorldSpec` 在窗口遍历前保存实体几何、位置、材质与种子，动态世界训练代码也按世界后按合法中心帧遍历；但真实的“三个世界×五个重叠窗口”共享帧逐槽检查没有运行，现有确定性证据仅覆盖控制世界三帧和小夹具。

证据:
- src/render.py:1356-1441
- src/train.py:1549-1658
- test_ajae.py:535-545

判断:
只能证明静态实现、局部运行或缺少 生成链路 的产物，尚不能证明可追溯正式路径完整采用；按审计规则不得升级为 PASS。

需要修改:
转为 PASS 所需条件：按固定世界规格运行 A00-06/U01 的重叠窗口逐槽一致性检查，并保存命令、提交、seed 和结果。

### [A00-04] 状态: PARTIAL

要求:
找到世界 seed 的生成、保存和恢复机制。

发现:
`WorldSpec` 在窗口遍历前保存实体几何、位置、材质与种子，动态世界训练代码也按世界后按合法中心帧遍历；但真实的“三个世界×五个重叠窗口”共享帧逐槽检查没有运行，现有确定性证据仅覆盖控制世界三帧和小夹具。

证据:
- src/render.py:1356-1441
- src/train.py:1549-1658
- test_ajae.py:535-545

判断:
只能证明静态实现、局部运行或缺少 生成链路 的产物，尚不能证明可追溯正式路径完整采用；按审计规则不得升级为 PASS。

需要修改:
转为 PASS 所需条件：按固定世界规格运行 A00-06/U01 的重叠窗口逐槽一致性检查，并保存命令、提交、seed 和结果。

### [A00-05] 状态: PARTIAL

要求:
验证重新执行同一个 world specification 可以得到完全相同结果。

发现:
`WorldSpec` 在窗口遍历前保存实体几何、位置、材质与种子，动态世界训练代码也按世界后按合法中心帧遍历；但真实的“三个世界×五个重叠窗口”共享帧逐槽检查没有运行，现有确定性证据仅覆盖控制世界三帧和小夹具。

证据:
- src/render.py:1356-1441
- src/train.py:1549-1658
- test_ajae.py:535-545

判断:
只能证明静态实现、局部运行或缺少 生成链路 的产物，尚不能证明可追溯正式路径完整采用；按审计规则不得升级为 PASS。

需要修改:
转为 PASS 所需条件：按固定世界规格运行 A00-06/U01 的重叠窗口逐槽一致性检查，并保存命令、提交、seed 和结果。

### [A00-06] 状态: NOT RUN

要求:
随机抽至少 3 个世界、每个世界至少 5 个重叠窗口，对共享帧做 hash/逐槽一致性检查。

发现:
没有运行要求的至少 3 个世界、每个至少 5 个重叠窗口的共享帧 hash/逐槽检查。

证据:
- src/render.py:1356-1441
- src/train.py:1549-1658
- test_ajae.py:535-545

判断:
本条要求的运行级审计、实验或正式产物没有发生；代码存在也不能替代执行，判定 NOT RUN。

需要修改:
转为 PASS 所需条件：按固定世界规格运行 A00-06/U01 的重叠窗口逐槽一致性检查，并保存命令、提交、seed 和结果。

### [A00-07] 状态: PASS

要求:
如果当前实现是“window-level augmentation”，直接 FAIL。

发现:
`WorldSpec` 在窗口遍历前保存实体几何、位置、材质与种子，动态世界训练代码也按世界后按合法中心帧遍历；但真实的“三个世界×五个重叠窗口”共享帧逐槽检查没有运行，现有确定性证据仅覆盖控制世界三帧和小夹具。

证据:
- src/render.py:1356-1441
- src/train.py:1549-1658
- test_ajae.py:535-545

判断:
证据直接覆盖该要求，或排除性全仓库搜索已证明冲突路径不在当前正式路线；可判 PASS。该结论只覆盖本条，不外推到整个模块或科学主张。

需要修改:
无需修改；保留当前实现和直接证据，并继续确保该局部结论不被外推为模块或科学主张已经成立。

## A01：正常控制和异常代理共享同一传感器渲染器

### [A01-01] 状态: PARTIAL

要求:
找到 normal-control 的完整渲染调用链。

发现:
normal-control 与 anomaly-proxy 最终都进入 `render_frame()` 的统一射线、回波竞争、回波概率、强度和槽位恢复路径；没有发现异常专用噪声分支。不过共享代码尚未以正式混合世界产物和严格来源审计证明能够消除来源捷径。

证据:
- src/render.py:2707-3073
- src/render.py:3594-3701
- test_ajae.py:414-545

判断:
只能证明静态实现、局部运行或缺少 生成链路 的产物，尚不能证明可追溯正式路径完整采用；按审计规则不得升级为 PASS。

需要修改:
转为 PASS 所需条件：用严格匹配的真实正常、normal-control 和 anomaly-proxy 产物证明三者经过同一传感器流程且来源不可轻易区分。

### [A01-02] 状态: PARTIAL

要求:
找到 anomaly-proxy 的完整渲染调用链。

发现:
normal-control 与 anomaly-proxy 最终都进入 `render_frame()` 的统一射线、回波竞争、回波概率、强度和槽位恢复路径；没有发现异常专用噪声分支。不过共享代码尚未以正式混合世界产物和严格来源审计证明能够消除来源捷径。

证据:
- src/render.py:2707-3073
- src/render.py:3594-3701
- test_ajae.py:414-545

判断:
只能证明静态实现、局部运行或缺少 生成链路 的产物，尚不能证明可追溯正式路径完整采用；按审计规则不得升级为 PASS。

需要修改:
转为 PASS 所需条件：用严格匹配的真实正常、normal-control 和 anomaly-proxy 产物证明三者经过同一传感器流程且来源不可轻易区分。

### [A01-03] 状态: PARTIAL

要求:
验证两者调用相同 ray grid。

发现:
normal-control 与 anomaly-proxy 最终都进入 `render_frame()` 的统一射线、回波竞争、回波概率、强度和槽位恢复路径；没有发现异常专用噪声分支。不过共享代码尚未以正式混合世界产物和严格来源审计证明能够消除来源捷径。

证据:
- src/render.py:2707-3073
- src/render.py:3594-3701
- test_ajae.py:414-545

判断:
只能证明静态实现、局部运行或缺少 生成链路 的产物，尚不能证明可追溯正式路径完整采用；按审计规则不得升级为 PASS。

需要修改:
转为 PASS 所需条件：用严格匹配的真实正常、normal-control 和 anomaly-proxy 产物证明三者经过同一传感器流程且来源不可轻易区分。

### [A01-04] 状态: PARTIAL

要求:
验证两者调用相同 occlusion / nearest-return 逻辑。

发现:
normal-control 与 anomaly-proxy 最终都进入 `render_frame()` 的统一射线、回波竞争、回波概率、强度和槽位恢复路径；没有发现异常专用噪声分支。不过共享代码尚未以正式混合世界产物和严格来源审计证明能够消除来源捷径。

证据:
- src/render.py:2707-3073
- src/render.py:3594-3701
- test_ajae.py:414-545

判断:
只能证明静态实现、局部运行或缺少 生成链路 的产物，尚不能证明可追溯正式路径完整采用；按审计规则不得升级为 PASS。

需要修改:
转为 PASS 所需条件：用严格匹配的真实正常、normal-control 和 anomaly-proxy 产物证明三者经过同一传感器流程且来源不可轻易区分。

### [A01-05] 状态: PARTIAL

要求:
验证两者调用相同 return probability 模型。

发现:
normal-control 与 anomaly-proxy 最终都进入 `render_frame()` 的统一射线、回波竞争、回波概率、强度和槽位恢复路径；没有发现异常专用噪声分支。不过共享代码尚未以正式混合世界产物和严格来源审计证明能够消除来源捷径。

证据:
- src/render.py:2707-3073
- src/render.py:3594-3701
- test_ajae.py:414-545

判断:
只能证明静态实现、局部运行或缺少 生成链路 的产物，尚不能证明可追溯正式路径完整采用；按审计规则不得升级为 PASS。

需要修改:
转为 PASS 所需条件：用严格匹配的真实正常、normal-control 和 anomaly-proxy 产物证明三者经过同一传感器流程且来源不可轻易区分。

### [A01-06] 状态: PARTIAL

要求:
验证两者调用相同 intensity 模型。

发现:
normal-control 与 anomaly-proxy 最终都进入 `render_frame()` 的统一射线、回波竞争、回波概率、强度和槽位恢复路径；没有发现异常专用噪声分支。不过共享代码尚未以正式混合世界产物和严格来源审计证明能够消除来源捷径。

证据:
- src/render.py:2707-3073
- src/render.py:3594-3701
- test_ajae.py:414-545

判断:
只能证明静态实现、局部运行或缺少 生成链路 的产物，尚不能证明可追溯正式路径完整采用；按审计规则不得升级为 PASS。

需要修改:
转为 PASS 所需条件：用严格匹配的真实正常、normal-control 和 anomaly-proxy 产物证明三者经过同一传感器流程且来源不可轻易区分。

### [A01-07] 状态: PARTIAL

要求:
验证两者使用相同 empty-ray → new-return 机制。

发现:
normal-control 与 anomaly-proxy 最终都进入 `render_frame()` 的统一射线、回波竞争、回波概率、强度和槽位恢复路径；没有发现异常专用噪声分支。不过共享代码尚未以正式混合世界产物和严格来源审计证明能够消除来源捷径。

证据:
- src/render.py:2707-3073
- src/render.py:3594-3701
- test_ajae.py:414-545

判断:
只能证明静态实现、局部运行或缺少 生成链路 的产物，尚不能证明可追溯正式路径完整采用；按审计规则不得升级为 PASS。

需要修改:
转为 PASS 所需条件：用严格匹配的真实正常、normal-control 和 anomaly-proxy 产物证明三者经过同一传感器流程且来源不可轻易区分。

### [A01-08] 状态: PARTIAL

要求:
验证两者使用相同 point identity / output recovery 流程。

发现:
normal-control 与 anomaly-proxy 最终都进入 `render_frame()` 的统一射线、回波竞争、回波概率、强度和槽位恢复路径；没有发现异常专用噪声分支。不过共享代码尚未以正式混合世界产物和严格来源审计证明能够消除来源捷径。

证据:
- src/render.py:2707-3073
- src/render.py:3594-3701
- test_ajae.py:414-545

判断:
只能证明静态实现、局部运行或缺少 生成链路 的产物，尚不能证明可追溯正式路径完整采用；按审计规则不得升级为 PASS。

需要修改:
转为 PASS 所需条件：用严格匹配的真实正常、normal-control 和 anomaly-proxy 产物证明三者经过同一传感器流程且来源不可轻易区分。

### [A01-09] 状态: PASS

要求:
检查是否存在 anomaly-only 噪声、强度扰动、点稀疏化或特殊渲染分支。

发现:
normal-control 与 anomaly-proxy 最终都进入 `render_frame()` 的统一射线、回波竞争、回波概率、强度和槽位恢复路径；没有发现异常专用噪声分支。不过共享代码尚未以正式混合世界产物和严格来源审计证明能够消除来源捷径。

证据:
- src/render.py:2707-3073
- src/render.py:3594-3701
- test_ajae.py:414-545

判断:
证据直接覆盖该要求，或排除性全仓库搜索已证明冲突路径不在当前正式路线；可判 PASS。该结论只覆盖本条，不外推到整个模块或科学主张。

需要修改:
无需修改；保留当前实现和直接证据，并继续确保该局部结论不被外推为模块或科学主张已经成立。

### [A01-10] 状态: PASS

要求:
如果“是否经过渲染器”仍然等价于异常标签，直接判 P0 FAIL。

发现:
normal-control 与 anomaly-proxy 最终都进入 `render_frame()` 的统一射线、回波竞争、回波概率、强度和槽位恢复路径；没有发现异常专用噪声分支。不过共享代码尚未以正式混合世界产物和严格来源审计证明能够消除来源捷径。

证据:
- src/render.py:2707-3073
- src/render.py:3594-3701
- test_ajae.py:414-545

判断:
证据直接覆盖该要求，或排除性全仓库搜索已证明冲突路径不在当前正式路线；可判 PASS。该结论只覆盖本条，不外推到整个模块或科学主张。

需要修改:
无需修改；保留当前实现和直接证据，并继续确保该局部结论不被外推为模块或科学主张已经成立。

## A02：五帧共享参数并接受同等监督

### [A02-01] 状态: PARTIAL

要求:
五个时间位置是否使用同一 AJAE 参数。

发现:
同一 `AJAEPointModel` 处理所有 q，模型可为五帧输出点级 logit，代码中未发现中心帧专用 head、中心加权或 center/last-only loss；正式 backward 从未执行，五帧全监督只能判为实现层证据。

证据:
- src/train.py:623-721
- src/train.py:908-987
- src/model.py:1045-1275

判断:
只能证明静态实现、局部运行或缺少 生成链路 的产物，尚不能证明可追溯正式路径完整采用；按审计规则不得升级为 PASS。

需要修改:
转为 PASS 所需条件：在上游 Gate1 通过后，以真实训练 trace 证明五帧有效点共同进入同一参数与同一 BCE。

### [A02-02] 状态: PARTIAL

要求:
五帧是否全部产生点级 anomaly logits。

发现:
同一 `AJAEPointModel` 处理所有 q，模型可为五帧输出点级 logit，代码中未发现中心帧专用 head、中心加权或 center/last-only loss；正式 backward 从未执行，五帧全监督只能判为实现层证据。

证据:
- src/train.py:623-721
- src/train.py:908-987
- src/model.py:1045-1275

判断:
只能证明静态实现、局部运行或缺少 生成链路 的产物，尚不能证明可追溯正式路径完整采用；按审计规则不得升级为 PASS。

需要修改:
转为 PASS 所需条件：在上游 Gate1 通过后，以真实训练 trace 证明五帧有效点共同进入同一参数与同一 BCE。

### [A02-03] 状态: BLOCKED

要求:
五帧有效点是否全部参与训练监督。

发现:
同一 `AJAEPointModel` 处理所有 q，模型可为五帧输出点级 logit，代码中未发现中心帧专用 head、中心加权或 center/last-only loss；正式 backward 从未执行，五帧全监督只能判为实现层证据。

证据:
- src/train.py:623-721
- src/train.py:908-987
- src/model.py:1045-1275

判断:
上游 Decision Gate 或冻结条件未满足，本条原则上不得继续；判定 BLOCKED。

需要修改:
转为 PASS 所需条件：在上游 Gate1 通过后，以真实训练 trace 证明五帧有效点共同进入同一参数与同一 BCE。

### [A02-04] 状态: PASS

要求:
检查是否存在 `center_only_loss`。

发现:
同一 `AJAEPointModel` 处理所有 q，模型可为五帧输出点级 logit，代码中未发现中心帧专用 head、中心加权或 center/last-only loss；正式 backward 从未执行，五帧全监督只能判为实现层证据。

证据:
- src/train.py:623-721
- src/train.py:908-987
- src/model.py:1045-1275

判断:
证据直接覆盖该要求，或排除性全仓库搜索已证明冲突路径不在当前正式路线；可判 PASS。该结论只覆盖本条，不外推到整个模块或科学主张。

需要修改:
无需修改；保留当前实现和直接证据，并继续确保该局部结论不被外推为模块或科学主张已经成立。

### [A02-05] 状态: PASS

要求:
检查是否存在 `last_frame_only_loss`。

发现:
同一 `AJAEPointModel` 处理所有 q，模型可为五帧输出点级 logit，代码中未发现中心帧专用 head、中心加权或 center/last-only loss；正式 backward 从未执行，五帧全监督只能判为实现层证据。

证据:
- src/train.py:623-721
- src/train.py:908-987
- src/model.py:1045-1275

判断:
证据直接覆盖该要求，或排除性全仓库搜索已证明冲突路径不在当前正式路线；可判 PASS。该结论只覆盖本条，不外推到整个模块或科学主张。

需要修改:
无需修改；保留当前实现和直接证据，并继续确保该局部结论不被外推为模块或科学主张已经成立。

### [A02-06] 状态: PASS

要求:
检查 center frame 是否被额外加权。

发现:
同一 `AJAEPointModel` 处理所有 q，模型可为五帧输出点级 logit，代码中未发现中心帧专用 head、中心加权或 center/last-only loss；正式 backward 从未执行，五帧全监督只能判为实现层证据。

证据:
- src/train.py:623-721
- src/train.py:908-987
- src/model.py:1045-1275

判断:
证据直接覆盖该要求，或排除性全仓库搜索已证明冲突路径不在当前正式路线；可判 PASS。该结论只覆盖本条，不外推到整个模块或科学主张。

需要修改:
无需修改；保留当前实现和直接证据，并继续确保该局部结论不被外推为模块或科学主张已经成立。

### [A02-07] 状态: PASS

要求:
检查不同 $q$ 是否使用不同 prediction head；若是，需判断是否违反共享参数定义。

发现:
同一 `AJAEPointModel` 处理所有 q，模型可为五帧输出点级 logit，代码中未发现中心帧专用 head、中心加权或 center/last-only loss；正式 backward 从未执行，五帧全监督只能判为实现层证据。

证据:
- src/train.py:623-721
- src/train.py:908-987
- src/model.py:1045-1275

判断:
证据直接覆盖该要求，或排除性全仓库搜索已证明冲突路径不在当前正式路线；可判 PASS。该结论只覆盖本条，不外推到整个模块或科学主张。

需要修改:
无需修改；保留当前实现和直接证据，并继续确保该局部结论不被外推为模块或科学主张已经成立。

### [A02-08] 状态: PASS

要求:
中心帧是否仅承担坐标系规范角色。

发现:
同一 `AJAEPointModel` 处理所有 q，模型可为五帧输出点级 logit，代码中未发现中心帧专用 head、中心加权或 center/last-only loss；正式 backward 从未执行，五帧全监督只能判为实现层证据。

证据:
- src/train.py:623-721
- src/train.py:908-987
- src/model.py:1045-1275

判断:
证据直接覆盖该要求，或排除性全仓库搜索已证明冲突路径不在当前正式路线；可判 PASS。该结论只覆盖本条，不外推到整个模块或科学主张。

需要修改:
无需修改；保留当前实现和直接证据，并继续确保该局部结论不被外推为模块或科学主张已经成立。

## A03：原始回波点级最终目标

### [A03-01] 状态: PARTIAL

要求:
最终 head 是否对原始可见 LiDAR 回波点输出。

发现:
模型在同帧 3NN 解码后对原始点输出 logit，并保存 ray/slot 身份；对象级诊断没有替代点级 head。正式 prediction 文件和官方槽位恢复尚不存在。

证据:
- src/model.py:974-1275
- src/scene.py:375-536
- src/evaluate.py:787-1269

判断:
只能证明静态实现、局部运行或缺少 生成链路 的产物，尚不能证明可追溯正式路径完整采用；按审计规则不得升级为 PASS。

需要修改:
转为 PASS 所需条件：保存真实 prediction，并通过官方 evaluator 的点顺序/槽位读取路径复核。

### [A03-02] 状态: PASS

要求:
是否错误地把 sparse voxel 当作最终预测单位。

发现:
模型在同帧 3NN 解码后对原始点输出 logit，并保存 ray/slot 身份；对象级诊断没有替代点级 head。正式 prediction 文件和官方槽位恢复尚不存在。

证据:
- src/model.py:974-1275
- src/scene.py:375-536
- src/evaluate.py:787-1269

判断:
证据直接覆盖该要求，或排除性全仓库搜索已证明冲突路径不在当前正式路线；可判 PASS。该结论只覆盖本条，不外推到整个模块或科学主张。

需要修改:
无需修改；保留当前实现和直接证据，并继续确保该局部结论不被外推为模块或科学主张已经成立。

### [A03-03] 状态: PARTIAL

要求:
多个原始点映射到同一 STU voxel 时，是否仍保持独立原始点身份。

发现:
模型在同帧 3NN 解码后对原始点输出 logit，并保存 ray/slot 身份；对象级诊断没有替代点级 head。正式 prediction 文件和官方槽位恢复尚不存在。

证据:
- src/model.py:974-1275
- src/scene.py:375-536
- src/evaluate.py:787-1269

判断:
只能证明静态实现、局部运行或缺少 生成链路 的产物，尚不能证明可追溯正式路径完整采用；按审计规则不得升级为 PASS。

需要修改:
转为 PASS 所需条件：保存真实 prediction，并通过官方 evaluator 的点顺序/槽位读取路径复核。

### [A03-04] 状态: BLOCKED

要求:
输出文件是否恢复为官方 evaluator 所要求的原始点顺序/槽位。

发现:
模型在同帧 3NN 解码后对原始点输出 logit，并保存 ray/slot 身份；对象级诊断没有替代点级 head。正式 prediction 文件和官方槽位恢复尚不存在。

证据:
- src/model.py:974-1275
- src/scene.py:375-536
- src/evaluate.py:787-1269

判断:
上游 Decision Gate 或冻结条件未满足，本条原则上不得继续；判定 BLOCKED。

需要修改:
转为 PASS 所需条件：保存真实 prediction，并通过官方 evaluator 的点顺序/槽位读取路径复核。

### [A03-05] 状态: PASS

要求:
是否存在对象级 head 替代点级 anomaly head。

发现:
模型在同帧 3NN 解码后对原始点输出 logit，并保存 ray/slot 身份；对象级诊断没有替代点级 head。正式 prediction 文件和官方槽位恢复尚不存在。

证据:
- src/model.py:974-1275
- src/scene.py:375-536
- src/evaluate.py:787-1269

判断:
证据直接覆盖该要求，或排除性全仓库搜索已证明冲突路径不在当前正式路线；可判 PASS。该结论只覆盖本条，不外推到整个模块或科学主张。

需要修改:
无需修改；保留当前实现和直接证据，并继续确保该局部结论不被外推为模块或科学主张已经成立。

### [A03-06] 状态: PASS

要求:
是否存在对象分数再回投点作为主模型输出；若有，FAIL。

发现:
模型在同帧 3NN 解码后对原始点输出 logit，并保存 ray/slot 身份；对象级诊断没有替代点级 head。正式 prediction 文件和官方槽位恢复尚不存在。

证据:
- src/model.py:974-1275
- src/scene.py:375-536
- src/evaluate.py:787-1269

判断:
证据直接覆盖该要求，或排除性全仓库搜索已证明冲突路径不在当前正式路线；可判 PASS。该结论只覆盖本条，不外推到整个模块或科学主张。

需要修改:
无需修改；保留当前实现和直接证据，并继续确保该局部结论不被外推为模块或科学主张已经成立。

## A04：冻结 STU

### [A04-01] 状态: FAIL

要求:
STU 参数是否全部 `requires_grad=False`。

发现:
静态源码会调用 `requires_grad_(False)`，但当前环境因缺少 Hydra 等官方依赖，无法产生一组真实 STU 参数对象来核验全部标志；“实际调用并冻结”不成立。

证据:
- src/model.py:372-563
- src/train.py:1742-1954
- 本次真实官方 STU 构造：train/206 frame 0，失败于 `ModuleNotFoundError: hydra`

判断:
存在与要求直接冲突的实现、数据或真实运行事实；判定 FAIL。

需要修改:
转为 PASS 所需条件：先让官方 STU 环境可构造，再保存 requires-grad、optimizer 排除、反向梯度和更新前后 state/buffer hash 的一次完整证据。

### [A04-02] 状态: PARTIAL

要求:
optimizer parameter groups 中是否完全不存在 STU 参数。

发现:
代码意图在构造后对 STU 调用 `eval()` 和 `requires_grad_(False)`，优化器也只接收 AJAE 参数；但当前环境无法实例化官方 STU，且没有真实 backward、梯度空值或训练前后状态散列证据。

证据:
- src/model.py:372-563
- src/train.py:1742-1954
- 本次真实官方 STU 构造：train/206 frame 0，失败于 `ModuleNotFoundError: hydra`

判断:
只能证明静态实现、局部运行或缺少 生成链路 的产物，尚不能证明可追溯正式路径完整采用；按审计规则不得升级为 PASS。

需要修改:
转为 PASS 所需条件：先让官方 STU 环境可构造，再保存 requires-grad、optimizer 排除、反向梯度和更新前后 state/buffer hash 的一次完整证据。

### [A04-03] 状态: NOT RUN

要求:
backward 后随机检查 STU 参数 `.grad` 是否为空。

发现:
没有任何真实 backward 后的 STU `.grad is None` 记录。

证据:
- src/model.py:372-563
- src/train.py:1742-1954
- 本次真实官方 STU 构造：train/206 frame 0，失败于 `ModuleNotFoundError: hydra`

判断:
本条要求的运行级审计、实验或正式产物没有发生；代码存在也不能替代执行，判定 NOT RUN。

需要修改:
转为 PASS 所需条件：先让官方 STU 环境可构造，再保存 requires-grad、optimizer 排除、反向梯度和更新前后 state/buffer hash 的一次完整证据。

### [A04-04] 状态: NOT RUN

要求:
训练前后 STU state dict hash 是否一致。

发现:
没有训练前后 STU state dict hash 对照。

证据:
- src/model.py:372-563
- src/train.py:1742-1954
- 本次真实官方 STU 构造：train/206 frame 0，失败于 `ModuleNotFoundError: hydra`

判断:
本条要求的运行级审计、实验或正式产物没有发生；代码存在也不能替代执行，判定 NOT RUN。

需要修改:
转为 PASS 所需条件：先让官方 STU 环境可构造，再保存 requires-grad、optimizer 排除、反向梯度和更新前后 state/buffer hash 的一次完整证据。

### [A04-05] 状态: PARTIAL

要求:
checkpoint 是否错误保存并更新了 STU。

发现:
代码意图在构造后对 STU 调用 `eval()` 和 `requires_grad_(False)`，优化器也只接收 AJAE 参数；但当前环境无法实例化官方 STU，且没有真实 backward、梯度空值或训练前后状态散列证据。

证据:
- src/model.py:372-563
- src/train.py:1742-1954
- 本次真实官方 STU 构造：train/206 frame 0，失败于 `ModuleNotFoundError: hydra`

判断:
只能证明静态实现、局部运行或缺少 生成链路 的产物，尚不能证明可追溯正式路径完整采用；按审计规则不得升级为 PASS。

需要修改:
转为 PASS 所需条件：先让官方 STU 环境可构造，再保存 requires-grad、optimizer 排除、反向梯度和更新前后 state/buffer hash 的一次完整证据。

### [A04-06] 状态: PARTIAL

要求:
BatchNorm / dropout / running statistics 是否可能在训练模式下改变 STU。

发现:
代码意图在构造后对 STU 调用 `eval()` 和 `requires_grad_(False)`，优化器也只接收 AJAE 参数；但当前环境无法实例化官方 STU，且没有真实 backward、梯度空值或训练前后状态散列证据。

证据:
- src/model.py:372-563
- src/train.py:1742-1954
- 本次真实官方 STU 构造：train/206 frame 0，失败于 `ModuleNotFoundError: hydra`

判断:
只能证明静态实现、局部运行或缺少 生成链路 的产物，尚不能证明可追溯正式路径完整采用；按审计规则不得升级为 PASS。

需要修改:
转为 PASS 所需条件：先让官方 STU 环境可构造，再保存 requires-grad、optimizer 排除、反向梯度和更新前后 state/buffer hash 的一次完整证据。

### [A04-07] 状态: FAIL

要求:
必须确认冻结不仅是“不进 optimizer”，而是权重和状态实际不变化。

发现:
没有真实 optimizer step 前后权重、buffer 与模式状态不变的证据；当前官方 STU 甚至不能实例化。

证据:
- src/model.py:372-563
- src/train.py:1742-1954
- 本次真实官方 STU 构造：train/206 frame 0，失败于 `ModuleNotFoundError: hydra`

判断:
存在与要求直接冲突的实现、数据或真实运行事实；判定 FAIL。

需要修改:
转为 PASS 所需条件：先让官方 STU 环境可构造，再保存 requires-grad、optimizer 排除、反向梯度和更新前后 state/buffer hash 的一次完整证据。

### [A04-08] 状态: BLOCKED

要求:
206 只允许更新 AJAE 新增参数。

发现:
代码意图在构造后对 STU 调用 `eval()` 和 `requires_grad_(False)`，优化器也只接收 AJAE 参数；但当前环境无法实例化官方 STU，且没有真实 backward、梯度空值或训练前后状态散列证据。

证据:
- src/model.py:372-563
- src/train.py:1742-1954
- 本次真实官方 STU 构造：train/206 frame 0，失败于 `ModuleNotFoundError: hydra`

判断:
上游 Decision Gate 或冻结条件未满足，本条原则上不得继续；判定 BLOCKED。

需要修改:
转为 PASS 所需条件：先让官方 STU 环境可构造，再保存 requires-grad、optimizer 排除、反向梯度和更新前后 state/buffer hash 的一次完整证据。

## A05：中心对称五帧离线主设置

### [A05-01] 状态: PARTIAL

要求:
主训练窗口是否为 $[t-2,t-1,t,t+1,t+2]$。

发现:
协议和窗口加载器把 B3/B4 定义为 `[t-2,t-1,t,t+1,t+2]`，B5 单独使用因果 `[t-4,t]`，边界只允许完整窗口；B3/B4 尚未真实运行。

证据:
- protocol.json:119-148
- src/scene.py:483-536
- src/evaluate.py:1045-1263

判断:
只能证明静态实现、局部运行或缺少 生成链路 的产物，尚不能证明可追溯正式路径完整采用；按审计规则不得升级为 PASS。

需要修改:
转为 PASS 所需条件：在冻结协议后的 B3/B4 resolved config 与日志中证明实际使用完整中心对称窗口。

### [A05-02] 状态: BLOCKED

要求:
主开发 B3/B4 是否使用中心对称五帧。

发现:
协议和窗口加载器把 B3/B4 定义为 `[t-2,t-1,t,t+1,t+2]`，B5 单独使用因果 `[t-4,t]`，边界只允许完整窗口；B3/B4 尚未真实运行。

证据:
- protocol.json:119-148
- src/scene.py:483-536
- src/evaluate.py:1045-1263

判断:
上游 Decision Gate 或冻结条件未满足，本条原则上不得继续；判定 BLOCKED。

需要修改:
转为 PASS 所需条件：在冻结协议后的 B3/B4 resolved config 与日志中证明实际使用完整中心对称窗口。

### [A05-03] 状态: PASS

要求:
是否错误使用“当前帧 + 四历史帧”作为主模型。

发现:
协议和窗口加载器把 B3/B4 定义为 `[t-2,t-1,t,t+1,t+2]`，B5 单独使用因果 `[t-4,t]`，边界只允许完整窗口；B3/B4 尚未真实运行。

证据:
- protocol.json:119-148
- src/scene.py:483-536
- src/evaluate.py:1045-1263

判断:
证据直接覆盖该要求，或排除性全仓库搜索已证明冲突路径不在当前正式路线；可判 PASS。该结论只覆盖本条，不外推到整个模块或科学主张。

需要修改:
无需修改；保留当前实现和直接证据，并继续确保该局部结论不被外推为模块或科学主张已经成立。

### [A05-04] 状态: PASS

要求:
causal $[t-4,t]$ 是否单独作为 B5，而非混入主模型。

发现:
协议和窗口加载器把 B3/B4 定义为 `[t-2,t-1,t,t+1,t+2]`，B5 单独使用因果 `[t-4,t]`，边界只允许完整窗口；B3/B4 尚未真实运行。

证据:
- protocol.json:119-148
- src/scene.py:483-536
- src/evaluate.py:1045-1263

判断:
证据直接覆盖该要求，或排除性全仓库搜索已证明冲突路径不在当前正式路线；可判 PASS。该结论只覆盖本条，不外推到整个模块或科学主张。

需要修改:
无需修改；保留当前实现和直接证据，并继续确保该局部结论不被外推为模块或科学主张已经成立。

### [A05-05] 状态: PASS

要求:
代码、配置、日志、README 中是否错误将 B3/B4 描述为 online / real-time。

发现:
协议和窗口加载器把 B3/B4 定义为 `[t-2,t-1,t,t+1,t+2]`，B5 单独使用因果 `[t-4,t]`，边界只允许完整窗口；B3/B4 尚未真实运行。

证据:
- protocol.json:119-148
- src/scene.py:483-536
- src/evaluate.py:1045-1263

判断:
证据直接覆盖该要求，或排除性全仓库搜索已证明冲突路径不在当前正式路线；可判 PASS。该结论只覆盖本条，不外推到整个模块或科学主张。

需要修改:
无需修改；保留当前实现和直接证据，并继续确保该局部结论不被外推为模块或科学主张已经成立。

### [A05-06] 状态: PARTIAL

要求:
序列边界是否只使用完整五帧，而不是 padding、重复或镜像。

发现:
协议和窗口加载器把 B3/B4 定义为 `[t-2,t-1,t,t+1,t+2]`，B5 单独使用因果 `[t-4,t]`，边界只允许完整窗口；B3/B4 尚未真实运行。

证据:
- protocol.json:119-148
- src/scene.py:483-536
- src/evaluate.py:1045-1263

判断:
只能证明静态实现、局部运行或缺少 生成链路 的产物，尚不能证明可追溯正式路径完整采用；按审计规则不得升级为 PASS。

需要修改:
转为 PASS 所需条件：在冻结协议后的 B3/B4 resolved config 与日志中证明实际使用完整中心对称窗口。

## D00：数据集角色

### [D00-01] 状态: NOT RUN

要求:
206 是否是 AJAE 新参数唯一梯度训练来源。

发现:
协议定义 206 仅用于训练/校准、201 仅用于开发、19/51 用于冻结后确认。正式评价入口有冻结检查，但通用场景检查入口可以直接读取公开验证标签，且仓库没有全局访问台账。

证据:
- protocol.json:14-42
- src/evaluate.py:1485-1592
- src/scene.py:1010-1046

判断:
本条要求的运行级审计、实验或正式产物没有发生；代码存在也不能替代执行，判定 NOT RUN。

需要修改:
转为合规所需条件：关闭或冻结保护所有可读 public19 标签的旁路，并建立不依赖结果目录推断的访问记录。

### [D00-02] 状态: NOT RUN

要求:
201 是否完全不参与 optimizer update。

发现:
协议定义 206 仅用于训练/校准、201 仅用于开发、19/51 用于冻结后确认。正式评价入口有冻结检查，但通用场景检查入口可以直接读取公开验证标签，且仓库没有全局访问台账。

证据:
- protocol.json:14-42
- src/evaluate.py:1485-1592
- src/scene.py:1010-1046

判断:
本条要求的运行级审计、实验或正式产物没有发生；代码存在也不能替代执行，判定 NOT RUN。

需要修改:
转为合规所需条件：关闭或冻结保护所有可读 public19 标签的旁路，并建立不依赖结果目录推断的访问记录。

### [D00-03] 状态: PARTIAL

要求:
19 条公开真实异常序列是否完全不参与开发阶段梯度。

发现:
协议定义 206 仅用于训练/校准、201 仅用于开发、19/51 用于冻结后确认。正式评价入口有冻结检查，但通用场景检查入口可以直接读取公开验证标签，且仓库没有全局访问台账。

证据:
- protocol.json:14-42
- src/evaluate.py:1485-1592
- src/scene.py:1010-1046

判断:
只能证明静态实现、局部运行或缺少 生成链路 的产物，尚不能证明可追溯正式路径完整采用；按审计规则不得升级为 PASS。

需要修改:
转为合规所需条件：关闭或冻结保护所有可读 public19 标签的旁路，并建立不依赖结果目录推断的访问记录。

### [D00-04] 状态: PARTIAL

要求:
51 条隐藏测试是否未参与开发。

发现:
协议定义 206 仅用于训练/校准、201 仅用于开发、19/51 用于冻结后确认。正式评价入口有冻结检查，但通用场景检查入口可以直接读取公开验证标签，且仓库没有全局访问台账。

证据:
- protocol.json:14-42
- src/evaluate.py:1485-1592
- src/scene.py:1010-1046

判断:
只能证明静态实现、局部运行或缺少 生成链路 的产物，尚不能证明可追溯正式路径完整采用；按审计规则不得升级为 PASS。

需要修改:
转为合规所需条件：关闭或冻结保护所有可读 public19 标签的旁路，并建立不依赖结果目录推断的访问记录。

### [D00-05] 状态: FAIL

要求:
检索全部 dataset path、split name、loader config，检查是否存在意外混用。

发现:
`src/scene.py:1010-1046` 的通用检查入口接受 `--partition val --labels required`，没有方法冻结检查，可绕过正式评价入口读取 public19 标签。

证据:
- protocol.json:14-42
- src/evaluate.py:1485-1592
- src/scene.py:1010-1046

判断:
存在与要求直接冲突的实现、数据或真实运行事实；判定 FAIL。

需要修改:
关闭该旁路，或让所有 public19 标签读取都强制验证方法冻结与访问台账；本次未修改代码。

### [D00-06] 状态: NOT RUN

要求:
检索所有训练日志，确认没有 201/真实异常样本进入 train dataloader。

发现:
协议定义 206 仅用于训练/校准、201 仅用于开发、19/51 用于冻结后确认。正式评价入口有冻结检查，但通用场景检查入口可以直接读取公开验证标签，且仓库没有全局访问台账。

证据:
- protocol.json:14-42
- src/evaluate.py:1485-1592
- src/scene.py:1010-1046

判断:
本条要求的运行级审计、实验或正式产物没有发生；代码存在也不能替代执行，判定 NOT RUN。

需要修改:
转为合规所需条件：关闭或冻结保护所有可读 public19 标签的旁路，并建立不依赖结果目录推断的访问记录。

### [D00-07] 状态: PARTIAL

要求:
如果曾经在当前研究周期利用 19 条真实异常结果修改模型，必须明确报告 confirmation set 已被触碰。

发现:
协议定义 206 仅用于训练/校准、201 仅用于开发、19/51 用于冻结后确认。正式评价入口有冻结检查，但通用场景检查入口可以直接读取公开验证标签，且仓库没有全局访问台账。

证据:
- protocol.json:14-42
- src/evaluate.py:1485-1592
- src/scene.py:1010-1046

判断:
只能证明静态实现、局部运行或缺少 生成链路 的产物，尚不能证明可追溯正式路径完整采用；按审计规则不得升级为 PASS。

需要修改:
转为合规所需条件：关闭或冻结保护所有可读 public19 标签的旁路，并建立不依赖结果目录推断的访问记录。

## D01：206 的合法用途

### [D01-01] 状态: PARTIAL

要求:
206 用于正常背景。

发现:
校准产物来自 train/206 全 449 帧，正常模板也记录 206 来源；正式异常代理训练和 AJAE 参数更新未发生，校准产物缺生成命令与提交绑定。

证据:
- runs/ajae/calibration.pt
- src/train.py:1742-1887
- src/render.py:2420-2646

判断:
只能证明静态实现、局部运行或缺少 生成链路 的产物，尚不能证明可追溯正式路径完整采用；按审计规则不得升级为 PASS。

需要修改:
转为 PASS 所需条件：为校准和模板产物补足可追溯命令/提交；训练项只能在 Gate1 通过后执行。

### [D01-02] 状态: PARTIAL

要求:
206 用于 normal-control 模板提取。

发现:
校准产物来自 train/206 全 449 帧，正常模板也记录 206 来源；正式异常代理训练和 AJAE 参数更新未发生，校准产物缺生成命令与提交绑定。

证据:
- runs/ajae/calibration.pt
- src/train.py:1742-1887
- src/render.py:2420-2646

判断:
只能证明静态实现、局部运行或缺少 生成链路 的产物，尚不能证明可追溯正式路径完整采用；按审计规则不得升级为 PASS。

需要修改:
转为 PASS 所需条件：为校准和模板产物补足可追溯命令/提交；训练项只能在 Gate1 通过后执行。

### [D01-03] 状态: NOT RUN

要求:
206 用于 anomaly-proxy 世界背景。

发现:
校准产物来自 train/206 全 449 帧，正常模板也记录 206 来源；正式异常代理训练和 AJAE 参数更新未发生，校准产物缺生成命令与提交绑定。

证据:
- runs/ajae/calibration.pt
- src/train.py:1742-1887
- src/render.py:2420-2646

判断:
本条要求的运行级审计、实验或正式产物没有发生；代码存在也不能替代执行，判定 NOT RUN。

需要修改:
转为 PASS 所需条件：为校准和模板产物补足可追溯命令/提交；训练项只能在 Gate1 通过后执行。

### [D01-04] 状态: PARTIAL

要求:
206 用于 return model 校准。

发现:
校准产物来自 train/206 全 449 帧，正常模板也记录 206 来源；正式异常代理训练和 AJAE 参数更新未发生，校准产物缺生成命令与提交绑定。

证据:
- runs/ajae/calibration.pt
- src/train.py:1742-1887
- src/render.py:2420-2646

判断:
只能证明静态实现、局部运行或缺少 生成链路 的产物，尚不能证明可追溯正式路径完整采用；按审计规则不得升级为 PASS。

需要修改:
转为 PASS 所需条件：为校准和模板产物补足可追溯命令/提交；训练项只能在 Gate1 通过后执行。

### [D01-05] 状态: PARTIAL

要求:
206 用于 intensity model 校准。

发现:
校准产物来自 train/206 全 449 帧，正常模板也记录 206 来源；正式异常代理训练和 AJAE 参数更新未发生，校准产物缺生成命令与提交绑定。

证据:
- runs/ajae/calibration.pt
- src/train.py:1742-1887
- src/render.py:2420-2646

判断:
只能证明静态实现、局部运行或缺少 生成链路 的产物，尚不能证明可追溯正式路径完整采用；按审计规则不得升级为 PASS。

需要修改:
转为 PASS 所需条件：为校准和模板产物补足可追溯命令/提交；训练项只能在 Gate1 通过后执行。

### [D01-06] 状态: BLOCKED

要求:
206 用于 AJAE 参数更新。

发现:
校准产物来自 train/206 全 449 帧，正常模板也记录 206 来源；正式异常代理训练和 AJAE 参数更新未发生，校准产物缺生成命令与提交绑定。

证据:
- runs/ajae/calibration.pt
- src/train.py:1742-1887
- src/render.py:2420-2646

判断:
上游 Decision Gate 或冻结条件未满足，本条原则上不得继续；判定 BLOCKED。

需要修改:
转为 PASS 所需条件：为校准和模板产物补足可追溯命令/提交；训练项只能在 Gate1 通过后执行。

## D02：201 的固定开发世界

### [D02-01] 状态: PARTIAL

要求:
是否存在固定的 30 条 201 counterfactual worlds。

发现:
`dev.json` 定义 24 个同机制和 6 个留出机制世界，但状态明确为 `definitions_only_unvalidated`，五项验证均为 false，文件还未被 Git 跟踪；固定评价域、选择规则和难度覆盖均未冻结。

证据:
- dev.json:10963-20892
- src/protocol.py:953-1223
- protocol.json:150-173

判断:
只能证明静态实现、局部运行或缺少 生成链路 的产物，尚不能证明可追溯正式路径完整采用；按审计规则不得升级为 PASS。

需要修改:
转为 PASS 所需条件：补齐 V=1..5 覆盖，完成五项验证，并冻结 fixed-201 评价域、难度标准与选择规则。

### [D02-02] 状态: PARTIAL

要求:
是否严格为 24 条 in-generator development worlds。

发现:
`dev.json` 定义 24 个同机制和 6 个留出机制世界，但状态明确为 `definitions_only_unvalidated`，五项验证均为 false，文件还未被 Git 跟踪；固定评价域、选择规则和难度覆盖均未冻结。

证据:
- dev.json:10963-20892
- src/protocol.py:953-1223
- protocol.json:150-173

判断:
只能证明静态实现、局部运行或缺少 生成链路 的产物，尚不能证明可追溯正式路径完整采用；按审计规则不得升级为 PASS。

需要修改:
转为 PASS 所需条件：补齐 V=1..5 覆盖，完成五项验证，并冻结 fixed-201 评价域、难度标准与选择规则。

### [D02-03] 状态: PARTIAL

要求:
是否严格为 6 条 generator-held-out diagnostic worlds。

发现:
`dev.json` 定义 24 个同机制和 6 个留出机制世界，但状态明确为 `definitions_only_unvalidated`，五项验证均为 false，文件还未被 Git 跟踪；固定评价域、选择规则和难度覆盖均未冻结。

证据:
- dev.json:10963-20892
- src/protocol.py:953-1223
- protocol.json:150-173

判断:
只能证明静态实现、局部运行或缺少 生成链路 的产物，尚不能证明可追溯正式路径完整采用；按审计规则不得升级为 PASS。

需要修改:
转为 PASS 所需条件：补齐 V=1..5 覆盖，完成五项验证，并冻结 fixed-201 评价域、难度标准与选择规则。

### [D02-04] 状态: PARTIAL

要求:
24 条开发世界的 world spec 是否已经固定存档。

发现:
`dev.json` 定义 24 个同机制和 6 个留出机制世界，但状态明确为 `definitions_only_unvalidated`，五项验证均为 false，文件还未被 Git 跟踪；固定评价域、选择规则和难度覆盖均未冻结。

证据:
- dev.json:10963-20892
- src/protocol.py:953-1223
- protocol.json:150-173

判断:
只能证明静态实现、局部运行或缺少 生成链路 的产物，尚不能证明可追溯正式路径完整采用；按审计规则不得升级为 PASS。

需要修改:
转为 PASS 所需条件：补齐 V=1..5 覆盖，完成五项验证，并冻结 fixed-201 评价域、难度标准与选择规则。

### [D02-05] 状态: PARTIAL

要求:
6 条 held-out world spec 是否已经固定存档。

发现:
`dev.json` 定义 24 个同机制和 6 个留出机制世界，但状态明确为 `definitions_only_unvalidated`，五项验证均为 false，文件还未被 Git 跟踪；固定评价域、选择规则和难度覆盖均未冻结。

证据:
- dev.json:10963-20892
- src/protocol.py:953-1223
- protocol.json:150-173

判断:
只能证明静态实现、局部运行或缺少 生成链路 的产物，尚不能证明可追溯正式路径完整采用；按审计规则不得升级为 PASS。

需要修改:
转为 PASS 所需条件：补齐 V=1..5 覆盖，完成五项验证，并冻结 fixed-201 评价域、难度标准与选择规则。

### [D02-06] 状态: PARTIAL

要求:
6 条 held-out 是否使用训练阶段完全未使用的程序化几何机制。

发现:
`dev.json` 定义 24 个同机制和 6 个留出机制世界，但状态明确为 `definitions_only_unvalidated`，五项验证均为 false，文件还未被 Git 跟踪；固定评价域、选择规则和难度覆盖均未冻结。

证据:
- dev.json:10963-20892
- src/protocol.py:953-1223
- protocol.json:150-173

判断:
只能证明静态实现、局部运行或缺少 生成链路 的产物，尚不能证明可追溯正式路径完整采用；按审计规则不得升级为 PASS。

需要修改:
转为 PASS 所需条件：补齐 V=1..5 覆盖，完成五项验证，并冻结 fixed-201 评价域、难度标准与选择规则。

### [D02-07] 状态: NOT RUN

要求:
held-out 结果是否没有用于 checkpoint ranking。

发现:
`dev.json` 定义 24 个同机制和 6 个留出机制世界，但状态明确为 `definitions_only_unvalidated`，五项验证均为 false，文件还未被 Git 跟踪；固定评价域、选择规则和难度覆盖均未冻结。

证据:
- dev.json:10963-20892
- src/protocol.py:953-1223
- protocol.json:150-173

判断:
本条要求的运行级审计、实验或正式产物没有发生；代码存在也不能替代执行，判定 NOT RUN。

需要修改:
转为 PASS 所需条件：补齐 V=1..5 覆盖，完成五项验证，并冻结 fixed-201 评价域、难度标准与选择规则。

### [D02-08] 状态: NOT RUN

要求:
held-out 结果是否没有用于超参数选择。

发现:
`dev.json` 定义 24 个同机制和 6 个留出机制世界，但状态明确为 `definitions_only_unvalidated`，五项验证均为 false，文件还未被 Git 跟踪；固定评价域、选择规则和难度覆盖均未冻结。

证据:
- dev.json:10963-20892
- src/protocol.py:953-1223
- protocol.json:150-173

判断:
本条要求的运行级审计、实验或正式产物没有发生；代码存在也不能替代执行，判定 NOT RUN。

需要修改:
转为 PASS 所需条件：补齐 V=1..5 覆盖，完成五项验证，并冻结 fixed-201 评价域、难度标准与选择规则。

### [D02-09] 状态: NOT RUN

要求:
held-out 结果是否没有用于阈值选择。

发现:
`dev.json` 定义 24 个同机制和 6 个留出机制世界，但状态明确为 `definitions_only_unvalidated`，五项验证均为 false，文件还未被 Git 跟踪；固定评价域、选择规则和难度覆盖均未冻结。

证据:
- dev.json:10963-20892
- src/protocol.py:953-1223
- protocol.json:150-173

判断:
本条要求的运行级审计、实验或正式产物没有发生；代码存在也不能替代执行，判定 NOT RUN。

需要修改:
转为 PASS 所需条件：补齐 V=1..5 覆盖，完成五项验证，并冻结 fixed-201 评价域、难度标准与选择规则。

### [D02-10] 状态: NOT RUN

要求:
held-out 结果是否没有用于方法修改。

发现:
`dev.json` 定义 24 个同机制和 6 个留出机制世界，但状态明确为 `definitions_only_unvalidated`，五项验证均为 false，文件还未被 Git 跟踪；固定评价域、选择规则和难度覆盖均未冻结。

证据:
- dev.json:10963-20892
- src/protocol.py:953-1223
- protocol.json:150-173

判断:
本条要求的运行级审计、实验或正式产物没有发生；代码存在也不能替代执行，判定 NOT RUN。

需要修改:
转为 PASS 所需条件：补齐 V=1..5 覆盖，完成五项验证，并冻结 fixed-201 评价域、难度标准与选择规则。

### [D02-11] 状态: PARTIAL

要求:
pure-normal 201 是否独立保存，不包含在这 30 个世界中。

发现:
`dev.json` 定义 24 个同机制和 6 个留出机制世界，但状态明确为 `definitions_only_unvalidated`，五项验证均为 false，文件还未被 Git 跟踪；固定评价域、选择规则和难度覆盖均未冻结。

证据:
- dev.json:10963-20892
- src/protocol.py:953-1223
- protocol.json:150-173

判断:
只能证明静态实现、局部运行或缺少 生成链路 的产物，尚不能证明可追溯正式路径完整采用；按审计规则不得升级为 PASS。

需要修改:
转为 PASS 所需条件：补齐 V=1..5 覆盖，完成五项验证，并冻结 fixed-201 评价域、难度标准与选择规则。

### [D02-12] 状态: PARTIAL

要求:
24 条世界是否同时包含 normal-control 和 anomaly-proxy。

发现:
`dev.json` 定义 24 个同机制和 6 个留出机制世界，但状态明确为 `definitions_only_unvalidated`，五项验证均为 false，文件还未被 Git 跟踪；固定评价域、选择规则和难度覆盖均未冻结。

证据:
- dev.json:10963-20892
- src/protocol.py:953-1223
- protocol.json:150-173

判断:
只能证明静态实现、局部运行或缺少 生成链路 的产物，尚不能证明可追溯正式路径完整采用；按审计规则不得升级为 PASS。

需要修改:
转为 PASS 所需条件：补齐 V=1..5 覆盖，完成五项验证，并冻结 fixed-201 评价域、难度标准与选择规则。

### [D02-13] 状态: PARTIAL

要求:
是否覆盖不同 $N^{vis}$。

发现:
`dev.json` 定义 24 个同机制和 6 个留出机制世界，但状态明确为 `definitions_only_unvalidated`，五项验证均为 false，文件还未被 Git 跟踪；固定评价域、选择规则和难度覆盖均未冻结。

证据:
- dev.json:10963-20892
- src/protocol.py:953-1223
- protocol.json:150-173

判断:
只能证明静态实现、局部运行或缺少 生成链路 的产物，尚不能证明可追溯正式路径完整采用；按审计规则不得升级为 PASS。

需要修改:
转为 PASS 所需条件：补齐 V=1..5 覆盖，完成五项验证，并冻结 fixed-201 评价域、难度标准与选择规则。

### [D02-14] 状态: PARTIAL

要求:
是否覆盖不同遮挡程度 $O$。

发现:
`dev.json` 定义 24 个同机制和 6 个留出机制世界，但状态明确为 `definitions_only_unvalidated`，五项验证均为 false，文件还未被 Git 跟踪；固定评价域、选择规则和难度覆盖均未冻结。

证据:
- dev.json:10963-20892
- src/protocol.py:953-1223
- protocol.json:150-173

判断:
只能证明静态实现、局部运行或缺少 生成链路 的产物，尚不能证明可追溯正式路径完整采用；按审计规则不得升级为 PASS。

需要修改:
转为 PASS 所需条件：补齐 V=1..5 覆盖，完成五项验证，并冻结 fixed-201 评价域、难度标准与选择规则。

### [D02-15] 状态: PARTIAL

要求:
是否覆盖不同距离 $d$。

发现:
`dev.json` 定义 24 个同机制和 6 个留出机制世界，但状态明确为 `definitions_only_unvalidated`，五项验证均为 false，文件还未被 Git 跟踪；固定评价域、选择规则和难度覆盖均未冻结。

证据:
- dev.json:10963-20892
- src/protocol.py:953-1223
- protocol.json:150-173

判断:
只能证明静态实现、局部运行或缺少 生成链路 的产物，尚不能证明可追溯正式路径完整采用；按审计规则不得升级为 PASS。

需要修改:
转为 PASS 所需条件：补齐 V=1..5 覆盖，完成五项验证，并冻结 fixed-201 评价域、难度标准与选择规则。

### [D02-16] 状态: FAIL

要求:
是否覆盖不同五帧可见性 $V$。

发现:
实际 60 个开发实体为 V=5 的 59 个、V=4 的 1 个，V=1、2、3 均为 0，直接不满足五个可见帧层级的覆盖要求。

证据:
- dev.json:10963-20892
- src/protocol.py:953-1223
- protocol.json:150-173

判断:
存在与要求直接冲突的实现、数据或真实运行事实；判定 FAIL。

需要修改:
重新定义并验证固定开发世界，使 V=1、2、3、4、5 均达到预先冻结的覆盖标准。

## D03：真实异常确认集

### [D03-01] 状态: FAIL

要求:
是否有明确 freeze manifest。

发现:
工作区不存在 `ajae-method-freeze-v1` 实例；方法冻结仅是验证代码，不是已完成事实。

证据:
- src/evaluate.py:1485-1592
- src/evaluate.py:2110-2215
- 全仓库未发现 `ajae-method-freeze-v1` 实例

判断:
存在与要求直接冲突的实现、数据或真实运行事实；判定 FAIL。

需要修改:
转为 PASS 所需条件：在任何 19 运行前生成并验证唯一方法冻结清单，同时封闭所有标签访问旁路。

### [D03-02] 状态: NOT RUN

要求:
freeze manifest 是否包含 generator 版本/hash。

发现:
代码具备方法冻结清单验证器，但没有任何真实 freeze manifest，也没有 19 序列正式结果；未发现当前工作区或 Git 中的公开确认结果，但无访问台账，不能证明全局从未访问。

证据:
- src/evaluate.py:1485-1592
- src/evaluate.py:2110-2215
- 全仓库未发现 `ajae-method-freeze-v1` 实例

判断:
本条要求的运行级审计、实验或正式产物没有发生；代码存在也不能替代执行，判定 NOT RUN。

需要修改:
转为 PASS 所需条件：在任何 19 运行前生成并验证唯一方法冻结清单，同时封闭所有标签访问旁路。

### [D03-03] 状态: NOT RUN

要求:
是否包含 normal-control 版本。

发现:
代码具备方法冻结清单验证器，但没有任何真实 freeze manifest，也没有 19 序列正式结果；未发现当前工作区或 Git 中的公开确认结果，但无访问台账，不能证明全局从未访问。

证据:
- src/evaluate.py:1485-1592
- src/evaluate.py:2110-2215
- 全仓库未发现 `ajae-method-freeze-v1` 实例

判断:
本条要求的运行级审计、实验或正式产物没有发生；代码存在也不能替代执行，判定 NOT RUN。

需要修改:
转为 PASS 所需条件：在任何 19 运行前生成并验证唯一方法冻结清单，同时封闭所有标签访问旁路。

### [D03-04] 状态: NOT RUN

要求:
是否包含 STU interface 版本。

发现:
代码具备方法冻结清单验证器，但没有任何真实 freeze manifest，也没有 19 序列正式结果；未发现当前工作区或 Git 中的公开确认结果，但无访问台账，不能证明全局从未访问。

证据:
- src/evaluate.py:1485-1592
- src/evaluate.py:2110-2215
- 全仓库未发现 `ajae-method-freeze-v1` 实例

判断:
本条要求的运行级审计、实验或正式产物没有发生；代码存在也不能替代执行，判定 NOT RUN。

需要修改:
转为 PASS 所需条件：在任何 19 运行前生成并验证唯一方法冻结清单，同时封闭所有标签访问旁路。

### [D03-05] 状态: NOT RUN

要求:
是否包含 model architecture。

发现:
代码具备方法冻结清单验证器，但没有任何真实 freeze manifest，也没有 19 序列正式结果；未发现当前工作区或 Git 中的公开确认结果，但无访问台账，不能证明全局从未访问。

证据:
- src/evaluate.py:1485-1592
- src/evaluate.py:2110-2215
- 全仓库未发现 `ajae-method-freeze-v1` 实例

判断:
本条要求的运行级审计、实验或正式产物没有发生；代码存在也不能替代执行，判定 NOT RUN。

需要修改:
转为 PASS 所需条件：在任何 19 运行前生成并验证唯一方法冻结清单，同时封闭所有标签访问旁路。

### [D03-06] 状态: NOT RUN

要求:
是否包含 loss。

发现:
代码具备方法冻结清单验证器，但没有任何真实 freeze manifest，也没有 19 序列正式结果；未发现当前工作区或 Git 中的公开确认结果，但无访问台账，不能证明全局从未访问。

证据:
- src/evaluate.py:1485-1592
- src/evaluate.py:2110-2215
- 全仓库未发现 `ajae-method-freeze-v1` 实例

判断:
本条要求的运行级审计、实验或正式产物没有发生；代码存在也不能替代执行，判定 NOT RUN。

需要修改:
转为 PASS 所需条件：在任何 19 运行前生成并验证唯一方法冻结清单，同时封闭所有标签访问旁路。

### [D03-07] 状态: NOT RUN

要求:
是否包含 hyperparameters。

发现:
代码具备方法冻结清单验证器，但没有任何真实 freeze manifest，也没有 19 序列正式结果；未发现当前工作区或 Git 中的公开确认结果，但无访问台账，不能证明全局从未访问。

证据:
- src/evaluate.py:1485-1592
- src/evaluate.py:2110-2215
- 全仓库未发现 `ajae-method-freeze-v1` 实例

判断:
本条要求的运行级审计、实验或正式产物没有发生；代码存在也不能替代执行，判定 NOT RUN。

需要修改:
转为 PASS 所需条件：在任何 19 运行前生成并验证唯一方法冻结清单，同时封闭所有标签访问旁路。

### [D03-08] 状态: NOT RUN

要求:
是否包含 checkpoint selection rule。

发现:
代码具备方法冻结清单验证器，但没有任何真实 freeze manifest，也没有 19 序列正式结果；未发现当前工作区或 Git 中的公开确认结果，但无访问台账，不能证明全局从未访问。

证据:
- src/evaluate.py:1485-1592
- src/evaluate.py:2110-2215
- 全仓库未发现 `ajae-method-freeze-v1` 实例

判断:
本条要求的运行级审计、实验或正式产物没有发生；代码存在也不能替代执行，判定 NOT RUN。

需要修改:
转为 PASS 所需条件：在任何 19 运行前生成并验证唯一方法冻结清单，同时封闭所有标签访问旁路。

### [D03-09] 状态: NOT RUN

要求:
是否包含 overlapping-window fusion rule。

发现:
代码具备方法冻结清单验证器，但没有任何真实 freeze manifest，也没有 19 序列正式结果；未发现当前工作区或 Git 中的公开确认结果，但无访问台账，不能证明全局从未访问。

证据:
- src/evaluate.py:1485-1592
- src/evaluate.py:2110-2215
- 全仓库未发现 `ajae-method-freeze-v1` 实例

判断:
本条要求的运行级审计、实验或正式产物没有发生；代码存在也不能替代执行，判定 NOT RUN。

需要修改:
转为 PASS 所需条件：在任何 19 运行前生成并验证唯一方法冻结清单，同时封闭所有标签访问旁路。

### [D03-10] 状态: NOT RUN

要求:
是否包含 point threshold。

发现:
代码具备方法冻结清单验证器，但没有任何真实 freeze manifest，也没有 19 序列正式结果；未发现当前工作区或 Git 中的公开确认结果，但无访问台账，不能证明全局从未访问。

证据:
- src/evaluate.py:1485-1592
- src/evaluate.py:2110-2215
- 全仓库未发现 `ajae-method-freeze-v1` 实例

判断:
本条要求的运行级审计、实验或正式产物没有发生；代码存在也不能替代执行，判定 NOT RUN。

需要修改:
转为 PASS 所需条件：在任何 19 运行前生成并验证唯一方法冻结清单，同时封闭所有标签访问旁路。

### [D03-11] 状态: NOT RUN

要求:
是否包含 DBSCAN 参数。

发现:
代码具备方法冻结清单验证器，但没有任何真实 freeze manifest，也没有 19 序列正式结果；未发现当前工作区或 Git 中的公开确认结果，但无访问台账，不能证明全局从未访问。

证据:
- src/evaluate.py:1485-1592
- src/evaluate.py:2110-2215
- 全仓库未发现 `ajae-method-freeze-v1` 实例

判断:
本条要求的运行级审计、实验或正式产物没有发生；代码存在也不能替代执行，判定 NOT RUN。

需要修改:
转为 PASS 所需条件：在任何 19 运行前生成并验证唯一方法冻结清单，同时封闭所有标签访问旁路。

### [D03-12] 状态: PARTIAL

要求:
是否只有上述内容全部冻结后才运行 19 条真实异常。

发现:
代码具备方法冻结清单验证器，但没有任何真实 freeze manifest，也没有 19 序列正式结果；未发现当前工作区或 Git 中的公开确认结果，但无访问台账，不能证明全局从未访问。

证据:
- src/evaluate.py:1485-1592
- src/evaluate.py:2110-2215
- 全仓库未发现 `ajae-method-freeze-v1` 实例

判断:
只能证明静态实现、局部运行或缺少 生成链路 的产物，尚不能证明可追溯正式路径完整采用；按审计规则不得升级为 PASS。

需要修改:
转为 PASS 所需条件：在任何 19 运行前生成并验证唯一方法冻结清单，同时封闭所有标签访问旁路。

### [D03-13] 状态: N/A

要求:
如已运行真实异常，记录首次运行日期、commit、checkpoint 和命令。

发现:
代码具备方法冻结清单验证器，但没有任何真实 freeze manifest，也没有 19 序列正式结果；未发现当前工作区或 Git 中的公开确认结果，但无访问台账，不能证明全局从未访问。

证据:
- src/evaluate.py:1485-1592
- src/evaluate.py:2110-2215
- 全仓库未发现 `ajae-method-freeze-v1` 实例

判断:
当前阶段尚未触发该条件，主线明确允许不执行；已说明适用边界，判定 N/A。

需要修改:
当前阶段条件尚未触发，无需修改；若未来触发，必须重新按本条要求审计。

### [D03-14] 状态: N/A

要求:
检查首次运行后是否又产生了方法修改。

发现:
代码具备方法冻结清单验证器，但没有任何真实 freeze manifest，也没有 19 序列正式结果；未发现当前工作区或 Git 中的公开确认结果，但无访问台账，不能证明全局从未访问。

证据:
- src/evaluate.py:1485-1592
- src/evaluate.py:2110-2215
- 全仓库未发现 `ajae-method-freeze-v1` 实例

判断:
当前阶段尚未触发该条件，主线明确允许不执行；已说明适用边界，判定 N/A。

需要修改:
当前阶段条件尚未触发，无需修改；若未来触发，必须重新按本条要求审计。

### [D03-15] 状态: N/A

要求:
如果修改过，明确标记“当前 19 条数据已成为开发信息，不可继续称未触碰 confirmation set”。

发现:
代码具备方法冻结清单验证器，但没有任何真实 freeze manifest，也没有 19 序列正式结果；未发现当前工作区或 Git 中的公开确认结果，但无访问台账，不能证明全局从未访问。

证据:
- src/evaluate.py:1485-1592
- src/evaluate.py:2110-2215
- 全仓库未发现 `ajae-method-freeze-v1` 实例

判断:
当前阶段尚未触发该条件，主线明确允许不执行；已说明适用边界，判定 N/A。

需要修改:
当前阶段条件尚未触发，无需修改；若未来触发，必须重新按本条要求审计。

## R00：射线与槽位身份

### [R00-01] 状态: PARTIAL

要求:
是否显式区分 file slot 与 physical ray identity。

发现:
`RayId=(beam,column)` 与 file slot 明确分离，17 帧审计记录未发现正反映射不一致；判定阈值和正式结论字段仍为空，审计范围也未覆盖多序列。

证据:
- src/scene.py:97-168
- src/render.py:1445-2083
- dev.json:6980-8993

判断:
只能证明静态实现、局部运行或缺少 生成链路 的产物，尚不能证明可追溯正式路径完整采用；按审计规则不得升级为 PASS。

需要修改:
转为 PASS 所需条件：冻结审计判据并扩展到足以覆盖 slot/ray 稳定性风险的真实帧/序列，保存独立报告。

### [R00-02] 状态: PARTIAL

要求:
规范射线是否定义为 $r=(b,a)$。

发现:
`RayId=(beam,column)` 与 file slot 明确分离，17 帧审计记录未发现正反映射不一致；判定阈值和正式结论字段仍为空，审计范围也未覆盖多序列。

证据:
- src/scene.py:97-168
- src/render.py:1445-2083
- dev.json:6980-8993

判断:
只能证明静态实现、局部运行或缺少 生成链路 的产物，尚不能证明可追溯正式路径完整采用；按审计规则不得升级为 PASS。

需要修改:
转为 PASS 所需条件：冻结审计判据并扩展到足以覆盖 slot/ray 稳定性风险的真实帧/序列，保存独立报告。

### [R00-03] 状态: PARTIAL

要求:
是否存在 $\rho_f(r)$ 或等价映射。

发现:
`RayId=(beam,column)` 与 file slot 明确分离，17 帧审计记录未发现正反映射不一致；判定阈值和正式结论字段仍为空，审计范围也未覆盖多序列。

证据:
- src/scene.py:97-168
- src/render.py:1445-2083
- dev.json:6980-8993

判断:
只能证明静态实现、局部运行或缺少 生成链路 的产物，尚不能证明可追溯正式路径完整采用；按审计规则不得升级为 PASS。

需要修改:
转为 PASS 所需条件：冻结审计判据并扩展到足以覆盖 slot/ray 稳定性风险的真实帧/序列，保存独立报告。

### [R00-04] 状态: PARTIAL

要求:
slot 数量是否被统计。

发现:
`RayId=(beam,column)` 与 file slot 明确分离，17 帧审计记录未发现正反映射不一致；判定阈值和正式结论字段仍为空，审计范围也未覆盖多序列。

证据:
- src/scene.py:97-168
- src/render.py:1445-2083
- dev.json:6980-8993

判断:
只能证明静态实现、局部运行或缺少 生成链路 的产物，尚不能证明可追溯正式路径完整采用；按审计规则不得升级为 PASS。

需要修改:
转为 PASS 所需条件：冻结审计判据并扩展到足以覆盖 slot/ray 稳定性风险的真实帧/序列，保存独立报告。

### [R00-05] 状态: PARTIAL

要求:
空槽规律是否被统计。

发现:
`RayId=(beam,column)` 与 file slot 明确分离，17 帧审计记录未发现正反映射不一致；判定阈值和正式结论字段仍为空，审计范围也未覆盖多序列。

证据:
- src/scene.py:97-168
- src/render.py:1445-2083
- dev.json:6980-8993

判断:
只能证明静态实现、局部运行或缺少 生成链路 的产物，尚不能证明可追溯正式路径完整采用；按审计规则不得升级为 PASS。

需要修改:
转为 PASS 所需条件：冻结审计判据并扩展到足以覆盖 slot/ray 稳定性风险的真实帧/序列，保存独立报告。

### [R00-06] 状态: PARTIAL

要求:
每个 slot 的 elevation 分布是否审计。

发现:
`RayId=(beam,column)` 与 file slot 明确分离，17 帧审计记录未发现正反映射不一致；判定阈值和正式结论字段仍为空，审计范围也未覆盖多序列。

证据:
- src/scene.py:97-168
- src/render.py:1445-2083
- dev.json:6980-8993

判断:
只能证明静态实现、局部运行或缺少 生成链路 的产物，尚不能证明可追溯正式路径完整采用；按审计规则不得升级为 PASS。

需要修改:
转为 PASS 所需条件：冻结审计判据并扩展到足以覆盖 slot/ray 稳定性风险的真实帧/序列，保存独立报告。

### [R00-07] 状态: PARTIAL

要求:
是否验证 128 beam 排列周期。

发现:
`RayId=(beam,column)` 与 file slot 明确分离，17 帧审计记录未发现正反映射不一致；判定阈值和正式结论字段仍为空，审计范围也未覆盖多序列。

证据:
- src/scene.py:97-168
- src/render.py:1445-2083
- dev.json:6980-8993

判断:
只能证明静态实现、局部运行或缺少 生成链路 的产物，尚不能证明可追溯正式路径完整采用；按审计规则不得升级为 PASS。

需要修改:
转为 PASS 所需条件：冻结审计判据并扩展到足以覆盖 slot/ray 稳定性风险的真实帧/序列，保存独立报告。

### [R00-08] 状态: PARTIAL

要求:
是否验证 azimuth column 连续性。

发现:
`RayId=(beam,column)` 与 file slot 明确分离，17 帧审计记录未发现正反映射不一致；判定阈值和正式结论字段仍为空，审计范围也未覆盖多序列。

证据:
- src/scene.py:97-168
- src/render.py:1445-2083
- dev.json:6980-8993

判断:
只能证明静态实现、局部运行或缺少 生成链路 的产物，尚不能证明可追溯正式路径完整采用；按审计规则不得升级为 PASS。

需要修改:
转为 PASS 所需条件：冻结审计判据并扩展到足以覆盖 slot/ray 稳定性风险的真实帧/序列，保存独立报告。

### [R00-09] 状态: PARTIAL

要求:
是否验证同一 slot 跨帧方向稳定性。

发现:
`RayId=(beam,column)` 与 file slot 明确分离，17 帧审计记录未发现正反映射不一致；判定阈值和正式结论字段仍为空，审计范围也未覆盖多序列。

证据:
- src/scene.py:97-168
- src/render.py:1445-2083
- dev.json:6980-8993

判断:
只能证明静态实现、局部运行或缺少 生成链路 的产物，尚不能证明可追溯正式路径完整采用；按审计规则不得升级为 PASS。

需要修改:
转为 PASS 所需条件：冻结审计判据并扩展到足以覆盖 slot/ray 稳定性风险的真实帧/序列，保存独立报告。

### [R00-10] 状态: PARTIAL

要求:
是否检查多回波导致的 slot reorder。

发现:
`RayId=(beam,column)` 与 file slot 明确分离，17 帧审计记录未发现正反映射不一致；判定阈值和正式结论字段仍为空，审计范围也未覆盖多序列。

证据:
- src/scene.py:97-168
- src/render.py:1445-2083
- dev.json:6980-8993

判断:
只能证明静态实现、局部运行或缺少 生成链路 的产物，尚不能证明可追溯正式路径完整采用；按审计规则不得升级为 PASS。

需要修改:
转为 PASS 所需条件：冻结审计判据并扩展到足以覆盖 slot/ray 稳定性风险的真实帧/序列，保存独立报告。

### [R00-11] 状态: PARTIAL

要求:
审计结果是否保存为正式报告/日志，而不是仅代码存在。

发现:
`RayId=(beam,column)` 与 file slot 明确分离，17 帧审计记录未发现正反映射不一致；判定阈值和正式结论字段仍为空，审计范围也未覆盖多序列。

证据:
- src/scene.py:97-168
- src/render.py:1445-2083
- dev.json:6980-8993

判断:
只能证明静态实现、局部运行或缺少 生成链路 的产物，尚不能证明可追溯正式路径完整采用；按审计规则不得升级为 PASS。

需要修改:
转为 PASS 所需条件：冻结审计判据并扩展到足以覆盖 slot/ray 稳定性风险的真实帧/序列，保存独立报告。

### [R00-12] 状态: N/A

要求:
如果原 slot 不稳定，是否重建 beam/azimuth 索引。

发现:
`RayId=(beam,column)` 与 file slot 明确分离，17 帧审计记录未发现正反映射不一致；判定阈值和正式结论字段仍为空，审计范围也未覆盖多序列。

证据:
- src/scene.py:97-168
- src/render.py:1445-2083
- dev.json:6980-8993

判断:
当前阶段尚未触发该条件，主线明确允许不执行；已说明适用边界，判定 N/A。

需要修改:
当前阶段条件尚未触发，无需修改；若未来触发，必须重新按本条要求审计。

### [R00-13] 状态: PASS

要求:
若该审计尚未通过，所有正式异常世界生成工作必须标记 BLOCKED。

发现:
`RayId=(beam,column)` 与 file slot 明确分离，17 帧审计记录未发现正反映射不一致；判定阈值和正式结论字段仍为空，审计范围也未覆盖多序列。

证据:
- src/scene.py:97-168
- src/render.py:1445-2083
- dev.json:6980-8993

判断:
证据直接覆盖该要求，或排除性全仓库搜索已证明冲突路径不在当前正式路线；可判 PASS。该结论只覆盖本条，不外推到整个模块或科学主张。

需要修改:
无需修改；保留当前实现和直接证据，并继续确保该局部结论不被外推为模块或科学主张已经成立。

## R01：距离图往返

### [R01-01] 状态: PARTIAL

要求:
是否实现 raw point → canonical ray grid。

发现:
train/206 全 449 帧、56,196,767 个回波完成数值往返，计数不一致为 0，最大点误差 `4.49388e-14 m`；证据只来自一个序列且缺独立生成命令与提交，因此不能整体 PASS。

证据:
- src/scene.py:539-660
- dev.json:8363-8368
- artifact: train/206，449 帧，56,196,767 回波

判断:
只能证明静态实现、局部运行或缺少 生成链路 的产物，尚不能证明可追溯正式路径完整采用；按审计规则不得升级为 PASS。

需要修改:
转为 PASS 所需条件：补充多序列覆盖和完整 生成链路；机械误差结果本身无需重做。

### [R01-02] 状态: PARTIAL

要求:
是否实现 canonical ray grid → original point/output。

发现:
train/206 全 449 帧、56,196,767 个回波完成数值往返，计数不一致为 0，最大点误差 `4.49388e-14 m`；证据只来自一个序列且缺独立生成命令与提交，因此不能整体 PASS。

证据:
- src/scene.py:539-660
- dev.json:8363-8368
- artifact: train/206，449 帧，56,196,767 回波

判断:
只能证明静态实现、局部运行或缺少 生成链路 的产物，尚不能证明可追溯正式路径完整采用；按审计规则不得升级为 PASS。

需要修改:
转为 PASS 所需条件：补充多序列覆盖和完整 生成链路；机械误差结果本身无需重做。

### [R01-03] 状态: PARTIAL

要求:
round-trip 是否恢复有效点数量。

发现:
train/206 全 449 帧、56,196,767 个回波完成数值往返，计数不一致为 0，最大点误差 `4.49388e-14 m`；证据只来自一个序列且缺独立生成命令与提交，因此不能整体 PASS。

证据:
- src/scene.py:539-660
- dev.json:8363-8368
- artifact: train/206，449 帧，56,196,767 回波

判断:
只能证明静态实现、局部运行或缺少 生成链路 的产物，尚不能证明可追溯正式路径完整采用；按审计规则不得升级为 PASS。

需要修改:
转为 PASS 所需条件：补充多序列覆盖和完整 生成链路；机械误差结果本身无需重做。

### [R01-04] 状态: PARTIAL

要求:
是否恢复 ray direction。

发现:
train/206 全 449 帧、56,196,767 个回波完成数值往返，计数不一致为 0，最大点误差 `4.49388e-14 m`；证据只来自一个序列且缺独立生成命令与提交，因此不能整体 PASS。

证据:
- src/scene.py:539-660
- dev.json:8363-8368
- artifact: train/206，449 帧，56,196,767 回波

判断:
只能证明静态实现、局部运行或缺少 生成链路 的产物，尚不能证明可追溯正式路径完整采用；按审计规则不得升级为 PASS。

需要修改:
转为 PASS 所需条件：补充多序列覆盖和完整 生成链路；机械误差结果本身无需重做。

### [R01-05] 状态: PARTIAL

要求:
是否恢复 range。

发现:
train/206 全 449 帧、56,196,767 个回波完成数值往返，计数不一致为 0，最大点误差 `4.49388e-14 m`；证据只来自一个序列且缺独立生成命令与提交，因此不能整体 PASS。

证据:
- src/scene.py:539-660
- dev.json:8363-8368
- artifact: train/206，449 帧，56,196,767 回波

判断:
只能证明静态实现、局部运行或缺少 生成链路 的产物，尚不能证明可追溯正式路径完整采用；按审计规则不得升级为 PASS。

需要修改:
转为 PASS 所需条件：补充多序列覆盖和完整 生成链路；机械误差结果本身无需重做。

### [R01-06] 状态: PARTIAL

要求:
是否记录数值误差。

发现:
train/206 全 449 帧、56,196,767 个回波完成数值往返，计数不一致为 0，最大点误差 `4.49388e-14 m`；证据只来自一个序列且缺独立生成命令与提交，因此不能整体 PASS。

证据:
- src/scene.py:539-660
- dev.json:8363-8368
- artifact: train/206，449 帧，56,196,767 回波

判断:
只能证明静态实现、局部运行或缺少 生成链路 的产物，尚不能证明可追溯正式路径完整采用；按审计规则不得升级为 PASS。

需要修改:
转为 PASS 所需条件：补充多序列覆盖和完整 生成链路；机械误差结果本身无需重做。

### [R01-07] 状态: PARTIAL

要求:
是否覆盖多帧、多序列，而不是单帧 toy case。

发现:
train/206 全 449 帧、56,196,767 个回波完成数值往返，计数不一致为 0，最大点误差 `4.49388e-14 m`；证据只来自一个序列且缺独立生成命令与提交，因此不能整体 PASS。

证据:
- src/scene.py:539-660
- dev.json:8363-8368
- artifact: train/206，449 帧，56,196,767 回波

判断:
只能证明静态实现、局部运行或缺少 生成链路 的产物，尚不能证明可追溯正式路径完整采用；按审计规则不得升级为 PASS。

需要修改:
转为 PASS 所需条件：补充多序列覆盖和完整 生成链路；机械误差结果本身无需重做。

### [R01-08] 状态: PARTIAL

要求:
是否保存 audit artifact。

发现:
train/206 全 449 帧、56,196,767 个回波完成数值往返，计数不一致为 0，最大点误差 `4.49388e-14 m`；证据只来自一个序列且缺独立生成命令与提交，因此不能整体 PASS。

证据:
- src/scene.py:539-660
- dev.json:8363-8368
- artifact: train/206，449 帧，56,196,767 回波

判断:
只能证明静态实现、局部运行或缺少 生成链路 的产物，尚不能证明可追溯正式路径完整采用；按审计规则不得升级为 PASS。

需要修改:
转为 PASS 所需条件：补充多序列覆盖和完整 生成链路；机械误差结果本身无需重做。

### [R01-09] 状态: PARTIAL

要求:
round-trip 未通过时 Gate 1 必须 FAIL。

发现:
train/206 全 449 帧、56,196,767 个回波完成数值往返，计数不一致为 0，最大点误差 `4.49388e-14 m`；证据只来自一个序列且缺独立生成命令与提交，因此不能整体 PASS。

证据:
- src/scene.py:539-660
- dev.json:8363-8368
- artifact: train/206，449 帧，56,196,767 回波

判断:
只能证明静态实现、局部运行或缺少 生成链路 的产物，尚不能证明可追溯正式路径完整采用；按审计规则不得升级为 PASS。

需要修改:
转为 PASS 所需条件：补充多序列覆盖和完整 生成链路；机械误差结果本身无需重做。

## G00：正常控制模板

### [G00-01] 状态: PARTIAL

要求:
normal-control 是否从 206 正常实例提取。

发现:
模板提取器限制为八类正常实例并支持刚体变换、合理尺度和材质重采样；当前 30 个开发世界实际没有类别 31/32，且模板库只要求整体非空，没有逐类覆盖清单。

证据:
- src/render.py:774-1030
- src/render.py:1146-1195
- dev.json（30 个固定定义中的 normal-control）

判断:
只能证明静态实现、局部运行或缺少 生成链路 的产物，尚不能证明可追溯正式路径完整采用；按审计规则不得升级为 PASS。

需要修改:
转为 PASS 所需条件：保存逐类模板 manifest，证明八类均有真实 206 来源或冻结缺失类处理规则。

### [G00-02] 状态: PARTIAL

要求:
是否只使用方案允许的正常类别或有明确记录的等价集合。

发现:
模板提取器限制为八类正常实例并支持刚体变换、合理尺度和材质重采样；当前 30 个开发世界实际没有类别 31/32，且模板库只要求整体非空，没有逐类覆盖清单。

证据:
- src/render.py:774-1030
- src/render.py:1146-1195
- dev.json（30 个固定定义中的 normal-control）

判断:
只能证明静态实现、局部运行或缺少 生成链路 的产物，尚不能证明可追溯正式路径完整采用；按审计规则不得升级为 PASS。

需要修改:
转为 PASS 所需条件：保存逐类模板 manifest，证明八类均有真实 206 来源或冻结缺失类处理规则。

### [G00-03] 状态: PARTIAL

要求:
car 支持。

发现:
模板提取器限制为八类正常实例并支持刚体变换、合理尺度和材质重采样；当前 30 个开发世界实际没有类别 31/32，且模板库只要求整体非空，没有逐类覆盖清单。

证据:
- src/render.py:774-1030
- src/render.py:1146-1195
- dev.json（30 个固定定义中的 normal-control）

判断:
只能证明静态实现、局部运行或缺少 生成链路 的产物，尚不能证明可追溯正式路径完整采用；按审计规则不得升级为 PASS。

需要修改:
转为 PASS 所需条件：保存逐类模板 manifest，证明八类均有真实 206 来源或冻结缺失类处理规则。

### [G00-04] 状态: PARTIAL

要求:
bicycle 支持。

发现:
模板提取器限制为八类正常实例并支持刚体变换、合理尺度和材质重采样；当前 30 个开发世界实际没有类别 31/32，且模板库只要求整体非空，没有逐类覆盖清单。

证据:
- src/render.py:774-1030
- src/render.py:1146-1195
- dev.json（30 个固定定义中的 normal-control）

判断:
只能证明静态实现、局部运行或缺少 生成链路 的产物，尚不能证明可追溯正式路径完整采用；按审计规则不得升级为 PASS。

需要修改:
转为 PASS 所需条件：保存逐类模板 manifest，证明八类均有真实 206 来源或冻结缺失类处理规则。

### [G00-05] 状态: PARTIAL

要求:
motorcycle 支持。

发现:
模板提取器限制为八类正常实例并支持刚体变换、合理尺度和材质重采样；当前 30 个开发世界实际没有类别 31/32，且模板库只要求整体非空，没有逐类覆盖清单。

证据:
- src/render.py:774-1030
- src/render.py:1146-1195
- dev.json（30 个固定定义中的 normal-control）

判断:
只能证明静态实现、局部运行或缺少 生成链路 的产物，尚不能证明可追溯正式路径完整采用；按审计规则不得升级为 PASS。

需要修改:
转为 PASS 所需条件：保存逐类模板 manifest，证明八类均有真实 206 来源或冻结缺失类处理规则。

### [G00-06] 状态: PARTIAL

要求:
truck 支持。

发现:
模板提取器限制为八类正常实例并支持刚体变换、合理尺度和材质重采样；当前 30 个开发世界实际没有类别 31/32，且模板库只要求整体非空，没有逐类覆盖清单。

证据:
- src/render.py:774-1030
- src/render.py:1146-1195
- dev.json（30 个固定定义中的 normal-control）

判断:
只能证明静态实现、局部运行或缺少 生成链路 的产物，尚不能证明可追溯正式路径完整采用；按审计规则不得升级为 PASS。

需要修改:
转为 PASS 所需条件：保存逐类模板 manifest，证明八类均有真实 206 来源或冻结缺失类处理规则。

### [G00-07] 状态: PARTIAL

要求:
other-vehicle 支持。

发现:
模板提取器限制为八类正常实例并支持刚体变换、合理尺度和材质重采样；当前 30 个开发世界实际没有类别 31/32，且模板库只要求整体非空，没有逐类覆盖清单。

证据:
- src/render.py:774-1030
- src/render.py:1146-1195
- dev.json（30 个固定定义中的 normal-control）

判断:
只能证明静态实现、局部运行或缺少 生成链路 的产物，尚不能证明可追溯正式路径完整采用；按审计规则不得升级为 PASS。

需要修改:
转为 PASS 所需条件：保存逐类模板 manifest，证明八类均有真实 206 来源或冻结缺失类处理规则。

### [G00-08] 状态: PARTIAL

要求:
person 支持。

发现:
模板提取器限制为八类正常实例并支持刚体变换、合理尺度和材质重采样；当前 30 个开发世界实际没有类别 31/32，且模板库只要求整体非空，没有逐类覆盖清单。

证据:
- src/render.py:774-1030
- src/render.py:1146-1195
- dev.json（30 个固定定义中的 normal-control）

判断:
只能证明静态实现、局部运行或缺少 生成链路 的产物，尚不能证明可追溯正式路径完整采用；按审计规则不得升级为 PASS。

需要修改:
转为 PASS 所需条件：保存逐类模板 manifest，证明八类均有真实 206 来源或冻结缺失类处理规则。

### [G00-09] 状态: PARTIAL

要求:
bicyclist 支持。

发现:
模板提取器限制为八类正常实例并支持刚体变换、合理尺度和材质重采样；当前 30 个开发世界实际没有类别 31/32，且模板库只要求整体非空，没有逐类覆盖清单。

证据:
- src/render.py:774-1030
- src/render.py:1146-1195
- dev.json（30 个固定定义中的 normal-control）

判断:
只能证明静态实现、局部运行或缺少 生成链路 的产物，尚不能证明可追溯正式路径完整采用；按审计规则不得升级为 PASS。

需要修改:
转为 PASS 所需条件：保存逐类模板 manifest，证明八类均有真实 206 来源或冻结缺失类处理规则。

### [G00-10] 状态: PARTIAL

要求:
motorcyclist 支持。

发现:
模板提取器限制为八类正常实例并支持刚体变换、合理尺度和材质重采样；当前 30 个开发世界实际没有类别 31/32，且模板库只要求整体非空，没有逐类覆盖清单。

证据:
- src/render.py:774-1030
- src/render.py:1146-1195
- dev.json（30 个固定定义中的 normal-control）

判断:
只能证明静态实现、局部运行或缺少 生成链路 的产物，尚不能证明可追溯正式路径完整采用；按审计规则不得升级为 PASS。

需要修改:
转为 PASS 所需条件：保存逐类模板 manifest，证明八类均有真实 206 来源或冻结缺失类处理规则。

### [G00-11] 状态: PASS

要求:
instance ID 是否仅生成器内部使用。

发现:
模板提取器限制为八类正常实例并支持刚体变换、合理尺度和材质重采样；当前 30 个开发世界实际没有类别 31/32，且模板库只要求整体非空，没有逐类覆盖清单。

证据:
- src/render.py:774-1030
- src/render.py:1146-1195
- dev.json（30 个固定定义中的 normal-control）

判断:
证据直接覆盖该要求，或排除性全仓库搜索已证明冲突路径不在当前正式路线；可判 PASS。该结论只覆盖本条，不外推到整个模块或科学主张。

需要修改:
无需修改；保留当前实现和直接证据，并继续确保该局部结论不被外推为模块或科学主张已经成立。

### [G00-12] 状态: PASS

要求:
AJAE 输入中是否完全没有 normal-control instance ID。

发现:
模板提取器限制为八类正常实例并支持刚体变换、合理尺度和材质重采样；当前 30 个开发世界实际没有类别 31/32，且模板库只要求整体非空，没有逐类覆盖清单。

证据:
- src/render.py:774-1030
- src/render.py:1146-1195
- dev.json（30 个固定定义中的 normal-control）

判断:
证据直接覆盖该要求，或排除性全仓库搜索已证明冲突路径不在当前正式路线；可判 PASS。该结论只覆盖本条，不外推到整个模块或科学主张。

需要修改:
无需修改；保留当前实现和直接证据，并继续确保该局部结论不被外推为模块或科学主张已经成立。

### [G00-13] 状态: PARTIAL

要求:
是否支持合理刚体旋转。

发现:
模板提取器限制为八类正常实例并支持刚体变换、合理尺度和材质重采样；当前 30 个开发世界实际没有类别 31/32，且模板库只要求整体非空，没有逐类覆盖清单。

证据:
- src/render.py:774-1030
- src/render.py:1146-1195
- dev.json（30 个固定定义中的 normal-control）

判断:
只能证明静态实现、局部运行或缺少 生成链路 的产物，尚不能证明可追溯正式路径完整采用；按审计规则不得升级为 PASS。

需要修改:
转为 PASS 所需条件：保存逐类模板 manifest，证明八类均有真实 206 来源或冻结缺失类处理规则。

### [G00-14] 状态: PARTIAL

要求:
是否支持平移。

发现:
模板提取器限制为八类正常实例并支持刚体变换、合理尺度和材质重采样；当前 30 个开发世界实际没有类别 31/32，且模板库只要求整体非空，没有逐类覆盖清单。

证据:
- src/render.py:774-1030
- src/render.py:1146-1195
- dev.json（30 个固定定义中的 normal-control）

判断:
只能证明静态实现、局部运行或缺少 生成链路 的产物，尚不能证明可追溯正式路径完整采用；按审计规则不得升级为 PASS。

需要修改:
转为 PASS 所需条件：保存逐类模板 manifest，证明八类均有真实 206 来源或冻结缺失类处理规则。

### [G00-15] 状态: PARTIAL

要求:
是否只进行类别合理尺度变化。

发现:
模板提取器限制为八类正常实例并支持刚体变换、合理尺度和材质重采样；当前 30 个开发世界实际没有类别 31/32，且模板库只要求整体非空，没有逐类覆盖清单。

证据:
- src/render.py:774-1030
- src/render.py:1146-1195
- dev.json（30 个固定定义中的 normal-control）

判断:
只能证明静态实现、局部运行或缺少 生成链路 的产物，尚不能证明可追溯正式路径完整采用；按审计规则不得升级为 PASS。

需要修改:
转为 PASS 所需条件：保存逐类模板 manifest，证明八类均有真实 206 来源或冻结缺失类处理规则。

### [G00-16] 状态: PARTIAL

要求:
是否支持材质重采样。

发现:
模板提取器限制为八类正常实例并支持刚体变换、合理尺度和材质重采样；当前 30 个开发世界实际没有类别 31/32，且模板库只要求整体非空，没有逐类覆盖清单。

证据:
- src/render.py:774-1030
- src/render.py:1146-1195
- dev.json（30 个固定定义中的 normal-control）

判断:
只能证明静态实现、局部运行或缺少 生成链路 的产物，尚不能证明可追溯正式路径完整采用；按审计规则不得升级为 PASS。

需要修改:
转为 PASS 所需条件：保存逐类模板 manifest，证明八类均有真实 206 来源或冻结缺失类处理规则。

### [G00-17] 状态: PARTIAL

要求:
放置 surface 是否符合基本正常语义。

发现:
模板提取器限制为八类正常实例并支持刚体变换、合理尺度和材质重采样；当前 30 个开发世界实际没有类别 31/32，且模板库只要求整体非空，没有逐类覆盖清单。

证据:
- src/render.py:774-1030
- src/render.py:1146-1195
- dev.json（30 个固定定义中的 normal-control）

判断:
只能证明静态实现、局部运行或缺少 生成链路 的产物，尚不能证明可追溯正式路径完整采用；按审计规则不得升级为 PASS。

需要修改:
转为 PASS 所需条件：保存逐类模板 manifest，证明八类均有真实 206 来源或冻结缺失类处理规则。

## G01：异常代理基本体

### [G01-01] 状态: PARTIAL

要求:
anomaly proxy 是否为类别无关程序化几何。

发现:
程序化 CSG 支持 1–5 个 superquadric、三种布尔运算、弯曲/扭转/渐缩与低频变形，并执行 finite、bounded 和单连通分量检查；独立射线稳定性或严格 watertight 运行审计缺失。

证据:
- src/render.py:174-645
- src/render.py:1032-1134
- test_ajae.py:402-411

判断:
只能证明静态实现、局部运行或缺少 生成链路 的产物，尚不能证明可追溯正式路径完整采用；按审计规则不得升级为 PASS。

需要修改:
转为 PASS 所需条件：保存多视角/多精度的求交稳定性与封闭性运行审计。

### [G01-02] 状态: PASS

要求:
是否没有直接使用 chair/box/trashcan 等现实语义 CAD 作为主生成来源。

发现:
程序化 CSG 支持 1–5 个 superquadric、三种布尔运算、弯曲/扭转/渐缩与低频变形，并执行 finite、bounded 和单连通分量检查；独立射线稳定性或严格 watertight 运行审计缺失。

证据:
- src/render.py:174-645
- src/render.py:1032-1134
- test_ajae.py:402-411

判断:
证据直接覆盖该要求，或排除性全仓库搜索已证明冲突路径不在当前正式路线；可判 PASS。该结论只覆盖本条，不外推到整个模块或科学主张。

需要修改:
无需修改；保留当前实现和直接证据，并继续确保该局部结论不被外推为模块或科学主张已经成立。

### [G01-03] 状态: PARTIAL

要求:
是否实现 superquadric 或方案等价连续 primitive。

发现:
程序化 CSG 支持 1–5 个 superquadric、三种布尔运算、弯曲/扭转/渐缩与低频变形，并执行 finite、bounded 和单连通分量检查；独立射线稳定性或严格 watertight 运行审计缺失。

证据:
- src/render.py:174-645
- src/render.py:1032-1134
- test_ajae.py:402-411

判断:
只能证明静态实现、局部运行或缺少 生成链路 的产物，尚不能证明可追溯正式路径完整采用；按审计规则不得升级为 PASS。

需要修改:
转为 PASS 所需条件：保存多视角/多精度的求交稳定性与封闭性运行审计。

### [G01-04] 状态: PARTIAL

要求:
primitive 是否包含独立三轴尺度参数。

发现:
程序化 CSG 支持 1–5 个 superquadric、三种布尔运算、弯曲/扭转/渐缩与低频变形，并执行 finite、bounded 和单连通分量检查；独立射线稳定性或严格 watertight 运行审计缺失。

证据:
- src/render.py:174-645
- src/render.py:1032-1134
- test_ajae.py:402-411

判断:
只能证明静态实现、局部运行或缺少 生成链路 的产物，尚不能证明可追溯正式路径完整采用；按审计规则不得升级为 PASS。

需要修改:
转为 PASS 所需条件：保存多视角/多精度的求交稳定性与封闭性运行审计。

### [G01-05] 状态: PARTIAL

要求:
是否包含形状指数/圆润度等参数。

发现:
程序化 CSG 支持 1–5 个 superquadric、三种布尔运算、弯曲/扭转/渐缩与低频变形，并执行 finite、bounded 和单连通分量检查；独立射线稳定性或严格 watertight 运行审计缺失。

证据:
- src/render.py:174-645
- src/render.py:1032-1134
- test_ajae.py:402-411

判断:
只能证明静态实现、局部运行或缺少 生成链路 的产物，尚不能证明可追溯正式路径完整采用；按审计规则不得升级为 PASS。

需要修改:
转为 PASS 所需条件：保存多视角/多精度的求交稳定性与封闭性运行审计。

### [G01-06] 状态: PARTIAL

要求:
单实体是否允许 1–5 个 primitive。

发现:
程序化 CSG 支持 1–5 个 superquadric、三种布尔运算、弯曲/扭转/渐缩与低频变形，并执行 finite、bounded 和单连通分量检查；独立射线稳定性或严格 watertight 运行审计缺失。

证据:
- src/render.py:174-645
- src/render.py:1032-1134
- test_ajae.py:402-411

判断:
只能证明静态实现、局部运行或缺少 生成链路 的产物，尚不能证明可追溯正式路径完整采用；按审计规则不得升级为 PASS。

需要修改:
转为 PASS 所需条件：保存多视角/多精度的求交稳定性与封闭性运行审计。

### [G01-07] 状态: PARTIAL

要求:
是否支持 union。

发现:
程序化 CSG 支持 1–5 个 superquadric、三种布尔运算、弯曲/扭转/渐缩与低频变形，并执行 finite、bounded 和单连通分量检查；独立射线稳定性或严格 watertight 运行审计缺失。

证据:
- src/render.py:174-645
- src/render.py:1032-1134
- test_ajae.py:402-411

判断:
只能证明静态实现、局部运行或缺少 生成链路 的产物，尚不能证明可追溯正式路径完整采用；按审计规则不得升级为 PASS。

需要修改:
转为 PASS 所需条件：保存多视角/多精度的求交稳定性与封闭性运行审计。

### [G01-08] 状态: PARTIAL

要求:
是否支持 difference。

发现:
程序化 CSG 支持 1–5 个 superquadric、三种布尔运算、弯曲/扭转/渐缩与低频变形，并执行 finite、bounded 和单连通分量检查；独立射线稳定性或严格 watertight 运行审计缺失。

证据:
- src/render.py:174-645
- src/render.py:1032-1134
- test_ajae.py:402-411

判断:
只能证明静态实现、局部运行或缺少 生成链路 的产物，尚不能证明可追溯正式路径完整采用；按审计规则不得升级为 PASS。

需要修改:
转为 PASS 所需条件：保存多视角/多精度的求交稳定性与封闭性运行审计。

### [G01-09] 状态: PARTIAL

要求:
是否支持 intersection。

发现:
程序化 CSG 支持 1–5 个 superquadric、三种布尔运算、弯曲/扭转/渐缩与低频变形，并执行 finite、bounded 和单连通分量检查；独立射线稳定性或严格 watertight 运行审计缺失。

证据:
- src/render.py:174-645
- src/render.py:1032-1134
- test_ajae.py:402-411

判断:
只能证明静态实现、局部运行或缺少 生成链路 的产物，尚不能证明可追溯正式路径完整采用；按审计规则不得升级为 PASS。

需要修改:
转为 PASS 所需条件：保存多视角/多精度的求交稳定性与封闭性运行审计。

### [G01-10] 状态: PARTIAL

要求:
是否支持 bending。

发现:
程序化 CSG 支持 1–5 个 superquadric、三种布尔运算、弯曲/扭转/渐缩与低频变形，并执行 finite、bounded 和单连通分量检查；独立射线稳定性或严格 watertight 运行审计缺失。

证据:
- src/render.py:174-645
- src/render.py:1032-1134
- test_ajae.py:402-411

判断:
只能证明静态实现、局部运行或缺少 生成链路 的产物，尚不能证明可追溯正式路径完整采用；按审计规则不得升级为 PASS。

需要修改:
转为 PASS 所需条件：保存多视角/多精度的求交稳定性与封闭性运行审计。

### [G01-11] 状态: PARTIAL

要求:
是否支持 twisting。

发现:
程序化 CSG 支持 1–5 个 superquadric、三种布尔运算、弯曲/扭转/渐缩与低频变形，并执行 finite、bounded 和单连通分量检查；独立射线稳定性或严格 watertight 运行审计缺失。

证据:
- src/render.py:174-645
- src/render.py:1032-1134
- test_ajae.py:402-411

判断:
只能证明静态实现、局部运行或缺少 生成链路 的产物，尚不能证明可追溯正式路径完整采用；按审计规则不得升级为 PASS。

需要修改:
转为 PASS 所需条件：保存多视角/多精度的求交稳定性与封闭性运行审计。

### [G01-12] 状态: PARTIAL

要求:
是否支持 tapering。

发现:
程序化 CSG 支持 1–5 个 superquadric、三种布尔运算、弯曲/扭转/渐缩与低频变形，并执行 finite、bounded 和单连通分量检查；独立射线稳定性或严格 watertight 运行审计缺失。

证据:
- src/render.py:174-645
- src/render.py:1032-1134
- test_ajae.py:402-411

判断:
只能证明静态实现、局部运行或缺少 生成链路 的产物，尚不能证明可追溯正式路径完整采用；按审计规则不得升级为 PASS。

需要修改:
转为 PASS 所需条件：保存多视角/多精度的求交稳定性与封闭性运行审计。

### [G01-13] 状态: PARTIAL

要求:
是否支持 smooth low-frequency deformation。

发现:
程序化 CSG 支持 1–5 个 superquadric、三种布尔运算、弯曲/扭转/渐缩与低频变形，并执行 finite、bounded 和单连通分量检查；独立射线稳定性或严格 watertight 运行审计缺失。

证据:
- src/render.py:174-645
- src/render.py:1032-1134
- test_ajae.py:402-411

判断:
只能证明静态实现、局部运行或缺少 生成链路 的产物，尚不能证明可追溯正式路径完整采用；按审计规则不得升级为 PASS。

需要修改:
转为 PASS 所需条件：保存多视角/多精度的求交稳定性与封闭性运行审计。

### [G01-14] 状态: PARTIAL

要求:
最终几何是否做 bounded 检查。

发现:
程序化 CSG 支持 1–5 个 superquadric、三种布尔运算、弯曲/扭转/渐缩与低频变形，并执行 finite、bounded 和单连通分量检查；独立射线稳定性或严格 watertight 运行审计缺失。

证据:
- src/render.py:174-645
- src/render.py:1032-1134
- test_ajae.py:402-411

判断:
只能证明静态实现、局部运行或缺少 生成链路 的产物，尚不能证明可追溯正式路径完整采用；按审计规则不得升级为 PASS。

需要修改:
转为 PASS 所需条件：保存多视角/多精度的求交稳定性与封闭性运行审计。

### [G01-15] 状态: PARTIAL

要求:
是否做 closed/watertight 或等价射线稳定性检查。

发现:
程序化 CSG 支持 1–5 个 superquadric、三种布尔运算、弯曲/扭转/渐缩与低频变形，并执行 finite、bounded 和单连通分量检查；独立射线稳定性或严格 watertight 运行审计缺失。

证据:
- src/render.py:174-645
- src/render.py:1032-1134
- test_ajae.py:402-411

判断:
只能证明静态实现、局部运行或缺少 生成链路 的产物，尚不能证明可追溯正式路径完整采用；按审计规则不得升级为 PASS。

需要修改:
转为 PASS 所需条件：保存多视角/多精度的求交稳定性与封闭性运行审计。

### [G01-16] 状态: PARTIAL

要求:
是否检查 finite numerical values。

发现:
程序化 CSG 支持 1–5 个 superquadric、三种布尔运算、弯曲/扭转/渐缩与低频变形，并执行 finite、bounded 和单连通分量检查；独立射线稳定性或严格 watertight 运行审计缺失。

证据:
- src/render.py:174-645
- src/render.py:1032-1134
- test_ajae.py:402-411

判断:
只能证明静态实现、局部运行或缺少 生成链路 的产物，尚不能证明可追溯正式路径完整采用；按审计规则不得升级为 PASS。

需要修改:
转为 PASS 所需条件：保存多视角/多精度的求交稳定性与封闭性运行审计。

### [G01-17] 状态: PARTIAL

要求:
是否检查至少一个 connected component。

发现:
程序化 CSG 支持 1–5 个 superquadric、三种布尔运算、弯曲/扭转/渐缩与低频变形，并执行 finite、bounded 和单连通分量检查；独立射线稳定性或严格 watertight 运行审计缺失。

证据:
- src/render.py:174-645
- src/render.py:1032-1134
- test_ajae.py:402-411

判断:
只能证明静态实现、局部运行或缺少 生成链路 的产物，尚不能证明可追溯正式路径完整采用；按审计规则不得升级为 PASS。

需要修改:
转为 PASS 所需条件：保存多视角/多精度的求交稳定性与封闭性运行审计。

### [G01-18] 状态: PARTIAL

要求:
多 component 时是否 reject，而非视作一个 entity。

发现:
程序化 CSG 支持 1–5 个 superquadric、三种布尔运算、弯曲/扭转/渐缩与低频变形，并执行 finite、bounded 和单连通分量检查；独立射线稳定性或严格 watertight 运行审计缺失。

证据:
- src/render.py:174-645
- src/render.py:1032-1134
- test_ajae.py:402-411

判断:
只能证明静态实现、局部运行或缺少 生成链路 的产物，尚不能证明可追溯正式路径完整采用；按审计规则不得升级为 PASS。

需要修改:
转为 PASS 所需条件：保存多视角/多精度的求交稳定性与封闭性运行审计。

### [G01-19] 状态: NOT RUN

要求:
是否验证 ray intersection 稳定性。

发现:
程序化 CSG 支持 1–5 个 superquadric、三种布尔运算、弯曲/扭转/渐缩与低频变形，并执行 finite、bounded 和单连通分量检查；独立射线稳定性或严格 watertight 运行审计缺失。

证据:
- src/render.py:174-645
- src/render.py:1032-1134
- test_ajae.py:402-411

判断:
本条要求的运行级审计、实验或正式产物没有发生；代码存在也不能替代执行，判定 NOT RUN。

需要修改:
转为 PASS 所需条件：保存多视角/多精度的求交稳定性与封闭性运行审计。

## G02：形状、尺度与材质解耦

### [G02-01] 状态: PARTIAL

要求:
overall scale 是否独立采样。

发现:
采样字段把总体尺度、轴比、形状复杂度、位置和材质分开；开发定义显示一定 Nvis/O/d 变化，但没有预注册覆盖分箱，且五帧可见性 V 缺少 1–3。

证据:
- src/render.py:1217-1352
- src/render.py:3465-3842
- dev.json:20883-20892

判断:
只能证明静态实现、局部运行或缺少 生成链路 的产物，尚不能证明可追溯正式路径完整采用；按审计规则不得升级为 PASS。

需要修改:
转为 PASS 所需条件：冻结并验证 scale/axis/complexity/material/Nvis/O/d/V 的覆盖标准。

### [G02-02] 状态: PARTIAL

要求:
axis ratio 是否独立采样。

发现:
采样字段把总体尺度、轴比、形状复杂度、位置和材质分开；开发定义显示一定 Nvis/O/d 变化，但没有预注册覆盖分箱，且五帧可见性 V 缺少 1–3。

证据:
- src/render.py:1217-1352
- src/render.py:3465-3842
- dev.json:20883-20892

判断:
只能证明静态实现、局部运行或缺少 生成链路 的产物，尚不能证明可追溯正式路径完整采用；按审计规则不得升级为 PASS。

需要修改:
转为 PASS 所需条件：冻结并验证 scale/axis/complexity/material/Nvis/O/d/V 的覆盖标准。

### [G02-03] 状态: PARTIAL

要求:
是否覆盖 sparse small targets。

发现:
采样字段把总体尺度、轴比、形状复杂度、位置和材质分开；开发定义显示一定 Nvis/O/d 变化，但没有预注册覆盖分箱，且五帧可见性 V 缺少 1–3。

证据:
- src/render.py:1217-1352
- src/render.py:3465-3842
- dev.json:20883-20892

判断:
只能证明静态实现、局部运行或缺少 生成链路 的产物，尚不能证明可追溯正式路径完整采用；按审计规则不得升级为 PASS。

需要修改:
转为 PASS 所需条件：冻结并验证 scale/axis/complexity/material/Nvis/O/d/V 的覆盖标准。

### [G02-04] 状态: PARTIAL

要求:
是否覆盖 medium-visible targets。

发现:
采样字段把总体尺度、轴比、形状复杂度、位置和材质分开；开发定义显示一定 Nvis/O/d 变化，但没有预注册覆盖分箱，且五帧可见性 V 缺少 1–3。

证据:
- src/render.py:1217-1352
- src/render.py:3465-3842
- dev.json:20883-20892

判断:
只能证明静态实现、局部运行或缺少 生成链路 的产物，尚不能证明可追溯正式路径完整采用；按审计规则不得升级为 PASS。

需要修改:
转为 PASS 所需条件：冻结并验证 scale/axis/complexity/material/Nvis/O/d/V 的覆盖标准。

### [G02-05] 状态: PARTIAL

要求:
是否覆盖 dense/clear targets。

发现:
采样字段把总体尺度、轴比、形状复杂度、位置和材质分开；开发定义显示一定 Nvis/O/d 变化，但没有预注册覆盖分箱，且五帧可见性 V 缺少 1–3。

证据:
- src/render.py:1217-1352
- src/render.py:3465-3842
- dev.json:20883-20892

判断:
只能证明静态实现、局部运行或缺少 生成链路 的产物，尚不能证明可追溯正式路径完整采用；按审计规则不得升级为 PASS。

需要修改:
转为 PASS 所需条件：冻结并验证 scale/axis/complexity/material/Nvis/O/d/V 的覆盖标准。

### [G02-06] 状态: PARTIAL

要求:
是否覆盖 block-like。

发现:
采样字段把总体尺度、轴比、形状复杂度、位置和材质分开；开发定义显示一定 Nvis/O/d 变化，但没有预注册覆盖分箱，且五帧可见性 V 缺少 1–3。

证据:
- src/render.py:1217-1352
- src/render.py:3465-3842
- dev.json:20883-20892

判断:
只能证明静态实现、局部运行或缺少 生成链路 的产物，尚不能证明可追溯正式路径完整采用；按审计规则不得升级为 PASS。

需要修改:
转为 PASS 所需条件：冻结并验证 scale/axis/complexity/material/Nvis/O/d/V 的覆盖标准。

### [G02-07] 状态: PARTIAL

要求:
是否覆盖 flat。

发现:
采样字段把总体尺度、轴比、形状复杂度、位置和材质分开；开发定义显示一定 Nvis/O/d 变化，但没有预注册覆盖分箱，且五帧可见性 V 缺少 1–3。

证据:
- src/render.py:1217-1352
- src/render.py:3465-3842
- dev.json:20883-20892

判断:
只能证明静态实现、局部运行或缺少 生成链路 的产物，尚不能证明可追溯正式路径完整采用；按审计规则不得升级为 PASS。

需要修改:
转为 PASS 所需条件：冻结并验证 scale/axis/complexity/material/Nvis/O/d/V 的覆盖标准。

### [G02-08] 状态: PARTIAL

要求:
是否覆盖 slender。

发现:
采样字段把总体尺度、轴比、形状复杂度、位置和材质分开；开发定义显示一定 Nvis/O/d 变化，但没有预注册覆盖分箱，且五帧可见性 V 缺少 1–3。

证据:
- src/render.py:1217-1352
- src/render.py:3465-3842
- dev.json:20883-20892

判断:
只能证明静态实现、局部运行或缺少 生成链路 的产物，尚不能证明可追溯正式路径完整采用；按审计规则不得升级为 PASS。

需要修改:
转为 PASS 所需条件：冻结并验证 scale/axis/complexity/material/Nvis/O/d/V 的覆盖标准。

### [G02-09] 状态: PARTIAL

要求:
是否覆盖 asymmetric。

发现:
采样字段把总体尺度、轴比、形状复杂度、位置和材质分开；开发定义显示一定 Nvis/O/d 变化，但没有预注册覆盖分箱，且五帧可见性 V 缺少 1–3。

证据:
- src/render.py:1217-1352
- src/render.py:3465-3842
- dev.json:20883-20892

判断:
只能证明静态实现、局部运行或缺少 生成链路 的产物，尚不能证明可追溯正式路径完整采用；按审计规则不得升级为 PASS。

需要修改:
转为 PASS 所需条件：冻结并验证 scale/axis/complexity/material/Nvis/O/d/V 的覆盖标准。

### [G02-10] 状态: PARTIAL

要求:
geometry complexity 与 scale 是否非固定绑定。

发现:
采样字段把总体尺度、轴比、形状复杂度、位置和材质分开；开发定义显示一定 Nvis/O/d 变化，但没有预注册覆盖分箱，且五帧可见性 V 缺少 1–3。

证据:
- src/render.py:1217-1352
- src/render.py:3465-3842
- dev.json:20883-20892

判断:
只能证明静态实现、局部运行或缺少 生成链路 的产物，尚不能证明可追溯正式路径完整采用；按审计规则不得升级为 PASS。

需要修改:
转为 PASS 所需条件：冻结并验证 scale/axis/complexity/material/Nvis/O/d/V 的覆盖标准。

### [G02-11] 状态: PARTIAL

要求:
geometry 与 placement 是否非固定绑定。

发现:
采样字段把总体尺度、轴比、形状复杂度、位置和材质分开；开发定义显示一定 Nvis/O/d 变化，但没有预注册覆盖分箱，且五帧可见性 V 缺少 1–3。

证据:
- src/render.py:1217-1352
- src/render.py:3465-3842
- dev.json:20883-20892

判断:
只能证明静态实现、局部运行或缺少 生成链路 的产物，尚不能证明可追溯正式路径完整采用；按审计规则不得升级为 PASS。

需要修改:
转为 PASS 所需条件：冻结并验证 scale/axis/complexity/material/Nvis/O/d/V 的覆盖标准。

### [G02-12] 状态: PARTIAL

要求:
geometry 与 material 是否非固定绑定。

发现:
采样字段把总体尺度、轴比、形状复杂度、位置和材质分开；开发定义显示一定 Nvis/O/d 变化，但没有预注册覆盖分箱，且五帧可见性 V 缺少 1–3。

证据:
- src/render.py:1217-1352
- src/render.py:3465-3842
- dev.json:20883-20892

判断:
只能证明静态实现、局部运行或缺少 生成链路 的产物，尚不能证明可追溯正式路径完整采用；按审计规则不得升级为 PASS。

需要修改:
转为 PASS 所需条件：冻结并验证 scale/axis/complexity/material/Nvis/O/d/V 的覆盖标准。

### [G02-13] 状态: PARTIAL

要求:
是否显式存在与正常表面强度相近的 anomaly proxy。

发现:
采样字段把总体尺度、轴比、形状复杂度、位置和材质分开；开发定义显示一定 Nvis/O/d 变化，但没有预注册覆盖分箱，且五帧可见性 V 缺少 1–3。

证据:
- src/render.py:1217-1352
- src/render.py:3465-3842
- dev.json:20883-20892

判断:
只能证明静态实现、局部运行或缺少 生成链路 的产物，尚不能证明可追溯正式路径完整采用；按审计规则不得升级为 PASS。

需要修改:
转为 PASS 所需条件：冻结并验证 scale/axis/complexity/material/Nvis/O/d/V 的覆盖标准。

### [G02-14] 状态: PARTIAL

要求:
held-out generator mechanism 是否真正训练时不可见。

发现:
采样字段把总体尺度、轴比、形状复杂度、位置和材质分开；开发定义显示一定 Nvis/O/d 变化，但没有预注册覆盖分箱，且五帧可见性 V 缺少 1–3。

证据:
- src/render.py:1217-1352
- src/render.py:3465-3842
- dev.json:20883-20892

判断:
只能证明静态实现、局部运行或缺少 生成链路 的产物，尚不能证明可追溯正式路径完整采用；按审计规则不得升级为 PASS。

需要修改:
转为 PASS 所需条件：冻结并验证 scale/axis/complexity/material/Nvis/O/d/V 的覆盖标准。

## G03：世界级实体

### [G03-01] 状态: PARTIAL

要求:
entity 是否具有稳定 world-space geometry。

发现:
实体在 `WorldSpec` 中保存稳定世界坐标几何、位置、材质、内部编号和标签，第一版没有运动字段；30 个开发世界仍未通过序列可见性和物理放置验证。

证据:
- src/render.py:1217-1441
- dev.json:20883-20892

判断:
只能证明静态实现、局部运行或缺少 生成链路 的产物，尚不能证明可追溯正式路径完整采用；按审计规则不得升级为 PASS。

需要修改:
转为 PASS 所需条件：完成开发世界实体参数一致性、物理放置和跨帧可见性验证。

### [G03-02] 状态: PARTIAL

要求:
是否具有稳定 world position。

发现:
实体在 `WorldSpec` 中保存稳定世界坐标几何、位置、材质、内部编号和标签，第一版没有运动字段；30 个开发世界仍未通过序列可见性和物理放置验证。

证据:
- src/render.py:1217-1441
- dev.json:20883-20892

判断:
只能证明静态实现、局部运行或缺少 生成链路 的产物，尚不能证明可追溯正式路径完整采用；按审计规则不得升级为 PASS。

需要修改:
转为 PASS 所需条件：完成开发世界实体参数一致性、物理放置和跨帧可见性验证。

### [G03-03] 状态: PARTIAL

要求:
是否具有稳定 material。

发现:
实体在 `WorldSpec` 中保存稳定世界坐标几何、位置、材质、内部编号和标签，第一版没有运动字段；30 个开发世界仍未通过序列可见性和物理放置验证。

证据:
- src/render.py:1217-1441
- dev.json:20883-20892

判断:
只能证明静态实现、局部运行或缺少 生成链路 的产物，尚不能证明可追溯正式路径完整采用；按审计规则不得升级为 PASS。

需要修改:
转为 PASS 所需条件：完成开发世界实体参数一致性、物理放置和跨帧可见性验证。

### [G03-04] 状态: PARTIAL

要求:
是否有 generator-internal ID。

发现:
实体在 `WorldSpec` 中保存稳定世界坐标几何、位置、材质、内部编号和标签，第一版没有运动字段；30 个开发世界仍未通过序列可见性和物理放置验证。

证据:
- src/render.py:1217-1441
- dev.json:20883-20892

判断:
只能证明静态实现、局部运行或缺少 生成链路 的产物，尚不能证明可追溯正式路径完整采用；按审计规则不得升级为 PASS。

需要修改:
转为 PASS 所需条件：完成开发世界实体参数一致性、物理放置和跨帧可见性验证。

### [G03-05] 状态: PARTIAL

要求:
是否有 `normal-control` / `anomaly-proxy` 标签。

发现:
实体在 `WorldSpec` 中保存稳定世界坐标几何、位置、材质、内部编号和标签，第一版没有运动字段；30 个开发世界仍未通过序列可见性和物理放置验证。

证据:
- src/render.py:1217-1441
- dev.json:20883-20892

判断:
只能证明静态实现、局部运行或缺少 生成链路 的产物，尚不能证明可追溯正式路径完整采用；按审计规则不得升级为 PASS。

需要修改:
转为 PASS 所需条件：完成开发世界实体参数一致性、物理放置和跨帧可见性验证。

### [G03-06] 状态: PARTIAL

要求:
第一版实体是否默认静止。

发现:
实体在 `WorldSpec` 中保存稳定世界坐标几何、位置、材质、内部编号和标签，第一版没有运动字段；30 个开发世界仍未通过序列可见性和物理放置验证。

证据:
- src/render.py:1217-1441
- dev.json:20883-20892

判断:
只能证明静态实现、局部运行或缺少 生成链路 的产物，尚不能证明可追溯正式路径完整采用；按审计规则不得升级为 PASS。

需要修改:
转为 PASS 所需条件：完成开发世界实体参数一致性、物理放置和跨帧可见性验证。

### [G03-07] 状态: PASS

要求:
是否不存在“运动 = 异常”捷径。

发现:
实体在 `WorldSpec` 中保存稳定世界坐标几何、位置、材质、内部编号和标签，第一版没有运动字段；30 个开发世界仍未通过序列可见性和物理放置验证。

证据:
- src/render.py:1217-1441
- dev.json:20883-20892

判断:
证据直接覆盖该要求，或排除性全仓库搜索已证明冲突路径不在当前正式路线；可判 PASS。该结论只覆盖本条，不外推到整个模块或科学主张。

需要修改:
无需修改；保留当前实现和直接证据，并继续确保该局部结论不被外推为模块或科学主张已经成立。

### [G03-08] 状态: PARTIAL

要求:
同一 entity 跨帧是否保持身份和参数。

发现:
实体在 `WorldSpec` 中保存稳定世界坐标几何、位置、材质、内部编号和标签，第一版没有运动字段；30 个开发世界仍未通过序列可见性和物理放置验证。

证据:
- src/render.py:1217-1441
- dev.json:20883-20892

判断:
只能证明静态实现、局部运行或缺少 生成链路 的产物，尚不能证明可追溯正式路径完整采用；按审计规则不得升级为 PASS。

需要修改:
转为 PASS 所需条件：完成开发世界实体参数一致性、物理放置和跨帧可见性验证。

## G04：放置合法性

### [G04-01] 状态: PARTIAL

要求:
候选位置是否来自允许的正常地面语义区域。

发现:
地面语义、局部支撑平面、法向、yaw、接触和已观测几何碰撞检查均有实现；这些检查只针对可观测表面，开发产物中的物理放置与序列可见性验证仍为 false。

证据:
- src/render.py:3115-3405
- AJAE.tex:205
- dev.json:20883-20892

判断:
只能证明静态实现、局部运行或缺少 生成链路 的产物，尚不能证明可追溯正式路径完整采用；按审计规则不得升级为 PASS。

需要修改:
转为 PASS 所需条件：对固定开发定义真实执行放置与序列可见性审计并保存失败样例。

### [G04-02] 状态: PARTIAL

要求:
road 支持。

发现:
地面语义、局部支撑平面、法向、yaw、接触和已观测几何碰撞检查均有实现；这些检查只针对可观测表面，开发产物中的物理放置与序列可见性验证仍为 false。

证据:
- src/render.py:3115-3405
- AJAE.tex:205
- dev.json:20883-20892

判断:
只能证明静态实现、局部运行或缺少 生成链路 的产物，尚不能证明可追溯正式路径完整采用；按审计规则不得升级为 PASS。

需要修改:
转为 PASS 所需条件：对固定开发定义真实执行放置与序列可见性审计并保存失败样例。

### [G04-03] 状态: PARTIAL

要求:
parking 支持。

发现:
地面语义、局部支撑平面、法向、yaw、接触和已观测几何碰撞检查均有实现；这些检查只针对可观测表面，开发产物中的物理放置与序列可见性验证仍为 false。

证据:
- src/render.py:3115-3405
- AJAE.tex:205
- dev.json:20883-20892

判断:
只能证明静态实现、局部运行或缺少 生成链路 的产物，尚不能证明可追溯正式路径完整采用；按审计规则不得升级为 PASS。

需要修改:
转为 PASS 所需条件：对固定开发定义真实执行放置与序列可见性审计并保存失败样例。

### [G04-04] 状态: PARTIAL

要求:
sidewalk 支持。

发现:
地面语义、局部支撑平面、法向、yaw、接触和已观测几何碰撞检查均有实现；这些检查只针对可观测表面，开发产物中的物理放置与序列可见性验证仍为 false。

证据:
- src/render.py:3115-3405
- AJAE.tex:205
- dev.json:20883-20892

判断:
只能证明静态实现、局部运行或缺少 生成链路 的产物，尚不能证明可追溯正式路径完整采用；按审计规则不得升级为 PASS。

需要修改:
转为 PASS 所需条件：对固定开发定义真实执行放置与序列可见性审计并保存失败样例。

### [G04-05] 状态: PARTIAL

要求:
是否估计局部 support plane。

发现:
地面语义、局部支撑平面、法向、yaw、接触和已观测几何碰撞检查均有实现；这些检查只针对可观测表面，开发产物中的物理放置与序列可见性验证仍为 false。

证据:
- src/render.py:3115-3405
- AJAE.tex:205
- dev.json:20883-20892

判断:
只能证明静态实现、局部运行或缺少 生成链路 的产物，尚不能证明可追溯正式路径完整采用；按审计规则不得升级为 PASS。

需要修改:
转为 PASS 所需条件：对固定开发定义真实执行放置与序列可见性审计并保存失败样例。

### [G04-06] 状态: PARTIAL

要求:
entity vertical axis 是否与地面法向对齐。

发现:
地面语义、局部支撑平面、法向、yaw、接触和已观测几何碰撞检查均有实现；这些检查只针对可观测表面，开发产物中的物理放置与序列可见性验证仍为 false。

证据:
- src/render.py:3115-3405
- AJAE.tex:205
- dev.json:20883-20892

判断:
只能证明静态实现、局部运行或缺少 生成链路 的产物，尚不能证明可追溯正式路径完整采用；按审计规则不得升级为 PASS。

需要修改:
转为 PASS 所需条件：对固定开发定义真实执行放置与序列可见性审计并保存失败样例。

### [G04-07] 状态: PARTIAL

要求:
yaw 是否独立随机采样。

发现:
地面语义、局部支撑平面、法向、yaw、接触和已观测几何碰撞检查均有实现；这些检查只针对可观测表面，开发产物中的物理放置与序列可见性验证仍为 false。

证据:
- src/render.py:3115-3405
- AJAE.tex:205
- dev.json:20883-20892

判断:
只能证明静态实现、局部运行或缺少 生成链路 的产物，尚不能证明可追溯正式路径完整采用；按审计规则不得升级为 PASS。

需要修改:
转为 PASS 所需条件：对固定开发定义真实执行放置与序列可见性审计并保存失败样例。

### [G04-08] 状态: PARTIAL

要求:
是否检查悬空。

发现:
地面语义、局部支撑平面、法向、yaw、接触和已观测几何碰撞检查均有实现；这些检查只针对可观测表面，开发产物中的物理放置与序列可见性验证仍为 false。

证据:
- src/render.py:3115-3405
- AJAE.tex:205
- dev.json:20883-20892

判断:
只能证明静态实现、局部运行或缺少 生成链路 的产物，尚不能证明可追溯正式路径完整采用；按审计规则不得升级为 PASS。

需要修改:
转为 PASS 所需条件：对固定开发定义真实执行放置与序列可见性审计并保存失败样例。

### [G04-09] 状态: PARTIAL

要求:
是否检查大面积埋地。

发现:
地面语义、局部支撑平面、法向、yaw、接触和已观测几何碰撞检查均有实现；这些检查只针对可观测表面，开发产物中的物理放置与序列可见性验证仍为 false。

证据:
- src/render.py:3115-3405
- AJAE.tex:205
- dev.json:20883-20892

判断:
只能证明静态实现、局部运行或缺少 生成链路 的产物，尚不能证明可追溯正式路径完整采用；按审计规则不得升级为 PASS。

需要修改:
转为 PASS 所需条件：对固定开发定义真实执行放置与序列可见性审计并保存失败样例。

### [G04-10] 状态: PARTIAL

要求:
是否检查与已观测非地面几何穿插。

发现:
地面语义、局部支撑平面、法向、yaw、接触和已观测几何碰撞检查均有实现；这些检查只针对可观测表面，开发产物中的物理放置与序列可见性验证仍为 false。

证据:
- src/render.py:3115-3405
- AJAE.tex:205
- dev.json:20883-20892

判断:
只能证明静态实现、局部运行或缺少 生成链路 的产物，尚不能证明可追溯正式路径完整采用；按审计规则不得升级为 PASS。

需要修改:
转为 PASS 所需条件：对固定开发定义真实执行放置与序列可见性审计并保存失败样例。

### [G04-11] 状态: PARTIAL

要求:
是否检查 inserted entities 互相穿插。

发现:
地面语义、局部支撑平面、法向、yaw、接触和已观测几何碰撞检查均有实现；这些检查只针对可观测表面，开发产物中的物理放置与序列可见性验证仍为 false。

证据:
- src/render.py:3115-3405
- AJAE.tex:205
- dev.json:20883-20892

判断:
只能证明静态实现、局部运行或缺少 生成链路 的产物，尚不能证明可追溯正式路径完整采用；按审计规则不得升级为 PASS。

需要修改:
转为 PASS 所需条件：对固定开发定义真实执行放置与序列可见性审计并保存失败样例。

### [G04-12] 状态: PASS

要求:
代码/文档是否避免声称“保证隐藏区域无碰撞”。

发现:
地面语义、局部支撑平面、法向、yaw、接触和已观测几何碰撞检查均有实现；这些检查只针对可观测表面，开发产物中的物理放置与序列可见性验证仍为 false。

证据:
- src/render.py:3115-3405
- AJAE.tex:205
- dev.json:20883-20892

判断:
证据直接覆盖该要求，或排除性全仓库搜索已证明冲突路径不在当前正式路线；可判 PASS。该结论只覆盖本条，不外推到整个模块或科学主张。

需要修改:
无需修改；保留当前实现和直接证据，并继续确保该局部结论不被外推为模块或科学主张已经成立。

## G05：世界类型和实体数量

### [G05-01] 状态: PARTIAL

要求:
anomaly count 是否允许 0。

发现:
代码允许 0–9 个实体并使非零质量集中于 1–3，混合世界按支撑语义与距离层配对 control/proxy；没有正式 anomaly-only 世界、匹配三元组或 normal/proxy 分布对照。

证据:
- src/render.py:3465-3759
- protocol.json:100-148
- test_ajae.py:414-532

判断:
只能证明静态实现、局部运行或缺少 生成链路 的产物，尚不能证明可追溯正式路径完整采用；按审计规则不得升级为 PASS。

需要修改:
转为 PASS 所需条件：运行冻结的世界类型与匹配分布审计，覆盖 anomaly-only 和 matched-trio。

### [G05-02] 状态: PARTIAL

要求:
最大是否允许到 9 或明确等价实现。

发现:
代码允许 0–9 个实体并使非零质量集中于 1–3，混合世界按支撑语义与距离层配对 control/proxy；没有正式 anomaly-only 世界、匹配三元组或 normal/proxy 分布对照。

证据:
- src/render.py:3465-3759
- protocol.json:100-148
- test_ajae.py:414-532

判断:
只能证明静态实现、局部运行或缺少 生成链路 的产物，尚不能证明可追溯正式路径完整采用；按审计规则不得升级为 PASS。

需要修改:
转为 PASS 所需条件：运行冻结的世界类型与匹配分布审计，覆盖 anomaly-only 和 matched-trio。

### [G05-03] 状态: PARTIAL

要求:
主概率质量是否集中在 1–3 个 anomaly proxy。

发现:
代码允许 0–9 个实体并使非零质量集中于 1–3，混合世界按支撑语义与距离层配对 control/proxy；没有正式 anomaly-only 世界、匹配三元组或 normal/proxy 分布对照。

证据:
- src/render.py:3465-3759
- protocol.json:100-148
- test_ajae.py:414-532

判断:
只能证明静态实现、局部运行或缺少 生成链路 的产物，尚不能证明可追溯正式路径完整采用；按审计规则不得升级为 PASS。

需要修改:
转为 PASS 所需条件：运行冻结的世界类型与匹配分布审计，覆盖 anomaly-only 和 matched-trio。

### [G05-04] 状态: PARTIAL

要求:
是否存在 pure-normal canonical worlds。

发现:
代码允许 0–9 个实体并使非零质量集中于 1–3，混合世界按支撑语义与距离层配对 control/proxy；没有正式 anomaly-only 世界、匹配三元组或 normal/proxy 分布对照。

证据:
- src/render.py:3465-3759
- protocol.json:100-148
- test_ajae.py:414-532

判断:
只能证明静态实现、局部运行或缺少 生成链路 的产物，尚不能证明可追溯正式路径完整采用；按审计规则不得升级为 PASS。

需要修改:
转为 PASS 所需条件：运行冻结的世界类型与匹配分布审计，覆盖 anomaly-only 和 matched-trio。

### [G05-05] 状态: PARTIAL

要求:
是否存在 only-normal-control worlds。

发现:
代码允许 0–9 个实体并使非零质量集中于 1–3，混合世界按支撑语义与距离层配对 control/proxy；没有正式 anomaly-only 世界、匹配三元组或 normal/proxy 分布对照。

证据:
- src/render.py:3465-3759
- protocol.json:100-148
- test_ajae.py:414-532

判断:
只能证明静态实现、局部运行或缺少 生成链路 的产物，尚不能证明可追溯正式路径完整采用；按审计规则不得升级为 PASS。

需要修改:
转为 PASS 所需条件：运行冻结的世界类型与匹配分布审计，覆盖 anomaly-only 和 matched-trio。

### [G05-06] 状态: PARTIAL

要求:
是否存在 normal-control + anomaly-proxy mixed worlds。

发现:
代码允许 0–9 个实体并使非零质量集中于 1–3，混合世界按支撑语义与距离层配对 control/proxy；没有正式 anomaly-only 世界、匹配三元组或 normal/proxy 分布对照。

证据:
- src/render.py:3465-3759
- protocol.json:100-148
- test_ajae.py:414-532

判断:
只能证明静态实现、局部运行或缺少 生成链路 的产物，尚不能证明可追溯正式路径完整采用；按审计规则不得升级为 PASS。

需要修改:
转为 PASS 所需条件：运行冻结的世界类型与匹配分布审计，覆盖 anomaly-only 和 matched-trio。

### [G05-07] 状态: BLOCKED

要求:
是否存在 anomaly-only inserted worlds。

发现:
代码允许 0–9 个实体并使非零质量集中于 1–3，混合世界按支撑语义与距离层配对 control/proxy；没有正式 anomaly-only 世界、匹配三元组或 normal/proxy 分布对照。

证据:
- src/render.py:3465-3759
- protocol.json:100-148
- test_ajae.py:414-532

判断:
上游 Decision Gate 或冻结条件未满足，本条原则上不得继续；判定 BLOCKED。

需要修改:
转为 PASS 所需条件：运行冻结的世界类型与匹配分布审计，覆盖 anomaly-only 和 matched-trio。

### [G05-08] 状态: NOT RUN

要求:
normal-control 与 anomaly proxy 距离分布是否可比较。

发现:
代码允许 0–9 个实体并使非零质量集中于 1–3，混合世界按支撑语义与距离层配对 control/proxy；没有正式 anomaly-only 世界、匹配三元组或 normal/proxy 分布对照。

证据:
- src/render.py:3465-3759
- protocol.json:100-148
- test_ajae.py:414-532

判断:
本条要求的运行级审计、实验或正式产物没有发生；代码存在也不能替代执行，判定 NOT RUN。

需要修改:
转为 PASS 所需条件：运行冻结的世界类型与匹配分布审计，覆盖 anomaly-only 和 matched-trio。

### [G05-09] 状态: NOT RUN

要求:
可见点数分布是否可比较。

发现:
代码允许 0–9 个实体并使非零质量集中于 1–3，混合世界按支撑语义与距离层配对 control/proxy；没有正式 anomaly-only 世界、匹配三元组或 normal/proxy 分布对照。

证据:
- src/render.py:3465-3759
- protocol.json:100-148
- test_ajae.py:414-532

判断:
本条要求的运行级审计、实验或正式产物没有发生；代码存在也不能替代执行，判定 NOT RUN。

需要修改:
转为 PASS 所需条件：运行冻结的世界类型与匹配分布审计，覆盖 anomaly-only 和 matched-trio。

### [G05-10] 状态: NOT RUN

要求:
遮挡程度是否可比较。

发现:
代码允许 0–9 个实体并使非零质量集中于 1–3，混合世界按支撑语义与距离层配对 control/proxy；没有正式 anomaly-only 世界、匹配三元组或 normal/proxy 分布对照。

证据:
- src/render.py:3465-3759
- protocol.json:100-148
- test_ajae.py:414-532

判断:
本条要求的运行级审计、实验或正式产物没有发生；代码存在也不能替代执行，判定 NOT RUN。

需要修改:
转为 PASS 所需条件：运行冻结的世界类型与匹配分布审计，覆盖 anomaly-only 和 matched-trio。

### [G05-11] 状态: NOT RUN

要求:
placement surface 是否可比较。

发现:
代码允许 0–9 个实体并使非零质量集中于 1–3，混合世界按支撑语义与距离层配对 control/proxy；没有正式 anomaly-only 世界、匹配三元组或 normal/proxy 分布对照。

证据:
- src/render.py:3465-3759
- protocol.json:100-148
- test_ajae.py:414-532

判断:
本条要求的运行级审计、实验或正式产物没有发生；代码存在也不能替代执行，判定 NOT RUN。

需要修改:
转为 PASS 所需条件：运行冻结的世界类型与匹配分布审计，覆盖 anomaly-only 和 matched-trio。

### [G05-12] 状态: NOT RUN

要求:
相同背景位置能否在不同世界放 normal / anomaly / nothing。

发现:
代码允许 0–9 个实体并使非零质量集中于 1–3，混合世界按支撑语义与距离层配对 control/proxy；没有正式 anomaly-only 世界、匹配三元组或 normal/proxy 分布对照。

证据:
- src/render.py:3465-3759
- protocol.json:100-148
- test_ajae.py:414-532

判断:
本条要求的运行级审计、实验或正式产物没有发生；代码存在也不能替代执行，判定 NOT RUN。

需要修改:
转为 PASS 所需条件：运行冻结的世界类型与匹配分布审计，覆盖 anomaly-only 和 matched-trio。

### [G05-13] 状态: PARTIAL

要求:
标签是否不能通过 placement position 单独推断。

发现:
代码允许 0–9 个实体并使非零质量集中于 1–3，混合世界按支撑语义与距离层配对 control/proxy；没有正式 anomaly-only 世界、匹配三元组或 normal/proxy 分布对照。

证据:
- src/render.py:3465-3759
- protocol.json:100-148
- test_ajae.py:414-532

判断:
只能证明静态实现、局部运行或缺少 生成链路 的产物，尚不能证明可追溯正式路径完整采用；按审计规则不得升级为 PASS。

需要修改:
转为 PASS 所需条件：运行冻结的世界类型与匹配分布审计，覆盖 anomaly-only 和 matched-trio。

### [G05-14] 状态: PARTIAL

要求:
“有新增回波”是否不能直接推断异常。

发现:
代码允许 0–9 个实体并使非零质量集中于 1–3，混合世界按支撑语义与距离层配对 control/proxy；没有正式 anomaly-only 世界、匹配三元组或 normal/proxy 分布对照。

证据:
- src/render.py:3465-3759
- protocol.json:100-148
- test_ajae.py:414-532

判断:
只能证明静态实现、局部运行或缺少 生成链路 的产物，尚不能证明可追溯正式路径完整采用；按审计规则不得升级为 PASS。

需要修改:
转为 PASS 所需条件：运行冻结的世界类型与匹配分布审计，覆盖 anomaly-only 和 matched-trio。

## S00：规范射线网格

### [S00-01] 状态: PARTIAL

要求:
每帧使用统一校准 ray set。

发现:
128×1024 beam-major 规范射线网格、beam/column、原点、单位方向和空射线无穷距离均有实现；校准和往返只绑定 train/206，Gate1 结论仍未冻结。

证据:
- src/render.py:1445-1745
- runs/ajae/calibration.pt
- dev.json:8363-8368

判断:
只能证明静态实现、局部运行或缺少 生成链路 的产物，尚不能证明可追溯正式路径完整采用；按审计规则不得升级为 PASS。

需要修改:
转为 PASS 所需条件：把 ray-grid 校准、映射和判定标准绑定同一可追溯产物。

### [S00-02] 状态: PARTIAL

要求:
ray 由 beam ID 唯一标识部分身份。

发现:
128×1024 beam-major 规范射线网格、beam/column、原点、单位方向和空射线无穷距离均有实现；校准和往返只绑定 train/206，Gate1 结论仍未冻结。

证据:
- src/render.py:1445-1745
- runs/ajae/calibration.pt
- dev.json:8363-8368

判断:
只能证明静态实现、局部运行或缺少 生成链路 的产物，尚不能证明可追溯正式路径完整采用；按审计规则不得升级为 PASS。

需要修改:
转为 PASS 所需条件：把 ray-grid 校准、映射和判定标准绑定同一可追溯产物。

### [S00-03] 状态: PARTIAL

要求:
ray 由 azimuth column 完成唯一身份。

发现:
128×1024 beam-major 规范射线网格、beam/column、原点、单位方向和空射线无穷距离均有实现；校准和往返只绑定 train/206，Gate1 结论仍未冻结。

证据:
- src/render.py:1445-1745
- runs/ajae/calibration.pt
- dev.json:8363-8368

判断:
只能证明静态实现、局部运行或缺少 生成链路 的产物，尚不能证明可追溯正式路径完整采用；按审计规则不得升级为 PASS。

需要修改:
转为 PASS 所需条件：把 ray-grid 校准、映射和判定标准绑定同一可追溯产物。

### [S00-04] 状态: PARTIAL

要求:
存储 sensor origin。

发现:
128×1024 beam-major 规范射线网格、beam/column、原点、单位方向和空射线无穷距离均有实现；校准和往返只绑定 train/206，Gate1 结论仍未冻结。

证据:
- src/render.py:1445-1745
- runs/ajae/calibration.pt
- dev.json:8363-8368

判断:
只能证明静态实现、局部运行或缺少 生成链路 的产物，尚不能证明可追溯正式路径完整采用；按审计规则不得升级为 PASS。

需要修改:
转为 PASS 所需条件：把 ray-grid 校准、映射和判定标准绑定同一可追溯产物。

### [S00-05] 状态: PARTIAL

要求:
存储 unit direction。

发现:
128×1024 beam-major 规范射线网格、beam/column、原点、单位方向和空射线无穷距离均有实现；校准和往返只绑定 train/206，Gate1 结论仍未冻结。

证据:
- src/render.py:1445-1745
- runs/ajae/calibration.pt
- dev.json:8363-8368

判断:
只能证明静态实现、局部运行或缺少 生成链路 的产物，尚不能证明可追溯正式路径完整采用；按审计规则不得升级为 PASS。

需要修改:
转为 PASS 所需条件：把 ray-grid 校准、映射和判定标准绑定同一可追溯产物。

### [S00-06] 状态: PARTIAL

要求:
原始有效回波能投到对应 canonical ray。

发现:
128×1024 beam-major 规范射线网格、beam/column、原点、单位方向和空射线无穷距离均有实现；校准和往返只绑定 train/206，Gate1 结论仍未冻结。

证据:
- src/render.py:1445-1745
- runs/ajae/calibration.pt
- dev.json:8363-8368

判断:
只能证明静态实现、局部运行或缺少 生成链路 的产物，尚不能证明可追溯正式路径完整采用；按审计规则不得升级为 PASS。

需要修改:
转为 PASS 所需条件：把 ray-grid 校准、映射和判定标准绑定同一可追溯产物。

### [S00-07] 状态: PARTIAL

要求:
empty ray 使用 $+\infty$ 或严格等价逻辑。

发现:
128×1024 beam-major 规范射线网格、beam/column、原点、单位方向和空射线无穷距离均有实现；校准和往返只绑定 train/206，Gate1 结论仍未冻结。

证据:
- src/render.py:1445-1745
- runs/ajae/calibration.pt
- dev.json:8363-8368

判断:
只能证明静态实现、局部运行或缺少 生成链路 的产物，尚不能证明可追溯正式路径完整采用；按审计规则不得升级为 PASS。

需要修改:
转为 PASS 所需条件：把 ray-grid 校准、映射和判定标准绑定同一可追溯产物。

## S01：射线与表面求交

### [S01-01] 状态: PARTIAL

要求:
对每个 inserted entity 求最近几何交点。

发现:
所有实体通过 shape-level `intersect` 进入统一最近候选竞争，数值无效交点被显式拒绝；真实 mixed 开发世界未完成可追溯验证。

证据:
- src/render.py:2707-2779
- src/render.py:2813-3073

判断:
只能证明静态实现、局部运行或缺少 生成链路 的产物，尚不能证明可追溯正式路径完整采用；按审计规则不得升级为 PASS。

需要修改:
转为 PASS 所需条件：对冻结 mixed worlds 保存交点/竞争数值稳定性运行证据。

### [S01-02] 状态: PARTIAL

要求:
normal-control 和 anomaly-proxy 完全共享 intersection code。

发现:
所有实体通过 shape-level `intersect` 进入统一最近候选竞争，数值无效交点被显式拒绝；真实 mixed 开发世界未完成可追溯验证。

证据:
- src/render.py:2707-2779
- src/render.py:2813-3073

判断:
只能证明静态实现、局部运行或缺少 生成链路 的产物，尚不能证明可追溯正式路径完整采用；按审计规则不得升级为 PASS。

需要修改:
转为 PASS 所需条件：对冻结 mixed worlds 保存交点/竞争数值稳定性运行证据。

### [S01-03] 状态: PARTIAL

要求:
多 inserted entities 时取最近候选。

发现:
所有实体通过 shape-level `intersect` 进入统一最近候选竞争，数值无效交点被显式拒绝；真实 mixed 开发世界未完成可追溯验证。

证据:
- src/render.py:2707-2779
- src/render.py:2813-3073

判断:
只能证明静态实现、局部运行或缺少 生成链路 的产物，尚不能证明可追溯正式路径完整采用；按审计规则不得升级为 PASS。

需要修改:
转为 PASS 所需条件：对冻结 mixed worlds 保存交点/竞争数值稳定性运行证据。

### [S01-04] 状态: PARTIAL

要求:
intersection numerical failure 有显式处理，而非 silent NaN。

发现:
所有实体通过 shape-level `intersect` 进入统一最近候选竞争，数值无效交点被显式拒绝；真实 mixed 开发世界未完成可追溯验证。

证据:
- src/render.py:2707-2779
- src/render.py:2813-3073

判断:
只能证明静态实现、局部运行或缺少 生成链路 的产物，尚不能证明可追溯正式路径完整采用；按审计规则不得升级为 PASS。

需要修改:
转为 PASS 所需条件：对冻结 mixed worlds 保存交点/竞争数值稳定性运行证据。

## S02：回波概率

### [S02-01] 状态: PARTIAL

要求:
几何命中是否不会自动成为有效回波。

发现:
几何命中还需通过由 beam、range、入射角和材质调制的回波概率，模型由 206 校准且 normal/proxy 共用；正式分布一致性尚未成立。

证据:
- src/render.py:2109-2675
- src/render.py:2707-2779

判断:
只能证明静态实现、局部运行或缺少 生成链路 的产物，尚不能证明可追溯正式路径完整采用；按审计规则不得升级为 PASS。

需要修改:
转为 PASS 所需条件：在 Gate1 传感器分布产物中证明 control/proxy 共用且分布符合冻结标准。

### [S02-02] 状态: PARTIAL

要求:
是否存在 $P(return\mid b,d,\mu,\rho)$ 或等价模型。

发现:
几何命中还需通过由 beam、range、入射角和材质调制的回波概率，模型由 206 校准且 normal/proxy 共用；正式分布一致性尚未成立。

证据:
- src/render.py:2109-2675
- src/render.py:2707-2779

判断:
只能证明静态实现、局部运行或缺少 生成链路 的产物，尚不能证明可追溯正式路径完整采用；按审计规则不得升级为 PASS。

需要修改:
转为 PASS 所需条件：在 Gate1 传感器分布产物中证明 control/proxy 共用且分布符合冻结标准。

### [S02-03] 状态: PARTIAL

要求:
使用 beam。

发现:
几何命中还需通过由 beam、range、入射角和材质调制的回波概率，模型由 206 校准且 normal/proxy 共用；正式分布一致性尚未成立。

证据:
- src/render.py:2109-2675
- src/render.py:2707-2779

判断:
只能证明静态实现、局部运行或缺少 生成链路 的产物，尚不能证明可追溯正式路径完整采用；按审计规则不得升级为 PASS。

需要修改:
转为 PASS 所需条件：在 Gate1 传感器分布产物中证明 control/proxy 共用且分布符合冻结标准。

### [S02-04] 状态: PARTIAL

要求:
使用 distance。

发现:
几何命中还需通过由 beam、range、入射角和材质调制的回波概率，模型由 206 校准且 normal/proxy 共用；正式分布一致性尚未成立。

证据:
- src/render.py:2109-2675
- src/render.py:2707-2779

判断:
只能证明静态实现、局部运行或缺少 生成链路 的产物，尚不能证明可追溯正式路径完整采用；按审计规则不得升级为 PASS。

需要修改:
转为 PASS 所需条件：在 Gate1 传感器分布产物中证明 control/proxy 共用且分布符合冻结标准。

### [S02-05] 状态: PARTIAL

要求:
使用 incidence cosine。

发现:
几何命中还需通过由 beam、range、入射角和材质调制的回波概率，模型由 206 校准且 normal/proxy 共用；正式分布一致性尚未成立。

证据:
- src/render.py:2109-2675
- src/render.py:2707-2779

判断:
只能证明静态实现、局部运行或缺少 生成链路 的产物，尚不能证明可追溯正式路径完整采用；按审计规则不得升级为 PASS。

需要修改:
转为 PASS 所需条件：在 Gate1 传感器分布产物中证明 control/proxy 共用且分布符合冻结标准。

### [S02-06] 状态: PARTIAL

要求:
使用 material state 或合理调制。

发现:
几何命中还需通过由 beam、range、入射角和材质调制的回波概率，模型由 206 校准且 normal/proxy 共用；正式分布一致性尚未成立。

证据:
- src/render.py:2109-2675
- src/render.py:2707-2779

判断:
只能证明静态实现、局部运行或缺少 生成链路 的产物，尚不能证明可追溯正式路径完整采用；按审计规则不得升级为 PASS。

需要修改:
转为 PASS 所需条件：在 Gate1 传感器分布产物中证明 control/proxy 共用且分布符合冻结标准。

### [S02-07] 状态: PARTIAL

要求:
return model 是否由 206 校准。

发现:
几何命中还需通过由 beam、range、入射角和材质调制的回波概率，模型由 206 校准且 normal/proxy 共用；正式分布一致性尚未成立。

证据:
- src/render.py:2109-2675
- src/render.py:2707-2779

判断:
只能证明静态实现、局部运行或缺少 生成链路 的产物，尚不能证明可追溯正式路径完整采用；按审计规则不得升级为 PASS。

需要修改:
转为 PASS 所需条件：在 Gate1 传感器分布产物中证明 control/proxy 共用且分布符合冻结标准。

### [S02-08] 状态: PARTIAL

要求:
normal/anomaly 是否共享同一 return model。

发现:
几何命中还需通过由 beam、range、入射角和材质调制的回波概率，模型由 206 校准且 normal/proxy 共用；正式分布一致性尚未成立。

证据:
- src/render.py:2109-2675
- src/render.py:2707-2779

判断:
只能证明静态实现、局部运行或缺少 生成链路 的产物，尚不能证明可追溯正式路径完整采用；按审计规则不得升级为 PASS。

需要修改:
转为 PASS 所需条件：在 Gate1 传感器分布产物中证明 control/proxy 共用且分布符合冻结标准。

## S03：最近回波竞争

### [S03-01] 状态: PARTIAL

要求:
inserted return 更近时替换原正常回波。

发现:
插入回波只有更近且被接受时替换原回波，空射线以无穷距离参与同一竞争；双向遮挡和 empty-ray 两项定向运行测试缺失，正式 loss 未执行。

证据:
- src/render.py:2813-3073
- test_ajae.py（未发现 U03/U04 定向测试）

判断:
只能证明静态实现、局部运行或缺少 生成链路 的产物，尚不能证明可追溯正式路径完整采用；按审计规则不得升级为 PASS。

需要修改:
转为 PASS 所需条件：补做双向遮挡和空射线接受/拒绝的运行级测试，并在真实监督张量核对被移除点。

### [S03-02] 状态: PARTIAL

要求:
original foreground 更近时 inserted entity 被遮挡。

发现:
插入回波只有更近且被接受时替换原回波，空射线以无穷距离参与同一竞争；双向遮挡和 empty-ray 两项定向运行测试缺失，正式 loss 未执行。

证据:
- src/render.py:2813-3073
- test_ajae.py（未发现 U03/U04 定向测试）

判断:
只能证明静态实现、局部运行或缺少 生成链路 的产物，尚不能证明可追溯正式路径完整采用；按审计规则不得升级为 PASS。

需要修改:
转为 PASS 所需条件：补做双向遮挡和空射线接受/拒绝的运行级测试，并在真实监督张量核对被移除点。

### [S03-03] 状态: PARTIAL

要求:
empty ray 可产生新回波。

发现:
插入回波只有更近且被接受时替换原回波，空射线以无穷距离参与同一竞争；双向遮挡和 empty-ray 两项定向运行测试缺失，正式 loss 未执行。

证据:
- src/render.py:2813-3073
- test_ajae.py（未发现 U03/U04 定向测试）

判断:
只能证明静态实现、局部运行或缺少 生成链路 的产物，尚不能证明可追溯正式路径完整采用；按审计规则不得升级为 PASS。

需要修改:
转为 PASS 所需条件：补做双向遮挡和空射线接受/拒绝的运行级测试，并在真实监督张量核对被移除点。

### [S03-04] 状态: PARTIAL

要求:
geometry hit 但 return rejected 时不得产生点。

发现:
插入回波只有更近且被接受时替换原回波，空射线以无穷距离参与同一竞争；双向遮挡和 empty-ray 两项定向运行测试缺失，正式 loss 未执行。

证据:
- src/render.py:2813-3073
- test_ajae.py（未发现 U03/U04 定向测试）

判断:
只能证明静态实现、局部运行或缺少 生成链路 的产物，尚不能证明可追溯正式路径完整采用；按审计规则不得升级为 PASS。

需要修改:
转为 PASS 所需条件：补做双向遮挡和空射线接受/拒绝的运行级测试，并在真实监督张量核对被移除点。

### [S03-05] 状态: PARTIAL

要求:
是否正确实现双向遮挡逻辑。

发现:
插入回波只有更近且被接受时替换原回波，空射线以无穷距离参与同一竞争；双向遮挡和 empty-ray 两项定向运行测试缺失，正式 loss 未执行。

证据:
- src/render.py:2813-3073
- test_ajae.py（未发现 U03/U04 定向测试）

判断:
只能证明静态实现、局部运行或缺少 生成链路 的产物，尚不能证明可追溯正式路径完整采用；按审计规则不得升级为 PASS。

需要修改:
转为 PASS 所需条件：补做双向遮挡和空射线接受/拒绝的运行级测试，并在真实监督张量核对被移除点。

### [S03-06] 状态: PARTIAL

要求:
被遮挡掉的原始点是否从反事实观测集合移除。

发现:
插入回波只有更近且被接受时替换原回波，空射线以无穷距离参与同一竞争；双向遮挡和 empty-ray 两项定向运行测试缺失，正式 loss 未执行。

证据:
- src/render.py:2813-3073
- test_ajae.py（未发现 U03/U04 定向测试）

判断:
只能证明静态实现、局部运行或缺少 生成链路 的产物，尚不能证明可追溯正式路径完整采用；按审计规则不得升级为 PASS。

需要修改:
转为 PASS 所需条件：补做双向遮挡和空射线接受/拒绝的运行级测试，并在真实监督张量核对被移除点。

### [S03-07] 状态: BLOCKED

要求:
被移除点是否不参与 loss。

发现:
插入回波只有更近且被接受时替换原回波，空射线以无穷距离参与同一竞争；双向遮挡和 empty-ray 两项定向运行测试缺失，正式 loss 未执行。

证据:
- src/render.py:2813-3073
- test_ajae.py（未发现 U03/U04 定向测试）

判断:
上游 Decision Gate 或冻结条件未满足，本条原则上不得继续；判定 BLOCKED。

需要修改:
转为 PASS 所需条件：补做双向遮挡和空射线接受/拒绝的运行级测试，并在真实监督张量核对被移除点。

## S04：强度模型

### [S04-01] 状态: PARTIAL

要求:
是否在 206 上估计条件 intensity 分布。

发现:
206 校准产物包含 beam×range×incidence 条件强度分位数，材质作平滑调制并裁剪到实际支持；产物缺 生成链路，normal/proxy 的真实强度分布对照没有完成。

证据:
- src/render.py:2109-2675
- src/render.py:2290-2328
- runs/ajae/calibration.pt

判断:
只能证明静态实现、局部运行或缺少 生成链路 的产物，尚不能证明可追溯正式路径完整采用；按审计规则不得升级为 PASS。

需要修改:
转为 PASS 所需条件：保存可追溯的完整条件强度分布，并包含 normal-control/proxy 对照。

### [S04-02] 状态: PARTIAL

要求:
条件包含 beam。

发现:
206 校准产物包含 beam×range×incidence 条件强度分位数，材质作平滑调制并裁剪到实际支持；产物缺 生成链路，normal/proxy 的真实强度分布对照没有完成。

证据:
- src/render.py:2109-2675
- src/render.py:2290-2328
- runs/ajae/calibration.pt

判断:
只能证明静态实现、局部运行或缺少 生成链路 的产物，尚不能证明可追溯正式路径完整采用；按审计规则不得升级为 PASS。

需要修改:
转为 PASS 所需条件：保存可追溯的完整条件强度分布，并包含 normal-control/proxy 对照。

### [S04-03] 状态: PARTIAL

要求:
条件包含 range。

发现:
206 校准产物包含 beam×range×incidence 条件强度分位数，材质作平滑调制并裁剪到实际支持；产物缺 生成链路，normal/proxy 的真实强度分布对照没有完成。

证据:
- src/render.py:2109-2675
- src/render.py:2290-2328
- runs/ajae/calibration.pt

判断:
只能证明静态实现、局部运行或缺少 生成链路 的产物，尚不能证明可追溯正式路径完整采用；按审计规则不得升级为 PASS。

需要修改:
转为 PASS 所需条件：保存可追溯的完整条件强度分布，并包含 normal-control/proxy 对照。

### [S04-04] 状态: PARTIAL

要求:
条件包含 incidence。

发现:
206 校准产物包含 beam×range×incidence 条件强度分位数，材质作平滑调制并裁剪到实际支持；产物缺 生成链路，normal/proxy 的真实强度分布对照没有完成。

证据:
- src/render.py:2109-2675
- src/render.py:2290-2328
- runs/ajae/calibration.pt

判断:
只能证明静态实现、局部运行或缺少 生成链路 的产物，尚不能证明可追溯正式路径完整采用；按审计规则不得升级为 PASS。

需要修改:
转为 PASS 所需条件：保存可追溯的完整条件强度分布，并包含 normal-control/proxy 对照。

### [S04-05] 状态: PARTIAL

要求:
material 是否只做平滑调制。

发现:
206 校准产物包含 beam×range×incidence 条件强度分位数，材质作平滑调制并裁剪到实际支持；产物缺 生成链路，normal/proxy 的真实强度分布对照没有完成。

证据:
- src/render.py:2109-2675
- src/render.py:2290-2328
- runs/ajae/calibration.pt

判断:
只能证明静态实现、局部运行或缺少 生成链路 的产物，尚不能证明可追溯正式路径完整采用；按审计规则不得升级为 PASS。

需要修改:
转为 PASS 所需条件：保存可追溯的完整条件强度分布，并包含 normal-control/proxy 对照。

### [S04-06] 状态: PARTIAL

要求:
是否有 noise 项。

发现:
206 校准产物包含 beam×range×incidence 条件强度分位数，材质作平滑调制并裁剪到实际支持；产物缺 生成链路，normal/proxy 的真实强度分布对照没有完成。

证据:
- src/render.py:2109-2675
- src/render.py:2290-2328
- runs/ajae/calibration.pt

判断:
只能证明静态实现、局部运行或缺少 生成链路 的产物，尚不能证明可追溯正式路径完整采用；按审计规则不得升级为 PASS。

需要修改:
转为 PASS 所需条件：保存可追溯的完整条件强度分布，并包含 normal-control/proxy 对照。

### [S04-07] 状态: PARTIAL

要求:
intensity 是否裁剪到 206 实际支持范围。

发现:
206 校准产物包含 beam×range×incidence 条件强度分位数，材质作平滑调制并裁剪到实际支持；产物缺 生成链路，normal/proxy 的真实强度分布对照没有完成。

证据:
- src/render.py:2109-2675
- src/render.py:2290-2328
- runs/ajae/calibration.pt

判断:
只能证明静态实现、局部运行或缺少 生成链路 的产物，尚不能证明可追溯正式路径完整采用；按审计规则不得升级为 PASS。

需要修改:
转为 PASS 所需条件：保存可追溯的完整条件强度分布，并包含 normal-control/proxy 对照。

### [S04-08] 状态: PARTIAL

要求:
normal-control 与 anomaly-proxy 完全共享 intensity pipeline。

发现:
206 校准产物包含 beam×range×incidence 条件强度分位数，材质作平滑调制并裁剪到实际支持；产物缺 生成链路，normal/proxy 的真实强度分布对照没有完成。

证据:
- src/render.py:2109-2675
- src/render.py:2290-2328
- runs/ajae/calibration.pt

判断:
只能证明静态实现、局部运行或缺少 生成链路 的产物，尚不能证明可追溯正式路径完整采用；按审计规则不得升级为 PASS。

需要修改:
转为 PASS 所需条件：保存可追溯的完整条件强度分布，并包含 normal-control/proxy 对照。

### [S04-09] 状态: PARTIAL

要求:
材质与 anomaly label 是否没有固定映射。

发现:
206 校准产物包含 beam×range×incidence 条件强度分位数，材质作平滑调制并裁剪到实际支持；产物缺 生成链路，normal/proxy 的真实强度分布对照没有完成。

证据:
- src/render.py:2109-2675
- src/render.py:2290-2328
- runs/ajae/calibration.pt

判断:
只能证明静态实现、局部运行或缺少 生成链路 的产物，尚不能证明可追溯正式路径完整采用；按审计规则不得升级为 PASS。

需要修改:
转为 PASS 所需条件：保存可追溯的完整条件强度分布，并包含 normal-control/proxy 对照。

## S05：物理主张边界

### [S05-01] 状态: PASS

要求:
是否使用“第一回波反事实近似”而非“真实 LiDAR 完整模拟”。

发现:
协议、渲染代码和论文均把该模块限定为 OS1-128 第一回波反事实近似，并明确排除运动畸变、光束发散、有限 footprint、多回波、多径和隐藏表面重建；全仓库未见相反正式主张。

证据:
- protocol.json:75-117
- src/render.py:1445-1453
- AJAE.tex:173-205

判断:
证据直接覆盖该要求，或排除性全仓库搜索已证明冲突路径不在当前正式路线；可判 PASS。该结论只覆盖本条，不外推到整个模块或科学主张。

需要修改:
无需修改；保留当前实现和直接证据，并继续确保该局部结论不被外推为模块或科学主张已经成立。

### [S05-02] 状态: PASS

要求:
是否没有声称模拟完整 intra-scan motion distortion。

发现:
协议、渲染代码和论文均把该模块限定为 OS1-128 第一回波反事实近似，并明确排除运动畸变、光束发散、有限 footprint、多回波、多径和隐藏表面重建；全仓库未见相反正式主张。

证据:
- protocol.json:75-117
- src/render.py:1445-1453
- AJAE.tex:173-205

判断:
证据直接覆盖该要求，或排除性全仓库搜索已证明冲突路径不在当前正式路线；可判 PASS。该结论只覆盖本条，不外推到整个模块或科学主张。

需要修改:
无需修改；保留当前实现和直接证据，并继续确保该局部结论不被外推为模块或科学主张已经成立。

### [S05-03] 状态: PASS

要求:
是否没有声称模拟 beam divergence。

发现:
协议、渲染代码和论文均把该模块限定为 OS1-128 第一回波反事实近似，并明确排除运动畸变、光束发散、有限 footprint、多回波、多径和隐藏表面重建；全仓库未见相反正式主张。

证据:
- protocol.json:75-117
- src/render.py:1445-1453
- AJAE.tex:173-205

判断:
证据直接覆盖该要求，或排除性全仓库搜索已证明冲突路径不在当前正式路线；可判 PASS。该结论只覆盖本条，不外推到整个模块或科学主张。

需要修改:
无需修改；保留当前实现和直接证据，并继续确保该局部结论不被外推为模块或科学主张已经成立。

### [S05-04] 状态: PASS

要求:
是否没有声称模拟 finite footprint。

发现:
协议、渲染代码和论文均把该模块限定为 OS1-128 第一回波反事实近似，并明确排除运动畸变、光束发散、有限 footprint、多回波、多径和隐藏表面重建；全仓库未见相反正式主张。

证据:
- protocol.json:75-117
- src/render.py:1445-1453
- AJAE.tex:173-205

判断:
证据直接覆盖该要求，或排除性全仓库搜索已证明冲突路径不在当前正式路线；可判 PASS。该结论只覆盖本条，不外推到整个模块或科学主张。

需要修改:
无需修改；保留当前实现和直接证据，并继续确保该局部结论不被外推为模块或科学主张已经成立。

### [S05-05] 状态: PASS

要求:
是否没有声称模拟 multi-return。

发现:
协议、渲染代码和论文均把该模块限定为 OS1-128 第一回波反事实近似，并明确排除运动畸变、光束发散、有限 footprint、多回波、多径和隐藏表面重建；全仓库未见相反正式主张。

证据:
- protocol.json:75-117
- src/render.py:1445-1453
- AJAE.tex:173-205

判断:
证据直接覆盖该要求，或排除性全仓库搜索已证明冲突路径不在当前正式路线；可判 PASS。该结论只覆盖本条，不外推到整个模块或科学主张。

需要修改:
无需修改；保留当前实现和直接证据，并继续确保该局部结论不被外推为模块或科学主张已经成立。

### [S05-06] 状态: PASS

要求:
是否没有声称模拟 multipath。

发现:
协议、渲染代码和论文均把该模块限定为 OS1-128 第一回波反事实近似，并明确排除运动畸变、光束发散、有限 footprint、多回波、多径和隐藏表面重建；全仓库未见相反正式主张。

证据:
- protocol.json:75-117
- src/render.py:1445-1453
- AJAE.tex:173-205

判断:
证据直接覆盖该要求，或排除性全仓库搜索已证明冲突路径不在当前正式路线；可判 PASS。该结论只覆盖本条，不外推到整个模块或科学主张。

需要修改:
无需修改；保留当前实现和直接证据，并继续确保该局部结论不被外推为模块或科学主张已经成立。

### [S05-07] 状态: PASS

要求:
是否没有声称重建 hidden surfaces。

发现:
协议、渲染代码和论文均把该模块限定为 OS1-128 第一回波反事实近似，并明确排除运动畸变、光束发散、有限 footprint、多回波、多径和隐藏表面重建；全仓库未见相反正式主张。

证据:
- protocol.json:75-117
- src/render.py:1445-1453
- AJAE.tex:173-205

判断:
证据直接覆盖该要求，或排除性全仓库搜索已证明冲突路径不在当前正式路线；可判 PASS。该结论只覆盖本条，不外推到整个模块或科学主张。

需要修改:
无需修改；保留当前实现和直接证据，并继续确保该局部结论不被外推为模块或科学主张已经成立。

### [S05-08] 状态: PASS

要求:
如果论文/README 存在过度物理等价表述，记录为 protocol issue。

发现:
协议、渲染代码和论文均把该模块限定为 OS1-128 第一回波反事实近似，并明确排除运动畸变、光束发散、有限 footprint、多回波、多径和隐藏表面重建；全仓库未见相反正式主张。

证据:
- protocol.json:75-117
- src/render.py:1445-1453
- AJAE.tex:173-205

判断:
证据直接覆盖该要求，或排除性全仓库搜索已证明冲突路径不在当前正式路线；可判 PASS。该结论只覆盖本条，不外推到整个模块或科学主张。

需要修改:
无需修改；保留当前实现和直接证据，并继续确保该局部结论不被外推为模块或科学主张已经成立。

## C00：真实正常与渲染正常来源泄漏

### [C00-01] 状态: PARTIAL

要求:
是否构造了匹配后的 real-normal / rendered-normal 数据。

发现:
真实来源审计使用按帧分组的低容量逻辑回归和规定的七类输入，但匹配距离偏松；测试平衡准确率 `0.9398887`、AUROC `0.9786149`，已接近饱和，Gate1 判定标准却仍为 null。

证据:
- src/render.py:3845-4106
- dev.json:8978-10958
- 本次前置检查拒绝正式训练且未写 run state

判断:
只能证明静态实现、局部运行或缺少 生成链路 的产物，尚不能证明可追溯正式路径完整采用；按审计规则不得升级为 PASS。

需要修改:
转为 PASS 所需条件：基于科学容忍度而非当前分数冻结 Gate1 标准，并用严格条件匹配样本重审来源泄漏。

### [C00-02] 状态: PARTIAL

要求:
是否存在低容量 classifier。

发现:
真实来源审计使用按帧分组的低容量逻辑回归和规定的七类输入，但匹配距离偏松；测试平衡准确率 `0.9398887`、AUROC `0.9786149`，已接近饱和，Gate1 判定标准却仍为 null。

证据:
- src/render.py:3845-4106
- dev.json:8978-10958
- 本次前置检查拒绝正式训练且未写 run state

判断:
只能证明静态实现、局部运行或缺少 生成链路 的产物，尚不能证明可追溯正式路径完整采用；按审计规则不得升级为 PASS。

需要修改:
转为 PASS 所需条件：基于科学容忍度而非当前分数冻结 Gate1 标准，并用严格条件匹配样本重审来源泄漏。

### [C00-03] 状态: PARTIAL

要求:
输入是否仅限 $x,y,z$、intensity、beam、range、local density。

发现:
真实来源审计使用按帧分组的低容量逻辑回归和规定的七类输入，但匹配距离偏松；测试平衡准确率 `0.9398887`、AUROC `0.9786149`，已接近饱和，Gate1 判定标准却仍为 null。

证据:
- src/render.py:3845-4106
- dev.json:8978-10958
- 本次前置检查拒绝正式训练且未写 run state

判断:
只能证明静态实现、局部运行或缺少 生成链路 的产物，尚不能证明可追溯正式路径完整采用；按审计规则不得升级为 PASS。

需要修改:
转为 PASS 所需条件：基于科学容忍度而非当前分数冻结 Gate1 标准，并用严格条件匹配样本重审来源泄漏。

### [C00-04] 状态: PARTIAL

要求:
是否没有偷偷输入 semantic/category/source flag。

发现:
真实来源审计使用按帧分组的低容量逻辑回归和规定的七类输入，但匹配距离偏松；测试平衡准确率 `0.9398887`、AUROC `0.9786149`，已接近饱和，Gate1 判定标准却仍为 null。

证据:
- src/render.py:3845-4106
- dev.json:8978-10958
- 本次前置检查拒绝正式训练且未写 run state

判断:
只能证明静态实现、局部运行或缺少 生成链路 的产物，尚不能证明可追溯正式路径完整采用；按审计规则不得升级为 PASS。

需要修改:
转为 PASS 所需条件：基于科学容忍度而非当前分数冻结 Gate1 标准，并用严格条件匹配样本重审来源泄漏。

### [C00-05] 状态: PARTIAL

要求:
是否记录 train/dev split。

发现:
真实来源审计使用按帧分组的低容量逻辑回归和规定的七类输入，但匹配距离偏松；测试平衡准确率 `0.9398887`、AUROC `0.9786149`，已接近饱和，Gate1 判定标准却仍为 null。

证据:
- src/render.py:3845-4106
- dev.json:8978-10958
- 本次前置检查拒绝正式训练且未写 run state

判断:
只能证明静态实现、局部运行或缺少 生成链路 的产物，尚不能证明可追溯正式路径完整采用；按审计规则不得升级为 PASS。

需要修改:
转为 PASS 所需条件：基于科学容忍度而非当前分数冻结 Gate1 标准，并用严格条件匹配样本重审来源泄漏。

### [C00-06] 状态: PARTIAL

要求:
是否真实训练过，而不是只实现代码。

发现:
真实来源审计使用按帧分组的低容量逻辑回归和规定的七类输入，但匹配距离偏松；测试平衡准确率 `0.9398887`、AUROC `0.9786149`，已接近饱和，Gate1 判定标准却仍为 null。

证据:
- src/render.py:3845-4106
- dev.json:8978-10958
- 本次前置检查拒绝正式训练且未写 run state

判断:
只能证明静态实现、局部运行或缺少 生成链路 的产物，尚不能证明可追溯正式路径完整采用；按审计规则不得升级为 PASS。

需要修改:
转为 PASS 所需条件：基于科学容忍度而非当前分数冻结 Gate1 标准，并用严格条件匹配样本重审来源泄漏。

### [C00-07] 状态: PARTIAL

要求:
是否报告 AUC/AP/accuracy 等区分能力。

发现:
真实来源审计使用按帧分组的低容量逻辑回归和规定的七类输入，但匹配距离偏松；测试平衡准确率 `0.9398887`、AUROC `0.9786149`，已接近饱和，Gate1 判定标准却仍为 null。

证据:
- src/render.py:3845-4106
- dev.json:8978-10958
- 本次前置检查拒绝正式训练且未写 run state

判断:
只能证明静态实现、局部运行或缺少 生成链路 的产物，尚不能证明可追溯正式路径完整采用；按审计规则不得升级为 PASS。

需要修改:
转为 PASS 所需条件：基于科学容忍度而非当前分数冻结 Gate1 标准，并用严格条件匹配样本重审来源泄漏。

### [C00-08] 状态: FAIL

要求:
是否有预先定义或明确解释的“来源泄漏不可接受”判断。

发现:
`protocol.json` 的 Gate1 criteria 和 `dev.json.threshold_conclusion` 均为 null，没有预先冻结的可接受来源泄漏标准。

证据:
- src/render.py:3845-4106
- dev.json:8978-10958
- 本次前置检查拒绝正式训练且未写 run state

判断:
存在与要求直接冲突的实现、数据或真实运行事实；判定 FAIL。

需要修改:
在查看新一轮结果前，按科学可接受泄漏水平冻结明确阈值和失败动作。

### [C00-09] 状态: FAIL

要求:
若 classifier 接近饱和，Gate 1 必须 FAIL。

发现:
真实来源分类器平衡准确率 0.9398887、AUROC 0.9786149，接近饱和；无论匹配距离是否混杂根因，Gate1 都不能通过。

证据:
- src/render.py:3845-4106
- dev.json:8978-10958
- 本次前置检查拒绝正式训练且未写 run state

判断:
存在与要求直接冲突的实现、数据或真实运行事实；判定 FAIL。

需要修改:
不得训练 AJAE；先以严格条件匹配样本重做来源审计并消除可分来源。

### [C00-10] 状态: PASS

要求:
若 Gate 1 FAIL，正式 AJAE train 应 BLOCKED。

发现:
真实来源审计使用按帧分组的低容量逻辑回归和规定的七类输入，但匹配距离偏松；测试平衡准确率 `0.9398887`、AUROC `0.9786149`，已接近饱和，Gate1 判定标准却仍为 null。

证据:
- src/render.py:3845-4106
- dev.json:8978-10958
- 本次前置检查拒绝正式训练且未写 run state

判断:
证据直接覆盖该要求，或排除性全仓库搜索已证明冲突路径不在当前正式路线；可判 PASS。该结论只覆盖本条，不外推到整个模块或科学主张。

需要修改:
无需修改；保留当前实现和直接证据，并继续确保该局部结论不被外推为模块或科学主张已经成立。

### [C00-11] 状态: PASS

要求:
检查当前工作区在该失败后是否仍偷偷进行了正式训练。

发现:
真实来源审计使用按帧分组的低容量逻辑回归和规定的七类输入，但匹配距离偏松；测试平衡准确率 `0.9398887`、AUROC `0.9786149`，已接近饱和，Gate1 判定标准却仍为 null。

证据:
- src/render.py:3845-4106
- dev.json:8978-10958
- 本次前置检查拒绝正式训练且未写 run state

判断:
证据直接覆盖该要求，或排除性全仓库搜索已证明冲突路径不在当前正式路线；可判 PASS。该结论只覆盖本条，不外推到整个模块或科学主张。

需要修改:
无需修改；保留当前实现和直接证据，并继续确保该局部结论不被外推为模块或科学主张已经成立。

## C01：异常代理难度检查

### [C01-01] 状态: NOT RUN

要求:
是否训练 low-capacity single-frame normal-control vs proxy classifier。

发现:
代理难度分类函数存在，但没有在固定 201 合成世界真实训练、评价或保存结果；当前只可确认主训练没有越过 Gate1。

证据:
- src/render.py:4091-4106
- 全仓库无 201 proxy-difficulty 结果产物

判断:
本条要求的运行级审计、实验或正式产物没有发生；代码存在也不能替代执行，判定 NOT RUN。

需要修改:
转为 PASS 所需条件：在冻结的 201 合成世界运行低容量难度分类并保存量化结论。

### [C01-02] 状态: NOT RUN

要求:
是否在 201 synthetic worlds 评估。

发现:
代理难度分类函数存在，但没有在固定 201 合成世界真实训练、评价或保存结果；当前只可确认主训练没有越过 Gate1。

证据:
- src/render.py:4091-4106
- 全仓库无 201 proxy-difficulty 结果产物

判断:
本条要求的运行级审计、实验或正式产物没有发生；代码存在也不能替代执行，判定 NOT RUN。

需要修改:
转为 PASS 所需条件：在冻结的 201 合成世界运行低容量难度分类并保存量化结论。

### [C01-03] 状态: NOT RUN

要求:
是否报告结果。

发现:
代理难度分类函数存在，但没有在固定 201 合成世界真实训练、评价或保存结果；当前只可确认主训练没有越过 Gate1。

证据:
- src/render.py:4091-4106
- 全仓库无 201 proxy-difficulty 结果产物

判断:
本条要求的运行级审计、实验或正式产物没有发生；代码存在也不能替代执行，判定 NOT RUN。

需要修改:
转为 PASS 所需条件：在冻结的 201 合成世界运行低容量难度分类并保存量化结论。

### [C01-04] 状态: N/A

要求:
若接近饱和，是否被识别为 proxy 过易，而不是宣称模型优秀。

发现:
代理难度分类函数存在，但没有在固定 201 合成世界真实训练、评价或保存结果；当前只可确认主训练没有越过 Gate1。

证据:
- src/render.py:4091-4106
- 全仓库无 201 proxy-difficulty 结果产物

判断:
当前阶段尚未触发该条件，主线明确允许不执行；已说明适用边界，判定 N/A。

需要修改:
当前阶段条件尚未触发，无需修改；若未来触发，必须重新按本条要求审计。

### [C01-05] 状态: PARTIAL

要求:
是否有 hard proxy / near-normal-boundary proxy。

发现:
代理难度分类函数存在，但没有在固定 201 合成世界真实训练、评价或保存结果；当前只可确认主训练没有越过 Gate1。

证据:
- src/render.py:4091-4106
- 全仓库无 201 proxy-difficulty 结果产物

判断:
只能证明静态实现、局部运行或缺少 生成链路 的产物，尚不能证明可追溯正式路径完整采用；按审计规则不得升级为 PASS。

需要修改:
转为 PASS 所需条件：在冻结的 201 合成世界运行低容量难度分类并保存量化结论。

### [C01-06] 状态: PASS

要求:
是否没有因为“代理越容易”就继续主实验。

发现:
代理难度分类函数存在，但没有在固定 201 合成世界真实训练、评价或保存结果；当前只可确认主训练没有越过 Gate1。

证据:
- src/render.py:4091-4106
- 全仓库无 201 proxy-difficulty 结果产物

判断:
证据直接覆盖该要求，或排除性全仓库搜索已证明冲突路径不在当前正式路线；可判 PASS。该结论只覆盖本条，不外推到整个模块或科学主张。

需要修改:
无需修改；保留当前实现和直接证据，并继续确保该局部结论不被外推为模块或科学主张已经成立。

## C02：传感器分布审计

### [C02-01] 状态: PARTIAL

要求:
per-beam return rate。

发现:
传感器审计保存 per-beam、per-range、empty→valid、可见点变化和遮挡统计；强度只保存 beam×range 均值，且产物记录 anomaly-proxy returns 为 0，无法比较 control/proxy。

证据:
- src/render.py:4109-4329
- dev.json:5-101
- dev.json:6980-6982

判断:
只能证明静态实现、局部运行或缺少 生成链路 的产物，尚不能证明可追溯正式路径完整采用；按审计规则不得升级为 PASS。

需要修改:
转为 PASS 所需条件：补全强度分布且让 normal-control/proxy 都有有效回波，再按冻结标准判定。

### [C02-02] 状态: PARTIAL

要求:
per-range-bin return rate。

发现:
传感器审计保存 per-beam、per-range、empty→valid、可见点变化和遮挡统计；强度只保存 beam×range 均值，且产物记录 anomaly-proxy returns 为 0，无法比较 control/proxy。

证据:
- src/render.py:4109-4329
- dev.json:5-101
- dev.json:6980-6982

判断:
只能证明静态实现、局部运行或缺少 生成链路 的产物，尚不能证明可追溯正式路径完整采用；按审计规则不得升级为 PASS。

需要修改:
转为 PASS 所需条件：补全强度分布且让 normal-control/proxy 都有有效回波，再按冻结标准判定。

### [C02-03] 状态: FAIL

要求:
beam × range intensity distribution。

发现:
现有产物只有 beam×range intensity mean，没有完整分布、分位数或不确定性对照。

证据:
- src/render.py:4109-4329
- dev.json:5-101
- dev.json:6980-6982

判断:
存在与要求直接冲突的实现、数据或真实运行事实；判定 FAIL。

需要修改:
保存 beam×range 条件下的强度分位数/分布，而非只有均值。

### [C02-04] 状态: PARTIAL

要求:
empty-ray → valid-ray ratio。

发现:
传感器审计保存 per-beam、per-range、empty→valid、可见点变化和遮挡统计；强度只保存 beam×range 均值，且产物记录 anomaly-proxy returns 为 0，无法比较 control/proxy。

证据:
- src/render.py:4109-4329
- dev.json:5-101
- dev.json:6980-6982

判断:
只能证明静态实现、局部运行或缺少 生成链路 的产物，尚不能证明可追溯正式路径完整采用；按审计规则不得升级为 PASS。

需要修改:
转为 PASS 所需条件：补全强度分布且让 normal-control/proxy 都有有效回波，再按冻结标准判定。

### [C02-05] 状态: PARTIAL

要求:
per-entity visible point count。

发现:
传感器审计保存 per-beam、per-range、empty→valid、可见点变化和遮挡统计；强度只保存 beam×range 均值，且产物记录 anomaly-proxy returns 为 0，无法比较 control/proxy。

证据:
- src/render.py:4109-4329
- dev.json:5-101
- dev.json:6980-6982

判断:
只能证明静态实现、局部运行或缺少 生成链路 的产物，尚不能证明可追溯正式路径完整采用；按审计规则不得升级为 PASS。

需要修改:
转为 PASS 所需条件：补全强度分布且让 normal-control/proxy 都有有效回波，再按冻结标准判定。

### [C02-06] 状态: PARTIAL

要求:
consecutive-frame visible-point variation。

发现:
传感器审计保存 per-beam、per-range、empty→valid、可见点变化和遮挡统计；强度只保存 beam×range 均值，且产物记录 anomaly-proxy returns 为 0，无法比较 control/proxy。

证据:
- src/render.py:4109-4329
- dev.json:5-101
- dev.json:6980-6982

判断:
只能证明静态实现、局部运行或缺少 生成链路 的产物，尚不能证明可追溯正式路径完整采用；按审计规则不得升级为 PASS。

需要修改:
转为 PASS 所需条件：补全强度分布且让 normal-control/proxy 都有有效回波，再按冻结标准判定。

### [C02-07] 状态: PARTIAL

要求:
occlusion ratio distribution。

发现:
传感器审计保存 per-beam、per-range、empty→valid、可见点变化和遮挡统计；强度只保存 beam×range 均值，且产物记录 anomaly-proxy returns 为 0，无法比较 control/proxy。

证据:
- src/render.py:4109-4329
- dev.json:5-101
- dev.json:6980-6982

判断:
只能证明静态实现、局部运行或缺少 生成链路 的产物，尚不能证明可追溯正式路径完整采用；按审计规则不得升级为 PASS。

需要修改:
转为 PASS 所需条件：补全强度分布且让 normal-control/proxy 都有有效回波，再按冻结标准判定。

### [C02-08] 状态: PARTIAL

要求:
real vs rendered normal 对照。

发现:
传感器审计保存 per-beam、per-range、empty→valid、可见点变化和遮挡统计；强度只保存 beam×range 均值，且产物记录 anomaly-proxy returns 为 0，无法比较 control/proxy。

证据:
- src/render.py:4109-4329
- dev.json:5-101
- dev.json:6980-6982

判断:
只能证明静态实现、局部运行或缺少 生成链路 的产物，尚不能证明可追溯正式路径完整采用；按审计规则不得升级为 PASS。

需要修改:
转为 PASS 所需条件：补全强度分布且让 normal-control/proxy 都有有效回波，再按冻结标准判定。

### [C02-09] 状态: FAIL

要求:
normal-control vs anomaly-proxy 分布。

发现:
sensor audit 明确记录 anomaly_proxy_returns=0，完全无法比较 normal-control 与 anomaly-proxy 传感器分布。

证据:
- src/render.py:4109-4329
- dev.json:5-101
- dev.json:6980-6982

判断:
存在与要求直接冲突的实现、数据或真实运行事实；判定 FAIL。

需要修改:
让同一冻结世界审计同时包含正常控制和异常代理有效回波。

### [C02-10] 状态: PARTIAL

要求:
audit artifact 是否保存。

发现:
传感器审计保存 per-beam、per-range、empty→valid、可见点变化和遮挡统计；强度只保存 beam×range 均值，且产物记录 anomaly-proxy returns 为 0，无法比较 control/proxy。

证据:
- src/render.py:4109-4329
- dev.json:5-101
- dev.json:6980-6982

判断:
只能证明静态实现、局部运行或缺少 生成链路 的产物，尚不能证明可追溯正式路径完整采用；按审计规则不得升级为 PASS。

需要修改:
转为 PASS 所需条件：补全强度分布且让 normal-control/proxy 都有有效回波，再按冻结标准判定。

## T00：128 维高层特征

### [T00-01] 状态: FAIL

要求:
确认实际调用官方 STU backbone。

发现:
本次在真实 train/206 frame 0 调用 `FrozenSTUPointEncoder.from_protocol`，构造阶段因 `ModuleNotFoundError: No module named hydra` 失败；当前工作区无法实际调用官方 backbone。

证据:
- src/model.py:372-563
- test_ajae.py:346-399
- 环境检查：MinkowskiEngine=True；hydra/omegaconf/pytorch3d=False

判断:
存在与要求直接冲突的实现、数据或真实运行事实；判定 FAIL。

需要修改:
补齐官方依赖并在受散列约束的真实 STU checkpoint 上成功运行真实帧；本次没有安装或改动环境。

### [T00-02] 状态: PARTIAL

要求:
确认使用 `all_features[-1]`。

发现:
官方接口代码注册 `point_features_head` hook 并要求 128 维，也使用官方稀疏逆映射；真实 train/206 帧的构造在导入 Hydra 时失败，现有模型测试只输入随机 128D 张量。

证据:
- src/model.py:372-563
- test_ajae.py:346-399
- 环境检查：MinkowskiEngine=True；hydra/omegaconf/pytorch3d=False

判断:
只能证明静态实现、局部运行或缺少 生成链路 的产物，尚不能证明可追溯正式路径完整采用；按审计规则不得升级为 PASS。

需要修改:
转为 PASS 所需条件：补齐官方依赖，真实加载受散列约束权重，并保存官方 128D feature-interface 测试。

### [T00-03] 状态: PARTIAL

要求:
确认调用官方 `point_features_head`。

发现:
官方接口代码注册 `point_features_head` hook 并要求 128 维，也使用官方稀疏逆映射；真实 train/206 帧的构造在导入 Hydra 时失败，现有模型测试只输入随机 128D 张量。

证据:
- src/model.py:372-563
- test_ajae.py:346-399
- 环境检查：MinkowskiEngine=True；hydra/omegaconf/pytorch3d=False

判断:
只能证明静态实现、局部运行或缺少 生成链路 的产物，尚不能证明可追溯正式路径完整采用；按审计规则不得升级为 PASS。

需要修改:
转为 PASS 所需条件：补齐官方依赖，真实加载受散列约束权重，并保存官方 128D feature-interface 测试。

### [T00-04] 状态: PARTIAL

要求:
输出维度必须验证为 128。

发现:
官方接口代码注册 `point_features_head` hook 并要求 128 维，也使用官方稀疏逆映射；真实 train/206 帧的构造在导入 Hydra 时失败，现有模型测试只输入随机 128D 张量。

证据:
- src/model.py:372-563
- test_ajae.py:346-399
- 环境检查：MinkowskiEngine=True；hydra/omegaconf/pytorch3d=False

判断:
只能证明静态实现、局部运行或缺少 生成链路 的产物，尚不能证明可追溯正式路径完整采用；按审计规则不得升级为 PASS。

需要修改:
转为 PASS 所需条件：补齐官方依赖，真实加载受散列约束权重，并保存官方 128D feature-interface 测试。

### [T00-05] 状态: PARTIAL

要求:
不得把其他层误认为主 128D feature。

发现:
官方接口代码注册 `point_features_head` hook 并要求 128 维，也使用官方稀疏逆映射；真实 train/206 帧的构造在导入 Hydra 时失败，现有模型测试只输入随机 128D 张量。

证据:
- src/model.py:372-563
- test_ajae.py:346-399
- 环境检查：MinkowskiEngine=True；hydra/omegaconf/pytorch3d=False

判断:
只能证明静态实现、局部运行或缺少 生成链路 的产物，尚不能证明可追溯正式路径完整采用；按审计规则不得升级为 PASS。

需要修改:
转为 PASS 所需条件：补齐官方依赖，真实加载受散列约束权重，并保存官方 128D feature-interface 测试。

### [T00-06] 状态: FAIL

要求:
是否保存 feature interface unit test。

发现:
没有保存的 feature-interface 单元测试真实实例化官方 STU 并验证 `all_features[-1] → point_features_head → 128D`；随机 128D tensor 夹具不满足要求。

证据:
- src/model.py:372-563
- test_ajae.py:346-399
- 环境检查：MinkowskiEngine=True；hydra/omegaconf/pytorch3d=False

判断:
存在与要求直接冲突的实现、数据或真实运行事实；判定 FAIL。

需要修改:
新增的测试必须真实实例化官方 STU 接口，随机 128D tensor 不能替代。

### [T00-07] 状态: PARTIAL

要求:
是否使用官方或严格等价 sparse voxel inverse map。

发现:
官方接口代码注册 `point_features_head` hook 并要求 128 维，也使用官方稀疏逆映射；真实 train/206 帧的构造在导入 Hydra 时失败，现有模型测试只输入随机 128D 张量。

证据:
- src/model.py:372-563
- test_ajae.py:346-399
- 环境检查：MinkowskiEngine=True；hydra/omegaconf/pytorch3d=False

判断:
只能证明静态实现、局部运行或缺少 生成链路 的产物，尚不能证明可追溯正式路径完整采用；按审计规则不得升级为 PASS。

需要修改:
转为 PASS 所需条件：补齐官方依赖，真实加载受散列约束权重，并保存官方 128D feature-interface 测试。

## T01：体素到原始点的逆映射

### [T01-01] 状态: PARTIAL

要求:
是否存在 $\pi_f(p)$ 等价映射。

发现:
逆映射允许多个原始点共享体素特征，同时分别保留坐标、强度、ray/slot 和最终分数接口；尚无真实官方 STU 帧级产物。

证据:
- src/model.py:310-364
- src/model.py:475-563
- src/scene.py:375-469

判断:
只能证明静态实现、局部运行或缺少 生成链路 的产物，尚不能证明可追溯正式路径完整采用；按审计规则不得升级为 PASS。

需要修改:
转为 PASS 所需条件：在真实官方 STU 帧上逐点核对 inverse map、重复体素和原始身份。

### [T01-02] 状态: PARTIAL

要求:
每个原始有效回波点是否都能恢复对应 STU voxel feature。

发现:
逆映射允许多个原始点共享体素特征，同时分别保留坐标、强度、ray/slot 和最终分数接口；尚无真实官方 STU 帧级产物。

证据:
- src/model.py:310-364
- src/model.py:475-563
- src/scene.py:375-469

判断:
只能证明静态实现、局部运行或缺少 生成链路 的产物，尚不能证明可追溯正式路径完整采用；按审计规则不得升级为 PASS。

需要修改:
转为 PASS 所需条件：在真实官方 STU 帧上逐点核对 inverse map、重复体素和原始身份。

### [T01-03] 状态: PARTIAL

要求:
多 raw points 共享 voxel feature 是否被正确允许。

发现:
逆映射允许多个原始点共享体素特征，同时分别保留坐标、强度、ray/slot 和最终分数接口；尚无真实官方 STU 帧级产物。

证据:
- src/model.py:310-364
- src/model.py:475-563
- src/scene.py:375-469

判断:
只能证明静态实现、局部运行或缺少 生成链路 的产物，尚不能证明可追溯正式路径完整采用；按审计规则不得升级为 PASS。

需要修改:
转为 PASS 所需条件：在真实官方 STU 帧上逐点核对 inverse map、重复体素和原始身份。

### [T01-04] 状态: PARTIAL

要求:
raw coordinates 是否仍分别保留。

发现:
逆映射允许多个原始点共享体素特征，同时分别保留坐标、强度、ray/slot 和最终分数接口；尚无真实官方 STU 帧级产物。

证据:
- src/model.py:310-364
- src/model.py:475-563
- src/scene.py:375-469

判断:
只能证明静态实现、局部运行或缺少 生成链路 的产物，尚不能证明可追溯正式路径完整采用；按审计规则不得升级为 PASS。

需要修改:
转为 PASS 所需条件：在真实官方 STU 帧上逐点核对 inverse map、重复体素和原始身份。

### [T01-05] 状态: PARTIAL

要求:
raw intensity 是否仍分别保留。

发现:
逆映射允许多个原始点共享体素特征，同时分别保留坐标、强度、ray/slot 和最终分数接口；尚无真实官方 STU 帧级产物。

证据:
- src/model.py:310-364
- src/model.py:475-563
- src/scene.py:375-469

判断:
只能证明静态实现、局部运行或缺少 生成链路 的产物，尚不能证明可追溯正式路径完整采用；按审计规则不得升级为 PASS。

需要修改:
转为 PASS 所需条件：在真实官方 STU 帧上逐点核对 inverse map、重复体素和原始身份。

### [T01-06] 状态: PARTIAL

要求:
canonical ray identity 是否仍分别保留。

发现:
逆映射允许多个原始点共享体素特征，同时分别保留坐标、强度、ray/slot 和最终分数接口；尚无真实官方 STU 帧级产物。

证据:
- src/model.py:310-364
- src/model.py:475-563
- src/scene.py:375-469

判断:
只能证明静态实现、局部运行或缺少 生成链路 的产物，尚不能证明可追溯正式路径完整采用；按审计规则不得升级为 PASS。

需要修改:
转为 PASS 所需条件：在真实官方 STU 帧上逐点核对 inverse map、重复体素和原始身份。

### [T01-07] 状态: PARTIAL

要求:
raw point final score 是否仍分别输出。

发现:
逆映射允许多个原始点共享体素特征，同时分别保留坐标、强度、ray/slot 和最终分数接口；尚无真实官方 STU 帧级产物。

证据:
- src/model.py:310-364
- src/model.py:475-563
- src/scene.py:375-469

判断:
只能证明静态实现、局部运行或缺少 生成链路 的产物，尚不能证明可追溯正式路径完整采用；按审计规则不得升级为 PASS。

需要修改:
转为 PASS 所需条件：在真实官方 STU 帧上逐点核对 inverse map、重复体素和原始身份。

## T02：官方查询分配语义证据

### [T02-01] 状态: PARTIAL

要求:
是否读取 query class logits $Q\times20$。

发现:
`assigned_stu_evidence()` 实现 20 类 query softmax、mask sigmoid、19 类 normal max、确定性 query 分配、19D 证据、可靠性和 no-object 概率；仅有小张量测试，没有真实 STU 输出运行。

证据:
- src/model.py:260-307
- test_ajae.py:320-334
- src/evaluate.py:1065-1111

判断:
只能证明静态实现、局部运行或缺少 生成链路 的产物，尚不能证明可追溯正式路径完整采用；按审计规则不得升级为 PASS。

需要修改:
转为 PASS 所需条件：用真实官方 query/mask 输出核对分配、19D 证据和 raw-point 恢复。

### [T02-02] 状态: PARTIAL

要求:
是否 softmax query class logits。

发现:
`assigned_stu_evidence()` 实现 20 类 query softmax、mask sigmoid、19 类 normal max、确定性 query 分配、19D 证据、可靠性和 no-object 概率；仅有小张量测试，没有真实 STU 输出运行。

证据:
- src/model.py:260-307
- test_ajae.py:320-334
- src/evaluate.py:1065-1111

判断:
只能证明静态实现、局部运行或缺少 生成链路 的产物，尚不能证明可追溯正式路径完整采用；按审计规则不得升级为 PASS。

需要修改:
转为 PASS 所需条件：用真实官方 query/mask 输出核对分配、19D 证据和 raw-point 恢复。

### [T02-03] 状态: PARTIAL

要求:
是否读取 mask logits $N_v\times Q$。

发现:
`assigned_stu_evidence()` 实现 20 类 query softmax、mask sigmoid、19 类 normal max、确定性 query 分配、19D 证据、可靠性和 no-object 概率；仅有小张量测试，没有真实 STU 输出运行。

证据:
- src/model.py:260-307
- test_ajae.py:320-334
- src/evaluate.py:1065-1111

判断:
只能证明静态实现、局部运行或缺少 生成链路 的产物，尚不能证明可追溯正式路径完整采用；按审计规则不得升级为 PASS。

需要修改:
转为 PASS 所需条件：用真实官方 query/mask 输出核对分配、19D 证据和 raw-point 恢复。

### [T02-04] 状态: PARTIAL

要求:
是否 sigmoid mask logits。

发现:
`assigned_stu_evidence()` 实现 20 类 query softmax、mask sigmoid、19 类 normal max、确定性 query 分配、19D 证据、可靠性和 no-object 概率；仅有小张量测试，没有真实 STU 输出运行。

证据:
- src/model.py:260-307
- test_ajae.py:320-334
- src/evaluate.py:1065-1111

判断:
只能证明静态实现、局部运行或缺少 生成链路 的产物，尚不能证明可追溯正式路径完整采用；按审计规则不得升级为 PASS。

需要修改:
转为 PASS 所需条件：用真实官方 query/mask 输出核对分配、19D 证据和 raw-point 恢复。

### [T02-05] 状态: PARTIAL

要求:
是否计算 query assignment confidence。

发现:
`assigned_stu_evidence()` 实现 20 类 query softmax、mask sigmoid、19 类 normal max、确定性 query 分配、19D 证据、可靠性和 no-object 概率；仅有小张量测试，没有真实 STU 输出运行。

证据:
- src/model.py:260-307
- test_ajae.py:320-334
- src/evaluate.py:1065-1111

判断:
只能证明静态实现、局部运行或缺少 生成链路 的产物，尚不能证明可追溯正式路径完整采用；按审计规则不得升级为 PASS。

需要修改:
转为 PASS 所需条件：用真实官方 query/mask 输出核对分配、19D 证据和 raw-point 恢复。

### [T02-06] 状态: PARTIAL

要求:
是否只在 19 个 normal semantic classes 中计算 normal max。

发现:
`assigned_stu_evidence()` 实现 20 类 query softmax、mask sigmoid、19 类 normal max、确定性 query 分配、19D 证据、可靠性和 no-object 概率；仅有小张量测试，没有真实 STU 输出运行。

证据:
- src/model.py:260-307
- test_ajae.py:320-334
- src/evaluate.py:1065-1111

判断:
只能证明静态实现、局部运行或缺少 生成链路 的产物，尚不能证明可追溯正式路径完整采用；按审计规则不得升级为 PASS。

需要修改:
转为 PASS 所需条件：用真实官方 query/mask 输出核对分配、19D 证据和 raw-point 恢复。

### [T02-07] 状态: PARTIAL

要求:
是否选择最强 query $q^*(v)$。

发现:
`assigned_stu_evidence()` 实现 20 类 query softmax、mask sigmoid、19 类 normal max、确定性 query 分配、19D 证据、可靠性和 no-object 概率；仅有小张量测试，没有真实 STU 输出运行。

证据:
- src/model.py:260-307
- test_ajae.py:320-334
- src/evaluate.py:1065-1111

判断:
只能证明静态实现、局部运行或缺少 生成链路 的产物，尚不能证明可追溯正式路径完整采用；按审计规则不得升级为 PASS。

需要修改:
转为 PASS 所需条件：用真实官方 query/mask 输出核对分配、19D 证据和 raw-point 恢复。

### [T02-08] 状态: PARTIAL

要求:
tie 是否确定性选择最小 query ID 或等价固定规则。

发现:
`assigned_stu_evidence()` 实现 20 类 query softmax、mask sigmoid、19 类 normal max、确定性 query 分配、19D 证据、可靠性和 no-object 概率；仅有小张量测试，没有真实 STU 输出运行。

证据:
- src/model.py:260-307
- test_ajae.py:320-334
- src/evaluate.py:1065-1111

判断:
只能证明静态实现、局部运行或缺少 生成链路 的产物，尚不能证明可追溯正式路径完整采用；按审计规则不得升级为 PASS。

需要修改:
转为 PASS 所需条件：用真实官方 query/mask 输出核对分配、19D 证据和 raw-point 恢复。

### [T02-09] 状态: PARTIAL

要求:
normal semantic evidence 是否为 19D。

发现:
`assigned_stu_evidence()` 实现 20 类 query softmax、mask sigmoid、19 类 normal max、确定性 query 分配、19D 证据、可靠性和 no-object 概率；仅有小张量测试，没有真实 STU 输出运行。

证据:
- src/model.py:260-307
- test_ajae.py:320-334
- src/evaluate.py:1065-1111

判断:
只能证明静态实现、局部运行或缺少 生成链路 的产物，尚不能证明可追溯正式路径完整采用；按审计规则不得升级为 PASS。

需要修改:
转为 PASS 所需条件：用真实官方 query/mask 输出核对分配、19D 证据和 raw-point 恢复。

### [T02-10] 状态: PARTIAL

要求:
是否计算 assignment reliability。

发现:
`assigned_stu_evidence()` 实现 20 类 query softmax、mask sigmoid、19 类 normal max、确定性 query 分配、19D 证据、可靠性和 no-object 概率；仅有小张量测试，没有真实 STU 输出运行。

证据:
- src/model.py:260-307
- test_ajae.py:320-334
- src/evaluate.py:1065-1111

判断:
只能证明静态实现、局部运行或缺少 生成链路 的产物，尚不能证明可追溯正式路径完整采用；按审计规则不得升级为 PASS。

需要修改:
转为 PASS 所需条件：用真实官方 query/mask 输出核对分配、19D 证据和 raw-point 恢复。

### [T02-11] 状态: PARTIAL

要求:
是否保留 no-object probability。

发现:
`assigned_stu_evidence()` 实现 20 类 query softmax、mask sigmoid、19 类 normal max、确定性 query 分配、19D 证据、可靠性和 no-object 概率；仅有小张量测试，没有真实 STU 输出运行。

证据:
- src/model.py:260-307
- test_ajae.py:320-334
- src/evaluate.py:1065-1111

判断:
只能证明静态实现、局部运行或缺少 生成链路 的产物，尚不能证明可追溯正式路径完整采用；按审计规则不得升级为 PASS。

需要修改:
转为 PASS 所需条件：用真实官方 query/mask 输出核对分配、19D 证据和 raw-point 恢复。

### [T02-12] 状态: PARTIAL

要求:
是否 inverse-map 到 raw point。

发现:
`assigned_stu_evidence()` 实现 20 类 query softmax、mask sigmoid、19 类 normal max、确定性 query 分配、19D 证据、可靠性和 no-object 概率；仅有小张量测试，没有真实 STU 输出运行。

证据:
- src/model.py:260-307
- test_ajae.py:320-334
- src/evaluate.py:1065-1111

判断:
只能证明静态实现、局部运行或缺少 生成链路 的产物，尚不能证明可追溯正式路径完整采用；按审计规则不得升级为 PASS。

需要修改:
转为 PASS 所需条件：用真实官方 query/mask 输出核对分配、19D 证据和 raw-point 恢复。

### [T02-13] 状态: PARTIAL

要求:
是否完全没有“所有 query 未归一化求和”旧实现进入正式路径。

发现:
`assigned_stu_evidence()` 实现 20 类 query softmax、mask sigmoid、19 类 normal max、确定性 query 分配、19D 证据、可靠性和 no-object 概率；仅有小张量测试，没有真实 STU 输出运行。

证据:
- src/model.py:260-307
- test_ajae.py:320-334
- src/evaluate.py:1065-1111

判断:
只能证明静态实现、局部运行或缺少 生成链路 的产物，尚不能证明可追溯正式路径完整采用；按审计规则不得升级为 PASS。

需要修改:
转为 PASS 所需条件：用真实官方 query/mask 输出核对分配、19D 证据和 raw-point 恢复。

### [T02-14] 状态: PASS

要求:
搜索 repository 中旧 semantic evidence 实现，确认是否仍可能被配置启用。

发现:
`assigned_stu_evidence()` 实现 20 类 query softmax、mask sigmoid、19 类 normal max、确定性 query 分配、19D 证据、可靠性和 no-object 概率；仅有小张量测试，没有真实 STU 输出运行。

证据:
- src/model.py:260-307
- test_ajae.py:320-334
- src/evaluate.py:1065-1111

判断:
证据直接覆盖该要求，或排除性全仓库搜索已证明冲突路径不在当前正式路线；可判 PASS。该结论只覆盖本条，不外推到整个模块或科学主张。

需要修改:
无需修改；保留当前实现和直接证据，并继续确保该局部结论不被外推为模块或科学主张已经成立。

## T03：AJAE 输入接口

### [T03-01] 状态: PARTIAL

要求:
128D STU feature。

发现:
输入投影维度为 150（128+19+1+1+1），再加入中心坐标位置编码与 q 编码；没有 query token、entropy、energy 或 MSP 输入。真实五帧 STU 张量未进入该路径。

证据:
- src/model.py:631-679
- src/model.py:1114-1275
- protocol.json:126-138

判断:
只能证明静态实现、局部运行或缺少 生成链路 的产物，尚不能证明可追溯正式路径完整采用；按审计规则不得升级为 PASS。

需要修改:
转为 PASS 所需条件：保存真实五帧输入 tensor schema、维度与模型首层接收记录。

### [T03-02] 状态: PARTIAL

要求:
19D normal semantic evidence。

发现:
输入投影维度为 150（128+19+1+1+1），再加入中心坐标位置编码与 q 编码；没有 query token、entropy、energy 或 MSP 输入。真实五帧 STU 张量未进入该路径。

证据:
- src/model.py:631-679
- src/model.py:1114-1275
- protocol.json:126-138

判断:
只能证明静态实现、局部运行或缺少 生成链路 的产物，尚不能证明可追溯正式路径完整采用；按审计规则不得升级为 PASS。

需要修改:
转为 PASS 所需条件：保存真实五帧输入 tensor schema、维度与模型首层接收记录。

### [T03-03] 状态: PARTIAL

要求:
assignment reliability。

发现:
输入投影维度为 150（128+19+1+1+1），再加入中心坐标位置编码与 q 编码；没有 query token、entropy、energy 或 MSP 输入。真实五帧 STU 张量未进入该路径。

证据:
- src/model.py:631-679
- src/model.py:1114-1275
- protocol.json:126-138

判断:
只能证明静态实现、局部运行或缺少 生成链路 的产物，尚不能证明可追溯正式路径完整采用；按审计规则不得升级为 PASS。

需要修改:
转为 PASS 所需条件：保存真实五帧输入 tensor schema、维度与模型首层接收记录。

### [T03-04] 状态: PARTIAL

要求:
no-object reliability。

发现:
输入投影维度为 150（128+19+1+1+1），再加入中心坐标位置编码与 q 编码；没有 query token、entropy、energy 或 MSP 输入。真实五帧 STU 张量未进入该路径。

证据:
- src/model.py:631-679
- src/model.py:1114-1275
- protocol.json:126-138

判断:
只能证明静态实现、局部运行或缺少 生成链路 的产物，尚不能证明可追溯正式路径完整采用；按审计规则不得升级为 PASS。

需要修改:
转为 PASS 所需条件：保存真实五帧输入 tensor schema、维度与模型首层接收记录。

### [T03-05] 状态: PARTIAL

要求:
intensity。

发现:
输入投影维度为 150（128+19+1+1+1），再加入中心坐标位置编码与 q 编码；没有 query token、entropy、energy 或 MSP 输入。真实五帧 STU 张量未进入该路径。

证据:
- src/model.py:631-679
- src/model.py:1114-1275
- protocol.json:126-138

判断:
只能证明静态实现、局部运行或缺少 生成链路 的产物，尚不能证明可追溯正式路径完整采用；按审计规则不得升级为 PASS。

需要修改:
转为 PASS 所需条件：保存真实五帧输入 tensor schema、维度与模型首层接收记录。

### [T03-06] 状态: PARTIAL

要求:
center-time coordinate。

发现:
输入投影维度为 150（128+19+1+1+1），再加入中心坐标位置编码与 q 编码；没有 query token、entropy、energy 或 MSP 输入。真实五帧 STU 张量未进入该路径。

证据:
- src/model.py:631-679
- src/model.py:1114-1275
- protocol.json:126-138

判断:
只能证明静态实现、局部运行或缺少 生成链路 的产物，尚不能证明可追溯正式路径完整采用；按审计规则不得升级为 PASS。

需要修改:
转为 PASS 所需条件：保存真实五帧输入 tensor schema、维度与模型首层接收记录。

### [T03-07] 状态: PARTIAL

要求:
relative time $q$。

发现:
输入投影维度为 150（128+19+1+1+1），再加入中心坐标位置编码与 q 编码；没有 query token、entropy、energy 或 MSP 输入。真实五帧 STU 张量未进入该路径。

证据:
- src/model.py:631-679
- src/model.py:1114-1275
- protocol.json:126-138

判断:
只能证明静态实现、局部运行或缺少 生成链路 的产物，尚不能证明可追溯正式路径完整采用；按审计规则不得升级为 PASS。

需要修改:
转为 PASS 所需条件：保存真实五帧输入 tensor schema、维度与模型首层接收记录。

### [T03-08] 状态: PARTIAL

要求:
unified input projection。

发现:
输入投影维度为 150（128+19+1+1+1），再加入中心坐标位置编码与 q 编码；没有 query token、entropy、energy 或 MSP 输入。真实五帧 STU 张量未进入该路径。

证据:
- src/model.py:631-679
- src/model.py:1114-1275
- protocol.json:126-138

判断:
只能证明静态实现、局部运行或缺少 生成链路 的产物，尚不能证明可追溯正式路径完整采用；按审计规则不得升级为 PASS。

需要修改:
转为 PASS 所需条件：保存真实五帧输入 tensor schema、维度与模型首层接收记录。

### [T03-09] 状态: PARTIAL

要求:
spatial positional encoding。

发现:
输入投影维度为 150（128+19+1+1+1），再加入中心坐标位置编码与 q 编码；没有 query token、entropy、energy 或 MSP 输入。真实五帧 STU 张量未进入该路径。

证据:
- src/model.py:631-679
- src/model.py:1114-1275
- protocol.json:126-138

判断:
只能证明静态实现、局部运行或缺少 生成链路 的产物，尚不能证明可追溯正式路径完整采用；按审计规则不得升级为 PASS。

需要修改:
转为 PASS 所需条件：保存真实五帧输入 tensor schema、维度与模型首层接收记录。

### [T03-10] 状态: PARTIAL

要求:
temporal encoding。

发现:
输入投影维度为 150（128+19+1+1+1），再加入中心坐标位置编码与 q 编码；没有 query token、entropy、energy 或 MSP 输入。真实五帧 STU 张量未进入该路径。

证据:
- src/model.py:631-679
- src/model.py:1114-1275
- protocol.json:126-138

判断:
只能证明静态实现、局部运行或缺少 生成链路 的产物，尚不能证明可追溯正式路径完整采用；按审计规则不得升级为 PASS。

需要修改:
转为 PASS 所需条件：保存真实五帧输入 tensor schema、维度与模型首层接收记录。

### [T03-11] 状态: PASS

要求:
第一版是否没有直接输入 query tokens。

发现:
输入投影维度为 150（128+19+1+1+1），再加入中心坐标位置编码与 q 编码；没有 query token、entropy、energy 或 MSP 输入。真实五帧 STU 张量未进入该路径。

证据:
- src/model.py:631-679
- src/model.py:1114-1275
- protocol.json:126-138

判断:
证据直接覆盖该要求，或排除性全仓库搜索已证明冲突路径不在当前正式路线；可判 PASS。该结论只覆盖本条，不外推到整个模块或科学主张。

需要修改:
无需修改；保留当前实现和直接证据，并继续确保该局部结论不被外推为模块或科学主张已经成立。

### [T03-12] 状态: PASS

要求:
第一版是否没有手工加入 entropy。

发现:
输入投影维度为 150（128+19+1+1+1），再加入中心坐标位置编码与 q 编码；没有 query token、entropy、energy 或 MSP 输入。真实五帧 STU 张量未进入该路径。

证据:
- src/model.py:631-679
- src/model.py:1114-1275
- protocol.json:126-138

判断:
证据直接覆盖该要求，或排除性全仓库搜索已证明冲突路径不在当前正式路线；可判 PASS。该结论只覆盖本条，不外推到整个模块或科学主张。

需要修改:
无需修改；保留当前实现和直接证据，并继续确保该局部结论不被外推为模块或科学主张已经成立。

### [T03-13] 状态: PASS

要求:
第一版是否没有手工加入 energy。

发现:
输入投影维度为 150（128+19+1+1+1），再加入中心坐标位置编码与 q 编码；没有 query token、entropy、energy 或 MSP 输入。真实五帧 STU 张量未进入该路径。

证据:
- src/model.py:631-679
- src/model.py:1114-1275
- protocol.json:126-138

判断:
证据直接覆盖该要求，或排除性全仓库搜索已证明冲突路径不在当前正式路线；可判 PASS。该结论只覆盖本条，不外推到整个模块或科学主张。

需要修改:
无需修改；保留当前实现和直接证据，并继续确保该局部结论不被外推为模块或科学主张已经成立。

### [T03-14] 状态: PASS

要求:
第一版是否没有手工加入 MSP。

发现:
输入投影维度为 150（128+19+1+1+1），再加入中心坐标位置编码与 q 编码；没有 query token、entropy、energy 或 MSP 输入。真实五帧 STU 张量未进入该路径。

证据:
- src/model.py:631-679
- src/model.py:1114-1275
- protocol.json:126-138

判断:
证据直接覆盖该要求，或排除性全仓库搜索已证明冲突路径不在当前正式路线；可判 PASS。该结论只覆盖本条，不外推到整个模块或科学主张。

需要修改:
无需修改；保留当前实现和直接证据，并继续确保该局部结论不被外推为模块或科学主张已经成立。

### [T03-15] 状态: PARTIAL

要求:
检查实际 tensor dimension 与配置/论文是否一致。

发现:
输入投影维度为 150（128+19+1+1+1），再加入中心坐标位置编码与 q 编码；没有 query token、entropy、energy 或 MSP 输入。真实五帧 STU 张量未进入该路径。

证据:
- src/model.py:631-679
- src/model.py:1114-1275
- protocol.json:126-138

判断:
只能证明静态实现、局部运行或缺少 生成链路 的产物，尚不能证明可追溯正式路径完整采用；按审计规则不得升级为 PASS。

需要修改:
转为 PASS 所需条件：保存真实五帧输入 tensor schema、维度与模型首层接收记录。

## W00：中心时刻自车对齐

### [W00-01] 状态: PARTIAL

要求:
是否获取每帧 sensor → world pose。

发现:
代码按 `T_ref←W T_W←source` 把五帧转换到中心 LiDAR 坐标并强制中心变换为单位阵；没有数值变换夹具或静态世界点跨帧重合误差报告。

证据:
- src/scene.py:539-660
- src/scene.py:744-952
- test_ajae.py（未发现 synthetic transform test）

判断:
只能证明静态实现、局部运行或缺少 生成链路 的产物，尚不能证明可追溯正式路径完整采用；按审计规则不得升级为 PASS。

需要修改:
转为 PASS 所需条件：补做合成矩阵方向测试和真实静态世界点跨帧重合误差报告。

### [W00-02] 状态: PARTIAL

要求:
是否正确计算 $T_{S_t\leftarrow W}$。

发现:
代码按 `T_ref←W T_W←source` 把五帧转换到中心 LiDAR 坐标并强制中心变换为单位阵；没有数值变换夹具或静态世界点跨帧重合误差报告。

证据:
- src/scene.py:539-660
- src/scene.py:744-952
- test_ajae.py（未发现 synthetic transform test）

判断:
只能证明静态实现、局部运行或缺少 生成链路 的产物，尚不能证明可追溯正式路径完整采用；按审计规则不得升级为 PASS。

需要修改:
转为 PASS 所需条件：补做合成矩阵方向测试和真实静态世界点跨帧重合误差报告。

### [W00-03] 状态: PARTIAL

要求:
是否将五帧点全部转换到 center-time LiDAR frame。

发现:
代码按 `T_ref←W T_W←source` 把五帧转换到中心 LiDAR 坐标并强制中心变换为单位阵；没有数值变换夹具或静态世界点跨帧重合误差报告。

证据:
- src/scene.py:539-660
- src/scene.py:744-952
- test_ajae.py（未发现 synthetic transform test）

判断:
只能证明静态实现、局部运行或缺少 生成链路 的产物，尚不能证明可追溯正式路径完整采用；按审计规则不得升级为 PASS。

需要修改:
转为 PASS 所需条件：补做合成矩阵方向测试和真实静态世界点跨帧重合误差报告。

### [W00-04] 状态: PARTIAL

要求:
是否存在坐标变换方向写反风险。

发现:
代码按 `T_ref←W T_W←source` 把五帧转换到中心 LiDAR 坐标并强制中心变换为单位阵；没有数值变换夹具或静态世界点跨帧重合误差报告。

证据:
- src/scene.py:539-660
- src/scene.py:744-952
- test_ajae.py（未发现 synthetic transform test）

判断:
只能证明静态实现、局部运行或缺少 生成链路 的产物，尚不能证明可追溯正式路径完整采用；按审计规则不得升级为 PASS。

需要修改:
转为 PASS 所需条件：补做合成矩阵方向测试和真实静态世界点跨帧重合误差报告。

### [W00-05] 状态: FAIL

要求:
是否有 synthetic transform unit test。

发现:
仓库没有合成刚体变换夹具验证矩阵方向和中心帧不变性。

证据:
- src/scene.py:539-660
- src/scene.py:744-952
- test_ajae.py（未发现 synthetic transform test）

判断:
存在与要求直接冲突的实现、数据或真实运行事实；判定 FAIL。

需要修改:
增加一个方向可判别的合成刚体变换测试并报告数值误差。

### [W00-06] 状态: PARTIAL

要求:
center frame 自己变换后是否保持不变。

发现:
代码按 `T_ref←W T_W←source` 把五帧转换到中心 LiDAR 坐标并强制中心变换为单位阵；没有数值变换夹具或静态世界点跨帧重合误差报告。

证据:
- src/scene.py:539-660
- src/scene.py:744-952
- test_ajae.py（未发现 synthetic transform test）

判断:
只能证明静态实现、局部运行或缺少 生成链路 的产物，尚不能证明可追溯正式路径完整采用；按审计规则不得升级为 PASS。

需要修改:
转为 PASS 所需条件：补做合成矩阵方向测试和真实静态世界点跨帧重合误差报告。

### [W00-07] 状态: NOT RUN

要求:
静态世界点跨帧转换后是否近似重合。

发现:
代码按 `T_ref←W T_W←source` 把五帧转换到中心 LiDAR 坐标并强制中心变换为单位阵；没有数值变换夹具或静态世界点跨帧重合误差报告。

证据:
- src/scene.py:539-660
- src/scene.py:744-952
- test_ajae.py（未发现 synthetic transform test）

判断:
本条要求的运行级审计、实验或正式产物没有发生；代码存在也不能替代执行，判定 NOT RUN。

需要修改:
转为 PASS 所需条件：补做合成矩阵方向测试和真实静态世界点跨帧重合误差报告。

### [W00-08] 状态: PARTIAL

要求:
moving objects 是否保留实际位移，而非被实例级 tracking 对齐。

发现:
代码按 `T_ref←W T_W←source` 把五帧转换到中心 LiDAR 坐标并强制中心变换为单位阵；没有数值变换夹具或静态世界点跨帧重合误差报告。

证据:
- src/scene.py:539-660
- src/scene.py:744-952
- test_ajae.py（未发现 synthetic transform test）

判断:
只能证明静态实现、局部运行或缺少 生成链路 的产物，尚不能证明可追溯正式路径完整采用；按审计规则不得升级为 PASS。

需要修改:
转为 PASS 所需条件：补做合成矩阵方向测试和真实静态世界点跨帧重合误差报告。

## W01：时间身份

### [W01-01] 状态: PARTIAL

要求:
每个 raw point 是否保存 $q\in{-2,-1,0,+1,+2}$。

发现:
窗口保存完整 q 序列并只向模型暴露相对 q；B5 的物理帧和模型 q 分离，未发现绝对帧编号捷径。正式真实窗口尚未进入模型。

证据:
- src/scene.py:375-536
- test_ajae.py:248-274
- src/model.py:647-679

判断:
只能证明静态实现、局部运行或缺少 生成链路 的产物，尚不能证明可追溯正式路径完整采用；按审计规则不得升级为 PASS。

需要修改:
转为 PASS 所需条件：用真实 B3/B5 窗口记录完整 q 与模型输入。

### [W01-02] 状态: PARTIAL

要求:
五个 $q$ 是否完整。

发现:
窗口保存完整 q 序列并只向模型暴露相对 q；B5 的物理帧和模型 q 分离，未发现绝对帧编号捷径。正式真实窗口尚未进入模型。

证据:
- src/scene.py:375-536
- test_ajae.py:248-274
- src/model.py:647-679

判断:
只能证明静态实现、局部运行或缺少 生成链路 的产物，尚不能证明可追溯正式路径完整采用；按审计规则不得升级为 PASS。

需要修改:
转为 PASS 所需条件：用真实 B3/B5 窗口记录完整 q 与模型输入。

### [W01-03] 状态: PARTIAL

要求:
q 编码方向是否一致。

发现:
窗口保存完整 q 序列并只向模型暴露相对 q；B5 的物理帧和模型 q 分离，未发现绝对帧编号捷径。正式真实窗口尚未进入模型。

证据:
- src/scene.py:375-536
- test_ajae.py:248-274
- src/model.py:647-679

判断:
只能证明静态实现、局部运行或缺少 生成链路 的产物，尚不能证明可追溯正式路径完整采用；按审计规则不得升级为 PASS。

需要修改:
转为 PASS 所需条件：用真实 B3/B5 窗口记录完整 q 与模型输入。

### [W01-04] 状态: PASS

要求:
是否不存在 frame index 泄漏替代 relative-time encoding。

发现:
窗口保存完整 q 序列并只向模型暴露相对 q；B5 的物理帧和模型 q 分离，未发现绝对帧编号捷径。正式真实窗口尚未进入模型。

证据:
- src/scene.py:375-536
- test_ajae.py:248-274
- src/model.py:647-679

判断:
证据直接覆盖该要求，或排除性全仓库搜索已证明冲突路径不在当前正式路线；可判 PASS。该结论只覆盖本条，不外推到整个模块或科学主张。

需要修改:
无需修改；保留当前实现和直接证据，并继续确保该局部结论不被外推为模块或科学主张已经成立。

### [W01-05] 状态: PARTIAL

要求:
同一模型参数是否处理所有 q。

发现:
窗口保存完整 q 序列并只向模型暴露相对 q；B5 的物理帧和模型 q 分离，未发现绝对帧编号捷径。正式真实窗口尚未进入模型。

证据:
- src/scene.py:375-536
- test_ajae.py:248-274
- src/model.py:647-679

判断:
只能证明静态实现、局部运行或缺少 生成链路 的产物，尚不能证明可追溯正式路径完整采用；按审计规则不得升级为 PASS。

需要修改:
转为 PASS 所需条件：用真实 B3/B5 窗口记录完整 q 与模型输入。

## M00：固定第一版结构

### [M00-01] 状态: PARTIAL

要求:
是否正好四级：L0/L1/L2/L3。

发现:
唯一主模型代码为 L0 原始点加 L1/L2/L3 三层体素、mean-max、时间分层注意力、同帧 3NN 与高分辨率 skip；目前只有小张量前后向。

证据:
- src/model.py:845-1275
- protocol.json:126-138
- test_ajae.py:346-399

判断:
只能证明静态实现、局部运行或缺少 生成链路 的产物，尚不能证明可追溯正式路径完整采用；按审计规则不得升级为 PASS。

需要修改:
转为 PASS 所需条件：在真实 STU 输入上运行唯一冻结结构并保存 resolved config。

### [M00-02] 状态: PASS

要求:
是否存在多个 decoder 版本运行后择优；若有，检查是否违反冻结协议。

发现:
唯一主模型代码为 L0 原始点加 L1/L2/L3 三层体素、mean-max、时间分层注意力、同帧 3NN 与高分辨率 skip；目前只有小张量前后向。

证据:
- src/model.py:845-1275
- protocol.json:126-138
- test_ajae.py:346-399

判断:
证据直接覆盖该要求，或排除性全仓库搜索已证明冲突路径不在当前正式路线；可判 PASS。该结论只覆盖本条，不外推到整个模块或科学主张。

需要修改:
无需修改；保留当前实现和直接证据，并继续确保该局部结论不被外推为模块或科学主张已经成立。

### [M00-03] 状态: PASS

要求:
是否把 attention pooling 等非协议版本混作正式主模型。

发现:
唯一主模型代码为 L0 原始点加 L1/L2/L3 三层体素、mean-max、时间分层注意力、同帧 3NN 与高分辨率 skip；目前只有小张量前后向。

证据:
- src/model.py:845-1275
- protocol.json:126-138
- test_ajae.py:346-399

判断:
证据直接覆盖该要求，或排除性全仓库搜索已证明冲突路径不在当前正式路线；可判 PASS。该结论只覆盖本条，不外推到整个模块或科学主张。

需要修改:
无需修改；保留当前实现和直接证据，并继续确保该局部结论不被外推为模块或科学主张已经成立。

### [M00-04] 状态: PARTIAL

要求:
model config 是否唯一对应论文主模型。

发现:
唯一主模型代码为 L0 原始点加 L1/L2/L3 三层体素、mean-max、时间分层注意力、同帧 3NN 与高分辨率 skip；目前只有小张量前后向。

证据:
- src/model.py:845-1275
- protocol.json:126-138
- test_ajae.py:346-399

判断:
只能证明静态实现、局部运行或缺少 生成链路 的产物，尚不能证明可追溯正式路径完整采用；按审计规则不得升级为 PASS。

需要修改:
转为 PASS 所需条件：在真实 STU 输入上运行唯一冻结结构并保存 resolved config。

## M01：每帧独立体素化

### [M01-01] 状态: PARTIAL

要求:
voxel key 是否显式包含 temporal position q。

发现:
体素键显式为 `(q,floor(x/v))`，各层保留相对时间，仓库没有正式的五帧无 q 合并路径；专门的同坐标异 q 测试缺失。

证据:
- src/model.py:845-971
- test_ajae.py（未发现 U06）

判断:
只能证明静态实现、局部运行或缺少 生成链路 的产物，尚不能证明可追溯正式路径完整采用；按审计规则不得升级为 PASS。

需要修改:
转为 PASS 所需条件：补做同坐标异 q 的显式体素分离测试并保存真实运行证据。

### [M01-02] 状态: PARTIAL

要求:
是否等价于 `(q, floor(x/v_l))`。

发现:
体素键显式为 `(q,floor(x/v))`，各层保留相对时间，仓库没有正式的五帧无 q 合并路径；专门的同坐标异 q 测试缺失。

证据:
- src/model.py:845-971
- test_ajae.py（未发现 U06）

判断:
只能证明静态实现、局部运行或缺少 生成链路 的产物，尚不能证明可追溯正式路径完整采用；按审计规则不得升级为 PASS。

需要修改:
转为 PASS 所需条件：补做同坐标异 q 的显式体素分离测试并保存真实运行证据。

### [M01-03] 状态: PARTIAL

要求:
不同 q 的点是否绝不会在 pooling 阶段直接合并。

发现:
体素键显式为 `(q,floor(x/v))`，各层保留相对时间，仓库没有正式的五帧无 q 合并路径；专门的同坐标异 q 测试缺失。

证据:
- src/model.py:845-971
- test_ajae.py（未发现 U06）

判断:
只能证明静态实现、局部运行或缺少 生成链路 的产物，尚不能证明可追溯正式路径完整采用；按审计规则不得升级为 PASS。

需要修改:
转为 PASS 所需条件：补做同坐标异 q 的显式体素分离测试并保存真实运行证据。

### [M01-04] 状态: PARTIAL

要求:
L1/L2/L3 均保持 temporal identity。

发现:
体素键显式为 `(q,floor(x/v))`，各层保留相对时间，仓库没有正式的五帧无 q 合并路径；专门的同坐标异 q 测试缺失。

证据:
- src/model.py:845-971
- test_ajae.py（未发现 U06）

判断:
只能证明静态实现、局部运行或缺少 生成链路 的产物，尚不能证明可追溯正式路径完整采用；按审计规则不得升级为 PASS。

需要修改:
转为 PASS 所需条件：补做同坐标异 q 的显式体素分离测试并保存真实运行证据。

### [M01-05] 状态: PASS

要求:
检查是否曾经直接对五帧 concat 后普通 voxelize；若正式路径仍如此，FAIL。

发现:
体素键显式为 `(q,floor(x/v))`，各层保留相对时间，仓库没有正式的五帧无 q 合并路径；专门的同坐标异 q 测试缺失。

证据:
- src/model.py:845-971
- test_ajae.py（未发现 U06）

判断:
证据直接覆盖该要求，或排除性全仓库搜索已证明冲突路径不在当前正式路线；可判 PASS。该结论只覆盖本条，不外推到整个模块或科学主张。

需要修改:
无需修改；保留当前实现和直接证据，并继续确保该局部结论不被外推为模块或科学主张已经成立。

## M02：均值—最大值池化

### [M02-01] 状态: PARTIAL

要求:
是否计算 mean。

发现:
`VoxelPool` 对实际非空体素计算 mean 和 max，拼接后经可学习投影；没有逐项数值单元测试，只有全模型夹具间接覆盖。

证据:
- src/model.py:845-886
- test_ajae.py:346-399

判断:
只能证明静态实现、局部运行或缺少 生成链路 的产物，尚不能证明可追溯正式路径完整采用；按审计规则不得升级为 PASS。

需要修改:
转为 PASS 所需条件：补做 mean/max/拼接/投影数值测试并在真实路径验证。

### [M02-02] 状态: PARTIAL

要求:
是否计算 max。

发现:
`VoxelPool` 对实际非空体素计算 mean 和 max，拼接后经可学习投影；没有逐项数值单元测试，只有全模型夹具间接覆盖。

证据:
- src/model.py:845-886
- test_ajae.py:346-399

判断:
只能证明静态实现、局部运行或缺少 生成链路 的产物，尚不能证明可追溯正式路径完整采用；按审计规则不得升级为 PASS。

需要修改:
转为 PASS 所需条件：补做 mean/max/拼接/投影数值测试并在真实路径验证。

### [M02-03] 状态: PARTIAL

要求:
是否 concatenate mean/max。

发现:
`VoxelPool` 对实际非空体素计算 mean 和 max，拼接后经可学习投影；没有逐项数值单元测试，只有全模型夹具间接覆盖。

证据:
- src/model.py:845-886
- test_ajae.py:346-399

判断:
只能证明静态实现、局部运行或缺少 生成链路 的产物，尚不能证明可追溯正式路径完整采用；按审计规则不得升级为 PASS。

需要修改:
转为 PASS 所需条件：补做 mean/max/拼接/投影数值测试并在真实路径验证。

### [M02-04] 状态: PARTIAL

要求:
是否通过 learned linear projection 融合。

发现:
`VoxelPool` 对实际非空体素计算 mean 和 max，拼接后经可学习投影；没有逐项数值单元测试，只有全模型夹具间接覆盖。

证据:
- src/model.py:845-886
- test_ajae.py:346-399

判断:
只能证明静态实现、局部运行或缺少 生成链路 的产物，尚不能证明可追溯正式路径完整采用；按审计规则不得升级为 PASS。

需要修改:
转为 PASS 所需条件：补做 mean/max/拼接/投影数值测试并在真实路径验证。

### [M02-05] 状态: PARTIAL

要求:
是否并非 mean-only。

发现:
`VoxelPool` 对实际非空体素计算 mean 和 max，拼接后经可学习投影；没有逐项数值单元测试，只有全模型夹具间接覆盖。

证据:
- src/model.py:845-886
- test_ajae.py:346-399

判断:
只能证明静态实现、局部运行或缺少 生成链路 的产物，尚不能证明可追溯正式路径完整采用；按审计规则不得升级为 PASS。

需要修改:
转为 PASS 所需条件：补做 mean/max/拼接/投影数值测试并在真实路径验证。

### [M02-06] 状态: PARTIAL

要求:
是否并非 max-only。

发现:
`VoxelPool` 对实际非空体素计算 mean 和 max，拼接后经可学习投影；没有逐项数值单元测试，只有全模型夹具间接覆盖。

证据:
- src/model.py:845-886
- test_ajae.py:346-399

判断:
只能证明静态实现、局部运行或缺少 生成链路 的产物，尚不能证明可追溯正式路径完整采用；按审计规则不得升级为 PASS。

需要修改:
转为 PASS 所需条件：补做 mean/max/拼接/投影数值测试并在真实路径验证。

### [M02-07] 状态: PARTIAL

要求:
空 voxel/数值边界正确。

发现:
`VoxelPool` 对实际非空体素计算 mean 和 max，拼接后经可学习投影；没有逐项数值单元测试，只有全模型夹具间接覆盖。

证据:
- src/model.py:845-886
- test_ajae.py:346-399

判断:
只能证明静态实现、局部运行或缺少 生成链路 的产物，尚不能证明可追溯正式路径完整采用；按审计规则不得升级为 PASS。

需要修改:
转为 PASS 所需条件：补做 mean/max/拼接/投影数值测试并在真实路径验证。

## M03：按时间差分层的邻域

### [M03-01] 状态: PARTIAL

要求:
neighbor search 是否按 $\delta=q_j-q_i$ 分层。

发现:
每个精确 delta 单独执行 radius-K 查询，候选不足保持为空，不跨 delta 或半径补点；配置给出逐层逐 delta 半径和 K。只有一个 delta=+1 小夹具实际运行。

证据:
- src/model.py:576-842
- protocol.json:132-135
- test_ajae.py:337-343

判断:
只能证明静态实现、局部运行或缺少 生成链路 的产物，尚不能证明可追溯正式路径完整采用；按审计规则不得升级为 PASS。

需要修改:
转为 PASS 所需条件：扩展到所有 delta、空候选和边界半径的运行测试，并保存正式配置。

### [M03-02] 状态: PARTIAL

要求:
是否支持 $\delta=-2$。

发现:
每个精确 delta 单独执行 radius-K 查询，候选不足保持为空，不跨 delta 或半径补点；配置给出逐层逐 delta 半径和 K。只有一个 delta=+1 小夹具实际运行。

证据:
- src/model.py:576-842
- protocol.json:132-135
- test_ajae.py:337-343

判断:
只能证明静态实现、局部运行或缺少 生成链路 的产物，尚不能证明可追溯正式路径完整采用；按审计规则不得升级为 PASS。

需要修改:
转为 PASS 所需条件：扩展到所有 delta、空候选和边界半径的运行测试，并保存正式配置。

### [M03-03] 状态: PARTIAL

要求:
是否支持 $\delta=-1$。

发现:
每个精确 delta 单独执行 radius-K 查询，候选不足保持为空，不跨 delta 或半径补点；配置给出逐层逐 delta 半径和 K。只有一个 delta=+1 小夹具实际运行。

证据:
- src/model.py:576-842
- protocol.json:132-135
- test_ajae.py:337-343

判断:
只能证明静态实现、局部运行或缺少 生成链路 的产物，尚不能证明可追溯正式路径完整采用；按审计规则不得升级为 PASS。

需要修改:
转为 PASS 所需条件：扩展到所有 delta、空候选和边界半径的运行测试，并保存正式配置。

### [M03-04] 状态: PARTIAL

要求:
是否支持 $\delta=0$。

发现:
每个精确 delta 单独执行 radius-K 查询，候选不足保持为空，不跨 delta 或半径补点；配置给出逐层逐 delta 半径和 K。只有一个 delta=+1 小夹具实际运行。

证据:
- src/model.py:576-842
- protocol.json:132-135
- test_ajae.py:337-343

判断:
只能证明静态实现、局部运行或缺少 生成链路 的产物，尚不能证明可追溯正式路径完整采用；按审计规则不得升级为 PASS。

需要修改:
转为 PASS 所需条件：扩展到所有 delta、空候选和边界半径的运行测试，并保存正式配置。

### [M03-05] 状态: PARTIAL

要求:
是否支持 $\delta=+1$。

发现:
每个精确 delta 单独执行 radius-K 查询，候选不足保持为空，不跨 delta 或半径补点；配置给出逐层逐 delta 半径和 K。只有一个 delta=+1 小夹具实际运行。

证据:
- src/model.py:576-842
- protocol.json:132-135
- test_ajae.py:337-343

判断:
只能证明静态实现、局部运行或缺少 生成链路 的产物，尚不能证明可追溯正式路径完整采用；按审计规则不得升级为 PASS。

需要修改:
转为 PASS 所需条件：扩展到所有 delta、空候选和边界半径的运行测试，并保存正式配置。

### [M03-06] 状态: PARTIAL

要求:
是否支持 $\delta=+2$。

发现:
每个精确 delta 单独执行 radius-K 查询，候选不足保持为空，不跨 delta 或半径补点；配置给出逐层逐 delta 半径和 K。只有一个 delta=+1 小夹具实际运行。

证据:
- src/model.py:576-842
- protocol.json:132-135
- test_ajae.py:337-343

判断:
只能证明静态实现、局部运行或缺少 生成链路 的产物，尚不能证明可追溯正式路径完整采用；按审计规则不得升级为 PASS。

需要修改:
转为 PASS 所需条件：扩展到所有 delta、空候选和边界半径的运行测试，并保存正式配置。

### [M03-07] 状态: PARTIAL

要求:
每个 delta 是否拥有独立 $K_{l,\delta}$ 或显式配置。

发现:
每个精确 delta 单独执行 radius-K 查询，候选不足保持为空，不跨 delta 或半径补点；配置给出逐层逐 delta 半径和 K。只有一个 delta=+1 小夹具实际运行。

证据:
- src/model.py:576-842
- protocol.json:132-135
- test_ajae.py:337-343

判断:
只能证明静态实现、局部运行或缺少 生成链路 的产物，尚不能证明可追溯正式路径完整采用；按审计规则不得升级为 PASS。

需要修改:
转为 PASS 所需条件：扩展到所有 delta、空候选和边界半径的运行测试，并保存正式配置。

### [M03-08] 状态: PARTIAL

要求:
每个 delta 是否拥有独立 radius $r_{l,\delta}$ 或显式配置。

发现:
每个精确 delta 单独执行 radius-K 查询，候选不足保持为空，不跨 delta 或半径补点；配置给出逐层逐 delta 半径和 K。只有一个 delta=+1 小夹具实际运行。

证据:
- src/model.py:576-842
- protocol.json:132-135
- test_ajae.py:337-343

判断:
只能证明静态实现、局部运行或缺少 生成链路 的产物，尚不能证明可追溯正式路径完整采用；按审计规则不得升级为 PASS。

需要修改:
转为 PASS 所需条件：扩展到所有 delta、空候选和边界半径的运行测试，并保存正式配置。

### [M03-09] 状态: PARTIAL

要求:
候选不足时是否允许为空。

发现:
每个精确 delta 单独执行 radius-K 查询，候选不足保持为空，不跨 delta 或半径补点；配置给出逐层逐 delta 半径和 K。只有一个 delta=+1 小夹具实际运行。

证据:
- src/model.py:576-842
- protocol.json:132-135
- test_ajae.py:337-343

判断:
只能证明静态实现、局部运行或缺少 生成链路 的产物，尚不能证明可追溯正式路径完整采用；按审计规则不得升级为 PASS。

需要修改:
转为 PASS 所需条件：扩展到所有 delta、空候选和边界半径的运行测试，并保存正式配置。

### [M03-10] 状态: PARTIAL

要求:
是否不会去其他 delta 补邻居。

发现:
每个精确 delta 单独执行 radius-K 查询，候选不足保持为空，不跨 delta 或半径补点；配置给出逐层逐 delta 半径和 K。只有一个 delta=+1 小夹具实际运行。

证据:
- src/model.py:576-842
- protocol.json:132-135
- test_ajae.py:337-343

判断:
只能证明静态实现、局部运行或缺少 生成链路 的产物，尚不能证明可追溯正式路径完整采用；按审计规则不得升级为 PASS。

需要修改:
转为 PASS 所需条件：扩展到所有 delta、空候选和边界半径的运行测试，并保存正式配置。

### [M03-11] 状态: PARTIAL

要求:
是否不会用远距离点强行凑 K。

发现:
每个精确 delta 单独执行 radius-K 查询，候选不足保持为空，不跨 delta 或半径补点；配置给出逐层逐 delta 半径和 K。只有一个 delta=+1 小夹具实际运行。

证据:
- src/model.py:576-842
- protocol.json:132-135
- test_ajae.py:337-343

判断:
只能证明静态实现、局部运行或缺少 生成链路 的产物，尚不能证明可追溯正式路径完整采用；按审计规则不得升级为 PASS。

需要修改:
转为 PASS 所需条件：扩展到所有 delta、空候选和边界半径的运行测试，并保存正式配置。

### [M03-12] 状态: PASS

要求:
是否不存在“五帧共同竞争一个 global K”的旧实现。

发现:
每个精确 delta 单独执行 radius-K 查询，候选不足保持为空，不跨 delta 或半径补点；配置给出逐层逐 delta 半径和 K。只有一个 delta=+1 小夹具实际运行。

证据:
- src/model.py:576-842
- protocol.json:132-135
- test_ajae.py:337-343

判断:
证据直接覆盖该要求，或排除性全仓库搜索已证明冲突路径不在当前正式路线；可判 PASS。该结论只覆盖本条，不外推到整个模块或科学主张。

需要修改:
无需修改；保留当前实现和直接证据，并继续确保该局部结论不被外推为模块或科学主张已经成立。

### [M03-13] 状态: PASS

要求:
搜索 repository 中旧 KNN 实现并确认未被正式配置引用。

发现:
每个精确 delta 单独执行 radius-K 查询，候选不足保持为空，不跨 delta 或半径补点；配置给出逐层逐 delta 半径和 K。只有一个 delta=+1 小夹具实际运行。

证据:
- src/model.py:576-842
- protocol.json:132-135
- test_ajae.py:337-343

判断:
证据直接覆盖该要求，或排除性全仓库搜索已证明冲突路径不在当前正式路线；可判 PASS。该结论只覆盖本条，不外推到整个模块或科学主张。

需要修改:
无需修改；保留当前实现和直接证据，并继续确保该局部结论不被外推为模块或科学主张已经成立。

### [M03-14] 状态: PARTIAL

要求:
attention 是否使用相对空间位置。

发现:
每个精确 delta 单独执行 radius-K 查询，候选不足保持为空，不跨 delta 或半径补点；配置给出逐层逐 delta 半径和 K。只有一个 delta=+1 小夹具实际运行。

证据:
- src/model.py:576-842
- protocol.json:132-135
- test_ajae.py:337-343

判断:
只能证明静态实现、局部运行或缺少 生成链路 的产物，尚不能证明可追溯正式路径完整采用；按审计规则不得升级为 PASS。

需要修改:
转为 PASS 所需条件：扩展到所有 delta、空候选和边界半径的运行测试，并保存正式配置。

### [M03-15] 状态: PARTIAL

要求:
attention 是否使用相对时间差 delta。

发现:
每个精确 delta 单独执行 radius-K 查询，候选不足保持为空，不跨 delta 或半径补点；配置给出逐层逐 delta 半径和 K。只有一个 delta=+1 小夹具实际运行。

证据:
- src/model.py:576-842
- protocol.json:132-135
- test_ajae.py:337-343

判断:
只能证明静态实现、局部运行或缺少 生成链路 的产物，尚不能证明可追溯正式路径完整采用；按审计规则不得升级为 PASS。

需要修改:
转为 PASS 所需条件：扩展到所有 delta、空候选和边界半径的运行测试，并保存正式配置。

### [M03-16] 状态: PARTIAL

要求:
主要 Q/K/V projection 是否跨 temporal branches 共享。

发现:
每个精确 delta 单独执行 radius-K 查询，候选不足保持为空，不跨 delta 或半径补点；配置给出逐层逐 delta 半径和 K。只有一个 delta=+1 小夹具实际运行。

证据:
- src/model.py:576-842
- protocol.json:132-135
- test_ajae.py:337-343

判断:
只能证明静态实现、局部运行或缺少 生成链路 的产物，尚不能证明可追溯正式路径完整采用；按审计规则不得升级为 PASS。

需要修改:
转为 PASS 所需条件：扩展到所有 delta、空候选和边界半径的运行测试，并保存正式配置。

## M04：同帧残差

### [M04-01] 状态: PARTIAL

要求:
$\delta=0$ 分支始终存在。

发现:
delta=0 分支作为独立同帧消息残差加入，跨帧 gate 只作用于非零 delta，随后还有 FFN 残差；空跨帧与运动正常真实验证缺失。

证据:
- src/model.py:797-842
- test_ajae.py:346-399

判断:
只能证明静态实现、局部运行或缺少 生成链路 的产物，尚不能证明可追溯正式路径完整采用；按审计规则不得升级为 PASS。

需要修改:
转为 PASS 所需条件：以空跨帧和真实 moving-normal 证据验证同帧残差始终可用。

### [M04-02] 状态: PARTIAL

要求:
same-frame message 独立计算。

发现:
delta=0 分支作为独立同帧消息残差加入，跨帧 gate 只作用于非零 delta，随后还有 FFN 残差；空跨帧与运动正常真实验证缺失。

证据:
- src/model.py:797-842
- test_ajae.py:346-399

判断:
只能证明静态实现、局部运行或缺少 生成链路 的产物，尚不能证明可追溯正式路径完整采用；按审计规则不得升级为 PASS。

需要修改:
转为 PASS 所需条件：以空跨帧和真实 moving-normal 证据验证同帧残差始终可用。

### [M04-03] 状态: PARTIAL

要求:
same-frame branch 不受跨帧 gate 影响而被完全抑制。

发现:
delta=0 分支作为独立同帧消息残差加入，跨帧 gate 只作用于非零 delta，随后还有 FFN 残差；空跨帧与运动正常真实验证缺失。

证据:
- src/model.py:797-842
- test_ajae.py:346-399

判断:
只能证明静态实现、局部运行或缺少 生成链路 的产物，尚不能证明可追溯正式路径完整采用；按审计规则不得升级为 PASS。

需要修改:
转为 PASS 所需条件：以空跨帧和真实 moving-normal 证据验证同帧残差始终可用。

### [M04-04] 状态: PARTIAL

要求:
block 具备 residual path。

发现:
delta=0 分支作为独立同帧消息残差加入，跨帧 gate 只作用于非零 delta，随后还有 FFN 残差；空跨帧与运动正常真实验证缺失。

证据:
- src/model.py:797-842
- test_ajae.py:346-399

判断:
只能证明静态实现、局部运行或缺少 生成链路 的产物，尚不能证明可追溯正式路径完整采用；按审计规则不得升级为 PASS。

需要修改:
转为 PASS 所需条件：以空跨帧和真实 moving-normal 证据验证同帧残差始终可用。

### [M04-05] 状态: PARTIAL

要求:
moving normal object 即使跨帧邻域不可靠仍有单帧信息通路。

发现:
delta=0 分支作为独立同帧消息残差加入，跨帧 gate 只作用于非零 delta，随后还有 FFN 残差；空跨帧与运动正常真实验证缺失。

证据:
- src/model.py:797-842
- test_ajae.py:346-399

判断:
只能证明静态实现、局部运行或缺少 生成链路 的产物，尚不能证明可追溯正式路径完整采用；按审计规则不得升级为 PASS。

需要修改:
转为 PASS 所需条件：以空跨帧和真实 moving-normal 证据验证同帧残差始终可用。

## M05：跨帧拒绝门

### [M05-01] 状态: PARTIAL

要求:
每个 $\delta\ne0$ 分支存在 gate。

发现:
每个非零 delta 用 current feature、branch message 和 delta 计算 sigmoid gate，空邻域代码意图令 message/gate 为 0；缺少直接数值断言和正式运行。

证据:
- src/model.py:713-842
- test_ajae.py（未发现 U08）

判断:
只能证明静态实现、局部运行或缺少 生成链路 的产物，尚不能证明可追溯正式路径完整采用；按审计规则不得升级为 PASS。

需要修改:
转为 PASS 所需条件：补做空分支 message=0、gate=0、finite output 测试并记录真实 gate。

### [M05-02] 状态: PARTIAL

要求:
gate 范围是否经 sigmoid 限制。

发现:
每个非零 delta 用 current feature、branch message 和 delta 计算 sigmoid gate，空邻域代码意图令 message/gate 为 0；缺少直接数值断言和正式运行。

证据:
- src/model.py:713-842
- test_ajae.py（未发现 U08）

判断:
只能证明静态实现、局部运行或缺少 生成链路 的产物，尚不能证明可追溯正式路径完整采用；按审计规则不得升级为 PASS。

需要修改:
转为 PASS 所需条件：补做空分支 message=0、gate=0、finite output 测试并记录真实 gate。

### [M05-03] 状态: PARTIAL

要求:
gate 输入至少包含 query/current feature。

发现:
每个非零 delta 用 current feature、branch message 和 delta 计算 sigmoid gate，空邻域代码意图令 message/gate 为 0；缺少直接数值断言和正式运行。

证据:
- src/model.py:713-842
- test_ajae.py（未发现 U08）

判断:
只能证明静态实现、局部运行或缺少 生成链路 的产物，尚不能证明可追溯正式路径完整采用；按审计规则不得升级为 PASS。

需要修改:
转为 PASS 所需条件：补做空分支 message=0、gate=0、finite output 测试并记录真实 gate。

### [M05-04] 状态: PARTIAL

要求:
gate 输入包含 temporal branch message。

发现:
每个非零 delta 用 current feature、branch message 和 delta 计算 sigmoid gate，空邻域代码意图令 message/gate 为 0；缺少直接数值断言和正式运行。

证据:
- src/model.py:713-842
- test_ajae.py（未发现 U08）

判断:
只能证明静态实现、局部运行或缺少 生成链路 的产物，尚不能证明可追溯正式路径完整采用；按审计规则不得升级为 PASS。

需要修改:
转为 PASS 所需条件：补做空分支 message=0、gate=0、finite output 测试并记录真实 gate。

### [M05-05] 状态: PARTIAL

要求:
gate 能识别 delta。

发现:
每个非零 delta 用 current feature、branch message 和 delta 计算 sigmoid gate，空邻域代码意图令 message/gate 为 0；缺少直接数值断言和正式运行。

证据:
- src/model.py:713-842
- test_ajae.py（未发现 U08）

判断:
只能证明静态实现、局部运行或缺少 生成链路 的产物，尚不能证明可追溯正式路径完整采用；按审计规则不得升级为 PASS。

需要修改:
转为 PASS 所需条件：补做空分支 message=0、gate=0、finite output 测试并记录真实 gate。

### [M05-06] 状态: NOT RUN

要求:
空 temporal neighborhood 时 message = 0。

发现:
每个非零 delta 用 current feature、branch message 和 delta 计算 sigmoid gate，空邻域代码意图令 message/gate 为 0；缺少直接数值断言和正式运行。

证据:
- src/model.py:713-842
- test_ajae.py（未发现 U08）

判断:
本条要求的运行级审计、实验或正式产物没有发生；代码存在也不能替代执行，判定 NOT RUN。

需要修改:
转为 PASS 所需条件：补做空分支 message=0、gate=0、finite output 测试并记录真实 gate。

### [M05-07] 状态: NOT RUN

要求:
空 temporal neighborhood 时 gate = 0。

发现:
每个非零 delta 用 current feature、branch message 和 delta 计算 sigmoid gate，空邻域代码意图令 message/gate 为 0；缺少直接数值断言和正式运行。

证据:
- src/model.py:713-842
- test_ajae.py（未发现 U08）

判断:
本条要求的运行级审计、实验或正式产物没有发生；代码存在也不能替代执行，判定 NOT RUN。

需要修改:
转为 PASS 所需条件：补做空分支 message=0、gate=0、finite output 测试并记录真实 gate。

### [M05-08] 状态: PASS

要求:
是否不存在 forced temporal aggregation。

发现:
每个非零 delta 用 current feature、branch message 和 delta 计算 sigmoid gate，空邻域代码意图令 message/gate 为 0；缺少直接数值断言和正式运行。

证据:
- src/model.py:713-842
- test_ajae.py（未发现 U08）

判断:
证据直接覆盖该要求，或排除性全仓库搜索已证明冲突路径不在当前正式路线；可判 PASS。该结论只覆盖本条，不外推到整个模块或科学主张。

需要修改:
无需修改；保留当前实现和直接证据，并继续确保该局部结论不被外推为模块或科学主张已经成立。

### [M05-09] 状态: PARTIAL

要求:
跨帧信息是否通过 gated residual 写入 feature。

发现:
每个非零 delta 用 current feature、branch message 和 delta 计算 sigmoid gate，空邻域代码意图令 message/gate 为 0；缺少直接数值断言和正式运行。

证据:
- src/model.py:713-842
- test_ajae.py（未发现 U08）

判断:
只能证明静态实现、局部运行或缺少 生成链路 的产物，尚不能证明可追溯正式路径完整采用；按审计规则不得升级为 PASS。

需要修改:
转为 PASS 所需条件：补做空分支 message=0、gate=0、finite output 测试并记录真实 gate。

## M06：层级感受野

### [M06-01] 状态: PARTIAL

要求:
L0 半径最小。

发现:
四层每个 delta 的半径严格递增，构造器会拒绝非递增配置；尚无真实对象尺度性能证据，文档也未把结构约束夸大为结果。

证据:
- src/model.py:889-971
- protocol.json:132-134
- AJAE.tex:397-405

判断:
只能证明静态实现、局部运行或缺少 生成链路 的产物，尚不能证明可追溯正式路径完整采用；按审计规则不得升级为 PASS。

需要修改:
转为 PASS 所需条件：在真实模型配置与对象尺度诊断中验证层级使用；不得把结构直接当性能证据。

### [M06-02] 状态: PARTIAL

要求:
L1 半径增大。

发现:
四层每个 delta 的半径严格递增，构造器会拒绝非递增配置；尚无真实对象尺度性能证据，文档也未把结构约束夸大为结果。

证据:
- src/model.py:889-971
- protocol.json:132-134
- AJAE.tex:397-405

判断:
只能证明静态实现、局部运行或缺少 生成链路 的产物，尚不能证明可追溯正式路径完整采用；按审计规则不得升级为 PASS。

需要修改:
转为 PASS 所需条件：在真实模型配置与对象尺度诊断中验证层级使用；不得把结构直接当性能证据。

### [M06-03] 状态: PARTIAL

要求:
L2 半径继续增大。

发现:
四层每个 delta 的半径严格递增，构造器会拒绝非递增配置；尚无真实对象尺度性能证据，文档也未把结构约束夸大为结果。

证据:
- src/model.py:889-971
- protocol.json:132-134
- AJAE.tex:397-405

判断:
只能证明静态实现、局部运行或缺少 生成链路 的产物，尚不能证明可追溯正式路径完整采用；按审计规则不得升级为 PASS。

需要修改:
转为 PASS 所需条件：在真实模型配置与对象尺度诊断中验证层级使用；不得把结构直接当性能证据。

### [M06-04] 状态: PARTIAL

要求:
L3 半径最大。

发现:
四层每个 delta 的半径严格递增，构造器会拒绝非递增配置；尚无真实对象尺度性能证据，文档也未把结构约束夸大为结果。

证据:
- src/model.py:889-971
- protocol.json:132-134
- AJAE.tex:397-405

判断:
只能证明静态实现、局部运行或缺少 生成链路 的产物，尚不能证明可追溯正式路径完整采用；按审计规则不得升级为 PASS。

需要修改:
转为 PASS 所需条件：在真实模型配置与对象尺度诊断中验证层级使用；不得把结构直接当性能证据。

### [M06-05] 状态: PARTIAL

要求:
对所有关键 temporal offsets 检查 $r_0<r_1<r_2<r_3$。

发现:
四层每个 delta 的半径严格递增，构造器会拒绝非递增配置；尚无真实对象尺度性能证据，文档也未把结构约束夸大为结果。

证据:
- src/model.py:889-971
- protocol.json:132-134
- AJAE.tex:397-405

判断:
只能证明静态实现、局部运行或缺少 生成链路 的产物，尚不能证明可追溯正式路径完整采用；按审计规则不得升级为 PASS。

需要修改:
转为 PASS 所需条件：在真实模型配置与对象尺度诊断中验证层级使用；不得把结构直接当性能证据。

### [M06-06] 状态: PASS

要求:
如果部分层半径不扩大，说明与方案不一致。

发现:
四层每个 delta 的半径严格递增，构造器会拒绝非递增配置；尚无真实对象尺度性能证据，文档也未把结构约束夸大为结果。

证据:
- src/model.py:889-971
- protocol.json:132-134
- AJAE.tex:397-405

判断:
证据直接覆盖该要求，或排除性全仓库搜索已证明冲突路径不在当前正式路线；可判 PASS。该结论只覆盖本条，不外推到整个模块或科学主张。

需要修改:
无需修改；保留当前实现和直接证据，并继续确保该局部结论不被外推为模块或科学主张已经成立。

### [M06-07] 状态: PASS

要求:
代码/报告不得把“对象尺度”写成结构自动保证的事实。

发现:
四层每个 delta 的半径严格递增，构造器会拒绝非递增配置；尚无真实对象尺度性能证据，文档也未把结构约束夸大为结果。

证据:
- src/model.py:889-971
- protocol.json:132-134
- AJAE.tex:397-405

判断:
证据直接覆盖该要求，或排除性全仓库搜索已证明冲突路径不在当前正式路线；可判 PASS。该结论只覆盖本条，不外推到整个模块或科学主张。

需要修改:
无需修改；保留当前实现和直接证据，并继续确保该局部结论不被外推为模块或科学主张已经成立。

## M07：同帧 3NN 解码器

### [M07-01] 状态: PARTIAL

要求:
coarse → fine 插值是否只搜索相同 q。

发现:
解码器按同一 q 选最多三个父节点、用逆距离归一化并连接高分辨率 skip；没有“其他 q 更近仍不可选”的定向测试或正式运行。

证据:
- src/model.py:974-1042
- src/model.py:1249-1273
- test_ajae.py（未发现 U09）

判断:
只能证明静态实现、局部运行或缺少 生成链路 的产物，尚不能证明可追溯正式路径完整采用；按审计规则不得升级为 PASS。

需要修改:
转为 PASS 所需条件：补做跨 q 更近反例测试，并在真实解码结果核对同帧约束。

### [M07-02] 状态: PARTIAL

要求:
每点最多/正好使用 3 nearest parents。

发现:
解码器按同一 q 选最多三个父节点、用逆距离归一化并连接高分辨率 skip；没有“其他 q 更近仍不可选”的定向测试或正式运行。

证据:
- src/model.py:974-1042
- src/model.py:1249-1273
- test_ajae.py（未发现 U09）

判断:
只能证明静态实现、局部运行或缺少 生成链路 的产物，尚不能证明可追溯正式路径完整采用；按审计规则不得升级为 PASS。

需要修改:
转为 PASS 所需条件：补做跨 q 更近反例测试，并在真实解码结果核对同帧约束。

### [M07-03] 状态: PARTIAL

要求:
权重是否基于 inverse distance 或方案等价实现。

发现:
解码器按同一 q 选最多三个父节点、用逆距离归一化并连接高分辨率 skip；没有“其他 q 更近仍不可选”的定向测试或正式运行。

证据:
- src/model.py:974-1042
- src/model.py:1249-1273
- test_ajae.py（未发现 U09）

判断:
只能证明静态实现、局部运行或缺少 生成链路 的产物，尚不能证明可追溯正式路径完整采用；按审计规则不得升级为 PASS。

需要修改:
转为 PASS 所需条件：补做跨 q 更近反例测试，并在真实解码结果核对同帧约束。

### [M07-04] 状态: PARTIAL

要求:
是否归一化权重。

发现:
解码器按同一 q 选最多三个父节点、用逆距离归一化并连接高分辨率 skip；没有“其他 q 更近仍不可选”的定向测试或正式运行。

证据:
- src/model.py:974-1042
- src/model.py:1249-1273
- test_ajae.py（未发现 U09）

判断:
只能证明静态实现、局部运行或缺少 生成链路 的产物，尚不能证明可追溯正式路径完整采用；按审计规则不得升级为 PASS。

需要修改:
转为 PASS 所需条件：补做跨 q 更近反例测试，并在真实解码结果核对同帧约束。

### [M07-05] 状态: PARTIAL

要求:
是否有 epsilon 防止除零。

发现:
解码器按同一 q 选最多三个父节点、用逆距离归一化并连接高分辨率 skip；没有“其他 q 更近仍不可选”的定向测试或正式运行。

证据:
- src/model.py:974-1042
- src/model.py:1249-1273
- test_ajae.py（未发现 U09）

判断:
只能证明静态实现、局部运行或缺少 生成链路 的产物，尚不能证明可追溯正式路径完整采用；按审计规则不得升级为 PASS。

需要修改:
转为 PASS 所需条件：补做跨 q 更近反例测试，并在真实解码结果核对同帧约束。

### [M07-06] 状态: PARTIAL

要求:
是否与 high-resolution skip feature 融合。

发现:
解码器按同一 q 选最多三个父节点、用逆距离归一化并连接高分辨率 skip；没有“其他 q 更近仍不可选”的定向测试或正式运行。

证据:
- src/model.py:974-1042
- src/model.py:1249-1273
- test_ajae.py（未发现 U09）

判断:
只能证明静态实现、局部运行或缺少 生成链路 的产物，尚不能证明可追溯正式路径完整采用；按审计规则不得升级为 PASS。

需要修改:
转为 PASS 所需条件：补做跨 q 更近反例测试，并在真实解码结果核对同帧约束。

### [M07-07] 状态: PASS

要求:
decoder 阶段是否没有重新执行 cross-frame matching。

发现:
解码器按同一 q 选最多三个父节点、用逆距离归一化并连接高分辨率 skip；没有“其他 q 更近仍不可选”的定向测试或正式运行。

证据:
- src/model.py:974-1042
- src/model.py:1249-1273
- test_ajae.py（未发现 U09）

判断:
证据直接覆盖该要求，或排除性全仓库搜索已证明冲突路径不在当前正式路线；可判 PASS。该结论只覆盖本条，不外推到整个模块或科学主张。

需要修改:
无需修改；保留当前实现和直接证据，并继续确保该局部结论不被外推为模块或科学主张已经成立。

### [M07-08] 状态: PASS

要求:
是否不存在跨帧 3NN 上采样。

发现:
解码器按同一 q 选最多三个父节点、用逆距离归一化并连接高分辨率 skip；没有“其他 q 更近仍不可选”的定向测试或正式运行。

证据:
- src/model.py:974-1042
- src/model.py:1249-1273
- test_ajae.py（未发现 U09）

判断:
证据直接覆盖该要求，或排除性全仓库搜索已证明冲突路径不在当前正式路线；可判 PASS。该结论只覆盖本条，不外推到整个模块或科学主张。

需要修改:
无需修改；保留当前实现和直接证据，并继续确保该局部结论不被外推为模块或科学主张已经成立。

## L00：点标签

### [L00-01] 状态: PARTIAL

要求:
canonical real normal return = 0。

发现:
训练张量把原始正常和 normal-control 标为 0、proxy 标为 1，并排除被替换原回波；生成器对象编号和 tracking 不进入 loss。正式监督张量未运行。

证据:
- src/train.py:623-721
- src/train.py:805-810

判断:
只能证明静态实现、局部运行或缺少 生成链路 的产物，尚不能证明可追溯正式路径完整采用；按审计规则不得升级为 PASS。

需要修改:
转为 PASS 所需条件：在允许训练后保存真实五帧 label/valid mask 统计。

### [L00-02] 状态: PARTIAL

要求:
rendered normal-control return = 0。

发现:
训练张量把原始正常和 normal-control 标为 0、proxy 标为 1，并排除被替换原回波；生成器对象编号和 tracking 不进入 loss。正式监督张量未运行。

证据:
- src/train.py:623-721
- src/train.py:805-810

判断:
只能证明静态实现、局部运行或缺少 生成链路 的产物，尚不能证明可追溯正式路径完整采用；按审计规则不得升级为 PASS。

需要修改:
转为 PASS 所需条件：在允许训练后保存真实五帧 label/valid mask 统计。

### [L00-03] 状态: PARTIAL

要求:
anomaly-proxy return = 1。

发现:
训练张量把原始正常和 normal-control 标为 0、proxy 标为 1，并排除被替换原回波；生成器对象编号和 tracking 不进入 loss。正式监督张量未运行。

证据:
- src/train.py:623-721
- src/train.py:805-810

判断:
只能证明静态实现、局部运行或缺少 生成链路 的产物，尚不能证明可追溯正式路径完整采用；按审计规则不得升级为 PASS。

需要修改:
转为 PASS 所需条件：在允许训练后保存真实五帧 label/valid mask 统计。

### [L00-04] 状态: PARTIAL

要求:
occluded original point 被删除而不是标 normal 继续监督。

发现:
训练张量把原始正常和 normal-control 标为 0、proxy 标为 1，并排除被替换原回波；生成器对象编号和 tracking 不进入 loss。正式监督张量未运行。

证据:
- src/train.py:623-721
- src/train.py:805-810

判断:
只能证明静态实现、局部运行或缺少 生成链路 的产物，尚不能证明可追溯正式路径完整采用；按审计规则不得升级为 PASS。

需要修改:
转为 PASS 所需条件：在允许训练后保存真实五帧 label/valid mask 统计。

### [L00-05] 状态: PARTIAL

要求:
raw semantic 0 默认 ignore。

发现:
训练张量把原始正常和 normal-control 标为 0、proxy 标为 1，并排除被替换原回波；生成器对象编号和 tracking 不进入 loss。正式监督张量未运行。

证据:
- src/train.py:623-721
- src/train.py:805-810

判断:
只能证明静态实现、局部运行或缺少 生成链路 的产物，尚不能证明可追溯正式路径完整采用；按审计规则不得升级为 PASS。

需要修改:
转为 PASS 所需条件：在允许训练后保存真实五帧 label/valid mask 统计。

### [L00-06] 状态: PARTIAL

要求:
如果 ignored raw slot 被 inserted valid return 替换，新点按 inserted entity label。

发现:
训练张量把原始正常和 normal-control 标为 0、proxy 标为 1，并排除被替换原回波；生成器对象编号和 tracking 不进入 loss。正式监督张量未运行。

证据:
- src/train.py:623-721
- src/train.py:805-810

判断:
只能证明静态实现、局部运行或缺少 生成链路 的产物，尚不能证明可追溯正式路径完整采用；按审计规则不得升级为 PASS。

需要修改:
转为 PASS 所需条件：在允许训练后保存真实五帧 label/valid mask 统计。

### [L00-07] 状态: PASS

要求:
不存在 object ID supervision。

发现:
训练张量把原始正常和 normal-control 标为 0、proxy 标为 1，并排除被替换原回波；生成器对象编号和 tracking 不进入 loss。正式监督张量未运行。

证据:
- src/train.py:623-721
- src/train.py:805-810

判断:
证据直接覆盖该要求，或排除性全仓库搜索已证明冲突路径不在当前正式路线；可判 PASS。该结论只覆盖本条，不外推到整个模块或科学主张。

需要修改:
无需修改；保留当前实现和直接证据，并继续确保该局部结论不被外推为模块或科学主张已经成立。

### [L00-08] 状态: PASS

要求:
不存在 tracking supervision。

发现:
训练张量把原始正常和 normal-control 标为 0、proxy 标为 1，并排除被替换原回波；生成器对象编号和 tracking 不进入 loss。正式监督张量未运行。

证据:
- src/train.py:623-721
- src/train.py:805-810

判断:
证据直接覆盖该要求，或排除性全仓库搜索已证明冲突路径不在当前正式路线；可判 PASS。该结论只覆盖本条，不外推到整个模块或科学主张。

需要修改:
无需修改；保留当前实现和直接证据，并继续确保该局部结论不被外推为模块或科学主张已经成立。

## L01：官方距离域

### [L01-01] 状态: PARTIAL

要求:
主分类 loss 只在 $2.5\le d\le50$m。

发现:
训练和自研评价代码都使用含端点的 2.5–50 m 范围，范围外点保留为输入但不进入 loss/metric；无正式优化器或官方评价运行。

证据:
- src/train.py:686-695
- src/evaluate.py:96-107

判断:
只能证明静态实现、局部运行或缺少 生成链路 的产物，尚不能证明可追溯正式路径完整采用；按审计规则不得升级为 PASS。

需要修改:
转为 PASS 所需条件：用同一 prediction 经训练掩码和官方评价域独立核对边界。

### [L01-02] 状态: PARTIAL

要求:
range 使用正确坐标/定义。

发现:
训练和自研评价代码都使用含端点的 2.5–50 m 范围，范围外点保留为输入但不进入 loss/metric；无正式优化器或官方评价运行。

证据:
- src/train.py:686-695
- src/evaluate.py:96-107

判断:
只能证明静态实现、局部运行或缺少 生成链路 的产物，尚不能证明可追溯正式路径完整采用；按审计规则不得升级为 PASS。

需要修改:
转为 PASS 所需条件：用同一 prediction 经训练掩码和官方评价域独立核对边界。

### [L01-03] 状态: PARTIAL

要求:
距离外点仍允许作为 context。

发现:
训练和自研评价代码都使用含端点的 2.5–50 m 范围，范围外点保留为输入但不进入 loss/metric；无正式优化器或官方评价运行。

证据:
- src/train.py:686-695
- src/evaluate.py:96-107

判断:
只能证明静态实现、局部运行或缺少 生成链路 的产物，尚不能证明可追溯正式路径完整采用；按审计规则不得升级为 PASS。

需要修改:
转为 PASS 所需条件：用同一 prediction 经训练掩码和官方评价域独立核对边界。

### [L01-04] 状态: PARTIAL

要求:
距离外点不进入 anomaly BCE。

发现:
训练和自研评价代码都使用含端点的 2.5–50 m 范围，范围外点保留为输入但不进入 loss/metric；无正式优化器或官方评价运行。

证据:
- src/train.py:686-695
- src/evaluate.py:96-107

判断:
只能证明静态实现、局部运行或缺少 生成链路 的产物，尚不能证明可追溯正式路径完整采用；按审计规则不得升级为 PASS。

需要修改:
转为 PASS 所需条件：用同一 prediction 经训练掩码和官方评价域独立核对边界。

### [L01-05] 状态: PARTIAL

要求:
evaluator 使用相同 official domain。

发现:
训练和自研评价代码都使用含端点的 2.5–50 m 范围，范围外点保留为输入但不进入 loss/metric；无正式优化器或官方评价运行。

证据:
- src/train.py:686-695
- src/evaluate.py:96-107

判断:
只能证明静态实现、局部运行或缺少 生成链路 的产物，尚不能证明可追溯正式路径完整采用；按审计规则不得升级为 PASS。

需要修改:
转为 PASS 所需条件：用同一 prediction 经训练掩码和官方评价域独立核对边界。

## L02：平衡二元交叉熵

### [L02-01] 状态: PARTIAL

要求:
正负类同时存在时，两类各自 mean。

发现:
损失对实际存在的正负类分别取 BCE 均值再等权，零正类夹具通过；零负类夹具缺失，正式训练从未运行。

证据:
- src/train.py:546-563
- test_ajae.py:548-564

判断:
只能证明静态实现、局部运行或缺少 生成链路 的产物，尚不能证明可追溯正式路径完整采用；按审计规则不得升级为 PASS。

需要修改:
转为 PASS 所需条件：补做 zero-negative 测试，并在真实优化步骤记录有效正负类。

### [L02-02] 状态: PARTIAL

要求:
两类各权重 $1/2$。

发现:
损失对实际存在的正负类分别取 BCE 均值再等权，零正类夹具通过；零负类夹具缺失，正式训练从未运行。

证据:
- src/train.py:546-563
- test_ajae.py:548-564

判断:
只能证明静态实现、局部运行或缺少 生成链路 的产物，尚不能证明可追溯正式路径完整采用；按审计规则不得升级为 PASS。

需要修改:
转为 PASS 所需条件：补做 zero-negative 测试，并在真实优化步骤记录有效正负类。

### [L02-03] 状态: PARTIAL

要求:
zero-positive window 不产生 NaN。

发现:
损失对实际存在的正负类分别取 BCE 均值再等权，零正类夹具通过；零负类夹具缺失，正式训练从未运行。

证据:
- src/train.py:546-563
- test_ajae.py:548-564

判断:
只能证明静态实现、局部运行或缺少 生成链路 的产物，尚不能证明可追溯正式路径完整采用；按审计规则不得升级为 PASS。

需要修改:
转为 PASS 所需条件：补做 zero-negative 测试，并在真实优化步骤记录有效正负类。

### [L02-04] 状态: PARTIAL

要求:
zero-positive 时只计算 negative mean。

发现:
损失对实际存在的正负类分别取 BCE 均值再等权，零正类夹具通过；零负类夹具缺失，正式训练从未运行。

证据:
- src/train.py:546-563
- test_ajae.py:548-564

判断:
只能证明静态实现、局部运行或缺少 生成链路 的产物，尚不能证明可追溯正式路径完整采用；按审计规则不得升级为 PASS。

需要修改:
转为 PASS 所需条件：补做 zero-negative 测试，并在真实优化步骤记录有效正负类。

### [L02-05] 状态: NOT RUN

要求:
zero-negative 时只计算 positive mean。

发现:
损失对实际存在的正负类分别取 BCE 均值再等权，零正类夹具通过；零负类夹具缺失，正式训练从未运行。

证据:
- src/train.py:546-563
- test_ajae.py:548-564

判断:
本条要求的运行级审计、实验或正式产物没有发生；代码存在也不能替代执行，判定 NOT RUN。

需要修改:
转为 PASS 所需条件：补做 zero-negative 测试，并在真实优化步骤记录有效正负类。

### [L02-06] 状态: PARTIAL

要求:
没有因为 anomaly points < 5 而丢弃训练窗口。

发现:
损失对实际存在的正负类分别取 BCE 均值再等权，零正类夹具通过；零负类夹具缺失，正式训练从未运行。

证据:
- src/train.py:546-563
- test_ajae.py:548-564

判断:
只能证明静态实现、局部运行或缺少 生成链路 的产物，尚不能证明可追溯正式路径完整采用；按审计规则不得升级为 PASS。

需要修改:
转为 PASS 所需条件：补做 zero-negative 测试，并在真实优化步骤记录有效正负类。

### [L02-07] 状态: PARTIAL

要求:
`<5 anomaly points` 规则只用于官方评价，不进入训练过滤。

发现:
损失对实际存在的正负类分别取 BCE 均值再等权，零正类夹具通过；零负类夹具缺失，正式训练从未运行。

证据:
- src/train.py:546-563
- test_ajae.py:548-564

判断:
只能证明静态实现、局部运行或缺少 生成链路 的产物，尚不能证明可追溯正式路径完整采用；按审计规则不得升级为 PASS。

需要修改:
转为 PASS 所需条件：补做 zero-negative 测试，并在真实优化步骤记录有效正负类。

### [L02-08] 状态: PARTIAL

要求:
BCE 输入使用 logits，避免错误重复 sigmoid。

发现:
损失对实际存在的正负类分别取 BCE 均值再等权，零正类夹具通过；零负类夹具缺失，正式训练从未运行。

证据:
- src/train.py:546-563
- test_ajae.py:548-564

判断:
只能证明静态实现、局部运行或缺少 生成链路 的产物，尚不能证明可追溯正式路径完整采用；按审计规则不得升级为 PASS。

需要修改:
转为 PASS 所需条件：补做 zero-negative 测试，并在真实优化步骤记录有效正负类。

## L03：第一版总损失

### [L03-01] 状态: PARTIAL

要求:
$L=L_{\mathrm{anom}}$。

发现:
训练代码唯一反向标量是 balanced BCE，协议加载器拒绝 EMA、`L_cf` 等旧字段，全仓库没有旧附加损失正式路径；实际 backward 因上游阻断未发生。

证据:
- src/train.py:908-987
- src/protocol.py:646-648
- repository-wide obsolete-route search

判断:
只能证明静态实现、局部运行或缺少 生成链路 的产物，尚不能证明可追溯正式路径完整采用；按审计规则不得升级为 PASS。

需要修改:
转为 PASS 所需条件：上游通过后执行一次真实 backward，确认只有 balanced BCE 产生 AJAE 梯度。

### [L03-02] 状态: PASS

要求:
没有 point-identity EMA memory。

发现:
训练代码唯一反向标量是 balanced BCE，协议加载器拒绝 EMA、`L_cf` 等旧字段，全仓库没有旧附加损失正式路径；实际 backward 因上游阻断未发生。

证据:
- src/train.py:908-987
- src/protocol.py:646-648
- repository-wide obsolete-route search

判断:
证据直接覆盖该要求，或排除性全仓库搜索已证明冲突路径不在当前正式路线；可判 PASS。该结论只覆盖本条，不外推到整个模块或科学主张。

需要修改:
无需修改；保留当前实现和直接证据，并继续确保该局部结论不被外推为模块或科学主张已经成立。

### [L03-03] 状态: PASS

要求:
没有 $L_{cf}$。

发现:
训练代码唯一反向标量是 balanced BCE，协议加载器拒绝 EMA、`L_cf` 等旧字段，全仓库没有旧附加损失正式路径；实际 backward 因上游阻断未发生。

证据:
- src/train.py:908-987
- src/protocol.py:646-648
- repository-wide obsolete-route search

判断:
证据直接覆盖该要求，或排除性全仓库搜索已证明冲突路径不在当前正式路线；可判 PASS。该结论只覆盖本条，不外推到整个模块或科学主张。

需要修改:
无需修改；保留当前实现和直接证据，并继续确保该局部结论不被外推为模块或科学主张已经成立。

### [L03-04] 状态: PASS

要求:
没有 explicit point smoothness loss。

发现:
训练代码唯一反向标量是 balanced BCE，协议加载器拒绝 EMA、`L_cf` 等旧字段，全仓库没有旧附加损失正式路径；实际 backward 因上游阻断未发生。

证据:
- src/train.py:908-987
- src/protocol.py:646-648
- repository-wide obsolete-route search

判断:
证据直接覆盖该要求，或排除性全仓库搜索已证明冲突路径不在当前正式路线；可判 PASS。该结论只覆盖本条，不外推到整个模块或科学主张。

需要修改:
无需修改；保留当前实现和直接证据，并继续确保该局部结论不被外推为模块或科学主张已经成立。

### [L03-05] 状态: PASS

要求:
没有 object ID loss。

发现:
训练代码唯一反向标量是 balanced BCE，协议加载器拒绝 EMA、`L_cf` 等旧字段，全仓库没有旧附加损失正式路径；实际 backward 因上游阻断未发生。

证据:
- src/train.py:908-987
- src/protocol.py:646-648
- repository-wide obsolete-route search

判断:
证据直接覆盖该要求，或排除性全仓库搜索已证明冲突路径不在当前正式路线；可判 PASS。该结论只覆盖本条，不外推到整个模块或科学主张。

需要修改:
无需修改；保留当前实现和直接证据，并继续确保该局部结论不被外推为模块或科学主张已经成立。

### [L03-06] 状态: PASS

要求:
没有 tracking loss。

发现:
训练代码唯一反向标量是 balanced BCE，协议加载器拒绝 EMA、`L_cf` 等旧字段，全仓库没有旧附加损失正式路径；实际 backward 因上游阻断未发生。

证据:
- src/train.py:908-987
- src/protocol.py:646-648
- repository-wide obsolete-route search

判断:
证据直接覆盖该要求，或排除性全仓库搜索已证明冲突路径不在当前正式路线；可判 PASS。该结论只覆盖本条，不外推到整个模块或科学主张。

需要修改:
无需修改；保留当前实现和直接证据，并继续确保该局部结论不被外推为模块或科学主张已经成立。

### [L03-07] 状态: PASS

要求:
没有 motion regression。

发现:
训练代码唯一反向标量是 balanced BCE，协议加载器拒绝 EMA、`L_cf` 等旧字段，全仓库没有旧附加损失正式路径；实际 backward 因上游阻断未发生。

证据:
- src/train.py:908-987
- src/protocol.py:646-648
- repository-wide obsolete-route search

判断:
证据直接覆盖该要求，或排除性全仓库搜索已证明冲突路径不在当前正式路线；可判 PASS。该结论只覆盖本条，不外推到整个模块或科学主张。

需要修改:
无需修改；保留当前实现和直接证据，并继续确保该局部结论不被外推为模块或科学主张已经成立。

### [L03-08] 状态: PASS

要求:
没有 reconstruction loss。

发现:
训练代码唯一反向标量是 balanced BCE，协议加载器拒绝 EMA、`L_cf` 等旧字段，全仓库没有旧附加损失正式路径；实际 backward 因上游阻断未发生。

证据:
- src/train.py:908-987
- src/protocol.py:646-648
- repository-wide obsolete-route search

判断:
证据直接覆盖该要求，或排除性全仓库搜索已证明冲突路径不在当前正式路线；可判 PASS。该结论只覆盖本条，不外推到整个模块或科学主张。

需要修改:
无需修改；保留当前实现和直接证据，并继续确保该局部结论不被外推为模块或科学主张已经成立。

### [L03-09] 状态: PASS

要求:
没有 contrastive auxiliary loss。

发现:
训练代码唯一反向标量是 balanced BCE，协议加载器拒绝 EMA、`L_cf` 等旧字段，全仓库没有旧附加损失正式路径；实际 backward 因上游阻断未发生。

证据:
- src/train.py:908-987
- src/protocol.py:646-648
- repository-wide obsolete-route search

判断:
证据直接覆盖该要求，或排除性全仓库搜索已证明冲突路径不在当前正式路线；可判 PASS。该结论只覆盖本条，不外推到整个模块或科学主张。

需要修改:
无需修改；保留当前实现和直接证据，并继续确保该局部结论不被外推为模块或科学主张已经成立。

### [L03-10] 状态: BLOCKED

要求:
optimizer 实际 backward 的 scalar loss 与上述一致。

发现:
训练代码唯一反向标量是 balanced BCE，协议加载器拒绝 EMA、`L_cf` 等旧字段，全仓库没有旧附加损失正式路径；实际 backward 因上游阻断未发生。

证据:
- src/train.py:908-987
- src/protocol.py:646-648
- repository-wide obsolete-route search

判断:
上游 Decision Gate 或冻结条件未满足，本条原则上不得继续；判定 BLOCKED。

需要修改:
转为 PASS 所需条件：上游通过后执行一次真实 backward，确认只有 balanced BCE 产生 AJAE 梯度。

### [L03-11] 状态: PASS

要求:
搜索旧 loss 代码，确认不能通过默认配置意外启用。

发现:
训练代码唯一反向标量是 balanced BCE，协议加载器拒绝 EMA、`L_cf` 等旧字段，全仓库没有旧附加损失正式路径；实际 backward 因上游阻断未发生。

证据:
- src/train.py:908-987
- src/protocol.py:646-648
- repository-wide obsolete-route search

判断:
证据直接覆盖该要求，或排除性全仓库搜索已证明冲突路径不在当前正式路线；可判 PASS。该结论只覆盖本条，不外推到整个模块或科学主张。

需要修改:
无需修改；保留当前实现和直接证据，并继续确保该局部结论不被外推为模块或科学主张已经成立。

## F00：稳定点身份

### [F00-01] 状态: PARTIAL

要求:
final measurement identity 是否定义为 `(frame, canonical ray)`。

发现:
`PointId=(frame_id,RayId)`，file slot 仅用于恢复，融合器按 frame+canonical ray 聚合；没有真实重叠窗口逐点身份检查或融合产物。

证据:
- src/scene.py:145-168
- src/evaluate.py:633-740

判断:
只能证明静态实现、局部运行或缺少 生成链路 的产物，尚不能证明可追溯正式路径完整采用；按审计规则不得升级为 PASS。

需要修改:
转为 PASS 所需条件：运行重叠窗口 point identity 一致性检查和真实融合。

### [F00-02] 状态: PARTIAL

要求:
canonical ray 是否为 `(beam, azimuth)`。

发现:
`PointId=(frame_id,RayId)`，file slot 仅用于恢复，融合器按 frame+canonical ray 聚合；没有真实重叠窗口逐点身份检查或融合产物。

证据:
- src/scene.py:145-168
- src/evaluate.py:633-740

判断:
只能证明静态实现、局部运行或缺少 生成链路 的产物，尚不能证明可追溯正式路径完整采用；按审计规则不得升级为 PASS。

需要修改:
转为 PASS 所需条件：运行重叠窗口 point identity 一致性检查和真实融合。

### [F00-03] 状态: PARTIAL

要求:
原始 slot 是否只用于 I/O recovery。

发现:
`PointId=(frame_id,RayId)`，file slot 仅用于恢复，融合器按 frame+canonical ray 聚合；没有真实重叠窗口逐点身份检查或融合产物。

证据:
- src/scene.py:145-168
- src/evaluate.py:633-740

判断:
只能证明静态实现、局部运行或缺少 生成链路 的产物，尚不能证明可追溯正式路径完整采用；按审计规则不得升级为 PASS。

需要修改:
转为 PASS 所需条件：运行重叠窗口 point identity 一致性检查和真实融合。

### [F00-04] 状态: PARTIAL

要求:
未审计的 raw slot 是否没有被直接作为物理 point ID。

发现:
`PointId=(frame_id,RayId)`，file slot 仅用于恢复，融合器按 frame+canonical ray 聚合；没有真实重叠窗口逐点身份检查或融合产物。

证据:
- src/scene.py:145-168
- src/evaluate.py:633-740

判断:
只能证明静态实现、局部运行或缺少 生成链路 的产物，尚不能证明可追溯正式路径完整采用；按审计规则不得升级为 PASS。

需要修改:
转为 PASS 所需条件：运行重叠窗口 point identity 一致性检查和真实融合。

### [F00-05] 状态: NOT RUN

要求:
同一物理量测点进入不同 window 时 identity 保持一致。

发现:
`PointId=(frame_id,RayId)`，file slot 仅用于恢复，融合器按 frame+canonical ray 聚合；没有真实重叠窗口逐点身份检查或融合产物。

证据:
- src/scene.py:145-168
- src/evaluate.py:633-740

判断:
本条要求的运行级审计、实验或正式产物没有发生；代码存在也不能替代执行，判定 NOT RUN。

需要修改:
转为 PASS 所需条件：运行重叠窗口 point identity 一致性检查和真实融合。

### [F00-06] 状态: BLOCKED

要求:
能否收集该点所有 window-level predictions。

发现:
`PointId=(frame_id,RayId)`，file slot 仅用于恢复，融合器按 frame+canonical ray 聚合；没有真实重叠窗口逐点身份检查或融合产物。

证据:
- src/scene.py:145-168
- src/evaluate.py:633-740

判断:
上游 Decision Gate 或冻结条件未满足，本条原则上不得继续；判定 BLOCKED。

需要修改:
转为 PASS 所需条件：运行重叠窗口 point identity 一致性检查和真实融合。

## F01：概率平均

### [F01-01] 状态: PARTIAL

要求:
模型输出 logits $z$。

发现:
推理代码先 sigmoid，再按 frame/ray 等权平均概率；没有 logit 平均、中心加权或 q 权重。现有测试直接输入概率，未检验固定 logits 的两种公式差异。

证据:
- src/evaluate.py:633-740
- src/evaluate.py:1035-1263
- test_ajae.py:634-647

判断:
只能证明静态实现、局部运行或缺少 生成链路 的产物，尚不能证明可追溯正式路径完整采用；按审计规则不得升级为 PASS。

需要修改:
转为 PASS 所需条件：用固定 logits 明确验证 mean(sigmoid(z))，再在正式 B4 产物核对。

### [F01-02] 状态: PARTIAL

要求:
每个 window prediction 先 sigmoid 为 probability。

发现:
推理代码先 sigmoid，再按 frame/ray 等权平均概率；没有 logit 平均、中心加权或 q 权重。现有测试直接输入概率，未检验固定 logits 的两种公式差异。

证据:
- src/evaluate.py:633-740
- src/evaluate.py:1035-1263
- test_ajae.py:634-647

判断:
只能证明静态实现、局部运行或缺少 生成链路 的产物，尚不能证明可追溯正式路径完整采用；按审计规则不得升级为 PASS。

需要修改:
转为 PASS 所需条件：用固定 logits 明确验证 mean(sigmoid(z))，再在正式 B4 产物核对。

### [F01-03] 状态: PARTIAL

要求:
最终融合平均 probability。

发现:
推理代码先 sigmoid，再按 frame/ray 等权平均概率；没有 logit 平均、中心加权或 q 权重。现有测试直接输入概率，未检验固定 logits 的两种公式差异。

证据:
- src/evaluate.py:633-740
- src/evaluate.py:1035-1263
- test_ajae.py:634-647

判断:
只能证明静态实现、局部运行或缺少 生成链路 的产物，尚不能证明可追溯正式路径完整采用；按审计规则不得升级为 PASS。

需要修改:
转为 PASS 所需条件：用固定 logits 明确验证 mean(sigmoid(z))，再在正式 B4 产物核对。

### [F01-04] 状态: PASS

要求:
没有 average logits。

发现:
推理代码先 sigmoid，再按 frame/ray 等权平均概率；没有 logit 平均、中心加权或 q 权重。现有测试直接输入概率，未检验固定 logits 的两种公式差异。

证据:
- src/evaluate.py:633-740
- src/evaluate.py:1035-1263
- test_ajae.py:634-647

判断:
证据直接覆盖该要求，或排除性全仓库搜索已证明冲突路径不在当前正式路线；可判 PASS。该结论只覆盖本条，不外推到整个模块或科学主张。

需要修改:
无需修改；保留当前实现和直接证据，并继续确保该局部结论不被外推为模块或科学主张已经成立。

### [F01-05] 状态: PASS

要求:
没有 center-weighted average。

发现:
推理代码先 sigmoid，再按 frame/ray 等权平均概率；没有 logit 平均、中心加权或 q 权重。现有测试直接输入概率，未检验固定 logits 的两种公式差异。

证据:
- src/evaluate.py:633-740
- src/evaluate.py:1035-1263
- test_ajae.py:634-647

判断:
证据直接覆盖该要求，或排除性全仓库搜索已证明冲突路径不在当前正式路线；可判 PASS。该结论只覆盖本条，不外推到整个模块或科学主张。

需要修改:
无需修改；保留当前实现和直接证据，并继续确保该局部结论不被外推为模块或科学主张已经成立。

### [F01-06] 状态: PASS

要求:
没有按 q 人工加权。

发现:
推理代码先 sigmoid，再按 frame/ray 等权平均概率；没有 logit 平均、中心加权或 q 权重。现有测试直接输入概率，未检验固定 logits 的两种公式差异。

证据:
- src/evaluate.py:633-740
- src/evaluate.py:1035-1263
- test_ajae.py:634-647

判断:
证据直接覆盖该要求，或排除性全仓库搜索已证明冲突路径不在当前正式路线；可判 PASS。该结论只覆盖本条，不外推到整个模块或科学主张。

需要修改:
无需修改；保留当前实现和直接证据，并继续确保该局部结论不被外推为模块或科学主张已经成立。

### [F01-07] 状态: NOT RUN

要求:
每个点 $1\le m_p\le5$。

发现:
推理代码先 sigmoid，再按 frame/ray 等权平均概率；没有 logit 平均、中心加权或 q 权重。现有测试直接输入概率，未检验固定 logits 的两种公式差异。

证据:
- src/evaluate.py:633-740
- src/evaluate.py:1035-1263
- test_ajae.py:634-647

判断:
本条要求的运行级审计、实验或正式产物没有发生；代码存在也不能替代执行，判定 NOT RUN。

需要修改:
转为 PASS 所需条件：用固定 logits 明确验证 mean(sigmoid(z))，再在正式 B4 产物核对。

### [F01-08] 状态: PASS

要求:
序列边缘只聚合真实完整窗口中的预测。

发现:
推理代码先 sigmoid，再按 frame/ray 等权平均概率；没有 logit 平均、中心加权或 q 权重。现有测试直接输入概率，未检验固定 logits 的两种公式差异。

证据:
- src/evaluate.py:633-740
- src/evaluate.py:1035-1263
- test_ajae.py:634-647

判断:
证据直接覆盖该要求，或排除性全仓库搜索已证明冲突路径不在当前正式路线；可判 PASS。该结论只覆盖本条，不外推到整个模块或科学主张。

需要修改:
无需修改；保留当前实现和直接证据，并继续确保该局部结论不被外推为模块或科学主张已经成立。

### [F01-09] 状态: PASS

要求:
没有 padding。

发现:
推理代码先 sigmoid，再按 frame/ray 等权平均概率；没有 logit 平均、中心加权或 q 权重。现有测试直接输入概率，未检验固定 logits 的两种公式差异。

证据:
- src/evaluate.py:633-740
- src/evaluate.py:1035-1263
- test_ajae.py:634-647

判断:
证据直接覆盖该要求，或排除性全仓库搜索已证明冲突路径不在当前正式路线；可判 PASS。该结论只覆盖本条，不外推到整个模块或科学主张。

需要修改:
无需修改；保留当前实现和直接证据，并继续确保该局部结论不被外推为模块或科学主张已经成立。

### [F01-10] 状态: PASS

要求:
没有 repeated frames。

发现:
推理代码先 sigmoid，再按 frame/ray 等权平均概率；没有 logit 平均、中心加权或 q 权重。现有测试直接输入概率，未检验固定 logits 的两种公式差异。

证据:
- src/evaluate.py:633-740
- src/evaluate.py:1035-1263
- test_ajae.py:634-647

判断:
证据直接覆盖该要求，或排除性全仓库搜索已证明冲突路径不在当前正式路线；可判 PASS。该结论只覆盖本条，不外推到整个模块或科学主张。

需要修改:
无需修改；保留当前实现和直接证据，并继续确保该局部结论不被外推为模块或科学主张已经成立。

### [F01-11] 状态: PASS

要求:
没有 mirrored frames。

发现:
推理代码先 sigmoid，再按 frame/ray 等权平均概率；没有 logit 平均、中心加权或 q 权重。现有测试直接输入概率，未检验固定 logits 的两种公式差异。

证据:
- src/evaluate.py:633-740
- src/evaluate.py:1035-1263
- test_ajae.py:634-647

判断:
证据直接覆盖该要求，或排除性全仓库搜索已证明冲突路径不在当前正式路线；可判 PASS。该结论只覆盖本条，不外推到整个模块或科学主张。

需要修改:
无需修改；保留当前实现和直接证据，并继续确保该局部结论不被外推为模块或科学主张已经成立。

### [F01-12] 状态: PASS

要求:
没有把平均解释为 Bayesian independent evidence fusion。

发现:
推理代码先 sigmoid，再按 frame/ray 等权平均概率；没有 logit 平均、中心加权或 q 权重。现有测试直接输入概率，未检验固定 logits 的两种公式差异。

证据:
- src/evaluate.py:633-740
- src/evaluate.py:1035-1263
- test_ajae.py:634-647

判断:
证据直接覆盖该要求，或排除性全仓库搜索已证明冲突路径不在当前正式路线；可判 PASS。该结论只覆盖本条，不外推到整个模块或科学主张。

需要修改:
无需修改；保留当前实现和直接证据，并继续确保该局部结论不被外推为模块或科学主张已经成立。

## F02：时间位置校准

### [F02-01] 状态: NOT RUN

要求:
q=-2 AP。

发现:
时间位置校准的接口和校准文件存在，但 B1/B3 dev predictions、拟合结果、冻结参数和 B4/B5 复用均未真实执行。

证据:
- src/evaluate.py:253-346
- src/evaluate.py:609-740
- 全仓库无 q-position calibration 结果

判断:
本条要求的运行级审计、实验或正式产物没有发生；代码存在也不能替代执行，判定 NOT RUN。

需要修改:
转为 PASS 所需条件：完成 B1/B3 后仅在 201 拟合并冻结 q 校准，再验证 B4/B5 复用。

### [F02-02] 状态: NOT RUN

要求:
q=-1 AP。

发现:
时间位置校准的接口和校准文件存在，但 B1/B3 dev predictions、拟合结果、冻结参数和 B4/B5 复用均未真实执行。

证据:
- src/evaluate.py:253-346
- src/evaluate.py:609-740
- 全仓库无 q-position calibration 结果

判断:
本条要求的运行级审计、实验或正式产物没有发生；代码存在也不能替代执行，判定 NOT RUN。

需要修改:
转为 PASS 所需条件：完成 B1/B3 后仅在 201 拟合并冻结 q 校准，再验证 B4/B5 复用。

### [F02-03] 状态: NOT RUN

要求:
q=0 AP。

发现:
时间位置校准的接口和校准文件存在，但 B1/B3 dev predictions、拟合结果、冻结参数和 B4/B5 复用均未真实执行。

证据:
- src/evaluate.py:253-346
- src/evaluate.py:609-740
- 全仓库无 q-position calibration 结果

判断:
本条要求的运行级审计、实验或正式产物没有发生；代码存在也不能替代执行，判定 NOT RUN。

需要修改:
转为 PASS 所需条件：完成 B1/B3 后仅在 201 拟合并冻结 q 校准，再验证 B4/B5 复用。

### [F02-04] 状态: NOT RUN

要求:
q=+1 AP。

发现:
时间位置校准的接口和校准文件存在，但 B1/B3 dev predictions、拟合结果、冻结参数和 B4/B5 复用均未真实执行。

证据:
- src/evaluate.py:253-346
- src/evaluate.py:609-740
- 全仓库无 q-position calibration 结果

判断:
本条要求的运行级审计、实验或正式产物没有发生；代码存在也不能替代执行，判定 NOT RUN。

需要修改:
转为 PASS 所需条件：完成 B1/B3 后仅在 201 拟合并冻结 q 校准，再验证 B4/B5 复用。

### [F02-05] 状态: NOT RUN

要求:
q=+2 AP。

发现:
时间位置校准的接口和校准文件存在，但 B1/B3 dev predictions、拟合结果、冻结参数和 B4/B5 复用均未真实执行。

证据:
- src/evaluate.py:253-346
- src/evaluate.py:609-740
- 全仓库无 q-position calibration 结果

判断:
本条要求的运行级审计、实验或正式产物没有发生；代码存在也不能替代执行，判定 NOT RUN。

需要修改:
转为 PASS 所需条件：完成 B1/B3 后仅在 201 拟合并冻结 q 校准，再验证 B4/B5 复用。

### [F02-06] 状态: NOT RUN

要求:
每个 q 的 normal-point mean score。

发现:
时间位置校准的接口和校准文件存在，但 B1/B3 dev predictions、拟合结果、冻结参数和 B4/B5 复用均未真实执行。

证据:
- src/evaluate.py:253-346
- src/evaluate.py:609-740
- 全仓库无 q-position calibration 结果

判断:
本条要求的运行级审计、实验或正式产物没有发生；代码存在也不能替代执行，判定 NOT RUN。

需要修改:
转为 PASS 所需条件：完成 B1/B3 后仅在 201 拟合并冻结 q 校准，再验证 B4/B5 复用。

### [F02-07] 状态: NOT RUN

要求:
每个 q 的 anomaly-point mean score。

发现:
时间位置校准的接口和校准文件存在，但 B1/B3 dev predictions、拟合结果、冻结参数和 B4/B5 复用均未真实执行。

证据:
- src/evaluate.py:253-346
- src/evaluate.py:609-740
- 全仓库无 q-position calibration 结果

判断:
本条要求的运行级审计、实验或正式产物没有发生；代码存在也不能替代执行，判定 NOT RUN。

需要修改:
转为 PASS 所需条件：完成 B1/B3 后仅在 201 拟合并冻结 q 校准，再验证 B4/B5 复用。

### [F02-08] 状态: NOT RUN

要求:
每个 q 的 score distribution / scale。

发现:
时间位置校准的接口和校准文件存在，但 B1/B3 dev predictions、拟合结果、冻结参数和 B4/B5 复用均未真实执行。

证据:
- src/evaluate.py:253-346
- src/evaluate.py:609-740
- 全仓库无 q-position calibration 结果

判断:
本条要求的运行级审计、实验或正式产物没有发生；代码存在也不能替代执行，判定 NOT RUN。

需要修改:
转为 PASS 所需条件：完成 B1/B3 后仅在 201 拟合并冻结 q 校准，再验证 B4/B5 复用。

### [F02-09] 状态: NOT RUN

要求:
是否检查 q 位置偏置。

发现:
时间位置校准的接口和校准文件存在，但 B1/B3 dev predictions、拟合结果、冻结参数和 B4/B5 复用均未真实执行。

证据:
- src/evaluate.py:253-346
- src/evaluate.py:609-740
- 全仓库无 q-position calibration 结果

判断:
本条要求的运行级审计、实验或正式产物没有发生；代码存在也不能替代执行，判定 NOT RUN。

需要修改:
转为 PASS 所需条件：完成 B1/B3 后仅在 201 拟合并冻结 q 校准，再验证 B4/B5 复用。

### [F02-10] 状态: BLOCKED

要求:
若 q calibration 明显不一致，是否禁止直接依赖 B4 fusion。

发现:
时间位置校准的接口和校准文件存在，但 B1/B3 dev predictions、拟合结果、冻结参数和 B4/B5 复用均未真实执行。

证据:
- src/evaluate.py:253-346
- src/evaluate.py:609-740
- 全仓库无 q-position calibration 结果

判断:
上游 Decision Gate 或冻结条件未满足，本条原则上不得继续；判定 BLOCKED。

需要修改:
转为 PASS 所需条件：完成 B1/B3 后仅在 201 拟合并冻结 q 校准，再验证 B4/B5 复用。

### [F02-11] 状态: NOT RUN

要求:
如果该诊断尚未真实运行，B4 的融合主张标 NOT RUN。

发现:
时间位置校准的接口和校准文件存在，但 B1/B3 dev predictions、拟合结果、冻结参数和 B4/B5 复用均未真实执行。

证据:
- src/evaluate.py:253-346
- src/evaluate.py:609-740
- 全仓库无 q-position calibration 结果

判断:
本条要求的运行级审计、实验或正式产物没有发生；代码存在也不能替代执行，判定 NOT RUN。

需要修改:
转为 PASS 所需条件：完成 B1/B3 后仅在 201 拟合并冻结 q 校准，再验证 B4/B5 复用。

## B00：B0

### [B00-01] 状态: BLOCKED

要求:
是否实现冻结 STU 单帧参考。

发现:
B0 冻结 STU 单帧 MaxLogit 路径有代码，但固定 201 权威评价器未绑定，且没有配置、预测、指标或提交产物。Gate1 失败使正式运行被阻断。

证据:
- src/train.py:196-203
- src/evaluate.py:1065-1090
- src/train.py:1674-1708

判断:
上游 Decision Gate 或冻结条件未满足，本条原则上不得继续；判定 BLOCKED。

需要修改:
保持 BLOCKED；只有 Gate1 通过且 fixed-201 权威评价器冻结后才可运行并保存完整产物。

### [B00-02] 状态: BLOCKED

要求:
是否使用 MaxLogit 或官方可复现 STU OOD 分数。

发现:
B0 冻结 STU 单帧 MaxLogit 路径有代码，但固定 201 权威评价器未绑定，且没有配置、预测、指标或提交产物。Gate1 失败使正式运行被阻断。

证据:
- src/train.py:196-203
- src/evaluate.py:1065-1090
- src/train.py:1674-1708

判断:
上游 Decision Gate 或冻结条件未满足，本条原则上不得继续；判定 BLOCKED。

需要修改:
保持 BLOCKED；只有 Gate1 通过且 fixed-201 权威评价器冻结后才可运行并保存完整产物。

### [B00-03] 状态: BLOCKED

要求:
是否使用完全相同 evaluator。

发现:
`src/train.py:1674-1708` 的 `_AUTHORITATIVE_DEVELOPMENT_EVALUATOR` 明确为 `None`，固定 201 权威评价没有绑定。

证据:
- src/train.py:196-203
- src/evaluate.py:1065-1090
- src/train.py:1674-1708

判断:
上游 Decision Gate 或冻结条件未满足，本条原则上不得继续；判定 BLOCKED。

需要修改:
保持 BLOCKED；只有 Gate1 通过且 fixed-201 权威评价器冻结后才可运行并保存完整产物。

### [B00-04] 状态: BLOCKED

要求:
是否保存完整预测和指标。

发现:
B0 冻结 STU 单帧 MaxLogit 路径有代码，但固定 201 权威评价器未绑定，且没有配置、预测、指标或提交产物。Gate1 失败使正式运行被阻断。

证据:
- src/train.py:196-203
- src/evaluate.py:1065-1090
- src/train.py:1674-1708

判断:
上游 Decision Gate 或冻结条件未满足，本条原则上不得继续；判定 BLOCKED。

需要修改:
保持 BLOCKED；只有 Gate1 通过且 fixed-201 权威评价器冻结后才可运行并保存完整产物。

### [B00-05] 状态: BLOCKED

要求:
是否记录 commit/config。

发现:
B0 冻结 STU 单帧 MaxLogit 路径有代码，但固定 201 权威评价器未绑定，且没有配置、预测、指标或提交产物。Gate1 失败使正式运行被阻断。

证据:
- src/train.py:196-203
- src/evaluate.py:1065-1090
- src/train.py:1674-1708

判断:
上游 Decision Gate 或冻结条件未满足，本条原则上不得继续；判定 BLOCKED。

需要修改:
保持 BLOCKED；只有 Gate1 通过且 fixed-201 权威评价器冻结后才可运行并保存完整产物。

## B01：B1

### [B01-01] 状态: BLOCKED

要求:
B1 是否真实实现。

发现:
B1 单帧 AJAE 配置和禁用 temporal neighbor 的路径存在；没有训练、固定 201、pure-normal 或 normal-control 安全性结果。

证据:
- src/train.py:201-203
- src/evaluate.py:1091-1111
- runs/ajae（无 B1 产物）

判断:
上游 Decision Gate 或冻结条件未满足，本条原则上不得继续；判定 BLOCKED。

需要修改:
保持 BLOCKED；只有 Gate1 通过后才可运行 B1 与固定 201 安全性比较。

### [B01-02] 状态: BLOCKED

要求:
B1 与主模型使用同一 STU point interface。

发现:
B1 单帧 AJAE 配置和禁用 temporal neighbor 的路径存在；没有训练、固定 201、pure-normal 或 normal-control 安全性结果。

证据:
- src/train.py:201-203
- src/evaluate.py:1091-1111
- runs/ajae（无 B1 产物）

判断:
上游 Decision Gate 或冻结条件未满足，本条原则上不得继续；判定 BLOCKED。

需要修改:
保持 BLOCKED；只有 Gate1 通过后才可运行 B1 与固定 201 安全性比较。

### [B01-03] 状态: BLOCKED

要求:
B1 使用同一 generator。

发现:
B1 单帧 AJAE 配置和禁用 temporal neighbor 的路径存在；没有训练、固定 201、pure-normal 或 normal-control 安全性结果。

证据:
- src/train.py:201-203
- src/evaluate.py:1091-1111
- runs/ajae（无 B1 产物）

判断:
上游 Decision Gate 或冻结条件未满足，本条原则上不得继续；判定 BLOCKED。

需要修改:
保持 BLOCKED；只有 Gate1 通过后才可运行 B1 与固定 201 安全性比较。

### [B01-04] 状态: BLOCKED

要求:
B1 使用同一训练数据角色。

发现:
B1 单帧 AJAE 配置和禁用 temporal neighbor 的路径存在；没有训练、固定 201、pure-normal 或 normal-control 安全性结果。

证据:
- src/train.py:201-203
- src/evaluate.py:1091-1111
- runs/ajae（无 B1 产物）

判断:
上游 Decision Gate 或冻结条件未满足，本条原则上不得继续；判定 BLOCKED。

需要修改:
保持 BLOCKED；只有 Gate1 通过后才可运行 B1 与固定 201 安全性比较。

### [B01-05] 状态: BLOCKED

要求:
B1 不读取 temporal neighbors。

发现:
B1 单帧 AJAE 配置和禁用 temporal neighbor 的路径存在；没有训练、固定 201、pure-normal 或 normal-control 安全性结果。

证据:
- src/train.py:201-203
- src/evaluate.py:1091-1111
- runs/ajae（无 B1 产物）

判断:
上游 Decision Gate 或冻结条件未满足，本条原则上不得继续；判定 BLOCKED。

需要修改:
保持 BLOCKED；只有 Gate1 通过后才可运行 B1 与固定 201 安全性比较。

### [B01-06] 状态: BLOCKED

要求:
B1 201 固定开发结果存在。

发现:
B1 单帧 AJAE 配置和禁用 temporal neighbor 的路径存在；没有训练、固定 201、pure-normal 或 normal-control 安全性结果。

证据:
- src/train.py:201-203
- src/evaluate.py:1091-1111
- runs/ajae（无 B1 产物）

判断:
上游 Decision Gate 或冻结条件未满足，本条原则上不得继续；判定 BLOCKED。

需要修改:
保持 BLOCKED；只有 Gate1 通过后才可运行 B1 与固定 201 安全性比较。

### [B01-07] 状态: BLOCKED

要求:
B1 pure-normal 201 结果存在。

发现:
B1 单帧 AJAE 配置和禁用 temporal neighbor 的路径存在；没有训练、固定 201、pure-normal 或 normal-control 安全性结果。

证据:
- src/train.py:201-203
- src/evaluate.py:1091-1111
- runs/ajae（无 B1 产物）

判断:
上游 Decision Gate 或冻结条件未满足，本条原则上不得继续；判定 BLOCKED。

需要修改:
保持 BLOCKED；只有 Gate1 通过后才可运行 B1 与固定 201 安全性比较。

### [B01-08] 状态: BLOCKED

要求:
B1 normal-control 结果存在。

发现:
B1 单帧 AJAE 配置和禁用 temporal neighbor 的路径存在；没有训练、固定 201、pure-normal 或 normal-control 安全性结果。

证据:
- src/train.py:201-203
- src/evaluate.py:1091-1111
- runs/ajae（无 B1 产物）

判断:
上游 Decision Gate 或冻结条件未满足，本条原则上不得继续；判定 BLOCKED。

需要修改:
保持 BLOCKED；只有 Gate1 通过后才可运行 B1 与固定 201 安全性比较。

## B02：B2

### [B02-01] 状态: BLOCKED

要求:
B2 是否与 B3 参数量/主体结构尽可能一致。

发现:
五帧全监督、关闭跨帧边并只评价 q=0 的对照路径存在；无正式配置、检查点或三种子结果。

证据:
- src/train.py:204-212
- src/evaluate.py:1112-1150
- runs/ajae（无 B2 产物）

判断:
上游 Decision Gate 或冻结条件未满足，本条原则上不得继续；判定 BLOCKED。

需要修改:
保持 BLOCKED；先完成 Gate2，再按三种子协议运行。

### [B02-02] 状态: BLOCKED

要求:
所有 cross-frame edges 确实被关闭。

发现:
五帧全监督、关闭跨帧边并只评价 q=0 的对照路径存在；无正式配置、检查点或三种子结果。

证据:
- src/train.py:204-212
- src/evaluate.py:1112-1150
- runs/ajae（无 B2 产物）

判断:
上游 Decision Gate 或冻结条件未满足，本条原则上不得继续；判定 BLOCKED。

需要修改:
保持 BLOCKED；先完成 Gate2，再按三种子协议运行。

### [B02-03] 状态: BLOCKED

要求:
不是把输入直接改成单帧从而变成 B1。

发现:
五帧全监督、关闭跨帧边并只评价 q=0 的对照路径存在；无正式配置、检查点或三种子结果。

证据:
- src/train.py:204-212
- src/evaluate.py:1112-1150
- runs/ajae（无 B2 产物）

判断:
上游 Decision Gate 或冻结条件未满足，本条原则上不得继续；判定 BLOCKED。

需要修改:
保持 BLOCKED；先完成 Gate2，再按三种子协议运行。

### [B02-04] 状态: BLOCKED

要求:
final evaluation 确实只取 q=0。

发现:
五帧全监督、关闭跨帧边并只评价 q=0 的对照路径存在；无正式配置、检查点或三种子结果。

证据:
- src/train.py:204-212
- src/evaluate.py:1112-1150
- runs/ajae（无 B2 产物）

判断:
上游 Decision Gate 或冻结条件未满足，本条原则上不得继续；判定 BLOCKED。

需要修改:
保持 BLOCKED；先完成 Gate2，再按三种子协议运行。

### [B02-05] 状态: BLOCKED

要求:
保存 B2 完整运行配置。

发现:
五帧全监督、关闭跨帧边并只评价 q=0 的对照路径存在；无正式配置、检查点或三种子结果。

证据:
- src/train.py:204-212
- src/evaluate.py:1112-1150
- runs/ajae（无 B2 产物）

判断:
上游 Decision Gate 或冻结条件未满足，本条原则上不得继续；判定 BLOCKED。

需要修改:
保持 BLOCKED；先完成 Gate2，再按三种子协议运行。

### [B02-06] 状态: BLOCKED

要求:
多 seed 结果存在。

发现:
五帧全监督、关闭跨帧边并只评价 q=0 的对照路径存在；无正式配置、检查点或三种子结果。

证据:
- src/train.py:204-212
- src/evaluate.py:1112-1150
- runs/ajae（无 B2 产物）

判断:
上游 Decision Gate 或冻结条件未满足，本条原则上不得继续；判定 BLOCKED。

需要修改:
保持 BLOCKED；先完成 Gate2，再按三种子协议运行。

## B03：B3

### [B03-01] 状态: BLOCKED

要求:
cross-frame attention 正常开启。

发现:
跨帧注意力、时间门和 q=0 输出路径存在；无固定 201 或三种子结果。

证据:
- src/train.py:213-221
- src/evaluate.py:1112-1150
- runs/ajae（无 B3 产物）

判断:
上游 Decision Gate 或冻结条件未满足，本条原则上不得继续；判定 BLOCKED。

需要修改:
保持 BLOCKED；先完成 Gate2，再与 B1/B2 做三种子配对比较。

### [B03-02] 状态: BLOCKED

要求:
temporal gates 正常开启。

发现:
跨帧注意力、时间门和 q=0 输出路径存在；无固定 201 或三种子结果。

证据:
- src/train.py:213-221
- src/evaluate.py:1112-1150
- runs/ajae（无 B3 产物）

判断:
上游 Decision Gate 或冻结条件未满足，本条原则上不得继续；判定 BLOCKED。

需要修改:
保持 BLOCKED；先完成 Gate2，再与 B1/B2 做三种子配对比较。

### [B03-03] 状态: BLOCKED

要求:
evaluation 不使用 multi-window fusion。

发现:
跨帧注意力、时间门和 q=0 输出路径存在；无固定 201 或三种子结果。

证据:
- src/train.py:213-221
- src/evaluate.py:1112-1150
- runs/ajae（无 B3 产物）

判断:
上游 Decision Gate 或冻结条件未满足，本条原则上不得继续；判定 BLOCKED。

需要修改:
保持 BLOCKED；先完成 Gate2，再与 B1/B2 做三种子配对比较。

### [B03-04] 状态: BLOCKED

要求:
每个 frame 只使用 center-window q=0 prediction。

发现:
跨帧注意力、时间门和 q=0 输出路径存在；无固定 201 或三种子结果。

证据:
- src/train.py:213-221
- src/evaluate.py:1112-1150
- runs/ajae（无 B3 产物）

判断:
上游 Decision Gate 或冻结条件未满足，本条原则上不得继续；判定 BLOCKED。

需要修改:
保持 BLOCKED；先完成 Gate2，再与 B1/B2 做三种子配对比较。

### [B03-05] 状态: BLOCKED

要求:
201 固定开发结果存在。

发现:
跨帧注意力、时间门和 q=0 输出路径存在；无固定 201 或三种子结果。

证据:
- src/train.py:213-221
- src/evaluate.py:1112-1150
- runs/ajae（无 B3 产物）

判断:
上游 Decision Gate 或冻结条件未满足，本条原则上不得继续；判定 BLOCKED。

需要修改:
保持 BLOCKED；先完成 Gate2，再与 B1/B2 做三种子配对比较。

### [B03-06] 状态: BLOCKED

要求:
多 seed 结果存在。

发现:
跨帧注意力、时间门和 q=0 输出路径存在；无固定 201 或三种子结果。

证据:
- src/train.py:213-221
- src/evaluate.py:1112-1150
- runs/ajae（无 B3 产物）

判断:
上游 Decision Gate 或冻结条件未满足，本条原则上不得继续；判定 BLOCKED。

需要修改:
保持 BLOCKED；先完成 Gate2，再与 B1/B2 做三种子配对比较。

## B04：B4

### [B04-01] 状态: BLOCKED

要求:
使用与 B3 完全相同 checkpoint/model。

发现:
代码要求从 B3 加载冻结权重并按 canonical point identity 融合最多五个窗口概率；没有 B3/B4 配对结果。

证据:
- src/train.py:222-230
- src/evaluate.py:1045-1150
- runs/ajae（无 B4 产物）

判断:
上游 Decision Gate 或冻结条件未满足，本条原则上不得继续；判定 BLOCKED。

需要修改:
保持 BLOCKED；只能复用已冻结 B3 权重并单独判定窗口融合贡献。

### [B04-02] 状态: BLOCKED

要求:
唯一增加的是 overlapping-window probability averaging。

发现:
代码要求从 B3 加载冻结权重并按 canonical point identity 融合最多五个窗口概率；没有 B3/B4 配对结果。

证据:
- src/train.py:222-230
- src/evaluate.py:1045-1150
- runs/ajae（无 B4 产物）

判断:
上游 Decision Gate 或冻结条件未满足，本条原则上不得继续；判定 BLOCKED。

需要修改:
保持 BLOCKED；只能复用已冻结 B3 权重并单独判定窗口融合贡献。

### [B04-03] 状态: BLOCKED

要求:
未重新训练一个“B4 model”。

发现:
代码要求从 B3 加载冻结权重并按 canonical point identity 融合最多五个窗口概率；没有 B3/B4 配对结果。

证据:
- src/train.py:222-230
- src/evaluate.py:1045-1150
- runs/ajae（无 B4 产物）

判断:
上游 Decision Gate 或冻结条件未满足，本条原则上不得继续；判定 BLOCKED。

需要修改:
保持 BLOCKED；只能复用已冻结 B3 权重并单独判定窗口融合贡献。

### [B04-04] 状态: BLOCKED

要求:
multi-window aggregation 实际最多 5 predictions。

发现:
代码要求从 B3 加载冻结权重并按 canonical point identity 融合最多五个窗口概率；没有 B3/B4 配对结果。

证据:
- src/train.py:222-230
- src/evaluate.py:1045-1150
- runs/ajae（无 B4 产物）

判断:
上游 Decision Gate 或冻结条件未满足，本条原则上不得继续；判定 BLOCKED。

需要修改:
保持 BLOCKED；只能复用已冻结 B3 权重并单独判定窗口融合贡献。

### [B04-05] 状态: BLOCKED

要求:
B4 结果与 B3 可直接配对比较。

发现:
代码要求从 B3 加载冻结权重并按 canonical point identity 融合最多五个窗口概率；没有 B3/B4 配对结果。

证据:
- src/train.py:222-230
- src/evaluate.py:1045-1150
- runs/ajae（无 B4 产物）

判断:
上游 Decision Gate 或冻结条件未满足，本条原则上不得继续；判定 BLOCKED。

需要修改:
保持 BLOCKED；只能复用已冻结 B3 权重并单独判定窗口融合贡献。

## B05：B5 因果设置

### [B05-01] 状态: BLOCKED

要求:
causal input 为 $[t-4,t]$。

发现:
因果 `[t-4,t]`、五帧训练监督、当前帧评价和延迟记录接口存在；没有性能或延迟产物。

证据:
- src/train.py:232-240
- src/evaluate.py:1151-1263
- runs/ajae（无 B5 产物）

判断:
上游 Decision Gate 或冻结条件未满足，本条原则上不得继续；判定 BLOCKED。

需要修改:
保持 BLOCKED；主开发结论成立后再测因果性能与延迟代价。

### [B05-02] 状态: BLOCKED

要求:
只输出当前帧。

发现:
因果 `[t-4,t]`、五帧训练监督、当前帧评价和延迟记录接口存在；没有性能或延迟产物。

证据:
- src/train.py:232-240
- src/evaluate.py:1151-1263
- runs/ajae（无 B5 产物）

判断:
上游 Decision Gate 或冻结条件未满足，本条原则上不得继续；判定 BLOCKED。

需要修改:
保持 BLOCKED；主开发结论成立后再测因果性能与延迟代价。

### [B05-03] 状态: BLOCKED

要求:
没有访问未来帧。

发现:
因果 `[t-4,t]`、五帧训练监督、当前帧评价和延迟记录接口存在；没有性能或延迟产物。

证据:
- src/train.py:232-240
- src/evaluate.py:1151-1263
- runs/ajae（无 B5 产物）

判断:
上游 Decision Gate 或冻结条件未满足，本条原则上不得继续；判定 BLOCKED。

需要修改:
保持 BLOCKED；主开发结论成立后再测因果性能与延迟代价。

### [B05-04] 状态: BLOCKED

要求:
单独保存配置。

发现:
因果 `[t-4,t]`、五帧训练监督、当前帧评价和延迟记录接口存在；没有性能或延迟产物。

证据:
- src/train.py:232-240
- src/evaluate.py:1151-1263
- runs/ajae（无 B5 产物）

判断:
上游 Decision Gate 或冻结条件未满足，本条原则上不得继续；判定 BLOCKED。

需要修改:
保持 BLOCKED；主开发结论成立后再测因果性能与延迟代价。

### [B05-05] 状态: BLOCKED

要求:
不与主模型混淆。

发现:
因果 `[t-4,t]`、五帧训练监督、当前帧评价和延迟记录接口存在；没有性能或延迟产物。

证据:
- src/train.py:232-240
- src/evaluate.py:1151-1263
- runs/ajae（无 B5 产物）

判断:
上游 Decision Gate 或冻结条件未满足，本条原则上不得继续；判定 BLOCKED。

需要修改:
保持 BLOCKED；主开发结论成立后再测因果性能与延迟代价。

### [B05-06] 状态: BLOCKED

要求:
报告性能。

发现:
因果 `[t-4,t]`、五帧训练监督、当前帧评价和延迟记录接口存在；没有性能或延迟产物。

证据:
- src/train.py:232-240
- src/evaluate.py:1151-1263
- runs/ajae（无 B5 产物）

判断:
上游 Decision Gate 或冻结条件未满足，本条原则上不得继续；判定 BLOCKED。

需要修改:
保持 BLOCKED；主开发结论成立后再测因果性能与延迟代价。

### [B05-07] 状态: BLOCKED

要求:
报告延迟。

发现:
因果 `[t-4,t]`、五帧训练监督、当前帧评价和延迟记录接口存在；没有性能或延迟产物。

证据:
- src/train.py:232-240
- src/evaluate.py:1151-1263
- runs/ajae（无 B5 产物）

判断:
上游 Decision Gate 或冻结条件未满足，本条原则上不得继续；判定 BLOCKED。

需要修改:
保持 BLOCKED；主开发结论成立后再测因果性能与延迟代价。

## B06：主张判定

### [B06-01] 状态: NOT RUN

要求:
是否真实满足 $B3>B1$。

发现:
B0–B5 没有正式实验，因此 B3>B1、B3>B2、B4>B3、因果代价和任何科学主张均未测试。

证据:
- runs/ajae（只有 calibration.pt）
- src/train.py:2101-2328

判断:
本条要求的运行级审计、实验或正式产物没有发生；代码存在也不能替代执行，判定 NOT RUN。

需要修改:
需要 B0–B5 的冻结、多种子、可追溯结果后才能判定任何主张。

### [B06-02] 状态: NOT RUN

要求:
是否真实满足 $B3>B2$。

发现:
B0–B5 没有正式实验，因此 B3>B1、B3>B2、B4>B3、因果代价和任何科学主张均未测试。

证据:
- runs/ajae（只有 calibration.pt）
- src/train.py:2101-2328

判断:
本条要求的运行级审计、实验或正式产物没有发生；代码存在也不能替代执行，判定 NOT RUN。

需要修改:
需要 B0–B5 的冻结、多种子、可追溯结果后才能判定任何主张。

### [B06-03] 状态: NOT RUN

要求:
如果不满足，是否正确判定 cross-frame claim 未成立。

发现:
B0–B5 没有正式实验，因此 B3>B1、B3>B2、B4>B3、因果代价和任何科学主张均未测试。

证据:
- runs/ajae（只有 calibration.pt）
- src/train.py:2101-2328

判断:
本条要求的运行级审计、实验或正式产物没有发生；代码存在也不能替代执行，判定 NOT RUN。

需要修改:
需要 B0–B5 的冻结、多种子、可追溯结果后才能判定任何主张。

### [B06-04] 状态: NOT RUN

要求:
是否满足 $B4>B3$。

发现:
B0–B5 没有正式实验，因此 B3>B1、B3>B2、B4>B3、因果代价和任何科学主张均未测试。

证据:
- runs/ajae（只有 calibration.pt）
- src/train.py:2101-2328

判断:
本条要求的运行级审计、实验或正式产物没有发生；代码存在也不能替代执行，判定 NOT RUN。

需要修改:
需要 B0–B5 的冻结、多种子、可追溯结果后才能判定任何主张。

### [B06-05] 状态: NOT RUN

要求:
如果不满足，是否没有声称 overlapping-window fusion 有额外价值。

发现:
B0–B5 没有正式实验，因此 B3>B1、B3>B2、B4>B3、因果代价和任何科学主张均未测试。

证据:
- runs/ajae（只有 calibration.pt）
- src/train.py:2101-2328

判断:
本条要求的运行级审计、实验或正式产物没有发生；代码存在也不能替代执行，判定 NOT RUN。

需要修改:
需要 B0–B5 的冻结、多种子、可追溯结果后才能判定任何主张。

### [B06-06] 状态: NOT RUN

要求:
是否报告 B5 相对 B3/B4 的性能代价。

发现:
B0–B5 没有正式实验，因此 B3>B1、B3>B2、B4>B3、因果代价和任何科学主张均未测试。

证据:
- runs/ajae（只有 calibration.pt）
- src/train.py:2101-2328

判断:
本条要求的运行级审计、实验或正式产物没有发生；代码存在也不能替代执行，判定 NOT RUN。

需要修改:
需要 B0–B5 的冻结、多种子、可追溯结果后才能判定任何主张。

### [B06-07] 状态: NOT RUN

要求:
禁止只挑某一个 seed 支持不稳定结论。

发现:
B0–B5 没有正式实验，因此 B3>B1、B3>B2、B4>B3、因果代价和任何科学主张均未测试。

证据:
- runs/ajae（只有 calibration.pt）
- src/train.py:2101-2328

判断:
本条要求的运行级审计、实验或正式产物没有发生；代码存在也不能替代执行，判定 NOT RUN。

需要修改:
需要 B0–B5 的冻结、多种子、可追溯结果后才能判定任何主张。

## N00：运动正常子集

### [N00-01] 状态: PARTIAL

要求:
是否能识别 moving car。

发现:
评价代码把语义 252–259 合并为 moving-normal 子集，模型输入和损失没有 moving 标签捷径；没有分类别或真实安全性结果。

证据:
- protocol.json:53-60
- src/evaluate.py:349-405
- src/evaluate.py:2230-2289

判断:
只能证明静态实现、局部运行或缺少 生成链路 的产物，尚不能证明可追溯正式路径完整采用；按审计规则不得升级为 PASS。

需要修改:
转为 PASS 所需条件：在 B1/B3/B4 上保存按运动类别和静态正常分开的真实结果。

### [N00-02] 状态: PARTIAL

要求:
是否能识别 moving person。

发现:
评价代码把语义 252–259 合并为 moving-normal 子集，模型输入和损失没有 moving 标签捷径；没有分类别或真实安全性结果。

证据:
- protocol.json:53-60
- src/evaluate.py:349-405
- src/evaluate.py:2230-2289

判断:
只能证明静态实现、局部运行或缺少 生成链路 的产物，尚不能证明可追溯正式路径完整采用；按审计规则不得升级为 PASS。

需要修改:
转为 PASS 所需条件：在 B1/B3/B4 上保存按运动类别和静态正常分开的真实结果。

### [N00-03] 状态: PARTIAL

要求:
是否能识别 moving bicycle / cyclist 类。

发现:
评价代码把语义 252–259 合并为 moving-normal 子集，模型输入和损失没有 moving 标签捷径；没有分类别或真实安全性结果。

证据:
- protocol.json:53-60
- src/evaluate.py:349-405
- src/evaluate.py:2230-2289

判断:
只能证明静态实现、局部运行或缺少 生成链路 的产物，尚不能证明可追溯正式路径完整采用；按审计规则不得升级为 PASS。

需要修改:
转为 PASS 所需条件：在 B1/B3/B4 上保存按运动类别和静态正常分开的真实结果。

### [N00-04] 状态: PASS

要求:
moving semantic 只用于 diagnostic。

发现:
评价代码把语义 252–259 合并为 moving-normal 子集，模型输入和损失没有 moving 标签捷径；没有分类别或真实安全性结果。

证据:
- protocol.json:53-60
- src/evaluate.py:349-405
- src/evaluate.py:2230-2289

判断:
证据直接覆盖该要求，或排除性全仓库搜索已证明冲突路径不在当前正式路线；可判 PASS。该结论只覆盖本条，不外推到整个模块或科学主张。

需要修改:
无需修改；保留当前实现和直接证据，并继续确保该局部结论不被外推为模块或科学主张已经成立。

### [N00-05] 状态: PASS

要求:
moving semantic 没有作为 model input。

发现:
评价代码把语义 252–259 合并为 moving-normal 子集，模型输入和损失没有 moving 标签捷径；没有分类别或真实安全性结果。

证据:
- protocol.json:53-60
- src/evaluate.py:349-405
- src/evaluate.py:2230-2289

判断:
证据直接覆盖该要求，或排除性全仓库搜索已证明冲突路径不在当前正式路线；可判 PASS。该结论只覆盖本条，不外推到整个模块或科学主张。

需要修改:
无需修改；保留当前实现和直接证据，并继续确保该局部结论不被外推为模块或科学主张已经成立。

### [N00-06] 状态: PASS

要求:
moving semantic 没有作为训练 shortcut。

发现:
评价代码把语义 252–259 合并为 moving-normal 子集，模型输入和损失没有 moving 标签捷径；没有分类别或真实安全性结果。

证据:
- protocol.json:53-60
- src/evaluate.py:349-405
- src/evaluate.py:2230-2289

判断:
证据直接覆盖该要求，或排除性全仓库搜索已证明冲突路径不在当前正式路线；可判 PASS。该结论只覆盖本条，不外推到整个模块或科学主张。

需要修改:
无需修改；保留当前实现和直接证据，并继续确保该局部结论不被外推为模块或科学主张已经成立。

## N01：安全性指标

### [N01-01] 状态: BLOCKED

要求:
moving-normal mean anomaly score。

发现:
B1/B3/B4 的 moving/static 正常均值、FPR 和差值全部不存在，Gate4 安全阈值也为 null。

证据:
- src/evaluate.py:349-405
- protocol.json:204-219
- 全仓库无 B1/B3/B4 safety 结果

判断:
上游 Decision Gate 或冻结条件未满足，本条原则上不得继续；判定 BLOCKED。

需要修改:
保持 BLOCKED；需先有 B1/B3/B4 真实预测和冻结安全阈值。

### [N01-02] 状态: BLOCKED

要求:
moving-normal false-positive rate。

发现:
B1/B3/B4 的 moving/static 正常均值、FPR 和差值全部不存在，Gate4 安全阈值也为 null。

证据:
- src/evaluate.py:349-405
- protocol.json:204-219
- 全仓库无 B1/B3/B4 safety 结果

判断:
上游 Decision Gate 或冻结条件未满足，本条原则上不得继续；判定 BLOCKED。

需要修改:
保持 BLOCKED；需先有 B1/B3/B4 真实预测和冻结安全阈值。

### [N01-03] 状态: BLOCKED

要求:
static-normal mean anomaly score。

发现:
B1/B3/B4 的 moving/static 正常均值、FPR 和差值全部不存在，Gate4 安全阈值也为 null。

证据:
- src/evaluate.py:349-405
- protocol.json:204-219
- 全仓库无 B1/B3/B4 safety 结果

判断:
上游 Decision Gate 或冻结条件未满足，本条原则上不得继续；判定 BLOCKED。

需要修改:
保持 BLOCKED；需先有 B1/B3/B4 真实预测和冻结安全阈值。

### [N01-04] 状态: BLOCKED

要求:
static-normal false-positive rate。

发现:
B1/B3/B4 的 moving/static 正常均值、FPR 和差值全部不存在，Gate4 安全阈值也为 null。

证据:
- src/evaluate.py:349-405
- protocol.json:204-219
- 全仓库无 B1/B3/B4 safety 结果

判断:
上游 Decision Gate 或冻结条件未满足，本条原则上不得继续；判定 BLOCKED。

需要修改:
保持 BLOCKED；需先有 B1/B3/B4 真实预测和冻结安全阈值。

### [N01-05] 状态: BLOCKED

要求:
moving vs static gap。

发现:
B1/B3/B4 的 moving/static 正常均值、FPR 和差值全部不存在，Gate4 安全阈值也为 null。

证据:
- src/evaluate.py:349-405
- protocol.json:204-219
- 全仓库无 B1/B3/B4 safety 结果

判断:
上游 Decision Gate 或冻结条件未满足，本条原则上不得继续；判定 BLOCKED。

需要修改:
保持 BLOCKED；需先有 B1/B3/B4 真实预测和冻结安全阈值。

### [N01-06] 状态: BLOCKED

要求:
B3 相对 B1 是否恶化 moving normals。

发现:
B1/B3/B4 的 moving/static 正常均值、FPR 和差值全部不存在，Gate4 安全阈值也为 null。

证据:
- src/evaluate.py:349-405
- protocol.json:204-219
- 全仓库无 B1/B3/B4 safety 结果

判断:
上游 Decision Gate 或冻结条件未满足，本条原则上不得继续；判定 BLOCKED。

需要修改:
保持 BLOCKED；需先有 B1/B3/B4 真实预测和冻结安全阈值。

### [N01-07] 状态: BLOCKED

要求:
B4 相对 B1 是否恶化 moving normals。

发现:
B1/B3/B4 的 moving/static 正常均值、FPR 和差值全部不存在，Gate4 安全阈值也为 null。

证据:
- src/evaluate.py:349-405
- protocol.json:204-219
- 全仓库无 B1/B3/B4 safety 结果

判断:
上游 Decision Gate 或冻结条件未满足，本条原则上不得继续；判定 BLOCKED。

需要修改:
保持 BLOCKED；需先有 B1/B3/B4 真实预测和冻结安全阈值。

### [N01-08] 状态: BLOCKED

要求:
若明显恶化，Decision Gate 4 / 主模型安全判断不得 PASS。

发现:
B1/B3/B4 的 moving/static 正常均值、FPR 和差值全部不存在，Gate4 安全阈值也为 null。

证据:
- src/evaluate.py:349-405
- protocol.json:204-219
- 全仓库无 B1/B3/B4 safety 结果

判断:
上游 Decision Gate 或冻结条件未满足，本条原则上不得继续；判定 BLOCKED。

需要修改:
保持 BLOCKED；需先有 B1/B3/B4 真实预测和冻结安全阈值。

## O00：实体内部方差

### [O00-01] 状态: NOT RUN

要求:
能按照 generator object ID 聚合 anomaly entity points。

发现:
对象尺度诊断类可计算实体内部方差、点数、可见性和邻域背景均分，但只在 toy test 中实例化，正式评价路径未调用。

证据:
- src/evaluate.py:408-561
- test_ajae.py:811-829

判断:
本条要求的运行级审计、实验或正式产物没有发生；代码存在也不能替代执行，判定 NOT RUN。

需要修改:
转为 PASS 所需条件：把诊断绑定正式 fused score 路径并保存 B1/B3/B4 结果。

### [O00-02] 状态: NOT RUN

要求:
计算实体内部 $\operatorname{Var}_{p\in O_m}(S_p)$。

发现:
对象尺度诊断类可计算实体内部方差、点数、可见性和邻域背景均分，但只在 toy test 中实例化，正式评价路径未调用。

证据:
- src/evaluate.py:408-561
- test_ajae.py:811-829

判断:
本条要求的运行级审计、实验或正式产物没有发生；代码存在也不能替代执行，判定 NOT RUN。

需要修改:
转为 PASS 所需条件：把诊断绑定正式 fused score 路径并保存 B1/B3/B4 结果。

### [O00-03] 状态: NOT RUN

要求:
对不同可见点数分层。

发现:
对象尺度诊断类可计算实体内部方差、点数、可见性和邻域背景均分，但只在 toy test 中实例化，正式评价路径未调用。

证据:
- src/evaluate.py:408-561
- test_ajae.py:811-829

判断:
本条要求的运行级审计、实验或正式产物没有发生；代码存在也不能替代执行，判定 NOT RUN。

需要修改:
转为 PASS 所需条件：把诊断绑定正式 fused score 路径并保存 B1/B3/B4 结果。

### [O00-04] 状态: NOT RUN

要求:
比较 B1/B3/B4。

发现:
对象尺度诊断类可计算实体内部方差、点数、可见性和邻域背景均分，但只在 toy test 中实例化，正式评价路径未调用。

证据:
- src/evaluate.py:408-561
- test_ajae.py:811-829

判断:
本条要求的运行级审计、实验或正式产物没有发生；代码存在也不能替代执行，判定 NOT RUN。

需要修改:
转为 PASS 所需条件：把诊断绑定正式 fused score 路径并保存 B1/B3/B4 结果。

### [O00-05] 状态: NOT RUN

要求:
检查多帧模型是否减少内部碎片。

发现:
对象尺度诊断类可计算实体内部方差、点数、可见性和邻域背景均分，但只在 toy test 中实例化，正式评价路径未调用。

证据:
- src/evaluate.py:408-561
- test_ajae.py:811-829

判断:
本条要求的运行级审计、实验或正式产物没有发生；代码存在也不能替代执行，判定 NOT RUN。

需要修改:
转为 PASS 所需条件：把诊断绑定正式 fused score 路径并保存 B1/B3/B4 结果。

## O01：边界泄漏

### [O01-01] 状态: PARTIAL

要求:
定义 anomaly surface points。

发现:
诊断代码可区分异常实体与 0.5 m 邻近普通背景，但没有道路/正常物体细分、尺度分层或 B1/B3 正式结果。

证据:
- src/evaluate.py:408-561
- 全仓库无正式 boundary leakage 结果

判断:
只能证明静态实现、局部运行或缺少 生成链路 的产物，尚不能证明可追溯正式路径完整采用；按审计规则不得升级为 PASS。

需要修改:
转为 PASS 所需条件：细分道路与邻近正常物体，并保存尺度分层 B1/B3 比较。

### [O01-02] 状态: PARTIAL

要求:
定义邻近 normal background。

发现:
诊断代码可区分异常实体与 0.5 m 邻近普通背景，但没有道路/正常物体细分、尺度分层或 B1/B3 正式结果。

证据:
- src/evaluate.py:408-561
- 全仓库无正式 boundary leakage 结果

判断:
只能证明静态实现、局部运行或缺少 生成链路 的产物，尚不能证明可追溯正式路径完整采用；按审计规则不得升级为 PASS。

需要修改:
转为 PASS 所需条件：细分道路与邻近正常物体，并保存尺度分层 B1/B3 比较。

### [O01-03] 状态: PARTIAL

要求:
比较 anomaly score。

发现:
诊断代码可区分异常实体与 0.5 m 邻近普通背景，但没有道路/正常物体细分、尺度分层或 B1/B3 正式结果。

证据:
- src/evaluate.py:408-561
- 全仓库无正式 boundary leakage 结果

判断:
只能证明静态实现、局部运行或缺少 生成链路 的产物，尚不能证明可追溯正式路径完整采用；按审计规则不得升级为 PASS。

需要修改:
转为 PASS 所需条件：细分道路与邻近正常物体，并保存尺度分层 B1/B3 比较。

### [O01-04] 状态: NOT RUN

要求:
检查道路背景高分扩散。

发现:
诊断代码可区分异常实体与 0.5 m 邻近普通背景，但没有道路/正常物体细分、尺度分层或 B1/B3 正式结果。

证据:
- src/evaluate.py:408-561
- 全仓库无正式 boundary leakage 结果

判断:
本条要求的运行级审计、实验或正式产物没有发生；代码存在也不能替代执行，判定 NOT RUN。

需要修改:
转为 PASS 所需条件：细分道路与邻近正常物体，并保存尺度分层 B1/B3 比较。

### [O01-05] 状态: NOT RUN

要求:
检查邻近正常物体高分扩散。

发现:
诊断代码可区分异常实体与 0.5 m 邻近普通背景，但没有道路/正常物体细分、尺度分层或 B1/B3 正式结果。

证据:
- src/evaluate.py:408-561
- 全仓库无正式 boundary leakage 结果

判断:
本条要求的运行级审计、实验或正式产物没有发生；代码存在也不能替代执行，判定 NOT RUN。

需要修改:
转为 PASS 所需条件：细分道路与邻近正常物体，并保存尺度分层 B1/B3 比较。

### [O01-06] 状态: NOT RUN

要求:
报告不同 pyramid scale / B1/B3 的差异。

发现:
诊断代码可区分异常实体与 0.5 m 邻近普通背景，但没有道路/正常物体细分、尺度分层或 B1/B3 正式结果。

证据:
- src/evaluate.py:408-561
- 全仓库无正式 boundary leakage 结果

判断:
本条要求的运行级审计、实验或正式产物没有发生；代码存在也不能替代执行，判定 NOT RUN。

需要修改:
转为 PASS 所需条件：细分道路与邻近正常物体，并保存尺度分层 B1/B3 比较。

## O02：五帧可见性

### [O02-01] 状态: FAIL

要求:
计算 entity 在窗口中 $V=1$。

发现:
开发世界没有任何 V=1 实体，无法形成该层级。

证据:
- src/evaluate.py:521-560
- dev.json（V=5:59，V=4:1，V=1/2/3:0）

判断:
存在与要求直接冲突的实现、数据或真实运行事实；判定 FAIL。

需要修改:
转为 PASS 所需条件：先补齐 V=1..5 固定世界覆盖，再报告各层配对性能与趋势。

### [O02-02] 状态: FAIL

要求:
$V=2$。

发现:
开发世界没有任何 V=2 实体，无法形成该层级。

证据:
- src/evaluate.py:521-560
- dev.json（V=5:59，V=4:1，V=1/2/3:0）

判断:
存在与要求直接冲突的实现、数据或真实运行事实；判定 FAIL。

需要修改:
转为 PASS 所需条件：先补齐 V=1..5 固定世界覆盖，再报告各层配对性能与趋势。

### [O02-03] 状态: FAIL

要求:
$V=3$。

发现:
开发世界没有任何 V=3 实体，无法形成该层级。

证据:
- src/evaluate.py:521-560
- dev.json（V=5:59，V=4:1，V=1/2/3:0）

判断:
存在与要求直接冲突的实现、数据或真实运行事实；判定 FAIL。

需要修改:
转为 PASS 所需条件：先补齐 V=1..5 固定世界覆盖，再报告各层配对性能与趋势。

### [O02-04] 状态: PARTIAL

要求:
$V=4$。

发现:
代码预留 V=1..5 报告槽；当前 60 个开发实体中 59 个 V=5、1 个 V=4，V=1/2/3 完全缺失，也没有分层性能结果。

证据:
- src/evaluate.py:521-560
- dev.json（V=5:59，V=4:1，V=1/2/3:0）

判断:
只能证明静态实现、局部运行或缺少 生成链路 的产物，尚不能证明可追溯正式路径完整采用；按审计规则不得升级为 PASS。

需要修改:
转为 PASS 所需条件：先补齐 V=1..5 固定世界覆盖，再报告各层配对性能与趋势。

### [O02-05] 状态: PARTIAL

要求:
$V=5$。

发现:
代码预留 V=1..5 报告槽；当前 60 个开发实体中 59 个 V=5、1 个 V=4，V=1/2/3 完全缺失，也没有分层性能结果。

证据:
- src/evaluate.py:521-560
- dev.json（V=5:59，V=4:1，V=1/2/3:0）

判断:
只能证明静态实现、局部运行或缺少 生成链路 的产物，尚不能证明可追溯正式路径完整采用；按审计规则不得升级为 PASS。

需要修改:
转为 PASS 所需条件：先补齐 V=1..5 固定世界覆盖，再报告各层配对性能与趋势。

### [O02-06] 状态: NOT RUN

要求:
按 V 分层报告性能。

发现:
代码预留 V=1..5 报告槽；当前 60 个开发实体中 59 个 V=5、1 个 V=4，V=1/2/3 完全缺失，也没有分层性能结果。

证据:
- src/evaluate.py:521-560
- dev.json（V=5:59，V=4:1，V=1/2/3:0）

判断:
本条要求的运行级审计、实验或正式产物没有发生；代码存在也不能替代执行，判定 NOT RUN。

需要修改:
转为 PASS 所需条件：先补齐 V=1..5 固定世界覆盖，再报告各层配对性能与趋势。

### [O02-07] 状态: NOT RUN

要求:
检查性能是否随多帧证据增加而改善。

发现:
代码预留 V=1..5 报告槽；当前 60 个开发实体中 59 个 V=5、1 个 V=4，V=1/2/3 完全缺失，也没有分层性能结果。

证据:
- src/evaluate.py:521-560
- dev.json（V=5:59，V=4:1，V=1/2/3:0）

判断:
本条要求的运行级审计、实验或正式产物没有发生；代码存在也不能替代执行，判定 NOT RUN。

需要修改:
转为 PASS 所需条件：先补齐 V=1..5 固定世界覆盖，再报告各层配对性能与趋势。

### [O02-08] 状态: NOT RUN

要求:
若没有该趋势，不得声称结果证明“对象尺度时空共识”。

发现:
代码预留 V=1..5 报告槽；当前 60 个开发实体中 59 个 V=5、1 个 V=4，V=1/2/3 完全缺失，也没有分层性能结果。

证据:
- src/evaluate.py:521-560
- dev.json（V=5:59，V=4:1，V=1/2/3:0）

判断:
本条要求的运行级审计、实验或正式产物没有发生；代码存在也不能替代执行，判定 NOT RUN。

需要修改:
转为 PASS 所需条件：先补齐 V=1..5 固定世界覆盖，再报告各层配对性能与趋势。

## TR00：动态完整世界

### [TR00-01] 状态: BLOCKED

要求:
训练期间是否持续采样新的 206 world specs。

发现:
按 seed/world index 动态构造世界并遍历全部合法中心帧的代码存在；Gate1 和未冻结开发规则使实际训练未运行。

证据:
- src/train.py:1549-1658
- src/train.py:2101-2328
- 本次 formal preflight 退出码 1

判断:
上游 Decision Gate 或冻结条件未满足，本条原则上不得继续；判定 BLOCKED。

需要修改:
保持 BLOCKED；Gate1 和训练预算/评价规则冻结前不得启动动态世界训练。

### [TR00-02] 状态: BLOCKED

要求:
每个 world 在开始遍历 window 前固定全部 entities。

发现:
按 seed/world index 动态构造世界并遍历全部合法中心帧的代码存在；Gate1 和未冻结开发规则使实际训练未运行。

证据:
- src/train.py:1549-1658
- src/train.py:2101-2328
- 本次 formal preflight 退出码 1

判断:
上游 Decision Gate 或冻结条件未满足，本条原则上不得继续；判定 BLOCKED。

需要修改:
保持 BLOCKED；Gate1 和训练预算/评价规则冻结前不得启动动态世界训练。

### [TR00-03] 状态: BLOCKED

要求:
固定 geometry。

发现:
按 seed/world index 动态构造世界并遍历全部合法中心帧的代码存在；Gate1 和未冻结开发规则使实际训练未运行。

证据:
- src/train.py:1549-1658
- src/train.py:2101-2328
- 本次 formal preflight 退出码 1

判断:
上游 Decision Gate 或冻结条件未满足，本条原则上不得继续；判定 BLOCKED。

需要修改:
保持 BLOCKED；Gate1 和训练预算/评价规则冻结前不得启动动态世界训练。

### [TR00-04] 状态: BLOCKED

要求:
固定 positions。

发现:
按 seed/world index 动态构造世界并遍历全部合法中心帧的代码存在；Gate1 和未冻结开发规则使实际训练未运行。

证据:
- src/train.py:1549-1658
- src/train.py:2101-2328
- 本次 formal preflight 退出码 1

判断:
上游 Decision Gate 或冻结条件未满足，本条原则上不得继续；判定 BLOCKED。

需要修改:
保持 BLOCKED；Gate1 和训练预算/评价规则冻结前不得启动动态世界训练。

### [TR00-05] 状态: BLOCKED

要求:
固定 materials。

发现:
按 seed/world index 动态构造世界并遍历全部合法中心帧的代码存在；Gate1 和未冻结开发规则使实际训练未运行。

证据:
- src/train.py:1549-1658
- src/train.py:2101-2328
- 本次 formal preflight 退出码 1

判断:
上游 Decision Gate 或冻结条件未满足，本条原则上不得继续；判定 BLOCKED。

需要修改:
保持 BLOCKED；Gate1 和训练预算/评价规则冻结前不得启动动态世界训练。

### [TR00-06] 状态: BLOCKED

要求:
固定 random seeds。

发现:
按 seed/world index 动态构造世界并遍历全部合法中心帧的代码存在；Gate1 和未冻结开发规则使实际训练未运行。

证据:
- src/train.py:1549-1658
- src/train.py:2101-2328
- 本次 formal preflight 退出码 1

判断:
上游 Decision Gate 或冻结条件未满足，本条原则上不得继续；判定 BLOCKED。

需要修改:
保持 BLOCKED；Gate1 和训练预算/评价规则冻结前不得启动动态世界训练。

### [TR00-07] 状态: BLOCKED

要求:
一个 world 是否遍历所有合法 center windows。

发现:
按 seed/world index 动态构造世界并遍历全部合法中心帧的代码存在；Gate1 和未冻结开发规则使实际训练未运行。

证据:
- src/train.py:1549-1658
- src/train.py:2101-2328
- 本次 formal preflight 退出码 1

判断:
上游 Decision Gate 或冻结条件未满足，本条原则上不得继续；判定 BLOCKED。

需要修改:
保持 BLOCKED；Gate1 和训练预算/评价规则冻结前不得启动动态世界训练。

### [TR00-08] 状态: BLOCKED

要求:
相邻 windows 是否复用同一世界中的共享帧。

发现:
按 seed/world index 动态构造世界并遍历全部合法中心帧的代码存在；Gate1 和未冻结开发规则使实际训练未运行。

证据:
- src/train.py:1549-1658
- src/train.py:2101-2328
- 本次 formal preflight 退出码 1

判断:
上游 Decision Gate 或冻结条件未满足，本条原则上不得继续；判定 BLOCKED。

需要修改:
保持 BLOCKED；Gate1 和训练预算/评价规则冻结前不得启动动态世界训练。

### [TR00-09] 状态: NOT RUN

要求:
日志中是否把 windows 错误统计成 independent worlds。

发现:
按 seed/world index 动态构造世界并遍历全部合法中心帧的代码存在；Gate1 和未冻结开发规则使实际训练未运行。

证据:
- src/train.py:1549-1658
- src/train.py:2101-2328
- 本次 formal preflight 退出码 1

判断:
本条要求的运行级审计、实验或正式产物没有发生；代码存在也不能替代执行，判定 NOT RUN。

需要修改:
保持 BLOCKED；Gate1 和训练预算/评价规则冻结前不得启动动态世界训练。

## TR01：有界缓存

### [TR01-01] 状态: BLOCKED

要求:
是否按时间块 deterministic render。

发现:
render/STU 两个 LRU 有容量和清理策略，可流式处理；缓存键只有 `frame_id`，不含 world、generator、renderer 或 STU 版本，且没有 cached/uncached 一致性或峰值内存结果。

证据:
- src/train.py:566-601
- src/train.py:1024-1088
- FrameCache key=`frame_id`

判断:
上游 Decision Gate 或冻结条件未满足，本条原则上不得继续；判定 BLOCKED。

需要修改:
转为合规所需条件：使用复合缓存身份并补做 cached/uncached 一致性与峰值资源审计。

### [TR01-02] 状态: BLOCKED

要求:
是否存在有限 frame cache / LRU。

发现:
render/STU 两个 LRU 有容量和清理策略，可流式处理；缓存键只有 `frame_id`，不含 world、generator、renderer 或 STU 版本，且没有 cached/uncached 一致性或峰值内存结果。

证据:
- src/train.py:566-601
- src/train.py:1024-1088
- FrameCache key=`frame_id`

判断:
上游 Decision Gate 或冻结条件未满足，本条原则上不得继续；判定 BLOCKED。

需要修改:
转为合规所需条件：使用复合缓存身份并补做 cached/uncached 一致性与峰值资源审计。

### [TR01-03] 状态: BLOCKED

要求:
frozen STU feature 是否可在共享 frame 间复用。

发现:
render/STU 两个 LRU 有容量和清理策略，可流式处理；缓存键只有 `frame_id`，不含 world、generator、renderer 或 STU 版本，且没有 cached/uncached 一致性或峰值内存结果。

证据:
- src/train.py:566-601
- src/train.py:1024-1088
- FrameCache key=`frame_id`

判断:
上游 Decision Gate 或冻结条件未满足，本条原则上不得继续；判定 BLOCKED。

需要修改:
转为合规所需条件：使用复合缓存身份并补做 cached/uncached 一致性与峰值资源审计。

### [TR01-04] 状态: BLOCKED

要求:
block 完成后是否释放无用缓存。

发现:
render/STU 两个 LRU 有容量和清理策略，可流式处理；缓存键只有 `frame_id`，不含 world、generator、renderer 或 STU 版本，且没有 cached/uncached 一致性或峰值内存结果。

证据:
- src/train.py:566-601
- src/train.py:1024-1088
- FrameCache key=`frame_id`

判断:
上游 Decision Gate 或冻结条件未满足，本条原则上不得继续；判定 BLOCKED。

需要修改:
转为合规所需条件：使用复合缓存身份并补做 cached/uncached 一致性与峰值资源审计。

### [TR01-05] 状态: FAIL

要求:
cache key 是否至少包含 world identity + frame identity + relevant generator/STU version。

发现:
`FrameCache` 两个字典均只以 `frame_id` 为键，没有 world identity、generator/renderer version 或 STU version，直接违反复合身份要求。

证据:
- src/train.py:566-601
- src/train.py:1024-1088
- FrameCache key=`frame_id`

判断:
存在与要求直接冲突的实现、数据或真实运行事实；判定 FAIL。

需要修改:
缓存键至少绑定 world、frame、generator/renderer 身份和 STU 权重身份。

### [TR01-06] 状态: PARTIAL

要求:
cache 是否可能跨不同 world 错误复用。

发现:
render/STU 两个 LRU 有容量和清理策略，可流式处理；缓存键只有 `frame_id`，不含 world、generator、renderer 或 STU 版本，且没有 cached/uncached 一致性或峰值内存结果。

证据:
- src/train.py:566-601
- src/train.py:1024-1088
- FrameCache key=`frame_id`

判断:
只能证明静态实现、局部运行或缺少 生成链路 的产物，尚不能证明可追溯正式路径完整采用；按审计规则不得升级为 PASS。

需要修改:
转为合规所需条件：使用复合缓存身份并补做 cached/uncached 一致性与峰值资源审计。

### [TR01-07] 状态: NOT RUN

要求:
cached 与 uncached output 是否逐点一致。

发现:
render/STU 两个 LRU 有容量和清理策略，可流式处理；缓存键只有 `frame_id`，不含 world、generator、renderer 或 STU 版本，且没有 cached/uncached 一致性或峰值内存结果。

证据:
- src/train.py:566-601
- src/train.py:1024-1088
- FrameCache key=`frame_id`

判断:
本条要求的运行级审计、实验或正式产物没有发生；代码存在也不能替代执行，判定 NOT RUN。

需要修改:
转为合规所需条件：使用复合缓存身份并补做 cached/uncached 一致性与峰值资源审计。

### [TR01-08] 状态: PASS

要求:
不要求整条 sequence 全物化到 RAM/disk。

发现:
render/STU 两个 LRU 有容量和清理策略，可流式处理；缓存键只有 `frame_id`，不含 world、generator、renderer 或 STU 版本，且没有 cached/uncached 一致性或峰值内存结果。

证据:
- src/train.py:566-601
- src/train.py:1024-1088
- FrameCache key=`frame_id`

判断:
证据直接覆盖该要求，或排除性全仓库搜索已证明冲突路径不在当前正式路线；可判 PASS。该结论只覆盖本条，不外推到整个模块或科学主张。

需要修改:
无需修改；保留当前实现和直接证据，并继续确保该局部结论不被外推为模块或科学主张已经成立。

### [TR01-09] 状态: NOT RUN

要求:
是否记录峰值显存/内存。

发现:
render/STU 两个 LRU 有容量和清理策略，可流式处理；缓存键只有 `frame_id`，不含 world、generator、renderer 或 STU 版本，且没有 cached/uncached 一致性或峰值内存结果。

证据:
- src/train.py:566-601
- src/train.py:1024-1088
- FrameCache key=`frame_id`

判断:
本条要求的运行级审计、实验或正式产物没有发生；代码存在也不能替代执行，判定 NOT RUN。

需要修改:
转为合规所需条件：使用复合缓存身份并补做 cached/uncached 一致性与峰值资源审计。

## TR02：批处理

### [TR02-01] 状态: BLOCKED

要求:
micro-batch 是否为 1 个完整五帧 window。

发现:
配置固定 micro-batch=1、accumulation=8，代码含 loss 缩放、partial-group 补偿和 step 时序；无正式训练证明实际生效。

证据:
- protocol.json:140-146
- src/train.py:948-987

判断:
上游 Decision Gate 或冻结条件未满足，本条原则上不得继续；判定 BLOCKED。

需要修改:
转为 PASS 所需条件：在允许训练后保存 resolved batch/accumulation 和 step trace。

### [TR02-02] 状态: BLOCKED

要求:
是否使用 gradient accumulation 获得 effective batch。

发现:
配置固定 micro-batch=1、accumulation=8，代码含 loss 缩放、partial-group 补偿和 step 时序；无正式训练证明实际生效。

证据:
- protocol.json:140-146
- src/train.py:948-987

判断:
上游 Decision Gate 或冻结条件未满足，本条原则上不得继续；判定 BLOCKED。

需要修改:
转为 PASS 所需条件：在允许训练后保存 resolved batch/accumulation 和 step trace。

### [TR02-03] 状态: BLOCKED

要求:
gradient accumulation 实现是否正确除/累积 loss。

发现:
配置固定 micro-batch=1、accumulation=8，代码含 loss 缩放、partial-group 补偿和 step 时序；无正式训练证明实际生效。

证据:
- protocol.json:140-146
- src/train.py:948-987

判断:
上游 Decision Gate 或冻结条件未满足，本条原则上不得继续；判定 BLOCKED。

需要修改:
转为 PASS 所需条件：在允许训练后保存 resolved batch/accumulation 和 step trace。

### [TR02-04] 状态: BLOCKED

要求:
optimizer step 时机正确。

发现:
配置固定 micro-batch=1、accumulation=8，代码含 loss 缩放、partial-group 补偿和 step 时序；无正式训练证明实际生效。

证据:
- protocol.json:140-146
- src/train.py:948-987

判断:
上游 Decision Gate 或冻结条件未满足，本条原则上不得继续；判定 BLOCKED。

需要修改:
转为 PASS 所需条件：在允许训练后保存 resolved batch/accumulation 和 step trace。

### [TR02-05] 状态: BLOCKED

要求:
scheduler step 时机正确。

发现:
配置固定 micro-batch=1、accumulation=8，代码含 loss 缩放、partial-group 补偿和 step 时序；无正式训练证明实际生效。

证据:
- protocol.json:140-146
- src/train.py:948-987

判断:
上游 Decision Gate 或冻结条件未满足，本条原则上不得继续；判定 BLOCKED。

需要修改:
转为 PASS 所需条件：在允许训练后保存 resolved batch/accumulation 和 step trace。

### [TR02-06] 状态: PASS

要求:
是否没有为了 batch size 大量随机删原始点。

发现:
配置固定 micro-batch=1、accumulation=8，代码含 loss 缩放、partial-group 补偿和 step 时序；无正式训练证明实际生效。

证据:
- protocol.json:140-146
- src/train.py:948-987

判断:
证据直接覆盖该要求，或排除性全仓库搜索已证明冲突路径不在当前正式路线；可判 PASS。该结论只覆盖本条，不外推到整个模块或科学主张。

需要修改:
无需修改；保留当前实现和直接证据，并继续确保该局部结论不被外推为模块或科学主张已经成立。

### [TR02-07] 状态: PARTIAL

要求:
small anomalies 是否没有因采样系统性消失。

发现:
配置固定 micro-batch=1、accumulation=8，代码含 loss 缩放、partial-group 补偿和 step 时序；无正式训练证明实际生效。

证据:
- protocol.json:140-146
- src/train.py:948-987

判断:
只能证明静态实现、局部运行或缺少 生成链路 的产物，尚不能证明可追溯正式路径完整采用；按审计规则不得升级为 PASS。

需要修改:
转为 PASS 所需条件：在允许训练后保存 resolved batch/accumulation 和 step trace。

## TR03：三训练种子

### [TR03-01] 状态: BLOCKED

要求:
正式开发至少 3 个独立 seeds。

发现:
代码要求至少三个独立 seed 并设计独立目录/世界流；三个 seed 的命令、配置、检查点和结果全部不存在。

证据:
- src/train.py:290-366
- src/train.py:1985-2049
- runs/ajae（无 seed 目录）

判断:
上游 Decision Gate 或冻结条件未满足，本条原则上不得继续；判定 BLOCKED。

需要修改:
保持 BLOCKED；上游通过后才可生成三套独立且可追溯的正式运行。

### [TR03-02] 状态: BLOCKED

要求:
三 seed 模型初始化独立。

发现:
代码要求至少三个独立 seed 并设计独立目录/世界流；三个 seed 的命令、配置、检查点和结果全部不存在。

证据:
- src/train.py:290-366
- src/train.py:1985-2049
- runs/ajae（无 seed 目录）

判断:
上游 Decision Gate 或冻结条件未满足，本条原则上不得继续；判定 BLOCKED。

需要修改:
保持 BLOCKED；上游通过后才可生成三套独立且可追溯的正式运行。

### [TR03-03] 状态: BLOCKED

要求:
三 seed 的 206 dynamic world stream 独立。

发现:
代码要求至少三个独立 seed 并设计独立目录/世界流；三个 seed 的命令、配置、检查点和结果全部不存在。

证据:
- src/train.py:290-366
- src/train.py:1985-2049
- runs/ajae（无 seed 目录）

判断:
上游 Decision Gate 或冻结条件未满足，本条原则上不得继续；判定 BLOCKED。

需要修改:
保持 BLOCKED；上游通过后才可生成三套独立且可追溯的正式运行。

### [TR03-04] 状态: BLOCKED

要求:
三 seed 使用完全相同 201 fixed worlds。

发现:
代码要求至少三个独立 seed 并设计独立目录/世界流；三个 seed 的命令、配置、检查点和结果全部不存在。

证据:
- src/train.py:290-366
- src/train.py:1985-2049
- runs/ajae（无 seed 目录）

判断:
上游 Decision Gate 或冻结条件未满足，本条原则上不得继续；判定 BLOCKED。

需要修改:
保持 BLOCKED；上游通过后才可生成三套独立且可追溯的正式运行。

### [TR03-05] 状态: BLOCKED

要求:
三 seed checkpoint-selection rule 完全相同。

发现:
代码要求至少三个独立 seed 并设计独立目录/世界流；三个 seed 的命令、配置、检查点和结果全部不存在。

证据:
- src/train.py:290-366
- src/train.py:1985-2049
- runs/ajae（无 seed 目录）

判断:
上游 Decision Gate 或冻结条件未满足，本条原则上不得继续；判定 BLOCKED。

需要修改:
保持 BLOCKED；上游通过后才可生成三套独立且可追溯的正式运行。

### [TR03-06] 状态: BLOCKED

要求:
三 seed hyperparameters 完全相同，除随机种子。

发现:
代码要求至少三个独立 seed 并设计独立目录/世界流；三个 seed 的命令、配置、检查点和结果全部不存在。

证据:
- src/train.py:290-366
- src/train.py:1985-2049
- runs/ajae（无 seed 目录）

判断:
上游 Decision Gate 或冻结条件未满足，本条原则上不得继续；判定 BLOCKED。

需要修改:
保持 BLOCKED；上游通过后才可生成三套独立且可追溯的正式运行。

### [TR03-07] 状态: BLOCKED

要求:
每个 seed 的 command/config/checkpoint 可追溯。

发现:
代码要求至少三个独立 seed 并设计独立目录/世界流；三个 seed 的命令、配置、检查点和结果全部不存在。

证据:
- src/train.py:290-366
- src/train.py:1985-2049
- runs/ajae（无 seed 目录）

判断:
上游 Decision Gate 或冻结条件未满足，本条原则上不得继续；判定 BLOCKED。

需要修改:
保持 BLOCKED；上游通过后才可生成三套独立且可追溯的正式运行。

### [TR03-08] 状态: BLOCKED

要求:
最终报告没有只挑最好 seed。

发现:
代码要求至少三个独立 seed 并设计独立目录/世界流；三个 seed 的命令、配置、检查点和结果全部不存在。

证据:
- src/train.py:290-366
- src/train.py:1985-2049
- runs/ajae（无 seed 目录）

判断:
上游 Decision Gate 或冻结条件未满足，本条原则上不得继续；判定 BLOCKED。

需要修改:
保持 BLOCKED；上游通过后才可生成三套独立且可追溯的正式运行。

## E00：官方点指标

### [E00-01] 状态: NOT RUN

要求:
AP。

发现:
自研评价器实现 AP、AUROC、FPR95、2.5–50 m、ignore 和每帧少于五个异常点过滤；只有 toy parity，没有正式 prediction 经官方 evaluator 复算或原始日志。

证据:
- src/evaluate.py:85-250
- test_ajae.py:679-731
- 全仓库无 official evaluator 正式输出

判断:
本条要求的运行级审计、实验或正式产物没有发生；代码存在也不能替代执行，判定 NOT RUN。

需要修改:
转为 PASS 所需条件：保存 prediction，由 STU 官方 evaluator 复算并保留原始日志。

### [E00-02] 状态: NOT RUN

要求:
AUROC。

发现:
自研评价器实现 AP、AUROC、FPR95、2.5–50 m、ignore 和每帧少于五个异常点过滤；只有 toy parity，没有正式 prediction 经官方 evaluator 复算或原始日志。

证据:
- src/evaluate.py:85-250
- test_ajae.py:679-731
- 全仓库无 official evaluator 正式输出

判断:
本条要求的运行级审计、实验或正式产物没有发生；代码存在也不能替代执行，判定 NOT RUN。

需要修改:
转为 PASS 所需条件：保存 prediction，由 STU 官方 evaluator 复算并保留原始日志。

### [E00-03] 状态: NOT RUN

要求:
FPR95。

发现:
自研评价器实现 AP、AUROC、FPR95、2.5–50 m、ignore 和每帧少于五个异常点过滤；只有 toy parity，没有正式 prediction 经官方 evaluator 复算或原始日志。

证据:
- src/evaluate.py:85-250
- test_ajae.py:679-731
- 全仓库无 official evaluator 正式输出

判断:
本条要求的运行级审计、实验或正式产物没有发生；代码存在也不能替代执行，判定 NOT RUN。

需要修改:
转为 PASS 所需条件：保存 prediction，由 STU 官方 evaluator 复算并保留原始日志。

### [E00-04] 状态: NOT RUN

要求:
距离范围严格 2.5–50 m。

发现:
自研评价器实现 AP、AUROC、FPR95、2.5–50 m、ignore 和每帧少于五个异常点过滤；只有 toy parity，没有正式 prediction 经官方 evaluator 复算或原始日志。

证据:
- src/evaluate.py:85-250
- test_ajae.py:679-731
- 全仓库无 official evaluator 正式输出

判断:
本条要求的运行级审计、实验或正式产物没有发生；代码存在也不能替代执行，判定 NOT RUN。

需要修改:
转为 PASS 所需条件：保存 prediction，由 STU 官方 evaluator 复算并保留原始日志。

### [E00-05] 状态: NOT RUN

要求:
ignore filtering 正确。

发现:
自研评价器实现 AP、AUROC、FPR95、2.5–50 m、ignore 和每帧少于五个异常点过滤；只有 toy parity，没有正式 prediction 经官方 evaluator 复算或原始日志。

证据:
- src/evaluate.py:85-250
- test_ajae.py:679-731
- 全仓库无 official evaluator 正式输出

判断:
本条要求的运行级审计、实验或正式产物没有发生；代码存在也不能替代执行，判定 NOT RUN。

需要修改:
转为 PASS 所需条件：保存 prediction，由 STU 官方 evaluator 复算并保留原始日志。

### [E00-06] 状态: NOT RUN

要求:
过滤后每帧 anomaly points < 5 时不加入 official metric accumulation。

发现:
自研评价器实现 AP、AUROC、FPR95、2.5–50 m、ignore 和每帧少于五个异常点过滤；只有 toy parity，没有正式 prediction 经官方 evaluator 复算或原始日志。

证据:
- src/evaluate.py:85-250
- test_ajae.py:679-731
- 全仓库无 official evaluator 正式输出

判断:
本条要求的运行级审计、实验或正式产物没有发生；代码存在也不能替代执行，判定 NOT RUN。

需要修改:
转为 PASS 所需条件：保存 prediction，由 STU 官方 evaluator 复算并保留原始日志。

### [E00-07] 状态: PARTIAL

要求:
该 `<5` 规则没有误用于训练。

发现:
自研评价器实现 AP、AUROC、FPR95、2.5–50 m、ignore 和每帧少于五个异常点过滤；只有 toy parity，没有正式 prediction 经官方 evaluator 复算或原始日志。

证据:
- src/evaluate.py:85-250
- test_ajae.py:679-731
- 全仓库无 official evaluator 正式输出

判断:
只能证明静态实现、局部运行或缺少 生成链路 的产物，尚不能证明可追溯正式路径完整采用；按审计规则不得升级为 PASS。

需要修改:
转为 PASS 所需条件：保存 prediction，由 STU 官方 evaluator 复算并保留原始日志。

### [E00-08] 状态: PARTIAL

要求:
自研 evaluator 与官方 evaluator 是否做过一致性测试。

发现:
自研评价器实现 AP、AUROC、FPR95、2.5–50 m、ignore 和每帧少于五个异常点过滤；只有 toy parity，没有正式 prediction 经官方 evaluator 复算或原始日志。

证据:
- src/evaluate.py:85-250
- test_ajae.py:679-731
- 全仓库无 official evaluator 正式输出

判断:
只能证明静态实现、局部运行或缺少 生成链路 的产物，尚不能证明可追溯正式路径完整采用；按审计规则不得升级为 PASS。

需要修改:
转为 PASS 所需条件：保存 prediction，由 STU 官方 evaluator 复算并保留原始日志。

### [E00-09] 状态: NOT RUN

要求:
正式结果是否由 STU official evaluator 读取 prediction files 复算。

发现:
自研评价器实现 AP、AUROC、FPR95、2.5–50 m、ignore 和每帧少于五个异常点过滤；只有 toy parity，没有正式 prediction 经官方 evaluator 复算或原始日志。

证据:
- src/evaluate.py:85-250
- test_ajae.py:679-731
- 全仓库无 official evaluator 正式输出

判断:
本条要求的运行级审计、实验或正式产物没有发生；代码存在也不能替代执行，判定 NOT RUN。

需要修改:
转为 PASS 所需条件：保存 prediction，由 STU 官方 evaluator 复算并保留原始日志。

### [E00-10] 状态: NOT RUN

要求:
保存 official evaluator 原始输出日志。

发现:
自研评价器实现 AP、AUROC、FPR95、2.5–50 m、ignore 和每帧少于五个异常点过滤；只有 toy parity，没有正式 prediction 经官方 evaluator 复算或原始日志。

证据:
- src/evaluate.py:85-250
- test_ajae.py:679-731
- 全仓库无 official evaluator 正式输出

判断:
本条要求的运行级审计、实验或正式产物没有发生；代码存在也不能替代执行，判定 NOT RUN。

需要修改:
转为 PASS 所需条件：保存 prediction，由 STU 官方 evaluator 复算并保留原始日志。

## E01：对象级评价

### [E01-01] 状态: NOT RUN

要求:
使用 fused $S_p$。

发现:
阈值、逐帧 DBSCAN 和对象指标接口存在，但阈值/参数尚未从 201 冻结，所有对象级正式结果均未运行。

证据:
- src/evaluate.py:1272-1448
- src/evaluate.py:2303-2550
- 全仓库无对象级正式结果

判断:
本条要求的运行级审计、实验或正式产物没有发生；代码存在也不能替代执行，判定 NOT RUN。

需要修改:
保持未运行；阈值和 DBSCAN 只能从冻结 201 选择，随后用 fused score 正式评价。

### [E01-02] 状态: NOT RUN

要求:
point threshold $\tau$。

发现:
阈值、逐帧 DBSCAN 和对象指标接口存在，但阈值/参数尚未从 201 冻结，所有对象级正式结果均未运行。

证据:
- src/evaluate.py:1272-1448
- src/evaluate.py:2303-2550
- 全仓库无对象级正式结果

判断:
本条要求的运行级审计、实验或正式产物没有发生；代码存在也不能替代执行，判定 NOT RUN。

需要修改:
保持未运行；阈值和 DBSCAN 只能从冻结 201 选择，随后用 fused score 正式评价。

### [E01-03] 状态: BLOCKED

要求:
threshold 只从 201 开发。

发现:
阈值、逐帧 DBSCAN 和对象指标接口存在，但阈值/参数尚未从 201 冻结，所有对象级正式结果均未运行。

证据:
- src/evaluate.py:1272-1448
- src/evaluate.py:2303-2550
- 全仓库无对象级正式结果

判断:
上游 Decision Gate 或冻结条件未满足，本条原则上不得继续；判定 BLOCKED。

需要修改:
保持未运行；阈值和 DBSCAN 只能从冻结 201 选择，随后用 fused score 正式评价。

### [E01-04] 状态: NOT RUN

要求:
每帧单独 3D DBSCAN。

发现:
阈值、逐帧 DBSCAN 和对象指标接口存在，但阈值/参数尚未从 201 冻结，所有对象级正式结果均未运行。

证据:
- src/evaluate.py:1272-1448
- src/evaluate.py:2303-2550
- 全仓库无对象级正式结果

判断:
本条要求的运行级审计、实验或正式产物没有发生；代码存在也不能替代执行，判定 NOT RUN。

需要修改:
保持未运行；阈值和 DBSCAN 只能从冻结 201 选择，随后用 fused score 正式评价。

### [E01-05] 状态: BLOCKED

要求:
DBSCAN 参数只从 201 开发。

发现:
阈值、逐帧 DBSCAN 和对象指标接口存在，但阈值/参数尚未从 201 冻结，所有对象级正式结果均未运行。

证据:
- src/evaluate.py:1272-1448
- src/evaluate.py:2303-2550
- 全仓库无对象级正式结果

判断:
上游 Decision Gate 或冻结条件未满足，本条原则上不得继续；判定 BLOCKED。

需要修改:
保持未运行；阈值和 DBSCAN 只能从冻结 201 选择，随后用 fused score 正式评价。

### [E01-06] 状态: NOT RUN

要求:
不执行跨帧 tracking。

发现:
阈值、逐帧 DBSCAN 和对象指标接口存在，但阈值/参数尚未从 201 冻结，所有对象级正式结果均未运行。

证据:
- src/evaluate.py:1272-1448
- src/evaluate.py:2303-2550
- 全仓库无对象级正式结果

判断:
本条要求的运行级审计、实验或正式产物没有发生；代码存在也不能替代执行，判定 NOT RUN。

需要修改:
保持未运行；阈值和 DBSCAN 只能从冻结 201 选择，随后用 fused score 正式评价。

### [E01-07] 状态: NOT RUN

要求:
RecallQ。

发现:
阈值、逐帧 DBSCAN 和对象指标接口存在，但阈值/参数尚未从 201 冻结，所有对象级正式结果均未运行。

证据:
- src/evaluate.py:1272-1448
- src/evaluate.py:2303-2550
- 全仓库无对象级正式结果

判断:
本条要求的运行级审计、实验或正式产物没有发生；代码存在也不能替代执行，判定 NOT RUN。

需要修改:
保持未运行；阈值和 DBSCAN 只能从冻结 201 选择，随后用 fused score 正式评价。

### [E01-08] 状态: NOT RUN

要求:
SQ。

发现:
阈值、逐帧 DBSCAN 和对象指标接口存在，但阈值/参数尚未从 201 冻结，所有对象级正式结果均未运行。

证据:
- src/evaluate.py:1272-1448
- src/evaluate.py:2303-2550
- 全仓库无对象级正式结果

判断:
本条要求的运行级审计、实验或正式产物没有发生；代码存在也不能替代执行，判定 NOT RUN。

需要修改:
保持未运行；阈值和 DBSCAN 只能从冻结 201 选择，随后用 fused score 正式评价。

### [E01-09] 状态: NOT RUN

要求:
RQ。

发现:
阈值、逐帧 DBSCAN 和对象指标接口存在，但阈值/参数尚未从 201 冻结，所有对象级正式结果均未运行。

证据:
- src/evaluate.py:1272-1448
- src/evaluate.py:2303-2550
- 全仓库无对象级正式结果

判断:
本条要求的运行级审计、实验或正式产物没有发生；代码存在也不能替代执行，判定 NOT RUN。

需要修改:
保持未运行；阈值和 DBSCAN 只能从冻结 201 选择，随后用 fused score 正式评价。

### [E01-10] 状态: NOT RUN

要求:
UQ。

发现:
阈值、逐帧 DBSCAN 和对象指标接口存在，但阈值/参数尚未从 201 冻结，所有对象级正式结果均未运行。

证据:
- src/evaluate.py:1272-1448
- src/evaluate.py:2303-2550
- 全仓库无对象级正式结果

判断:
本条要求的运行级审计、实验或正式产物没有发生；代码存在也不能替代执行，判定 NOT RUN。

需要修改:
保持未运行；阈值和 DBSCAN 只能从冻结 201 选择，随后用 fused score 正式评价。

### [E01-11] 状态: NOT RUN

要求:
PQ。

发现:
阈值、逐帧 DBSCAN 和对象指标接口存在，但阈值/参数尚未从 201 冻结，所有对象级正式结果均未运行。

证据:
- src/evaluate.py:1272-1448
- src/evaluate.py:2303-2550
- 全仓库无对象级正式结果

判断:
本条要求的运行级审计、实验或正式产物没有发生；代码存在也不能替代执行，判定 NOT RUN。

需要修改:
保持未运行；阈值和 DBSCAN 只能从冻结 201 选择，随后用 fused score 正式评价。

### [E01-12] 状态: NOT RUN

要求:
TP。

发现:
阈值、逐帧 DBSCAN 和对象指标接口存在，但阈值/参数尚未从 201 冻结，所有对象级正式结果均未运行。

证据:
- src/evaluate.py:1272-1448
- src/evaluate.py:2303-2550
- 全仓库无对象级正式结果

判断:
本条要求的运行级审计、实验或正式产物没有发生；代码存在也不能替代执行，判定 NOT RUN。

需要修改:
保持未运行；阈值和 DBSCAN 只能从冻结 201 选择，随后用 fused score 正式评价。

### [E01-13] 状态: NOT RUN

要求:
FP。

发现:
阈值、逐帧 DBSCAN 和对象指标接口存在，但阈值/参数尚未从 201 冻结，所有对象级正式结果均未运行。

证据:
- src/evaluate.py:1272-1448
- src/evaluate.py:2303-2550
- 全仓库无对象级正式结果

判断:
本条要求的运行级审计、实验或正式产物没有发生；代码存在也不能替代执行，判定 NOT RUN。

需要修改:
保持未运行；阈值和 DBSCAN 只能从冻结 201 选择，随后用 fused score 正式评价。

### [E01-14] 状态: NOT RUN

要求:
FN。

发现:
阈值、逐帧 DBSCAN 和对象指标接口存在，但阈值/参数尚未从 201 冻结，所有对象级正式结果均未运行。

证据:
- src/evaluate.py:1272-1448
- src/evaluate.py:2303-2550
- 全仓库无对象级正式结果

判断:
本条要求的运行级审计、实验或正式产物没有发生；代码存在也不能替代执行，判定 NOT RUN。

需要修改:
保持未运行；阈值和 DBSCAN 只能从冻结 201 选择，随后用 fused score 正式评价。

## E02：不确定性与稳定性报告

### [E02-01] 状态: BLOCKED

要求:
三训练 seed 独立结果。

发现:
没有三种子、逐世界、配对差或逐序列结果；论文文字正确避免把点当独立重复，但这不替代稳定性产物。

证据:
- src/evaluate.py:564-606
- AJAE.tex:440
- 全仓库无三种子/逐世界结果

判断:
上游 Decision Gate 或冻结条件未满足，本条原则上不得继续；判定 BLOCKED。

需要修改:
保持 BLOCKED；三种子和逐世界结果出现后再做配对稳定性汇总。

### [E02-02] 状态: BLOCKED

要求:
mean/std 或等价稳定性汇总。

发现:
没有三种子、逐世界、配对差或逐序列结果；论文文字正确避免把点当独立重复，但这不替代稳定性产物。

证据:
- src/evaluate.py:564-606
- AJAE.tex:440
- 全仓库无三种子/逐世界结果

判断:
上游 Decision Gate 或冻结条件未满足，本条原则上不得继续；判定 BLOCKED。

需要修改:
保持 BLOCKED；三种子和逐世界结果出现后再做配对稳定性汇总。

### [E02-03] 状态: BLOCKED

要求:
24 fixed dev worlds 逐世界结果。

发现:
没有三种子、逐世界、配对差或逐序列结果；论文文字正确避免把点当独立重复，但这不替代稳定性产物。

证据:
- src/evaluate.py:564-606
- AJAE.tex:440
- 全仓库无三种子/逐世界结果

判断:
上游 Decision Gate 或冻结条件未满足，本条原则上不得继续；判定 BLOCKED。

需要修改:
保持 BLOCKED；三种子和逐世界结果出现后再做配对稳定性汇总。

### [E02-04] 状态: BLOCKED

要求:
B1/B2/B3/B4 逐世界 paired difference。

发现:
没有三种子、逐世界、配对差或逐序列结果；论文文字正确避免把点当独立重复，但这不替代稳定性产物。

证据:
- src/evaluate.py:564-606
- AJAE.tex:440
- 全仓库无三种子/逐世界结果

判断:
上游 Decision Gate 或冻结条件未满足，本条原则上不得继续；判定 BLOCKED。

需要修改:
保持 BLOCKED；三种子和逐世界结果出现后再做配对稳定性汇总。

### [E02-05] 状态: BLOCKED

要求:
19 real sequences 若已解锁，逐序列结果。

发现:
没有三种子、逐世界、配对差或逐序列结果；论文文字正确避免把点当独立重复，但这不替代稳定性产物。

证据:
- src/evaluate.py:564-606
- AJAE.tex:440
- 全仓库无三种子/逐世界结果

判断:
上游 Decision Gate 或冻结条件未满足，本条原则上不得继续；判定 BLOCKED。

需要修改:
保持 BLOCKED；三种子和逐世界结果出现后再做配对稳定性汇总。

### [E02-06] 状态: PASS

要求:
没有用“数百万个 point”作为独立重复来夸大统计可信度。

发现:
没有三种子、逐世界、配对差或逐序列结果；论文文字正确避免把点当独立重复，但这不替代稳定性产物。

证据:
- src/evaluate.py:564-606
- AJAE.tex:440
- 全仓库无三种子/逐世界结果

判断:
证据直接覆盖该要求，或排除性全仓库搜索已证明冲突路径不在当前正式路线；可判 PASS。该结论只覆盖本条，不外推到整个模块或科学主张。

需要修改:
无需修改；保留当前实现和直接证据，并继续确保该局部结论不被外推为模块或科学主张已经成立。

### [E02-07] 状态: BLOCKED

要求:
pooled official metric 与 world/sequence-level stability 同时存在。

发现:
没有三种子、逐世界、配对差或逐序列结果；论文文字正确避免把点当独立重复，但这不替代稳定性产物。

证据:
- src/evaluate.py:564-606
- AJAE.tex:440
- 全仓库无三种子/逐世界结果

判断:
上游 Decision Gate 或冻结条件未满足，本条原则上不得继续；判定 BLOCKED。

需要修改:
保持 BLOCKED；三种子和逐世界结果出现后再做配对稳定性汇总。

## K00：计算成本与公平性

### [K00-01] 状态: NOT RUN

要求:
报告 B1 inference cost。

发现:
推理路径能计时、记录显存/吞吐和缓存命中，但 B1/B3/B4/B5 均无真实成本结果，产物设计还缺 GPU 型号和完整 batch 条件。

证据:
- src/evaluate.py:1035-1269
- AJAE.tex:503
- 全仓库无成本结果产物

判断:
本条要求的运行级审计、实验或正式产物没有发生；代码存在也不能替代执行，判定 NOT RUN。

需要修改:
转为 PASS 所需条件：固定 GPU/device/batch/cache 条件实测 B1/B3/B4/B5，并明确时间输入差异。

### [K00-02] 状态: NOT RUN

要求:
报告 B3 window latency。

发现:
推理路径能计时、记录显存/吞吐和缓存命中，但 B1/B3/B4/B5 均无真实成本结果，产物设计还缺 GPU 型号和完整 batch 条件。

证据:
- src/evaluate.py:1035-1269
- AJAE.tex:503
- 全仓库无成本结果产物

判断:
本条要求的运行级审计、实验或正式产物没有发生；代码存在也不能替代执行，判定 NOT RUN。

需要修改:
转为 PASS 所需条件：固定 GPU/device/batch/cache 条件实测 B1/B3/B4/B5，并明确时间输入差异。

### [K00-03] 状态: NOT RUN

要求:
报告 B4 overall window/fusion cost。

发现:
推理路径能计时、记录显存/吞吐和缓存命中，但 B1/B3/B4/B5 均无真实成本结果，产物设计还缺 GPU 型号和完整 batch 条件。

证据:
- src/evaluate.py:1035-1269
- AJAE.tex:503
- 全仓库无成本结果产物

判断:
本条要求的运行级审计、实验或正式产物没有发生；代码存在也不能替代执行，判定 NOT RUN。

需要修改:
转为 PASS 所需条件：固定 GPU/device/batch/cache 条件实测 B1/B3/B4/B5，并明确时间输入差异。

### [K00-04] 状态: NOT RUN

要求:
报告 B5 causal latency。

发现:
推理路径能计时、记录显存/吞吐和缓存命中，但 B1/B3/B4/B5 均无真实成本结果，产物设计还缺 GPU 型号和完整 batch 条件。

证据:
- src/evaluate.py:1035-1269
- AJAE.tex:503
- 全仓库无成本结果产物

判断:
本条要求的运行级审计、实验或正式产物没有发生；代码存在也不能替代执行，判定 NOT RUN。

需要修改:
转为 PASS 所需条件：固定 GPU/device/batch/cache 条件实测 B1/B3/B4/B5，并明确时间输入差异。

### [K00-05] 状态: NOT RUN

要求:
报告 peak VRAM。

发现:
推理路径能计时、记录显存/吞吐和缓存命中，但 B1/B3/B4/B5 均无真实成本结果，产物设计还缺 GPU 型号和完整 batch 条件。

证据:
- src/evaluate.py:1035-1269
- AJAE.tex:503
- 全仓库无成本结果产物

判断:
本条要求的运行级审计、实验或正式产物没有发生；代码存在也不能替代执行，判定 NOT RUN。

需要修改:
转为 PASS 所需条件：固定 GPU/device/batch/cache 条件实测 B1/B3/B4/B5，并明确时间输入差异。

### [K00-06] 状态: NOT RUN

要求:
报告 throughput。

发现:
推理路径能计时、记录显存/吞吐和缓存命中，但 B1/B3/B4/B5 均无真实成本结果，产物设计还缺 GPU 型号和完整 batch 条件。

证据:
- src/evaluate.py:1035-1269
- AJAE.tex:503
- 全仓库无成本结果产物

判断:
本条要求的运行级审计、实验或正式产物没有发生；代码存在也不能替代执行，判定 NOT RUN。

需要修改:
转为 PASS 所需条件：固定 GPU/device/batch/cache 条件实测 B1/B3/B4/B5，并明确时间输入差异。

### [K00-07] 状态: NOT RUN

要求:
明确 STU frontend 是否 cached。

发现:
推理路径能计时、记录显存/吞吐和缓存命中，但 B1/B3/B4/B5 均无真实成本结果，产物设计还缺 GPU 型号和完整 batch 条件。

证据:
- src/evaluate.py:1035-1269
- AJAE.tex:503
- 全仓库无成本结果产物

判断:
本条要求的运行级审计、实验或正式产物没有发生；代码存在也不能替代执行，判定 NOT RUN。

需要修改:
转为 PASS 所需条件：固定 GPU/device/batch/cache 条件实测 B1/B3/B4/B5，并明确时间输入差异。

### [K00-08] 状态: NOT RUN

要求:
测量条件固定 GPU/device。

发现:
推理路径能计时、记录显存/吞吐和缓存命中，但 B1/B3/B4/B5 均无真实成本结果，产物设计还缺 GPU 型号和完整 batch 条件。

证据:
- src/evaluate.py:1035-1269
- AJAE.tex:503
- 全仓库无成本结果产物

判断:
本条要求的运行级审计、实验或正式产物没有发生；代码存在也不能替代执行，判定 NOT RUN。

需要修改:
转为 PASS 所需条件：固定 GPU/device/batch/cache 条件实测 B1/B3/B4/B5，并明确时间输入差异。

### [K00-09] 状态: NOT RUN

要求:
测量 batch 设置明确。

发现:
推理路径能计时、记录显存/吞吐和缓存命中，但 B1/B3/B4/B5 均无真实成本结果，产物设计还缺 GPU 型号和完整 batch 条件。

证据:
- src/evaluate.py:1035-1269
- AJAE.tex:503
- 全仓库无成本结果产物

判断:
本条要求的运行级审计、实验或正式产物没有发生；代码存在也不能替代执行，判定 NOT RUN。

需要修改:
转为 PASS 所需条件：固定 GPU/device/batch/cache 条件实测 B1/B3/B4/B5，并明确时间输入差异。

### [K00-10] 状态: PARTIAL

要求:
与 NDP/REL/LIDO 等单帧方法比较时明确 AJAE 使用 temporal inputs。

发现:
推理路径能计时、记录显存/吞吐和缓存命中，但 B1/B3/B4/B5 均无真实成本结果，产物设计还缺 GPU 型号和完整 batch 条件。

证据:
- src/evaluate.py:1035-1269
- AJAE.tex:503
- 全仓库无成本结果产物

判断:
只能证明静态实现、局部运行或缺少 生成链路 的产物，尚不能证明可追溯正式路径完整采用；按审计规则不得升级为 PASS。

需要修改:
转为 PASS 所需条件：固定 GPU/device/batch/cache 条件实测 B1/B3/B4/B5，并明确时间输入差异。

### [K00-11] 状态: PASS

要求:
明确 B3/B4 使用 future frames。

发现:
推理路径能计时、记录显存/吞吐和缓存命中，但 B1/B3/B4/B5 均无真实成本结果，产物设计还缺 GPU 型号和完整 batch 条件。

证据:
- src/evaluate.py:1035-1269
- AJAE.tex:503
- 全仓库无成本结果产物

判断:
证据直接覆盖该要求，或排除性全仓库搜索已证明冲突路径不在当前正式路线；可判 PASS。该结论只覆盖本条，不外推到整个模块或科学主张。

需要修改:
无需修改；保留当前实现和直接证据，并继续确保该局部结论不被外推为模块或科学主张已经成立。

### [K00-12] 状态: PASS

要求:
不得暗示和单帧 baseline 输入预算完全相同。

发现:
推理路径能计时、记录显存/吞吐和缓存命中，但 B1/B3/B4/B5 均无真实成本结果，产物设计还缺 GPU 型号和完整 batch 条件。

证据:
- src/evaluate.py:1035-1269
- AJAE.tex:503
- 全仓库无成本结果产物

判断:
证据直接覆盖该要求，或排除性全仓库搜索已证明冲突路径不在当前正式路线；可判 PASS。该结论只覆盖本条，不外推到整个模块或科学主张。

需要修改:
无需修改；保留当前实现和直接证据，并继续确保该局部结论不被外推为模块或科学主张已经成立。

## X00：废弃路线残留扫描

### [X00-01] 状态: PASS

要求:
当前帧 + 四历史帧主从结构。

发现:
对当前正式源码、协议、配置和文档进行了全仓库搜索与人工检查；旧 EMA、对象槽、Hungarian、tracking、`L_cf`、global-K 和在线中心五帧路线未进入 schema 30 正式路径。旧路线只在 Git 删除记录或明确失效文本中出现。

证据:
- exact command: `rg -n "EMA|L_cf|Hungarian|object slot|tracking|center.?only|last.?frame|global.?knn|query sum|smoothness|object head|realtime|online"`
- protocol.json schema 30
- Git 状态中的旧路线为已删除文件

判断:
证据直接覆盖该要求，或排除性全仓库搜索已证明冲突路径不在当前正式路线；可判 PASS。该结论只覆盖本条，不外推到整个模块或科学主张。

需要修改:
无需修改；保留当前实现和直接证据，并继续确保该局部结论不被外推为模块或科学主张已经成立。

### [X00-02] 状态: PASS

要求:
center-frame-only training。

发现:
对当前正式源码、协议、配置和文档进行了全仓库搜索与人工检查；旧 EMA、对象槽、Hungarian、tracking、`L_cf`、global-K 和在线中心五帧路线未进入 schema 30 正式路径。旧路线只在 Git 删除记录或明确失效文本中出现。

证据:
- exact command: `rg -n "EMA|L_cf|Hungarian|object slot|tracking|center.?only|last.?frame|global.?knn|query sum|smoothness|object head|realtime|online"`
- protocol.json schema 30
- Git 状态中的旧路线为已删除文件

判断:
证据直接覆盖该要求，或排除性全仓库搜索已证明冲突路径不在当前正式路线；可判 PASS。该结论只覆盖本条，不外推到整个模块或科学主张。

需要修改:
无需修改；保留当前实现和直接证据，并继续确保该局部结论不被外推为模块或科学主张已经成立。

### [X00-03] 状态: PASS

要求:
last-frame-only training。

发现:
对当前正式源码、协议、配置和文档进行了全仓库搜索与人工检查；旧 EMA、对象槽、Hungarian、tracking、`L_cf`、global-K 和在线中心五帧路线未进入 schema 30 正式路径。旧路线只在 Git 删除记录或明确失效文本中出现。

证据:
- exact command: `rg -n "EMA|L_cf|Hungarian|object slot|tracking|center.?only|last.?frame|global.?knn|query sum|smoothness|object head|realtime|online"`
- protocol.json schema 30
- Git 状态中的旧路线为已删除文件

判断:
证据直接覆盖该要求，或排除性全仓库搜索已证明冲突路径不在当前正式路线；可判 PASS。该结论只覆盖本条，不外推到整个模块或科学主张。

需要修改:
无需修改；保留当前实现和直接证据，并继续确保该局部结论不被外推为模块或科学主张已经成立。

### [X00-04] 状态: PASS

要求:
explicit object adapter。

发现:
对当前正式源码、协议、配置和文档进行了全仓库搜索与人工检查；旧 EMA、对象槽、Hungarian、tracking、`L_cf`、global-K 和在线中心五帧路线未进入 schema 30 正式路径。旧路线只在 Git 删除记录或明确失效文本中出现。

证据:
- exact command: `rg -n "EMA|L_cf|Hungarian|object slot|tracking|center.?only|last.?frame|global.?knn|query sum|smoothness|object head|realtime|online"`
- protocol.json schema 30
- Git 状态中的旧路线为已删除文件

判断:
证据直接覆盖该要求，或排除性全仓库搜索已证明冲突路径不在当前正式路线；可判 PASS。该结论只覆盖本条，不外推到整个模块或科学主张。

需要修改:
无需修改；保留当前实现和直接证据，并继续确保该局部结论不被外推为模块或科学主张已经成立。

### [X00-05] 状态: PASS

要求:
object slots。

发现:
对当前正式源码、协议、配置和文档进行了全仓库搜索与人工检查；旧 EMA、对象槽、Hungarian、tracking、`L_cf`、global-K 和在线中心五帧路线未进入 schema 30 正式路径。旧路线只在 Git 删除记录或明确失效文本中出现。

证据:
- exact command: `rg -n "EMA|L_cf|Hungarian|object slot|tracking|center.?only|last.?frame|global.?knn|query sum|smoothness|object head|realtime|online"`
- protocol.json schema 30
- Git 状态中的旧路线为已删除文件

判断:
证据直接覆盖该要求，或排除性全仓库搜索已证明冲突路径不在当前正式路线；可判 PASS。该结论只覆盖本条，不外推到整个模块或科学主张。

需要修改:
无需修改；保留当前实现和直接证据，并继续确保该局部结论不被外推为模块或科学主张已经成立。

### [X00-06] 状态: PASS

要求:
object ID prediction。

发现:
对当前正式源码、协议、配置和文档进行了全仓库搜索与人工检查；旧 EMA、对象槽、Hungarian、tracking、`L_cf`、global-K 和在线中心五帧路线未进入 schema 30 正式路径。旧路线只在 Git 删除记录或明确失效文本中出现。

证据:
- exact command: `rg -n "EMA|L_cf|Hungarian|object slot|tracking|center.?only|last.?frame|global.?knn|query sum|smoothness|object head|realtime|online"`
- protocol.json schema 30
- Git 状态中的旧路线为已删除文件

判断:
证据直接覆盖该要求，或排除性全仓库搜索已证明冲突路径不在当前正式路线；可判 PASS。该结论只覆盖本条，不外推到整个模块或科学主张。

需要修改:
无需修改；保留当前实现和直接证据，并继续确保该局部结论不被外推为模块或科学主张已经成立。

### [X00-07] 状态: PASS

要求:
Hungarian matching。

发现:
对当前正式源码、协议、配置和文档进行了全仓库搜索与人工检查；旧 EMA、对象槽、Hungarian、tracking、`L_cf`、global-K 和在线中心五帧路线未进入 schema 30 正式路径。旧路线只在 Git 删除记录或明确失效文本中出现。

证据:
- exact command: `rg -n "EMA|L_cf|Hungarian|object slot|tracking|center.?only|last.?frame|global.?knn|query sum|smoothness|object head|realtime|online"`
- protocol.json schema 30
- Git 状态中的旧路线为已删除文件

判断:
证据直接覆盖该要求，或排除性全仓库搜索已证明冲突路径不在当前正式路线；可判 PASS。该结论只覆盖本条，不外推到整个模块或科学主张。

需要修改:
无需修改；保留当前实现和直接证据，并继续确保该局部结论不被外推为模块或科学主张已经成立。

### [X00-08] 状态: PASS

要求:
cross-frame object tracking。

发现:
对当前正式源码、协议、配置和文档进行了全仓库搜索与人工检查；旧 EMA、对象槽、Hungarian、tracking、`L_cf`、global-K 和在线中心五帧路线未进入 schema 30 正式路径。旧路线只在 Git 删除记录或明确失效文本中出现。

证据:
- exact command: `rg -n "EMA|L_cf|Hungarian|object slot|tracking|center.?only|last.?frame|global.?knn|query sum|smoothness|object head|realtime|online"`
- protocol.json schema 30
- Git 状态中的旧路线为已删除文件

判断:
证据直接覆盖该要求，或排除性全仓库搜索已证明冲突路径不在当前正式路线；可判 PASS。该结论只覆盖本条，不外推到整个模块或科学主张。

需要修改:
无需修改；保留当前实现和直接证据，并继续确保该局部结论不被外推为模块或科学主张已经成立。

### [X00-09] 状态: PASS

要求:
object-level anomaly classification head。

发现:
对当前正式源码、协议、配置和文档进行了全仓库搜索与人工检查；旧 EMA、对象槽、Hungarian、tracking、`L_cf`、global-K 和在线中心五帧路线未进入 schema 30 正式路径。旧路线只在 Git 删除记录或明确失效文本中出现。

证据:
- exact command: `rg -n "EMA|L_cf|Hungarian|object slot|tracking|center.?only|last.?frame|global.?knn|query sum|smoothness|object head|realtime|online"`
- protocol.json schema 30
- Git 状态中的旧路线为已删除文件

判断:
证据直接覆盖该要求，或排除性全仓库搜索已证明冲突路径不在当前正式路线；可判 PASS。该结论只覆盖本条，不外推到整个模块或科学主张。

需要修改:
无需修改；保留当前实现和直接证据，并继续确保该局部结论不被外推为模块或科学主张已经成立。

### [X00-10] 状态: PASS

要求:
object score → point projection 作为主预测。

发现:
对当前正式源码、协议、配置和文档进行了全仓库搜索与人工检查；旧 EMA、对象槽、Hungarian、tracking、`L_cf`、global-K 和在线中心五帧路线未进入 schema 30 正式路径。旧路线只在 Git 删除记录或明确失效文本中出现。

证据:
- exact command: `rg -n "EMA|L_cf|Hungarian|object slot|tracking|center.?only|last.?frame|global.?knn|query sum|smoothness|object head|realtime|online"`
- protocol.json schema 30
- Git 状态中的旧路线为已删除文件

判断:
证据直接覆盖该要求，或排除性全仓库搜索已证明冲突路径不在当前正式路线；可判 PASS。该结论只覆盖本条，不外推到整个模块或科学主张。

需要修改:
无需修改；保留当前实现和直接证据，并继续确保该局部结论不被外推为模块或科学主张已经成立。

### [X00-11] 状态: PASS

要求:
point identity EMA memory。

发现:
对当前正式源码、协议、配置和文档进行了全仓库搜索与人工检查；旧 EMA、对象槽、Hungarian、tracking、`L_cf`、global-K 和在线中心五帧路线未进入 schema 30 正式路径。旧路线只在 Git 删除记录或明确失效文本中出现。

证据:
- exact command: `rg -n "EMA|L_cf|Hungarian|object slot|tracking|center.?only|last.?frame|global.?knn|query sum|smoothness|object head|realtime|online"`
- protocol.json schema 30
- Git 状态中的旧路线为已删除文件

判断:
证据直接覆盖该要求，或排除性全仓库搜索已证明冲突路径不在当前正式路线；可判 PASS。该结论只覆盖本条，不外推到整个模块或科学主张。

需要修改:
无需修改；保留当前实现和直接证据，并继续确保该局部结论不被外推为模块或科学主张已经成立。

### [X00-12] 状态: PASS

要求:
$L_{cf}$。

发现:
对当前正式源码、协议、配置和文档进行了全仓库搜索与人工检查；旧 EMA、对象槽、Hungarian、tracking、`L_cf`、global-K 和在线中心五帧路线未进入 schema 30 正式路径。旧路线只在 Git 删除记录或明确失效文本中出现。

证据:
- exact command: `rg -n "EMA|L_cf|Hungarian|object slot|tracking|center.?only|last.?frame|global.?knn|query sum|smoothness|object head|realtime|online"`
- protocol.json schema 30
- Git 状态中的旧路线为已删除文件

判断:
证据直接覆盖该要求，或排除性全仓库搜索已证明冲突路径不在当前正式路线；可判 PASS。该结论只覆盖本条，不外推到整个模块或科学主张。

需要修改:
无需修改；保留当前实现和直接证据，并继续确保该局部结论不被外推为模块或科学主张已经成立。

### [X00-13] 状态: PASS

要求:
explicit point smoothness loss。

发现:
对当前正式源码、协议、配置和文档进行了全仓库搜索与人工检查；旧 EMA、对象槽、Hungarian、tracking、`L_cf`、global-K 和在线中心五帧路线未进入 schema 30 正式路径。旧路线只在 Git 删除记录或明确失效文本中出现。

证据:
- exact command: `rg -n "EMA|L_cf|Hungarian|object slot|tracking|center.?only|last.?frame|global.?knn|query sum|smoothness|object head|realtime|online"`
- protocol.json schema 30
- Git 状态中的旧路线为已删除文件

判断:
证据直接覆盖该要求，或排除性全仓库搜索已证明冲突路径不在当前正式路线；可判 PASS。该结论只覆盖本条，不外推到整个模块或科学主张。

需要修改:
无需修改；保留当前实现和直接证据，并继续确保该局部结论不被外推为模块或科学主张已经成立。

### [X00-14] 状态: PASS

要求:
uncontrolled semantic evidence summation。

发现:
对当前正式源码、协议、配置和文档进行了全仓库搜索与人工检查；旧 EMA、对象槽、Hungarian、tracking、`L_cf`、global-K 和在线中心五帧路线未进入 schema 30 正式路径。旧路线只在 Git 删除记录或明确失效文本中出现。

证据:
- exact command: `rg -n "EMA|L_cf|Hungarian|object slot|tracking|center.?only|last.?frame|global.?knn|query sum|smoothness|object head|realtime|online"`
- protocol.json schema 30
- Git 状态中的旧路线为已删除文件

判断:
证据直接覆盖该要求，或排除性全仓库搜索已证明冲突路径不在当前正式路线；可判 PASS。该结论只覆盖本条，不外推到整个模块或科学主张。

需要修改:
无需修改；保留当前实现和直接证据，并继续确保该局部结论不被外推为模块或科学主张已经成立。

### [X00-15] 状态: PASS

要求:
five-frame global-K neighborhood。

发现:
对当前正式源码、协议、配置和文档进行了全仓库搜索与人工检查；旧 EMA、对象槽、Hungarian、tracking、`L_cf`、global-K 和在线中心五帧路线未进入 schema 30 正式路径。旧路线只在 Git 删除记录或明确失效文本中出现。

证据:
- exact command: `rg -n "EMA|L_cf|Hungarian|object slot|tracking|center.?only|last.?frame|global.?knn|query sum|smoothness|object head|realtime|online"`
- protocol.json schema 30
- Git 状态中的旧路线为已删除文件

判断:
证据直接覆盖该要求，或排除性全仓库搜索已证明冲突路径不在当前正式路线；可判 PASS。该结论只覆盖本条，不外推到整个模块或科学主张。

需要修改:
无需修改；保留当前实现和直接证据，并继续确保该局部结论不被外推为模块或科学主张已经成立。

### [X00-16] 状态: PASS

要求:
realtime/online claim for centered five-frame model。

发现:
对当前正式源码、协议、配置和文档进行了全仓库搜索与人工检查；旧 EMA、对象槽、Hungarian、tracking、`L_cf`、global-K 和在线中心五帧路线未进入 schema 30 正式路径。旧路线只在 Git 删除记录或明确失效文本中出现。

证据:
- exact command: `rg -n "EMA|L_cf|Hungarian|object slot|tracking|center.?only|last.?frame|global.?knn|query sum|smoothness|object head|realtime|online"`
- protocol.json schema 30
- Git 状态中的旧路线为已删除文件

判断:
证据直接覆盖该要求，或排除性全仓库搜索已证明冲突路径不在当前正式路线；可判 PASS。该结论只覆盖本条，不外推到整个模块或科学主张。

需要修改:
无需修改；保留当前实现和直接证据，并继续确保该局部结论不被外推为模块或科学主张已经成立。

### [X00-17] 状态: PASS

要求:
把 synthetic anomaly proxy 直接称为已证明 real OOD。

发现:
对当前正式源码、协议、配置和文档进行了全仓库搜索与人工检查；旧 EMA、对象槽、Hungarian、tracking、`L_cf`、global-K 和在线中心五帧路线未进入 schema 30 正式路径。旧路线只在 Git 删除记录或明确失效文本中出现。

证据:
- exact command: `rg -n "EMA|L_cf|Hungarian|object slot|tracking|center.?only|last.?frame|global.?knn|query sum|smoothness|object head|realtime|online"`
- protocol.json schema 30
- Git 状态中的旧路线为已删除文件

判断:
证据直接覆盖该要求，或排除性全仓库搜索已证明冲突路径不在当前正式路线；可判 PASS。该结论只覆盖本条，不外推到整个模块或科学主张。

需要修改:
无需修改；保留当前实现和直接证据，并继续确保该局部结论不被外推为模块或科学主张已经成立。

### [X00-18] 状态: PASS

要求:
如果存在旧配置，确认不能被误选为 default。

发现:
对当前正式源码、协议、配置和文档进行了全仓库搜索与人工检查；旧 EMA、对象槽、Hungarian、tracking、`L_cf`、global-K 和在线中心五帧路线未进入 schema 30 正式路径。旧路线只在 Git 删除记录或明确失效文本中出现。

证据:
- exact command: `rg -n "EMA|L_cf|Hungarian|object slot|tracking|center.?only|last.?frame|global.?knn|query sum|smoothness|object head|realtime|online"`
- protocol.json schema 30
- Git 状态中的旧路线为已删除文件

判断:
证据直接覆盖该要求，或排除性全仓库搜索已证明冲突路径不在当前正式路线；可判 PASS。该结论只覆盖本条，不外推到整个模块或科学主张。

需要修改:
无需修改；保留当前实现和直接证据，并继续确保该局部结论不被外推为模块或科学主张已经成立。

### [X00-19] 状态: N/A

要求:
如果旧 checkpoint 存在，明确标记其协议版本。

发现:
对当前正式源码、协议、配置和文档进行了全仓库搜索与人工检查；旧 EMA、对象槽、Hungarian、tracking、`L_cf`、global-K 和在线中心五帧路线未进入 schema 30 正式路径。旧路线只在 Git 删除记录或明确失效文本中出现。

证据:
- exact command: `rg -n "EMA|L_cf|Hungarian|object slot|tracking|center.?only|last.?frame|global.?knn|query sum|smoothness|object head|realtime|online"`
- protocol.json schema 30
- Git 状态中的旧路线为已删除文件

判断:
当前阶段尚未触发该条件，主线明确允许不执行；已说明适用边界，判定 N/A。

需要修改:
当前阶段条件尚未触发，无需修改；若未来触发，必须重新按本条要求审计。

## CFG：配置真实性

### [CFG-01] 状态: NOT RUN

要求:
有唯一 experiment config。

发现:
不存在任何 schema 30 正式实验 resolved config。未来 payload 设计含部分协议身份和 seed，但缺 Git commit、有效批大小和 exact CLI，且当前核心 schema 30 文件未提交。

证据:
- src/train.py:290-366
- src/train.py:1095-1498
- git status --short（schema 30 核心文件未提交）

判断:
本条要求的运行级审计、实验或正式产物没有发生；代码存在也不能替代执行，判定 NOT RUN。

需要修改:
转为 PASS 所需条件：每次正式运行保存唯一 resolved config、Git commit、exact CLI、seed 和所有运行身份。

### [CFG-02] 状态: NOT RUN

要求:
config 保存到结果目录。

发现:
不存在任何 schema 30 正式实验 resolved config。未来 payload 设计含部分协议身份和 seed，但缺 Git commit、有效批大小和 exact CLI，且当前核心 schema 30 文件未提交。

证据:
- src/train.py:290-366
- src/train.py:1095-1498
- git status --short（schema 30 核心文件未提交）

判断:
本条要求的运行级审计、实验或正式产物没有发生；代码存在也不能替代执行，判定 NOT RUN。

需要修改:
转为 PASS 所需条件：每次正式运行保存唯一 resolved config、Git commit、exact CLI、seed 和所有运行身份。

### [CFG-03] 状态: FAIL

要求:
config 包含 git commit。

发现:
未来实验 payload 不保存 Git commit；当前 schema 30 核心文件还是未跟踪/未提交状态。

证据:
- src/train.py:290-366
- src/train.py:1095-1498
- git status --short（schema 30 核心文件未提交）

判断:
存在与要求直接冲突的实现、数据或真实运行事实；判定 FAIL。

需要修改:
正式 config 必须记录 clean Git commit；当前未提交 schema 30 状态不能作为可复现实验基线。

### [CFG-04] 状态: NOT RUN

要求:
config 包含 random seed。

发现:
不存在任何 schema 30 正式实验 resolved config。未来 payload 设计含部分协议身份和 seed，但缺 Git commit、有效批大小和 exact CLI，且当前核心 schema 30 文件未提交。

证据:
- src/train.py:290-366
- src/train.py:1095-1498
- git status --short（schema 30 核心文件未提交）

判断:
本条要求的运行级审计、实验或正式产物没有发生；代码存在也不能替代执行，判定 NOT RUN。

需要修改:
转为 PASS 所需条件：每次正式运行保存唯一 resolved config、Git commit、exact CLI、seed 和所有运行身份。

### [CFG-05] 状态: NOT RUN

要求:
config 包含 dataset split。

发现:
不存在任何 schema 30 正式实验 resolved config。未来 payload 设计含部分协议身份和 seed，但缺 Git commit、有效批大小和 exact CLI，且当前核心 schema 30 文件未提交。

证据:
- src/train.py:290-366
- src/train.py:1095-1498
- git status --short（schema 30 核心文件未提交）

判断:
本条要求的运行级审计、实验或正式产物没有发生；代码存在也不能替代执行，判定 NOT RUN。

需要修改:
转为 PASS 所需条件：每次正式运行保存唯一 resolved config、Git commit、exact CLI、seed 和所有运行身份。

### [CFG-06] 状态: PARTIAL

要求:
config 包含 world-generator version。

发现:
不存在任何 schema 30 正式实验 resolved config。未来 payload 设计含部分协议身份和 seed，但缺 Git commit、有效批大小和 exact CLI，且当前核心 schema 30 文件未提交。

证据:
- src/train.py:290-366
- src/train.py:1095-1498
- git status --short（schema 30 核心文件未提交）

判断:
只能证明静态实现、局部运行或缺少 生成链路 的产物，尚不能证明可追溯正式路径完整采用；按审计规则不得升级为 PASS。

需要修改:
转为 PASS 所需条件：每次正式运行保存唯一 resolved config、Git commit、exact CLI、seed 和所有运行身份。

### [CFG-07] 状态: PARTIAL

要求:
config 包含 renderer version。

发现:
不存在任何 schema 30 正式实验 resolved config。未来 payload 设计含部分协议身份和 seed，但缺 Git commit、有效批大小和 exact CLI，且当前核心 schema 30 文件未提交。

证据:
- src/train.py:290-366
- src/train.py:1095-1498
- git status --short（schema 30 核心文件未提交）

判断:
只能证明静态实现、局部运行或缺少 生成链路 的产物，尚不能证明可追溯正式路径完整采用；按审计规则不得升级为 PASS。

需要修改:
转为 PASS 所需条件：每次正式运行保存唯一 resolved config、Git commit、exact CLI、seed 和所有运行身份。

### [CFG-08] 状态: PARTIAL

要求:
config 包含 STU checkpoint hash/path。

发现:
不存在任何 schema 30 正式实验 resolved config。未来 payload 设计含部分协议身份和 seed，但缺 Git commit、有效批大小和 exact CLI，且当前核心 schema 30 文件未提交。

证据:
- src/train.py:290-366
- src/train.py:1095-1498
- git status --short（schema 30 核心文件未提交）

判断:
只能证明静态实现、局部运行或缺少 生成链路 的产物，尚不能证明可追溯正式路径完整采用；按审计规则不得升级为 PASS。

需要修改:
转为 PASS 所需条件：每次正式运行保存唯一 resolved config、Git commit、exact CLI、seed 和所有运行身份。

### [CFG-09] 状态: NOT RUN

要求:
config 包含 model architecture。

发现:
不存在任何 schema 30 正式实验 resolved config。未来 payload 设计含部分协议身份和 seed，但缺 Git commit、有效批大小和 exact CLI，且当前核心 schema 30 文件未提交。

证据:
- src/train.py:290-366
- src/train.py:1095-1498
- git status --short（schema 30 核心文件未提交）

判断:
本条要求的运行级审计、实验或正式产物没有发生；代码存在也不能替代执行，判定 NOT RUN。

需要修改:
转为 PASS 所需条件：每次正式运行保存唯一 resolved config、Git commit、exact CLI、seed 和所有运行身份。

### [CFG-10] 状态: NOT RUN

要求:
config 包含 voxel scales。

发现:
不存在任何 schema 30 正式实验 resolved config。未来 payload 设计含部分协议身份和 seed，但缺 Git commit、有效批大小和 exact CLI，且当前核心 schema 30 文件未提交。

证据:
- src/train.py:290-366
- src/train.py:1095-1498
- git status --short（schema 30 核心文件未提交）

判断:
本条要求的运行级审计、实验或正式产物没有发生；代码存在也不能替代执行，判定 NOT RUN。

需要修改:
转为 PASS 所需条件：每次正式运行保存唯一 resolved config、Git commit、exact CLI、seed 和所有运行身份。

### [CFG-11] 状态: NOT RUN

要求:
config 包含 per-level/per-delta radii。

发现:
不存在任何 schema 30 正式实验 resolved config。未来 payload 设计含部分协议身份和 seed，但缺 Git commit、有效批大小和 exact CLI，且当前核心 schema 30 文件未提交。

证据:
- src/train.py:290-366
- src/train.py:1095-1498
- git status --short（schema 30 核心文件未提交）

判断:
本条要求的运行级审计、实验或正式产物没有发生；代码存在也不能替代执行，判定 NOT RUN。

需要修改:
转为 PASS 所需条件：每次正式运行保存唯一 resolved config、Git commit、exact CLI、seed 和所有运行身份。

### [CFG-12] 状态: NOT RUN

要求:
config 包含 per-level/per-delta K。

发现:
不存在任何 schema 30 正式实验 resolved config。未来 payload 设计含部分协议身份和 seed，但缺 Git commit、有效批大小和 exact CLI，且当前核心 schema 30 文件未提交。

证据:
- src/train.py:290-366
- src/train.py:1095-1498
- git status --short（schema 30 核心文件未提交）

判断:
本条要求的运行级审计、实验或正式产物没有发生；代码存在也不能替代执行，判定 NOT RUN。

需要修改:
转为 PASS 所需条件：每次正式运行保存唯一 resolved config、Git commit、exact CLI、seed 和所有运行身份。

### [CFG-13] 状态: PARTIAL

要求:
config 包含 optimizer。

发现:
不存在任何 schema 30 正式实验 resolved config。未来 payload 设计含部分协议身份和 seed，但缺 Git commit、有效批大小和 exact CLI，且当前核心 schema 30 文件未提交。

证据:
- src/train.py:290-366
- src/train.py:1095-1498
- git status --short（schema 30 核心文件未提交）

判断:
只能证明静态实现、局部运行或缺少 生成链路 的产物，尚不能证明可追溯正式路径完整采用；按审计规则不得升级为 PASS。

需要修改:
转为 PASS 所需条件：每次正式运行保存唯一 resolved config、Git commit、exact CLI、seed 和所有运行身份。

### [CFG-14] 状态: NOT RUN

要求:
config 包含 LR。

发现:
不存在任何 schema 30 正式实验 resolved config。未来 payload 设计含部分协议身份和 seed，但缺 Git commit、有效批大小和 exact CLI，且当前核心 schema 30 文件未提交。

证据:
- src/train.py:290-366
- src/train.py:1095-1498
- git status --short（schema 30 核心文件未提交）

判断:
本条要求的运行级审计、实验或正式产物没有发生；代码存在也不能替代执行，判定 NOT RUN。

需要修改:
转为 PASS 所需条件：每次正式运行保存唯一 resolved config、Git commit、exact CLI、seed 和所有运行身份。

### [CFG-15] 状态: NOT RUN

要求:
config 包含 accumulation steps。

发现:
不存在任何 schema 30 正式实验 resolved config。未来 payload 设计含部分协议身份和 seed，但缺 Git commit、有效批大小和 exact CLI，且当前核心 schema 30 文件未提交。

证据:
- src/train.py:290-366
- src/train.py:1095-1498
- git status --short（schema 30 核心文件未提交）

判断:
本条要求的运行级审计、实验或正式产物没有发生；代码存在也不能替代执行，判定 NOT RUN。

需要修改:
转为 PASS 所需条件：每次正式运行保存唯一 resolved config、Git commit、exact CLI、seed 和所有运行身份。

### [CFG-16] 状态: FAIL

要求:
config 包含 effective batch。

发现:
没有显式 `effective_batch` 字段，只能从 micro-batch 与 accumulation 推导，不能审计 runtime-resolved 值。

证据:
- src/train.py:290-366
- src/train.py:1095-1498
- git status --short（schema 30 核心文件未提交）

判断:
存在与要求直接冲突的实现、数据或真实运行事实；判定 FAIL。

需要修改:
在 resolved config 中显式保存 effective batch 及其计算成分。

### [CFG-17] 状态: NOT RUN

要求:
config 包含 distance range。

发现:
不存在任何 schema 30 正式实验 resolved config。未来 payload 设计含部分协议身份和 seed，但缺 Git commit、有效批大小和 exact CLI，且当前核心 schema 30 文件未提交。

证据:
- src/train.py:290-366
- src/train.py:1095-1498
- git status --short（schema 30 核心文件未提交）

判断:
本条要求的运行级审计、实验或正式产物没有发生；代码存在也不能替代执行，判定 NOT RUN。

需要修改:
转为 PASS 所需条件：每次正式运行保存唯一 resolved config、Git commit、exact CLI、seed 和所有运行身份。

### [CFG-18] 状态: NOT RUN

要求:
config 包含 fusion mode。

发现:
不存在任何 schema 30 正式实验 resolved config。未来 payload 设计含部分协议身份和 seed，但缺 Git commit、有效批大小和 exact CLI，且当前核心 schema 30 文件未提交。

证据:
- src/train.py:290-366
- src/train.py:1095-1498
- git status --short（schema 30 核心文件未提交）

判断:
本条要求的运行级审计、实验或正式产物没有发生；代码存在也不能替代执行，判定 NOT RUN。

需要修改:
转为 PASS 所需条件：每次正式运行保存唯一 resolved config、Git commit、exact CLI、seed 和所有运行身份。

### [CFG-19] 状态: NOT RUN

要求:
config 包含 evaluator mode。

发现:
不存在任何 schema 30 正式实验 resolved config。未来 payload 设计含部分协议身份和 seed，但缺 Git commit、有效批大小和 exact CLI，且当前核心 schema 30 文件未提交。

证据:
- src/train.py:290-366
- src/train.py:1095-1498
- git status --short（schema 30 核心文件未提交）

判断:
本条要求的运行级审计、实验或正式产物没有发生；代码存在也不能替代执行，判定 NOT RUN。

需要修改:
转为 PASS 所需条件：每次正式运行保存唯一 resolved config、Git commit、exact CLI、seed 和所有运行身份。

### [CFG-20] 状态: NOT RUN

要求:
config 与 runtime-resolved config 一致。

发现:
不存在任何 schema 30 正式实验 resolved config。未来 payload 设计含部分协议身份和 seed，但缺 Git commit、有效批大小和 exact CLI，且当前核心 schema 30 文件未提交。

证据:
- src/train.py:290-366
- src/train.py:1095-1498
- git status --short（schema 30 核心文件未提交）

判断:
本条要求的运行级审计、实验或正式产物没有发生；代码存在也不能替代执行，判定 NOT RUN。

需要修改:
转为 PASS 所需条件：每次正式运行保存唯一 resolved config、Git commit、exact CLI、seed 和所有运行身份。

### [CFG-21] 状态: FAIL

要求:
CLI override 被完整记录。

发现:
不保存 exact CLI，`device`、`resume` 等覆盖值无法完整追溯。

证据:
- src/train.py:290-366
- src/train.py:1095-1498
- git status --short（schema 30 核心文件未提交）

判断:
存在与要求直接冲突的实现、数据或真实运行事实；判定 FAIL。

需要修改:
保存原始命令和全部 CLI override 的 runtime-resolved 值。

### [CFG-22] 状态: PARTIAL

要求:
无“默认参数悄悄覆盖 YAML”的情况。

发现:
不存在任何 schema 30 正式实验 resolved config。未来 payload 设计含部分协议身份和 seed，但缺 Git commit、有效批大小和 exact CLI，且当前核心 schema 30 文件未提交。

证据:
- src/train.py:290-366
- src/train.py:1095-1498
- git status --short（schema 30 核心文件未提交）

判断:
只能证明静态实现、局部运行或缺少 生成链路 的产物，尚不能证明可追溯正式路径完整采用；按审计规则不得升级为 PASS。

需要修改:
转为 PASS 所需条件：每次正式运行保存唯一 resolved config、Git commit、exact CLI、seed 和所有运行身份。

## REP：实验产物真实性与可复现性

### [REP-01] 状态: BLOCKED

要求:
有启动命令。

发现:
没有任何 schema 30 B0–B5 正式实验目录、日志、配置、检查点、预测或评价输出；`runs/ajae` 只有被 `.gitignore` 忽略的 `calibration.pt`。

证据:
- runs/ajae/calibration.pt
- .gitignore:5
- git status --short（旧 oracle 产物已删除）

判断:
上游 Decision Gate 或冻结条件未满足，本条原则上不得继续；判定 BLOCKED。

需要修改:
保持 BLOCKED；正式实验解锁后必须原位生成完整 生成链路，不得用旧 oracle 结果替代。

### [REP-02] 状态: BLOCKED

要求:
有完整 stdout/stderr 日志。

发现:
没有任何 schema 30 B0–B5 正式实验目录、日志、配置、检查点、预测或评价输出；`runs/ajae` 只有被 `.gitignore` 忽略的 `calibration.pt`。

证据:
- runs/ajae/calibration.pt
- .gitignore:5
- git status --short（旧 oracle 产物已删除）

判断:
上游 Decision Gate 或冻结条件未满足，本条原则上不得继续；判定 BLOCKED。

需要修改:
保持 BLOCKED；正式实验解锁后必须原位生成完整 生成链路，不得用旧 oracle 结果替代。

### [REP-03] 状态: BLOCKED

要求:
有 resolved config。

发现:
没有任何 schema 30 B0–B5 正式实验目录、日志、配置、检查点、预测或评价输出；`runs/ajae` 只有被 `.gitignore` 忽略的 `calibration.pt`。

证据:
- runs/ajae/calibration.pt
- .gitignore:5
- git status --short（旧 oracle 产物已删除）

判断:
上游 Decision Gate 或冻结条件未满足，本条原则上不得继续；判定 BLOCKED。

需要修改:
保持 BLOCKED；正式实验解锁后必须原位生成完整 生成链路，不得用旧 oracle 结果替代。

### [REP-04] 状态: BLOCKED

要求:
有 git commit。

发现:
没有任何 schema 30 B0–B5 正式实验目录、日志、配置、检查点、预测或评价输出；`runs/ajae` 只有被 `.gitignore` 忽略的 `calibration.pt`。

证据:
- runs/ajae/calibration.pt
- .gitignore:5
- git status --short（旧 oracle 产物已删除）

判断:
上游 Decision Gate 或冻结条件未满足，本条原则上不得继续；判定 BLOCKED。

需要修改:
保持 BLOCKED；正式实验解锁后必须原位生成完整 生成链路，不得用旧 oracle 结果替代。

### [REP-05] 状态: BLOCKED

要求:
有 seed。

发现:
没有任何 schema 30 B0–B5 正式实验目录、日志、配置、检查点、预测或评价输出；`runs/ajae` 只有被 `.gitignore` 忽略的 `calibration.pt`。

证据:
- runs/ajae/calibration.pt
- .gitignore:5
- git status --short（旧 oracle 产物已删除）

判断:
上游 Decision Gate 或冻结条件未满足，本条原则上不得继续；判定 BLOCKED。

需要修改:
保持 BLOCKED；正式实验解锁后必须原位生成完整 生成链路，不得用旧 oracle 结果替代。

### [REP-06] 状态: BLOCKED

要求:
有 checkpoint。

发现:
没有任何 schema 30 B0–B5 正式实验目录、日志、配置、检查点、预测或评价输出；`runs/ajae` 只有被 `.gitignore` 忽略的 `calibration.pt`。

证据:
- runs/ajae/calibration.pt
- .gitignore:5
- git status --short（旧 oracle 产物已删除）

判断:
上游 Decision Gate 或冻结条件未满足，本条原则上不得继续；判定 BLOCKED。

需要修改:
保持 BLOCKED；正式实验解锁后必须原位生成完整 生成链路，不得用旧 oracle 结果替代。

### [REP-07] 状态: BLOCKED

要求:
有 checkpoint-selection 依据。

发现:
没有任何 schema 30 B0–B5 正式实验目录、日志、配置、检查点、预测或评价输出；`runs/ajae` 只有被 `.gitignore` 忽略的 `calibration.pt`。

证据:
- runs/ajae/calibration.pt
- .gitignore:5
- git status --short（旧 oracle 产物已删除）

判断:
上游 Decision Gate 或冻结条件未满足，本条原则上不得继续；判定 BLOCKED。

需要修改:
保持 BLOCKED；正式实验解锁后必须原位生成完整 生成链路，不得用旧 oracle 结果替代。

### [REP-08] 状态: BLOCKED

要求:
有预测文件。

发现:
没有任何 schema 30 B0–B5 正式实验目录、日志、配置、检查点、预测或评价输出；`runs/ajae` 只有被 `.gitignore` 忽略的 `calibration.pt`。

证据:
- runs/ajae/calibration.pt
- .gitignore:5
- git status --short（旧 oracle 产物已删除）

判断:
上游 Decision Gate 或冻结条件未满足，本条原则上不得继续；判定 BLOCKED。

需要修改:
保持 BLOCKED；正式实验解锁后必须原位生成完整 生成链路，不得用旧 oracle 结果替代。

### [REP-09] 状态: BLOCKED

要求:
有 evaluator 输出。

发现:
没有任何 schema 30 B0–B5 正式实验目录、日志、配置、检查点、预测或评价输出；`runs/ajae` 只有被 `.gitignore` 忽略的 `calibration.pt`。

证据:
- runs/ajae/calibration.pt
- .gitignore:5
- git status --short（旧 oracle 产物已删除）

判断:
上游 Decision Gate 或冻结条件未满足，本条原则上不得继续；判定 BLOCKED。

需要修改:
保持 BLOCKED；正式实验解锁后必须原位生成完整 生成链路，不得用旧 oracle 结果替代。

### [REP-10] 状态: BLOCKED

要求:
有 per-world / per-sequence 输出。

发现:
没有任何 schema 30 B0–B5 正式实验目录、日志、配置、检查点、预测或评价输出；`runs/ajae` 只有被 `.gitignore` 忽略的 `calibration.pt`。

证据:
- runs/ajae/calibration.pt
- .gitignore:5
- git status --short（旧 oracle 产物已删除）

判断:
上游 Decision Gate 或冻结条件未满足，本条原则上不得继续；判定 BLOCKED。

需要修改:
保持 BLOCKED；正式实验解锁后必须原位生成完整 生成链路，不得用旧 oracle 结果替代。

### [REP-11] 状态: BLOCKED

要求:
能从 checkpoint 重跑至少一个固定 201 world 并复现指标。

发现:
没有任何 schema 30 B0–B5 正式实验目录、日志、配置、检查点、预测或评价输出；`runs/ajae` 只有被 `.gitignore` 忽略的 `calibration.pt`。

证据:
- runs/ajae/calibration.pt
- .gitignore:5
- git status --short（旧 oracle 产物已删除）

判断:
上游 Decision Gate 或冻结条件未满足，本条原则上不得继续；判定 BLOCKED。

需要修改:
保持 BLOCKED；正式实验解锁后必须原位生成完整 生成链路，不得用旧 oracle 结果替代。

### [REP-12] 状态: BLOCKED

要求:
训练结果目录没有被后来实验覆盖。

发现:
没有任何 schema 30 B0–B5 正式实验目录、日志、配置、检查点、预测或评价输出；`runs/ajae` 只有被 `.gitignore` 忽略的 `calibration.pt`。

证据:
- runs/ajae/calibration.pt
- .gitignore:5
- git status --short（旧 oracle 产物已删除）

判断:
上游 Decision Gate 或冻结条件未满足，本条原则上不得继续；判定 BLOCKED。

需要修改:
保持 BLOCKED；正式实验解锁后必须原位生成完整 生成链路，不得用旧 oracle 结果替代。

### [REP-13] 状态: BLOCKED

要求:
文件命名能区分 B1/B2/B3/B4/B5。

发现:
没有任何 schema 30 B0–B5 正式实验目录、日志、配置、检查点、预测或评价输出；`runs/ajae` 只有被 `.gitignore` 忽略的 `calibration.pt`。

证据:
- runs/ajae/calibration.pt
- .gitignore:5
- git status --short（旧 oracle 产物已删除）

判断:
上游 Decision Gate 或冻结条件未满足，本条原则上不得继续；判定 BLOCKED。

需要修改:
保持 BLOCKED；正式实验解锁后必须原位生成完整 生成链路，不得用旧 oracle 结果替代。

### [REP-14] 状态: BLOCKED

要求:
文件命名能区分 seed。

发现:
没有任何 schema 30 B0–B5 正式实验目录、日志、配置、检查点、预测或评价输出；`runs/ajae` 只有被 `.gitignore` 忽略的 `calibration.pt`。

证据:
- runs/ajae/calibration.pt
- .gitignore:5
- git status --short（旧 oracle 产物已删除）

判断:
上游 Decision Gate 或冻结条件未满足，本条原则上不得继续；判定 BLOCKED。

需要修改:
保持 BLOCKED；正式实验解锁后必须原位生成完整 生成链路，不得用旧 oracle 结果替代。

### [REP-15] 状态: BLOCKED

要求:
没有手工复制/改名造成 lineage 不明的结果。

发现:
没有任何 schema 30 B0–B5 正式实验目录、日志、配置、检查点、预测或评价输出；`runs/ajae` 只有被 `.gitignore` 忽略的 `calibration.pt`。

证据:
- runs/ajae/calibration.pt
- .gitignore:5
- git status --short（旧 oracle 产物已删除）

判断:
上游 Decision Gate 或冻结条件未满足，本条原则上不得继续；判定 BLOCKED。

需要修改:
保持 BLOCKED；正式实验解锁后必须原位生成完整 生成链路，不得用旧 oracle 结果替代。

## 最小运行级检查 U00–U12

### [U00] 状态: PASS

要求:
渲染器确定性：同一世界规格和帧重复渲染时，输出必须逐槽一致。

发现:
真实 control-only 三帧重复渲染记录为 0 mismatch，小夹具也重复通过。

证据:
- dev.json:10942-10958
- test_ajae.py:535-545

判断:
证据直接覆盖该要求，或排除性全仓库搜索已证明冲突路径不在当前正式路线；可判 PASS。该结论只覆盖本条，不外推到整个模块或科学主张。

需要修改:
无需修改；保留当前实现和直接证据，并继续确保该局部结论不被外推为模块或科学主张已经成立。

### [U01] 状态: NOT RUN

要求:
重叠窗口帧一致性：同一世界、同一共享帧经不同窗口请求时必须逐槽一致。

发现:
未发现跨至少三个世界、每个至少五个重叠窗口的共享帧逐槽检查。

证据:
- test_ajae.py（未发现对应测试）

判断:
本条要求的运行级审计、实验或正式产物没有发生；代码存在也不能替代执行，判定 NOT RUN。

需要修改:
转为 PASS 所需条件：执行并保存该测试要求的直接数值断言；本次审计没有代替开发补测。

### [U02] 状态: PARTIAL

要求:
规范射线往返：真实回波经 canonical ray grid 往返后恢复数量、方向和距离。

发现:
train/206 全 449 帧往返数值优秀，但仅一个序列且缺完整 生成链路，因此保守判 PARTIAL。

证据:
- dev.json:8363-8368
- artifact: 449 帧、56,196,767 回波

判断:
只能证明静态实现、局部运行或缺少 生成链路 的产物，尚不能证明可追溯正式路径完整采用；按审计规则不得升级为 PASS。

需要修改:
转为 PASS 所需条件：执行并保存该测试要求的直接数值断言；本次审计没有代替开发补测。

### [U03] 状态: NOT RUN

要求:
遮挡：分别验证插入物在前替换背景、原前景在前保留原回波。

发现:
没有构造两种有向遮挡夹具。

证据:
- test_ajae.py（未发现对应测试）

判断:
本条要求的运行级审计、实验或正式产物没有发生；代码存在也不能替代执行，判定 NOT RUN。

需要修改:
转为 PASS 所需条件：执行并保存该测试要求的直接数值断言；本次审计没有代替开发补测。

### [U04] 状态: NOT RUN

要求:
空射线：分别验证接受回波时生成新点、拒绝回波时保持为空。

发现:
没有 empty-ray accepted/rejected 两分支夹具。

证据:
- test_ajae.py（未发现对应测试）

判断:
本条要求的运行级审计、实验或正式产物没有发生；代码存在也不能替代执行，判定 NOT RUN。

需要修改:
转为 PASS 所需条件：执行并保存该测试要求的直接数值断言；本次审计没有代替开发补测。

### [U05] 状态: NOT RUN

要求:
STU 冻结：一次反向/更新前后状态散列一致且全部梯度为空。

发现:
官方 STU 当前不能构造，也没有 optimizer step 前后 hash/grad 检查。

证据:
- src/model.py:372-563
- 真实 STU 构造失败：No module named hydra

判断:
本条要求的运行级审计、实验或正式产物没有发生；代码存在也不能替代执行，判定 NOT RUN。

需要修改:
转为 PASS 所需条件：执行并保存该测试要求的直接数值断言；本次审计没有代替开发补测。

### [U06] 状态: NOT RUN

要求:
时间体素分离：相同坐标但不同 q 的点不得落入同一体素。

发现:
体素键代码含 q，但没有同坐标异 q 的直接断言。

证据:
- src/model.py:845-885
- test_ajae.py（未发现对应测试）

判断:
本条要求的运行级审计、实验或正式产物没有发生；代码存在也不能替代执行，判定 NOT RUN。

需要修改:
转为 PASS 所需条件：执行并保存该测试要求的直接数值断言；本次审计没有代替开发补测。

### [U07] 状态: PASS

要求:
分层 KNN：指定 delta 的候选不得被其他时间位置占用配额。

发现:
小夹具验证 delta=+1 只取精确时间层，其他 q 不占配额。

证据:
- test_ajae.py:337-343
- exact command: `PYTHONDONTWRITEBYTECODE=1 python -m pytest -q -p no:cacheprovider`

判断:
证据直接覆盖该要求，或排除性全仓库搜索已证明冲突路径不在当前正式路线；可判 PASS。该结论只覆盖本条，不外推到整个模块或科学主张。

需要修改:
无需修改；保留当前实现和直接证据，并继续确保该局部结论不被外推为模块或科学主张已经成立。

### [U08] 状态: NOT RUN

要求:
空跨帧分支：消息和门值为零，输出保持有限。

发现:
没有直接断言空跨帧消息、门值和有限输出。

证据:
- src/model.py:797-842
- test_ajae.py（未发现对应测试）

判断:
本条要求的运行级审计、实验或正式产物没有发生；代码存在也不能替代执行，判定 NOT RUN。

需要修改:
转为 PASS 所需条件：执行并保存该测试要求的直接数值断言；本次审计没有代替开发补测。

### [U09] 状态: NOT RUN

要求:
同帧 3NN：即使其他 q 的节点更近，也只能选择同 q 父节点。

发现:
没有构造其他 q 更近的反例。

证据:
- src/model.py:974-1042
- test_ajae.py（未发现对应测试）

判断:
本条要求的运行级审计、实验或正式产物没有发生；代码存在也不能替代执行，判定 NOT RUN。

需要修改:
转为 PASS 所需条件：执行并保存该测试要求的直接数值断言；本次审计没有代替开发补测。

### [U10] 状态: PASS

要求:
零正类 BCE：只有负类时损失必须有限并等于负类均值。

发现:
现有测试覆盖 zero-positive/only-negative，损失有限并与负类均值一致。

证据:
- test_ajae.py:548-564
- pytest result: 25 passed

判断:
证据直接覆盖该要求，或排除性全仓库搜索已证明冲突路径不在当前正式路线；可判 PASS。该结论只覆盖本条，不外推到整个模块或科学主张。

需要修改:
无需修改；保留当前实现和直接证据，并继续确保该局部结论不被外推为模块或科学主张已经成立。

### [U11] 状态: NOT RUN

要求:
零负类 BCE：只有正类时损失必须有限并等于正类均值。

发现:
没有 only-positive/zero-negative 测试。

证据:
- test_ajae.py:548-564（只覆盖 zero-positive）

判断:
本条要求的运行级审计、实验或正式产物没有发生；代码存在也不能替代执行，判定 NOT RUN。

需要修改:
转为 PASS 所需条件：执行并保存该测试要求的直接数值断言；本次审计没有代替开发补测。

### [U12] 状态: PARTIAL

要求:
概率融合：必须验证 mean(sigmoid(logit))，且不等于 sigmoid(mean(logit))。

发现:
测试验证已给定概率的平均，但没有从固定 logits 同时比较两种公式。

证据:
- test_ajae.py:634-647

判断:
只能证明静态实现、局部运行或缺少 生成链路 的产物，尚不能证明可追溯正式路径完整采用；按审计规则不得升级为 PASS。

需要修改:
转为 PASS 所需条件：执行并保存该测试要求的直接数值断言；本次审计没有代替开发补测。

## 当前状态核验 STATUS-1–STATUS-6

### [STATUS-1] 状态: PARTIAL

要求:
核验 ray/slot identity 是否真实执行、有报告并得到通过结论。

发现:
17 帧 ray/slot 审计和 449 帧 range round-trip 已真实执行，但判定阈值与 `threshold_conclusion` 为空，且多序列覆盖不足。Current status: UNCONFIRMED。

证据:
- src/render.py:1445-2083
- dev.json:8363-8993

判断:
只能证明静态实现、局部运行或缺少 生成链路 的产物，尚不能证明可追溯正式路径完整采用；按审计规则不得升级为 PASS。

需要修改:
先解决对应上游 Gate 并生成要求的真实、冻结、可追溯结果；本次不执行下游实验。

### [STATUS-2] 状态: FAIL

要求:
核验 normal-control 是否已通过匹配来源泄漏实验消除 renderer/source confounding。

发现:
有匹配样本、分组划分和量化结果，但平衡准确率 0.9399、AUROC 0.9786 接近饱和，未消除来源混杂。Current status: FAILED FOR GATE1。

证据:
- dev.json:8978-10958
- protocol.json:204-219

判断:
存在与要求直接冲突的实现、数据或真实运行事实；判定 FAIL。

需要修改:
先解决对应上游 Gate 并生成要求的真实、冻结、可追溯结果；本次不执行下游实验。

### [STATUS-3] 状态: PARTIAL

要求:
核验 beam、range 与 intensity 条件下的传感器一致性。

发现:
beam/range 统计和强度均值存在，但异常代理回波为 0，强度不是完整分布，判据为空。Current status: UNCONFIRMED。

证据:
- dev.json:5-101
- dev.json:6980-6982

判断:
只能证明静态实现、局部运行或缺少 生成链路 的产物，尚不能证明可追溯正式路径完整采用；按审计规则不得升级为 PASS。

需要修改:
先解决对应上游 Gate 并生成要求的真实、冻结、可追溯结果；本次不执行下游实验。

### [STATUS-4] 状态: BLOCKED

要求:
核验 B0、B1、固定 201 比较和正常安全性是否支持 proxy supervision。

发现:
B0、B1 和 fixed-201 comparison 均未运行，不能写 `proxy supervision is validated`。Current status: BLOCKED / NOT RUN。

证据:
- runs/ajae（无 B0/B1）
- src/train.py:1674-1708

判断:
上游 Decision Gate 或冻结条件未满足，本条原则上不得继续；判定 BLOCKED。

需要修改:
先解决对应上游 Gate 并生成要求的真实、冻结、可追溯结果；本次不执行下游实验。

### [STATUS-5] 状态: BLOCKED

要求:
核验三种子 B1/B2/B3 以及 B3>B1、B3>B2 是否支持时间上下文贡献。

发现:
B1/B2/B3/B4 三种子结果不存在，不能写 `temporal context is validated`。Current status: BLOCKED / NOT RUN。

证据:
- runs/ajae（无 B1/B2/B3/B4 seed 结果）

判断:
上游 Decision Gate 或冻结条件未满足，本条原则上不得继续；判定 BLOCKED。

需要修改:
先解决对应上游 Gate 并生成要求的真实、冻结、可追溯结果；本次不执行下游实验。

### [STATUS-6] 状态: BLOCKED

要求:
核验冻结后 19 序列是否支持 anomaly-proxy 向真实 OOD 转移。

发现:
没有方法冻结或 19 序列正式运行，不能写 `proxy supervision transfers to real OOD`。Current status: BLOCKED / NOT RUN。

证据:
- 全仓库无 method-freeze/public19 正式产物
- src/evaluate.py:1485-1592

判断:
上游 Decision Gate 或冻结条件未满足，本条原则上不得继续；判定 BLOCKED。

需要修改:
先解决对应上游 Gate 并生成要求的真实、冻结、可追溯结果；本次不执行下游实验。

## DECISION_GATES

### Gate 1：FAIL

Evidence:
- ray/slot：17 帧详细审计有数值产物，但判定阈值与结论字段为空。
- range-image：train/206 全 449 帧、56,196,767 回波；计数不一致为 0，最大点误差 `4.49388e-14 m`。
- renderer source leakage：平衡准确率 `0.9398887`、AUROC `0.9786149`，接近饱和；匹配距离中位数 `1.790 m`、95% 分位 `3.205 m`，还妨碍根因归属。
- beam/range/intensity：强度只有均值，sensor audit 中 anomaly-proxy returns=0。
- shared renderer：代码路径共享，但真实来源泄漏结果否定了“已经消除来源捷径”。

If FAIL:
Formal AJAE training permitted? NO

### Gate 2：NOT RUN

B0: NOT RUN
B1: NOT RUN
Difference: unavailable

Pure-normal safety: NOT RUN
Normal-control safety: NOT RUN

Conclusion: Gate1 失败且 fixed-201 权威评价器未绑定，Gate2 被阻断。

Proceed to five-frame scientific claim? NO

### Gate 3：NOT RUN

Cross-frame claim: NOT YET TESTED

Window-fusion claim: NOT YET TESTED

没有 B1/B2/B3/B4 的三种子结果，不能从代码结构或 B4 设计推断时间贡献。

### Gate 4：NOT RUN

Real-OOD transfer: NOT TESTED

Confirmation-set integrity: INTACT

边界说明：这里的 INTACT 仅表示当前工作区和可见 Git 历史中未发现本周期运行 19 序列的正面证据；没有访问台账，且 `src/scene.py:1010-1046` 存在未冻结的验证标签读取旁路，因此不能把它解释为全局、独立证明的未触碰状态。

## 严重程度排序

### 1. P0 blocking violations

1. A04-01：当前环境无法实例化官方 STU，不能证实实际 STU 参数全部被冻结。
2. A04-07：没有真实 backward/update 前后的参数、buffer 和状态散列证据。
3. 其余 P0 中仍有 29 项 PARTIAL、NOT RUN 或 BLOCKED；最重要的是完整重叠窗口一致性和五帧真实监督没有执行。

### 2. P1 protocol mismatches

1. Gate1 来源分类近饱和且没有冻结标准；传感器产物又不含 anomaly-proxy 回波。
2. `src/scene.py` 存在公开验证标签读取旁路；方法冻结清单不存在。
3. 30 个开发世界未验证、未提交，V=1/2/3 完全缺失；fixed-201 scope、comparison frame domain、checkpoint rule、maximum_worlds 和 Gate criteria 未冻结。
4. `_AUTHORITATIVE_DEVELOPMENT_EVALUATOR = None`，正式 201 评价路径没有绑定。
5. 缓存身份和正式配置 生成链路 不符合要求：cache key 只有 frame_id，配置缺 commit、effective batch 和 exact CLI。

### 3. P2 missing diagnostics

缺失的直接诊断包括：重叠窗口一致性、双向遮挡、空射线接受/拒绝、STU 冻结 hash/grad、时间体素分离、空跨帧分支、同帧 3NN 反例、zero-negative BCE、完整概率融合、proxy 难度、对象尺度/边界、官方评价、对象级评价和成本报告。

### 4. Already-completed and trustworthy components

1. 本轮真实运行的 25 个局部单元测试全部通过；它们只支持被直接断言的局部行为。
2. train/206 全 449 帧 range-image 机械往返数值一致性证据可信，但只覆盖一个序列且 生成链路 不完整。
3. 第一回波反事实渲染的物理主张边界清楚，没有把近似夸大为完整真实 LiDAR 模拟。
4. schema 30 正式路径未发现 EMA、对象槽、Hungarian、tracking、`L_cf`、global-K 或中心五帧在线主张残留。

### 5. Earliest unresolved decision gate

最早未解决且会使后续结果失去科学解释性的决策门是 Gate 1。当前来源泄漏结果已直接失败，所以即使下游 B0–B5 代码完整或未来误跑出数值，也不能倒推 Gate1 成立。

### 6. Exactly one recommended next scientific action

只做一次完整的 Gate1 重审：在不访问 19/51、也不训练 AJAE 的前提下，先按科学容忍度而非当前分类分数冻结来源泄漏判据，再构造严格条件匹配的 real-normal/rendered-normal 样本，并在同一可追溯产物中同时加入 normal-control/anomaly-proxy 的 beam、range、完整 intensity distribution 与 visibility 对照；只有这一次 Gate1 重审通过后才允许进入正式训练。

## 最终裁决

```text
FINAL VERDICT:
NOT READY FOR FORMAL AJAE TRAINING

Blocking items:
1. Gate 1 fails: renderer/source leakage is nearly saturated and no acceptance criterion is frozen.
2. The sensor audit has no anomaly-proxy returns and does not report a full intensity distribution.
3. The official STU cannot be instantiated in the current environment, fixed-201 evaluation is unbound, and development/freeze rules are unresolved.

The first item that must be resolved:
Gate 1 renderer/source-leakage and sensor-consistency credibility.
```

## 协议落实矩阵

| 模块 | 实现 | 真实运行 | 有证据 | 协议一致 | 结论 |
| --- | --- | --- | --- | --- | --- |
| Ray identity audit | 是 | 部分：17 帧 | 有 | 否：判据为空 | PARTIAL |
| Range-image round trip | 是 | 是：206/449 帧 | 有 | 部分：单序列/生成链路 缺 | PARTIAL |
| Normal-control generator | 是 | 部分 | 有 | 部分 | PARTIAL |
| Anomaly-proxy generator | 是 | 部分：开发定义 | 有 | 部分：未验证 | PARTIAL |
| Shared renderer | 是 | 部分 | 有 | 否：来源审计失败 | FAIL |
| Return model | 是 | 部分 | 有 | 部分 | PARTIAL |
| Intensity model | 是 | 部分 | 有 | 否：分布不完整 | FAIL |
| Renderer leakage audit | 是 | 是 | 有 | 否：近饱和 | FAIL |
| STU frozen interface | 代码有 | 否：构造失败 | 有失败证据 | 否 | FAIL |
| Official semantic evidence | 是 | 仅 toy | 有 | 部分 | PARTIAL |
| Centered five-frame loader | 是 | 是：窗口检查 | 有 | 是（加载层） | PARTIAL |
| Ego alignment | 是 | 仅加载检查 | 有 | 未数值验证 | PARTIAL |
| Four-level pyramid | 是 | 仅 toy | 有 | 代码一致 | PARTIAL |
| Mean-max pooling | 是 | 仅 toy | 有 | 代码一致 | PARTIAL |
| Temporal-stratified KNN | 是 | 仅局部 toy | 有 | 代码一致 | PARTIAL |
| Cross-frame gates | 是 | 仅 toy | 有 | 未正式验证 | PARTIAL |
| Same-frame 3NN decoder | 是 | 仅 toy | 有 | 未定向验证 | PARTIAL |
| Point-level BCE | 是 | 局部测试 | 有 | 部分 | PARTIAL |
| Stable frame/ray identity | 是 | 部分 | 有 | 未做重叠窗口审计 | PARTIAL |
| Probability fusion | 是 | 概率 toy | 有 | 公式端到端未验证 | PARTIAL |
| q-position calibration | 接口有 | 否 | 无结果 | 未冻结 | NOT RUN |
| B0 | 代码有 | 否 | 无正式产物 | 被阻断 | BLOCKED |
| B1 | 代码有 | 否 | 无正式产物 | 被阻断 | BLOCKED |
| B2 | 代码有 | 否 | 无正式产物 | 被阻断 | BLOCKED |
| B3 | 代码有 | 否 | 无正式产物 | 被阻断 | BLOCKED |
| B4 | 代码有 | 否 | 无正式产物 | 被阻断 | BLOCKED |
| B5 | 代码有 | 否 | 无正式产物 | 被阻断 | BLOCKED |
| Moving-normal safety | 代码有 | 否 | 无结果 | 未验证 | BLOCKED |
| Object-scale diagnostics | 类存在 | 仅 toy | 无正式结果 | 未绑定正式路径 | NOT RUN |
| 3-seed development | 代码有 | 否 | 无 | 被阻断 | BLOCKED |
| Official evaluator | 自研/子进程接口有 | 仅 toy parity | 无正式官方输出 | 固定 201 未绑定 | FAIL |
| Object-level evaluator | 接口有 | 否 | 无 | 阈值/DBSCAN 未冻结 | NOT RUN |
| 19-real-sequence freeze protocol | 验证器有 | 否 | 无 freeze manifest | 旁路存在 | BLOCKED |

## 已证实、已否定与仍未知

- 已证实：schema 30 的主要静态结构已经搭建；若干局部数值行为和 206 单序列距离图往返真实运行。
- 已否定：当前工作区不具备正式 AJAE 训练资格；Gate1 没有通过，官方 STU 当前环境不可运行，固定 201 权威评价也未绑定。
- 仍未知：在消除来源泄漏并补齐传感器一致性后，B1 是否优于 B0；五帧跨帧贡献、窗口融合贡献和真实 OOD 转移均未测试。
- 适用边界：本报告只审计 2026-08-25 当前工作区和可见 Git/产物证据，不声称证明工作区外部从未访问 19/51。
