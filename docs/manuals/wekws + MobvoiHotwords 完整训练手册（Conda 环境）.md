# wekws + MobvoiHotwords 完整训练手册（Conda 环境）

> 当前外层 `scripts/` 工作流的命令速查与固定约束见[当前脚本工作流指导手册](script_workflow_guide.md)。本文继续保留 Conda、显存、负样本策略等专题细节。

> **当前推荐流程**：从外层仓库根目录执行 `scripts/` 入口，上游 `wekws/` 保持只读。
> 下文出现的 `run_ctc.sh` stage 说明仅作为**上游 recipe 历史参考**，不再作为推荐命令；迁移后对应的外层命令见下表与 `docs/plans/scripts_migration_and_upstream_readonly_plan.md`。

## 〇、流程总览与前置条件

| 阶段 | 外层入口（推荐） | 上游 `run_ctc.sh` stage（历史参考） |
| :--- | :--- | :--- |
| 环境 | `source scripts/common/env.sh`、`bash scripts/common/check_deps.sh` | — |
| 数据下载 | `scripts/01_data/download_mobvoi.sh`、`download_mobvoi_transcription.sh` | `-3` |
| 格式转换 | `scripts/01_data/prepare_mobvoi.sh`、`regenerate_data_lists.sh`、`prepare_dict.sh` | `-2` ~ `0` |
| 训练 | `scripts/02_train/train_ctc.sh` | `1` ~ `4` |
| 续训 | `scripts/02_train/resume_ctc.sh --checkpoint <file>` | `1` ~ `4` |
| 评估 | `scripts/04_eval/eval_ctc.sh` | `3` |
| 导出 | `scripts/03_export/export_onnx.sh` / `export_jit.sh` / `export_android_onnx.py` | `4` |

**前置条件：** Linux / macOS，NVIDIA GPU（CPU 可训练但较慢），约 **20 GB** 磁盘空间。显存建议 **≥ 16 GiB**；即使显存达到 16 GiB，也要按下文第三节调低 batch size。


## 一、Conda 环境配置（含 torchaudio 版本锁定）

wekws 官方推荐 **Python 3.10**、**PyTorch 2.5.1 + CUDA 12.4**，sox 通过 `conda-forge` 安装。**必须显式指定 `torchaudio==2.5.1`**，否则 `pip install torchaudio` 会解析到最新版（≥2.9），触发 `AttributeError: module 'torchaudio' has no attribute 'info'`。

### 1.1 克隆仓库并创建环境

```bash
git clone https://github.com/wenet-e2e/wekws.git
cd wekws/examples/hi_xiaowen/s0

conda create -n wekws python=3.10 -y
conda activate wekws
```

### 1.2 安装 sox（Conda 频道）

```bash
conda install conda-forge::sox -y
sox --version   # 验证
```

### 1.3 锁定版本安装 PyTorch 与 torchaudio

**这是本手册最关键的一步，直接决定数据准备阶段能否跑通。**

```bash
pip install torch==2.5.1 torchaudio==2.5.1
```

**为什么必须锁定 `torchaudio==2.5.1`：** TorchAudio 自 **2.9** 版本起移除了 `torchaudio.info`，并将 `torchaudio.load` 改为 TorchCodec 的轻量封装。wekws 的 `tools/compute_cmvn_stats.py` 和 `tools/wav2dur.py` 仍在调用 `torchaudio.info(...)`，不锁定版本必然崩溃，并连带导致 `make_list.py` 报 `AssertionError: key in duration_table`。

**验证：**

```bash
python -c "import torch, torchaudio; print(torch.__version__, torchaudio.__version__); print(hasattr(torchaudio, 'info'))"
```

期望输出包含 `2.5.1 2.5.1` 和 `True`。

### 1.4 安装剩余 Python 依赖

```bash
pip install -r requirements.txt
pre-commit install   # 可选，仅代码贡献时需要
```


## 二、数据准备：MobvoiHotwords

MobvoiHotwords 来自 OpenSLR SLR87，包含 **Hi Xiaowen** 和 **Nihao Wenwen** 两个中文唤醒词，采集自 788 位说话人（3–65 岁），附带约 **220 小时**非关键词数据作为负样本。

### 2.1 需要下载的两个文件

| 文件 | 大小 | 内容 |
| :--- | :--- | :--- |
| `mobvoi_hotword_dataset.tgz` | **~17 GB** | 关键词 + 非关键词音频 |
| `mobvoi_hotword_dataset_resources.tgz` | **~6.3 MB** | 标签、说话人、信道信息（JSON） |

下载地址：`openslr.org/resources/87/`。推荐直接执行外层下载脚本，所有资源落在 `work/downloads/`，不修改上游 recipe：

```bash
cd /workspace/wekws-workflow
source scripts/common/env.sh
bash scripts/01_data/download_mobvoi.sh
bash scripts/01_data/download_mobvoi_transcription.sh
```

两个脚本都是幂等的：目标目录已存在时直接跳过，不修改已有资源。国内下载 17 GB 较慢时，可使用 OpenSLR 国内镜像 `openslr.magicdatatech.com/resources/87/` 手动下载后放入 `$MOBVOI_DIR`；不要放回 `wekws/examples/hi_xiaowen/s0/`。

### 2.2 格式转换：从 tgz 到 `data.list`

外层 `prepare_mobvoi.sh` 调用上游数据准备工具，但所有输出都写到 `work/data/hi_xiaowen/`。它生成 train / dev / test 的 `wav.scp`、`text`、`wav.dur`、`data.list` 以及 `train/global_cmvn`；标签映射沿用 2599 维词表，不再手工照抄 `<filler> -1 / Hi_Xiaowen 0 / Nihao_Wenwen 1`。

**执行：**

```bash
source scripts/common/env.sh
bash scripts/01_data/prepare_mobvoi.sh
bash scripts/01_data/regenerate_data_lists.sh
bash scripts/01_data/prepare_dict.sh
```

`regenerate_data_lists.sh` 会强制把所有 wav 路径重写为 `$EXAMPLE_DATA/...`，逐条检查文件存在，并扫描旧上游路径；发现残留即失败。旧 recipe 的 `mobvoi_kws_transcription/` 已迁到 `$MOBVOI_TRANSCRIPTION_DIR`，不要删除该目录，脚本会识别并跳过重复下载。

### 2.3 降低 DataLoader worker 数（可选）

外层 `prepare_mobvoi.sh` 已统一使用 `--num_workers 8`，无需修改上游脚本。若日志仍出现 workers 警告，优先检查 CPU 配额和 `$WORK/data/hi_xiaowen/train/global_cmvn` 是否正常生成。


## 三、训练配置与启动（含显存调优与续训）

### 3.1 外层训练入口显式参数

| 参数 | 默认值 | 说明 |
| :--- | :--- | :--- |
| `--config` | `$WORKSPACE_ROOT/config/hi_xiaowen/ds_tcn_ctc.yaml` | 外层配置，含 `batch_size: 64` 与 `num_keywords: 2599` |
| `--gpus` | `0` | GPU 编号 |
| `--num_keywords` | `2599` | 输出类别数，与 `23.pt` 和现有词表绑定 |
| `--cmvn_file` | `$EXAMPLE_DATA/train/global_cmvn` | 显式绝对路径，不依赖 recipe cwd |
| `--checkpoint` | `$MOBVOI_TRANSCRIPTION_DIR/23.pt` | 初始微调的 base checkpoint；可显式覆盖 |
| `--model_dir` | `$EXAMPLE_EXP` | 模型与 checkpoint 输出目录 |
| `--tensorboard_dir` | `$EXAMPLE_TB` | TensorBoard 输出目录 |

base checkpoint 固定为 `work/downloads/mobvoi_kws_transcription/23.pt`。若要完全随机初始化，需显式传 `--checkpoint init.pt`；正常微调或续训不要使用 `init.pt`。该 checkpoint 的输出维度为 2599，修改词表时必须同步调整 `num_keywords`。

### 3.2 外层入口速查

| 入口 | 作用 |
| :--- | :--- |
| `scripts/01_data/download_mobvoi.sh` | 下载并解压 MobvoiHotwords |
| `scripts/01_data/download_mobvoi_transcription.sh` | 下载 `mobvoi_kws_transcription`（含 `23.pt` / `tokens.txt`） |
| `scripts/01_data/prepare_mobvoi.sh` | 生成 `wav.scp` / `text` / `wav.dur` / CMVN |
| `scripts/01_data/regenerate_data_lists.sh` | 重新生成指向 `work/` 的 `data.list` |
| `scripts/01_data/prepare_dict.sh` | 从 `tokens.txt` 生成 `dict.txt` / `words.txt` |
| `scripts/02_train/train_ctc.sh` | 初始微调与训练循环 |
| `scripts/02_train/resume_ctc.sh` | 独立目录续训 |
| `scripts/04_eval/eval_ctc.sh` | 模型平均、score、DET 与 FRR/FAR 报告 |
| `scripts/03_export/*.sh` | 通用 ONNX / JIT 导出 |

上游 `run_ctc.sh` 的 stage 编号仅作为历史参考，不再作为推荐命令。

### 3.3 显存与批次配置（首次训练前必做）

外层 `config/hi_xiaowen/ds_tcn_ctc.yaml` 已固定 `batch_size: 64`，并显式声明 `num_keywords: 2599`；不要修改上游 `conf/ds_tcn_ctc.yaml`。在 14–16 GiB 显存上，更大的 batch 会在 CTC `log_softmax` 阶段分配数 GiB 临时显存并触发 OOM，典型报错为：

```text
torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 3.62 GiB.
GPU 0 has a total capacity of 14.56 GiB of which 2.37 GiB is free.
```

如需调整，只改外层 yaml 的 `dataset_conf.batch_conf.batch_size`，并通过 `scripts/02_train/train_ctc.sh` 传入；确认稳定后可逐步上调，同时关注有效学习率变化。

**辅助缓解显存碎片：** 如果日志中的 `reserved but unallocated memory` 偏大，可在启动训练前设置环境变量，启用可扩展内存段：

```bash
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
```

### 3.4 启动训练

数据准备完成后，直接从外层入口启动训练：

```bash
cd /workspace/wekws-workflow
source scripts/common/env.sh
bash scripts/02_train/train_ctc.sh --gpus 0
```

如需指定其他 base checkpoint：

```bash
bash scripts/02_train/train_ctc.sh --gpus 0 --checkpoint "$MOBVOI_TRANSCRIPTION_DIR/23.pt"
```

checkpoint 写入 `$EXAMPLE_EXP`，TensorBoard 写入 `$EXAMPLE_TB`，不会回写 `wekws/`。模型平均、评分和导出由 `scripts/04_eval/eval_ctc.sh` 与 `scripts/03_export/` 独立执行；训练 OOM 导致的缺 checkpoint 是连锁故障，先修训练本身，不要改上游脚本。

### 3.5 续训：从已有 checkpoint 继续训练

续训使用独立入口和配置，**不修改上游脚本**：

```bash
cd /workspace/wekws-workflow
source scripts/common/env.sh
bash scripts/02_train/resume_ctc.sh \
  --checkpoint "$EXAMPLE_EXP/avg_30.pt"
```

`scripts/02_train/resume_ctc.sh` 使用 `config/hi_xiaowen/ds_tcn_ctc_resume.yaml`（`lr: 0.0005`，低于基础配置的 `0.001`），输出目录为 `$WORK/exp/hi_xiaowen/ds_tcn_ctc_resume_<timestamp>`，与原始训练分离。传入 `init.pt` 会直接失败，避免 epoch 从 0 重新计数造成混淆。

**续训行为：** wekws 的 `train.py` 只加载 `model.state_dict()`，不恢复优化器、调度器和 epoch 计数，因此日志可能再次出现 `Epoch 0`；这是正常现象。学习率由 resume 配置显式降低，不需要改基础配置或上游 yaml。


## 四、负样本策略：决定实际可用性

Mobvoi 数据集提供了约 220 小时负样本，但**正负比例并非 1:1**。一个常见的失败模式是强制 1:1 训练：有开发者在自定义数据集中采用正负 1:1、负样本两万条的配置，微调后**误检率非常高**。分析指出，真实场景中负样本（日常对话、环境噪声）远多于正样本，训练集应反映这一分布，而不是人为等量平衡。

**使用外层 `scripts/01_data/prepare_mobvoi.sh` 时，Mobvoi 的负样本全量参与训练，不需要手动裁剪到 1:1。** 如需补充困难负样本，应更新 `$EXAMPLE_DATA/{train,dev,test}/wav.scp` 与 `text`，再执行 `scripts/01_data/regenerate_data_lists.sh`。评估时关注 FPPH（每小时误唤醒次数）；语音唤醒工程中正负样本比常见参考区间为 **1:5 到 1:10**，核心是让训练分布贴近部署分布。

**一个值得注意的现象：** 如果训练 loss 长期在 40 附近震荡不降，但 CV loss 很低（如 1 以下）、CV acc 很高（80%+），这通常不是模型没学会，而是**训练集与验证集的负样本比例或难度差异很大**。训练集 filler 占比极高时，大量容易的 filler 样本把平均 loss 拉高，而验证集相对“干净”，指标自然好看。这种情况下优先检查两边的标签分布，而不是盲目调学习率。可以用下面这段脚本快速统计：

```bash
python -c "
import json
from collections import Counter
for split in ['train', 'dev']:
    labels = []
    with open(f'/workspace/wekws-workflow/work/data/hi_xiaowen/{split}/data.list') as f:
        for line in f:
            labels.append(json.loads(line)['txt'])
    c = Counter(labels)
    total = sum(c.values())
    print(f'{split}: total={total}, filler={c.get(-1,0)} ({c.get(-1,0)/total:.1%}), pos0={c.get(0,0)}, pos1={c.get(1,0)}')
"
```


## 五、模型导出（ONNX）

训练完成后，使用外层导出脚本；参数是 `--onnx_model` / `--jit_model`，不是旧手册中的 `--output_dir`：

```bash
cd /workspace/wekws-workflow
source scripts/common/env.sh
bash scripts/03_export/export_onnx.sh
bash scripts/03_export/export_jit.sh
```

产物位于 `$WORK/exports/hi_xiaowen/`：`avg_30.onnx`、`avg_30.zip` 与带绝对 CMVN 路径的 `config_abs_cmvn.yaml`。Android 部署资产单独使用：

```bash
$PYTHON_BIN scripts/03_export/export_android_onnx.py \
  --keyword_thresholds '嗨小问=0.303,你好问问=0.039'
```

Android 固定使用 `$DEPLOY_CHECKPOINT`（`avg_30.pt`），不会自动切换到更新 epoch。


## 六、关键排查速查

| 现象 | 原因 | 修复 |
| :--- | :--- | :--- |
| `AttributeError: module 'torchaudio' has no attribute 'info'` | `torchaudio` ≥ 2.9 移除了 `info` | `pip install torchaudio==2.5.1` |
| `ImportError: TorchCodec is required` | `torchaudio.load` 在 ≥2.9 走 TorchCodec 路径 | 同上，降级 `torchaudio` |
| `destination path 'mobvoi_kws_transcription' already exists` | 旧流程残留目录 | 不要删除；`download_mobvoi_transcription.sh` 会识别并跳过 |
| `make_list.py` 报 `AssertionError: key in duration_table` | `wav.dur` 缺失或 torchaudio 版本错误 | 修复 torchaudio 后重跑 `scripts/01_data/prepare_mobvoi.sh` 与 `regenerate_data_lists.sh` |
| **`torch.OutOfMemoryError: CUDA out of memory`（首轮即崩）** | batch 过大 | 修改外层 `config/hi_xiaowen/ds_tcn_ctc.yaml` 的 `batch_size`，不要改上游 yaml |
| `average_model.py` 报 `IndexError: too many indices for array` | checkpoint/yaml 不足 30 个 | `eval_ctc.sh` 已回退到 `avg_N.pt` / `final.pt` |
| `score_ctc.py` 报 `FileNotFoundError: data/train/global_cmvn` | 直接用了训练产物里的相对路径 config | 使用 `eval_ctc.sh` 生成的 `$RESULT_DIR/config.yaml`（绝对 CMVN 路径） |
| `score_ctc.py` / `export_*.py` 报 `FileNotFoundError: avg_30.pt` | checkpoint 路径错误或训练失败 | 部署固定 `$DEPLOY_CHECKPOINT`；训练失败时先修训练 |
| **续训后日志又从 `Epoch 0` 开始** | **wekws 只加载权重，不恢复 epoch 计数和优化器状态** | **正常行为；使用 `resume_ctc.sh` 的独立输出目录，学习率由 resume config 调低** |
| **续训后 loss 突然升高** | **优化器状态未恢复** | **检查 `ds_tcn_ctc_resume.yaml` 的 `lr: 0.0005`** |
| **训练 loss 高但 CV loss 很低、acc 很高** | **训练集与验证集负样本比例/难度差异大** | **用第四节脚本检查 `data.list` 标签分布，不要盲目调学习率** |
| DataLoader worker 警告 | workers 超过系统建议 | 外层 `prepare_mobvoi.sh` 已使用 8 workers |
| 误唤醒率偏高 | 负样本分布与真实场景不匹配 | 保持负样本全量参与，必要时补充困难负样本，关注 FPPH |


**一句话总结：** 本手册与“照着官方 README 直接跑”相比有**三个关键修正**——环境阶段**锁定 `torchaudio==2.5.1`**，确保数据准备阶段不崩；训练阶段**把 `batch_size` 从 256 降到 32/64 并开启 `expandable_segments`**，确保首轮不 OOM；续训阶段**明确只加载权重、epoch 会重记、学习率需手动调低**，避免误以为训练现场被完整恢复。做到这三步，其余从数据下载、训练到 ONNX 导出的流程与官方 `hi_xiaowen` 示例一致；第四节的负样本策略则决定模型在实际场景中是否“好用”。