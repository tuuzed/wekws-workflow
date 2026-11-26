# wekws + MobvoiHotwords 完整训练手册（Conda 环境）


## 〇、流程总览与前置条件

| 阶段 | 关键动作 | 对应 `run_ctc.sh` stage |
| :--- | :--- | :--- |
| 环境 | Conda 创建 `wekws` 环境，安装 sox、锁定版本的 PyTorch/torchaudio | — |
| 数据下载 | 下载 17 GB 音频 + 6.3 MB 资源 | `-3` |
| 格式转换 | 生成 `wav.scp` / `text` / `data.list` | `-2` ~ `0` |
| 训练 | 调整 batch size 后启动 CTC 训练 | `1` ~ `4` |
| 续训 | 从已有 checkpoint 热启动继续训练 | `1` ~ `4` |
| 导出 | 转 ONNX 用于部署 | 训练后手动执行 |

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

下载地址：`openslr.org/resources/87/`。`run_ctc.sh` 的 **stage -3** 已内置下载逻辑，只需把脚本中的 `download_dir` 指向数据存放位置：

```bash
# 编辑 run_ctc.sh，找到：
download_dir=/your/path/to/MobvoiHotwords
```

然后执行：

```bash
bash run_ctc.sh --stage -3 --stop_stage -3
```

国内下载 17 GB 较慢时，可使用 OpenSLR 国内镜像 `openslr.magicdatatech.com/resources/87/` 手动下载，放入 `download_dir` 后跳过 `-3` 阶段。

### 2.2 格式转换：从 tgz 到 `data.list`

`run_ctc.sh` 的 **stage -2** 会写入标签映射，并调用 `local/prepare_data.py` 分别处理 train / dev / test 的正样本（`p_`）和负样本（`n_`）：

```text
<filler> -1
Hi_Xiaowen 0
Nihao_Wenwen 1
```

**执行：**

```bash
bash run_ctc.sh --stage -2 --stop_stage -2
```

如果日志中出现 `fatal: destination path 'mobvoi_kws_transcription' already exists`，这是之前中断运行残留的目录，直接删除即可：

```bash
rm -rf mobvoi_kws_transcription
```

**stage -1 是可选的**：它用 Paraformer 对负样本做转写，服务于基于文本的建模需求。标准 CTC 唤醒词训练可以直接从 **-2 跳到 0**。

**stage 0** 计算 CMVN 并生成最终的 `data.list`（JSON 行格式，含 `key`、`txt` 标签、`duration` 和 `wav` 路径）。推荐从 `-2` 一次执行到 `0`：

```bash
bash run_ctc.sh --stage -2 --stop_stage 0
```

### 2.3 降低 DataLoader worker 数（可选）

`run_ctc.sh` stage 0 默认调用 `compute_cmvn_stats.py --num_workers 16`，日志中会出现 “suggested max number... is 8” 的警告。如果后续处理变慢或卡住，可以将 `run_ctc.sh` 中该行改为 `--num_workers 8`。这不是致命错误，但可避免不必要的资源争抢。


## 三、训练配置与启动（含显存调优与续训）

### 3.1 `run_ctc.sh` 中值得确认的变量

| 变量 | 默认值 | 说明 |
| :--- | :--- | :--- |
| `config` | `conf/ds_tcn_ctc.yaml` | 模型结构与超参 |
| `gpus` | `"0"` | GPU 编号，多卡写 `"0,1"` |
| `num_keywords` | `2599` | 输出类别数，与预训练 checkpoint 一致 |
| `norm_mean` / `norm_var` | `true` | CMVN 归一化开关 |
| `checkpoint` | 空 | **非空即加载权重**；脚本在 stage 1 会检测 `mobvoi_kws_transcription/23.pt` 是否存在，存在则自动填入该路径用于微调；续训时手动指向自己的模型文件 |
| `average_model` | `true` | 对最后 `num_average`（默认 30）个 checkpoint 做平均 |

**关于 `checkpoint` 的说明：** `run_ctc.sh` 的 stage 1 开头有一段**条件判断**——若 `mobvoi_kws_transcription/23.pt` 存在，就把它赋给 `checkpoint` 并打印 `Use the base model trained with WenetSpeech as checkpoint`。这属于**微调**而非从头训练，通常收敛更快。如果你想完全从随机初始化开始，**不能只把 `checkpoint=` 清空**（它初始本来就是空的，会被后面的条件判断重新赋值），而应把 `23.pt` 移走（`mv mobvoi_kws_transcription/23.pt mobvoi_kws_transcription/23.pt.bak`）或注释掉那段 `if [ -f ... ]` 逻辑。注意该 checkpoint 的输出维度与 `num_keywords=2599` 绑定，若你修改了 `dict/words.txt` 的类别数，需要相应调整配置。

### 3.2 Stage 速查

| Stage | 作用 |
| :--- | :--- |
| `-3` | 下载并解压数据集 |
| `-2` | 生成 `wav.scp` / `text` / `dict` |
| `-1` | （可选）负样本转写 |
| `0` | 计算 CMVN，生成 `data.list` |
| `1` ~ `4` | 训练主流程（初始化、训练循环、模型平均、评分与导出） |

### 3.3 显存与批次配置（首次训练前必做）

`conf/ds_tcn_ctc.yaml` 中默认 `batch_size: 256`。在 14–16 GiB 显存的 GPU 上，这个规模在 CTC 的 `log_softmax` 步骤会需要一次性分配数 GiB 临时显存，**首轮即 OOM**，典型报错为：

```text
torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 3.62 GiB.
GPU 0 has a total capacity of 14.56 GiB of which 2.37 GiB is free.
```

**修改方法：** 编辑 `conf/ds_tcn_ctc.yaml`，找到 `batch_conf`，先把 `batch_size` 降到 32 或 64：

```yaml
dataset_conf:
  ...
  batch_conf:
    batch_size: 32   # 默认 256，14–16 GiB 显存建议先设为 32
```

确认训练稳定后，可以逐步上调（64 → 96 → 128）观察显存占用，找到你这块卡的上限。batch size 变化会影响有效学习率，若大幅调低，可酌情同步下调学习率或增加梯度累积步数。

**辅助缓解显存碎片：** 如果日志中的 `reserved but unallocated memory` 偏大，可在启动训练前设置环境变量，启用可扩展内存段：

```bash
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
```

### 3.4 启动训练

数据准备完成（至少到 stage 0）且已调低 `batch_size` 后：

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
bash run_ctc.sh --stage 1 --stop_stage 4 --gpus "0"
```

或从 stage 0 一气呵成：

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
bash run_ctc.sh --stage 0 --stop_stage 4 --gpus "0"
```

日志和模型默认保存在 `exp/ds_tcn_ctc/` 下。启用 `average_model=true` 时，最终用于评分或导出的通常是 `avg_30.pt`；否则为 `final.pt`。

**注意 stage 1~4 是一条链：** 只有训练正常完成并产出 checkpoint，后续的模型平均（`average_model.py`）、评分（`score_ctc.py`）和导出（`export_jit.py` / `export_onnx.py`）才能依次进行。训练 OOM 会让后续步骤接连失败：模型平均因缺少验证分数报 `IndexError`，评分和导出因找不到 `avg_30.pt` 报 `FileNotFoundError`。这些是**连锁故障**，修好训练本身即可自动消失，不需要单独改脚本。

### 3.5 续训：从已有 checkpoint 继续训练

当训练中断，或想基于已有权重做热启动时，用 `checkpoint` 变量指向目标模型文件即可。**续训的核心是“加载权重”，不是“恢复训练现场”。**

#### 操作方式

**推荐做法：在脚本 stage 1 中强制覆盖 `checkpoint`。** 找到 stage 1 开头自动设置 `checkpoint` 的那段 `if [ -f "mobvoi_kws_transcription/23.pt" ]`，在其**之后**加一行：

```bash
checkpoint="exp/ds_tcn_ctc/0.pt"   # 换成你实际想续训的 epoch 文件
```

然后正常启动：

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
bash run_ctc.sh --stage 1 --stop_stage 4 --gpus "0"
```

**为什么推荐改脚本而非命令行参数：** `run_ctc.sh` 使用 `parse_options.sh`，理论上支持 `--checkpoint ...` 覆盖，但由于脚本内有自动赋值逻辑，命令行传入的值可能被后续代码重新覆盖。直接改脚本最可控。

**不要从 `init.pt` 续训：** `init.pt` 是训练开始时的初始权重，从它续训等同于重新开始。应选择带 epoch 编号的模型文件（如 `0.pt`、`1.pt`）。

#### 续训时会发生什么

| 项目 | 是否恢复 | 说明 |
| :--- | :--- | :--- |
| 模型权重 | ✅ | `model.state_dict()` 被加载 |
| 优化器状态（Adam 动量等） | ❌ | 重新初始化 |
| 学习率调度器状态 | ❌ | 从配置文件的初始 `lr` 重新开始 |
| **epoch 计数器** | ❌ | **日志中的 `Epoch 0` 会重新出现** |

**epoch 会重记：** wekws 的 `train.py` 只从 checkpoint 加载权重，不恢复训练元信息，因此训练循环从头开始，日志会再次打印 `Epoch 0 TRAIN info lr 0.001`。这不是错误——模型确实在之前权重基础上继续优化，只是编号重置了。

**学习率需要手动调低：** 由于优化器状态不恢复，学习率会回到配置文件中的初始值（默认 0.001）。续训时建议在 `conf/ds_tcn_ctc.yaml` 中把 `lr` 降到 **0.0005 或 0.0001**，避免热启动后因学习率过大破坏已有权重。

#### 避免日志混淆的建议

epoch 编号重置会让新旧日志混在一起，难以对比。推荐**为续训创建独立实验目录**：

```bash
# 复制 checkpoint 到新目录
mkdir -p exp/ds_tcn_ctc_resume
cp exp/ds_tcn_ctc/0.pt exp/ds_tcn_ctc_resume/init_resume.pt

# 修改 run_ctc.sh 中的 model_dir 和 checkpoint
model_dir=exp/ds_tcn_ctc_resume
checkpoint=exp/ds_tcn_ctc_resume/init_resume.pt
```

这样续训的日志、模型、TensorBoard 记录都是独立的，便于和原始训练对比。如果不新建目录，续训会写入同一个 `train.log`，旧日志可能被覆盖或追加。


## 四、负样本策略：决定实际可用性

Mobvoi 数据集提供了约 220 小时负样本，但**正负比例并非 1:1**。一个常见的失败模式是强制 1:1 训练：有开发者在自定义数据集中采用正负 1:1、负样本两万条的配置，微调后**误检率非常高**。分析指出，真实场景中负样本（日常对话、环境噪声）远多于正样本，训练集应反映这一分布，而不是人为等量平衡。

**使用 `run_ctc.sh` 默认流程时，Mobvoi 的负样本是全量参与训练的，不需要手动裁剪到 1:1。** 如需进一步提升实际场景表现，可以补充**困难负样本**（近音词、相似韵律短语），扩充 `wav.scp` 和 `text` 后重新生成 `data.list`。评估时应关注验证集上的 **FPPH（每小时误唤醒次数）**；相比 loss 或准确率，它更能反映真实可用性。语音唤醒工程中，正负样本比的常见参考区间是 **1:5 到 1:10**，核心原则是让训练分布贴近部署分布。

**一个值得注意的现象：** 如果训练 loss 长期在 40 附近震荡不降，但 CV loss 很低（如 1 以下）、CV acc 很高（80%+），这通常不是模型没学会，而是**训练集与验证集的负样本比例或难度差异很大**。训练集 filler 占比极高时，大量容易的 filler 样本把平均 loss 拉高，而验证集相对“干净”，指标自然好看。这种情况下优先检查两边的标签分布，而不是盲目调学习率。可以用下面这段脚本快速统计：

```bash
python -c "
import json
from collections import Counter
for split in ['train', 'dev']:
    labels = []
    with open(f'data/{split}/data.list') as f:
        for line in f:
            labels.append(json.loads(line)['txt'])
    c = Counter(labels)
    total = sum(c.values())
    print(f'{split}: total={total}, filler={c.get(-1,0)} ({c.get(-1,0)/total:.1%}), pos0={c.get(0,0)}, pos1={c.get(1,0)}')
"
```


## 五、模型导出（ONNX）

训练完成后，使用 `wekws/bin/export_onnx.py` 导出：

```bash
exp=exp/ds_tcn_ctc   # 换成你的实验目录

python -m wekws.bin.export_onnx \
  --config $exp/train.yaml \
  --checkpoint $exp/avg_30.pt \
  --output_dir final.onnx
```

如果未启用模型平均，将 `--checkpoint` 换成 `$exp/final.pt`。导出的 ONNX 可用于 wekws 自带的 `runtime/onnxruntime`，也支持进一步转换为 Android 或 Raspberry Pi 的部署格式。


## 六、关键排查速查

| 现象 | 原因 | 修复 |
| :--- | :--- | :--- |
| `AttributeError: module 'torchaudio' has no attribute 'info'` | `torchaudio` ≥ 2.9 移除了 `info` | `pip install torchaudio==2.5.1` |
| `ImportError: TorchCodec is required` | `torchaudio.load` 在 ≥2.9 走 TorchCodec 路径 | 同上，降级 `torchaudio` |
| `destination path 'mobvoi_kws_transcription' already exists` | 之前中断残留目录 | `rm -rf mobvoi_kws_transcription` |
| `make_list.py` 报 `AssertionError: key in duration_table` | `wav2dur.py` 因 torchaudio 错误未产出时长 | 根因同上，修复 torchaudio 版本后重跑 stage 0 |
| **`torch.OutOfMemoryError: CUDA out of memory`（首轮即崩）** | **`batch_size: 256` 在 ≤16 GiB 显存上过大** | **`conf/ds_tcn_ctc.yaml` 中 `batch_size` 降到 32/64；加 `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`** |
| `average_model.py` 报 `IndexError: too many indices for array` | 训练一个 epoch 都没完成，无验证分数 | 修好 OOM 让训练正常产出 checkpoint 后自动消失 |
| `score_ctc.py` / `export_*.py` 报 `FileNotFoundError: avg_30.pt` | 训练失败未产出平均模型 | 同上，属 OOM 的连锁故障 |
| **续训后日志又从 `Epoch 0` 开始** | **wekws 只加载权重，不恢复 epoch 计数和优化器状态** | **正常行为；可新建实验目录避免日志混淆，学习率手动调低** |
| **续训后 loss 突然升高** | **优化器状态未恢复，学习率回到初始值** | **在 `conf/ds_tcn_ctc.yaml` 中把 `lr` 降到 0.0005/0.0001** |
| **训练 loss 高但 CV loss 很低、acc 很高** | **训练集与验证集负样本比例/难度差异大** | **用第四节脚本检查两边 `data.list` 标签分布，不要盲目调学习率** |
| DataLoader worker 警告 | 16 workers > 系统建议的 8 | 将 `compute_cmvn_stats.py` 的 `--num_workers` 改为 8 |
| 误唤醒率偏高 | 负样本分布与真实场景不匹配 | 保持负样本全量参与，必要时补充困难负样本，关注 FPPH |


**一句话总结：** 本手册与“照着官方 README 直接跑”相比有**三个关键修正**——环境阶段**锁定 `torchaudio==2.5.1`**，确保数据准备阶段不崩；训练阶段**把 `batch_size` 从 256 降到 32/64 并开启 `expandable_segments`**，确保首轮不 OOM；续训阶段**明确只加载权重、epoch 会重记、学习率需手动调低**，避免误以为训练现场被完整恢复。做到这三步，其余从数据下载、训练到 ONNX 导出的流程与官方 `hi_xiaowen` 示例一致；第四节的负样本策略则决定模型在实际场景中是否“好用”。