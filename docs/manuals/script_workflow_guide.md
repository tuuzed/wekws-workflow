# 当前脚本工作流指导手册

> 适用范围：外层仓库 `/workspace/wekws-workflow` 的当前 `scripts/` 工作流。
> 上游 `wekws/` 是 pin 在固定 commit 的只读 submodule；所有推荐命令都从仓库根目录执行，且不修改上游 recipe。
> 关联专题：[完整训练手册（Conda 环境）](<wekws + MobvoiHotwords 完整训练手册（Conda 环境）.md>)、[FRR/FAR 评测操作手册](../evaluation/avg30_frr_far_evaluation.md)、[FRR/FAR 评测结果](../evaluation/avg30_frr_far_report.md)。

## 0. 事实优先级与固定约束

本手册按以下优先级取事实：

1. **`scripts/` 源码与 `scripts/common/env.sh`**：命令、参数、默认值和路径的最终依据。
2. **部署资产与评测报告**：`android/app/src/main/assets/`、`work/eval/avg_30_pt/` 和 `docs/evaluation/avg30_frr_far_report.md`。
3. **历史手册与上游 stage**：仅作追溯；上游 `run_ctc.sh` stage、`15.pt` 联调资产、历史阈值 `0.297 / 0.000`、历史 FAR/FRR `0.175212 / 3.31%` 都不是当前推荐配置。

固定工作点：

| 项目 | 当前值 |
| :--- | :--- |
| 部署 checkpoint | `work/exp/hi_xiaowen/ds_tcn_ctc/avg_30.pt`（`$DEPLOY_CHECKPOINT`） |
| 初始微调 base checkpoint | `work/downloads/mobvoi_kws_transcription/23.pt`（`$BASE_CHECKPOINT`） |
| 部署阈值 | `嗨小问=0.303`、`你好问问=0.039` |
| 阈值归属 | 每个关键词独立写入 `keywords[].trigger_threshold`；runtime config 不含全局 `ctc.trigger_threshold` |
| Android 模型 | `android/app/src/main/assets/kws.onnx`，直接加载，不转 `.ort`（历史 ORT Mobile 方案已废弃） |
| 输出语义 | posterior，Native 侧不再做第二次 softmax |
| score | `ctc_prefix_beam_search` 的 `sqrt(∏ token_prob)` |
| 调试字段 | `token_probs` / `frame_topk` 仅用于调试，不改变 score、阈值、检测结果或 score 文件格式 |

命令以当前脚本实际参数为准；旧式输出目录参数、旧 recipe 相对路径和上游 stage 编号都不作为推荐命令。

---

## 1. 当前工作流总览

| 阶段 | 外层入口 | 主要产物 |
| :--- | :--- | :--- |
| 环境 | `scripts/common/env.sh`、`scripts/common/check_deps.sh` | 统一路径、Python、pycache、版本检查 |
| 下载 | `scripts/01_data/download_mobvoi.sh`、`download_mobvoi_transcription.sh` | `work/downloads/` |
| 数据 | `scripts/01_data/prepare_dict.sh`、`prepare_mobvoi.sh`、`regenerate_data_lists.sh` | `work/dict/hi_xiaowen/`、`work/data/hi_xiaowen/` |
| 训练 | `scripts/02_train/train_ctc.sh` | `work/exp/hi_xiaowen/ds_tcn_ctc/` |
| 续训 | `scripts/02_train/resume_ctc.sh --checkpoint <file>` | `work/exp/hi_xiaowen/ds_tcn_ctc_resume/` |
| 评估 | `scripts/04_eval/eval_ctc.sh` | `work/eval/avg_30_pt/`、`docs/evaluation/avg30_frr_far_report.md` |
| 通用导出 | `scripts/03_export/export_onnx.sh`、`export_jit.sh` | `work/exports/hi_xiaowen/` |
| Android 导出 | `scripts/03_export/export_android_onnx.py` | `android/app/src/main/assets/` |
| 推理 | `scripts/04_eval/infer_cli.py`、`infer_webui.py` | score/detail 输出、WebUI 调试 |
| 部署验证 | `scripts/05_deploy/build_android_host.sh`、`compare_scores.sh`、`build_apk.sh` | Native host、三方 score 对齐、APK |

`scripts/common/env.sh` 关键变量：

| 变量 | 含义 / 默认值 |
| :--- | :--- |
| `WORKSPACE_ROOT` | `/workspace/wekws-workflow` |
| `WEKWS_ROOT` | `$WORKSPACE_ROOT/wekws` |
| `WORK` | `$WORKSPACE_ROOT/work` |
| `CONFIG_DIR` | `$WORKSPACE_ROOT/config/hi_xiaowen` |
| `MOBVOI_DIR` | `$WORK/downloads/mobvoi` |
| `MOBVOI_TRANSCRIPTION_DIR` | `$WORK/downloads/mobvoi_kws_transcription` |
| `EXAMPLE_DATA` | `$WORK/data/hi_xiaowen` |
| `EXAMPLE_DICT` | `$WORK/dict/hi_xiaowen` |
| `EXAMPLE_EXP` | `$WORK/exp/hi_xiaowen/ds_tcn_ctc` |
| `EXAMPLE_TB` | `$WORK/tensorboard/hi_xiaowen/ds_tcn_ctc` |
| `EXPORT_DIR` | `$WORK/exports/hi_xiaowen` |
| `RESULT_DIR` | `$WORK/eval/avg_30_pt` |
| `BASE_CHECKPOINT` | `$MOBVOI_TRANSCRIPTION_DIR/23.pt` |
| `DEPLOY_CHECKPOINT` | `$EXAMPLE_EXP/avg_30.pt` |
| `KEYWORD_THRESHOLDS` | `嗨小问=0.303,你好问问=0.039` |
| `PYTHON_BIN` | `/root/miniconda/envs/wekws/bin/python` |
| `TORCHRUN_BIN` | `/root/miniconda/envs/wekws/bin/torchrun` |
| `SCORE_BATCH_SIZE` | `32` |

---

## 2. 环境与数据

### 2.1 环境检查

```bash
cd /workspace/wekws-workflow
source scripts/common/env.sh
bash scripts/common/check_deps.sh
```

`check_deps.sh` 检查 `torch` / `torchaudio` 为 2.5.1、`torchaudio.info` 可用，并要求 `sox` 来自 conda-forge。`env.sh` 同时设置 `PYTHONPATH`、`PYTHONIOENCODING=UTF-8`、`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`，并把 bytecode 缓存指向 `work/pycache`，避免在只读上游目录生成缓存。

详细 Conda/显存/负样本背景见[完整训练手册](<wekws + MobvoiHotwords 完整训练手册（Conda 环境）.md>)。

### 2.2 下载

```bash
source scripts/common/env.sh
bash scripts/01_data/download_mobvoi.sh
bash scripts/01_data/download_mobvoi_transcription.sh
```

- `download_mobvoi.sh` 下载并解压 MobvoiHotwords 到 `$MOBVOI_DIR`。
- `download_mobvoi_transcription.sh` 克隆完整 `mobvoi_kws_transcription` 到 `$MOBVOI_TRANSCRIPTION_DIR`，保留其 `.git`，并校验 `23.pt` / `tokens.txt`。
- 两个脚本都是幂等的：目标存在时跳过，不把下载物放回 `wekws/examples/hi_xiaowen/s0/`。

### 2.3 数据准备

推荐顺序是先生成字典，再准备数据，最后重写数据清单：

```bash
source scripts/common/env.sh
bash scripts/01_data/prepare_dict.sh
bash scripts/01_data/prepare_mobvoi.sh
bash scripts/01_data/regenerate_data_lists.sh
```

- `prepare_dict.sh` 从 `tokens.txt` 生成 `work/dict/hi_xiaowen/dict.txt` 与 `words.txt`，并校验关键字 token id。
- `prepare_mobvoi.sh` 生成 train/dev/test 的 `wav.scp`、`text`、`wav.dur`、`data.list` 和 `train/global_cmvn`；输入/输出均为绝对路径，产物在 `work/`。
- `regenerate_data_lists.sh` 强制把 wav 路径重写到 `$EXAMPLE_DATA`，逐条检查文件存在，并拒绝残留旧上游路径。

---

## 3. 训练与续训

### 3.1 初始训练

```bash
cd /workspace/wekws-workflow
source scripts/common/env.sh
bash scripts/02_train/train_ctc.sh --gpus 0
```

默认行为：

- config：`config/hi_xiaowen/ds_tcn_ctc.yaml`，外层副本包含 `batch_size: 64` 和 `num_keywords: 2599`。
- checkpoint：`$BASE_CHECKPOINT`，即 `work/downloads/mobvoi_kws_transcription/23.pt`。
- dict：`$EXAMPLE_DICT`，显式传入，不依赖 cwd。
- 数据：`$EXAMPLE_DATA/train/data.list`、`$EXAMPLE_DATA/dev/data.list`。
- 产物：`model_dir=$EXAMPLE_EXP`、`tensorboard_dir=$EXAMPLE_TB`。
- worker 数默认 8；可选 `--num_workers`。

显式覆盖 base checkpoint：

```bash
bash scripts/02_train/train_ctc.sh --gpus 0 --checkpoint "$BASE_CHECKPOINT"
```

不要把 `init.pt` 用于正常微调或续训；它会破坏当前 2599 维 token 映射与 checkpoint 链。

### 3.2 续训

```bash
source scripts/common/env.sh
bash scripts/02_train/resume_ctc.sh \
  --checkpoint "$EXAMPLE_EXP/avg_30.pt" \
  --gpus 0
```

- `--checkpoint` 必填，且拒绝 `init.pt`。
- 使用 `config/hi_xiaowen/ds_tcn_ctc_resume.yaml`（`lr: 0.0005`），不修改基础配置。
- 默认写入 `work/exp/hi_xiaowen/ds_tcn_ctc_resume/`，与原始训练目录分离。
- 上游 `train.py` 只加载权重，不恢复优化器、调度器和 epoch 计数；日志再次出现 `Epoch 0` 是正常行为。
- 学习率只能通过 resume yaml 调整，不通过不存在的训练参数传入。

---

## 4. FRR/FAR 评估

完整参数和原理解释见[FRR/FAR 评测操作手册](../evaluation/avg30_frr_far_evaluation.md)；当前结果见[FRR/FAR 评测结果](../evaluation/avg30_frr_far_report.md)。

### 4.1 一键评估

```bash
cd /workspace/wekws-workflow
source scripts/common/env.sh
bash scripts/04_eval/eval_ctc.sh --checkpoint "$DEPLOY_CHECKPOINT" --result_dir "$RESULT_DIR"
```

脚本完成：

1. checkpoint 缺失时按可用 checkpoint/yaml 数生成 `avg_N.pt` 或回退 `final.pt`；不会静默切换到更新 epoch。
2. 派生 `$RESULT_DIR/config.yaml`，把 CMVN 写成 `work/data/hi_xiaowen/train/global_cmvn` 绝对路径。
3. 调用 `score_ctc.py` 生成 `$RESULT_DIR/score.txt`，关键词使用 ASCII escape：`\u55e8\u5c0f\u95ee,\u4f60\u597d\u95ee\u95ee`。
4. 调用 `compute_det_ctc.py`：阈值步长 `--step 0.001`、`--window_shift 50`，生成 stats 与 `det.png`。
5. 写出基础报告 `docs/evaluation/avg30_frr_far_report.md`；正式报告还应补齐 selected epochs、test wav 数、负样本总时长与 `det.png` 路径。

产物格式：

```text
$RESULT_DIR/score.txt
$RESULT_DIR/stats/stats.嗨_小_问.txt
$RESULT_DIR/stats/stats.你_好_问_问.txt
$RESULT_DIR/stats/det.png
```

`score.txt` 每行只保留 3 位小数：

```text
<key> detected <关键词> <score>
<key> rejected
```

stats 每行三列为 `threshold`、`false_alarm_per_hour`、`FRR`。因此阈值步长使用 `0.001`，更细没有意义。

### 4.2 逐关键词选点

```bash
source scripts/common/env.sh
awk '$2 <= 0.2 {print; exit}' "$RESULT_DIR/stats/stats.嗨_小_问.txt"
awk '$2 <= 0.2 {print; exit}' "$RESULT_DIR/stats/stats.你_好_问_问.txt"
```

三类阈值必须分清：

| 类型 | 嗨小问 | 你好问问 | 用途 |
| :--- | ---: | ---: | :--- |
| 当前复算理论最低点 | `0.293` | `0.038` | 当前 `score.txt` 按 `awk '$2 <= 0.2 {print; exit}'` 复算得到 |
| 历史理论点（仅追溯） | `0.297` | `0.000` | 早期评测记录，不作为当前选点依据 |
| 部署工作点 | `0.303` | `0.039` | 留 FAR 余量，Android runtime 实际使用 |

部署点当前 stats 对应值：

| 关键词 | 阈值 | FAR（次/小时） | FRR |
| :--- | ---: | ---: | ---: |
| 嗨小问 | `0.303` | `0.174880` | `4.0598%` |
| 你好问问 | `0.039` | `0.189813` | `3.2986%` |

报告模板：

| 模型 | 关键词 | FAR 目标（次/小时） | 阈值 | FRR |
| :--- | :--- | ---: | ---: | ---: |
| avg_30.pt | 嗨小问 | 0.2 | 0.303 | 4.0598% |
| avg_30.pt | 你好问问 | 0.2 | 0.039 | 3.2986% |

---

## 5. 模型导出

### 5.1 通用 ONNX / JIT

```bash
cd /workspace/wekws-workflow
source scripts/common/env.sh
bash scripts/03_export/export_onnx.sh
bash scripts/03_export/export_jit.sh
```

- ONNX 入口使用 `--onnx_model`，默认输出 `$EXPORT_DIR/avg_30.onnx`。
- JIT 入口使用 `--jit_model`，默认输出 `$EXPORT_DIR/avg_30.zip`。
- 两者都先派生带绝对 CMVN 路径的 `$EXPORT_DIR/config_abs_cmvn.yaml`。
- 默认 checkpoint 为 `$DEPLOY_CHECKPOINT`；更新 epoch 不会自动替换通用导出产物。

### 5.2 Android 部署资产

```bash
cd /workspace/wekws-workflow
source scripts/common/env.sh
$PYTHON_BIN scripts/03_export/export_android_onnx.py \
  --keyword_thresholds '嗨小问=0.303,你好问问=0.039'
```

固定输出：

```text
android/app/src/main/assets/kws.onnx
android/app/src/main/assets/config.json
android/app/src/main/assets/keywords.txt
android/app/src/main/assets/id_to_token.json
```

要求与校验：

- checkpoint 默认固定 `$DEPLOY_CHECKPOINT`，显式 `--checkpoint` 才会覆盖。
- `--keyword_thresholds` 必填；缺少任一关键词或传入未知关键词都会失败。
- 关键词 token 映射：`嗨小问` 为 `[1302,1462,2494]`，`你好问问` 为 `[1027,1401,2494,2494]`。
- 旧路径 `android/export/export_ctc_onnx.py` 仅保留兼容 wrapper，不作为推荐入口。

### 5.3 Android Runtime 契约

| 项目 | 契约 |
| :--- | :--- |
| 模型加载 | Native 直接加载 `kws.onnx`，不转 `.ort`，不做 ORT Mobile 格式转换 |
| 输入 | `[N,T,40]` 特征与 `[N,256,105]` cache，batch/帧数动态 |
| 输出 | posterior `[N,T,2599]` 与 cache `[N,256,105]` |
| 后处理 | posterior 直接参与 CTC prefix beam search，不再 softmax |
| 搜索参数 | beam 3 / path 20 / token `prob > 0.05` / `abs_tol=1e-6` |
| score | `sqrt(∏ token_prob)`，与 `score_ctc.py` 一致 |
| 阈值 | per-keyword `trigger_threshold`；同一 channel 可同时上报多个关键词，Engine 不仲裁 |
| 事件语义 | `currentEvents()` 保存当前 beam 中最近一次命中的事件，不是历史最高分 |
| 调试字段 | `token_probs` / `frame_topk` 只用于观察概率与逐帧 top-k，不参与 score、阈值、检测结果或 `score_file` 格式 |

---

## 6. 推理

CLI 与 WebUI 都复用 `scripts/04_eval/infer_core.py` 中的 `KwsEngine`，根目录 `infer_cli.py` / `infer_webui.py` / `infer_core.py` 只是兼容 wrapper。

### 6.1 CLI

```bash
source scripts/common/env.sh
$PYTHON_BIN scripts/04_eval/infer_cli.py \
  --wav android/tests/data/ch0.wav \
  --checkpoint "$DEPLOY_CHECKPOINT" \
  --detail_file work/eval/cli/ch0.detail.jsonl \
  --dump_frame_topk \
  --score_file work/eval/cli/score.txt
```

- 单文件用 `--wav`，批量用 `--data_list`，二者互斥且必选其一。
- `--checkpoint` 默认为 `auto`（最新 epoch）；部署验证必须显式传 `$DEPLOY_CHECKPOINT` 或 `avg_30.pt`。
- 正确参数是 `--detail_file`，不是 `--detail`。
- `--dump_frame_topk` 必须与 `--detail_file` 同时使用；token 概率和逐帧 top-k 只用于调试。
- `score_file` 沿用 `<key> detected <keyword> <score.3f>` / `<key> rejected` 格式。

批量示例：

```bash
$PYTHON_BIN scripts/04_eval/infer_cli.py \
  --data_list "$EXAMPLE_DATA/test/data.list" \
  --checkpoint "$DEPLOY_CHECKPOINT" \
  --score_file work/eval/cli/test.score.txt
```

### 6.2 WebUI

```bash
source scripts/common/env.sh
$PYTHON_BIN scripts/04_eval/infer_webui.py --checkpoint avg_30.pt --port 7860
```

浏览器访问 `http://127.0.0.1:7860`。WebUI 可选择 `avg_30.pt`、命名 checkpoint 或 epoch checkpoint，并可打开逐帧 top-k 调试；检测结果中的 token 概率同 CLI。

---

## 7. 部署验证

### 7.1 Native host 与三方对齐

```bash
cd /workspace/wekws-workflow
source scripts/common/env.sh
bash scripts/05_deploy/build_android_host.sh
bash scripts/05_deploy/compare_scores.sh android/tests/data/ch0.wav
```

`compare_scores.sh` 默认 `CHECKPOINT=$DEPLOY_CHECKPOINT`，依次输出：

1. ONNX Runtime posterior 与 Python reference score；
2. PyTorch checkpoint 与 ONNX posterior 的最大差异；
3. `infer_core` 检测结果；
4. Native CTC decoder 在 ONNX posterior 上的 score；
5. Native streaming engine（fbank → ONNX → CTC）的结果。

当前对齐记录：

| 音频 | 关键词 | Native / ORT | infer_core | score_file（3 位） |
| :--- | :--- | ---: | ---: | ---: |
| `android/tests/data/ch0.wav` | 嗨小问 | `0.783478` | `0.783476` | `0.783` |
| `android/tests/data/ch_nihao.wav` | 你好问问 | `0.224239` | `0.224240` | `0.224` |

说明：

- 三方 score 在既定容差内一致；差异来自计算路径与显示精度，不改变事件判定。
- `token_probs` / `frame_topk` 只用于调试展示，不纳入对齐判定。
- 若需覆盖 checkpoint，可显式设置环境变量 `CHECKPOINT`，但部署验收默认必须使用 `$DEPLOY_CHECKPOINT`。

### 7.2 APK 构建

```bash
source scripts/common/env.sh
bash scripts/05_deploy/build_apk.sh
```

前置条件：设置 `ANDROID_HOME` 或 `ANDROID_SDK_ROOT`，且 `platforms/` 下存在 `android.jar`。本机当前没有 Android SDK，`build_apk.sh` 会明确失败；因此只完成 Native host 与三方 score 对齐，不声称 APK 已构建或安装验证。

---

## 8. 故障排查

| 现象 | 原因 | 处理 |
| :--- | :--- | :--- |
| `torchaudio` 版本不是 2.5.1，或缺少 `torchaudio.info` | 新版 torchaudio 移除 `torchaudio.info`，数据准备会崩 | 使用 conda 环境并锁定 `torch==2.5.1 torchaudio==2.5.1`；重跑 `check_deps.sh` |
| `sox not found` | sox 不在 pip requirements 中 | 用 `conda install conda-forge::sox -y` 安装 |
| `data.list` 仍指向 `wekws/examples/hi_xiaowen/s0/data/` | 旧清单被直接复用 | 重跑 `regenerate_data_lists.sh`，脚本会拒绝旧路径并检查文件存在 |
| 评测报 CMVN 文件找不到 | 训练 config 里的 CMVN 是相对路径 | 使用 `eval_ctc.sh` 派生的 `$RESULT_DIR/config.yaml`，不要直接拿训练产物 config |
| 关键词变成乱码 | 评测入口传了中文字面量 | 使用 `\u55e8\u5c0f\u95ee,\u4f60\u597d\u95ee\u95ee` |
| `average_model.py` 报 checkpoint/yaml 数量不足 | 可用 checkpoint 少于 30 | 使用实际 N 生成 `avg_N.pt`，或直接用 `final.pt`；不要用 DRY_RUN 伪装训练 |
| CUDA OOM | batch 过大或训练与评测共用 GPU | 只改外层 `config/hi_xiaowen/ds_tcn_ctc.yaml` 的 batch；等待训练结束或降低评估 batch |
| 续训日志从 `Epoch 0` 开始 | 上游只加载权重，不恢复 epoch/优化器 | 正常行为；使用独立 resume 模型和 tensorboard 目录，学习率由 resume config 控制 |
| `build_apk.sh` 报 SDK 未找到 | 本机未配置 Android SDK | 设置 `ANDROID_HOME` / `ANDROID_SDK_ROOT` 并提供 `android.jar`；否则只做 Native host 与 score 对齐 |
| CLI 报 `--dump_frame_topk requires --detail_file` | 打开逐帧调试却未指定 detail 文件 | 同时传入 `--detail_file`；该文件只用于调试 |

---

## 9. 推荐验收清单

1. `bash scripts/common/check_deps.sh` 通过。
2. 数据清单中所有 wav 都在 `work/` 下且文件存在。
3. 训练/续训产物与 tensorboard 日志位于 `work/`，`wekws/` 内无项目落地文件或本地修改。
4. `eval_ctc.sh` 生成 3 位小数 `score.txt`、`stats/*.txt`、`det.png`，并保留当前/历史/部署三类阈值语义。
5. Android 导出资产成组一致，per-keyword 阈值写入 `config.json`，无全局 `ctc.trigger_threshold`。
6. Native / ORT / infer_core 对齐数据符合上表，score 与事件语义未被调试字段改变。
7. 若无 Android SDK，APK 构建保持“未执行/环境缺失”状态，不以 Native host 成功替代 APK 验证。
