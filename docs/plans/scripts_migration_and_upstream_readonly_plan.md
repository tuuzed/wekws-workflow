# 脚本归置与 wekws 上游只读迁移计划

> 状态：执行中 / Phase 0-1 部分完成  
> 日期：2026-11-25  
> 更新日期：2026-11-26  
> 参考手册：`docs/manuals/wekws + MobvoiHotwords 完整训练手册（Conda 环境）.md`  
> 注意：参考手册中 `export_onnx.py --output_dir` 与实际代码不一致，本计划已按实际代码修正为 `--onnx_model`。

---

## 1. 目标

将项目侧的数据准备、训练、评估、导出、部署脚本统一归置到：

```text
/workspace/wekws-my/scripts
```

同时将 `wekws` 作为**只读上游依赖**管理：

- 不修改 `wekws` 的源码、脚本、配置；
- 不在 `wekws/` 目录内产生 `data/`、`dict/`、`exp/`、`tensorboard/`、`mobvoi_kws_transcription/`、`__pycache__/` 等任何项目落地文件；
- 外层 `/workspace/wekws-my` 新建独立 git 仓库；
- `wekws` 通过 git submodule 引入并 pin 到指定 commit；
- 迁移验证无误后，恢复 `wekws` 当前已有的两处上游修改。

---

## 2. 已确认决策

| 编号 | 决策 | 说明 |
|---:|---|---|
| 1 | `wekws` 只读，不产生落地文件 | 所有数据、模型、实验、日志、下载物、bytecode 都写到外层 `work/` |
| 2 | 外层创建独立 git 仓库，`wekws` 用 submodule 管理 | `/workspace/wekws-my` 成为项目仓库根 |
| 3 | 迁移验证无误后恢复上游文件 | 先迁移并验证，再恢复 `wekws` 内已修改文件 |
| 4 | 通用导出和 Android 导出都保留 | 同时保留 `wekws/bin/export_onnx.py` / `export_jit.py` 链路和 Android 动态 batch ONNX 导出 |
| 5 | `mobvoi_kws_transcription` 由脚本下载 | 保存完整下载仓库，不手工把 `23.pt` 拷贝到 `models/base/` |
| 6 | 当前 2599 维 checkpoint 方案下，dict 生成是必需步骤 | 必须从 `mobvoi_kws_transcription/tokens.txt` 生成，不能只生成三行简化 dict |

---

## 3. 当前状态盘点

### 3.1 `wekws` 仓库状态

当前观察到：

```text
 M examples/hi_xiaowen/s0/conf/ds_tcn_ctc.yaml
 M examples/hi_xiaowen/s0/run_ctc.sh
?? examples/hi_xiaowen/s0/mobvoi_kws_transcription/
```

具体改动：

- `conf/ds_tcn_ctc.yaml`：`batch_size: 256` 改为 `batch_size: 64`；
- `run_ctc.sh`：增加显式 `--checkpoint` 续训支持，调整自动 base checkpoint 逻辑；
- `mobvoi_kws_transcription/`：从 ModelScope 下载的外部资源仓库，自身含 `.git`。

### 3.2 recipe 目录已有落地产物

```text
examples/hi_xiaowen/s0/data                       29G
examples/hi_xiaowen/s0/dict                       28K
examples/hi_xiaowen/s0/exp                        85M
examples/hi_xiaowen/s0/mobvoi_kws_transcription   71M
examples/hi_xiaowen/s0/tensorboard                12K
```

这些都需要迁出 `wekws/`，使上游目录保持只读、干净。

特别注意：现有 `data/{train,dev,test}/data.list` 中的 `wav` 是旧绝对路径，指向：

```text
/workspace/wekws-my/wekws/examples/hi_xiaowen/s0/data/local/...
```

迁移后必须重新生成 `data.list`，不能只移动目录。

### 3.3 外层已有项目资产

```text
/workspace/wekws-my/
├── .git/
├── .gitignore
├── android/
│   └── app/src/main/assets/id_to_token.json
├── docs/
│   ├── README.md
│   ├── architecture/
│   │   ├── android_deployment_architecture.md
│   │   ├── ctc_inference_score.md
│   │   └── custom_keywords_support.md
│   ├── evaluation/
│   │   └── avg30_frr_far_evaluation.md
│   ├── manuals/
│   │   └── wekws + MobvoiHotwords 完整训练手册（Conda 环境）.md
│   └── plans/
│       └── scripts_migration_and_upstream_readonly_plan.md
├── infer_cli.py
├── infer_core.py
├── infer_webui.py
├── requirements.txt
└── wekws/
```

当前 `/workspace/wekws-my` 已初始化独立 git 仓库，但尚未提交；`.gitignore` 已创建，`requirements.txt` 已锁定 `torch==2.5.1` / `torchaudio==2.5.1`。`scripts/`、`work/`、`config/` 尚未创建。`wekws` 仍是普通目录，当前 commit 为 `6a45aeb994dd81c0969ff877a5a7c46d60ed0c86`，保留两处修改和未跟踪的 `mobvoi_kws_transcription/`。

---

## 4. 目标目录结构

```text
/workspace/wekws-my/                         # 新项目 git 仓库
├── .git/
├── .gitmodules                              # wekws submodule 配置
├── .gitignore
├── config/
│   └── hi_xiaowen/
│       ├── ds_tcn_ctc.yaml                  # 上游 yaml 副本，batch_size=64
│       └── ds_tcn_ctc_resume.yaml           # 续训用，lr 单独调低
├── scripts/
│   ├── common/
│   │   └── env.sh                           # 路径、Python、CUDA、pycache 环境
│   ├── 01_data/
│   │   ├── download_mobvoi.sh               # 下载 MobvoiHotwords 音频/资源
│   │   ├── download_mobvoi_transcription.sh # 下载完整 mobvoi_kws_transcription 仓库
│   │   └── prepare_mobvoi.sh                # dict/wav.scp/text/wav.dur/data.list/CMVN
│   ├── 02_train/
│   │   ├── train_ctc.sh                     # 训练入口，显式 config/dict/checkpoint/model_dir
│   │   └── resume_ctc.sh                    # 续训入口，显式 resume config + checkpoint
│   ├── 03_export/
│   │   ├── export_onnx.sh                   # 手册链路：wekws/bin/export_onnx.py
│   │   ├── export_jit.sh                    # 手册链路：wekws/bin/export_jit.py
│   │   ├── export_android_onnx.py           # Android 动态 batch ONNX 导出
│   │   └── download_ort_mobile.sh           # ORT Mobile headers/libs 下载
│   ├── 04_eval/
│   │   ├── eval_ctc.sh                      # average + score_ctc + compute_det_ctc
│   │   ├── infer_cli.py                     # CLI 推理实体脚本
│   │   └── infer_webui.py                   # WebUI 推理实体脚本
│   └── 05_deploy/
│       ├── build_android_host.sh
│       ├── compare_scores.sh
│       └── build_apk.sh
├── work/                                    # gitignore，所有落地产物
│   ├── pycache/                             # PYTHONPYCACHEPREFIX 目标
│   ├── downloads/
│   │   ├── mobvoi/                          # MobvoiHotwords tgz / 解压音频 / JSON
│   │   └── mobvoi_kws_transcription/        # 完整下载仓库，含 23.pt / *.text / tokens.txt
│   ├── data/hi_xiaowen/
│   │   ├── train/
│   │   ├── dev/
│   │   └── test/
│   ├── dict/hi_xiaowen/
│   ├── tensorboard/hi_xiaowen/ds_tcn_ctc/
│   ├── exports/hi_xiaowen/
│   └── exp/hi_xiaowen/ds_tcn_ctc/
├── android/
├── docs/
├── infer_core.py                            # 公共库，保留根目录，供 CLI/WebUI/脚本复用
├── requirements.txt                         # 锁定 torch/torchaudio 2.5.1
└── wekws/                                   # git submodule，只读
```

说明：

- `mobvoi_kws_transcription` 完整仓库放在 `work/downloads/mobvoi_kws_transcription/`，不是 `models/base/`；
- `23.pt` 直接引用下载仓库中的路径：
  ```text
  work/downloads/mobvoi_kws_transcription/23.pt
  ```
- `models/` 不再是必要目录。

---

## 5. Git 与 submodule 计划

### 5.1 外层仓库初始化

已完成：`/workspace/wekws-my` 已创建独立 git 仓库和 `.gitignore`，后续只需核对覆盖范围并提交。

建议 `.gitignore`（当前已落地）：

```gitignore
# 所有项目运行产物
work/

# Python
__pycache__/
*.py[cod]

# Android 构建产物
android/.gradle/
android/app/build/
android/build_host/
android/local.properties
android/third_party/onnxruntime/

# 推理输出
infer_cli/
*.onnx

# 系统文件
.DS_Store
```

说明：

- `android/app/src/main/assets/config.json`、`keywords.txt` 纳入 git；
- `android/app/src/main/assets/id_to_token.json` 是文本调试资产，纳入 git；
- `android/app/src/main/assets/kws.onnx` 是二进制部署产物，默认忽略，由导出脚本生成；
- 新 clone 后 Android 工程不会自带 ONNX，必须先执行：
  ```bash
  python scripts/03_export/export_android_onnx.py --checkpoint <部署 checkpoint>
  ```
  才能构建可运行 APK。

### 5.2 `wekws` submodule 转换方案

采用**重新 clone + 旧目录备份**方案，避免在当前含本地修改和未跟踪大目录的 repo 上直接做 submodule absorb。

当前观察 commit：

```text
6a45aeb994dd81c0969ff877a5a7c46d60ed0c86
```

当前 diff/备份尚未生成；`wekws` 内仍有两处修改和未跟踪的 `mobvoi_kws_transcription/`。

步骤：

1. 记录当前上游 commit：

   ```bash
   git -C /workspace/wekws-my/wekws rev-parse HEAD > /tmp/wekws_commit.txt
   ```

2. 保存当前 diff 到外层备份位置：

   ```bash
   git -C /workspace/wekws-my/wekws diff \
     > /workspace/wekws-my/docs/plans/backups/wekws_local_changes.diff
   ```

3. 完成所有脚本迁移和验证后，在验证通过的前提下：

   ```bash
   cd /workspace/wekws-my
   mv wekws wekws.old
   git submodule add https://github.com/wenet-e2e/wekws.git wekws
   git -C wekws checkout "$(cat /tmp/wekws_commit.txt)"
   ```

4. 确认：

   ```bash
   git -C wekws status --short
   ```

   输出应为空。

5. `wekws.old` 保留到最终验收完成后再删除，避免误删用户数据。

`.gitmodules` URL：

```text
https://github.com/wenet-e2e/wekws.git
```

后续更新上游时只在 submodule 内显式 pull/checkout，不修改其文件。

---

## 6. 脚本职责与上游调用方式

### 6.1 统一环境

`scripts/common/env.sh` 提供：

```bash
WORKSPACE_ROOT=/workspace/wekws-my
WEKWS_ROOT=$WORKSPACE_ROOT/wekws
WORK=$WORKSPACE_ROOT/work
CONFIG_DIR=$WORKSPACE_ROOT/config/hi_xiaowen
DOWNLOADS=$WORK/downloads
MOBVOI_TRANSCRIPTION_DIR=$DOWNLOADS/mobvoi_kws_transcription
EXAMPLE_DATA=$WORK/data/hi_xiaowen
EXAMPLE_DICT=$WORK/dict/hi_xiaowen
EXAMPLE_EXP=$WORK/exp/hi_xiaowen/ds_tcn_ctc
EXAMPLE_TB=$WORK/tensorboard/hi_xiaowen/ds_tcn_ctc

export PYTHONPATH="$WEKWS_ROOT:${PYTHONPATH:-}"
export PYTHONIOENCODING=UTF-8
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

# 防止 Python 在只读上游 wekws/ 内写 __pycache__
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPYCACHEPREFIX="$WORK/pycache"
```

Python 环境按手册使用 conda `wekws`，并在 `requirements.txt` 中锁定：

```text
torch==2.5.1
torchaudio==2.5.1
```

还需通过 conda-forge 安装 `sox`。`env.sh` 可加入版本检查，发现 `torchaudio >= 2.9` 或缺少 `torchaudio.info` 时明确报错。

### 6.2 数据准备

| 脚本 | 作用 | 上游依赖 |
|---|---|---|
| `download_mobvoi.sh` | 下载 MobvoiHotwords 音频/resource tgz 到 `work/downloads/mobvoi/` | 也可调用 `local/mobvoi_data_download.sh`，但输出重定向到外层 |
| `download_mobvoi_transcription.sh` | clone `mobvoi_kws_transcription` 到 `work/downloads/mobvoi_kws_transcription/` | ModelScope git repo |
| `prepare_mobvoi.sh` | 生成 dict / wav.scp / text / wav.dur / data.list / global_cmvn | `local/prepare_data.py`、`tools/compute_cmvn_stats.py`、`tools/wav_to_duration.sh`、`tools/make_list.py` |

原则：

- 调用上游文件时全部使用绝对路径；
- 输出路径全部指向 `$WORK`；
- 不在 `wekws/` 内创建任何文件；
- `data.list` 必须重新生成，确保其中 wav 路径指向迁移后的 `$WORK`；
- 大目录迁移前后用 `du -sh` 和文件数对比校验，验证通过前不删除原目录。

#### dict 生成：必需步骤

当前 base checkpoint `23.pt` 是 2599 维输出，token id 与 `mobvoi_kws_transcription/tokens.txt` 绑定：

```text
你=1027
嗨=1302
好=1401
小=1462
问=2494
```

因此不能只生成手册 stage -2 的三行简化 dict。`prepare_mobvoi.sh` 必须执行等价的 stage -1 dict 转换：

```bash
awk '{print $1, $2-1}' \
  "$MOBVOI_TRANSCRIPTION_DIR/tokens.txt" \
  > "$EXAMPLE_DICT/dict.txt"

sed -i 's/& 1/<filler> 1/' "$EXAMPLE_DICT/dict.txt"

printf '<SILENCE>\n<EPS>\n<BLK>\n' > "$EXAMPLE_DICT/words.txt"
```

随后再用该 dict 调用 `local/prepare_data.py` 生成 `text`。

> 手册中“stage -1 可选”的说法只适用于不依赖 2599 维 base checkpoint 的其他建模方案；本工程当前方案下，dict 生成步骤是必需的。

### 6.3 训练与续训

`train_ctc.sh` 直接调用上游训练入口，不再依赖修改后的上游 `run_ctc.sh`：

```bash
torchrun --standalone --nnodes=1 --nproc_per_node="$NUM_GPUS" \
  "$WEKWS_ROOT/wekws/bin/train.py" \
    --gpus "$GPUS" \
    --config "$CONFIG_DIR/ds_tcn_ctc.yaml" \
    --dict "$EXAMPLE_DICT" \
    --train_data "$EXAMPLE_DATA/train/data.list" \
    --cv_data "$EXAMPLE_DATA/dev/data.list" \
    --model_dir "$EXAMPLE_EXP" \
    --tensorboard_dir "$EXAMPLE_TB" \
    --num_workers 8 \
    --num_keywords 2599 \
    --min_duration 50 \
    --seed 666 \
    --cmvn_file "$EXAMPLE_DATA/train/global_cmvn" \
    --norm_var \
    --checkpoint "${CHECKPOINT:-$MOBVOI_TRANSCRIPTION_DIR/23.pt}"
```

关键点：

- 默认 checkpoint 来自完整下载仓库：
  ```text
  $MOBVOI_TRANSCRIPTION_DIR/23.pt
  ```
- 外层 config 副本承载 `batch_size=64`；
- 必须显式传 `--dict`，否则 `train.py` 会用默认 `./dict`；
- 必须显式传 `--tensorboard_dir`，否则日志写到外层 cwd；
- `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` 固化在 env。

#### 续训脚本

`train.py` **没有 `--lr` 参数**。学习率只存在于 yaml。因此 `resume_ctc.sh` 的做法是：

1. 复制外层 config：

   ```bash
   cp "$CONFIG_DIR/ds_tcn_ctc.yaml" "$CONFIG_DIR/ds_tcn_ctc_resume.yaml"
   ```

2. 修改学习率：

   ```bash
   sed -i 's/lr: 0.001/lr: 0.0005/' "$CONFIG_DIR/ds_tcn_ctc_resume.yaml"
   ```

3. 用 resume config 启动训练：

   ```bash
   --config "$CONFIG_DIR/ds_tcn_ctc_resume.yaml"
   ```

其他要求：

- 必须显式传 `--checkpoint`；
- 拒绝从 `init.pt` 续训；
- 续训默认写入独立模型目录，例如：
  ```text
  $WORK/exp/hi_xiaowen/ds_tcn_ctc_resume_<timestamp>
  ```
- 避免手册中提到的 epoch 重记导致日志混淆。

### 6.4 评估

`eval_ctc.sh` 对应手册 stage 3，调用上游：

```bash
python "$WEKWS_ROOT/wekws/bin/average_model.py" \
  --dst_model "$EXAMPLE_EXP/avg_30.pt" \
  --src_path "$EXAMPLE_EXP" \
  --num 30 \
  --val_best

python "$WEKWS_ROOT/wekws/bin/score_ctc.py" \
  --config "$EXAMPLE_EXP/config.yaml" \
  --dict "$EXAMPLE_DICT" \
  --test_data "$EXAMPLE_DATA/test/data.list" \
  --gpu 0 \
  --batch_size "$SCORE_BATCH_SIZE" \
  --checkpoint "$EXAMPLE_EXP/avg_30.pt" \
  --score_file "$RESULT_DIR/score.txt" \
  --num_workers 8 \
  --keywords '\u55e8\u5c0f\u95ee,\u4f60\u597d\u95ee\u95ee'

python "$WEKWS_ROOT/wekws/bin/compute_det_ctc.py" \
  --test_data "$EXAMPLE_DATA/test/data.list" \
  --dict "$EXAMPLE_DICT" \
  --keywords '\u55e8\u5c0f\u95ee,\u4f60\u597d\u95ee\u95ee' \
  --score_file "$RESULT_DIR/score.txt" \
  --stats_dir "$RESULT_DIR/stats" \
  --det_curve_path "$RESULT_DIR/stats/det.png" \
  --window_shift 50 \
  --step 0.001
```

注意：

- `score_ctc.py` / `compute_det_ctc.py` 会对 `--keywords` 做 `encode('utf-8').decode('unicode_escape')`，因此必须传 ASCII 转义，不能传中文字面量；
- 两者都必须显式传 `--dict`；
- 输出目录：
  ```text
  $RESULT_DIR=$EXAMPLE_EXP/test_avg_30.pt
  ```

#### average fallback

`average_model.py --val_best --num 30` 要求至少 30 个可用 checkpoint/yaml，否则会触发：

```text
assert num == len(path_list)
```

脚本应先检查可用 checkpoint 数：

- 若 >= 30：使用 `avg_30.pt`；
- 若 < 30：使用实际 N 生成 `avg_N.pt`，或直接用 `final.pt`；
- 不把“缺少 30 个 checkpoint”误判为训练失败。

FRR / FAR 评测已完成，工作点按 `FAR ≤ 0.2/h` 选定为独立阈值 `嗨小问=0.303`、`你好问问=0.039`，依据见 `docs/evaluation/avg30_frr_far_evaluation.md`。阈值通过导出脚本参数写入：

```bash
--keyword_thresholds '嗨小问=0.303,你好问问=0.039'
```

### 6.5 导出

保留两套导出。

#### A. 通用 ONNX / JIT（手册链路）

实际 `export_onnx.py` 参数是 `--onnx_model`，不是手册中的 `--output_dir`：

```text
scripts/03_export/export_onnx.sh
scripts/03_export/export_jit.sh
```

命令：

```bash
python -m wekws.bin.export_onnx \
  --config "$EXAMPLE_EXP/config.yaml" \
  --checkpoint "$EXAMPLE_EXP/avg_30.pt" \
  --onnx_model "$EXPORT_DIR/final.onnx"

python -m wekws.bin.export_jit \
  --config "$EXAMPLE_EXP/config.yaml" \
  --checkpoint "$EXAMPLE_EXP/avg_30.pt" \
  --jit_model "$EXPORT_DIR/avg_30.zip"
```

其中：

```text
EXPORT_DIR=$WORK/exports/hi_xiaowen
```

#### B. Android 动态 batch ONNX

```text
scripts/03_export/export_android_onnx.py
```

由现有 `android/export/export_ctc_onnx.py` 迁移而来，修正路径锚点：

```python
WORKSPACE_ROOT = Path(__file__).resolve().parents[2]
WEKWS_ROOT = WORKSPACE_ROOT / "wekws"
ANDROID_ROOT = WORKSPACE_ROOT / "android"
WORK = WORKSPACE_ROOT / "work"
```

默认 checkpoint 改为外层实验目录：

```text
$WORK/exp/hi_xiaowen/ds_tcn_ctc/avg_30.pt
```

输出：

```text
android/app/src/main/assets/kws.onnx
android/app/src/main/assets/config.json
android/app/src/main/assets/keywords.txt
android/app/src/main/assets/id_to_token.json
```

导出顺序必须固定：

1. 用 `$WORK/exp/hi_xiaowen/ds_tcn_ctc/avg_30.pt` 导出 ONNX / config / keywords / id_to_token；
2. 导出时通过 `--keyword_thresholds` 写入每个关键词的 `trigger_threshold`；
3. `ctc.trigger_threshold` 已删除，不再需要导出后 patch；缺少任一关键词阈值时导出失败。

保持动态 batch：

```text
input:   [N,T,40]
cache:   [N,256,105]
output:  [N,T,2599]
r_cache: [N,256,105]
```

ONNX 输出是 softmax posterior，不是 logits；Android Native 不做二次 softmax。

### 6.6 推理脚本路径修正

当前 `infer_core.py` 的常量仍指向上游 recipe：

```python
ROOT = Path(__file__).resolve().parent
REPO = ROOT / "wekws"
EXAMPLE = REPO / "examples" / "hi_xiaowen" / "s0"
EXP_DIR = EXAMPLE / "exp" / "ds_tcn_ctc"
```

迁移后必须改为读取外层环境：

```python
WORKSPACE_ROOT = Path(os.environ.get(
    "WORKSPACE_ROOT", str(Path(__file__).resolve().parent)))
WORK = Path(os.environ.get("WORK", str(WORKSPACE_ROOT / "work")))
REPO = WORKSPACE_ROOT / "wekws"

EXAMPLE = WORK / "data" / "hi_xiaowen"
DICT_DIR = WORK / "dict" / "hi_xiaowen"
EXP_DIR = WORK / "exp" / "hi_xiaowen" / "ds_tcn_ctc"
```

同时：

- `infer_cli.py` / `infer_webui.py` 实体迁移到 `scripts/04_eval/`，根目录保留薄 wrapper 兼容旧命令；
- `infer_cli.py` 的默认 `score_file` 从 `ROOT/infer_cli/score.txt` 改为 `WORK/eval/score.txt`；
- 所有默认 config / dict / checkpoint 路径都指向 `WORK`，不得指向 `wekws/examples/.../exp`。

迁移时必须保留现有推理能力，不能只搬路径：

- `infer_core.py`：`list_named_checkpoints()`、命中结果 `token_probs`、`include_frame_topk=True`；
- `infer_cli.py`：`--detail_file`、`--dump_frame_topk`，同时保持原 `--score_file` 格式不变；
- `infer_webui.py`：checkpoint 下拉框包含 `avg_30.pt`，支持“输出每帧 top-k（调试）”复选框。

### 6.7 部署验证脚本默认值修正

必须修正以下默认值，否则仍会读取上游目录：

| 文件 | 当前默认值 | 修正后 |
|---|---|---|
| `android/tests/run_compare.sh` | `$ROOT/../wekws/examples/.../15.pt` | `$WORK/exp/hi_xiaowen/ds_tcn_ctc/avg_30.pt`，可由 `CHECKPOINT` 覆盖 |
| `android/export/export_ctc_onnx.py` | 上游 `15.pt` | `$WORK/exp/hi_xiaowen/ds_tcn_ctc/avg_30.pt`，可由 `--checkpoint` 覆盖 |
| `android/export/export_ctc_onnx.py` | 无 `--keyword_thresholds` 参数 | 必填参数，写入 `keywords[].trigger_threshold` |
| `android/app/src/main/assets/config.json` | 当前仍是旧工作点 | 迁移后写入 `嗨小问=0.303`、`你好问问=0.039`，并在导出后校验 |
| `docs/*.md`、`android/README.md` | 旧 `run_ctc.sh` / 上游 exp 路径 | 新 `scripts/` 命令和 `$WORK` 路径 |

部署脚本：

| 脚本 | 作用 |
|---|---|
| `build_android_host.sh` | 封装 `android/tests/build_host.sh`，构建 Native host CLI |
| `compare_scores.sh` | 封装 `android/tests/run_compare.sh`，跑 Python/ONNX/Native score 对齐 |
| `build_apk.sh` | 在 Android SDK 可用时执行 `./gradlew assembleDebug` |

---

## 7. 迁移执行步骤

### Phase 0：准备与备份

1. 记录 `wekws` 当前 commit 和 diff。
2. 盘点 recipe 下已有产物，确认没有未知用户工作。
3. 把上游 diff 备份到：
   ```text
   docs/plans/backups/wekws_local_changes.diff
   ```
4. 外层 git 仓库已初始化，后续迁移只在新仓库内操作；仍先不修改上游文件。

### Phase 1：建立外层结构

1. 外层 `.gitignore` 已创建，后续核对覆盖范围并提交。
2. 创建 `config/hi_xiaowen/ds_tcn_ctc.yaml`，复制上游 yaml 并保留 `batch_size: 64`。
3. 创建 `config/hi_xiaowen/ds_tcn_ctc_resume.yaml`，`lr: 0.0005`。
4. 创建 `scripts/` 和 `common/env.sh`。
5. 创建 `work/` 及 `pycache/downloads/data/dict/tensorboard/exports/exp` 子目录。
6. `requirements.txt` 已锁定 `torch==2.5.1`、`torchaudio==2.5.1`，后续只做校验。

### Phase 2：迁移下载资源与产物

1. 将整个 `mobvoi_kws_transcription/` 从 `wekws/examples/hi_xiaowen/s0/` 移到：
   ```text
   work/downloads/mobvoi_kws_transcription/
   ```
2. 保留其自身 `.git`。
3. 实现 `download_mobvoi_transcription.sh`：目录不存在时 clone，存在时跳过。
4. 将 `data/`、`dict/`、`exp/`、`tensorboard/` 迁出到 `work/` 对应目录。
5. 重新生成 `data.list`，确保 wav 路径指向迁移后的 `$WORK`。
6. 大目录迁移前后对比 `du -sh` 和文件数；验证通过前不删除旧目录。

### Phase 3：实现脚本

1. 数据准备脚本，包含强制 dict 生成。
2. 训练 / 续训脚本。
3. 评估脚本，修正 keywords 转义和 dict 参数。
4. 通用 ONNX/JIT 导出脚本，使用 `--onnx_model` / `--jit_model`。
5. Android 导出脚本迁移与默认值修正。
6. `infer_core.py` 路径常量修正；CLI/WebUI 迁移与 wrapper。
7. 部署验证脚本默认值修正。
8. 文档路径更新。

### Phase 4：等价验证

必须验证：

1. 外层 config 训练参数与当前有效配置一致（尤其 `batch_size=64`、`num_keywords=2599`）；
2. 默认 checkpoint 指向 `work/downloads/mobvoi_kws_transcription/23.pt`；
3. dict 中 `你/嗨/好/小/问` 的 token id 与原 dict 一致；
4. `data.list` 中 wav 路径全部位于 `$WORK`；
5. `train_ctc.sh` 能正常启动训练，产物写 `$EXAMPLE_EXP`，tensorboard 写 `$EXAMPLE_TB`；
6. `resume_ctc.sh` 使用 resume config，`lr` 已调低，输出独立目录；
7. `eval_ctc.sh` 能产出 `avg_N.pt` / `final.pt` 和 FRR/FAR；
8. `export_onnx.sh` 能用 `--onnx_model` 导出通用 ONNX；
9. `export_jit.sh` 能用 `--jit_model` 导出 JIT；
10. `export_android_onnx.py` 能导出动态 batch `kws.onnx`；
11. `infer_cli.py` 对现有测试 wav 的 score 与迁移前一致；
12. `compare_scores.sh` 中 Python / ONNX / Native score 对齐不回归；
13. `wekws/` 内没有 `data/`、`dict/`、`exp/`、`tensorboard/`、`mobvoi_kws_transcription/`、`__pycache__/`；
14. CLI / WebUI / Android 事件 JSON 均包含 `token_probs`，且 `sqrt(prod(token_probs)) == score`；
15. `--dump_frame_topk` / `frameDebug()` 的 token id 与 ORT 参考一致，默认关闭时输出为空；
16. `id_to_token.json` 能正确映射 `1302/1462/2494` 等关键词 token；
17. 用 `avg_30.pt` 导出后，`android/app/src/main/assets/config.json` 的 `keywords[].trigger_threshold` 为 `嗨小问=0.303`、`你好问问=0.039`，且不含 `ctc.trigger_threshold`；
18. `--score_file` 原格式保持不变，不影响 DET 评测。

### Phase 5：恢复上游并转 submodule

1. 验证通过后恢复：

   ```bash
   git -C /workspace/wekws-my/wekws checkout -- \
     examples/hi_xiaowen/s0/conf/ds_tcn_ctc.yaml \
     examples/hi_xiaowen/s0/run_ctc.sh
   ```

2. 确认 `mobvoi_kws_transcription/` 已离开 `wekws/`。
3. 确认 `git -C wekws status --short` 为空。
4. 按 5.2 的方案把 `wekws` 转为 submodule 并 pin commit。
5. 最终验收后再删除 `wekws.old`。

---

## 8. 风险与处理

| 风险 | 处理 |
|---|---|
| 29G `data/` 迁移耗时 | 同一文件系统内优先 `mv`；迁移前后校验大小和文件数 |
| `data.list` 保留旧绝对路径 | 迁移后重新生成 `data.list`，不直接复用 |
| `mobvoi_kws_transcription` 自身是 git repo | 作为独立下载资源整体移动，不复制单文件、不拆散 `.git` |
| dict 生成错误导致 token id 错位 | 强制从 `tokens.txt` 生成，并校验 `你/嗨/好/小/问` 的 id |
| 上游转 submodule 操作失误 | 记录 commit/diff，使用重新 clone + 旧目录备份方案 |
| 外层路径未完全覆盖 | 所有脚本统一 source `env.sh`，禁止硬编码 recipe 相对路径 |
| Python 在 `wekws/` 写 `__pycache__` | `PYTHONDONTWRITEBYTECODE=1` + `PYTHONPYCACHEPREFIX=$WORK/pycache` |
| `torchaudio` 版本错误 | `requirements.txt` 锁 2.5.1，env 检查 `torchaudio.info` |
| 续训日志混淆 | `resume_ctc.sh` 使用独立模型目录和 resume config |
| `score_ctc.py` keywords 乱码 | 传 ASCII unicode escape，不传中文字面量 |
| ONNX 导出参数错误 | 使用 `--onnx_model`，不是手册中的 `--output_dir` |
| 误唤醒阈值未校准 | FRR/FAR 已完成，按 `FAR ≤ 0.2/h` 独立阈值定为 `嗨小问=0.303`、`你好问问=0.039`，依据见 `docs/evaluation/avg30_frr_far_evaluation.md` |
| 导出脚本覆盖阈值 | `--keyword_thresholds` 为必填参数，写入 per-keyword 阈值，并加入 Phase 4 验证 |
| debug 输出导致 JSON 过大 | 默认关闭 `dump_frame_topk`，WebUI 只展示前 100 帧 |
| `avg_30.pt` 已生成但训练仍运行 | 部署来源固定为 `avg_30.pt`，后续新 checkpoint 不自动覆盖部署模型 |

---

## 9. 验收标准

满足以下条件视为迁移完成：

1. `/workspace/wekws-my` 是独立 git 仓库；
2. `wekws` 是 submodule，且 `git -C wekws status --short` 为空；
3. `wekws/` 内无任何项目落地文件，包括 `data/`、`dict/`、`exp/`、`tensorboard/`、`mobvoi_kws_transcription/`、`__pycache__/`；
4. 所有项目入口命令都在 `scripts/` 下有对应脚本；
5. 数据、模型、实验、日志、下载物、bytecode 全部落在 `work/`；
6. `mobvoi_kws_transcription` 由下载脚本管理，完整仓库位于 `work/downloads/`；
7. dict 从 `tokens.txt` 生成，关键词 token id 与迁移前一致；
8. `data.list` wav 路径全部指向 `$WORK`；
9. 训练、续训、评估、通用 ONNX/JIT 导出、Android 导出、CLI/WebUI 推理、host 对齐均验证通过；
10. 文档已更新为新的推荐命令路径；
11. Android ONNX 由 `avg_30.pt` 导出，`id_to_token.json` 已生成并纳入 git；
12. `android/app/src/main/assets/config.json` 的 `keywords[].trigger_threshold` 为 `嗨小问=0.303`、`你好问问=0.039`，且不含 `ctc.trigger_threshold`；
13. CLI / WebUI / Android 均可输出 `token_probs`，且 score 公式未变；
14. 逐帧 top-k 默认关闭，开启后与 ORT 参考 token id 一致。

---

## 10. 一句话总结

本计划的核心是：**外层项目仓库接管所有脚本、配置、下载物、dict、数据和实验产物；`wekws` 只作为 git submodule 提供上游代码，保持只读干净；训练与部署链路通过外层脚本显式传 config/dict/checkpoint/model_dir 完成，迁移验证通过后再恢复上游文件并转 submodule。**
