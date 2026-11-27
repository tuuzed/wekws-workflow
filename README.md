# wekws-workflow

基于 [wekws](https://github.com/wenet-e2e/wekws) 的中文关键词唤醒（KWS-CTC）工程：从数据准备、训练/续训、FRR/FAR 评估、ONNX 导出，到 Android 端部署与三方 score 对齐的完整工作流。

外层仓库接管所有入口脚本、配置、下载物与运行产物；`wekws/` 只是 pin 在固定 commit 的只读 submodule，不直接修改上游 recipe。

## 固定工作点

| 项目 | 当前值 |
| :--- | :--- |
| 上游代码 | `wekws/` submodule，pin 在 `6a45aeb994dd81c0969ff877a5a7c46d60ed0c86` |
| 部署 checkpoint | `work/exp/hi_xiaowen/ds_tcn_ctc/avg_30.pt`（`$DEPLOY_CHECKPOINT`） |
| 初始微调 base | `work/downloads/mobvoi_kws_transcription/23.pt`（`$BASE_CHECKPOINT`） |
| 关键词与阈值 | `嗨小问=0.303`、`你好问问=0.039`，按关键词独立写入，无全局阈值 |
| Android 模型 | `android/app/src/main/assets/kws.onnx`，直接加载，不转 `.ort` |
| 输出语义 | posterior；Native 侧不再做第二次 softmax |
| score | `ctc_prefix_beam_search` 的 `sqrt(∏ token_prob)`；调试字段不改变 score 或检测结果 |
| 运行产物 | 全部在 `work/`；上游 `wekws/` 内不写入任何项目产物 |

## 目录结构

```text
wekws-workflow/
├── scripts/
│   ├── common/        env.sh（路径与环境唯一来源）、check_deps.sh
│   ├── 01_data/       下载、字典、数据准备、预置产物获取
│   ├── 02_train/      训练与续训
│   ├── 03_export/     通用 ONNX/JIT、Android ONNX 导出
│   ├── 04_eval/       FRR/FAR 评估、CLI/WebUI 推理核心
│   └── 05_deploy/     Native host 构建、score 对比、APK 构建
├── config/
│   ├── hi_xiaowen/    训练与续训配置
│   └── pretrained/    预置归档的下载地址与 sha256
├── android/           Android 部署工程（com.haibingtech.kws）
├── docs/              架构、评测、操作手册与实施计划
├── openspec/          项目契约与已归档变更
├── wekws/             只读 upstream submodule
├── work/              本地产物（gitignore，不入库）
└── infer_*.py         根目录兼容 wrapper，实体在 scripts/04_eval/
```

## 快速开始

以下命令均从仓库根目录执行。先加载统一环境，再按阶段依次运行；`DRY_RUN=1 <命令>` 可以只打印实际命令而不启动训练或评测。

### 1. 环境检查

需要 conda 环境（torch / torchaudio 2.5.1、onnxruntime 1.23.2），`sox` 需通过 conda-forge 安装，不在 `requirements.txt` 中。

```bash
source scripts/common/env.sh
bash scripts/common/check_deps.sh
```

`env.sh` 提供所有路径与 Python 变量，并把 bytecode 缓存指向 `work/pycache`，避免污染只读 submodule。默认 Python 为 `/root/miniconda/envs/wekws/bin/python`，可用 `PYTHON_BIN` / `TORCHRUN_BIN` 覆盖。

### 2. 下载与数据准备

```bash
bash scripts/01_data/download_mobvoi.sh
bash scripts/01_data/download_mobvoi_transcription.sh
bash scripts/01_data/prepare_dict.sh
bash scripts/01_data/prepare_mobvoi.sh
bash scripts/01_data/regenerate_data_lists.sh
```

`regenerate_data_lists.sh` 会重写 `data.list`，确保 wav 路径都指向 `work/`，并拒绝残留的上游旧路径。

完整的 80-epoch 训练产物归档不进 Git（规避平台 100MB LFS 单文件限制），需要单独获取：

默认下载地址写在 `config/pretrained/ds_tcn_ctc.env`（ModelScope 上的 `ds_tcn_ctc.tar.gz`），也可直接传 `--url` 覆盖：

```bash
bash scripts/01_data/download_pretrained.sh
bash scripts/01_data/download_pretrained.sh --download-only   # 只下载与校验
```

脚本校验归档与 `avg_30.pt` 的 sha256、检查 epoch 链，再以不覆盖方式恢复 `work/exp/hi_xiaowen/ds_tcn_ctc/` 与 `work/eval/avg_30_pt/`；已有同名文件不同时会报错并提示 `--force`。

### 3. 训练与续训

```bash
bash scripts/02_train/train_ctc.sh --gpus 0
bash scripts/02_train/resume_ctc.sh --checkpoint "$EXAMPLE_EXP/avg_30.pt" --gpus 0
```

续训使用 `config/hi_xiaowen/ds_tcn_ctc_resume.yaml`（`lr: 0.0005`），写入独立的 `ds_tcn_ctc_resume/` 目录，不覆盖部署产物。上游只加载权重，`epoch` / `lr` 来自 checkpoint 的同名 yaml；`avg_30.pt` 无同名 yaml，因此从 `Epoch 0` 开始是正常的。从预置归档续训的 checkpoint 选择与 `79.pt` 注意事项见[脚本工作流手册](docs/manuals/script_workflow_guide.md) §3.3。

### 4. 评估

```bash
bash scripts/04_eval/eval_ctc.sh
```

生成 `work/eval/avg_30_pt/` 下的 score、stats 与 DET；阈值选点与当前工作点见 [FRR/FAR 评测结果](docs/evaluation/avg30_frr_far_report.md)。

### 5. 导出与推理

```bash
"$PYTHON_BIN" scripts/03_export/export_android_onnx.py \
  --keyword_thresholds '嗨小问=0.303,你好问问=0.039'

"$PYTHON_BIN" scripts/04_eval/infer_cli.py --wav <wav路径> --checkpoint avg_30.pt
"$PYTHON_BIN" scripts/04_eval/infer_webui.py --checkpoint avg_30.pt --port 7860
```

`--keyword_thresholds` 必填；导出会重写 `kws.onnx`、`config.json`、`keywords.txt`、`id_to_token.json`。推理默认 checkpoint 是 `auto`（最新 epoch），部署验证必须显式传 `avg_30.pt`。`token_probs` / `frame_topk` 只用于调试，不改变 score、阈值或 `score_file` 格式。

WebUI 顶部可切换「PyTorch（整段检测）」与「Android ONNX（部署口径，流式）」，下方按「上传检测 / 麦克风流式」两个 Tab 分区，PyTorch 专用设置和日志默认折叠。「上传检测」支持上传音频或直接录音；「麦克风流式」用 Android ONNX 直接加载部署资产，录音中显示 live 事件，停止后点击「结束并生成 final」标记 final。浏览器录音的 48 kHz 会在输入边界自动重采样到 16 kHz，每个会话状态隔离，详见 [ONNX 流式后端说明](docs/manuals/onnx_streaming_backend.md) 与 [脚本工作流指导](docs/manuals/script_workflow_guide.md#62-webui)。

### 6. Android 部署验证

```bash
bash scripts/05_deploy/build_android_host.sh
bash scripts/05_deploy/compare_scores.sh android/tests/data/ch0.wav
```

`compare_scores.sh` 默认使用 `$DEPLOY_CHECKPOINT`，可由 `CHECKPOINT` 覆盖。当前环境没有 Android SDK，`scripts/05_deploy/build_apk.sh` 会明确失败，不会伪装成功。

## 约束提醒

- 训练、评估、导出、推理都走外层 `scripts/` 入口，不要修改上游 `run_ctc.sh` 或 `wekws/` 内文件。
- 部署阈值和 `avg_30.pt` 固定，不要随更训练 checkpoint 静默替换部署资产。
- Android 直接加载 `kws.onnx`，不做 `.ort` 转换；score 与事件语义的变更必须先通过对齐验证。

## 文档

- [文档索引](docs/README.md)
- [当前脚本工作流指导手册](docs/manuals/script_workflow_guide.md)
- [完整训练手册（Conda 环境）](<docs/manuals/wekws + MobvoiHotwords 完整训练手册（Conda 环境）.md>)
- [FRR/FAR 评测操作手册](docs/evaluation/avg30_frr_far_evaluation.md)
- [FRR/FAR 评测结果](docs/evaluation/avg30_frr_far_report.md)
- [Android 部署架构](docs/architecture/android_deployment_architecture.md)
