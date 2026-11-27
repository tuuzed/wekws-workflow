# AGENTS.md

## 边界与事实来源
- 主仓库根目录 `/workspace/wekws-workflow`；`wekws/` 是 pin 在 `6a45aeb994dd81c0969ff877a5a7c46d60ed0c86` 的只读 submodule（URL `https://github.com/wenet-e2e/wekws.git`），只提交 gitlink，不写入上游 `data/exp/__pycache__`。
- `scripts/common/env.sh` 是路径、Python、checkpoint、阈值的唯一来源；跑任何入口前先 `source scripts/common/env.sh`。根目录 `infer_cli.py`/`infer_webui.py`/`infer_core.py` 只是兼容 wrapper，实体在 `scripts/04_eval/`。
- `work/`、`infer_cli/`、`__pycache__`、`*.onnx` 已忽略；只有 `android/app/src/main/assets/kws.onnx` 显式入库。Android 包名 `com.haibingtech.kws`。

## 当前固定事实
- Python 默认 `/root/miniconda/envs/wekws/bin/python`（torch/torchaudio 2.5.1、onnxruntime 1.23.2）；`sox` 来自 conda-forge，不在 `requirements.txt`。`env.sh` 设置 `PYTHONDONTWRITEBYTECODE=1` 与 `PYTHONPYCACHEPREFIX=$WORK/pycache`，别让字节码落到 `wekws/`。
- 初始微调 base 是 `$BASE_CHECKPOINT=work/downloads/mobvoi_kws_transcription/23.pt`；部署 checkpoint 固定 `$DEPLOY_CHECKPOINT=work/exp/hi_xiaowen/ds_tcn_ctc/avg_30.pt`，不要静默换成更新 epoch。
- 阈值按关键词独立：`嗨小问=0.303`、`你好问问=0.039`；没有全局 `ctc.trigger_threshold`，缺失/非法 `keywords[].trigger_threshold` 会直接报错。
- Android 直接加载 `kws.onnx`（绝不转 `.ort`）：动态 batch `[N,T,40]`/`[N,256,105]` -> `[N,T,2599]`，输出 posterior，Native 不再 softmax。一个 ORT Session、单 decode 线程、每 channel 独立 mutex/frontend/cache/decoder/offset；“多路”=独立音频流。
- score 严格对齐 `ctc_prefix_beam_search`：`sqrt(∏ token_prob)`，beam 3 / path 20 / `prob>0.05` / `abs_tol=1e-6`。`token_probs`/`frame_topk` 仅供调试，不得改变 score、阈值、检测结果或 `score_file` 格式。
- `currentEvents()` 是每 channel/关键词最近一次在当前 beam 中观察到的命中，不是历史最高分；同一 channel 可上报多个关键词，Engine 不仲裁。

## 命令与坑（均从仓库根目录）
- 导出部署资产：`"$PYTHON_BIN" scripts/03_export/export_android_onnx.py --keyword_thresholds '嗨小问=0.303,你好问问=0.039'`；`--keyword_thresholds` 必填，默认 checkpoint 已固定，会重写 `kws.onnx`/`config.json`/`keywords.txt`/`id_to_token.json`。旧 `android/export/export_ctc_onnx.py` 只是 wrapper。
- 推理：`"$PYTHON_BIN" scripts/04_eval/infer_cli.py --wav ... --checkpoint avg_30.pt`；WebUI 用 `"$PYTHON_BIN" scripts/04_eval/infer_webui.py --checkpoint avg_30.pt --port 7860`。两者默认 `auto`（最新 epoch），部署验证必须显式传 `avg_30.pt`；`--dump_frame_topk` 需要 `--detail_file`，CLI 默认 `score_file` 是 `work/eval/cli/score.txt`。
- 评估：`bash scripts/04_eval/eval_ctc.sh`；训练 config 的 CMVN 是相对路径，不能直接拿它跑 `score_ctc.py`，脚本派生 `$RESULT_DIR/config.yaml` 绝对路径；score/DET 必须显式 `--dict` 且关键词传 ASCII escape `'\u55e8\u5c0f\u95ee,\u4f60\u597d\u95ee\u95ee'`。当前 `work/data` 缺 `global_cmvn`/`wav.dur`/`data.list`，重跑前先 `bash scripts/01_data/prepare_mobvoi.sh && bash scripts/01_data/regenerate_data_lists.sh`，否则报 CMVN 找不到。
- 训练/续训：`bash scripts/02_train/train_ctc.sh --gpus 0`、`bash scripts/02_train/resume_ctc.sh --checkpoint "$EXAMPLE_EXP/avg_30.pt" --gpus 0`。上游 `train.py` 没有 `--lr`，lr 只来自配置；`avg_30.pt` 无同名 yaml，续训从 Epoch 0 正常；`79.pt` 带 epoch79/lr1e-6 的同名 yaml，当前 `max_epoch: 80` 会使其不训练；不要用 `train_ctc.sh` 指向归档 checkpoint 覆盖 `$EXAMPLE_EXP`。
- Native 对齐：`bash scripts/05_deploy/build_android_host.sh` 需要 ORT headers，缺失时按提示跑 `bash android/export/download_ort_mobile.sh`；再 `bash scripts/05_deploy/compare_scores.sh android/tests/data/ch0.wav`，`CHECKPOINT` 默认 `$DEPLOY_CHECKPOINT`。无 Android SDK 时 `build_apk.sh` 明确失败。
- 预置训练产物：`bash scripts/01_data/download_pretrained.sh`；默认地址/sha 在 `config/pretrained/ds_tcn_ctc.env`，可用 `--url` 覆盖；归档到 `work/downloads/pretrained/`，校验 sha256 与 epoch 链 `0..79` 后落入 `$EXAMPLE_EXP`/`$RESULT_DIR`，默认不覆盖（`--force` 才覆盖），不写 `wekws/`。
- `DRY_RUN=1` 只对走 `kws_run` 的训练/续训/评估/通用导出入口生效；数据准备和 Android 导出仍会写文件。

## 溯源
- 历史文档中的 `15.pt`、“联调 ONNX”、阈值 `0.297/0.000` 仅作追溯；当前部署以导出脚本、`assets/config.json` 与 `compare_scores.sh` 输出为准（ch0/ch_nihao score `0.783478 / 0.224239`）。
- FRR/FAR 当前复算点（`awk '$2 <= 0.2 {print; exit}'`）是 `嗨小问=0.293`、`你好问问=0.038`；部署工作点仍是 `0.303/0.039`，报告在 `docs/evaluation/avg30_frr_far_report.md`。
- OpenSpec 当前无 active change；契约在 `openspec/specs/`，历史 change 在 `openspec/changes/archive/`。不改归档产物；需要变更时按 propose/apply 流程。
