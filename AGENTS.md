# AGENTS.md

## 作用域与归属
- 主仓库是 `/workspace/wekws-my`；`wekws/` 已是 pin 在 `6a45aeb994dd81c0969ff877a5a7c46d60ed0c86` 的只读 submodule（URL `https://github.com/wenet-e2e/wekws.git`），只提交 gitlink，不提交上游源码。不要把 `data/`、`exp/`、`__pycache__` 留在里面。
- `android/` 是部署工程（包名 `com.haibingtech.kws`）；`infer_core.py` 由 `infer_cli.py` 和 `infer_webui.py` 共用；文档在 `docs/`。
- `work/`、`infer_cli/`、`__pycache__`、`*.onnx` 已加入 gitignore；只有 `android/app/src/main/assets/kws.onnx` 被显式重新纳入。

## 环境与部署事实
- 使用 `/root/miniconda/envs/wekws/bin/python`（torch/torchaudio 2.5.1、onnxruntime 1.23.2）。`sox` 来自 conda-forge，不在 pip requirements 里。设置 `PYTHONDONTWRITEBYTECODE=1`，并把 `PYTHONPYCACHEPREFIX` 指向 `work/`。
- 部署 checkpoint 固定为 `work/exp/hi_xiaowen/ds_tcn_ctc/avg_30.pt`；不要悄悄改用更新的 checkpoint。
- 阈值按关键词独立：`嗨小问=0.303`、`你好问问=0.039`。没有全局 `ctc.trigger_threshold`；缺少 `keywords[].trigger_threshold` 会直接报错。
- Android 直接加载 `kws.onnx`（绝不转 `.ort`）：动态 batch `[N,T,40]`/`[N,256,105]` -> `[N,T,2599]`，输出 posterior（不再二次 softmax）。一个 ORT Session、单 decode worker、每 channel 独立 mutex/frontend/cache/decoder/offset；"多路"指独立音频流。
- score 是 `ctc_prefix_beam_search` 的 `sqrt(∏ token_prob)`（beam 3 / path 20 / `prob>0.05` / `abs_tol=1e-6`）。`token_probs` 和 `frame_topk` 仅用于调试：不得改变 score、阈值、检测结果或 `score_file` 格式。
- `currentEvents()` 保存的是每 channel / 每关键词最近一次在当前 beam 中观察到的命中，不是历史最高分；同一 channel 可以上报多个关键词，Engine 不仲裁。

## 命令
- 重新导出部署资产（会重写 `kws.onnx`、`config.json`、`keywords.txt`、`id_to_token.json`）：
  `/root/miniconda/envs/wekws/bin/python scripts/03_export/export_android_onnx.py --keyword_thresholds '嗨小问=0.303,你好问问=0.039'`
  默认 checkpoint 已固定为 `$WORK/exp/hi_xiaowen/ds_tcn_ctc/avg_30.pt`；`--keyword_thresholds` 必填，不传会失败。旧路径 `android/export/export_ctc_onnx.py` 仅保留薄 wrapper。
- Python 推理：`python scripts/04_eval/infer_cli.py --wav ... --checkpoint avg_30.pt`；默认 checkpoint 是 `auto`（最新 epoch），部署验证必须显式传 `avg_30.pt`。WebUI：`python scripts/04_eval/infer_webui.py --checkpoint avg_30.pt --port 7860`。根目录 `infer_cli.py` / `infer_webui.py` / `infer_core.py` 仍保留兼容 wrapper。
- Native host 校验：`bash scripts/05_deploy/build_android_host.sh`，然后 `bash scripts/05_deploy/compare_scores.sh android/tests/data/ch0.wav`。`run_compare.sh` / `compare_scores.sh` 的 `CHECKPOINT` 默认已是 `work/exp/hi_xiaowen/ds_tcn_ctc/avg_30.pt`，仍可由环境变量覆盖。
- 当前环境没有 Android SDK，`scripts/05_deploy/build_apk.sh` 会明确失败，不会伪装成功。
- 外层入口统一由 `scripts/common/env.sh` 提供路径与 Python 环境（`source scripts/common/env.sh`）。上游 recipe 只有在直接调用其工具时才需在 recipe 目录下 `source path.sh`；训练/评估/导出/推理均走外层 `scripts/` 入口，不再推荐修改上游 `run_ctc.sh`。

## 文档注意
- `android/README.md` 中 `15.pt` / "联调 ONNX" 仅作为历史记录保留；当前部署资产以导出脚本、assets/config.json 和 `tests/` 输出为准。旧 `kws.onnx` 与 `avg_30.pt` posterior 差约 `0.008`，已重导出，当前对齐 score 为 `0.783478 / 0.224239`。
- FRR/FAR 当前复算点（`awk '$2 <= 0.2 {print; exit}'`）为 `嗨小问=0.293`、`你好问问=0.038`；历史点 `0.297 / 0.000` 仅作追溯。部署阈值不变：`0.303 / 0.039`。
