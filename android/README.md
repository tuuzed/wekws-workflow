# Android 多路 KWS-CTC 工程

工程目录：`/workspace/wekws-workflow/android`

包名 / applicationId / namespace：

```text
com.haibingtech.kws
```

模型格式：**ONNX**，Native 直接加载 `app/src/main/assets/kws.onnx`，不转换 `.ort`。

## 架构

```text
多路 PCM -> 每路独立 fbank/cache/decoder/frame_offset
         -> 一次 ORT Run: [C,T,40] + [C,256,105] -> [C,T,2599] + [C,256,105]
         -> 每路 CTC prefix beam search -> 事件 JSON
```

- 一个 `Ort::Session` 被所有 channel 共享。
- 每个 channel 有独立的 `FeaturePipeline`、`cache`、`CtcBeamSearch` 和 frame offset。
- C++ 内部按 channel 加 mutex；Java 侧推荐单 decode worker。
- score 严格移植 `wekws/model/loss.py::ctc_prefix_beam_search`，保留 Python `is_sublist` 边界行为。
- fbank 对齐 `torchaudio.compliance.kaldi.fbank` 默认参数：Povey window、25ms/10ms、40 mel bins、dither=0。

关键词：

```text
嗨小问   = [1302, 1462, 2494]
你好问问 = [1027, 1401, 2494, 2494]
blank    = 0
```

## 目录说明

```text
export/export_ctc_onnx.py          动态 batch ONNX 导出
export/download_ort_mobile.sh      下载 ORT Mobile headers/.so
app/src/main/assets/kws.onnx       由 avg_30.pt 导出的动态 batch ONNX
app/src/main/assets/config.json    Native 运行时配置
app/src/main/assets/keywords.txt   关键词 token 文件
app/src/main/assets/id_to_token.json  token id 到字符的调试映射
app/src/main/cpp/engine/           Native 推理、frontend、CTC decoder、JNI
app/src/main/java/com/haibingtech/kws/  Java API 与示例 UI
tests/onnx_reference.py            Python/ORT 参考分数
tests/build_host.sh                Linux host native 构建
tests/run_compare.sh               Python 与 Native 对齐验证
```

## 1. 导出 ONNX

当前 assets 中已有用于联调的 `kws.onnx`。训练完成并生成 `avg_30.pt` 后重新导出：

```bash
cd /workspace/wekws-workflow/android
/root/miniconda/envs/wekws/bin/python export/export_ctc_onnx.py \
  --checkpoint /workspace/wekws-workflow/wekws/examples/hi_xiaowen/s0/exp/ds_tcn_ctc/avg_30.pt
```

导出脚本会同时刷新：

```text
app/src/main/assets/kws.onnx
app/src/main/assets/config.json
app/src/main/assets/keywords.txt
app/src/main/assets/id_to_token.json
```

动态轴：

```python
{
  "input":   {0: "N", 1: "T"},
  "cache":   {0: "N"},
  "output":  {0: "N", 1: "T"},
  "r_cache": {0: "N"},
}
```

## 2. 下载 ORT Mobile

```bash
cd /workspace/wekws-workflow/android
bash export/download_ort_mobile.sh
```

产物：

```text
third_party/onnxruntime/include/
third_party/onnxruntime/lib/arm64-v8a/libonnxruntime.so
third_party/onnxruntime/lib/armeabi-v7a/libonnxruntime.so
third_party/onnxruntime/lib/x86/libonnxruntime.so
third_party/onnxruntime/lib/x86_64/libonnxruntime.so
```

默认版本是 `ORT_VERSION=1.20.0`，可通过环境变量覆盖。

## 3. Host 对齐验证

```bash
cd /workspace/wekws-workflow
bash scripts/05_deploy/build_android_host.sh
bash scripts/05_deploy/compare_scores.sh android/tests/data/ch0.wav
bash scripts/05_deploy/compare_scores.sh android/tests/data/ch_nihao.wav
bash scripts/05_deploy/compare_scores.sh android/tests/data/stereo.wav
```

`compare_scores.sh` 默认使用 `work/exp/hi_xiaowen/ds_tcn_ctc/avg_30.pt`，可用 `CHECKPOINT` 环境变量覆盖。

当前部署资产的对齐记录（`avg_30.pt`，重导出 ONNX 后复测）：

```text
ch0.wav       检测到 嗨小问   score=0.783478
ch_nihao.wav  检测到 你好问问 score=0.224239
ch1.wav       无事件
stereo.wav    channel 0 命中，channel 1 无事件
```

Python `infer_core.py` 为 `0.783476 / 0.224240`，ONNX 参考与 Native decoder/streaming 为 `0.783478 / 0.224239`，差异在 `2e-6` 内，3 位小数 `score.txt` 均为 `0.783 / 0.224`。每帧 top-k 的 token id 与 ORT 参考一致，prob 最大差约 `6.2e-6`。

历史值 `0.787617 / 0.223722` 来自旧 ONNX 资产；其与 `avg_30.pt` 的 posterior 最大差约 `0.008`，已被重新导出修复，不再作为当前对齐基线。

历史记录：早期 `15.pt` 联调模型的 score 为 `0.723124` / `0.927641`，已不再是部署工作点，仅作追溯。

## 4. Android Gradle 构建

需要本机配置 Android SDK（当前工作区未安装 Android SDK，因此这里只验证了 Native host 构建和 Python 对齐）：

```bash
cd /workspace/wekws-workflow/android
./gradlew assembleDebug
```

安装：

```bash
adb install -r app/build/outputs/apk/debug/app-debug.apk
```

## Java 调用方式

```java
KwsEngine engine = new KwsEngine(context, "kws.onnx", "config.json", 0);
// 0 表示读取 config.json 的 runtime.default_channels，也可显式传 2。

// 采集线程：每路独立 PCM
engine.feed(channel, pcm16, 0, n);

// 单 decode 线程：持续推理，不在这里读取事件
while (running) {
    if (!engine.decodeOnce()) {
        Thread.sleep(5);
    }
}
```

结束时：

```java
for (int c = 0; c < engine.channels(); ++c) engine.finish(c);
while (engine.decodeOnce()) {
    // 排空剩余帧
}
for (DetectionEvent event : engine.currentEvents()) {
    // 这里拿到的是每个关键词最近一次当前 beam 命中，score 与 Python 离线口径一致
}
engine.close();
```

注意：当前 `currentEvents()` 返回每个 channel、每个关键词最近一次在当前 beam 中观察到的命中，同一 channel 可能有多个事件；每个关键词保存最近一次命中而不是历史最高分，早期高分路径被 beam 剪掉后不会覆盖最终结果。示例 UI 在 `Finish()` 后输出最终精确分数。如果需要“越阈值立即触发”的实时事件队列，需要单独定义触发时的 score 口径；实时分数可能与离线最终分数不同。

## 阈值说明

阈值配置在每个关键词对象里，`keywords[].trigger_threshold` 是必填字段：

```json
{
  "text": "嗨小问",
  "tokens": [1302, 1462, 2494],
  "trigger_threshold": 0.303
}
{
  "text": "你好问问",
  "tokens": [1027, 1401, 2494, 2494],
  "trigger_threshold": 0.039
}
```

`trigger_threshold` 只控制该关键词是否上报，不改变 score。同一 channel 可以同时上报多个关键词事件，使用方自行仲裁。当前工作点来自 `docs/evaluation/avg30_frr_far_evaluation.md`。

`cooldown_ms` / `reset_decoder_after_trigger` / `reset_cache_after_trigger` 是为后续实时触发队列预留的字段，当前命中缓存口径下不参与 score 计算。

## 事件 JSON

JNI `currentEventsJson()` 返回：

```json
[
  {
    "channel": 0,
    "keyword": "嗨小问",
    "score": 0.792815,
    "start_ms": 690,
    "end_ms": 1290,
    "token_probs": [
      {"token": "嗨", "token_id": 1302, "prob": 0.986345, "frame": 69},
      {"token": "小", "token_id": 1462, "prob": 0.997565, "frame": 109},
      {"token": "问", "token_id": 2494, "prob": 0.638812, "frame": 129}
    ]
  }
]
```

同一 channel 同时命中多个关键词时，数组中会有多个对象，由使用方自行仲裁。

`token_probs` 是命中关键词各 token 在对应帧上的 posterior，满足：

```text
score = sqrt(prod(token_probs[i].prob))
```

### 每帧 top-k 调试输出

`config.json` 中默认关闭：

```json
"debug": {
  "dump_frame_topk": false,
  "frame_topk_size": 3,
  "max_debug_frames": 2000
}
```

打开后 native 会为每路保存 `raw_topk`（全词表 top-3）和 `kept`（通过 `prob > 0.05` 且属于关键词 token set 的 token），用于确认某个字为什么被 beam search 保留或丢弃。

host CLI 导出 JSON：

```bash
cd /workspace/wekws-workflow/android
LD_LIBRARY_PATH=build_host ./build_host/kws_host_cli \
  app/src/main/assets/kws.onnx \
  /tmp/kws_config_debug.json \
  tests/data/ch0.wav 1 \
  --debug_json /tmp/android_debug.json
```

其中 `/tmp/kws_config_debug.json` 是把 `config.json` 的 `debug.dump_frame_topk` 改为 `true` 的副本。

Java API：

```java
for (FrameDebug frame : engine.frameDebug(channel, 200)) {
    // frame.rawTopk / frame.kept
}
```

示例 UI 中增加了 `Debug` 按钮，会把当前选中 channel 的最多 200 帧 debug 信息打到日志区：先输出 `frame debug channel=<id> frames=<n>`，随后每帧一行 `frame=<帧号> ms=<时间> raw=[...] kept=[...]`。

按钮只读取已有的 `frame_debug` buffer，与录音状态无关（Engine 初始化失败时才禁用），输出是追加到日志区而非清空日志；它不改变 score、阈值或检测结果。`resetChannel()` 会清空该 channel 的 `frame_debug`，因此重置后（或 `dump_frame_topk=false` 时）点击按钮会得到 `frames=0`。

## 注意事项

- 导出脚本默认 checkpoint 已是 `work/exp/hi_xiaowen/ds_tcn_ctc/avg_30.pt`，阈值由 `--keyword_thresholds` 显式传入（`嗨小问=0.303,你好问问=0.039`）；历史 `15.pt` 联调模型不再作为部署工作点。
- 如果需要修 `is_sublist` 边界行为，必须同时修改 Python `score_ctc.py` 和 Native `ctc_beam_search.cc`。
- `build_host/`、`tests/out/`、`third_party/onnxruntime/` 是本地产物目录，不需要提交。
