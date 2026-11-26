# Android 端多路 KWS-CTC 部署架构设计与实现

工程目录：

```text
/workspace/wekws-workflow/android
```

包名 / applicationId / namespace：

```text
com.haibingtech.kws
```

模型格式：**ONNX**。Native 通过 ONNX Runtime Mobile 直接加载 `kws.onnx`，不转换 `.ort`。

本文描述 Android 端整体部署架构、模块职责、多路隔离策略、线程模型、构建链路和当前验证状态。score 公式细节见：

```text
docs/architecture/ctc_inference_score.md
```

---

## 1. 设计目标

- 支持多路独立音频输入，每路独立检测关键词。
- 一个进程内只创建一个 ORT Session，降低内存和初始化成本。
- 多路通过动态 batch 合并推理，但 frontend / cache / decoder / frame offset 完全隔离。
- Android Native score 与 Python `infer_core.py` / `score_ctc.py` 对齐。
- 模型直接加载 ONNX，不引入 `.ort` 转换步骤。
- Java 层 API 足够薄，主要做 assets 加载、线程调度和事件展示。
- 可选输出命中 token 概率与逐帧 top-k 调试数据，但不改变 score 和触发行为。

---

## 2. 总体架构

```text
Android App / Java
  ├── KwsEngine.java           Java API
  ├── KwsJni.java              JNI 声明
  ├── DetectionEvent.java      事件结构
  ├── TokenProb.java           命中 token 概率
  ├── FrameTokenProb.java      逐帧 top-k token 项
  ├── FrameDebug.java          逐帧 top-k 调试结构
  └── MainActivity.java        麦克风 demo / 线程调度
          │
          │ JNI
          ▼
Native libkws.so
  ├── jni_bridge.cc            JNI handle 管理、事件 JSON
  └── engine/
      ├── KwsEngine            多路引擎、batch ORT Run
      ├── RuntimeConfig        config.json 解析
      ├── FeaturePipeline      每路流式 fbank + 帧队列
      ├── Fbank / Fft           Kaldi 风格 40维 fbank
      ├── CtcBeamSearch        每路 CTC prefix beam search
      ├── json.cc              轻量 JSON 解析
      └── wav_reader.cc        host 验证用 wav 读取
          │
          ▼
ONNX Runtime Mobile
  └── libonnxruntime.so        jniLibs 按 ABI 打包
          │
          ▼
app/src/main/assets/
  ├── kws.onnx                 动态 batch ONNX 模型
  ├── config.json              Native 运行配置
  ├── keywords.txt             关键词 token 可读文件
  └── id_to_token.json         token id 到字符的调试映射
```

数据流：

```text
多路 PCM
  -> 每路独立 FeaturePipeline 生成 40维 fbank
  -> 组装 [N,T,40] + [N,256,105]
  -> 一次 ORT Run
  -> 得到 [N,T,2599] + [N,256,105]
  -> 按 batch 索引拆回每一路
  -> 每路更新自己的 cache / decoder / frame_offset
  -> 每路、每关键词独立当前 beam 命中缓存
```

---

## 3. 关键设计取舍

| 设计点 | 选择 | 原因 |
|---|---|---|
| ORT Session | 全局共享一个 | 避免每个 channel 重复加载模型，减少内存和初始化开销 |
| 推理方式 | 动态 batch 合并 | 多路对齐推理，降低 ORT Run 次数 |
| fbank 位置 | Native C++ | 减少 JNI 数据往返，便于和 Python torchaudio 对齐 |
| decode 线程 | 单 worker | 避免并发调用同一个 ORT Session |
| 每路状态 | 完全独立 | 保证 channel 之间不串 cache/decoder/frame |
| 模型格式 | 直接 ONNX | ORT Mobile 可直接加载，不再转换 `.ort` |
| 事件口径 | 每 channel / 每关键词最近一次当前 beam 命中 | 保证最终 score 与 Python 离线 score 对齐，同时允许同一 channel 上报多个关键词 |

---

## 4. 多路隔离设计

`KwsEngine` 内部为每个 channel 维护一个 `ChannelState`：

```cpp
struct ChannelState {
  mutable std::mutex mutex;
  FeaturePipeline frontend;
  std::unique_ptr<CtcBeamSearch> decoder;
  std::vector<float> cache;
  int frame_offset = 0;
  int64_t last_trigger_ms = -1;
  std::vector<FrameDebug> frame_debug;
};
```

每个 channel 独立持有：

- `FeaturePipeline`：残留样本、帧队列、finished 状态；
- `cache`：TCN 流式 cache，形状 `[256, 105]`；
- `CtcBeamSearch`：CTC prefix beam 当前 prefix / pb / pnb / nodes；
- `frame_offset`：该路全局帧偏移，用于计算事件起止帧；
- `frame_debug`：可选的逐帧 top-k buffer，仅在 `debug.dump_frame_topk=true` 时写入，不参与 score；
- `mutex`：保护该路状态，允许采集线程和 decode 线程并发访问。

共享的内容只有：

- 一个 `Ort::Env`；
- 一个 `Ort::SessionOptions`；
- 一个 `Ort::Session`；
- 一份 `RuntimeConfig`；
- 每路、每关键词最近一次当前 beam 命中缓存。

---

## 5. 流式推理流程

### 5.1 音频输入

Java 采集线程调用：

```java
engine.feed(channel, pcm16, offset, length);
```

JNI 将 `short[]` 拷贝为 `std::vector<int16_t>`，调用：

```cpp
engine->AcceptWaveform(channel, pcm, num_samples);
```

`AcceptWaveform()` 根据 channel 加锁，把 PCM 送入该路 `FeaturePipeline`。

---

### 5.2 流式 fbank

`FeaturePipeline::AcceptWaveform()` 维护 `remained_` 残留样本。

关键实现：

```cpp
const int num_frames =
    1 + (static_cast<int>(waves.size()) - frame_length) / frame_shift;
const int consumed = num_frames * frame_shift;
```

消费样本数使用：

```text
num_frames * frame_shift
```

而不是最后一个窗口的结尾。这样避免 chunk 边界漏帧，是与 Python 连续分帧对齐的关键修正。

fbank 参数：

- sample rate：16000；
- frame length：25ms，即 400 samples；
- frame shift：10ms，即 160 samples；
- FFT points：512；
- mel bins：40；
- low freq：20Hz；
- high freq：8000Hz；
- preemphasis：0.97；
- window：Povey window，`Hann(periodic=False) ^ 0.85`；
- dither：0；
- 每帧 DC removal；
- `log(energy)`，用 float epsilon 做下限。

当前 C++ fbank 与 Python `feats.npy` 验证结果：

```text
shape:    (200, 40) 一致
max diff: 约 2.5e-5
```

---

### 5.3 DecodeOnce 批量推理

`DecodeOnce()` 每次执行一个 batch step：

1. 遍历所有 channel，读取各自 `NumQueuedFrames()`；
2. 取：

   ```cpp
   frames = min(chunk_frames, max_queued_frames)
   ```

   当前 `chunk_frames = 80`，即最多 800ms；
3. 只选择 `queued >= frames` 的 channel 组成 `active_channels`；
4. 从这些 channel 读取 frames，拼成：

   ```text
   feats_flat: [N, frames, 40]
   cache_flat: [N, 256, 105]
   ```

5. 调用一次 ORT Run：

   ```text
   input:  input, cache
   output: output, r_cache
   ```

6. 按 batch 索引把 `r_cache` 和 `output` 写回对应 channel；
7. 每路逐帧调用 `decoder->ProcessFrame(probs, vocab_size, frame)`；
8. 若 `debug.dump_frame_topk=true`，每帧记录 `raw_topk` 和 `kept` 到该路 `frame_debug`；
9. 每路 `frame_offset += frames`。

只把有足够帧的 channel 放进 batch，避免快的 channel 被慢的 channel 阻塞，也保证每路 offset 独立推进。

---

### 5.4 ONNX 动态 batch

导出脚本：

```text
scripts/03_export/export_android_onnx.py
```

`android/export/export_ctc_onnx.py` 仅保留为兼容 wrapper，不作为推荐入口。

动态轴：

```python
{
  "input":   {0: "N", 1: "T"},
  "cache":   {0: "N"},
  "output":  {0: "N", 1: "T"},
  "r_cache": {0: "N"},
}
```

输入 / 输出：

```text
input:   [N, T, 40]
cache:   [N, 256, 105]
output:  [N, T, 2599]
r_cache: [N, 256, 105]
```

其中：

- `N`：本次参与推理的 active channel 数；
- `T`：本次 chunk 的帧数；
- `2599`：词表大小；
- `105`：TCN cache 长度。

`cache_len = 105` 来自当前 TCN 配置：

```text
kernel_size = 8
num_layers = 4
padding = 7 + 14 + 28 + 56 = 105
```

导出脚本同时写入：

```text
app/src/main/assets/kws.onnx
app/src/main/assets/config.json
app/src/main/assets/keywords.txt
app/src/main/assets/id_to_token.json
```

CTC 导出时使用 `model.forward_softmax`，因此 ONNX 输出已经是 posterior，Native 侧不做二次 softmax。

`id_to_token.json` 只用于调试展示时把 token id 显示为字符，不参与推理；缺失时 Java 层会退化为 `<id:n>` 标签。

---

## 6. CTC decoder 设计

Native decoder 文件：

```text
app/src/main/cpp/engine/ctc_beam_search.cc
```

实现严格移植：

```text
wekws/wekws/model/loss.py::ctc_prefix_beam_search
```

关键参数：

```text
score_beam_size = 3
path_beam_size  = 20
prob_threshold  = 0.05
blank_id        = 0
isclose_abs_tol = 1e-6
```

每帧处理：

1. 取 top-3 token；
2. 过滤 `prob <= 0.05` 或非关键词 token；
3. 按 CTC blank / repeat / new token 规则扩展 prefix；
4. 维护 `pb` / `pnb`；
5. 记录每个 token 的 `frame` 和 `prob`；
6. 按 `pb + pnb` 排序，保留 top-20 prefix。

命中关键词后：

```text
score = sqrt(∏ token_prob)
start_frame = 第一个 token 的 frame
end_frame   = 最后一个 token 的 frame
```

详细公式和原因见 `docs/architecture/ctc_inference_score.md`。

Native 保留了 Python `is_sublist` 的边界行为，避免关键词匹配位置不一致。

命中结果中的 `token_probs` 来自 beam 选中的节点，概率是 `ctc_probs[frame][token_id]` 的 posterior，不是严格 CTC 对齐边缘概率；但 `sqrt(∏ token_prob)` 与当前 score 定义完全一致。

---

## 7. 事件策略

当前 `CurrentEvents()` 返回每个 channel、每个关键词各自观察到的 best-hit：

```cpp
for (size_t k = 0; k < config_.keywords.size(); ++k) {
  if (!has_hit_[c][k]) continue;
  const KeywordHit& hit = best_hits_[c][k];
  if (hit.score < config_.keywords[k].trigger_threshold) continue;
  events.push_back(MakeEvent(c, hit));
}
```

特点：

- 每个关键词使用自己的 `keywords[k].trigger_threshold`；
- 同一 channel 可以同时上报多个关键词事件，Engine 不做仲裁，由使用方决定；
- 每个关键词保存的是最近一次在当前 beam 中观察到的命中，不是历史最高分；
- 早期出现但随后被 beam 剪掉的高分路径不会覆盖后续命中，因此最终分数与 Python 离线口径一致；
- 已观察到的关键词即使暂时离开 beam 也不会清除，保证多关键词流水中先前的事件仍可上报；
- 最终精确分数应在 `Finish()` 并 drain 完成后读取；阈值只过滤事件，不改变 score。

JNI 返回 JSON：

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

如果同一 channel 的两个关键词都过各自阈值，数组中会包含两个对象，使用方自行仲裁。

其中：

```text
start_ms = start_frame * frame_shift_ms
end_ms   = end_frame * frame_shift_ms
```

### 7.1 调试输出：token_probs 与逐帧 top-k

- `token_probs`：命中关键词各 token 在 beam 选中帧上的 posterior，满足 `score = sqrt(∏ prob)`；
- `raw_topk`：每帧全词表 top-3，对应 `score_beam_size=3`；
- `kept`：每帧经过 `prob > 0.05` 且属于关键词 token set 后，实际进入 beam search 的 token；
- `id_to_token.json`：只用于把 token id 显示为字符，缺失时显示 `<id:n>`；
- 这些调试字段默认关闭或不影响推理，不改变 score、trigger 阈值与检测结果。

host CLI 导出：

```bash
cd /workspace/wekws-workflow/android
LD_LIBRARY_PATH=build_host ./build_host/kws_host_cli \
  app/src/main/assets/kws.onnx \
  /tmp/kws_config_debug.json \
  tests/data/ch0.wav 1 \
  --debug_json /tmp/android_debug.json
```

其中 `/tmp/kws_config_debug.json` 是把 `config.json` 的 `debug.dump_frame_topk` 改为 `true` 的副本。Java 侧通过 `engine.frameDebug(channel, 200)` 读取，`MainActivity` 的 `Debug` 按钮会打印当前 channel 最多 200 帧。

---

## 8. Java API 设计

主要类：

```text
com.haibingtech.kws.KwsEngine
com.haibingtech.kws.DetectionEvent
com.haibingtech.kws.TokenProb
com.haibingtech.kws.FrameTokenProb
com.haibingtech.kws.FrameDebug
com.haibingtech.kws.KwsJni
com.haibingtech.kws.MainActivity
```

### 8.1 KwsEngine

构造：

```java
KwsEngine engine = new KwsEngine(context, "kws.onnx", "config.json", 0);
```

- `channels = 0` 表示读取 `config.json` 的 `runtime.default_channels`；
- Java 会把 assets 中的模型复制到 `context.getFilesDir()`；
- Native 拿到模型绝对路径和 config JSON 字符串；
- Java 构造时读取 `assets/id_to_token.json`，仅用于 `frameDebug()` 显示 token 字符，缺失时不影响推理。

主要方法：

```java
engine.feed(channel, pcm16, offset, length);
boolean decoded = engine.decodeOnce();
engine.finish(channel);
List<DetectionEvent> events = engine.currentEvents();
List<FrameDebug> frames = engine.frameDebug(channel, 200);
engine.resetChannel(channel);
engine.close();
```

### 8.2 JNI 接口

`KwsJni.java` 声明：

```java
static native long create(String modelPath, String configJson, int channels);
static native void feed(long handle, int channel, short[] pcm, int offset, int length);
static native void finish(long handle, int channel);
static native boolean decodeOnce(long handle);
static native String currentEventsJson(long handle);
static native String frameDebugJson(long handle, int channel, int maxFrames);
static native void resetChannel(long handle, int channel);
static native void destroy(long handle);
```

Native 库：

```java
System.loadLibrary("kws");
```

对应 CMake target：

```text
libkws.so
```

### 8.3 推荐的线程模型

```text
采集线程 1..N: engine.feed(channel_i, ...)
decode 单线程: while (running) engine.decodeOnce()
UI 线程:       读取/展示 currentEvents()
```

`MainActivity` 中已有：

- `kws-record` 采集线程，当前 demo 只采集 spinner 选中的一路；
- `kws-decode` 单 decode 线程；
- UI 通过 `Handler` 展示事件；
- `Debug` 按钮读取当前选中 channel 的逐帧 top-k 并打到日志区。

真实多路采集时，业务层需要为每路独立 PCM 调用 `feed(channel, ...)`。

---

## 9. 运行配置

部署目标 `config.json` 核心内容：

```json
{
  "model_path": "kws.onnx",
  "model_type": "ctc",
  "sample_rate": 16000,
  "feature_dim": 40,
  "frame_length_ms": 25,
  "frame_shift_ms": 10,
  "hidden_dim": 256,
  "cache_len": 105,
  "vocab_size": 2599,
  "blank_id": 0,
  "keywords": [
    {
      "text": "嗨小问",
      "chars": ["嗨", "小", "问"],
      "tokens": [1302, 1462, 2494],
      "trigger_threshold": 0.303
    },
    {
      "text": "你好问问",
      "chars": ["你", "好", "问", "问"],
      "tokens": [1027, 1401, 2494, 2494],
      "trigger_threshold": 0.039
    }
  ],
  "ctc": {
    "score_beam_size": 3,
    "path_beam_size": 20,
    "prob_threshold": 0.05,
    "isclose_abs_tol": 1e-06,
    "cooldown_ms": 1000,
    "reset_decoder_after_trigger": true,
    "reset_cache_after_trigger": false
  },
  "runtime": {
    "default_channels": 2,
    "chunk_frames": 80,
    "intra_op_threads": 1,
    "inter_op_threads": 1
  },
  "debug": {
    "dump_frame_topk": false,
    "frame_topk_size": 3,
    "max_debug_frames": 2000
  }
}
```

说明：

- `keywords[].trigger_threshold` 是必填字段，缺失时配置解析失败；
- 每个关键词使用自己的阈值，只控制是否上报，不改变 score；
- `cooldown_ms` / `reset_decoder_after_trigger` / `reset_cache_after_trigger` 是当前 best-hit 口径下的预留字段，不参与 score 计算；
- `intra_op_threads` / `inter_op_threads` 当前设为 1，避免移动端多线程调度带来的延迟抖动；
- `debug.dump_frame_topk` 默认关闭，开启后才会填充每路 `frame_debug`，不影响 score 与事件输出。

独立阈值 `嗨小问=0.303`、`你好问问=0.039` 来自 `docs/evaluation/avg30_frr_far_evaluation.md` 的 `FAR ≤ 0.2/h` 保守工作点。`export_ctc_onnx.py` 不再写全局阈值，导出时必须通过 `--keyword_thresholds` 显式传入每个关键词的阈值。

---

## 10. 构建与部署

### 10.1 Gradle 配置

```text
compileSdk 34
minSdk 24
targetSdk 34
Java 8
C++14
CMake 3.22.1
```

`jniLibs` 路径：

```groovy
jniLibs.srcDirs = ["../third_party/onnxruntime/lib", "src/main/jniLibs"]
```

CMake 中 ORT 路径：

```cmake
set(ORT_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../../../../third_party/onnxruntime")
```

链接：

```cmake
target_link_libraries(kws onnxruntime android log)
```

### 10.2 ORT Mobile

下载脚本：

```bash
bash export/download_ort_mobile.sh
```

默认版本：

```text
ORT_VERSION=1.20.0
```

下载 Maven AAR 后解压：

```text
third_party/onnxruntime/include/
third_party/onnxruntime/lib/arm64-v8a/libonnxruntime.so
third_party/onnxruntime/lib/armeabi-v7a/libonnxruntime.so
third_party/onnxruntime/lib/x86/libonnxruntime.so
third_party/onnxruntime/lib/x86_64/libonnxruntime.so
```

### 10.3 APK 构建

需要 Android SDK：

```bash
cd /workspace/wekws-workflow/android
./gradlew assembleDebug
```

安装：

```bash
adb install -r app/build/outputs/apk/debug/app-debug.apk
```

当前工作区没有 Android SDK，因此尚未跑通完整 APK 构建；已完成 Native host 构建、JNI 对象编译检查和 Java stub `javac` 语法检查。

---

## 11. Host 验证链路

构建：

```bash
cd /workspace/wekws-workflow/android
bash tests/build_host.sh
```

对齐验证：

```bash
bash tests/run_compare.sh tests/data/ch0.wav
bash tests/run_compare.sh tests/data/ch_nihao.wav
bash tests/run_compare.sh tests/data/ch1.wav
bash tests/run_compare.sh tests/data/stereo.wav
```

历史 `15.pt` 联调模型结果：

```text
ch0.wav       嗨小问   score=0.723124  start=69  end=129
ch_nihao.wav  你好问问 score=0.927641  start=63  end=143
ch1.wav       无事件
stereo.wav    channel 0 命中，channel 1 无事件
```

当前 `assets/kws.onnx` 已用 `avg_30.pt` 重新导出，观察到的 Native streaming 结果为：

```text
ch0.wav       嗨小问   score=0.783478  start=69  end=129
              token_probs = [嗨=0.986322@69, 小=0.997568@109, 问=0.623867@129]
ch_nihao.wav  你好问问 score=0.224239  start=63  end=125
              token_probs = [你=0.953913@63, 好=0.989329@85, 问=0.994196@108, 问=0.053592@125]
ch1.wav       无事件
stereo.wav    channel 0 命中，channel 1 无事件
```

验证覆盖：

- Python `infer_core.py`
- Python ONNX 参考
- Native CTC decoder CLI
- Native streaming engine
- 事件 `token_probs`
- 逐帧 top-k debug JSON

`stereo.wav` 证明多路的 batch/cache/decoder/offset 没有串路。逐帧 top-k 的 token id 与 ORT 参考一致，prob 最大差约 `6.24e-6`。重导出后已重跑本节全部对齐验证：Python `0.783476 / 0.224240`、ONNX 参考与 Native decoder `0.783478 / 0.224239`、Native streaming `0.783478 / 0.224239`。早期 `0.787617 / 0.223722` 对应旧 ONNX 资产，posterior 与 `avg_30.pt` 最大差约 `0.008`，不再作为当前工作点。

---

## 12. 当前限制与后续工作

### 当前限制

1. assets 中的 `kws.onnx` 已由 `avg_30.pt` 导出，PyTorch / ONNX posterior 最大差约 `6.44e-6`。
2. FRR / FAR 评测已完成，按 `FAR ≤ 0.2/h` 选定独立工作点 `嗨小问=0.303`、`你好问问=0.039`，见 `docs/evaluation/avg30_frr_far_evaluation.md`；训练进程仍在继续，后续新 checkpoint 不自动覆盖部署模型。
3. 当前环境没有 Android SDK，未执行完整 Gradle APK 构建。
4. `MainActivity` demo 只采集一路麦克风；多路能力已实现在 Native/Java API，但真实多路采集需要业务侧接入。
5. `currentEvents()` 是每 channel / 每关键词“最近一次当前 beam 命中”语义，不是“越阈值立即触发”的事件队列；同一 channel 可返回多个关键词事件。
6. `Finish()` 标记输入结束，当前没有额外尾部静音补帧逻辑。

### 后续工作

1. 已用 `avg_30.pt` 重新导出 Android ONNX：

   ```bash
   cd /workspace/wekws-workflow
   source scripts/common/env.sh
   "$PYTHON_BIN" scripts/03_export/export_android_onnx.py \
     --checkpoint "$DEPLOY_CHECKPOINT" \
     --keyword_thresholds '嗨小问=0.303,你好问问=0.039'
   ```

2. 导出脚本会按 `--keyword_thresholds` 写入每个关键词的 `trigger_threshold`，不再需要导出后手工 patch。
3. Python / ONNX / Native 三方 score、`token_probs` 和逐帧 top-k 对齐已重跑；`--score_file` 格式未变。
4. 在装有 Android SDK 的环境执行 `./gradlew assembleDebug`，修复可能的 Java / CMake / packaging 问题。
5. 如需实时触发队列，单独设计触发时刻的 score 口径，不要直接复用当前命中缓存的最终 score。

---

## 13. 一句话总结

Android 端采用“**一个共享 ORT Session + 多路独立 frontend/cache/decoder/offset + 动态 batch 推理 + 单 decode worker**”的架构，在保证多路隔离和移动端资源可控的同时，score 与 Python 离线口径保持一致。
