# CTC 推理 Score 的计算方式

本文说明 `hi_xiaowen/s0` CTC KWS 推理结果中 `score` 的来源，适用于：

- Python CLI / WebUI：`infer_core.py`
- stage 3 评估：`wekws/wekws/bin/score_ctc.py`
- Android Native：`android/app/src/main/cpp/engine/ctc_beam_search.cc`

当前 Android 工程中的 score 口径与 Python 严格对齐，Python、ONNX 参考结果和 Native host 结果可以到 6 位小数一致。

---

## 1. 结论

推理 score 不是模型直接输出的一个数，而是：

> CTC prefix beam search 命中关键词后，把关键词每个 token 在对应帧上的 posterior 概率连乘，再开平方。

设关键词 token 序列为：

```text
w0, w1, ..., w(L-1)
```

beam 命中时，每个 token 对应的帧和概率为：

```text
node_i = { token: wi, frame: ti, prob: pi }
```

其中：

```text
pi = P(wi | x_ti)
```

即模型在第 `ti` 帧输出 token `wi` 的 softmax 后验概率。

最终 score 为：

```text
score = sqrt(p0 * p1 * ... * p(L-1))
```

也可以写成数值稳定形式：

```text
score = exp(0.5 * sum(log(pi)))
```

注意：这里没有除以关键词长度，也没有使用平均概率。这是 `wekws` 原有的 score 定义，Android Native 严格照搬。

---

## 2. 模型输出概率从哪里来

模型输入 fbank 特征后输出 logits：

```text
[T, 2599]
```

其中：

- `T`：帧数；
- `2599`：词表大小；
- `0`：blank token。

Python 侧会显式做 softmax：

```python
logits, _ = model(feats)
probs = logits.softmax(2).cpu()
```

Android 导出的 `kws.onnx` 已经把 softmax 包含在图里，因此 Native 侧直接拿到每帧 posterior，不再二次 softmax。

于是每一帧得到：

```text
p_t(k) = 第 t 帧 token k 的概率
```

---

## 3. CTC prefix beam search 做什么

核心实现：

```text
wekws/wekws/model/loss.py::ctc_prefix_beam_search
```

Android 对应实现：

```text
android/app/src/main/cpp/engine/ctc_beam_search.cc
```

beam search 为每个 prefix 维护两个概率：

```text
pb  ：路径以 blank 结尾的概率
pnb ：路径不以 blank 结尾的概率
```

每一帧执行：

1. 取当前帧 top-k token：

   ```text
   score_beam_size = 3
   ```

2. 过滤低概率和非关键词 token：

   ```text
   prob > 0.05
   token 属于 keywords_token_set 或 blank
   ```

3. 按 CTC 规则扩展 prefix：

   - `blank`：prefix 不变；
   - 重复 token：处理 `*s -> *ss`；
   - 新 token：prefix 追加该 token。

4. 每帧按 `pb + pnb` 排序，保留：

   ```text
   path_beam_size = 20
   ```

   条 prefix。

部署目标 `config.json` 中相关参数：

```json
{
  "blank_id": 0,
  "keywords": [
    {
      "text": "嗨小问",
      "tokens": [1302, 1462, 2494],
      "trigger_threshold": 0.303
    },
    {
      "text": "你好问问",
      "tokens": [1027, 1401, 2494, 2494],
      "trigger_threshold": 0.039
    }
  ],
  "ctc": {
    "score_beam_size": 3,
    "path_beam_size": 20,
    "prob_threshold": 0.05,
    "isclose_abs_tol": 1e-06
  }
}
```

---

## 4. beam node 中记录了什么

除了 prefix 和路径概率，每条 beam 还会记录 nodes：

```text
node = {
  token: token_id,
  frame: 该 token 被加入 prefix 的帧号,
  prob: 该帧上该 token 的 posterior
}
```

例如 prefix 是：

```text
[1302, 1462, 2494]
```

它可能对应：

```text
token 1302 -> frame 69,  prob p0
token 1462 -> frame 101, prob p1
token 2494 -> frame 129, prob p2
```

最终 score 和起止帧都来自这些 nodes。

对于重复 token，例如 `2494, 2494`，实现会保留对应帧上更大的 posterior 和 frame。Python 和 Native 两边的这一行为保持一致。

---

## 5. 关键词命中与 score 计算

搜索结束后，按 beam 顺序遍历 prefix，检查 prefix 中是否包含关键词 token 序列。

当前关键词：

```text
嗨小问   = [1302, 1462, 2494]
你好问问 = [1027, 1401, 2494, 2494]
blank    = 0
```

伪代码：

```python
probs = softmax(model(feats))

hyps = ctc_prefix_beam_search(
    probs,
    keywords_tokenset,
    score_beam_size=3,
    path_beam_size=20,
)

for prefix, path_score, nodes in hyps:
    for keyword in keywords:
        offset = is_sublist(prefix, keyword.tokens)
        if offset != -1:
            score = 1.0
            for i in range(len(keyword.tokens)):
                score *= nodes[offset + i]["prob"]

            score = math.sqrt(score)

            start_frame = nodes[offset]["frame"]
            end_frame = nodes[offset + len(keyword.tokens) - 1]["frame"]
            break
```

关键词匹配使用连续子序列判断 `is_sublist`。Native 实现保留了 Python 原实现中的边界行为，避免 score 与起止帧发生偏移。

---

## 6. 当前验证样例

### 6.1 `ch0.wav`

```text
关键词：嗨小问
token： [1302, 1462, 2494]
score：0.792632
start_frame：69
end_frame：129
start_ms：690
end_ms：1290
```

对应公式：

```text
score = sqrt(0.986359 * 0.997566 * 0.638508)
      ≈ 0.792632
```

### 6.2 `ch_nihao.wav`

```text
关键词：你好问问
token： [1027, 1401, 2494, 2494]
score：0.225955
start_frame：63
end_frame：125
start_ms：630
end_ms：1250
```

对应公式：

```text
score = sqrt(0.954417 * 0.988056 * 0.994529 * 0.054439)
      ≈ 0.225955
```

注意第二个 `问` 的 posterior 较低，`avg_30.pt` 下该样本分数明显低于旧联调模型。

### 6.3 负样本

```text
ch1.wav：无关键词命中
stereo.wav：只有 channel 0 命中，channel 1 无事件
```

说明多路独立 frontend / cache / decoder / frame offset 没有串路。

---

## 7. score 与 beam 路径概率不是同一个东西

`ctc_prefix_beam_search` 返回的每条 beam 里有：

```text
prefix
path_score = pb + pnb
nodes
```

但最终 score **不使用** `path_score`。

区别是：

| 项目 | 用途 | 含义 |
|---|---|---|
| `pb + pnb` | beam 排序 | 整条 CTC prefix 路径的概率 |
| `score` | 关键词置信度 | 关键词各 token 对应帧 posterior 的连乘开根 |

也就是说：

- beam search 用 `pb + pnb` 找到可能包含关键词的路径；
- 最终 score 只看关键词 token 本身在对应帧上的 posterior。

---

## 8. `trigger_threshold` 不改变 score

Android `config.json` 中，每个关键词有自己的阈值：

```json
{
  "text": "嗨小问",
  "trigger_threshold": 0.303
}
```

它只控制该关键词是否上报事件：

```text
score >= keywords[k].trigger_threshold -> 允许上报
score <  keywords[k].trigger_threshold -> 不上报
```

输出事件中的 score 仍然是上述公式计算出的原始分数，不会被 threshold 修改。

当前 `currentEvents()` 返回每个 channel、每个关键词最近一次在当前 beam 中观察到的命中，同一 channel 可以有多个事件；它不保存历史最高分，因此与 Python 离线最终 beam 口径一致。当前部署工作点按 `FAR ≤ 0.2/h` 选择为 `嗨小问=0.303`、`你好问问=0.039`，依据见 `docs/evaluation/avg30_frr_far_evaluation.md`；新增关键词时必须重新跑 FRR / FAR 工作点，不要直接复用。

---

## 9. 代码位置

Python 推理：

```text
/workspace/wekws-workflow/infer_core.py
```

关键逻辑：

```python
logits = logits.softmax(2).cpu()
hyps = ctc_prefix_beam_search(score, lengths[i].cpu(), self.keywords_idxset)
hit_score *= prefix_nodes[idx]["prob"]
hit_score = math.sqrt(hit_score)
```

stage 3 评估：

```text
/workspace/wekws-workflow/wekws/wekws/bin/score_ctc.py
```

CTC beam search：

```text
/workspace/wekws-workflow/wekws/wekws/model/loss.py
```

Android Native：

```text
/workspace/wekws-workflow/android/app/src/main/cpp/engine/ctc_beam_search.cc
```

关键函数：

```text
ProcessFrame()
FindBestHit()
```

---

## 10. 输出每个 token 的 prob 与每帧 top-k

### 10.1 Python CLI / WebUI

命中关键词后，结果里会带 `token_probs`：

```json
{
  "keyword": "嗨小问",
  "score": 0.792632,
  "token_probs": [
    {"token": "嗨", "token_id": 1302, "prob": 0.986359, "frame": 69, "time_ms": 690},
    {"token": "小", "token_id": 1462, "prob": 0.997566, "frame": 109, "time_ms": 1090},
    {"token": "问", "token_id": 2494, "prob": 0.638508, "frame": 129, "time_ms": 1290}
  ]
}
```

满足：

```text
score = sqrt(prod(token_probs[i].prob))
```

CLI 用法：

```bash
cd /workspace/wekws-workflow
/root/miniconda/envs/wekws/bin/python infer_cli.py \
  --wav android/tests/data/ch0.wav \
  --checkpoint avg_30.pt \
  --detail_file /tmp/detail.jsonl
```

如果需要每帧 top-k：

```bash
/root/miniconda/envs/wekws/bin/python infer_cli.py \
  --wav android/tests/data/ch0.wav \
  --checkpoint avg_30.pt \
  --detail_file /tmp/detail_with_topk.jsonl \
  --dump_frame_topk
```

WebUI 中勾选「输出每帧 top-k（调试）」即可；完整 JSON 在日志框，结果框只展示前 100 帧避免刷屏。

每帧 top-k 字段：

```json
{
  "frame": 12,
  "time_ms": 120,
  "raw_topk": [
    {"token_id": 0, "token": "<blk>", "prob": 0.91},
    {"token_id": 1302, "token": "嗨", "prob": 0.07},
    {"token_id": 2494, "token": "问", "prob": 0.02}
  ],
  "kept": [
    {"token_id": 1302, "token": "嗨", "prob": 0.07}
  ]
}
```

- `raw_topk`：全词表 top-3，对应 `score_beam_size=3`。
- `kept`：经过 `prob > 0.05` 且属于关键词 token set 后，实际进入 beam search 的 token。

### 10.2 Android Native / Java

事件 JSON 增加 `token_probs`：

```json
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
```

每帧 top-k 是可选的调试 buffer，默认关闭：

```json
"debug": {
  "dump_frame_topk": false,
  "frame_topk_size": 3,
  "max_debug_frames": 2000
}
```

host CLI 导出：

```bash
cd /workspace/wekws-workflow/android
LD_LIBRARY_PATH=build_host ./build_host/kws_host_cli \
  app/src/main/assets/kws.onnx \
  app/src/main/assets/config.json \
  tests/data/ch0.wav 1 \
  --debug_json /tmp/android_debug.json
```

Java 侧：

```java
for (DetectionEvent event : engine.currentEvents()) {
    // event.tokenProbs
}

for (FrameDebug frame : engine.frameDebug(channel, 200)) {
    // frame.rawTopk / frame.kept
}
```

`id_to_token.json` 由导出脚本生成，只用于把 token id 显示成字符；缺失时会退化成 `<id:123>`，不影响推理和 score。

### 10.3 语义提醒

这里的 token prob 是 beam 选中帧上的 posterior：

```text
p = ctc_probs[frame][token_id]
```

它不是完整 CTC 对齐上的严格边缘概率。要得到严格边缘概率需要改变 decoder 的路径累加方式，也会改变当前 score 定义，因此没有采用。

---

## 11. 一句话总结

```text
score = sqrt(关键词每个 token 在其命中帧上的 posterior 概率连乘)
```

beam search 负责找到关键词 token 序列以及每个 token 对应的帧；score 只对这些 token 的后验概率做连乘开根，不使用整条 CTC 路径概率。
