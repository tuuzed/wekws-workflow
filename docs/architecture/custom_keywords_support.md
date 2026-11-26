# 自定义唤醒词支持设计

本文说明当前 Android / Python CTC KWS 推理链路对自定义唤醒词的支持能力、限制和接入方式。

score 公式见：

```text
docs/architecture/ctc_inference_score.md
```

Android 部署架构见：

```text
docs/architecture/android_deployment_architecture.md
```

---

## 1. 结论

**理论上可以支持其他自定义唤醒词。**

原因是模型输出的是 2599 维 token posterior，而 decoder 做的是：

```text
任意 keyword token 序列 -> CTC prefix 中查找连续子序列 -> 命中则计算 score
```

所以 decoder 并没有写死只识别：

```text
嗨小问
你好问问
```

它识别的是 `config.json` 里配置的 token id 序列。

但是否能稳定唤醒，取决于三个条件：

1. 自定义词能被当前 tokenizer / dict 正确转成 token id；
2. 模型能对这些 token 发出足够高的 posterior；
3. beam 搜索参数不会把所需 token 过滤掉。

---

## 2. 当前工程中已经具备的通用能力

### 2.1 关键词来自配置

Android Native 的运行时配置里包含：

```json
"keywords": [
  {
    "text": "嗨小问",
    "chars": ["嗨", "小", "问"],
    "tokens": [1302, 1462, 2494]
  },
  {
    "text": "你好问问",
    "chars": ["你", "好", "问", "问"],
    "tokens": [1027, 1401, 2494, 2494]
  }
]
```

`CtcBeamSearch` 从配置读取关键词列表，不依赖硬编码。

---

### 2.2 token 集合来自关键词配置

当前：

```json
"keywords_token_set": [
  0,
  1027,
  1302,
  1401,
  1462,
  2494
]
```

其中 `0` 是 blank。

每帧 decoder 会过滤：

```text
token 必须在 keywords_token_set 中，或为 blank
```

所以新增自定义词时，必须把新词的 token id 加入 `keywords_token_set`。

---

### 2.3 关键词匹配是通用子序列匹配

Native 和 Python 都使用 `is_sublist` 语义：

```text
prefix 中是否包含 keyword token 序列
```

例如：

```text
prefix:  [..., 1027, 1401, 2494, 2494, ...]
keyword: [1027, 1401, 2494, 2494]
```

命中后取对应 nodes 的 frame / prob 计算 score。

重复 token，例如 `问问 -> [2494, 2494]`，由 CTC prefix beam search 的 blank / repeat 规则处理，不需要额外改 decoder。

---

### 2.4 导出脚本支持 `--keywords`

导出脚本：

```text
android/export/export_ctc_onnx.py
```

支持：

```bash
python export/export_ctc_onnx.py \
  --checkpoint /workspace/wekws-my/wekws/examples/hi_xiaowen/s0/exp/ds_tcn_ctc/15.pt \
  --keywords "嗨小问,你好问问,自定义词"
```

它会：

- 用同一套 `CharTokenizer` 把关键词转成 token id；
- 更新 `app/src/main/assets/config.json`；
- 更新 `app/src/main/assets/keywords.txt`；
- 同时重新导出 `kws.onnx`。

---

## 3. 三种支持级别

| 级别 | 做法 | 是否需要重训模型 | 说明 |
|---|---|---|---|
| 配置级 | 新词由现有字典 token 组成，直接改 `config.json` | 不需要 | 最快，但依赖模型已能识别这些 token |
| 模型级 | 新词发音建模差，补语料微调模型 | 需要 | 适合新增难词、生僻字、英文词 |
| 产品级 | App 运行时动态增删唤醒词 | 不需要重训，但要改 API | 需要新增 Java/Native 更新关键词接口 |

---

## 4. 配置级接入流程

如果只是把自定义词加入固定构建配置，推荐流程如下。

### 4.1 用同一 tokenizer 得到 token id

可以使用和导出脚本一致的 tokenizer：

```python
import sys
from pathlib import Path

sys.path.insert(0, "/workspace/wekws-my/wekws")

from wenet.text.char_tokenizer import CharTokenizer

dict_dir = Path("/workspace/wekws-my/wekws/examples/hi_xiaowen/s0/dict")

tokenizer = CharTokenizer(
    str(dict_dir / "dict.txt"),
    str(dict_dir / "words.txt"),
    unk="<filler>",
    split_with_space=True,
)

keywords = ["嗨小问", "你好问问", "自定义词"]

for keyword in keywords:
    chars, token_ids = tokenizer.tokenize(" ".join(list(keyword)))
    print(keyword, chars, token_ids)
```

输出中的 `token_ids` 就是要写入 `config.json` 的 `tokens`。

如果某个字被转成 `<filler>` 或 unk，说明当前字典 / 模型不适合直接承载该词，需要考虑模型级方案。

---

### 4.2 更新 `config.json`

新增一个关键词对象：

```json
{
  "text": "自定义词",
  "chars": ["自", "定", "义", "词"],
  "tokens": [1111, 2222, 3333, 4444]
}
```

同时更新：

```json
"keywords_token_set": [
  0,
  1027,
  1302,
  1401,
  1462,
  2494,
  1111,
  2222,
  3333,
  4444
]
```

注意：

- `0` 必须保留，表示 blank；
- `keywords_token_set` 应包含所有关键词的所有 token；
- Native decoder 不读取 `keywords.txt`，真正生效的是 `config.json`；
- `keywords.txt` 仅作为人工可读文件。

---

### 4.3 是否需要重新导出 ONNX

如果只是新增 / 修改关键词，且 tokenizer、字典、模型都不变，理论上**不需要重新导出 ONNX**。

但当前导出脚本会把 ONNX、config、keywords 一起重写。因此有两种做法：

1. 手动编辑 `config.json` 和 `keywords.txt`；
2. 运行导出脚本，同时重新生成 ONNX。

后续如果要做产品级自定义词，建议增加一个“只更新 config”的工具，避免每次改词都重新导出模型。

---

## 5. 关键限制

### 5.1 字典和词表限制

模型输出 2599 维，但这 2599 维只覆盖训练字典中的 token。

如果自定义词包含字典外字符，tokenizer 可能产生：

```text
<filler> / <unk>
```

这类 token 不具备明确关键词语义，不能可靠唤醒。

---

### 5.2 模型声学建模限制

输出层有某个 token，不代表模型能稳定识别这个 token 的实际发音。

如果自定义词或其中的字在训练集中很少出现，可能出现：

- posterior 偏低；
- 不同人 / 不同环境下不稳定；
- 误触发或漏触发。

这种情况需要补数据微调或重新训练。

---

### 5.3 beam 过滤限制

当前每帧只取：

```text
score_beam_size = 3
```

并且要求：

```text
prob > 0.05
token 属于 keywords_token_set 或 blank
```

一个 token 要参与解码，必须同时满足：

```text
该 token 在该帧 top-3 里
且 prob > 0.05
且 token 在 keywords_token_set 中
```

因此：

- 关键词越多，top-3 越容易不够；
- 难词的 token prob 可能低于 0.05；
- 长词的中间 token 可能在某帧被过滤掉；
- 多个相似词同时配置时，可能相互竞争 beam。

必要时可以调大：

```json
"score_beam_size": 5
```

但要重新测 FRR / FAR 和性能。

---

### 5.4 阈值不能复用

score 为：

```text
sqrt(∏ token_prob)
```

它与关键词长度和 token 难度有关，不同词之间 score 尺度不可直接比较。

所以新增自定义词后，不能简单复用当前部署词的工作点，例如：

```json
{
  "text": "嗨小问",
  "trigger_threshold": 0.303
}
```

每个关键词都必须在自己的配置对象里显式写 `trigger_threshold`：

```json
{
  "text": "新唤醒词",
  "chars": ["新", "唤", "醒", "词"],
  "tokens": [/* ... */],
  "trigger_threshold": /* 需要单独评测后填写 */
}
```

建议为每个关键词单独确定阈值：

```text
正样本集 -> 统计 score 分布 -> 目标 FRR
负样本集 -> 统计 score 分布 -> 目标 FAR
```

运行时不再有全局 `ctc.trigger_threshold`；新增关键词若没有 `trigger_threshold`，配置解析会直接失败。

---

### 5.5 当前事件策略限制

当前 `currentEvents()` 是每 channel / 每关键词“最近一次当前 beam 命中”语义：

- 返回每个 channel、每个关键词最近一次在当前 beam 中观察到的命中；
- 不保存历史最高分，避免早期高分路径被剪掉后覆盖 Python 离线最终结果；
- 已观察到的关键词暂时离开 beam 时不清空，保证多关键词流水中先前的事件仍可上报；
- 每个关键词使用自己的 `trigger_threshold`；
- 同一 channel 可以同时返回多个关键词事件，Engine 不做仲裁；
- 不维护“触发即上报”的实时事件队列；
- `cooldown_ms` 等字段目前是预留字段，不参与 score 计算。

如果产品上存在多个自定义词，可由使用方自行仲裁：

```text
每词阈值 + 每词 cooldown + 事件去重 + 优先级
```

---

## 6. 产品级动态自定义方案

当前 Java API 不支持运行时添加关键词。现有流程是：

```text
编辑 config.json -> 重新构建 APK -> 生效
```

如果要在 App 内动态添加 / 删除唤醒词，需要新增一条链路：

```text
用户输入词
  -> Java 侧或 Native 侧 tokenize
  -> 生成 keywords / keywords_token_set
  -> 调用 native updateKeywords
  -> 重置每路 decoder 状态
  -> 保留模型 cache 或按策略重置
```

建议接口形态：

```java
engine.updateKeywords(keywordSpecs);
```

Native 侧新增：

```cpp
void KwsEngine::UpdateKeywords(
    const std::vector<Keyword>& keywords,
    const std::vector<int>& keyword_token_set);
```

需要注意：

- 不需要重新加载 ONNX；
- 需要重置每路 `CtcBeamSearch`；
- 是否重置 `cache` 取决于策略，通常建议重置 decoder，不一定重置 cache；
- 每路 `frame_offset` 可以继续保留，重新开始一段唤醒时再 `ResetChannel()`；
- 需要把每词阈值一起传入。

---

## 7. 验证清单

新增自定义词后建议至少验证：

1. Python 侧 `infer_core.py` 使用同样关键词能命中；
2. Python ORT 参考 score 与 PyTorch 一致；
3. Native host CLI score 与 Python 一致；
4. 多路场景下只有目标 channel 触发；
5. 正样本不同说话人 / 距离 / 噪声；
6. 负样本包含相似发音和日常语音；
7. 长词、重复字、相近词分别测试；
8. 根据 FRR / FAR 校准每个词阈值；
9. 如果调大 `score_beam_size`，重新测 CPU / 内存 / 延迟。

---

## 8. 推荐落地顺序

### 第一阶段：固定配置自定义词

- 用 tokenizer 得到 token id；
- 编辑 `config.json`；
- 跑 Python / Native 对齐；
- 校准阈值。

### 第二阶段：构建期工具

- 增加 `generate_keyword_config.py`；
- 只刷新 `config.json` / `keywords.txt`；
- 不重新导出 ONNX。

### 第三阶段：运行时动态自定义

- Java/Native 新增 `updateKeywords`；
- 每词独立阈值；
- 每词 cooldown；
- 事件队列和去重策略；
- App 持久化用户词表。

---

## 9. 一句话总结

当前 decoder 是通用的 token 序列匹配器，不是只能识别两个固定唤醒词。只要新词能被字典 tokenize，且模型能可靠输出对应 token posterior，就可以通过更新 `config.json` 支持；否则需要补数据微调模型。产品级动态自定义词还需要扩展运行时 API 和每词阈值策略。
