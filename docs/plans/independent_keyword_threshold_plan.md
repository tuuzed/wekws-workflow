# 独立关键词阈值方案实施计划

> 状态：已完成  
> 日期：2026-11-26  
> 依据评测：`docs/evaluation/avg30_frr_far_evaluation.md`

---

## 1. 背景

当前 Android Native 运行时只有一个全局 `ctc.trigger_threshold`，同一 channel 只保留一个 best hit，并用同一个阈值决定是否上报。

`avg_30.pt` 的 FRR/FAR 评测表明：

- `嗨小问` 和 `你好问问` 的 score 尺度不同；
- 共享阈值会为了压低 `嗨小问` 的误唤醒而显著抬高 `你好问问` 的漏唤醒；
- 在 `FAR ≤ 0.2/h` 约束下，独立阈值明显优于共享阈值。

因此改为每个关键词独立阈值，并允许同一 channel 同时输出多个关键词事件，由使用方仲裁。

---

## 2. 已确认决策

| 编号 | 决策 | 说明 |
|---:|---|---|
| 1 | 同一 channel 允许多个事件 | `CurrentEvents()` 返回所有过各自阈值的关键词事件；Engine 不做仲裁 |
| 2 | 每关键词独立阈值 | 阈值配置在 `keywords[].trigger_threshold` |
| 3 | 不做旧配置兼容 | 删除 `ctc.trigger_threshold`；缺少 per-keyword 阈值时解析报错 |
| 4 | 保留 FAR 余量 | 不取 `awk` 首个满足值，取更保守工作点 |
| 5 | score 公式不变 | 阈值只过滤事件，不改变 score、token_probs、frame_topk 和 score_file |

---

## 3. 最终工作点

| 关键词 | 阈值 | FAR（次/小时） | FRR |
|---|---:|---:|---:|
| 嗨小问 | 0.303 | 0.174880 | 4.03% |
| 你好问问 | 0.039 | 0.175212 | 3.31% |

两词 FAR 均约 `0.175/h`，相对 `0.2/h` 上限保留约 `0.025/h` 余量。

对比共享阈值 `0.297`：

| 指标 | 共享阈值 0.297 | 独立阈值 | 变化 |
|---|---:|---:|---:|
| 嗨小问 FRR | 3.96% | 4.03% | +0.07pp |
| 你好问问 FRR | 12.85% | 3.31% | -9.54pp |
| 你好问问 FAR | 0.0438/h | 0.1752/h | 上升，但仍满足约束 |

---

## 4. 目标配置格式

删除 `ctc.trigger_threshold`，改为：

```json
{
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
  ]
}
```

`keywords[].trigger_threshold` 是必填字段；缺失时配置解析直接失败，不使用默认值。

---

## 5. 改动清单

### 5.1 Native 配置

- `android/app/src/main/cpp/engine/runtime_config.h`
  - `Keyword` 增加 `trigger_threshold`
  - `CtcConfig` 删除 `trigger_threshold`
- `android/app/src/main/cpp/engine/runtime_config.cc`
  - 删除 `ctc.trigger_threshold` 解析
  - 强制读取 `keywords[].trigger_threshold`
  - 缺失时抛出包含关键词文本的异常

### 5.2 Decoder / Engine

- `android/app/src/main/cpp/engine/ctc_beam_search.h` / `.cc`
  - 新增按关键词返回各自最佳 hit 的接口
  - 每个关键词独立取最高分 hit
- `android/app/src/main/cpp/engine/kws_engine.h` / `.cc`
  - best hit 从每 channel 一组改为每 channel、每关键词一组
  - `CurrentEvents()` 对每个关键词使用自己的阈值
  - 同一 channel 输出多个过阈值事件
  - `ResetChannel()` 清空所有关键词槽位

### 5.3 JNI / Java

- `jni_bridge.cc` 继续输出事件数组，无需改结构
- `MainActivity` 已按 `channel/keyword/start/end` de-dupe，无需改

### 5.4 导出脚本

- `android/export/export_ctc_onnx.py`
  - 删除 `ctc.trigger_threshold = 0.5` 硬编码
  - 新增 `--keyword_thresholds '嗨小问=0.303,你好问问=0.039'`
  - 未传或关键词不匹配时失败，不再导出后手工 patch

### 5.5 配置资产与文档

- `android/app/src/main/assets/config.json`
- `android/README.md`
- `docs/evaluation/avg30_frr_far_evaluation.md`
- `docs/architecture/android_deployment_architecture.md`
- `docs/architecture/ctc_inference_score.md`
- `docs/architecture/custom_keywords_support.md`
- `docs/plans/scripts_migration_and_upstream_readonly_plan.md`

---

## 6. 验证项

1. `android/tests/build_host.sh` 通过。
2. `ch0.wav`：输出 `嗨小问`，score 与迁移前一致。
3. `ch_nihao.wav`：独立阈值下输出 `你好问问`。
4. `ch1.wav`：无事件。
5. `stereo.wav`：仅对应 channel 输出。
6. 同一 channel 同时命中两个关键词时，JSON 中出现两个事件。
7. Python / ORT / Native score 仍按 6 位小数对齐。
8. `token_probs`、逐帧 top-k、`score_file` 格式不变。
9. 缺少 `keywords[].trigger_threshold` 的 config 解析失败。
10. 导出脚本生成的 config 包含 per-keyword 阈值，且不含 `ctc.trigger_threshold`。

---

## 7. 预期行为变化

- `你好问问` 的低分 hit 不再被 `嗨小问` 的全局阈值压制。
- 同一 channel 可能同时上报两个关键词，使用方需要自行仲裁。
- 阈值只影响事件输出，不影响 beam search、score 和调试数据。
- 旧配置文件不再可用，必须补齐每个关键词的 `trigger_threshold`。

---

## 8. 执行结果

- `android/tests/build_host.sh` 通过。
- `ch0.wav`：`channel=0 detected=嗨小问 score=0.787618`。
- `ch_nihao.wav`：现在输出 `channel=0 detected=你好问问 score=0.223723`；之前会被共享阈值过滤。
- `ch1.wav`：`no events`。
- `stereo.wav`：仅 `channel=0` 输出 `嗨小问`。
- 用 `ch0.wav + 静音 + ch_nihao.wav` 拼接样本验证：同一 channel 同时输出 `嗨小问` 和 `你好问问` 两个事件。
- `avg_30.pt` 的 Python / ONNX posterior / Native `ctc_decoder_cli` / Native streaming score 对齐到 6 位小数：
  - `ch0.wav`：`0.787617`（streaming `0.787618`）
  - `ch_nihao.wav`：`0.223722`（streaming `0.223723`）
- 修正了 best-hit 口径：不再保留历史最高分，而是保留每个 channel / 关键词最近一次在当前 beam 中观察到的命中，避免早期高分路径覆盖 Python 最终 beam 结果。
- 缺失 `keywords[].trigger_threshold` 的 config 解析失败：`ERROR: keyword '嗨小问' missing or invalid trigger_threshold`。
- `export_ctc_onnx.py` 未传 `--keyword_thresholds` 时直接失败；传入后生成的 config 含 `0.303 / 0.039`，且 `ctc` 段无 `trigger_threshold`。
- assets 中的 `kws.onnx` 已换为 `avg_30.pt` 导出模型；导出时 PyTorch / ONNX posterior 最大差约 `6.44e-6`。
