#include "engine/kws_engine.h"

#include <algorithm>
#include <limits>
#include <stdexcept>
#include <utility>

namespace haibingtech {
namespace kws {

namespace {

std::vector<FrameTokenProb> ExtractTopK(const float* probs, int vocab_size,
                                        int top_k) {
  std::vector<FrameTokenProb> result;
  if (probs == nullptr || vocab_size <= 0 || top_k <= 0) return result;
  const int k = std::min(top_k, vocab_size);

  std::vector<std::pair<float, int>> all;
  all.reserve(static_cast<size_t>(vocab_size));
  for (int i = 0; i < vocab_size; ++i) {
    all.emplace_back(probs[i], i);
  }
  std::partial_sort(
      all.begin(), all.begin() + k, all.end(),
      [](const std::pair<float, int>& a, const std::pair<float, int>& b) {
        return a.first > b.first;
      });

  result.reserve(static_cast<size_t>(k));
  for (int i = 0; i < k; ++i) {
    result.push_back({all[static_cast<size_t>(i)].second,
                      all[static_cast<size_t>(i)].first});
  }
  return result;
}

bool IsKeywordToken(const std::vector<int>& token_set, int token_id) {
  return std::find(token_set.begin(), token_set.end(), token_id) !=
         token_set.end();
}

}  // namespace

Ort::SessionOptions KwsEngine::MakeSessionOptions(const RuntimeConfig& config) {
  Ort::SessionOptions options;
  options.SetIntraOpNumThreads(config.intra_op_threads);
  options.SetInterOpNumThreads(config.inter_op_threads);
  options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
  return options;
}

KwsEngine::KwsEngine(const std::string& model_path, const RuntimeConfig& config,
                     int channels)
    : config_(config),
      channels_(channels),
      frame_shift_ms_(config.frame_shift_ms),
      chunk_frames_(config.chunk_frames),
      env_(ORT_LOGGING_LEVEL_WARNING, "haibingtech_kws"),
      session_options_(MakeSessionOptions(config)),
      session_(env_, model_path.c_str(), session_options_) {
  if (channels_ <= 0) throw std::invalid_argument("channels must be positive");
  if (config_.keywords.empty()) throw std::invalid_argument("no keywords configured");

  input_names_ = {"input", "cache"};
  output_names_ = {"output", "r_cache"};

  FeatureConfig feature_config;
  feature_config.num_bins = config_.feature_dim;
  feature_config.sample_rate = config_.sample_rate;
  feature_config.frame_length_ms = config_.frame_length_ms;
  feature_config.frame_shift_ms = config_.frame_shift_ms;

  for (int c = 0; c < channels_; ++c) {
    states_.emplace_back(new ChannelState(feature_config));
    ChannelState& state = *states_.back();
    state.decoder.reset(new CtcBeamSearch(config_.ctc, config_.keywords));
    state.cache.assign(
        static_cast<size_t>(config_.hidden_dim) * config_.cache_len, 0.0f);
    best_hits_.emplace_back(config_.keywords.size());
    has_hit_.emplace_back(config_.keywords.size(), false);
  }
}

void KwsEngine::AcceptWaveform(int channel, const int16_t* pcm,
                               size_t num_samples) {
  if (channel < 0 || channel >= channels_) {
    throw std::out_of_range("invalid channel id");
  }
  ChannelState& state = *states_[channel];
  std::lock_guard<std::mutex> lock(state.mutex);
  state.frontend.AcceptWaveform(pcm, num_samples);
}

void KwsEngine::Finish(int channel) {
  if (channel < 0 || channel >= channels_) {
    throw std::out_of_range("invalid channel id");
  }
  ChannelState& state = *states_[channel];
  std::lock_guard<std::mutex> lock(state.mutex);
  state.frontend.Finish();
}

bool KwsEngine::DecodeOnce() {
  size_t max_frames = 0;
  std::vector<size_t> queued(static_cast<size_t>(channels_), 0);
  for (int c = 0; c < channels_; ++c) {
    ChannelState& state = *states_[c];
    std::lock_guard<std::mutex> lock(state.mutex);
    queued[static_cast<size_t>(c)] = state.frontend.NumQueuedFrames();
    max_frames = std::max(max_frames, queued[static_cast<size_t>(c)]);
  }
  if (max_frames == 0) return false;

  const int frames =
      static_cast<int>(std::min<size_t>(chunk_frames_, max_frames));
  if (frames <= 0) return false;

  // Batch only the channels that have enough frames. This keeps channels with
  // different stream lengths independent while still using one ONNX Run.
  std::vector<int> active_channels;
  for (int c = 0; c < channels_; ++c) {
    if (queued[static_cast<size_t>(c)] >= static_cast<size_t>(frames)) {
      active_channels.push_back(c);
    }
  }
  if (active_channels.empty()) return false;

  const int batch = static_cast<int>(active_channels.size());
  const int feature_dim = config_.feature_dim;
  const int hidden_dim = config_.hidden_dim;
  const int cache_len = config_.cache_len;
  const int vocab_size = config_.vocab_size;

  std::vector<std::vector<std::vector<float>>> per_channel_feats(batch);
  std::vector<float> feats_flat(static_cast<size_t>(batch) * frames *
                                feature_dim);
  std::vector<float> cache_flat(static_cast<size_t>(batch) * hidden_dim *
                                cache_len);

  for (int b = 0; b < batch; ++b) {
    const int c = active_channels[b];
    ChannelState& state = *states_[c];
    std::lock_guard<std::mutex> lock(state.mutex);
    if (!state.frontend.ReadFrames(frames, &per_channel_feats[b])) {
      return false;
    }
    for (int t = 0; t < frames; ++t) {
      const float* src = per_channel_feats[b][t].data();
      float* dst = feats_flat.data() +
                   (static_cast<size_t>(b) * frames + t) * feature_dim;
      std::copy(src, src + feature_dim, dst);
    }
    const float* src_cache = state.cache.data();
    float* dst_cache = cache_flat.data() +
                       static_cast<size_t>(b) * hidden_dim * cache_len;
    std::copy(src_cache, src_cache + hidden_dim * cache_len, dst_cache);
  }

  Ort::MemoryInfo memory_info =
      Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);
  const int64_t feats_shape[] = {batch, frames, feature_dim};
  const int64_t cache_shape[] = {batch, hidden_dim, cache_len};
  Ort::Value feats_value = Ort::Value::CreateTensor<float>(
      memory_info, feats_flat.data(), feats_flat.size(), feats_shape, 3);
  Ort::Value cache_value = Ort::Value::CreateTensor<float>(
      memory_info, cache_flat.data(), cache_flat.size(), cache_shape, 3);

  std::vector<Ort::Value> inputs;
  inputs.reserve(2);
  inputs.emplace_back(std::move(feats_value));
  inputs.emplace_back(std::move(cache_value));

  std::vector<Ort::Value> outputs = session_.Run(
      Ort::RunOptions{nullptr}, input_names_.data(), inputs.data(),
      inputs.size(), output_names_.data(), output_names_.size());
  if (outputs.size() != 2) throw std::runtime_error("unexpected ORT outputs");

  const float* output_data = outputs[0].GetTensorData<float>();
  const float* r_cache_data = outputs[1].GetTensorData<float>();
  for (int b = 0; b < batch; ++b) {
    const int c = active_channels[b];
    ChannelState& state = *states_[c];
    std::lock_guard<std::mutex> lock(state.mutex);
    const float* src = r_cache_data +
                       static_cast<size_t>(b) * hidden_dim * cache_len;
    std::copy(src, src + hidden_dim * cache_len, state.cache.begin());

    for (int t = 0; t < frames; ++t) {
      const int frame = state.frame_offset + t;
      const float* probs = output_data +
                           (static_cast<size_t>(b) * frames + t) * vocab_size;
      state.decoder->ProcessFrame(probs, vocab_size, frame);

      if (config_.debug.dump_frame_topk &&
          config_.debug.frame_topk_size > 0 &&
          static_cast<int>(state.frame_debug.size()) <
              config_.debug.max_debug_frames) {
        FrameDebug frame_debug;
        frame_debug.frame = frame;
        frame_debug.raw_topk = ExtractTopK(
            probs, vocab_size, config_.debug.frame_topk_size);
        for (const FrameTokenProb& item : frame_debug.raw_topk) {
          if (item.prob > config_.ctc.prob_threshold &&
              IsKeywordToken(config_.keywords_token_set, item.token_id)) {
            frame_debug.kept.push_back(item);
          }
        }
        state.frame_debug.push_back(std::move(frame_debug));
      }

      std::vector<KeywordHit> hits;
      state.decoder->FindBestHits(&hits);
      // Keep the most recent hit observed for each keyword, not the historical
      // maximum. A transient high score early in the beam can disappear from
      // the final beam; Python/score_ctc.py reports the current beam hit.
      // Do not clear an old hit when the keyword temporarily drops out, so a
      // multi-keyword stream can still report earlier keywords.
      for (size_t k = 0; k < hits.size(); ++k) {
        if (hits[k].keyword.empty()) continue;
        best_hits_[c][k] = hits[k];
        has_hit_[c][k] = true;
      }
    }
    state.frame_offset += frames;
  }
  return true;
}

std::vector<DetectionEvent> KwsEngine::CurrentEvents() const {
  std::vector<DetectionEvent> events;
  for (int c = 0; c < channels_; ++c) {
    ChannelState& state = *states_[c];
    std::lock_guard<std::mutex> lock(state.mutex);
    for (size_t k = 0; k < config_.keywords.size(); ++k) {
      if (!has_hit_[c][k]) continue;
      const KeywordHit& hit = best_hits_[c][k];
      if (hit.score < config_.keywords[k].trigger_threshold) continue;
      DetectionEvent event;
      event.channel = c;
      event.keyword = hit.keyword;
      event.score = hit.score;
      event.start_frame = hit.start_frame;
      event.end_frame = hit.end_frame;
      event.token_probs = hit.token_probs;
      events.push_back(event);
    }
  }
  return events;
}

std::vector<FrameDebug> KwsEngine::FrameDebugDump(int channel,
                                                  size_t max_frames) const {
  if (channel < 0 || channel >= channels_) {
    throw std::out_of_range("invalid channel id");
  }
  const ChannelState& state = *states_[channel];
  std::lock_guard<std::mutex> lock(state.mutex);
  if (max_frames == 0 || max_frames > state.frame_debug.size()) {
    return state.frame_debug;
  }
  return std::vector<FrameDebug>(
      state.frame_debug.begin(), state.frame_debug.begin() + max_frames);
}

std::vector<DetectionEvent> KwsEngine::DecodeAll() {
  while (DecodeOnce()) {
  }
  return CurrentEvents();
}

void KwsEngine::ResetChannel(int channel) {
  if (channel < 0 || channel >= channels_) {
    throw std::out_of_range("invalid channel id");
  }
  ChannelState& state = *states_[channel];
  std::lock_guard<std::mutex> lock(state.mutex);
  state.frontend.Reset();
  state.decoder->Reset();
  state.frame_debug.clear();
  std::fill(state.cache.begin(), state.cache.end(), 0.0f);
  state.frame_offset = 0;
  state.last_trigger_ms = -1;
  std::fill(has_hit_[channel].begin(), has_hit_[channel].end(), false);
  std::fill(best_hits_[channel].begin(), best_hits_[channel].end(),
            KeywordHit());
}

}  // namespace kws
}  // namespace haibingtech
