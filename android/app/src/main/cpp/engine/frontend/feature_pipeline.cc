#include "engine/frontend/feature_pipeline.h"

#include <algorithm>
#include <stdexcept>

namespace haibingtech {
namespace kws {

FeaturePipeline::FeaturePipeline(const FeatureConfig& config)
    : fbank_(MakeFbankConfig(config)) {}

FbankConfig FeaturePipeline::MakeFbankConfig(const FeatureConfig& config) const {
  FbankConfig fbank_config;
  fbank_config.num_bins = config.num_bins;
  fbank_config.sample_rate = config.sample_rate;
  fbank_config.frame_length = config.sample_rate * config.frame_length_ms / 1000;
  fbank_config.frame_shift = config.sample_rate * config.frame_shift_ms / 1000;
  fbank_config.low_freq = 20.0f;
  fbank_config.high_freq = static_cast<float>(config.sample_rate / 2);
  return fbank_config;
}

void FeaturePipeline::AcceptWaveform(const int16_t* pcm, size_t num_samples) {
  std::vector<float> float_pcm(num_samples);
  for (size_t i = 0; i < num_samples; ++i) {
    // Python frontend multiplies normalized float wav by (1 << 15).
    float_pcm[i] = static_cast<float>(pcm[i]);
  }
  AcceptWaveform(float_pcm.data(), float_pcm.size());
}

void FeaturePipeline::AcceptWaveform(const float* pcm, size_t num_samples) {
  if (finished_) {
    throw std::runtime_error("AcceptWaveform called after Finish");
  }
  std::vector<float> waves;
  waves.reserve(remained_.size() + num_samples);
  waves.insert(waves.end(), remained_.begin(), remained_.end());
  waves.insert(waves.end(), pcm, pcm + num_samples);

  const int frame_length = fbank_.config().frame_length;
  const int frame_shift = fbank_.config().frame_shift;
  if (static_cast<int>(waves.size()) < frame_length) {
    remained_ = std::move(waves);
    return;
  }

  const int num_frames =
      1 + (static_cast<int>(waves.size()) - frame_length) / frame_shift;
  // Keep the overlapping tail for the next frame. Consuming up to the end of
  // the last window would silently skip frames at chunk boundaries.
  const int consumed = num_frames * frame_shift;
  for (int i = 0; i < num_frames; ++i) {
    const float* start = waves.data() + i * frame_shift;
    queue_.push_back(fbank_.Compute(std::vector<float>(start, start + frame_length)));
  }

  remained_.assign(waves.begin() + consumed, waves.end());
}

bool FeaturePipeline::ReadFrames(int num_frames,
                                 std::vector<std::vector<float>>* feats) {
  feats->clear();
  for (int i = 0; i < num_frames; ++i) {
    if (queue_.empty()) {
      return false;
    }
    feats->push_back(std::move(queue_.front()));
    queue_.pop_front();
  }
  return true;
}

void FeaturePipeline::Reset() {
  finished_ = false;
  remained_.clear();
  queue_.clear();
}

}  // namespace kws
}  // namespace haibingtech
