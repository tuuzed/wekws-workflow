#ifndef HAIBINGTECH_KWS_FRONTEND_FEATURE_PIPELINE_H_
#define HAIBINGTECH_KWS_FRONTEND_FEATURE_PIPELINE_H_

#include <cstdint>
#include <deque>
#include <vector>

#include "engine/frontend/fbank.h"

namespace haibingtech {
namespace kws {

struct FeatureConfig {
  int num_bins = 40;
  int sample_rate = 16000;
  int frame_length_ms = 25;
  int frame_shift_ms = 10;
};

// Streaming fbank frontend for one audio channel.
class FeaturePipeline {
 public:
  explicit FeaturePipeline(const FeatureConfig& config);

  void AcceptWaveform(const int16_t* pcm, size_t num_samples);
  void AcceptWaveform(const float* pcm, size_t num_samples);
  void Finish() { finished_ = true; }

  bool ReadFrames(int num_frames, std::vector<std::vector<float>>* feats);
  size_t NumQueuedFrames() const { return queue_.size(); }
  bool finished() const { return finished_; }
  int frame_shift() const { return fbank_.config().frame_shift; }

  void Reset();

 private:
  FbankConfig MakeFbankConfig(const FeatureConfig& config) const;

  Fbank fbank_;
  bool finished_ = false;
  std::vector<float> remained_;
  std::deque<std::vector<float>> queue_;
};

}  // namespace kws
}  // namespace haibingtech

#endif  // HAIBINGTECH_KWS_FRONTEND_FEATURE_PIPELINE_H_
