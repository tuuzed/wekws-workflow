#ifndef HAIBINGTECH_KWS_KWS_ENGINE_H_
#define HAIBINGTECH_KWS_KWS_ENGINE_H_

#include <cstddef>
#include <cstdint>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

#include <onnxruntime_cxx_api.h>  // NOLINT

#include "engine/ctc_beam_search.h"
#include "engine/frontend/feature_pipeline.h"
#include "engine/runtime_config.h"

namespace haibingtech {
namespace kws {

struct FrameTokenProb {
  int token_id = 0;
  float prob = 0.0f;
};

struct FrameDebug {
  int frame = 0;
  std::vector<FrameTokenProb> raw_topk;
  std::vector<FrameTokenProb> kept;
};

struct DetectionEvent {
  int channel = 0;
  std::string keyword;
  float score = 0.0f;
  int start_frame = 0;
  int end_frame = 0;
  std::vector<TokenProb> token_probs;
};

// Multi-channel streaming KWS engine. One ORT session is shared by all
// channels; every channel owns its own frontend, cache and CTC decoder state.
class KwsEngine {
 public:
  KwsEngine(const std::string& model_path, const RuntimeConfig& config,
            int channels);

  void AcceptWaveform(int channel, const int16_t* pcm, size_t num_samples);
  void Finish(int channel);

  // Returns true when one inference chunk was processed.
  bool DecodeOnce();
  std::vector<DetectionEvent> DecodeAll();

  // Best hit seen so far for each channel/keyword pair. Scores are refined
  // while more frames arrive, so the final value matches the offline Python
  // score. Every hit passing its own keyword threshold is returned.
  std::vector<DetectionEvent> CurrentEvents() const;

  // Debug buffer of per-frame top-k posterior values for one channel.
  std::vector<FrameDebug> FrameDebugDump(int channel,
                                         size_t max_frames = 0) const;

  void ResetChannel(int channel);

  int channels() const { return channels_; }
  int frame_shift_ms() const { return frame_shift_ms_; }

 private:
  struct ChannelState {
    explicit ChannelState(const FeatureConfig& feature_config)
        : frontend(feature_config) {}

    mutable std::mutex mutex;
    FeaturePipeline frontend;
    std::unique_ptr<CtcBeamSearch> decoder;
    std::vector<float> cache;
    std::vector<FrameDebug> frame_debug;
    int frame_offset = 0;
    int64_t last_trigger_ms = -1;
  };

  RuntimeConfig config_;
  int channels_;
  int frame_shift_ms_;
  int chunk_frames_;

  Ort::Env env_;
  Ort::SessionOptions session_options_;
  Ort::Session session_;
  std::vector<std::unique_ptr<ChannelState>> states_;
  std::vector<std::vector<KeywordHit>> best_hits_;
  std::vector<std::vector<bool>> has_hit_;
  std::vector<const char*> input_names_;
  std::vector<const char*> output_names_;

  static Ort::SessionOptions MakeSessionOptions(const RuntimeConfig& config);
};

}  // namespace kws
}  // namespace haibingtech

#endif  // HAIBINGTECH_KWS_KWS_ENGINE_H_
