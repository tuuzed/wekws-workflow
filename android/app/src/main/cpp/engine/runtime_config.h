#ifndef HAIBINGTECH_KWS_RUNTIME_CONFIG_H_
#define HAIBINGTECH_KWS_RUNTIME_CONFIG_H_

#include <string>
#include <vector>

namespace haibingtech {
namespace kws {

struct Keyword {
  std::string text;
  std::vector<std::string> chars;
  std::vector<int> tokens;
  float trigger_threshold = -1.0f;
};

struct CtcConfig {
  int blank_id = 0;
  int score_beam_size = 3;
  int path_beam_size = 20;
  float prob_threshold = 0.05f;
  float isclose_abs_tol = 1e-6f;
  int cooldown_ms = 1000;
  bool reset_decoder_after_trigger = true;
  bool reset_cache_after_trigger = false;
};

struct DebugConfig {
  bool dump_frame_topk = false;
  int frame_topk_size = 3;
  int max_debug_frames = 2000;
};

struct RuntimeConfig {
  std::string model_path = "kws.onnx";
  std::string model_type = "ctc";
  int sample_rate = 16000;
  int feature_dim = 40;
  int frame_length_ms = 25;
  int frame_shift_ms = 10;
  int hidden_dim = 256;
  int cache_len = 105;
  int vocab_size = 2599;
  int blank_id = 0;
  int default_channels = 2;
  int chunk_frames = 80;
  int intra_op_threads = 1;
  int inter_op_threads = 1;
  std::vector<int> keywords_token_set;
  std::vector<Keyword> keywords;
  CtcConfig ctc;
  DebugConfig debug;
};

RuntimeConfig ParseRuntimeConfig(const std::string& json_text);
RuntimeConfig LoadRuntimeConfig(const std::string& json_path);

}  // namespace kws
}  // namespace haibingtech

#endif  // HAIBINGTECH_KWS_RUNTIME_CONFIG_H_
