#include "engine/runtime_config.h"

#include <fstream>
#include <sstream>
#include <stdexcept>

#include "engine/json.h"

namespace haibingtech {
namespace kws {

namespace {

int JsonInt(const Json& root, const std::string& key, int default_value) {
  return root.get_int(key, default_value);
}

std::vector<int> JsonIntArray(const Json& value) {
  std::vector<int> result;
  if (!value.is_array()) return result;
  for (const Json& item : value.array()) {
    if (item.is_number()) result.push_back(static_cast<int>(item.number()));
  }
  return result;
}

std::vector<std::string> JsonStringArray(const Json& value) {
  std::vector<std::string> result;
  if (!value.is_array()) return result;
  for (const Json& item : value.array()) {
    if (item.is_string()) result.push_back(item.string());
  }
  return result;
}

}  // namespace

RuntimeConfig ParseRuntimeConfig(const std::string& json_text) {
  Json root = ParseJson(json_text);

  RuntimeConfig config;
  config.model_path = root.get_string("model_path", config.model_path);
  config.model_type = root.get_string("model_type", config.model_type);
  config.sample_rate = JsonInt(root, "sample_rate", config.sample_rate);
  config.feature_dim = JsonInt(root, "feature_dim", config.feature_dim);
  config.frame_length_ms = JsonInt(root, "frame_length_ms", config.frame_length_ms);
  config.frame_shift_ms = JsonInt(root, "frame_shift_ms", config.frame_shift_ms);
  config.hidden_dim = JsonInt(root, "hidden_dim", config.hidden_dim);
  config.cache_len = JsonInt(root, "cache_len", config.cache_len);
  config.vocab_size = JsonInt(root, "vocab_size", config.vocab_size);
  config.blank_id = JsonInt(root, "blank_id", config.blank_id);

  if (root.has("keywords_token_set")) {
    config.keywords_token_set = JsonIntArray(root.at("keywords_token_set"));
  }

  if (root.has("keywords") && root.at("keywords").is_array()) {
    for (const Json& item : root.at("keywords").array()) {
      Keyword keyword;
      keyword.text = item.get_string("text", "");
      if (item.has("chars")) keyword.chars = JsonStringArray(item.at("chars"));
      if (item.has("tokens")) keyword.tokens = JsonIntArray(item.at("tokens"));
      if (!item.has("trigger_threshold") ||
          !item.at("trigger_threshold").is_number()) {
        throw std::runtime_error("keyword '" + keyword.text +
                                 "' missing or invalid trigger_threshold");
      }
      keyword.trigger_threshold =
          static_cast<float>(item.get_double("trigger_threshold", -1.0));
      if (keyword.trigger_threshold < 0.0f) {
        throw std::runtime_error("keyword '" + keyword.text +
                                 "' has negative trigger_threshold");
      }
      if (!keyword.text.empty() && !keyword.tokens.empty()) {
        config.keywords.push_back(keyword);
      }
    }
  }

  if (root.has("ctc")) {
    const Json& ctc = root.at("ctc");
    config.ctc.blank_id = config.blank_id;
    config.ctc.score_beam_size = ctc.get_int("score_beam_size", config.ctc.score_beam_size);
    config.ctc.path_beam_size = ctc.get_int("path_beam_size", config.ctc.path_beam_size);
    config.ctc.prob_threshold =
        static_cast<float>(ctc.get_double("prob_threshold", config.ctc.prob_threshold));
    config.ctc.isclose_abs_tol = static_cast<float>(
        ctc.get_double("isclose_abs_tol", config.ctc.isclose_abs_tol));
    config.ctc.cooldown_ms = ctc.get_int("cooldown_ms", config.ctc.cooldown_ms);
    config.ctc.reset_decoder_after_trigger = ctc.get_bool(
        "reset_decoder_after_trigger", config.ctc.reset_decoder_after_trigger);
    config.ctc.reset_cache_after_trigger = ctc.get_bool(
        "reset_cache_after_trigger", config.ctc.reset_cache_after_trigger);
  }

  if (root.has("runtime")) {
    const Json& runtime = root.at("runtime");
    config.default_channels =
        runtime.get_int("default_channels", config.default_channels);
    config.chunk_frames = runtime.get_int("chunk_frames", config.chunk_frames);
    config.intra_op_threads =
        runtime.get_int("intra_op_threads", config.intra_op_threads);
    config.inter_op_threads =
        runtime.get_int("inter_op_threads", config.inter_op_threads);
  }

  if (root.has("debug")) {
    const Json& debug = root.at("debug");
    config.debug.dump_frame_topk =
        debug.get_bool("dump_frame_topk", config.debug.dump_frame_topk);
    config.debug.frame_topk_size =
        debug.get_int("frame_topk_size", config.debug.frame_topk_size);
    config.debug.max_debug_frames =
        debug.get_int("max_debug_frames", config.debug.max_debug_frames);
  }

  return config;
}

RuntimeConfig LoadRuntimeConfig(const std::string& json_path) {
  std::ifstream fin(json_path);
  if (!fin.good()) {
    throw std::runtime_error("config json not found: " + json_path);
  }
  std::ostringstream oss;
  oss << fin.rdbuf();
  return ParseRuntimeConfig(oss.str());
}

}  // namespace kws
}  // namespace haibingtech
