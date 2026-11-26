#include <algorithm>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "engine/kws_engine.h"
#include "engine/runtime_config.h"
#include "engine/wav_reader.h"

namespace {

std::string FormatTokenProbs(
    const std::vector<haibingtech::kws::TokenProb>& items) {
  std::ostringstream oss;
  oss << "[";
  for (size_t i = 0; i < items.size(); ++i) {
    if (i > 0) oss << ",";
    oss << items[i].token << "=" << std::fixed << std::setprecision(6)
        << items[i].prob << "@" << items[i].frame;
  }
  oss << "]";
  return oss.str();
}

void WriteFrameTokenProbJson(
    std::ostream& oss,
    const std::vector<haibingtech::kws::FrameTokenProb>& items) {
  oss << "[";
  for (size_t i = 0; i < items.size(); ++i) {
    if (i > 0) oss << ",";
    const auto& item = items[i];
    oss << "{\"token_id\":" << item.token_id
        << ",\"prob\":" << std::fixed << std::setprecision(6) << item.prob
        << "}";
  }
  oss << "]";
}

void WriteFrameDebugJson(const haibingtech::kws::KwsEngine& engine,
                         int channels, const std::string& path) {
  std::ofstream fout(path);
  if (!fout.good()) throw std::runtime_error("cannot open debug json: " + path);
  fout << "[";
  for (int c = 0; c < channels; ++c) {
    if (c > 0) fout << ",";
    std::vector<haibingtech::kws::FrameDebug> frames =
        engine.FrameDebugDump(c, 0);
    fout << "{\"channel\":" << c << ",\"frames\":[";
    for (size_t i = 0; i < frames.size(); ++i) {
      if (i > 0) fout << ",";
      const auto& frame = frames[i];
      fout << "{\"frame\":" << frame.frame
           << ",\"time_ms\":" << frame.frame * engine.frame_shift_ms()
           << ",\"raw_topk\":";
      WriteFrameTokenProbJson(fout, frame.raw_topk);
      fout << ",\"kept\":";
      WriteFrameTokenProbJson(fout, frame.kept);
      fout << "}";
    }
    fout << "]}";
  }
  fout << "]";
}

}  // namespace

int main(int argc, char** argv) {
  if (argc < 4 || argc > 7) {
    std::cerr << "Usage: kws_host_cli model.onnx config.json input.wav "
                 "[channels] [--debug_json path]"
              << std::endl;
    return 1;
  }

  try {
    const std::string model_path = argv[1];
    haibingtech::kws::RuntimeConfig config =
        haibingtech::kws::LoadRuntimeConfig(argv[2]);
    haibingtech::kws::WavData wav = haibingtech::kws::ReadWav(argv[3]);
    if (wav.sample_rate != config.sample_rate) {
      throw std::runtime_error("wav sample rate mismatch");
    }

    int channels = wav.channels;
    std::string debug_json;
    for (int i = 4; i < argc; ++i) {
      const std::string arg = argv[i];
      if (arg == "--debug_json" && i + 1 < argc) {
        debug_json = argv[++i];
      } else {
        channels = std::stoi(arg);
      }
    }
    if (channels <= 0 || channels > wav.channels) {
      throw std::runtime_error("invalid channel count");
    }

    haibingtech::kws::KwsEngine engine(model_path, config, channels);

    const size_t frames_per_chunk = 1600;  // 100 ms
    for (size_t offset = 0; offset < wav.data.size();
         offset += static_cast<size_t>(frames_per_chunk) * wav.channels) {
      const size_t remaining = wav.data.size() - offset;
      const size_t frames = std::min(
          frames_per_chunk, remaining / static_cast<size_t>(wav.channels));
      if (frames == 0) break;
      std::vector<std::vector<int16_t>> pcm(channels);
      for (int c = 0; c < channels; ++c) pcm[c].resize(frames);
      for (size_t i = 0; i < frames; ++i) {
        for (int c = 0; c < channels; ++c) {
          pcm[c][i] = wav.data[offset + i * wav.channels + c];
        }
      }
      for (int c = 0; c < channels; ++c) {
        engine.AcceptWaveform(c, pcm[c].data(), pcm[c].size());
      }
    }
    for (int c = 0; c < channels; ++c) engine.Finish(c);

    std::vector<haibingtech::kws::DetectionEvent> events;
    while (engine.DecodeOnce()) {
      events = engine.CurrentEvents();
      for (const auto& event : events) {
        std::cerr << "chunk ch=" << event.channel << " " << event.keyword
                  << " score=" << event.score << " " << event.start_frame
                  << "-" << event.end_frame << std::endl;
      }
    }
    events = engine.CurrentEvents();
    for (const auto& event : events) {
      const int start_ms = event.start_frame * engine.frame_shift_ms();
      const int end_ms = event.end_frame * engine.frame_shift_ms();
      std::cout << "channel=" << event.channel << " detected=" << event.keyword
                << " score=" << std::fixed << std::setprecision(6) << event.score
                << " start_ms=" << start_ms << " end_ms=" << end_ms
                << " token_probs=" << FormatTokenProbs(event.token_probs)
                << std::endl;
    }
    if (events.empty()) std::cout << "no events" << std::endl;
    if (!debug_json.empty()) {
      WriteFrameDebugJson(engine, channels, debug_json);
      std::cout << "debug_json=" << debug_json << std::endl;
    }
    return 0;
  } catch (const std::exception& e) {
    std::cerr << "ERROR: " << e.what() << std::endl;
    return 2;
  }
}
