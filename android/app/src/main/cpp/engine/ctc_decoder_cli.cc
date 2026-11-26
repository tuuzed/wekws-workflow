#include <cstdio>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include "engine/ctc_beam_search.h"
#include "engine/runtime_config.h"

int main(int argc, char** argv) {
  if (argc != 4) {
    std::cerr << "Usage: ctc_decoder_cli config.json probs.bin T" << std::endl;
    return 1;
  }

  try {
    haibingtech::kws::RuntimeConfig config =
        haibingtech::kws::LoadRuntimeConfig(argv[1]);
    const int frames = std::stoi(argv[3]);
    if (frames <= 0) throw std::runtime_error("invalid frame count");

    std::ifstream fin(argv[2], std::ios::binary);
    if (!fin.good()) throw std::runtime_error("probs.bin not found");
    const size_t count = static_cast<size_t>(frames) * config.vocab_size;
    std::vector<float> probs(count);
    fin.read(reinterpret_cast<char*>(probs.data()),
             static_cast<std::streamsize>(count * sizeof(float)));
    if (!fin) throw std::runtime_error("failed to read probs.bin");

    haibingtech::kws::CtcBeamSearch decoder(config.ctc, config.keywords);
    for (int t = 0; t < frames; ++t) {
      decoder.ProcessFrame(probs.data() + static_cast<size_t>(t) * config.vocab_size,
                           config.vocab_size, t);
    }

    haibingtech::kws::KeywordHit hit;
    if (decoder.FindBestHit(&hit)) {
      std::cout << "detected " << hit.keyword << " " << std::fixed
                << std::setprecision(6) << hit.score << " " << hit.start_frame
                << " " << hit.end_frame << std::endl;
    } else {
      std::cout << "rejected" << std::endl;
    }
    return 0;
  } catch (const std::exception& e) {
    std::cerr << "ERROR: " << e.what() << std::endl;
    return 2;
  }
}
