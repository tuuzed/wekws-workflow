#include <algorithm>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include "engine/frontend/feature_pipeline.h"
#include "engine/wav_reader.h"

int main(int argc, char** argv) {
  if (argc != 3) {
    std::cerr << "Usage: fbank_dump_cli input.wav out.bin\n";
    return 1;
  }
  auto wav = haibingtech::kws::ReadWav(argv[1]);
  haibingtech::kws::FeatureConfig cfg;
  cfg.num_bins = 40;
  cfg.sample_rate = 16000;
  cfg.frame_length_ms = 25;
  cfg.frame_shift_ms = 10;
  haibingtech::kws::FeaturePipeline fp(cfg);
  for (size_t offset = 0; offset < wav.data.size(); offset += 1600) {
    size_t n = std::min<size_t>(1600, wav.data.size() - offset);
    fp.AcceptWaveform(wav.data.data() + offset, n);
  }
  fp.Finish();
  std::vector<std::vector<float>> feats;
  int total = 0;
  while (fp.ReadFrames(1, &feats)) {
    std::ofstream out(argv[2], total == 0 ? std::ios::binary : std::ios::binary | std::ios::app);
    out.write(reinterpret_cast<const char*>(feats[0].data()),
              feats[0].size() * sizeof(float));
    ++total;
  }
  std::cout << "frames " << total << std::endl;
  return 0;
}
