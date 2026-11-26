#ifndef HAIBINGTECH_KWS_WAV_READER_H_
#define HAIBINGTECH_KWS_WAV_READER_H_

#include <cstdint>
#include <string>
#include <vector>

namespace haibingtech {
namespace kws {

struct WavData {
  int sample_rate = 0;
  int channels = 0;
  std::vector<int16_t> data;  // interleaved
};

WavData ReadWav(const std::string& path);

}  // namespace kws
}  // namespace haibingtech

#endif  // HAIBINGTECH_KWS_WAV_READER_H_
