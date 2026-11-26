#include "engine/wav_reader.h"

#include <algorithm>
#include <cstring>
#include <fstream>
#include <iterator>
#include <stdexcept>
#include <vector>

namespace haibingtech {
namespace kws {

namespace {

uint32_t ReadU32LE(const unsigned char* p) {
  return static_cast<uint32_t>(p[0]) |
         (static_cast<uint32_t>(p[1]) << 8) |
         (static_cast<uint32_t>(p[2]) << 16) |
         (static_cast<uint32_t>(p[3]) << 24);
}

uint16_t ReadU16LE(const unsigned char* p) {
  return static_cast<uint16_t>(p[0] | (p[1] << 8));
}

}  // namespace

WavData ReadWav(const std::string& path) {
  std::ifstream fin(path, std::ios::binary);
  if (!fin.good()) throw std::runtime_error("wav not found: " + path);

  std::vector<unsigned char> bytes((std::istreambuf_iterator<char>(fin)),
                                   std::istreambuf_iterator<char>());
  if (bytes.size() < 44) throw std::runtime_error("wav too short");

  auto pos = bytes.begin();
  if (std::memcmp(&*pos, "RIFF", 4) != 0 ||
      std::memcmp(&bytes[8], "WAVE", 4) != 0) {
    throw std::runtime_error("not a RIFF/WAVE file");
  }

  WavData wav;
  bool has_fmt = false;
  bool has_data = false;
  size_t cursor = 12;
  while (cursor + 8 <= bytes.size()) {
    const unsigned char* chunk = bytes.data() + cursor;
    const std::string id(reinterpret_cast<const char*>(chunk), 4);
    const uint32_t size = ReadU32LE(chunk + 4);
    const size_t payload = cursor + 8;
    if (payload + size > bytes.size()) break;

    if (id == "fmt ") {
      if (size < 16) throw std::runtime_error("invalid fmt chunk");
      const unsigned char* fmt = bytes.data() + payload;
      const uint16_t format = ReadU16LE(fmt);
      if (format != 1) throw std::runtime_error("only PCM wav is supported");
      wav.channels = ReadU16LE(fmt + 2);
      wav.sample_rate = static_cast<int>(ReadU32LE(fmt + 4));
      const uint16_t bits = ReadU16LE(fmt + 14);
      if (bits != 16) throw std::runtime_error("only 16-bit wav is supported");
      has_fmt = true;
    } else if (id == "data") {
      const size_t num_bytes = size;
      wav.data.resize(num_bytes / 2);
      std::memcpy(wav.data.data(), bytes.data() + payload, num_bytes);
      has_data = true;
    }

    cursor = payload + size + (size % 2);
  }

  if (!has_fmt || !has_data || wav.channels <= 0) {
    throw std::runtime_error("invalid wav chunks");
  }
  return wav;
}

}  // namespace kws
}  // namespace haibingtech
