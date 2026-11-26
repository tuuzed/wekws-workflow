#ifndef HAIBINGTECH_KWS_FRONTEND_FBANK_H_
#define HAIBINGTECH_KWS_FRONTEND_FBANK_H_

#include <cmath>
#include <utility>
#include <vector>

namespace haibingtech {
namespace kws {

struct FbankConfig {
  int num_bins = 40;
  int sample_rate = 16000;
  int frame_length = 400;  // samples, 25 ms at 16 kHz
  int frame_shift = 160;   // samples, 10 ms at 16 kHz
  float preemphasis_coeff = 0.97f;
  float low_freq = 20.0f;
  float high_freq = 8000.0f;
};

// Kaldi-style log-mel fbank, matching the frontend used by WeKWS training.
class Fbank {
 public:
  explicit Fbank(const FbankConfig& config);
  std::vector<float> Compute(const std::vector<float>& frame) const;
  int num_bins() const { return config_.num_bins; }
  const FbankConfig& config() const { return config_; }

 private:
  static float MelScale(float freq) {
    return 1127.0f * std::log(1.0f + freq / 700.0f);
  }
  static float InverseMelScale(float mel) {
    return 700.0f * (std::exp(mel / 1127.0f) - 1.0f);
  }
  static int NextPowerOfTwo(int n);

  FbankConfig config_;
  int fft_points_ = 0;
  std::vector<std::pair<int, std::vector<float>>> bins_;
  std::vector<float> window_;
};

}  // namespace kws
}  // namespace haibingtech

#endif  // HAIBINGTECH_KWS_FRONTEND_FBANK_H_
