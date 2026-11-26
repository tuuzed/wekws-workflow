#include "engine/frontend/fbank.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

#include "engine/frontend/fft.h"

namespace haibingtech {
namespace kws {

int Fbank::NextPowerOfTwo(int n) {
  int p = 1;
  while (p < n) p <<= 1;
  return p;
}

Fbank::Fbank(const FbankConfig& config) : config_(config) {
  if (config_.frame_length <= 0 || config_.frame_shift <= 0 ||
      config_.num_bins <= 0) {
    throw std::invalid_argument("Invalid fbank config");
  }
  fft_points_ = NextPowerOfTwo(config_.frame_length);
  const int num_fft_bins = fft_points_ / 2;
  const float fft_bin_width =
      static_cast<float>(config_.sample_rate) / static_cast<float>(fft_points_);

  const float mel_low = MelScale(config_.low_freq);
  const float mel_high = MelScale(config_.high_freq);
  const float mel_delta = (mel_high - mel_low) / (config_.num_bins + 1);

  bins_.resize(config_.num_bins);
  for (int bin = 0; bin < config_.num_bins; ++bin) {
    const float left_mel = mel_low + bin * mel_delta;
    const float center_mel = mel_low + (bin + 1) * mel_delta;
    const float right_mel = mel_low + (bin + 2) * mel_delta;

    std::vector<float> weights(num_fft_bins, 0.0f);
    int first = -1;
    int last = -1;
    for (int i = 0; i < num_fft_bins; ++i) {
      const float freq = fft_bin_width * i;
      const float mel = MelScale(freq);
      const float up_slope = (mel - left_mel) / (center_mel - left_mel);
      const float down_slope = (right_mel - mel) / (right_mel - center_mel);
      const float weight = std::max(0.0f, std::min(up_slope, down_slope));
      weights[i] = weight;
      if (weight > 0.0f) {
        if (first < 0) first = i;
        last = i;
      }
    }
    if (first < 0 || last < first) {
      throw std::runtime_error("Invalid mel filter bank");
    }
    bins_[bin].first = first;
    bins_[bin].second.assign(weights.begin() + first, weights.begin() + last + 1);
  }

  // torchaudio.compliance.kaldi.fbank defaults to the Povey window:
  // Hann(window_size, periodic=False) ** 0.85.
  window_.resize(config_.frame_length);
  const double step = 2.0 * 3.14159265358979323846 /
                      static_cast<double>(config_.frame_length - 1);
  for (int i = 0; i < config_.frame_length; ++i) {
    const float hann = static_cast<float>(0.5 - 0.5 * std::cos(step * i));
    window_[i] = std::pow(hann, 0.85f);
  }
}

std::vector<float> Fbank::Compute(const std::vector<float>& frame) const {
  if (static_cast<int>(frame.size()) != config_.frame_length) {
    throw std::invalid_argument("Frame length mismatch");
  }

  std::vector<float> data(frame.begin(), frame.end());

  // Remove DC offset before preemphasis, matching Kaldi/torchaudio defaults.
  float mean = 0.0f;
  for (float v : data) mean += v;
  mean /= static_cast<float>(data.size());
  for (float& v : data) v -= mean;

  if (config_.preemphasis_coeff != 0.0f) {
    for (size_t i = data.size() - 1; i > 0; --i) {
      data[i] -= config_.preemphasis_coeff * data[i - 1];
    }
    data[0] -= config_.preemphasis_coeff * data[0];
  }

  for (size_t i = 0; i < data.size(); ++i) {
    data[i] *= window_[i];
  }

  std::vector<float> real(fft_points_, 0.0f);
  std::vector<float> imag(fft_points_, 0.0f);
  std::copy(data.begin(), data.end(), real.begin());
  Fft(&real, &imag);

  std::vector<float> power(fft_points_ / 2);
  for (size_t i = 0; i < power.size(); ++i) {
    power[i] = real[i] * real[i] + imag[i] * imag[i];
  }

  const float eps = std::numeric_limits<float>::epsilon();
  std::vector<float> feat(config_.num_bins);
  for (int bin = 0; bin < config_.num_bins; ++bin) {
    float energy = 0.0f;
    const int first = bins_[bin].first;
    const auto& weights = bins_[bin].second;
    for (size_t k = 0; k < weights.size(); ++k) {
      energy += weights[k] * power[first + k];
    }
    feat[bin] = std::log(energy < eps ? eps : energy);
  }
  return feat;
}

}  // namespace kws
}  // namespace haibingtech
