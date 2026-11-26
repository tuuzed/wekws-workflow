#include "engine/frontend/fft.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <stdexcept>

namespace haibingtech {
namespace kws {

void Fft(std::vector<float>* real, std::vector<float>* imag) {
  const size_t n = real->size();
  if (n == 0 || (n & (n - 1)) != 0 || imag->size() != n) {
    throw std::invalid_argument("FFT size must be a positive power of two");
  }

  std::vector<double> re(real->begin(), real->end());
  std::vector<double> im(imag->begin(), imag->end());

  // Bit-reversal permutation.
  size_t j = 0;
  for (size_t i = 1; i < n; ++i) {
    size_t bit = n >> 1;
    while (j & bit) {
      j ^= bit;
      bit >>= 1;
    }
    j ^= bit;
    if (i < j) {
      std::swap(re[i], re[j]);
      std::swap(im[i], im[j]);
    }
  }

  const double kPi = 3.141592653589793238462643383279502884;
  for (size_t len = 2; len <= n; len <<= 1) {
    const double angle = -2.0 * kPi / static_cast<double>(len);
    const double wr_step = std::cos(angle);
    const double wi_step = std::sin(angle);
    for (size_t start = 0; start < n; start += len) {
      double wr = 1.0;
      double wi = 0.0;
      for (size_t k = 0; k < len / 2; ++k) {
        const size_t a = start + k;
        const size_t b = a + len / 2;
        const double tr = re[b] * wr - im[b] * wi;
        const double ti = re[b] * wi + im[b] * wr;
        re[b] = re[a] - tr;
        im[b] = im[a] - ti;
        re[a] += tr;
        im[a] += ti;
        const double next_wr = wr * wr_step - wi * wi_step;
        wi = wr * wi_step + wi * wr_step;
        wr = next_wr;
      }
    }
  }

  for (size_t i = 0; i < n; ++i) {
    (*real)[i] = static_cast<float>(re[i]);
    (*imag)[i] = static_cast<float>(im[i]);
  }
}

}  // namespace kws
}  // namespace haibingtech
