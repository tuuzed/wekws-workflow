#ifndef HAIBINGTECH_KWS_FRONTEND_FFT_H_
#define HAIBINGTECH_KWS_FRONTEND_FFT_H_

#include <vector>

namespace haibingtech {
namespace kws {

// In-place radix-2 complex FFT. The input size must be a power of two.
void Fft(std::vector<float>* real, std::vector<float>* imag);

}  // namespace kws
}  // namespace haibingtech

#endif  // HAIBINGTECH_KWS_FRONTEND_FFT_H_
