#ifndef WAVEFLOW_CPU_KERNEL_UTILS_H
#define WAVEFLOW_CPU_KERNEL_UTILS_H
#include <cmath>

namespace waveflow {
namespace utils {
namespace cpu {

template<class T>
T interpolate_1d(const T *input, const float approxSampleNo) {
  int sampleNoFloor = static_cast<int>(std::floor(approxSampleNo));
  float ratio = approxSampleNo - sampleNoFloor;
  return (1.0f - ratio) * input[sampleNoFloor]
      + ratio * input[sampleNoFloor + 1];
}

static float sq(const float x) {
  return x * x;
}

// Computes L2 distance between (x1,y1) and (x2,y2)
static float euclidean_distance(const float x1,
                                const float y1,
                                const float x2,
                                const float y2) {
  return sqrtf(sq(x1 - x2) + sq(y1 - y2));
}
}
}
}

#endif //WAVEFLOW_CPU_KERNEL_UTILS_H
