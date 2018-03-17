#ifndef WAVEFLOW_STA_H
#define WAVEFLOW_STA_H
#include "waveflow/core/common/device.h"
#include "waveflow/core/common/data_types.h"

namespace waveflow {
namespace functor {
template<typename Device, typename T>
struct STA {
  // Intentionally left unimplemented.
  void operator()(
      const Device &d,
      const T *input,
      int64 receiversCount,
      int64 samplesCount,
      float speedOfSound,
      float receiverWidth,
      float samplingFrequency,
      float startDepth,
      float areaHeight,
      int64 outputHeight,
      int64 outputWidth,
      T *output
  );
};

// CPU specialization
template<typename T>
struct STA<CPUDevice, T> {
  // Intentionally left unimplemented.
  void operator()(
      const CPUDevice &d,
      const T *input,
      int64 receiversCount,
      int64 samplesCount,
      float speedOfSound,
      float receiverWidth,
      float samplingFrequency,
      float startDepth,
      float areaHeight,
      int64 outputHeight,
      int64 outputWidth,
      T *output
  );
};
// GPU specialization
#ifdef GOOGLE_CUDA
template<typename T>
struct STA<GPUDevice, T> {
  // Intentionally left unimplemented.
  void operator()(
      const GPUDevice &d,
      const T *input,
      int64 receiversCount,
      int64 samplesCount,
      float speedOfSound,
      float receiverWidth,
      float samplingFrequency,
      float startDepth,
      float areaHeight,
      int64 outputHeight,
      int64 outputWidth,
      T *output
  );
};
#endif
}
}

#endif //WAVEFLOW_STA_H
