#ifndef WAVEFLOW_FIR_H
#define WAVEFLOW_FIR_H
#include "waveflow/core/common/device.h"
#include "waveflow/core/common/data_types.h"

namespace waveflow {
namespace functor {

template<typename Device, typename T>
struct FIRFilter {
  // Intentionally left unimplemented.
  void operator()(const Device &d,
                  const T *input,
                  const int64 inputSize,
                  const int64 axisLength,
                  const T *filter,
                  const int64 filterSize,
                  T *output
  );
};

// CPU specialization
template<typename T>
struct FIRFilter<CPUDevice, T> {
  // Intentionally left unimplemented.
  void operator()(const CPUDevice &d,
                  const T *input,
                  const int64 inputSize,
                  const int64 axisLength,
                  const T *filter,
                  const int64 filterSize,
                  T *output
  );
};
// GPU specialization
#ifdef GOOGLE_CUDA
template<typename T>
struct FIRFilter<GPUDevice, T> {
  // Intentionally left unimplemented.
  void operator()(const GPUDevice &d,
                  const T *input,
                  const int64 inputSize,
                  const int64 axisLength,
                  const T *filter,
                  const int64 filterSize,
                  T *output);
};
#endif
}
}

#endif //WAVEFLOW_FIR_H

