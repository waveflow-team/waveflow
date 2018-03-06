#ifndef WAVEFLOW_STA_OP_H
#define WAVEFLOW_STA_OP_H
#include "waveflow/core/kernels/device.h"

namespace waveflow {
namespace functor {
/**
 * Test op algo interface.
 *
 * @tparam Device device specialization
 * @tparam T IO data type
 */
// TODO(pjarosik) remove it soon
template<typename Device, typename T>
struct Test {
  // Intentionally left unimplemented.
  void operator()(const Device &d, int size, const T *in, T *out);
};

// CPU specialization
template<typename T>
struct Test<CPUDevice, T> {
  // Intentionally left unimplemented.
  void operator()(const CPUDevice &d, int size, const T *in, T *out);
};
// GPU specialization
#ifdef GOOGLE_CUDA
template<typename T>
struct Test<GPUDevice, T> {
  // Intentionally left unimplemented.
  void operator()(const GPUDevice &d, int size, const T *in, T *out);
};
#endif
}
}

#endif //WAVEFLOW_STA_OP_H

