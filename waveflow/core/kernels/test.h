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
#include "waveflow/core/kernels/test.inc"
}
}

#endif //WAVEFLOW_STA_OP_H

