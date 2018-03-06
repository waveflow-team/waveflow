#include "tensorflow/core/framework/op_kernel.h"
#include "waveflow/core/kernels/device.h"
#include "waveflow/core/kernels/test.h"

namespace waveflow {
namespace functor {

/**
 *
 * Implementation of test op kernel for CPU.
 *
 * @tparam T data type
 */
template<typename T>
void Test<CPUDevice, T>::operator()(const CPUDevice &d,
                                    int size,
                                    const T *in,
                                    T *out) {
  for (int i = 0; i < size; ++i) {
    out[i] = 2 * in[i];
  }
}

template struct Test<CPUDevice, float>;
template struct Test<CPUDevice, double>;
}
}





