#include "device.h"
#include "tensorflow/core/framework/op_kernel.h"

using namespace waveflow;
/**
 * Implementation of test op kernel for CPU.
 *
 * @tparam T data type
 */
template <typename T>
struct Test<CPUDevice, T> {
  void operator()(const CPUDevice & d, int size, const T* in, T* out) {
    for (int i = 0; i < size; ++i) {
      out[i] = 2 * in[i];
    }
  }
};



