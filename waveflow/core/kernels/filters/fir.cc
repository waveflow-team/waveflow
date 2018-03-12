#include "tensorflow/core/framework/op_kernel.h"
#include "waveflow/core/common/device.h"
#include "waveflow/core/kernels/filters/fir.h"

namespace waveflow {
namespace functor {

// computes conv in[i-j] * filter[j], input is padded from the left with zeros
// naive: O(n*k)
template<typename T>
void conv_cpu(const int64 start,
              const T *input,
              const int64 axisLength,
              const T *filter,
              const int64 filterSize,
              T *output) {
  for (int i = 0; i < axisLength; ++i) {
    T result = T(0);
    for (int j = 0; j < filterSize; ++j) {
      if(i < j) {
        break;
      }
      result += input[start + i - j] * filter[j];
    }
    output[start + i] = result;
  }
}

template<typename T>
void fir_cpu(const T *input,
             const int64 inputSize,
             const int64 axisLength,
             const T *filter,
             const int64 filterSize,
             T *output
) {
  for (int i = 0; i < inputSize; i += axisLength) {
    conv_cpu(i, input, axisLength, filter, filterSize, output);
  }
}

template<typename T>
void FIRFilter<CPUDevice, T>::operator()(const CPUDevice &d,
                                         const T *input,
                                         const int64 inputSize,
                                         const int64 axisLength,
                                         const T *filter,
                                         const int64 filterSize,
                                         T *output
) {
  fir_cpu(input, inputSize, axisLength, filter, filterSize, output);
}

template
struct FIRFilter<CPUDevice, float>;
template
struct FIRFilter<CPUDevice, double>;
template
struct FIRFilter<CPUDevice, int32>;
}
}





