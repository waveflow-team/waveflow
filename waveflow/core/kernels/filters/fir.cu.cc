// CUDA implementation introduced by
// Jakub Domaradzki (Warsaw University of Technology) and us4us team.
// Maintained by waveflow community.

#ifdef GOOGLE_CUDA
#define EIGEN_USE_GPU
#include "waveflow/core/kernels/common/device.h"
#include "waveflow/core/kernels/filters/fir.h"
#include "waveflow/core/kernels/common/cuda_kernel_helper.h"
#include <algorithm>

namespace waveflow {
namespace functor {

// Input should have shape: (... ,filtered axis)
// In other words, input array is treated as a one large list of vectors of
// 'axisLength' size. Each vector will be convolved with the filter.
template<typename T>
__global__ void fir_gpu(const T *input,
                        const int64 inputSize,
                        const int64 axisLength,
                        const T *filter,
                        const int64 filterSize,
                        const int64 alignedFilterSize,
                        T *output,
                        const int64 shmSize) {
  int idx = threadIdx.x + blockIdx.x * blockDim.x;
  // (ch)annel - number of receiver
  int64 ch = idx / axisLength;
  // (s)ample - number of sample
  int64 s = idx % axisLength;

  // Creates copy of input block, padded with values from neighbouring blocks
  // on the left (or zeros, if 'i' is out of bounds):
  // [prev_block[algnFilterSize]...,prev_block[length], curr_block[0], ..., c_block[length]]
  extern __shared__ float shm[];
  T *cachedInput = (T *) shm;
  for (int64 i = s - alignedFilterSize, localIdx = threadIdx.x;
       localIdx < shmSize;
       i += blockDim.x, localIdx += blockDim.x) {
    if (i < 0) {
      cachedInput[localIdx] = T(0);
    }
    else {
      cachedInput[localIdx] = input[ch * axisLength + i];
    }
  }

  __syncthreads();

  if (idx >= inputSize)
    return;

  // Computes output[idx] value.
  T result = T(0);
  int localN = threadIdx.x + alignedFilterSize;
  for (int64 i = 0; i < filterSize; ++i) {
    result += cachedInput[localN - i] * filter[i];
  }
  output[idx] = result;
}

template<typename T>
void FIRFilter<GPUDevice, T>::operator()(const GPUDevice &d,
                                         const T *input,
                                         const int64 inputSize,
                                         const int64 axisLength,
                                         const T *filter,
                                         const int64 filterSize,
                                         T *output) {
  dim3 blockDim(std::min<int64>(512, axisLength));
  dim3 gridDim((inputSize + blockDim.x - 1) / blockDim.x);
  const int64 alignedFilterSize = ((filterSize + 31) / 32) * 32;
  // shared memory size (number of array elements)
  const int64 shmSize = (blockDim.x + alignedFilterSize);
  const cudaStream_t &stream = d.stream();
  fir_gpu <<< gridDim, blockDim, shmSize * sizeof(T), stream >>> (
      input, inputSize, axisLength, filter, filterSize,
      alignedFilterSize, output,
      shmSize);
  REQUIRE_CUDA_SUCCESS(cudaGetLastError());
}

template
struct FIRFilter<GPUDevice, float>;
template
struct FIRFilter<GPUDevice, double>;
template
struct FIRFilter<GPUDevice, int32>;
}
}
#endif

