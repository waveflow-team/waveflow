#ifndef WAVEFLOW_CUDA_KERNEL_HELPER_H
#define WAVEFLOW_CUDA_KERNEL_HELPER_H

#ifdef GOOGLE_CUDA
#include <cuda/include/driver_types.h>
#include <tensorflow/core/platform/default/logging.h>
#include <thrust/system_error.h>
#include <thrust/system/cuda/error.h>
#include <sstream>

namespace waveflow {
namespace utils {
namespace cuda {

// throws exception, when code != success
// idea from Jared Hoberock (stackoverflow q: 14038589)
#define REQUIRE_CUDA_SUCCESS(code) { waveflow::utils::cuda::require_cuda_success((code), __FILE__, __LINE__); }
inline void require_cuda_success(cudaError_t code, const char *file, int line) {
  if (code != cudaSuccess) {
    std::stringstream ss;
    ss << file << ":" << line;
    std::string position;
    ss >> position;
    throw thrust::system_error(code, thrust::cuda_category(), position);
  }
}

// Computes value interpolated between input[sampleNo] and input[sampleNo+1]
template<class T>
static __device__ __forceinline__
    T
interpolate_1d(const T *input, const float approxSampleNo) {
  int sampleNoFloor = floorf(approxSampleNo);
  float ratio = approxSampleNo - sampleNoFloor;
  return (1.0f - ratio) * input[sampleNoFloor]
      + ratio * input[sampleNoFloor + 1];
}

static __device__ __forceinline__ float sq(const float x) {
  return x * x;
}

// Computes L2 distance between (x1,y1) and (x2,y2)
static __device__ __forceinline__
float euclidean_distance(const float x1,
                         const float y1,
                         const float x2,
                         const float y2) {
  return sqrtf(sq(x1 - x2) + sq(y1 - y2));
}

}
}
}
#endif

#endif //WAVEFLOW_CUDA_KERNEL_HELPER_H
