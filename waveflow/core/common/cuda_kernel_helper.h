#ifndef WAVEFLOW_CUDA_KERNEL_HELPER_H
#define WAVEFLOW_CUDA_KERNEL_HELPER_H

#include <cuda/include/driver_types.h>
#include <tensorflow/core/platform/default/logging.h>
#include <thrust/system_error.h>
#include <thrust/system/cuda/error.h>
#include <sstream>

// throws exception, when code != success
// idea from Jared Hoberock (stackoverflow q: 14038589)
#define REQUIRE_CUDA_SUCCESS(code) { require_cuda_success((code), __FILE__, __LINE__); }
inline void require_cuda_success(cudaError_t code, const char *file, int line) {
  if (code != cudaSuccess) {
    std::stringstream ss;
    ss << file << ":" << line;
    std::string position;
    ss >> position;
    throw thrust::system_error(code, thrust::cuda_category(), position);
  }
}

#endif //WAVEFLOW_CUDA_KERNEL_HELPER_H
