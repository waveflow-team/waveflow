// CUDA implementation introduced by
// Marcin Lewandowski, Jakub Domaradzki (Warsaw University of Technology)
// and us4us team.
// Maintained by waveflow community.

#ifdef GOOGLE_CUDA
#define EIGEN_USE_GPU
#include "waveflow/core/common/device.h"
#include "waveflow/core/kernels/beamforming/sta.h"
#include "waveflow/core/common/cuda_kernel_utils.h"

using namespace waveflow::utils::cuda;
namespace waveflow {
namespace functor {

// Computes position of given component, relative to the first component of
// given entity.
static __device__ __forceinline__
float get_relative_position(const int componentNo,
                            const int componentsCount,
                            const float entityWidth) {
  return (float) componentNo / (float) (componentsCount - 1) * entityWidth;
}

static __device__ __forceinline__

float time_lapse(const float distance,
                 const float speedOfSound) {
  return distance / speedOfSound;
}

template<typename T>
__forceinline__ __device__ T
delay_and_sum_and_interpolate(
    const T *input,
    const int64 receiversCount,
    const int64 samplesCount,
    const float speedOfSound,
    const float receiverWidth,
    const float samplingFrequency,
    const float startDepth,
    const float transmitDistance,
    const float outputX,
    const float outputY
) {
  T result = T(0);
  for (int r = 0; r < receiversCount; ++r) {
    float receiverX = get_relative_position(r, receiversCount, receiverWidth);
    float receiveDistance = euclidean_distance(outputX, outputY, receiverX, 0);
    float totalDistance = transmitDistance + receiveDistance;
    // approximate sample number
    float approxSampleNo = time_lapse(totalDistance - startDepth, speedOfSound)
        * samplingFrequency;

    T recvImpact = T(0);
    auto sampleNo = (int) approxSampleNo;
    if ((sampleNo < samplesCount) && (sampleNo >= 0)) {
      const int64 inputOffset = samplesCount * r;
      recvImpact = interpolate_1d(input + inputOffset, approxSampleNo);
    }
    result += recvImpact;
  }
  return result;
}

// Coordinate system:
// OX(C) --> rec_1, rec_2, ....
// OY(S)     s1_1   s2_1
// |t1       s1_2   ...
// |t2       s1_3
// .t3       ...
template<typename T>
__global__ void sta_with_focusing_gpu(
    const T *input,
    const int64 receiversCount,
    const int64 samplesCount,
    const float speedOfSound,
    const float receiverWidth,
    const float samplingFrequency,
    const float areaHeight,
    const float startDepth,
    const int64 outputHeight,
    const int64 outputWidth,
    T *output) {
  // Here is computed one pixel output[y][x].
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  // (ph)ysical (x,y) coordinates of pixel computed by this thread
  float posX = receiverWidth / (float) 2;
  float posY = (float) y / (float) (outputHeight - 1) * areaHeight + startDepth;

  float transmitDistance = posY;
  // below implies, that x < eventsCount
  const int64 inputOffset = receiversCount * samplesCount * x;

  T result = delay_and_sum_and_interpolate(
      input + inputOffset,
      receiversCount,
      samplesCount,
      speedOfSound,
      receiverWidth,
      samplingFrequency,
      startDepth,
      transmitDistance,
      posX,
      posY);
  // blockDim.x*gridDim.x == outputWidth, see STA implementation
  output[x + y * blockDim.x * gridDim.x] = result;
}

template<typename T>
void STA<GPUDevice, T>::operator()(const GPUDevice &d,
                                   const T *input,
                                   const int64 receiversCount,
                                   const int64 samplesCount,
                                   const float speedOfSound,
                                   const float receiverWidth,
                                   const float samplingFrequency,
                                   const float startDepth,
                                   const float areaHeight,
                                   const int64 outputHeight,
                                   const int64 outputWidth,
                                   T *output) {
  const cudaStream_t &stream = d.stream();

  dim3 blockDim(8, 32);
  dim3 gridDim(outputWidth / blockDim.x, outputHeight / blockDim.y);

  sta_with_focusing_gpu<T> << < gridDim, blockDim, 0, stream >> > (
      input, receiversCount, samplesCount, speedOfSound, receiverWidth,
          samplingFrequency, areaHeight, startDepth, outputHeight,
          outputWidth, output
  );
  REQUIRE_CUDA_SUCCESS(cudaGetLastError());
}

template
struct STA<GPUDevice, float>;
template
struct STA<GPUDevice, double>;
}
}
#endif
