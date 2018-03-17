#include "tensorflow/core/framework/op_kernel.h"
#include "waveflow/core/common/device.h"
#include "waveflow/core/kernels/beamforming/sta.h"
#include "waveflow/core/common/cpu_kernel_utils.h"

namespace waveflow {
namespace functor {

static float get_relative_position(const int64 componentNo,
                                   const int64 componentsCount,
                                   const float entityWidth) {
  return (float) componentNo / (float) (componentsCount - 1) * entityWidth;
}

float time_lapse(const float distance,
                 const float speedOfSound) {
  return distance / speedOfSound;
}

template<typename T>
T delay_and_sum_and_interpolate(
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
    float receiveDistance =
        utils::cpu::euclidean_distance(outputX, outputY, receiverX, 0);
    float totalDistance = transmitDistance + receiveDistance;
    // approximate sample number
    float approxSampleNo = time_lapse(totalDistance - startDepth, speedOfSound)
        * samplingFrequency;

    T recvImpact = T(0);
    auto sampleNo = (int) approxSampleNo;
    if ((sampleNo < samplesCount) && (sampleNo >= 0)) {
      const int64 inputOffset = samplesCount * r;
      recvImpact =
          utils::cpu::interpolate_1d(input + inputOffset, approxSampleNo);
    }
    result += recvImpact;
  }
  return result;
}

template<typename T>
void STA<CPUDevice, T>::operator()(
    const CPUDevice &d,
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
    T *output
) {
  for (int x = 0; x < outputWidth; ++x) {
    for (int y = 0; y < outputHeight; ++y) {
      // physical (x,y) coordinates of pixel computed by this thread
      float posX = receiverWidth / (float) 2;
      float posY =
          (float) y / (float) (outputHeight - 1) * areaHeight + startDepth;

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
      output[x + y * outputWidth] = result;
    }
  }
}

template
struct STA<CPUDevice, float>;
template
struct STA<CPUDevice, double>;
}
}

