#ifndef WAVEFLOW_DEVICE_H
#define WAVEFLOW_DEVICE_H
#include "tensorflow/core/framework/op_kernel.h"

namespace waveflow {
using CPUDevice = Eigen::ThreadPoolDevice;
using GPUDevice = Eigen::GpuDevice;
}

#endif //WAVEFLOW_DEVICE_H
