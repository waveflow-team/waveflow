#ifndef WAVEFLOW_REGISTER_KERNEL_H
#define WAVEFLOW_REGISTER_KERNEL_H
#include "waveflow/core/common/device.h"
#include "waveflow/core/common/waveflow_op.h"
#include "tensorflow/core/framework/op_kernel.h"

// Convenience macro for cpu op kernel registration.
#define REGISTER_WF_CPU_KERNEL(name, opKernel, T)                     \
  REGISTER_KERNEL_BUILDER(Name(WAVEFLOW_OP_NAME(name))                \
                              .Device(DEVICE_CPU)                     \
                              .TypeConstraint<T>("T"),                \
                          opKernel<waveflow::CPUDevice, T>);

// Convenience macro for gpu op kernel registration.
#define REGISTER_WF_GPU_KERNEL(name, opKernel, T)                     \
  REGISTER_KERNEL_BUILDER(Name(WAVEFLOW_OP_NAME(name))                \
                              .Device(DEVICE_GPU)                     \
                              .TypeConstraint<T>("T"),                \
                          opKernel<waveflow::GPUDevice, T>);

#endif //WAVEFLOW_REGISTER_KERNEL_H
