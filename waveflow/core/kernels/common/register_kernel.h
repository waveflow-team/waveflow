#ifndef WAVEFLOW_REGISTER_KERNEL_H
#define WAVEFLOW_REGISTER_KERNEL_H
#include "device.h"
#include "tensorflow/core/framework/op_kernel.h"

// Convenience macro for cpu op kernel registration.
#define REGISTER_CPU_KERNEL(name, opKernel, T)                        \
  REGISTER_KERNEL_BUILDER(Name(name)                                  \
                              .Device(DEVICE_CPU)                     \
                              .TypeConstraint<T>("T"),                \
                          opKernel<waveflow::CPUDevice, T>);

// Convenience macro for gpu op kernel registration.
#define REGISTER_GPU_KERNEL(name, opKernel, T)                        \
  REGISTER_KERNEL_BUILDER(Name(name)                                  \
                              .Device(DEVICE_GPU)                     \
                              .TypeConstraint<T>("T"),                \
                          opKernel<waveflow::GPUDevice, T>);

#endif //WAVEFLOW_REGISTER_KERNEL_H
