#ifndef WAVEFLOW_REGISTER_KERNEL_H
#define WAVEFLOW_REGISTER_KERNEL_H
#include "device.h"
#include "tensorflow/core/framework/op_kernel.h"

// Convenience macro for cpu op kernel registration.
#define REGISTER_CPU_KERNEL(name, opKernel, dtype)                    \
  REGISTER_KERNEL_BUILDER(Name(name)                                  \
                              .Device(DEVICE_CPU)                     \
                              .TypeConstraint<dtype>("dtype"),        \
                          opKernel<waveflow::CPUDevice, dtype>);


#endif //WAVEFLOW_REGISTER_KERNEL_H
