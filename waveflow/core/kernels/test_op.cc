#include "test.h"
#include "device.h"
#include "register_kernel.h"
#include "tensorflow/core/framework/op_kernel.h"

using namespace tensorflow;

namespace waveflow {

namespace functor {
extern template struct Test<CPUDevice, float>;
extern template struct Test<CPUDevice, double>;
}

template <typename Device, typename T>
class TestOpKernel: public OpKernel {
 public:
  TestOpKernel(OpKernelConstruction *ctx) : OpKernel(ctx) {}

  void Compute(OpKernelContext *ctx) override {
    const Tensor& in = ctx->input(0);

    Tensor* out = nullptr;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(0, in.shape(), &out));

    functor::Test<Device, T>()(
        ctx->eigen_device<Device>(),
        static_cast<int>(in.NumElements()),
        in.flat<T>().data(),
        out->flat<T>().data());
  }
};
#define REGISTER_CPU(dtype) REGISTER_CPU_KERNEL("TestOp", TestOpKernel, dtype)
REGISTER_CPU(float)
REGISTER_CPU(double)
#undef REGISTER_CPU
}
