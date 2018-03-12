#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/lib/core/errors.h"
#include "waveflow/core/kernels/filters/fir.h"
#include "waveflow/core/common/device.h"
#include "waveflow/core/common/register_kernel.h"

using namespace tensorflow;

namespace waveflow {

namespace functor {
extern template
struct FIRFilter<CPUDevice, float>;
extern template
struct FIRFilter<CPUDevice, double>;
extern template
struct FIRFilter<CPUDevice, int32>;
extern template
struct FIRFilter<GPUDevice, float>;
extern template
struct FIRFilter<GPUDevice, double>;
extern template
struct FIRFilter<GPUDevice, int32>;
}

template<typename Device, typename T>
class FIROpKernel : public OpKernel {
 public:
  explicit FIROpKernel(OpKernelConstruction *ctx) : OpKernel(ctx) {}

  void Compute(OpKernelContext *ctx) override {
    const Tensor &in = ctx->input(0);
    const Tensor &filter = ctx->input(1);
    int64 axisSize = in.dim_size(in.dims() - 1);

    OP_REQUIRES(ctx,
                TensorShapeUtils::IsVectorOrHigher(in.shape()),
                errors::InvalidArgument("Input cannot be a scalar."));
    OP_REQUIRES(ctx,
                TensorShapeUtils::IsVector(filter.shape()),
                errors::InvalidArgument("Filter must be a 1-D vector."));
    OP_REQUIRES(ctx,
                filter.dim_size(0) > 0,
                errors::InvalidArgument("Filter cannot be empty."));
    OP_REQUIRES(ctx,
                axisSize >= filter.NumElements(),
                errors::InvalidArgument(
                    "Input's last dimension size must be >= filter length."));
    Tensor *out = nullptr;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(0, in.shape(), &out));
    functor::FIRFilter<Device, T>()(
        ctx->eigen_device<Device>(),
        in.flat<T>().data(),
        in.NumElements(),
        axisSize,
        filter.flat<T>().data(),
        filter.NumElements(),
        out->flat<T>().data());

  }
};
#define REGISTER_CPU(T) REGISTER_WF_CPU_KERNEL("Fir", FIROpKernel, T)
REGISTER_CPU(float)
REGISTER_CPU(double)
REGISTER_CPU(int32)
#undef REGISTER_CPU

#ifdef GOOGLE_CUDA
#define REGISTER_GPU(T) REGISTER_WF_GPU_KERNEL("Fir", FIROpKernel, T)
REGISTER_GPU(float)
REGISTER_GPU(double)
REGISTER_GPU(int32)
#undef REGISTER_GPU
#endif
}
