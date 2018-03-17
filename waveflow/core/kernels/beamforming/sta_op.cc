#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/lib/core/errors.h"
#include "waveflow/core/common/device.h"
#include "waveflow/core/common/register_kernel.h"
#include "waveflow/core/common/tensor_utils.h"
#include "waveflow/core/kernels/beamforming/sta.h"

using namespace tensorflow;

namespace waveflow {

namespace functor {
extern template
struct STA<CPUDevice, float>;
extern template
struct STA<CPUDevice, double>;
extern template
struct STA<GPUDevice, float>;
extern template
struct STA<GPUDevice, double>;
}

template<typename Device, typename T>
class STAOpKernel : public OpKernel {
 public:
  explicit STAOpKernel(OpKernelConstruction *ctx) : OpKernel(ctx) {
    int output_height, output_width;
    OP_REQUIRES_OK(ctx, ctx->GetAttr("output_height", &output_height));
    OP_REQUIRES(ctx,
                output_height > 1,
                errors::InvalidArgument("output height should be > 1"));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("output_width", &output_width));
    OP_REQUIRES(ctx,
                output_width > 1,
                errors::InvalidArgument("output width should be > 1"));
    output_shape_ = TensorShape();
    output_shape_.AddDim(output_height);
    output_shape_.AddDim(output_width);
  }

  void Compute(OpKernelContext *ctx) override {
    const Tensor *input, *speedOfSoundSc,
        *receiverWidthSc, *samplingFrequencySc,
        *startDepthSc;

    GET_INPUT_TENSOR(ctx, "input", input);
    GET_INPUT_SCALAR(ctx, "speed_of_sound", speedOfSoundSc);
    GET_INPUT_SCALAR(ctx, "receiver_width", receiverWidthSc);
    GET_INPUT_SCALAR(ctx, "sampling_frequency", samplingFrequencySc);
    GET_INPUT_SCALAR(ctx, "start_depth", startDepthSc);

    const int64 eventsCount = input->dim_size(0);
    const int64 channelsCount = input->dim_size(1);
    const int64 samplesCount = input->dim_size(2);

    const float speedOfSound = get_scalar_value<float>(speedOfSoundSc);

    const float receiverWidth = get_scalar_value<float>(receiverWidthSc);
    const float
        samplingFrequency = get_scalar_value<float>(samplingFrequencySc);

    const float startDepth = get_scalar_value<float>(startDepthSc);
    float areaHeight = samplesCount * speedOfSound / samplingFrequency * 0.5f;

    OP_REQUIRES(ctx,
                input->dims() == 3,
                errors::InvalidArgument("Input must be 3-dimensional."));
    OP_REQUIRES(ctx,
                eventsCount <= output_shape_.dim_size(1),
                errors::InvalidArgument(
                    "Number of events can not be larger than the "
                        "output width."));

    Tensor *out = nullptr;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(0, output_shape_, &out));

    functor::STA<Device, T>()(
        ctx->eigen_device<Device>(),
        input->flat<T>().data(),
        channelsCount,
        samplesCount,
        speedOfSound,
        receiverWidth,
        samplingFrequency,
        startDepth,
        areaHeight,
        output_shape_.dim_size(0),
        output_shape_.dim_size(1),
        out->flat<T>().data()
    );
  }
 private:
  TensorShape output_shape_;
};
#define REGISTER_CPU(T) REGISTER_WF_CPU_KERNEL("Sta", STAOpKernel, T)
REGISTER_CPU(float)
REGISTER_CPU(double)
#undef REGISTER_CPU

#ifdef GOOGLE_CUDA
#define REGISTER_GPU(T) REGISTER_WF_GPU_KERNEL("Sta", STAOpKernel, T)
REGISTER_GPU(float)
REGISTER_GPU(double)
#undef REGISTER_GPU
#endif
}
