#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/common_shape_fns.h"
#include "waveflow/core/common/waveflow_op.h"

using namespace tensorflow;

namespace waveflow {
namespace beamforming {

REGISTER_WF_OP("Sta")
    .Input("input: T")
    .Input("speed_of_sound: float32")
    .Input("receiver_width: float32")
    .Input("sampling_frequency: float32")
    .Input("start_depth: float32")
    .Output("output: T")
    .Attr("output_height: int")
    .Attr("output_width: int")
    .Attr("T: {float, double, int32}")
    .SetShapeFn([](::tensorflow::shape_inference::InferenceContext *c) {
      int height, width;
      TF_RETURN_IF_ERROR(c->GetAttr("output_height", &height));
      TF_RETURN_IF_ERROR(c->GetAttr("output_width", &width));
      c->set_output(0, c->Matrix(height, width));
      return Status::OK();
    })
    .Doc(R"doc(
Synthetic Transmit Aperture imaging in ultrasound.
The operator is a composition of delay-and-sum method & image 1-d interpolation.

This op currently works only in 'FOCUSING' mode.

input: input echo tensor in ECS format (Event, Channel, Sample)
output_height: output's height (in pixels)
output_width: output's width (in pixels)

speed_of_sound: environments speed of sound
receiver_width: receiver's width (pitch * number of receivers), in meters
sampling_frequency: signal sampling frequency
start_depth: starting depth of imaged environment, in meters

output: ultrasound image (shape=output_height, output_width).
)doc");
}
}


