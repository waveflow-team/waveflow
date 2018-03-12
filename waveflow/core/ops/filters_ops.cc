#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/common_shape_fns.h"

using namespace tensorflow;

namespace waveflow {
namespace filters {

REGISTER_OP("Fir")
    .Input("input: T")
    .Input("filter: T")
    .Output("output: T")
    .Attr("T: {float, double, int32}")
    .SetShapeFn(shape_inference::UnchangedShape)
    .Doc(R"doc(
Finite Impulse Response Filter.

Requires, that:
- the size of computation axis >= filter size

How it works:
- computes true 1D conv along the last (-1) axis
 (i.e. if input is a signal tensor with dims ECS (Event, Channel, Sample),
 FIR will be applied to each sample vector);
- supports only left-side padding (to keep unchaged shape)
)doc");
}
}


