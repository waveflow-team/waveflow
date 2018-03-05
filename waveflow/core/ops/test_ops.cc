#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/common_shape_fns.h"

using namespace tensorflow;

namespace waveflow {

REGISTER_OP("TestOp") // TODO(pjarosik) remove it soon
    .Input("input: dtype")
    .Output("output: dtype")
    .Attr("dtype: {float, double}")
    .SetShapeFn(shape_inference::UnchangedShape)
    .Doc(R"doc(
     Test
)doc");


}


