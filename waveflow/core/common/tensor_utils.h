#ifndef WAVEFLOW_TENSOR_UTILS_H
#define WAVEFLOW_TENSOR_UTILS_H
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"

using namespace tensorflow;

namespace waveflow {

// Loads input scalar with given name to buffer pointed by PTR.
// Should be called only within op kernel context (CTX must be a ptr to
// OpKernelContext object). Input with given name must be a scalar.
#define GET_INPUT_SCALAR(CTX, INPUT_NAME, PTR)                                 \
  do {                                                                   \
  OP_REQUIRES_OK((CTX), (CTX)->input(INPUT_NAME, &PTR));                 \
  OP_REQUIRES((CTX), TensorShapeUtils::IsScalar((PTR)->shape()),         \
              errors::InvalidArgument(INPUT_NAME " must be a scalar.")); \
  } while (0)

// Loads input scalar with given name to buffer pointed by PTR.
// Should be called only within op kernel context (CTX must be a ptr to
// OpKernelContext object). Input with given name must be a scalar.
#define GET_INPUT_VECTOR(CTX, INPUT_NAME, PTR)                                 \
  do {                                                                   \
  OP_REQUIRES_OK((CTX), (CTX)->input(INPUT_NAME, &PTR));                 \
  OP_REQUIRES((CTX), TensorShapeUtils::IsVector((PTR)->shape()),         \
              errors::InvalidArgument(INPUT_NAME " must be a vector.")); \
  } while (0)

// Loads input tensor with given name to buffer pointed by PTR.
// Should be called only within op kernel context (CTX must be a ptr to
// OpKernelContext object).
#define GET_INPUT_TENSOR(CTX, INPUT_NAME, PTR)                                 \
  OP_REQUIRES_OK((CTX), (CTX)->input(INPUT_NAME, &PTR));

template<typename T>
inline T get_scalar_value(const Tensor &scalar) {
  return *(scalar.scalar<T>().data());
}

template<typename T>
inline T get_scalar_value(const Tensor *scalar) {
  return *(scalar->scalar<T>().data());
}

}


#endif //WAVEFLOW_TENSOR_UTILS_H
