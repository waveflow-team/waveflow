//
// Created by pjarosik on 3/9/18.
//

#ifndef WAVEFLOW_DATA_TYPES_H
#define WAVEFLOW_DATA_TYPES_H
#include <tensorflow/core/platform/default/integral_types.h>

namespace waveflow {
 // We currently fully base on tensorflow data types;
typedef tensorflow::int8 int8;
typedef tensorflow::int16 int16;
typedef tensorflow::int32 int32;
typedef tensorflow::int64 int64;

typedef tensorflow::uint8 uint8;
typedef tensorflow::uint16 uint16;
typedef tensorflow::uint32 uint32;
typedef tensorflow::uint64 uint64;

}

#endif //WAVEFLOW_DATA_TYPES_H
