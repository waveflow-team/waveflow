#ifndef WAVEFLOW_WAVEFLOW_OP_H
#define WAVEFLOW_WAVEFLOW_OP_H
#include "tensorflow/core/framework/op.h"
#include <string>

// This header contains code common for all waveflow CC ops.

// Name prefix common to all waveflow ops.
#define _WAVEFLOW_OP_PREFIX "Waveflow"

// Creates new const string containing proper waveflow op name.
#define WAVEFLOW_OP_NAME(name) _WAVEFLOW_OP_PREFIX name

// Use this macro to register new waveflow op.
// Using this macro you protect your code against possible name clash with
// other ops registered in tensorflow.
#define REGISTER_WF_OP(name) REGISTER_OP(WAVEFLOW_OP_NAME(name))




#endif //WAVEFLOW_WAVEFLOW_OP_H

