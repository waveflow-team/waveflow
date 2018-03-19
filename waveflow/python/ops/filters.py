import waveflow.python.op_util as op_util
import tensorflow as tf

__module = op_util.load_op_library('core/libfilters.so')

fir = __module.fir
