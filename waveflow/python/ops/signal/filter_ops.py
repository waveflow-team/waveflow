from waveflow.python.ops import op_util

__module = op_util.load_op_library('core/libfilters.so')

fir = __module.fir
