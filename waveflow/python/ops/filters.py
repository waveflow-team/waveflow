import waveflow.python.op_util as op_util

__module = op_util.load_op_library('waveflow/core/libfilters.so')

def fir(**kwargs):
  return __module.fir(**kwargs)




