import waveflow.python.op_util as op_util

__module = op_util.load_op_library('waveflow/core/libbeamforming.so')

def sta(**kwargs):
  return __module.sta(**kwargs)




