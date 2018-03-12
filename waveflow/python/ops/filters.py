import tensorflow as tf

__module = tf.load_op_library('waveflow/core/libfilters.so')

def fir(**kwargs):
  return __module.waveflow_fir(**kwargs)




