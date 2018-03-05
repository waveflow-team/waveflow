import tensorflow as tf

_test_module = tf.load_op_library('test.so')

def test(x):
  return _test_module.test(x)