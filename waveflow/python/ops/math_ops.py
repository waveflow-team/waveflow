import tensorflow as tf
import waveflow.python.op_util as op_util

def log(x, base, name=None):
  """
  Computes log_{base}(x)
  """
  with tf.name_scope(name, op_util.resolve_op_name("Log"), [x, base]):
    n = tf.log(x)
    base = tf.convert_to_tensor(base, dtype=n.dtype)
    d = tf.log(base)
    return n/d


def log10(x, name=None):
  """
  Computes log_10(x)

  numpy compatibility:
  Equivalent to numpy.log10
  """
  with tf.name_scope(name, op_util.resolve_op_name("Log10"), [x]):
    return log(x, 10)


def log2(x, name=None):
  """
  Computes log_2(x)

  numpy compatibility:
  Equivalent to numpy.log2
  """
  with tf.name_scope(name, op_util.resolve_op_name("Log2"), [x]):
    return log(x, 2)
