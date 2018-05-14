"""
Functions for conversion between common SI units.
"""
import tensorflow as tf

from waveflow.python.ops import op_util
from . import math_ops

def to_decibel(x, ref_val, name=None):
  """
  Converts to decibel scale.

  :param x Tensor with values to convert to dB scale.
  :param ref_val Scalar representing reference value in bel scale
  :return: 10*log10(value/reference_value)
  """
  with tf.name_scope(name, op_util.resolve_op_name("ToDecibel"), [x]):
    zero = tf.constant(0, dtype=ref_val.dtype)
    with tf.control_dependencies([
      tf.assert_greater(ref_val, zero, data=[ref_val],
                        message="reference value must be > 0")
      ]):
      return 10*math_ops.log10(x / ref_val)

