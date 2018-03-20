"""
Module providing function for conversion between SI units.
"""
import tensorflow as tf
import waveflow.python.ops.math_ops as math_ops
import waveflow.python.op_util as op_util

def to_decibel(x, ref_val, name=None):
  """
  Converts to decibel scale.

  :param x Tensor with values to convert to dB scale.
  :param ref_val Scalar representing reference value in bel scale
  :return: 10*log10(value/reference_value)
  """
  with tf.name_scope(name, op_util.resolve_op_name("Bel"), [x]):
    with tf.control_dependencies(
      tf.assert_greater(ref_val, 0, data=[ref_val],
                        message="reference value must be > 0")):
      return 10*math_ops.log10(x / ref_val)

