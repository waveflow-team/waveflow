"""
Common functions for signal processing.
"""

import tensorflow as tf
import waveflow as wf
from waveflow.python.ops import op_util
from waveflow import hilbert

def analytic_signal(input, dt=1, axis=None, name=None):
  """
  Computes the analytic signal:
  x_a = x + j*h(x)
  Where x is the input signal, h(x) - the Hilbert transform of x.

  For more information see:
  https://en.wikipedia.org/wiki/Analytic_signal

  scipy compatibility
  Equivalent to scipy.signal.hilbert(input)
  """
  with tf.name_scope(name, op_util.resolve_op_name("AnalyticSignal"), [input]):
    return input + 1j * tf.cast(hilbert(input, dt=dt, axis=axis), dtype=tf.complex64)


def clip_by_decibel(input, clip_value_min, clip_value_max, name=None):
  """
  Converts to dB scale and limits dynamic range of input signal.

  :param range pair (min, max) of allowed dB values. Each output value will be
         limited to this range, i.e.:
          if result[i] > max => result[i] = max
          if result[i] < min => result[i] = min
  :return: input in dB scale, with values limited to given range
  """
  with tf.name_scope(name, op_util.resolve_op_name("ClipByDecibel"),
                     [input, clip_value_min, clip_value_max]):
    clip_value_max = tf.convert_to_tensor(clip_value_max, dtype=input.dtype)
    clip_value_min = tf.convert_to_tensor(clip_value_min, dtype=input.dtype)
    zero = tf.constant(0, dtype=input.dtype)
    with tf.control_dependencies([
      tf.assert_greater_equal(clip_value_max, clip_value_min,
                              data=[clip_value_min, clip_value_max],
                              message="must be: clip_value_max >= clip_value_min"),
      tf.assert_greater_equal(clip_value_max, zero, data=[clip_value_max],
                              message="clip value max must be >= 0 "),
      tf.assert_greater_equal(clip_value_min, zero, data=[clip_value_min],
                              message="clip value min must be >= 0 ")
      ]):
      abs_input = tf.abs(input)
      ref_value = tf.reduce_max(abs_input)
      input_db = tf.cond(
        tf.equal(ref_value, zero),
        # We cannot use 0 as ref. value in dB scale.
        true_fn=lambda: abs_input,
        false_fn=lambda: wf.to_decibel(abs_input, ref_value)
      )
      # input_db has non-positive values
      return tf.clip_by_value(input_db, -1*clip_value_max, -1*clip_value_min)
