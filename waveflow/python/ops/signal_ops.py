import waveflow.python.ops.units as units
import tensorflow as tf
import waveflow.python.op_util as op_util

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
    with tf.control_dependencies(
      tf.assert_greater_equal(clip_value_max, clip_value_min,
                              data=[clip_value_min, clip_value_max],
                              message="must be: clip_value_max >= clip_value_min"),
      tf.assert_greater_equal(clip_value_max, 0, data=[clip_value_max],
                              message="clip value max must be >= 0 "),
      tf.assert_greater_equal(clip_value_min, 0, data=[clip_value_min],
                              message="clip value min must be >= 0 ")
    ):
      abs_input = tf.abs(input)
      ref_value = tf.maximum(abs_input)
      input_db = tf.cond(
        tf.equal(ref_value, 0),
        # We cannot use 0 as ref. value in dB scale.
        true_fn=lambda: abs_input,
        false_fn=lambda: units.to_decibel(abs_input, ref_value)
      )
      # input_db has non-positive values
      return input_db.clip_by_value(input_db, -1*clip_value_max, -1*clip_value_min)
