import tensorflow as tf
import waveflow.python.op_util as op_util


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


def hilbert(input, dt=1, axis=None, name=None):
  """
  Computes Hilbert transform of complex-valued signal over the inner-most
  dimension of input. Computations will be done along given axis.

  Disclaimer:
  This function currently does not support tensors with undefined dimensions.

  scipy compatibility
  Equivalent to (-1)*scipy.fftpack.hilbert(input)

  :param input: tensor of tf.complex64 values
  :param dt: time step between consecutive samples
  :param axis input's dimension
  :return: tensor with values of type tf.complex64
  """
  with tf.name_scope(name, op_util.resolve_op_name("Hilbert"), [input]):
    input = tf.convert_to_tensor(value=input, name="input", dtype=tf.complex64)
    # TODO(pjarosik) axis parameter should be handled by tf.fft
    if axis is not None:
      rank = len(input.get_shape().as_list())
      assert 0 <= axis < rank
      perm = [x for x in range(0, rank)]
      perm[axis] = rank - 1
      perm[rank - 1] = axis
      input = tf.transpose(input, perm=perm)

    input_shape = tf.shape(input)
    input_rank = tf.rank(input)
    with tf.control_dependencies([
      tf.assert_greater(input_rank, 0, data=[input],
                        message="input can not be a scalar")
    ]):
      n = input_shape[input_rank - 1]
      y_f = tf.fft(input)
      f = tf.cast(fftfreq(n, d=dt), dtype=tf.complex64)
      output = tf.real(tf.ifft(-1j * tf.sign(f) * y_f))
      if axis is not None:
        output = tf.transpose(output, perm = perm)
      return output


def fftfreq(n, d=1.0, name=None):
  """
  Returns tf.fft sample frequencies.

  numpy compatibility
  Equivalent to np.fft.fftfreq

  :param n: 0-D tensor - number of samples of signal in time-domain
  :param d: value - sample spacing
  :return: 1-D tensor of length n with sample frequencies.
  """
  with tf.name_scope(name, op_util.resolve_op_name("Fftfreq"), [n]):
    n = tf.cast(n, dtype=tf.float32)
    with tf.control_dependencies([
      tf.assert_equal(tf.rank(n), 0, data=[n], message="n must be a scalar."),
      tf.assert_greater_equal(n, 2., data=[n], message="n must be > 2")
    ]):
      T = d * n
      dtype = tf.float32
      non_negative_range = tf.cond(
        tf.equal(tf.mod(n, 2), 0),
        true_fn=lambda: tf.range(start=0, limit=n // 2, dtype=dtype),
        false_fn=lambda: tf.range(start=0, limit=((n - 1) // 2) + 1,
                                  dtype=dtype)
      )
      negative_range = tf.cond(
        tf.equal(tf.mod(n, 2), 0),
        true_fn=lambda: tf.range(start=-1 * (n // 2), limit=0, dtype=dtype),
        false_fn=lambda: tf.range(start=-1 * ((n - 1) // 2), limit=0,
                                  dtype=dtype)
      )
      return tf.concat(
        [tf.div(non_negative_range, T), tf.div(negative_range, T)], -1)


