import unittest
import tensorflow as tf
from tensorflow.python.framework import errors_impl as tf_errors
import waveflow.python.ops.filters as filters
import waveflow.python.test_util as test_util
from waveflow.python.test_util import ParamTest
import numpy as np
import scipy.fftpack


class FIRFilterTest(test_util.WaveFlowTestCase):

  def test_fir_validation(self):
    """Checks ops validation procedure."""
    tests = [
      ParamTest(
        name="scalar_input",
        params={
          "input": 42,
          "filter": [1, 2, 3]
        },
        exception=tf_errors.InvalidArgumentError
      ),
      ParamTest(
        name="empty_filter",
        params={
          "input": [1, 2, 3],
          "filter": []
        },
        exception=tf_errors.InvalidArgumentError
      ),
      ParamTest(
        name="scalar_filter",
        params={
          "input": [1, 2, 3],
          "filter": 1
        },
        exception=tf_errors.InvalidArgumentError
      ),
      ParamTest(
        name="2d_filter",
        params={
          "input": [1, 2, 3],
          "filter": [[1, 2],
                     [3, 4]]
        },
        exception=tf_errors.InvalidArgumentError
      ),
      ParamTest(
        name="4d_filter",
        params={
          "input": [1, 2, 3],
          "filter": [[[[1]], [[2]]],
                     [[[3]], [[4]]]]
        },
        exception=tf_errors.InvalidArgumentError
      ),
      ParamTest(
        name="filter_larger_than_input_vector",
        params={
          "input": [1, 2],
          "filter": [1, 2, 3]
        },
        exception=tf_errors.InvalidArgumentError
      ),
      ParamTest(
        name="filter_larger_than_2d_input_axis",
        params={
          "input": [[1, 2, 3],
                    [4, 5, 6]],
          "filter": [1, 2, 3, 4]
        },
        exception=tf_errors.InvalidArgumentError
      ),
    ]
    self.run_test(filters.fir, tests)

  def test_fir_int32_pass_edge_cases(self):
    tests = [
      ParamTest(
        name="singular_input_singular_filter",
        params={
          "input": [2],
          "filter": [3]
        },
        expected=[6]
      ),
      ParamTest(
        name="singular_filter",
        params={
          "input": [1, 2],
          "filter": [3]
        },
        expected=[3, 6]
      ),
      ParamTest(
        name="input_and_filter_same_size",
        params={
          "input": [1, 2],
          "filter": [3, 4]
        },
        expected=[3, 10]
      ),
    ]
    self.run_test(filters.fir, tests, assert_func=np.testing.assert_array_equal)

  def test_fir_int32_works_for_n_dims(self):
    """ FIR should work for n > 1 dimensional input tensors,
        and always compute conv. (left-padded with zeros), along n-th axis.
    """
    tests = [
      ParamTest(
        name="vector_5",
        params={
          "input": [1, 2, 3, 5, 8],
          "filter": [3, 1, 4]
        },
        expected=[3, 7, 15, 26, 41]
      ),
      ParamTest(
        name="matrix_2x4",
        params={
          "input": [[3, 12, 4, 17],
                    [9, 1, 13, 7]],
          "filter": [8, 7, 1]
        },
        expected=[[24, 117, 119, 176],
                  [72, 71, 120, 148]]
      ),

      ParamTest(
        name="cube_2x2x3",
        params={
          "input": [[[1, 2, 3],
                     [7, 8, 9]],
                    [[4, 5, 6],
                     [0, 1, 2]]],
          "filter": [3, 13]
        },
        expected=[[[3, 19, 35],
                   [21, 115, 131]],
                  [[12, 67, 83],
                   [0, 3, 19]]]
      ),
    ]
    self.run_test(func=filters.fir,
                  tests=tests,
                  assert_func=np.testing.assert_array_equal)

  def test_fir_double(self):
    tests = [
      ParamTest(
        name="vector_4",
        params={
          "input": [1.1, 2, 3.3, 4.4],
          "filter": [2., 3.1, 4.]
        },
        expected=[2.2, 7.41, 17.2, 27.03]
      ),
    ]
    self.run_test(func=filters.fir,
                  tests=tests,
                  assert_func=np.testing.assert_allclose)


class FFTFreqTest(test_util.WaveFlowTestCase):
  def test_fftfreq(self):
    tests = [
      ParamTest(
        name="even",
        params={
          "n": 8,
        },
        expected=np.divide([0, 1, 2, 3, -4, -3, -2, -1], 8)
      ),
      ParamTest(
        name="odd",
        params={
          "n": 7
        },
        expected=np.divide([0, 1, 2, 3, -3, -2, -1], 7)
      ),
      ParamTest(
        name="even_with_step",
        params={
          "n": 4,
          "d": .5
        },
        expected=np.divide([0, 1, -2, -1], 4 * .5)
      ),
      ParamTest(
        name="odd_with_step",
        params={
          "n": 9,
          "d": .2
        },
        expected=np.divide([0, 1, 2, 3, 4, -4, -3, -2, -1], 9 * .2)
      ),
    ]
    self.run_test(filters.fftfreq, tests,
                  assert_func=np.testing.assert_allclose)

  def test_fftfreq_edge_cases(self):
    tests = [
      ParamTest(
        name="zero",
        params={
          "n": 0,
        },
        exception=tf_errors.InvalidArgumentError
      ),
      ParamTest(
        name="one",
        params={
          "n": 1
        },
        exception=tf_errors.InvalidArgumentError
      ),
      ParamTest(
        name="two",
        params={
          "n": 2
        },
        expected=[0, -.5],
      )
    ]
    self.run_test(filters.fftfreq, tests,
                  assert_func=np.testing.assert_allclose)

  def test_fftfreq_numpy_equivalence(self):
    """
    Checks, if given implementation is equivalent to np.fft.fftfreq.
    """
    tests = [
      ParamTest(
        name="even",
        params={
          "n": 10,
        },
        expected=np.fft.fftfreq(n=10)
      ),
      ParamTest(
        name="odd",
        params={
          "n": 13
        },
        expected=np.fft.fftfreq(n=13)
      ),
      ParamTest(
        name="even_with_step",
        params={
          "n": 12,
          "d": .3
        },
        expected=np.fft.fftfreq(n=12, d=.3)
      ),
      ParamTest(
        name="odd_with_step",
        params={
          "n": 15,
          "d": .2
        },
        expected=np.fft.fftfreq(n=15, d=.2)
      ),
    ]
    self.run_test(filters.fftfreq, tests,
                  assert_func=np.testing.assert_allclose)

    class FFTFreqTest(test_util.WaveFlowTestCase):
      def test_fftfreq(self):
        tests = [
          ParamTest(
            name="even",
            params={
              "n": 8,
            },
            expected=np.divide([0, 1, 2, 3, -4, -3, -2, -1], 8)
          ),
          ParamTest(
            name="odd",
            params={
              "n": 7
            },
            expected=np.divide([0, 1, 2, 3, -3, -2, -1], 7)
          ),
          ParamTest(
            name="even_with_step",
            params={
              "n": 4,
              "d": .5
            },
            expected=np.divide([0, 1, -2, -1], 4 * .5)
          ),
          ParamTest(
            name="odd_with_step",
            params={
              "n": 9,
              "d": .2
            },
            expected=np.divide([0, 1, 2, 3, 4, -4, -3, -2, -1], 9 * .2)
          ),
        ]
        self.run_test(filters.fftfreq, tests,
                      assert_func=np.testing.assert_allclose)

      def test_fftfreq_edge_cases(self):
        tests = [
          ParamTest(
            name="zero",
            params={
              "n": 0,
            },
            exception=tf_errors.InvalidArgumentError
          ),
          ParamTest(
            name="one",
            params={
              "n": 1
            },
            exception=tf_errors.InvalidArgumentError
          ),
          ParamTest(
            name="two",
            params={
              "n": 2
            },
            expected=[0, -.5],
          )
        ]
        self.run_test(filters.fftfreq, tests,
                      assert_func=np.testing.assert_allclose)

      def test_fftfreq_numpy_equivalence(self):
        """
        Checks, if given implementation is equivalent to np.fft.fftfreq.
        """
        tests = [
          ParamTest(
            name="even",
            params={
              "n": 10,
            },
            expected=np.fft.fftfreq(n=10)
          ),
          ParamTest(
            name="odd",
            params={
              "n": 13
            },
            expected=np.fft.fftfreq(n=13)
          ),
          ParamTest(
            name="even_with_step",
            params={
              "n": 12,
              "d": .3
            },
            expected=np.fft.fftfreq(n=12, d=.3)
          ),
          ParamTest(
            name="odd_with_step",
            params={
              "n": 15,
              "d": .2
            },
            expected=np.fft.fftfreq(n=15, d=.2)
          ),
        ]
        self.run_test(filters.fftfreq, tests,
                      assert_func=np.testing.assert_allclose)


class HilbertTest(test_util.WaveFlowTestCase):

  def test_hilbert_common_functions(self):
    """
      Checks hilbert's correctness for common, simple functions.
    :return:
    """
    tests = [
      ParamTest(
        name="sin_should_give_minus_cos",
        params={
          "input": np.sin(np.arange(0, 6.28, step=.001)),
          "dt": .001
        },
        expected=-1 * np.cos(np.arange(0, 6.28, step=.001))
      ),
      ParamTest(
        name="cos_should_give_sin",
        params={
          "input": np.cos(np.arange(0, 6.28, step=.0001)),
          "dt": .0001
        },
        expected=np.sin(np.arange(0, 6.28, step=.0001))
      ),
    ]
    self.run_test(lambda input, dt: tf.real(filters.hilbert(input, dt)),
                  tests,
                  assert_func=lambda a, d: np.testing.assert_allclose(a, d,
                                                                      atol=1e-2))

  def test_hilbert_computes_1d_along_last_axis(self):
    """
    Checks hilbert's correcetness for 2d tensor.
    """
    tests = [
      ParamTest(
        name="input_sin_cos_expected_minus_cos_sin",
        params={
          "input": np.asarray(
            [
              np.sin(np.arange(0, 6.28, step=.001)),
              np.cos(np.arange(0, 6.28, step=.001))
            ]
          ),
          "dt": .001
        },
        expected=np.asarray(
          [
            -1 * np.cos(np.arange(0, 6.28, step=.001)),
            np.sin(np.arange(0, 6.28, step=.001))
          ]
        )
      )
    ]
    self.run_test(lambda input, dt: tf.real(filters.hilbert(input, dt)),
                  tests,
                  assert_func=lambda a, d: np.testing.assert_allclose(a, d,
                                                                      atol=1e-2))


  def test_hilbert_scipy_equivalence(self):
    """
    Checks equivalance with tf.fftpack.hilbert()
    """
    tests = [
      ParamTest(
        name="sin",
        params={
          "input": np.sin(np.arange(0, 6.28, step=.001)),
          "dt": .001
        },
        expected=-1 * scipy.fftpack.hilbert(
          np.sin(np.arange(0, 6.28, step=.001)),
        )
      ),
      ParamTest(
        name="cos",
        params={
          "input": np.cos(np.arange(0, 6.28, step=.001)),
          "dt": .001
        },
        expected=-1 * scipy.fftpack.hilbert(
          np.cos(np.arange(0, 6.28, step=.001))
        )
      ),
    ]
    self.run_test(lambda input, dt: tf.real(filters.hilbert(input, dt)),
                  tests,
                  assert_func=lambda a, d: np.testing.assert_allclose(a, d,
                                                                      atol=1e-2))


if __name__ == '__main__':
  unittest.main()
