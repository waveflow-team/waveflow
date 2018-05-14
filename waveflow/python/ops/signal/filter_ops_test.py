import unittest
import tensorflow as tf
from tensorflow.python.framework import errors_impl as tf_errors
import waveflow.python.ops.signal.filter_ops as filter_ops
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
    self.run_test(filter_ops.fir, tests)

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
    self.run_test(filter_ops.fir, tests, assert_func=np.testing.assert_array_equal)

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
    self.run_test(func=filter_ops.fir,
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
    self.run_test(func=filter_ops.fir,
                  tests=tests,
                  assert_func=np.testing.assert_allclose)

if __name__ == '__main__':
  unittest.main()
