import unittest
import waveflow.python.test_util as test_util
from waveflow.python.test_util import ParamTest
import waveflow.python.ops.math_ops as math_ops
import numpy as np
import math as math

_RTOL = 1e-5
_assert_func = lambda x,y: np.testing.assert_allclose(x, y, rtol = _RTOL)

class MathTest(test_util.WaveFlowTestCase):

  def test_log_python_math_equivalance(self):
    x0_range = np.arange(.01, 1, step=.01).tolist()
    x_range = np.arange(1, 10e3, step=1).tolist()
    tests = [
      ParamTest(
        name="base 3, range[.01, 1)",
        params={
          "x": x0_range,
          "base": 3
        },
        expected=[math.log(x, 3) for x in x0_range]
      ),
      ParamTest(
        name="base 3, range[1, 10e3)",
        params={
          "x": x_range,
          "base": 3
        },
        expected=[math.log(x, 3) for x in x_range]
      ),
      ParamTest(
        name="base 123, range[.01, 1)",
        params={
          "x": x0_range,
          "base": 123
        },
        expected=[math.log(x, 123) for x in x0_range]
      ),
      ParamTest(
        name="base 123, range[1, 10e3)",
        params={
          "x": x_range,
          "base": 123
        },
        expected=[math.log(x, 123) for x in x_range]
      ),
    ]
    self.run_test(func=math_ops.log, tests=tests, assert_func=_assert_func)

  def test_log10_numpy_equivalance(self):
    x0_range = np.arange(.01, 1, step = .01)
    x_range = np.arange(1, 10e3, step=1)
    tests = [
      ParamTest(
        name="range [.01, 1)",
        params={
          "x": x0_range
        },
        expected=np.log10(x0_range)
      ),
      ParamTest(
        name="range [1, 10e3)",
        params={
          "x": x_range
        },
        expected=np.log10(x_range)
      ),
    ]
    self.run_test(func=math_ops.log10, tests=tests, assert_func=_assert_func)

  def test_log2_numpy_equivalance(self):
    x0_range = np.arange(.01, 1, step=.01)
    x_range = np.arange(1, 10e3, step=1)
    tests = [
      ParamTest(
        name="range [.01, 1)",
        params={
          "x": x0_range
        },
        expected=np.log2(x0_range)
      ),
      ParamTest(
        name="range [1, 10e3)",
        params={
          "x": x_range
        },
        expected=np.log2(x_range)
      ),
    ]
    self.run_test(func=math_ops.log2, tests=tests, assert_func=_assert_func)


if __name__ == '__main__':
  unittest.main()
