import tensorflow as tf
from tensorflow.python.framework.test_util import TensorFlowTestCase


class ParamTest:
  """ Single parametric test."""

  def __init__(self, name, params, expected = None, exception = None):
    """
    :param name: name of parametric test
    :param params: dictionary of kwargs for tested function
    :param expected: expected value
    :param exception: expected exception
    """
    assert (expected is None) ^ (exception is None)
    self.name = name,
    self.params = params
    self.expected = expected
    self.exception = exception

class WaveFlowTestCase(TensorFlowTestCase):
  """
  Wrapper over tensorflow test case class.
  This class should be used in every waveflow test case.

  Default settings:
  - if gpu is available, use it (you can turn off gpu tests by excluding
    config=cuda from bazel call parameters)
  - log device placement TODO(pjarosik) just for convenience, remove soon
  """

  def test_session(self,
                   graph=None,
                   config=tf.ConfigProto(log_device_placement=True),
                   use_gpu=True,
                   force_gpu=False):
    return super(WaveFlowTestCase, self).test_session(
      graph, config, use_gpu, force_gpu)

  def run_test(self, func, tests, assert_func=None):
    """
    :param func: tested function.
    :param tests: list of subtests to run.
    :param assert_func: assertion to call. Ignored, when test expects exception.
    :return:
    """
    with self.test_session():
      for test in tests:
        with self.subTest(test.name):
          kwargs = test.params
          # TODO(pjarosik) make more detailed Exception (or OpError) check below
          if test.exception is not None:
            with self.assertRaises(test.exception):
              func(**kwargs).eval()
          else:
            result = func(**kwargs).eval()
            assert_func(result, test.expected)
