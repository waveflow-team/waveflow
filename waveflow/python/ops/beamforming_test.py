import unittest
import tensorflow as tf
from tensorflow.python.framework import errors_impl as tf_errors
import waveflow.python.ops.beamforming as beamforming
import waveflow.python.test_util as test_util
from waveflow.python.test_util import ParamTest
import numpy as np

class STATest(test_util.WaveFlowTestCase):

  def test_sta_with_focusing_double(self):
    tests = [
      ParamTest(
        name="test",
        params={
            #           t0, t1, t2, t3,
            "input": [[[0., 0., 0., 0.],
                       [0., 0., 0., 0.],
                       [0., 0., 0., 0.],
                       [0., 0., 0., 0.]],
                      [[0., 0., 1., 0.],
                       [0., 0., 1., 0.],
                       [0., 0., 1., 0.],
                       [0., 0., 1., 0.]]],
            "receiver_width": 1,
            "speed_of_sound": 1,
            "sampling_frequency": 1,
            "start_depth": 0,
            "output_shape": (4, 2),
        },
        expected=[[0.,0.],
                  [0.,1.71],
                  [0.,1.13],
                  [0.,0.]]
    ),
    ]
    self.run_test(func=beamforming.sta,
              tests=tests,
              assert_func= lambda x,y : np.testing.assert_allclose(x, y, rtol = 1e-2))

if __name__ == '__main__':
  unittest.main()

