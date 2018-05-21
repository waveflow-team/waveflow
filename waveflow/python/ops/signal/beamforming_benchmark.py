import unittest
import tensorflow as tf
from tensorflow.python.framework import errors_impl as tf_errors
import waveflow.python.ops.beamforming as beamforming
from waveflow.python import test_util
from waveflow.python.test_util import ParamTest
import numpy as np

class STABenchmark(test_util.WaveFlowTestCase):

  def test_sta_with_focusing_double(self):
    print('after transpose')
    x = np.load('/home/pjarosik/sandbox/usg/sta/usg1_sta_nitki_reduced.npy')
    y = x.transpose((0, 2, 1))
    tests = [
      ParamTest(
        name="test",
        params={
            "input": x,
            "receiver_width": 64*.21e-3,
            "speed_of_sound": 1490.,
            "sampling_frequency": 50e6,
            "start_depth": 0.005,
            "output_height": 256,
            "output_width": 128,
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

