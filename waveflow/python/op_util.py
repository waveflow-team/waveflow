import tensorflow as tf

# Prefix, which should be removed from ops name
_WAVEFLOW_PREFIX = "waveflow_"


def load_op_library(path):
  """
  Loads given waveflow module.

  Currently, THIS IS THE ONLY RECOMMENDED WAY TO ACCESS WAVEFLOW MODULES.
  Accessing tensorflow modules directly is risky and may not work in future.

  :param path: path to op library
  :return: waveflow module
  """
  module = tf.load_op_library(path)
  return wrap_module(module)


def wrap_module(module):
  """
  Creates waveflow module from given tensorflow module.

  Currently THIS IS THE ONLY RECOMMENDED WAY TO ACCESS WAVEFLOW MODULES.
  Accessing tensorflow modules directly is risky and may not work in future.

  :param module: tensorflow module to wrap
  :return: waveflow module
  """
  return WaveflowModule(module)


class WaveflowModule:
  """
  Simple wrapper over tensorflow module, which allows to access
  waveflow ops in convenient way.
  """

  def __init__(self, module):
    self._tf_module = module
    for tf_op_name in dir(module):
      op = getattr(module, tf_op_name)
      if callable(op) and tf_op_name.startswith(_WAVEFLOW_PREFIX):
        wf_op_name = tf_op_name[len(_WAVEFLOW_PREFIX):]
        setattr(self, wf_op_name, op)
