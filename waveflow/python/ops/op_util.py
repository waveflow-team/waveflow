import tensorflow as tf
import os
import waveflow as _wf

# Prefix, which should be removed from ops name
_WAVEFLOW_PREFIX = "waveflow_"
_WAVEFLOW_CAMELCASE_PREFIX="Waveflow"


def load_op_library(op_lib_name):
  """
  Loads given waveflow module.

  Currently, THIS IS THE ONLY RECOMMENDED WAY TO ACCESS WAVEFLOW MODULES.
  Accessing tensorflow modules directly is risky and may not work in future.

  :param op_lib_name: path to op library, relative to 'waveflow' directory
  :return: waveflow module
  """
  module = tf.load_op_library(_resolve_op_path(op_lib_name))
  return wrap_module(module)

def _resolve_op_path(op_path):
  """
  Resolves absolute path to given op.

  :param op_path op path relative to 'waveflow' directory
  :return: absolute path to waveflow op
  """
  dir_path = os.path.dirname(_wf.__file__)
  return os.path.join(dir_path, op_path)


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

def resolve_op_name(name):
  return _WAVEFLOW_CAMELCASE_PREFIX + name
