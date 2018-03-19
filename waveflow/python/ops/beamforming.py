import waveflow.python.op_util as op_util

__module = op_util.load_op_library('core/libbeamforming.so')

def sta(input,
        output_shape,
        start_depth = 0.005,
        us_env = None,
        name=None,
        **kwargs):
  if not ((us_env is not None) ^ bool(kwargs)):
    raise ValueError("Exactly one of these must be provided: us_env, kwargs")
  if us_env is not None:
    kwargs = {
      'speed_of_sound': us_env.get_speed_of_sound(),
      'receiver_width': us_env.get_receiver_width(),
      'sampling_frequency': us_env.probe.sampling_frequency,
    }
  kwargs['output_height'] = output_shape[0]
  kwargs['output_width'] = output_shape[1]
  kwargs['start_depth'] = start_depth
  kwargs['input'] = input
  return __module.sta(**kwargs)




