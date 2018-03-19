import waveflow.python.envs.physics as physics
import waveflow.python.envs.io as io


class Probe():
  """
  Ultrasound probe.
  """
  def __init__(self,
               name: str,
               num_elements: int,
               subaperture_size: int,
               pitch: float,
               sampling_frequency: float,
               input_source: io.InputSource):
    self.name = name,
    self.subaperture_size = subaperture_size
    self.num_elements = num_elements
    self.pitch = pitch
    self.sampling_frequency = sampling_frequency
    self.input_source = input_source


class UltrasoundEnv():
  """
  Ultrasound environment.
  """
  def __init__(self,
               probe: Probe,
               physical_env: physics.PhysicalEnv):
    # validate
    if physical_env.speed_of_sound is None:
      raise ValueError(
        "Speed of sound should be provided by physical env.")
    # assing
    self.probe = probe
    self.physical_env = physical_env

  def __enter__(self):
    self.probe.input_source.__enter__()
    return self

  def __exit__(self, *args):
    self.probe.input_source.__exit__(*args)

  def step(self):
    return self.probe.input_source.fetch()

  def get_probe(self):
    return self.probe

  def get_speed_of_sound(self):
    return self.physical_env.speed_of_sound

  def get_receiver_width(self):
    return self.probe.subaperture_size * self.probe.pitch
