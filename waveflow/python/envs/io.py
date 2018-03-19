
class InputSource():
  """
  Input data source.
  """
  def __enter__(self):
    raise NotImplementedError("NYI")

  def __exit__(self, exc_type, exc_val, exc_tb):
    raise NotImplementedError("NYI")

  def fetch(self):
    """
    Returns next portion of available data
    """
    raise NotImplementedError("NYI")

class NumpyArraySource(InputSource):
  """
  Input mock source, that allows to access numpy array instance.

  This class is just a dummy mock, which every time returns the same array.
  """
  def __init__(self, array):
    self.array = array

  def __enter__(self):
    return self

  def __exit__(self, exc_type, exc_val, exc_tb):
    pass

  def fetch(self):
    return self.array