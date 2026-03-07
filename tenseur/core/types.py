"""Type aliases for the tenseur library."""

from typing import Union

import numpy as np
from numpy.typing import NDArray

Tensor = NDArray[np.float64]
TensorSize = Union[int, tuple[int, int], tuple[int, int, int]]
PitchSet = list[int]
