from abc import ABC
from copy import deepcopy
from typing import Tuple

import numpy as np
from numpy.typing import NDArray

from src.quantum_knapsack.utils import pretty_format


class Matrix(ABC):
    """Abstract base class for all matrix types.
    
    Provides common matrix functionality and interface definitions.
    """

    def __init__(self) -> None:
        """Initialize an empty matrix."""
        self._matrix: NDArray[np.complex128] = np.zeros((0, 0), dtype=np.complex128)

    @property
    def matrix(self) -> NDArray[np.complex128]:
        """Get the underlying matrix data.
        
        Returns:
            NDArray[np.complex128]: Copy of the matrix data
        """
        return deepcopy(self._matrix)

    @property
    def shape(self) -> tuple[int, ...]:
        """Get the matrix dimensions.
        
        Returns:
            Tuple[int, int]: Matrix shape (rows, columns)
        """
        return self._matrix.shape

    def __repr__(self) -> str:
        """Return string representation for debugging."""
        return f"{self.__class__.__name__}(shape={self.shape})"

    def __str__(self) -> str:
        """Return string representation for printing."""
        return pretty_format(self._matrix)
