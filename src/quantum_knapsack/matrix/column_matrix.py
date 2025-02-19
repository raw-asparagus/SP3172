from typing import Union

import numpy as np
from numpy.typing import NDArray

from .matrix import Matrix


class ColumnMatrix(Matrix):
    """Matrix class specifically for column vectors.

    Provides specialized operations for quantum state vectors.
    """

    @staticmethod
    def normalise(state: NDArray[np.complex128]) -> NDArray[np.complex128]:
        """Normalize the state vector to unit length.

        Args:
            state: State vector to normalize

        Returns:
            NDArray[np.complex128]: Normalized state vector

        Raises:
            ValueError: If state is zero vector
        """
        norm = np.sqrt(np.abs(state.conj().T @ state)[0, 0])
        if np.isclose(norm, 0):
            raise ValueError("Cannot normalize zero vector")
        return state / norm

    @classmethod
    def from_array(cls, data: Union[NDArray[np.complex128], NDArray[np.float64]]) -> 'ColumnMatrix':
        """Create a ColumnMatrix from numpy array.

        Args:
            data: Input array data

        Returns:
            ColumnMatrix: New column matrix instance

        Raises:
            ValueError: If input is not a column vector
        """
        instance = cls()
        if data.ndim != 2 or data.shape[1] != 1:
            raise ValueError("Data must be a column vector (n×1 matrix)")
        instance._matrix = np.asarray(data, dtype=np.complex128)
        return instance
