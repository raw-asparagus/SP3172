from abc import ABC
from copy import deepcopy

import numpy as np
from numpy.typing import NDArray

from .basis import Basis


class QumodeBasis(Basis, ABC):
    """
    Abstract base class for quantum mode basis implementations with ladder operators.

    This class extends the basic Basis with creation and annihilation operators
    commonly used in quantum optics and quantum mechanics.
    """

    def __init__(self) -> None:
        """Initialize the quantum mode basis with ladder operators."""
        super().__init__()
        self._creation_operator: NDArray[np.complex128] = np.zeros((self.dimension, self.dimension), dtype=np.complex128)
        self._annihilation_operator: NDArray[np.complex128] = np.zeros((self.dimension, self.dimension), dtype=np.complex128)

    def _create_ladder_operators(self) -> None:
        """
        Create the ladder operators (creation and annihilation) for the quantum mode.

        The creation operator increases the quantum number by 1,
        while the annihilation operator decreases it by 1.
        """
        # Initialize operators with proper size
        shape = (self._dimension, self._dimension)
        self._creation_operator = np.zeros(shape, dtype=np.complex128)
        self._annihilation_operator = np.zeros(shape, dtype=np.complex128)

        # Create creation operator
        n_indices = np.arange(self._dimension - 1)
        self._creation_operator[n_indices + 1, n_indices] = np.sqrt(n_indices + 1)

        # Create annihilation operator
        n_indices = np.arange(1, self._dimension)
        self._annihilation_operator[n_indices - 1, n_indices] = np.sqrt(n_indices)

    @property
    def creation_operator(self) -> NDArray[np.complex128]:
        """Get the creation operator matrix."""
        if not self._creation_operator.any():
            raise RuntimeError("Ladder operators have not been initialized.")
        return deepcopy(self._creation_operator)

    @property
    def annihilation_operator(self) -> NDArray[np.complex128]:
        """Get the annihilation operator matrix."""
        if not self._annihilation_operator.any():
            raise RuntimeError("Ladder operators have not been initialized.")
        return deepcopy(self._annihilation_operator)
