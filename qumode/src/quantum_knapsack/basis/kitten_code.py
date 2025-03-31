from typing import List

import numpy as np

from .qumode_basis import QumodeBasis


class StandardBasis(QumodeBasis):
    """Standard computational basis implementation for quantum modes.

    This class represents the standard computational basis states in a quantum system,
    where each basis state is a column vector with a single 1 and zeros elsewhere.

    Attributes:
        _dimension (int): Dimension of the basis space
        _basis_states (List[np.ndarray]): List of basis state vectors
    """

    def __init__(self, dimension: int) -> None:
        """Initialize a standard basis with given dimension.

        Args:
            dimension (int): Dimension of the basis space

        Raises:
            ValueError: If dimension is less than 1
        """
        if dimension < 1:
            raise ValueError("Dimension must be a positive integer")

        super().__init__()
        self._dimension: int = dimension
        self._basis_states: List[np.ndarray] = []

        self._create_basis()
        self._create_ladder_operators()

    def _create_basis(self) -> None:
        """Create the standard basis states.

        Generates an orthonormal basis where each vector has a 1 in one position
        and 0s elsewhere.
        """
        standard_basis

        self._basis_states = []
        for i in range(self._dimension):
            state: np.ndarray = np.zeros((self._dimension, 1), dtype=complex)
            state[i, 0] = 1.0
            self._basis_states.append(state)

        self._create_matrix()
