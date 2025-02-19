from abc import ABC, abstractmethod
from typing import List

import numpy as np
from numpy.typing import NDArray

from src.quantum_knapsack.utils import pretty_format


class Basis(ABC):
    """Abstract base class for quantum basis implementations."""

    def __init__(self) -> None:
        """Initialize the basis with zero dimension."""
        self._dimension: int = 0
        self._basis_states: List[NDArray[np.complex128]] = [
            np.zeros((self._dimension, 1), dtype=np.complex128)
        ]
        self._basis_matrix: NDArray[np.complex128] = np.zeros(
            (self._dimension, self._dimension), dtype=np.complex128
        )

    @abstractmethod
    def _create_basis(self) -> None:
        """Create the basis states. Must be implemented by subclasses."""
        pass

    def _create_matrix(self) -> None:
        """Create the basis matrix from basis states."""
        self._basis_matrix = np.column_stack([state.flatten() for state in self._basis_states])

    @property
    def basis_states(self) -> List[NDArray[np.complex128]]:
        """Get the list of basis states."""
        return self._basis_states

    def get_basis_state(self, idx: int) -> NDArray[np.complex128]:
        """
        Get a specific basis state by index.

        Args:
            idx: Index of the desired basis state

        Returns:
            The basis state vector

        Raises:
            IndexError: If idx is out of range
        """
        if not 0 <= idx < self._dimension:
            raise IndexError(f"Basis state index {idx} out of range [0, {self._dimension - 1}]")
        return self._basis_states[idx]

    @property
    def basis_matrix(self) -> NDArray[np.complex128]:
        """Get the basis matrix."""
        return self._basis_matrix

    @property
    def dimension(self) -> int:
        """Get the dimension of the basis."""
        return self._dimension

    def is_orthonormal(self, tolerance: float = 1e-10) -> bool:
        """
        Check if the basis is orthonormal within given tolerance.

        Args:
            tolerance: Maximum allowed deviation from orthonormality

        Returns:
            True if basis is orthonormal, False otherwise
        """
        for i, state_i in enumerate(self._basis_states):
            for j, state_j in enumerate(self._basis_states):
                inner_prod = (state_i.conj().T @ state_j)[0, 0]
                expected = 1.0 if i == j else 0.0

                if not np.isclose(inner_prod, expected, atol=tolerance):
                    print(
                        f"{'State ' + str(i) + ' is not normalized' if i == j else 'States ' + str(i) + ' and ' + str(j) + ' are not orthogonal'}: "
                        f"expected {expected:.1f}+0j but got {inner_prod:.3f}"
                    )
                    return False

        print("Basis is orthonormal.")
        return True

    def get_completeness(self) -> NDArray[np.complex128]:
        """
        Calculate and return the completeness relation matrix.

        Returns:
            The completeness relation matrix
        """
        return np.sum(np.outer(state, state.conj()) for state in self._basis_states)

    def verify_completeness(self, tolerance: float = 1e-10) -> bool:
        """
        Verify if the completeness relation is satisfied.

        Args:
            tolerance: Maximum allowed deviation from identity

        Returns:
            True if completeness relation is satisfied, False otherwise
        """
        completeness = self.get_completeness()
        return np.allclose(completeness, np.eye(self._dimension, dtype=np.complex128), atol=tolerance)

    def __str__(self) -> str:
        """Return string representation of the basis.

        Returns:
            str: Formatted string showing basis matrix
        """
        return f"Cat basis matrix of dimension {self.dimension}:\n{pretty_format(self.basis_matrix)}"
