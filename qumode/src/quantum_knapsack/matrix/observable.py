from copy import deepcopy

import numpy as np
from numpy.typing import NDArray

from .exceptions import ObservableException
from .square_matrix import SquareMatrix


class Observable(SquareMatrix):
    """Quantum observable matrix with measurement capabilities."""

    def __init__(
            self,
            matrix: NDArray[np.float64],
            energy_scale: float,
            dt: float
    ) -> None:
        """Initialize Observable with matrix and evolution parameters.
        
        Args:
            matrix: Observable operator matrix
            energy_scale: Energy scaling factor
            dt: Time step for evolution
            
        Raises:
            ObservableException: If matrix is invalid
            ValueError: If parameters are invalid
        """
        super().__init__()

        if not isinstance(matrix, np.ndarray):
            raise TypeError("Matrix must be a numpy array")
        if matrix.shape[0] != matrix.shape[1]:
            raise ValueError("Matrix must be square")
        if energy_scale <= 0:
            raise ValueError("Energy scale must be positive")
        if dt <= 0:
            raise ValueError("Time step must be positive")

        self._matrix = matrix.real.astype(np.float64) if np.all(np.isreal(matrix)) else matrix.astype(np.complex128)
        self._unitary_evolution: NDArray[np.complex128] = np.zeros(self.shape, dtype=np.complex128)

        # Perform eigendecomposition and compute evolution
        self._eigen_decompose()
        self._compute_unitary_evolution(energy_scale, dt)

    def _compute_unitary_evolution(self, energy_scale: float, dt: float) -> None:
        """Compute unitary evolution operator.
        
        Args:
            energy_scale: Energy scaling factor
            dt: Time step
        """
        # Get eigenvector matrix
        p: NDArray[np.float64] = np.hstack(self.eigenvectors)

        # Compute diagonal evolution matrix
        exp_d: NDArray[np.complex128] = np.diag(
            np.exp(-1j * self.eigenvalues / energy_scale * dt)
        )

        # Calculate evolution operator
        self._unitary_evolution = p @ exp_d @ p.conj().T

    @property
    def unitary_evolution(self) -> NDArray[np.complex128]:
        """Get the unitary evolution operator.
        
        Returns:
            NDArray[np.complex128]: Evolution operator matrix
        """
        if not self._unitary_evolution.any():
            raise RuntimeError("Unitary evolution not computed")
        return deepcopy(self._unitary_evolution)

    def measure(self, state: NDArray[np.complex128]) -> float:
        """Measure the observable on the given state.
        
        Args:
            state: Quantum state vector
            
        Returns:
            float: Expectation value of measurement
            
        Raises:
            ValueError: If state dimensions don't match
        """
        if state.shape[0] != self._matrix.shape[0]:
            raise ValueError("State vector dimension mismatch")

        result = (state.conj().T @ self._matrix @ state)[0, 0]
        if not np.isclose(result.imag, 0):
            raise ObservableException("Measurement resulted in complex value")

        return float(result.real)

    def evolve(self, state: NDArray[np.complex128]) -> NDArray[np.complex128]:
        """Evolve quantum state using unitary evolution operator.
        
        Args:
            state: Initial quantum state
            
        Returns:
            NDArray[np.complex128]: Evolved quantum state
            
        Raises:
            ValueError: If state dimensions don't match
        """
        if not self._unitary_evolution.any():
            raise RuntimeError("Unitary evolution not computed")
        if state.shape[0] != self._unitary_evolution.shape[0]:
            raise ValueError(
                f"State vector dimension {state.shape[0]} does not match "
                f"operator dimension {self._unitary_evolution.shape[0]}"
            )

        return self._unitary_evolution @ state
