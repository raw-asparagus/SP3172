import numpy as np
from numpy.typing import NDArray

from .square_matrix import SquareMatrix


class Unitary(SquareMatrix):
    """A unitary matrix class representing quantum evolution.

    Attributes:
        _matrix (NDArray[np.float64]): The matrix representation
        _unitary_evolution (NDArray[np.complex128]): Computed unitary evolution operator
    """

    def __init__(
            self,
            matrix: NDArray[np.float64],
            energy_scale: float
    ) -> None:
        """Initialize unitary matrix with time evolution parameters.

        Args:
            matrix: Matrix for unitary evolution
            energy_scale: Energy scaling factor

        Raises:
            ValueError: If end_time <= start_time or energy_scale <= 0
        """
        super().__init__()

        # Initialize attributes
        self._matrix = matrix.real.astype(np.float64) if np.all(np.isreal(matrix)) else matrix.astype(np.complex128)
        self._unitary_evolution: NDArray[np.complex128] = np.zeros(self.shape, dtype=np.complex128)

        # Perform eigendecomposition and compute evolution
        self._eigen_decompose()
        self._compute_unitary(energy_scale)

    def _compute_unitary(self, energy_scale: float) -> None:
        """Compute the unitary evolution operator.

        Args:
            energy_scale (float): Energy scale factor for the evolution

        Note:
            This method updates the _unitary_evolution attribute.
        """
        # Get eigenvector matrix
        p: NDArray[np.float64] = np.hstack(self.eigenvectors)

        exp_d: NDArray[np.complex128] = np.diag(
            np.exp(-1j * self.eigenvalues / energy_scale)
        )
        self._unitary_evolution = p @ exp_d @ p.conj().T

    def evolve(self, state: NDArray[np.complex128]) -> NDArray[np.complex128]:
        """Evolve quantum state using unitary evolution operator.

        Args:
            state (NDArray[np.complex128]): Initial quantum state vector

        Returns:
            NDArray[np.complex128]: Evolved quantum state

        Raises:
            RuntimeError: If unitary evolution operator hasn't been computed
            ValueError: If state dimensions don't match the operator dimensions
        """
        if not self._unitary_evolution.any():
            raise RuntimeError("Unitary evolution not computed")

        if state.shape[0] != self._unitary_evolution.shape[0]:
            raise ValueError(
                f"State vector dimension {state.shape[0]} does not match "
                f"operator dimension {self._unitary_evolution.shape[0]}"
            )

        return self._unitary_evolution @ state