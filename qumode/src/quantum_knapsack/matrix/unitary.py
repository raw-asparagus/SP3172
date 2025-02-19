from typing import Optional
import numpy as np
from numpy._typing import NDArray

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
            start_time (float): Initial time point
            end_time (float): Final time point
            func (Callable[[ArrayLike], NDArray[np.float64]]): Function that returns F(end) - F(start)
                where integral of -i/hbar H(t)dt = F(t) + C
            energy_scale (float): Energy scale factor in natural units

        Raises:
            ValueError: If end_time <= start_time or energy_scale <= 0
        """
        super().__init__()

        # Initialize attributes
        self._unitary_evolution: Optional[NDArray[np.complex128]] = None

        self._matrix = matrix

        # Compute the unitary matrix
        self._compute_unitary(energy_scale)

    def _compute_unitary(self, energy_scale: float) -> None:
        """Compute the unitary evolution operator.

        Args:
            energy_scale (float): Energy scale factor for the evolution

        Note:
            This method updates the _unitary_evolution attribute.
        """
        self._eigen_decompose()
        p: NDArray[np.complex128] = np.hstack(self.eigenvectors)
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
        if self._unitary_evolution is None:
            raise RuntimeError("Unitary evolution has not been computed")

        if state.shape[0] != self._unitary_evolution.shape[0]:
            raise ValueError(
                f"State vector dimension {state.shape[0]} does not match "
                f"operator dimension {self._unitary_evolution.shape[0]}"
            )

        return self._unitary_evolution @ state