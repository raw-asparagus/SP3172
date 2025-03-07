from copy import deepcopy
from typing import List, Tuple

import numpy as np
from numpy.typing import NDArray

from .exceptions import DegenerateException, ObservableException
from .matrix import Matrix


class SquareMatrix(Matrix):
    """Square matrix with eigendecomposition capabilities."""

    def __init__(self) -> None:
        """Initialize square matrix with empty eigendecomposition."""
        super().__init__()
        self._eigenvalues: NDArray[np.float64] = np.zeros(self.shape, dtype=np.float64)
        self._eigenvectors: List[NDArray[np.float64]] = []

    def _eigen_decompose(self) -> None:
        """Perform eigendecomposition of the matrix.
        
        Raises:
            ObservableException: If matrix is not positive definite
            ValueError: If matrix is not square
        """
        if self._matrix.shape[0] != self._matrix.shape[1]:
            raise ValueError("Matrix must be square for eigendecomposition")

        try:
            # # Ensure matrix is Hermitian
            if not np.allclose(self._matrix, self._matrix.conj().T):
                raise ObservableException("Matrix must be Hermitian")

            eigenvalues, eigenvectors = np.linalg.eigh(self._matrix)

            self._eigenvalues = eigenvalues
            self._eigenvectors = [v.reshape(-1, 1) for v in eigenvectors.T]

        except np.linalg.LinAlgError as e:
            raise ObservableException("Failed to compute eigendecomposition") from e

    @property
    def eigenvalues(self) -> NDArray[np.float64]:
        """Get eigenvalues of the matrix.
        
        Returns:
            NDArray[np.float64]: Array of eigenvalues
            
        Raises:
            RuntimeError: If eigendecomposition hasn't been performed
        """
        if not self._eigenvalues.any():
            raise RuntimeError("Eigendecomposition not performed")
        return self._eigenvalues.copy()

    @property
    def eigenvectors(self) -> List[NDArray[np.float64]]:
        """Get eigenvectors of the matrix.
        
        Returns:
            List[NDArray[np.float64]]: List of eigenvector column matrices
            
        Raises:
            RuntimeError: If eigendecomposition hasn't been performed
        """
        if len(self._eigenvectors) == 0:
            raise RuntimeError("Eigendecomposition not performed")
        return [deepcopy(ev) for ev in self._eigenvectors]

    @property
    def ground_state(self) -> Tuple[float, NDArray[np.float64]]:
        """Get ground state (lowest eigenvalue and corresponding eigenvector).
        
        Returns:
            Tuple[float, NDArray[np.float64]]: (eigenvalue, eigenvector)
            
        Raises:
            DegenerateException: If ground state is degenerate
            RuntimeError: If eigendecomposition hasn't been performed
        """
        if not self._eigenvalues.any() or not len(self._eigenvectors) != 0:
            raise RuntimeError("Eigendecomposition not performed")

        if len(self._eigenvalues) > 1 and np.isclose(
                self._eigenvalues[0],
                self._eigenvalues[1]
        ):
            raise DegenerateException("Ground state is degenerate")

        return float(self._eigenvalues[0]), deepcopy(self._eigenvectors[0])
