from copy import deepcopy
from typing import List

import numpy as np

from src.quantum_knapsack.matrix import ColumnMatrix
from .qumode_basis import QumodeBasis
from .standard_basis import StandardBasis


class CatBasis(QumodeBasis):
    """Cat state basis implementation for quantum modes.

    This class implements the cat state basis, which consists of superpositions
    of coherent states with specific phase relationships.

    Attributes:
        _dimension (int): Dimension of the basis space
        _N (int): Fock space dimension for encoding
        _alpha (complex): Complex amplitude parameter
        _factorials (List[int]): Precomputed factorials for efficiency
        _fock_basis (StandardBasis): Standard basis
        _coherent_states (List[np.ndarray]): Generated coherent states
    """

    def __init__(self, dimension: int, n: int, alpha: complex) -> None:
        """Initialize a cat state basis.

        Args:
            dimension (int): Dimension of the basis space
            n (int): Fock space dimension for encoding
            alpha (complex): Complex amplitude parameter

        Raises:
            ValueError: If dimension or n is less than 1
        """
        if dimension < 1 or n < 1:
            raise ValueError("Dimension and n must be positive integers")

        super().__init__()
        self._dimension: int = dimension
        self._N: int = n
        self._alpha: complex = alpha

        # Initialize components
        self._factorials: List[int] = self._compute_factorials(n)
        self._fock_basis: StandardBasis = StandardBasis(dimension)
        self._coherent_states: List[np.ndarray] = self._generate_coherent_states(n)

        self._create_basis()
        self._create_ladder_operators()

    def _coherent_state(self, k: int) -> np.ndarray:
        """Generate a coherent state for given index k.

        Args:
            k (int): Index of the coherent state

        Returns:
            np.ndarray: Generated coherent state vector
        """
        coherent_state: np.ndarray = np.zeros((self._N, 1), dtype=complex)

        for n in range(self._N):
            temp: np.ndarray = self._fock_basis.get_basis_state(n)

            # Calculate coefficient
            coeff: np.complex128 = np.exp(2j * np.pi * n * k / self._N)
            coeff *= (self._alpha ** n) / np.sqrt(self._factorials[n])

            coherent_state += coeff * temp

        # Apply normalization factor
        coherent_state *= np.exp(-abs(self._alpha) ** 2 / 2)
        return coherent_state

    def _generate_coherent_states(self, n: int) -> List[np.ndarray]:
        """Generate all coherent states for the basis.

        Args:
            n (int): Number of coherent states to generate

        Returns:
            List[np.ndarray]: List of coherent state vectors
        """
        return [self._coherent_state(k) for k in range(n)]

    def _cat_state(self, ell: int) -> np.ndarray:
        """Generate a cat state for given index ell.

        Args:
            ell (int): Index of the cat state

        Returns:
            np.ndarray: Generated cat state vector
        """
        cat_state: np.ndarray = np.zeros((self._N, 1), dtype=complex)

        for k in range(self._N):
            temp: np.ndarray = deepcopy(self._coherent_states[k])
            coeff: np.complex128 = np.exp(-2j * np.pi * ell * k / self._N)
            cat_state += coeff * temp

        return ColumnMatrix.normalise(cat_state)

    def _create_basis(self) -> None:
        """Create the cat state basis vectors."""
        self._basis_states = [self._cat_state(ell) for ell in range(self._N)]
        self._create_matrix()

    @staticmethod
    def _compute_factorials(n: int) -> List[int]:
        """Precompute factorials up to n for efficiency.

        Args:
            n (int): Maximum number to compute factorial for

        Returns:
            List[int]: List of factorials from 0! to n!
        """
        result: List[int] = [1] * (n + 1)
        for i in range(1, n + 1):
            result[i] = result[i - 1] * i
        return result
