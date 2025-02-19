from abc import ABC, abstractmethod
from typing import Optional

import numpy as np
from numpy.typing import NDArray

from src.quantum_knapsack.basis import Basis
from src.quantum_knapsack.knapsack import Knapsack


class Mapping(ABC):
    """Abstract base class for quantum problem mappings.
    
    Maps classical optimization problems to quantum hamiltonians.
    """

    def __init__(self) -> None:
        """Initialize mapping with empty state."""
        self._knapsack: Optional[Knapsack] = None
        self._basis: Optional[Basis] = None
        self._dimension: int = 0
        self._penalty_scale: float = 0.0

        # Initialize empty hamiltonians
        self._penalty_hamiltonian: NDArray[np.float64] = np.zeros((0, 0), dtype=np.float64)
        self._profit_hamiltonian: NDArray[np.float64] = np.zeros((0, 0), dtype=np.float64)
        self._problem_hamiltonian: NDArray[np.float64] = np.zeros((0, 0), dtype=np.float64)

    @abstractmethod
    def initialize(self, basis: Basis) -> None:
        """Initialize the mapping with a given basis.
        
        Args:
            basis: Quantum basis to use for mapping
            
        Raises:
            ValueError: If basis is invalid
        """
        pass

    @abstractmethod
    def _compute_hamiltonians(self) -> None:
        """Compute penalty and profit hamiltonians."""
        pass

    @property
    def dimension(self) -> int:
        """Get dimension of the quantum system.
        
        Returns:
            int: Dimension of the Hilbert space
        """
        return self._dimension

    @property
    def problem_hamiltonian(self) -> NDArray[np.float64]:
        """Get the complete problem hamiltonian.
        
        Returns:
            NDArray[np.float64]: Problem hamiltonian matrix
            
        Raises:
            RuntimeError: If hamiltonians haven't been computed
        """
        if self._problem_hamiltonian.size == 0:
            raise RuntimeError("Problem hamiltonian not initialized")
        return self._problem_hamiltonian.copy()

    @property
    def penalty_scale(self) -> float:
        """Get the penalty scaling factor.
        
        Returns:
            float: Penalty scale value
        """
        return self._penalty_scale

    @property
    def knapsack(self) -> Optional[Knapsack]:
        """Get the associated knapsack problem.
        
        Returns:
            Optional[Knapsack]: Knapsack problem instance or None
        """
        return self._knapsack
