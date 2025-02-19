from abc import ABC, abstractmethod
from typing import Optional

from src.quantum_knapsack.basis import Basis
from src.quantum_knapsack.mapping import Mapping
from src.quantum_knapsack.solution import Result


class Solver(ABC):
    """Abstract base class for quantum optimization solvers."""

    def __init__(self) -> None:
        """Initialize solver with empty state."""
        self._basis: Optional[Basis] = None
        self._mapping: Optional[Mapping] = None

    @abstractmethod
    def initialize(self, **kwargs) -> None:
        """Initialize the solver with problem-specific parameters.
        
        Args:
            **kwargs: Implementation-specific parameters
            
        Raises:
            ValueError: If parameters are invalid
        """
        pass

    @abstractmethod
    def solve(self) -> None:
        """Execute the solving algorithm.
        
        Raises:
            RuntimeError: If solver is not initialized
        """
        pass

    @abstractmethod
    def get_result(self, **kwargs) -> Result:
        """Get the solution result.
        
        Args:
            **kwargs: Implementation-specific parameters
            
        Returns:
            Result: Solution results
            
        Raises:
            RuntimeError: If solution is not available
        """
        pass

    @property
    def basis(self) -> Optional[Basis]:
        """Get the quantum basis used by the solver.
        
        Returns:
            Optional[Basis]: Quantum basis or None if not set
        """
        return self._basis

    @property
    def mapping(self) -> Optional[Mapping]:
        """Get the problem mapping used by the solver.
        
        Returns:
            Optional[Mapping]: Problem mapping or None if not set
        """
        return self._mapping
