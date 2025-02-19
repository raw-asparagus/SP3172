import math
from typing import Tuple

import numpy as np

from src.quantum_knapsack.basis import Basis
from src.quantum_knapsack.knapsack import Knapsack
from .mapping import Mapping


class Coffey(Mapping):
    """Coffey mapping for quantum knapsack problems.
    
    Implements the Coffey scheme for mapping knapsack problems
    to quantum hamiltonians using ancillary qubits.
    """

    def __init__(self, knapsack: Knapsack, penalty_scale: float) -> None:
        """Initialize Coffey mapping.
        
        Args:
            knapsack: Knapsack problem to map
            penalty_scale: Scaling factor for penalty terms
            
        Raises:
            ValueError: If parameters are invalid
        """
        super().__init__()

        if penalty_scale <= 0:
            raise ValueError("Penalty scale must be positive")

        self._knapsack = knapsack
        self._penalty_scale = penalty_scale

        # Calculate required bits
        self._item_bits = knapsack.num_items
        self._ancillary_bits = math.ceil(math.log2(knapsack.capacity + 1))
        self._total_bits = self._item_bits + self._ancillary_bits

        # Calculate system dimension
        self._dimension = 2 ** self._total_bits

        # Calculate ancillary bit modifier
        self._ancillary_modifier = (
                knapsack.capacity + 1 - 2 ** (self._ancillary_bits - 1)
        )

        # Initialize energy arrays
        self._penalties = np.zeros(self._dimension)
        self._profits = np.zeros(self._dimension)

    def initialize(self, basis: Basis) -> None:
        """Initialize the mapping with quantum basis.
        
        Args:
            basis: Quantum basis for the mapping
            
        Raises:
            ValueError: If basis dimension doesn't match
        """
        if basis.dimension != self._dimension:
            raise ValueError(
                f"Basis dimension {basis.dimension} doesn't match "
                f"required dimension {self._dimension}"
            )

        self._basis = basis
        self._compute_state_energies()
        self._compute_hamiltonians()

    def _compute_state_energies(self) -> None:
        """Compute penalties and profits for all basis states."""
        for i in range(self._dimension):
            item_bits, ancilla_bits, modifier_bits = self._decompose_state(i)

            # Calculate ancillary penalty
            penalty = np.sum(2 ** np.arange(len(ancilla_bits)) * ancilla_bits)

            # Add modifier penalty
            if modifier_bits[0]:
                penalty += self._ancillary_modifier

            # Process item bits
            weights = np.array([
                self._knapsack.get_weight(n) for n in range(self._item_bits)
            ])
            profits = np.array([
                self._knapsack.get_profit(n) for n in range(self._item_bits)
            ])

            # Update penalties and profits
            penalty -= np.sum(weights * item_bits)
            self._penalties[i] = penalty * penalty  # Square the penalty
            self._profits[i] = np.sum(profits * item_bits)

    def _compute_hamiltonians(self) -> None:
        """Compute system hamiltonians efficiently."""
        # Initialize diagonal hamiltonians
        self._penalty_hamiltonian = np.diag(self._penalties)
        self._profit_hamiltonian = -np.diag(self._profits)  # Note negative sign

        # Compute final problem hamiltonian
        self._problem_hamiltonian = (
                self._penalty_scale * self._penalty_hamiltonian +
                self._profit_hamiltonian
        )

    def _decompose_state(
            self,
            state_index: int
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Decompose state index into bit arrays.
        
        Args:
            state_index: Index of quantum basis state
            
        Returns:
            Tuple containing:
                - item bits array (np.ndarray)
                - ancillary bits array (np.ndarray)
                - modifier bits array (np.ndarray)
        """
        # Convert to binary and pad
        binary = format(state_index, f'0{self._total_bits}b')

        # Split into components
        item_bits = np.array([
            bit == '1' for bit in binary[:self._item_bits]
        ])
        ancilla_bits = np.array([
            bit == '1' for bit in binary[self._item_bits:-1]
        ])
        modifier_bits = np.array([
            binary[-1] == '1'
        ])

        return item_bits, ancilla_bits, modifier_bits

    @property
    def item_bits(self) -> int:
        """Get number of item bits.
        
        Returns:
            int: Number of bits representing items
        """
        return self._item_bits

    @property
    def ancillary_bits(self) -> int:
        """Get number of ancillary bits.
        
        Returns:
            int: Number of ancillary bits
        """
        return self._ancillary_bits
