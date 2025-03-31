# TODO Inherit from coffey.py

from dataclasses import dataclass
from typing import List, Tuple, Sequence

import numpy as np
from numpy.typing import NDArray

from .knapsack import Knapsack


@dataclass
class Solution:
    """Data class representing a knapsack solution.

    Attributes:
        items (List[int]): Binary list representing selected items (1) and unselected items (0)
        value (float): Total value/profit of the solution
        weight (float): Total weight of the solution
    """
    binary: str
    items: List[int]
    value: float
    weight: float


class SolutionAnalyzer:
    """Analyzes solutions for the Knapsack Problem.

    This class handles the analysis of quantum solutions for the knapsack problem,
    including finding correct states and calculating solution probabilities.

    Attributes:
        _weights (NDArray[np.int64]): Array of item weights
        _profits (NDArray[np.int64]): Array of item profits/values
        _capacity (int): Maximum weight capacity
        _correct_states (List[int]): List of correct quantum state indices
        _n_items (int): Number of items in the problem
        _n_ancilla (int): Number of ancillary qubits needed
    """

    def __init__(self, knapsack: Knapsack) -> None:
        """Initialize the SolutionAnalyzer.

        Args:
            knapsack: Knapsack problem instance to analyze

        Raises:
            TypeError: If knapsack is not a Knapsack instance
        """
        if not isinstance(knapsack, Knapsack):
            raise TypeError("Expected Knapsack instance")

        self._weights: NDArray[np.int64] = knapsack.weights
        self._profits: NDArray[np.int64] = knapsack.profits
        self._capacity: int = knapsack.capacity
        self._n_items: int = len(self._weights)
        self._n_ancilla: int = self._calculate_ancilla_bits()
        self._correct_states: List[int] = self._find_correct_states()

    def _calculate_ancilla_bits(self) -> int:
        """Calculate the number of ancillary qubits needed.

        Returns:
            int: Number of ancillary qubits
        """
        return int(np.ceil(np.log2(self._capacity + 1)))

    def _get_ancilla_weights(self) -> List[int]:
        """Calculate the weights for ancillary qubits.

        Returns:
            List[int]: Weights for ancillary qubits
        """
        powers = [2 ** i for i in range(self._n_ancilla - 1)]
        remainder = self._capacity + 1 - 2 ** (self._n_ancilla - 1)
        return powers + [remainder]

    def _evaluate_state(self, items: Sequence[int]) -> Tuple[float, float]:
        """Evaluate a solution state by calculating its total value and weight.

        Args:
            items: Binary sequence representing item selection

        Returns:
            Tuple[float, float]: (total_value, total_weight)
        """
        total_value = np.dot(self._profits, items)
        total_weight = np.dot(self._weights, items)
        return total_value, total_weight

    def _find_correct_states(self) -> List[int]:
        """Find the correct Fock states that correspond to optimal solutions.

        This method enumerates all possible quantum states and identifies those
        that represent valid and optimal solutions to the knapsack problem.

        Returns:
            List[int]: State indices that represent optimal solutions
        """
        total_bits = self._n_items + self._n_ancilla
        ancilla_weights = self._get_ancilla_weights()

        valid_states: List[int] = []
        best_value: float = 0.0

        for state_idx in range(1 << total_bits):
            binary = format(state_idx, f'0{total_bits}b')

            # Split binary string into items and ancilla bits
            items = [int(b) for b in binary[:self._n_items]]
            ancilla = [int(b) for b in binary[self._n_items:]]

            # Calculate solution metrics
            total_value, total_weight = self._evaluate_state(items)
            ancilla_sum = np.dot(ancilla_weights, ancilla)

            # Check if solution is valid
            if total_weight == ancilla_sum and total_weight <= self._capacity:
                if total_value > best_value:
                    valid_states = [state_idx]
                    best_value = total_value
                elif total_value == best_value:
                    valid_states.append(state_idx)

        return valid_states

    def get_solution_probability(self, probabilities: NDArray[np.float64]) -> float:
        """Calculate the probability of measuring a correct solution.

        Args:
            probabilities: Array of measurement probabilities for each state

        Returns:
            float: Total probability of measuring a correct solution

        Raises:
            ValueError: If probabilities array has incorrect length
            TypeError: If probabilities is not a numpy array
        """
        if not isinstance(probabilities, np.ndarray):
            raise TypeError("Probabilities must be a numpy array")
        expected_length = 1 << (self._n_items + self._n_ancilla)
        if len(probabilities) != expected_length:
            raise ValueError(f"Probabilities array must have length {expected_length}")

        return sum(probabilities[state] for state in self._correct_states)

    def get_solutions(self) -> List[Solution]:
        """Returns the list of correct solutions with their values and weights.

        Returns:
            List[Solution]: List of valid solutions with their properties
        """
        solutions: List[Solution] = []

        for state in self._correct_states:
            binary = format(state, f'0{self._n_items + self._n_ancilla}b')
            items = [int(b) for b in binary[:self._n_items]]
            value, weight = self._evaluate_state(items)
            solutions.append(Solution(binary=binary, items=items, value=value, weight=weight))

        return solutions

    @property
    def correct_states(self) -> List[int]:
        """Get the list of correct state indices.

        Returns:
            List[int]: Indices of correct quantum states
        """
        return self._correct_states.copy()

    def __str__(self) -> str:
        """Return a string representation of the analyzer state.

        Returns:
            str: Human-readable representation of the analyzer
        """
        solutions = self.get_solutions()
        lines = [
            f"Solution Analyzer Results",
            f"Number of items: {self._n_items}",
            f"Number of ancilla qubits: {self._n_ancilla}",
            f"Number of valid solutions: {len(solutions)}",
            "\nValid Solutions:"
        ]

        for i, sol in enumerate(solutions, 1):
            lines.append(
                f"Solution {i}: "
                f"Binary={sol.binary}, "
                f"Items={sol.items}, "
                f"Value={sol.value:.2f}, "
                f"Weight={sol.weight:.2f}"
            )

        return "\n".join(lines)
