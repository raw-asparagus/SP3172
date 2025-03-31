from copy import deepcopy
from dataclasses import dataclass
from typing import List, Sequence, Dict, Set, Tuple

import numpy as np
from numpy.typing import NDArray

from src.quantum_knapsack.basis import QumodeBasis
from src.quantum_knapsack.mapping import Mapping
from src.quantum_knapsack.matrix import Observable


@dataclass(frozen=True)
class ProblemParameters:
    """Problem-specific parameters."""

    profits: NDArray[np.int64]
    weights: NDArray[np.int64]
    capacity: int

    def validate(self) -> None:
        """Validate parameters.
        
        Raises:
            ValueError: If parameters are invalid
        """
        if len(self.weights) != len(self.profits):
            raise ValueError("Weights and profits must have same length")
        if np.any(self.weights <= 0) or np.any(self.profits <= 0):
            raise ValueError("Weights and profits must be positive")
        if self.capacity <= 0:
            raise ValueError("Capacity must be positive")


@dataclass(frozen=True)
class AlgorithmParameters:
    """Algorithm-specific parameters."""

    energy_scale: float
    initial_h_strength: float
    penalty_scale: float

    def validate(self) -> None:
        """Validate parameters.
        
        Raises:
            ValueError: If parameters are invalid
        """
        if self.energy_scale <= 0:
            raise ValueError("Energy scale must be positive")
        if self.initial_h_strength <= 0:
            raise ValueError("Initial H strength must be positive")
        if self.penalty_scale <= 0:
            raise ValueError("Penalty scale must be positive")


class Result:
    """Stores and analyzes quantum optimization results."""

    def __init__(
            self,
            basis: QumodeBasis,
            mapping: Mapping,
            times: NDArray[np.float64],
            observables: List[Observable],
            states: List[NDArray[np.complex128]],
            initial_h_strength: float,
            energy_scale: float,
            num_points: int
    ) -> None:
        """Initialize optimization result.
        
        Args:
            basis: Quantum basis used
            mapping: Problem mapping
            times: Evolution time points
            observables: System observables
            states: Quantum states
            initial_h_strength: Initial Hamiltonian strength
            energy_scale: Energy scale
            num_points: Number of analysis points
            
        Raises:
            ValueError: If parameters are invalid
        """
        self._validate_inputs(
            basis, mapping, times, observables, states, num_points
        )

        # Store basic parameters
        self._basis = basis
        self._mapping = mapping
        self._times = times
        self._observables = observables
        self._states = states

        # Store problem parameters
        self._problem_params = ProblemParameters(
            mapping.knapsack.profits,
            mapping.knapsack.weights,
            mapping.knapsack.capacity
        )

        # Validate problem parameters
        self._problem_params.validate()

        # Store algorithm parameters
        self._algorithm_params = AlgorithmParameters(
            energy_scale,
            initial_h_strength,
            mapping.penalty_scale
        )

        # Validate algorithm parameters
        self._algorithm_params.validate()

        # Initialize computed results
        self._graph_times: NDArray[np.float64] = np.array([])
        self._probabilities: NDArray[np.float64] = np.array([])
        self._expectation_values: NDArray[np.float64] = np.array([])
        self._eigenspectrum: List[Tuple[float, NDArray[np.complex128]]] = []

        # Compute initial results
        self._compute_basic_quantities(num_points)

    @staticmethod
    def _validate_inputs(
            basis: QumodeBasis,
            mapping: Mapping,
            times: NDArray[np.float64],
            observables: List[Observable],
            states: List[NDArray[np.complex128]],
            num_points: int
    ) -> None:
        """Validate input parameters.
        
        Args:
            basis: Quantum basis
            mapping: Problem mapping
            times: Time points
            observables: Observables
            states: Quantum states
            num_points: Analysis points
            
        Raises:
            ValueError: If inputs are invalid
        """
        if len(times) != len(observables) or len(times) != len(states):
            raise ValueError("Inconsistent sequence lengths")
        if basis.dimension != mapping.dimension:
            raise ValueError("Basis and mapping dimensions mismatch")
        if num_points <= 0:
            raise ValueError("Number of points must be positive")

    def _compute_basic_quantities(self, num_points: int) -> None:
        """Compute basic result quantities.
        
        Args:
            num_points: Number of analysis points
        """
        self._compute_graph_times(num_points)
        self._compute_probabilities(num_points)
        # if np.allclose(self._states[1], self._states[-2]):
        #     self._compute_expectation_values_alt()
        # else:
        #     self._compute_expectation_values(num_points)
        self._compute_expectation_values(num_points)
        self._compute_eigenspectrum(num_points)

    def _compute_graph_times(self, num_points: int) -> None:
        """Compute graph times.

        Args:
            num_points: Number of analysis points
        """
        indices = np.linspace(0, len(self._times) - 1, num_points, dtype=int)
        self._graph_times: NDArray[np.float64] = np.zeros(num_points)

        for idx, t_idx in enumerate(indices):
            self._graph_times[idx] = self._times[t_idx]

    def _compute_probabilities(self, num_points: int) -> None:
        """Compute state probabilities.
        
        Args:
            num_points: Number of analysis points
        """
        indices = np.linspace(0, len(self._times) - 1, num_points, dtype=int)
        dim = self._basis.dimension
        self._probabilities = np.zeros((num_points, dim))

        for idx, t_idx in enumerate(indices):
            for i in range(dim):
                self._probabilities[idx, i] = np.abs(self._basis.get_basis_state(i).conj().T @ self._states[t_idx] @ self._states[t_idx].conj().T @ self._basis.get_basis_state(i))

    def _compute_expectation_values(self, num_points: int) -> None:
        """Compute expectation values."""
        indices = np.linspace(0, len(self._times) - 1, num_points, dtype=int)
        self._expectation_values = np.zeros(num_points)

        for idx, t_idx in enumerate(indices):
            self._expectation_values[idx] = self._observables[t_idx].measure(self._states[t_idx])

    def _compute_expectation_values_alt(self) -> None:
        """Compute expectation values for start and end states."""
        self._expectation_values = np.array([
            self._observables[0].measure(self._states[0]), self._observables[-1].measure(self._states[-1])
        ])

    def _compute_eigenspectrum(self, num_points: int) -> None:
        """Compute energy eigenspectrum."""
        indices = np.linspace(0, len(self._times) - 1, num_points, dtype=int)
        dim = self._basis.dimension
        self._eigenspectrum = np.zeros((num_points, dim))

        for idx, t_idx in enumerate(indices):
            self._eigenspectrum[idx] = self._observables[t_idx].eigenvalues

    def get_correct_solutions(
            self,
            probability_threshold: float = 0.01
    ) -> Dict[str, str]:
        """Get correct solutions with their probabilities.
        
        Args:
            probability_threshold: Minimum probability threshold
            
        Returns:
            Dict[int, float]: Mapping of solution index to probability
            
        Raises:
            ValueError: If threshold is invalid
        """
        if not 0 <= probability_threshold <= 1:
            raise ValueError("Probability threshold must be between 0 and 1")

        # Get final probabilities
        final_probs = self._probabilities[-1]

        # Find valid solutions
        valid_solutions = self._find_valid_solutions()

        # Filter solutions by probability
        return {
            str(bin(idx)): f"{prob:.3f}" for idx, prob in enumerate(final_probs)
            if idx in valid_solutions and prob >= probability_threshold
        }

    def _find_valid_solutions(self) -> Set[int]:
        """Find valid solution states.
        
        Returns:
            Set[int]: Indices of valid solution states
        """
        valid = set()
        weights = self._problem_params.weights
        capacity = self._problem_params.capacity

        # Check all basis states
        for state_idx in range(self._basis.dimension):
            binary = format(state_idx, f'0{len(weights)}b')
            total_weight = sum(
                w for w, b in zip(weights, binary) if b == '1'
            )
            if total_weight <= capacity:
                valid.add(state_idx)

        return valid

    @property
    def basis(self) -> QumodeBasis:
        return self._basis

    @property
    def mapping(self) -> Mapping:
        return self._mapping

    @property
    def dimension(self) -> int:
        return self._basis.dimension

    @property
    def observables(self) -> List[Observable]:
        return self._observables

    @property
    def times(self) -> NDArray[np.float64]:
        """Get evolution times.
        
        Returns:
            NDArray[np.float64]: Time points array
        """
        return self._times.copy()

    @property
    def graph_times(self) -> NDArray[np.float64]:
        """Get graph times.

        Returns:
            NDArray[np.float64]: Time points array
        """
        return self._graph_times.copy()

    @property
    def probabilities(self) -> NDArray[np.float64]:
        """Get state probabilities.
        
        Returns:
            NDArray[np.float64]: Probability arrays
        """
        return self._probabilities.copy()

    @property
    def expectation_values(self) -> NDArray[np.float64]:
        """Get expectation values.
        
        Returns:
            NDArray[np.float64]: Expectation value array
        """
        return self._expectation_values.copy()

    @property
    def eigenspectrum(self) -> List[Tuple[float, NDArray[np.complex128]]]:
        """Get energy eigenspectrum.
        
        Returns:
            List[Tuple[float, NDArray[np.complex128]]]:
                Eigenvalues and eigenvectors
        """
        return deepcopy(self._eigenspectrum)

    @property
    def problem_parameters(self) -> ProblemParameters:
        """Get problem parameters.
        
        Returns:
            ProblemParameters: Problem-specific parameters
        """
        return self._problem_params

    @property
    def algorithm_parameters(self) -> AlgorithmParameters:
        """Get algorithm parameters.
        
        Returns:
            AlgorithmParameters: Algorithm-specific parameters
        """
        return self._algorithm_params
