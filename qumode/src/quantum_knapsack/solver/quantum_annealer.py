import itertools
import multiprocessing
from concurrent.futures.process import ProcessPoolExecutor
from copy import deepcopy
from dataclasses import dataclass
from typing import List, Sequence, Optional, Tuple

import numpy as np
from numpy.typing import NDArray

from src.quantum_knapsack.basis import QumodeBasis
from src.quantum_knapsack.mapping import Mapping
from src.quantum_knapsack.matrix import Observable
from src.quantum_knapsack.solution import Result
from .solver import Solver
from scipy.integrate import quad


@dataclass
class AnnealingParameters:
    """Parameters for quantum annealing process."""

    initial_h_strength: float
    energy_scale: float
    time_scale: float
    num_iterations: int

    def validate(self) -> None:
        """Validate parameter values.

        Raises:
            ValueError: If parameters are invalid
        """
        if self.initial_h_strength <= 0:
            raise ValueError("Initial hamiltonian strength must be positive")
        if self.energy_scale <= 0:
            raise ValueError("Energy scale must be positive")
        if self.num_iterations < 2:
            raise ValueError("Number of iterations must be at least 2")


class QuantumAnnealer(Solver):
    """Quantum annealing solver implementation.

    Performs quantum annealing by evolving system from initial
    to problem hamiltonian.
    """

    def __init__(self, basis: QumodeBasis, mapping: Mapping) -> None:
        """Initialize quantum annealer.

        Args:
            basis: Quantum basis for evolution
            mapping: Problem mapping to solve

        Raises:
            ValueError: If basis and mapping dimensions don't match
        """
        super().__init__()

        if basis.dimension != mapping.dimension:
            raise ValueError(
                f"Basis dimension {basis.dimension} doesn't match "
                f"mapping dimension {mapping.dimension}"
            )

        self._basis = basis
        self._mapping = mapping

        # Initialize the mixing hamiltonian
        # Explicitly cast as real since observables must have real-valued entries
        self._mixing_hamiltonian = (
                basis.annihilation_operator + basis.creation_operator
        ).real.astype(np.float64)

        # Initialize evolution arrays
        self._times: NDArray[np.float64] = np.array([])
        self._dt: float = 0.0
        self._observables: List[Observable] = []
        self._states: List[NDArray[np.complex128]] = []

        # Initialize parameters
        self._params: Optional[AnnealingParameters] = None

    def initialize(self, params: AnnealingParameters) -> None:
        """Initialize annealing process.

        Args:
            params: Annealing parameters

        Raises:
            ValueError: If parameters are invalid
        """
        params.validate()
        self._params = params

        # Scale mixing hamiltonian
        self._mixing_hamiltonian *= params.initial_h_strength

        # Setup time evolution
        self._setup_evolution(params.num_iterations)

        # Initialize quantum states
        self._initialize_states()

    def _setup_evolution(self, num_iterations: int) -> None:
        """Setup time evolution arrays and operators.

        Args:
            num_iterations: Number of time steps
        """
        # Create time array
        self._times = np.linspace(0, 1.0 * self._params.time_scale, num_iterations)
        self._dt = float(self._times[1] - self._times[0])

        # Create observables for each time step
        self._observables = [
            Observable(
                self._get_hamiltonian(t),
                self._params.energy_scale,
                self._dt
            )
            for t in self._times
        ]

    def _initialize_states(self) -> None:
        """Initialize quantum state arrays."""
        dim = self._basis.dimension
        self._states = [
            np.zeros((dim, 1), dtype=np.complex128)
            for _ in self._times
        ]
        # Set initial state to ground state
        self._states[0] = self._observables[0].ground_state[1]

    def solve(self) -> None:
        """Execute quantum annealing evolution.

        Raises:
            RuntimeError: If solver is not initialized
        """
        if not self._params:
            raise RuntimeError("Annealer not initialized")

        # Evolve system through time steps
        for i in range(len(self._times) - 1):
            self._states[i + 1] = self._observables[i].evolve(
                self._states[i]
            )

    def get_result(self, num_points: int) -> Result:
        """Get annealing results.

        Args:
            num_points: Number of points for result analysis

        Returns:
            Result: Annealing results

        Raises:
            RuntimeError: If solution is not available
        """
        if not self._states or not self._params:
            raise RuntimeError("No solution available")

        return Result(
            self._basis,
            self._mapping,
            self._times,
            self._observables,
            self._states,
            self._params.initial_h_strength,
            self._params.energy_scale,
            num_points
        )

    def _get_hamiltonian(self, time: float) -> NDArray[np.float64]:
        """Get time-dependent hamiltonian.

        Args:
            time: Normalized time point (0 to 1)

        Returns:
            NDArray[np.float64]: Hamiltonian matrix
        """
        end: float = 1.0 * self._params.time_scale

        return (
                (1 - time / end) * self._mixing_hamiltonian +
                time / end * self._mapping.problem_hamiltonian
        )

    @property
    def observables(self) -> List[Observable]:
        """Get system observables.

        Returns:
            Sequence[Observable]: Observable operators
        """
        return deepcopy(self._observables)

    @property
    def evolution_times(self) -> NDArray[np.float64]:
        """Get evolution time points.

        Returns:
            NDArray[np.float64]: Time points array
        """
        return self._times.copy()

    @property
    def quantum_states(self) -> List[NDArray[np.complex128]]:
        """Get quantum states at each time point.

        Returns:
            Sequence[NDArray[np.complex128]]: Quantum state vectors
        """
        return [deepcopy(state) for state in self._states]

    @property
    def mixing_hamiltonian(self) -> NDArray[np.float64]:
        """Get mixing hamiltonian.

        Returns:
            NDArray[np.float64]: Mixing hamiltonian matrix
        """
        return deepcopy(self._mixing_hamiltonian)

    @property
    def initial_state(self) -> NDArray[np.complex128]:
        """Get initial quantum state.
        
        Returns:
            NDArray[np.complex128]: Initial state vector
            
        Raises:
            RuntimeError: If states not initialized
        """
        if not self._states:
            raise RuntimeError("States not initialized")
        return deepcopy(self._states[0])
