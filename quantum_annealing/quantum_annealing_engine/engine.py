import numpy as np
from qutip import Qobj, basis, qeye, sigmax, sigmaz, tensor


class Gates:
    @staticmethod
    def tensor_sigmax(idx: int, n: int) -> Qobj:
        """Tensor product such that i-th element is sigma_x and rest are identity."""
        op = tensor(qeye(2) if i != idx else sigmax() for i in range(n))
        return op

    @staticmethod
    def tensor_sigmaz(idx: int, n: int) -> Qobj:
        """Tensor product such that i-th element is sigma_z and rest are identity."""
        op = tensor(qeye(2) if i != idx else sigmaz() for i in range(n))
        return op

    @staticmethod
    def tensor_bin(idx: int, n: int) -> Qobj:
        """Tensor product such that i-th element is (I - sigma_z) / 2 and rest are identity."""
        op = tensor(qeye(2) if i != idx else (qeye(2) - sigmaz()) / 2 for i in range(n))
        return op


class Observable:
    @staticmethod
    def get_ground_eigenstate(op: Qobj) -> Qobj:
        """Return ground state of observable."""
        _, ground_eigenstate = op.groundstate()
        return ground_eigenstate


class Basis:
    def __init__(self, num_qubits: int) -> None:
        self.set_num_qubits(num_qubits)
        self.generate_basis_states()

    #   Core setters
    def set_num_qubits(self, num_qubits: int) -> None:
        self.num_qubits = num_qubits

    def generate_basis_states(self) -> None:
        """Generate standard basis states (in column vector and matrix form)."""
        N = np.power(2, self.get_num_qubits())

        basis_states = tuple(
            tensor(basis(2, (i >> j) % 2) for j in range(self.get_num_qubits())[::-1])
            for i in range(N)
        )

        self.basis_states = basis_states
        self.basis_matrix = np.hstack([psi.full() for psi in basis_states])

    #   Core getters
    def get_num_qubits(self) -> int:
        return self.num_qubits

    def get_basis_states(self) -> tuple:
        return self.basis_states

    def get_basis_matrix(self) -> np.ndarray:
        return self.basis_matrix


class KnapsackProblem:
    def __init__(self, profits: np.ndarray, weights: np.ndarray, capacity: int) -> None:
        self.set_profits(profits)
        self.set_weights(weights)
        self.set_capacity(capacity)
        self.set_num_items()

    #   Core Setters
    def set_profits(self, profits: np.ndarray) -> None:
        self.profits = profits

    def set_weights(self, weights: np.ndarray) -> None:
        self.weights = weights

    def set_capacity(self, capacity: int) -> None:
        self.capacity = capacity

    def set_num_items(self) -> None:
        self.num_items = self.get_profits().shape[0]

    #   Core Getters
    def get_profits(self) -> np.ndarray:
        return self.profits

    def get_weights(self) -> np.ndarray:
        return self.weights

    def get_profit(self, idx: int) -> np.int64:
        return self.profits[idx]

    def get_weight(self, idx: int) -> np.int64:
        return self.weights[idx]

    def get_capacity(self) -> int:
        return self.capacity

    def get_num_items(self) -> int:
        return self.num_items

    # Getters
    def calculate_total_weight(self, item_bits: str) -> np.int64:
        """Calculate total weight from item bits."""
        return np.dot(self.get_weights(), np.array(tuple(item_bits), dtype=int))

    def calculate_total_profit(self, item_bits: str) -> np.int64:
        """Calculate total profit from item bits."""
        return np.dot(self.get_profits(), np.array(tuple(item_bits), dtype=int))


class MakeGraph:
    def __init__(self) -> None:
        self.probs = None
        self.simulated_spectrum = None
        self.computed_spectrum = None

    #   Core setters
    def set_probs(self, probs: list) -> None:
        self.probs = probs

    def set_simulated_spectrum(self, simulated_spectrum: list) -> None:
        self.simulated_spectrum = simulated_spectrum

    def set_computed_spectrum(self, computed_spectrum: list) -> None:
        self.computed_spectrum = computed_spectrum

    #   Core getters
    def get_probs(self) -> list:
        return self.probs

    def get_simulated_spectrum(self) -> tuple:
        return self.simulated_spectrum

    def get_computed_spectrum(self) -> tuple:
        return self.computed_spectrum
