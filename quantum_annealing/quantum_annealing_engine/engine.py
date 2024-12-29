import numpy as np
from qutip import Qobj, basis, qeye, sigmax, sigmaz, tensor


class Gates:
    @staticmethod
    def tensor_sigmax(i: int, n: int) -> Qobj:
        """Tensor product such that i-th element is sigma_x and rest are identity"""
        ops = [qeye(2) for _ in range(n)]
        ops[i] = sigmax()
        return tensor(ops)

    @staticmethod
    def tensor_sigmaz(i: int, n: int) -> Qobj:
        """Tensor product such that i-th element is sigma_z and rest are identity"""
        ops = [qeye(2) for _ in range(n)]
        ops[i] = sigmaz()
        return tensor(ops)

    @staticmethod
    def tensor_bin(i: int, n: int) -> Qobj:
        """Tensor product such that i-th element is (I - sigma_z) / 2 and rest are identity"""
        ops = [qeye(2) for _ in range(n)]
        ops[i] = (qeye(2) - sigmaz()) / 2
        return tensor(ops)


class Observable:
    @staticmethod
    def get_ground_eigenstate(op: Qobj) -> Qobj:
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
        N = np.power(2, self.get_num_qubits())
        basis_states = [
            tensor(
                [
                    basis(2, (i >> m) % 2)
                    for m in range(self.get_num_qubits() - 1, -1, -1)
                ]
            )
            for i in range(N)
        ]
        self.basis_states = basis_states
        self.basis_matrix = np.hstack([psi.full() for psi in basis_states])

    #   Core getters
    def get_num_qubits(self) -> int:
        return self.num_qubits

    def get_basis_states(self) -> list:
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

    def get_profit(self, idx: int) -> np.ndarray:
        return self.profits[idx]

    def get_weight(self, idx: int) -> np.ndarray:
        return self.weights[idx]

    def get_capacity(self) -> int:
        return self.capacity

    def get_num_items(self) -> int:
        return self.num_items

    # Getters
    def calculate_total_weight(self, item_bits: str) -> int:
        """Calculate total weight from item bits."""
        return sum(self.get_weight(i) for i, bit in enumerate(item_bits) if bit == "1")

    def calculate_total_profit(self, item_bits: str) -> int:
        """Calculate total profit from item bits."""
        return sum(self.get_profit(i) for i, bit in enumerate(item_bits) if bit == "1")


class MakeGraph:
    def __init__(self) -> None:
        pass

    #   Core setters
    def set_probs(self, probs: list) -> None:
        self.probs = probs

    def set_spectrum(self, spectrum: list) -> None:
        self.spectrum = spectrum

    #   Core getters
    def get_probs(self) -> list:
        return self.probs

    def get_spectrum(self) -> list:
        return self.spectrum
