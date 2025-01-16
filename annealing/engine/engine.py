from functools import reduce
import numpy as np
import time


class Matrix:
    SIGMAX = np.array([[0, 1], [1, 0]])
    SIGMAZ = np.array([[1, 0], [0, -1]])
    IDENTITY = np.eye(2)

    @staticmethod
    def eye(n: int) -> np.ndarray:
        return np.eye(n)

    @staticmethod
    def tensor_product(ops: list) -> np.ndarray:
        return reduce(np.kron, ops)


class Gates:
    @staticmethod
    def tensor_sigmax(idx: int, n: int) -> np.ndarray:
        ops = [Matrix.SIGMAX if i == idx else Matrix.IDENTITY for i in range(n)]
        return Matrix.tensor_product(ops)

    @staticmethod
    def tensor_sigmaz(idx: int, n: int) -> np.ndarray:
        ops = [Matrix.SIGMAZ if i == idx else Matrix.IDENTITY for i in range(n)]
        return Matrix.tensor_product(ops)

    @staticmethod
    def tensor_bin(idx: int, n: int) -> np.ndarray:
        s = (Matrix.IDENTITY - Matrix.SIGMAZ) / 2
        ops = [s if i == idx else Matrix.IDENTITY for i in range(n)]
        return Matrix.tensor_product(ops)


class Observable:
    @staticmethod
    def eigen_decompose(op: np.ndarray) -> tuple:
        return np.linalg.eigh(op)

    @staticmethod
    def get_eigenvalues(op: np.ndarray) -> np.ndarray:
        return Observable.eigen_decompose(op)[0]

    @staticmethod
    def get_eigenstates(op: np.ndarray) -> np.ndarray:
        return Observable.eigen_decompose(op)[1]

    @staticmethod
    def get_ground_energy(op: np.ndarray) -> float:
        return Observable.get_eigenvalues(op)[0]

    @staticmethod
    def get_ground_state(op: np.ndarray) -> np.ndarray:
        return Observable.get_eigenstates(op)[:, 0]


class StandardBasis:
    def __init__(self, num_qubits: int) -> None:
        self.set_num_qubits(num_qubits)
        self.generate_basis_states()

    # Core mutators #
    def set_num_qubits(self, num_qubits: int) -> None:
        self.num_qubits = num_qubits

    def generate_basis_states(self) -> None:
        N = np.power(2, self.get_num_qubits())
        self.basis_matrix = np.eye(N, dtype=complex)
        self.basis_states = tuple(self.basis_matrix[:, i : i + 1] for i in range(N))

    # Core accessors #
    def get_num_qubits(self) -> int:
        return self.num_qubits

    def get_basis_states(self) -> tuple:
        return self.basis_states

    def get_basis_matrix(self) -> np.ndarray:
        return self.basis_matrix


class Result:
    def __init__(self, states: np.ndarray, times: np.ndarray) -> None:
        self.set_len(times.shape[0])
        self.set_states(states)
        self.set_times(times)

    #   Core mutators
    def set_states(self, states: np.ndarray):
        self.states = states

    def set_times(self, times: np.ndarray):
        self.times = times

    def set_len(self, length: int) -> None:
        self.length = length

    #   Core accessors
    def get_states(self) -> np.ndarray:
        return self.states

    def get_final_state(self) -> np.ndarray:
        return self.states[-1]

    def get_len(self) -> int:
        return self.length

    def get_times(self) -> np.ndarray:
        return self.times

    #   Functionalities
    def interpolate_times(self, num_steps: int) -> np.ndarray:
        interpolate = np.round(np.linspace(0, self.get_len() - 1, num_steps)).astype(
            int
        )
        return self.get_times()[interpolate]

    def interpolate_states(self, num_steps: int) -> np.ndarray:
        interpolate = np.round(np.linspace(0, self.get_len() - 1, num_steps)).astype(
            int
        )
        return self.get_states()[interpolate]


class KnapsackProblem:
    def __init__(self, profits: np.ndarray, weights: np.ndarray, capacity: int) -> None:
        self.set_profits(profits)
        self.set_weights(weights)
        self.set_capacity(capacity)
        self.set_num_items()

    #   Core mutators
    def set_profits(self, profits: np.ndarray) -> None:
        self.profits = profits

    def set_weights(self, weights: np.ndarray) -> None:
        self.weights = weights

    def set_capacity(self, capacity: int) -> None:
        self.capacity = capacity

    def set_num_items(self) -> None:
        self.num_items = self.get_profits().shape[0]

    #   Core accessors
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

    # Accessors
    def calculate_total_weight(self, item_bits: str) -> np.int64:
        total_weight = np.dot(self.weights, np.fromiter(map(int, item_bits), dtype=int))
        return total_weight

    def calculate_total_profit(self, item_bits: str) -> np.int64:
        total_profit = np.dot(self.profits, np.fromiter(map(int, item_bits), dtype=int))
        return total_profit


class MakeGraph:
    def __init__(self) -> None:
        self.probs = []
        self.simulated_spectrum = []
        self.computed_spectrum = []

    #   Core mutators
    def set_probs(self, probs: list) -> None:
        self.probs = probs

    def set_simulated_spectrum(self, simulated_spectrum: list) -> None:
        self.simulated_spectrum = simulated_spectrum

    def set_computed_spectrum(self, computed_spectrum: list) -> None:
        self.computed_spectrum = computed_spectrum

    #   Core accessors
    def get_probs(self) -> list:
        return self.probs

    def get_simulated_spectrum(self) -> tuple:
        return self.simulated_spectrum

    def get_computed_spectrum(self) -> tuple:

        return self.computed_spectrum


def wrap(func, name: str, args: tuple):
    start = time.time()
    result = func(args)
    end = time.time()

    print(f"{name}() executed in {end - start:.3f}s")
    return result
