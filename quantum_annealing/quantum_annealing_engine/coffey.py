from .engine import *
from qutip import Result, mesolve
from matplotlib import gridspec
import matplotlib.pyplot as plt


class Coffey(KnapsackProblem):
    def __init__(self, profits: np.ndarray, weights: np.ndarray, capacity: int) -> None:
        super().__init__(profits, weights, capacity)
        self.set_params()
        self.set_alpha()
        self.set_H_0("transverse")
        print(
            "Note that by default Initial Hamiltonian will use transverse Hamiltonian!\n"
        )
        self.set_H_P()

    #   Setters
    #   Idea: rework to add 'old' flag to swap between old and new
    def set_params(self) -> None:
        #   Based on eqn 3
        self.M = int(np.floor(np.log2(self.get_capacity())))

        #   Based on eqn 3
        self.total_qubits = self.get_num_items() + self.get_M() + 1

        self.num_states = np.power(2, self.total_qubits)

    def set_alpha(self, alpha: float = 0.0) -> None:
        if alpha == 0.0:
            self.alpha = np.max(self.get_profits()) + 1
        else:
            self.alpha = alpha
            print(f"alpha parameter set to: {alpha}!\n")

    def set_H_0(self, H_0_state: str) -> None:
        N = self.get_total_qubits()

        #   Select between mixed and transverse Hamiltonian
        H_0 = 0
        match H_0_state:
            case "mixed":
                self.H_0_state = "mixed"
                for i in range(N):
                    for j in range(i + 1, N):
                        # Pauli-X on the ith and jth qubit, identity on others
                        operators = [
                            qeye(2) if k != i and k != j else sigmax() for k in range(N)
                        ]
                        H_0 -= tensor(operators)
                print("Initial Hamiltonian set as mixed!\n")
            case "transverse":
                self.H_0_state = "transverse"
                H_0 -= sum(Pauli.tensor_sigmax(i, N) for i in range(N))
                print("Initial Hamiltonian set as transverse!\n")
            case _:
                print("Set choice of initial Hamiltonian failed! Please try again!\n")

        self.H_0 = H_0

    #   Idea: rework to add 'old' flag to swap between old and new
    def set_H_P(self) -> None:
        N = self.get_total_qubits()

        #   Based on eqn 3
        i_upper = j_lower = self.get_num_items()
        j_upper = self.get_num_items() + self.get_M()
        H_A1 = sum(
            np.power(2, j - j_lower) * Pauli.tensor_sigmaz(j, N)
            for j in range(j_lower, j_upper)
        )  #   starting from the 1st ancillary qubit

        H_A2 = (
            self.get_capacity() + 1 - np.power(2, self.get_M())
        ) * Pauli.tensor_sigmaz(j_upper, N)

        H_A3 = -sum(
            self.get_weight(i) * Pauli.tensor_sigmaz(i, N) for i in range(i_upper)
        )

        H_A = (H_A1 + H_A2 + H_A3) ** 2

        H_B = -sum(
            self.get_profit(i) * Pauli.tensor_sigmaz(i, N) for i in range(i_upper)
        )

        self.H_P = self.get_alpha() * H_A + H_B

    #   Core Getters
    def get_M(self) -> int:
        return self.M

    def get_total_qubits(self) -> int:
        return self.total_qubits

    def get_num_states(self) -> int:
        return self.num_states

    def get_alpha(self) -> float:
        return self.alpha

    def get_H_0_state(self) -> str:
        return self.H_0_state

    #   Getters
    #   Idea: rework to add 'old' flag to swap between old and new
    def calculate_total_ancillary_weight(
        self, ancillary_bits, modifier_bit=None
    ) -> int:
        if modifier_bit is not None:
            modifier = int(modifier_bit) * (
                self.get_capacity() - np.power(2, len(ancillary_bits) - 1)
            )
        else:
            modifier = 0
        ancillary_weight = sum(
            np.power(2, idx) for idx, val in enumerate(ancillary_bits) if int(val) == 1
        )
        return modifier + ancillary_weight
    
    def get_H_0(self) -> Qobj:
        return self.H_0
    
    def get_H_P(self) -> Qobj:
        return self.H_P

    def get_H(self, s: float) -> Qobj:
        return (1 - s) * self.get_H_0() + s * self.get_H_P()

    #   Functionalities
    def anneal(self, num_steps: int) -> Result:
        ts = np.linspace(0, 1, num_steps)
        psi0 = Observable.get_ground_eigenstate(self.get_H_0())
        res = mesolve(self.get_H, psi0, ts, e_ops=[])
        print("Quantum annealing complete!\n")
        return res

    def compute_probs(self, res: Result) -> list:
        """Using result from anneal"""
        state_matrix = np.hstack([psi.full() for psi in res.states])
        qubit_basis = Basis(self.get_total_qubits())
        probs_lst = np.power(
            np.abs(np.dot(qubit_basis.basis_matrix.T.conj(), state_matrix)), 2
        ).T.tolist()
        print("Probabilities computed!\n")
        return probs_lst

    def compute_spectrum(self, num_steps: int) -> list:
        """Independent from anneal"""
        ts = np.linspace(0, 1, num_steps)
        hs = [self.get_H(t) for t in ts]
        spectrum_lst = [h.eigenenergies() for h in hs]
        print("Spectrum computed!\n")
        return spectrum_lst


class MakeGraphCoffey(MakeGraph):
    def __init__(self) -> None:
        super().__init__()

    #   Functionalities
    def display_probs(self, coffey: Coffey) -> None:
        final_probs = [
            (format(idx, f"0{coffey.get_total_qubits()}b"), prob)
            for idx, prob in enumerate(self.get_probs()[-1])
        ]

        #   Get the top 5 most probable basis in the final state
        table_data = []
        for state_label, prob in sorted(final_probs, key=lambda x: x[1], reverse=True)[
            :5
        ]:
            item_bits = state_label[: coffey.get_num_items()]
            ancillary_bits = state_label[
                coffey.get_num_items() : coffey.get_num_items() + coffey.get_M()
            ]
            modifier_bit = state_label[coffey.get_num_items() + coffey.get_M()]

            item_weight = coffey.calculate_total_weight(item_bits)
            total_profit = coffey.calculate_total_profit(item_bits)
            ancillary_weight = coffey.calculate_total_ancillary_weight(
                ancillary_bits, modifier_bit
            )

            table_data.append(
                [
                    f"| {item_bits} {ancillary_bits + modifier_bit} >",
                    f"{prob:.4f}",
                    item_weight,
                    ancillary_weight,
                    total_profit,
                ]
            )

        # Create figure and axes with dynamic size
        num_rows = len(table_data)
        num_columns = len(table_data[0])
        row_height = 0.3
        col_width = max(len(str(x)) for row in table_data for x in row) * 0.2

        fig_width = col_width * num_columns
        fig_height = row_height * num_rows
        _, ax = plt.subplots(figsize=(fig_width, fig_height), dpi=300)
        ax.axis("tight")
        ax.axis("off")
        ax.set(title="Top 5 states sorted by probability in descending order")
        table = ax.table(
            cellText=table_data,
            colLabels=[
                "State",
                "Probability",
                "Item Weight",
                "Ancillary Weight",
                "Total Profit",
            ],
            cellLoc="center",
            loc="center",
        )
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1.2, 1.2)

        for (i, _), cell in table.get_celld().items():
            if i == 0:
                cell.set_fontsize(12)
                cell.set_facecolor("#40466e")
                cell.set_text_props(color="w")

        plt.show()

    def display_filtered_probs(self, coffey: Coffey) -> None:
        final_probs = [
            (format(idx, f"0{coffey.get_total_qubits()}b"), prob)
            for idx, prob in enumerate(self.get_probs()[-1])
        ]

        subset_prob = 0
        table_data = []
        for state_label, prob in sorted(final_probs, key=lambda x: x[1], reverse=True):
            item_bits = state_label[: coffey.get_num_items()]
            ancillary_bits = state_label[
                coffey.get_num_items() : coffey.get_num_items() + coffey.get_M()
            ]
            modifier_bit = state_label[coffey.get_num_items() + coffey.get_M()]

            item_weight = coffey.calculate_total_weight(item_bits)
            total_profit = coffey.calculate_total_profit(item_bits)
            ancillary_weight = coffey.calculate_total_ancillary_weight(
                ancillary_bits, modifier_bit
            )

            if item_weight == ancillary_weight:
                subset_prob += prob
                table_data.append(
                    [
                        f"| {item_bits} {ancillary_bits + modifier_bit} >",
                        prob,
                        item_weight,
                        ancillary_weight,
                        total_profit,
                    ]
                )

        for idx, val in enumerate(table_data):
            table_data[idx][1] = f"{val[1] / subset_prob:.4f}"

        # Create figure and axes with dynamic size
        num_rows = len(table_data)
        num_columns = len(table_data[0])
        row_height = 0.3
        col_width = max(len(str(x)) for row in table_data for x in row) * 0.2

        fig_width = col_width * num_columns
        fig_height = row_height * num_rows
        _, ax = plt.subplots(figsize=(fig_width, fig_height), dpi=300)
        ax.axis("tight")
        ax.axis("off")
        ax.set(title=r"Filtered states (Item Weight $=$ Ancillary Weight)")
        table = ax.table(
            cellText=table_data,
            colLabels=[
                "Filtered State",
                "Adjusted Probability",
                "Item Weight",
                "Ancillary Weight",
                "Total Profit",
            ],
            cellLoc="center",
            loc="center",
        )
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1.2, 1.2)

        for (i, ), cell in table.get_celld().items():
            if i == 0:
                cell.set_fontsize(12)
                cell.set_facecolor("#40466e")
                cell.set_text_props(color="w")
            elif i > 0:
                item_weight = float(table_data[i - 1][2])
                if item_weight > coffey.get_capacity():
                    for k in range(len(table_data[i - 1])):
                        table[(i, k)].set_facecolor("#ffcccc")

        plt.show()

    def display_graph(self, coffey: Coffey, num_energies=5) -> None:
        probs_ts = np.linspace(0, 1, len(self.get_probs()))
        low_spectrum = np.array([ev[:num_energies] for ev in self.get_spectrum()])
        spectrum_ts = np.linspace(0, 1, len(self.get_spectrum()))

        fig = plt.figure(figsize=(20, 7), dpi=300)
        gs = gridspec.GridSpec(2, 2, height_ratios=[5, 1])

        ax1 = fig.add_subplot(gs[0, 0])

        for idx in range(coffey.get_num_states()):
            ax1.plot(
                probs_ts,
                [prob[idx] for prob in self.get_probs()],
                label=f'|{format(idx, "0" + str(coffey.get_total_qubits()) + "b")}>',
            )
            ax1.text(
                probs_ts[-1],
                self.get_probs()[-1][idx],
                f'|{format(idx, "0" + str(coffey.get_total_qubits()) + "b")}>',
                fontsize=8,
                horizontalalignment="center",
                verticalalignment="bottom",
            )

        ax1.set(
            xlabel="Time units",
            ylabel="Probability",
            title="Probabilities of states against time",
        )
        ax1.grid(True, alpha=0.25)

        ax3 = fig.add_subplot(gs[1, 0])

        ax3.semilogy(
            probs_ts,
            [1 - sum(prob) for prob in self.get_probs()],
            ls="",
            marker=".",
            ms=1,
        )

        ax3.set(
            xlabel="Time units",
            ylabel=r"$1 - \sum P$",
            title=r"Total probability difference from $1$",
        )
        ax3.grid(True, alpha=0.25)

        ax2 = fig.add_subplot(gs[:, 1])

        for idx in range(num_energies):
            ax2.plot(spectrum_ts, low_spectrum[:, idx])

        ax2.set(
            xlabel="Time units",
            ylabel="Energy eigenvalues",
            title="Evolution of energy eigenvalues against time",
        )
        ax2.grid(True, alpha=0.25)

        plt.tight_layout()

        plt.show()
