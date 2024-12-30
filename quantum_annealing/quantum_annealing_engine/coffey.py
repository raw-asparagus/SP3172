from .engine import *
from qutip import Result, expect, mesolve
from matplotlib import gridspec
import matplotlib.pyplot as plt


class Coffey(KnapsackProblem):
    def __init__(self, profits: np.ndarray, weights: np.ndarray, capacity: int) -> None:
        super().__init__(profits, weights, capacity)
        self.set_params()

        self.H_0_state = "transverse"
        print(
            "Note that by default, H_0 will use take the form of a transverse Hamiltonian!\n"
        )

        self.H_0 = None
        self.set_gamma()
        self.H_P = None
        self.set_alpha()

    #   Idea: add eqn 2a Hamiltonian for H_A
    #   Rework to include old flag to toggle between eqn 2a, 3

    #   Core Setters
    def set_params(self) -> None:
        self.M = int(np.floor(np.log2(self.get_capacity())))
        self.total_qubits = self.get_num_items() + self.get_M() + 1
        self.num_states = np.power(2, self.total_qubits)

    def set_gamma(self, gamma: float = 1.0) -> None:
        self.gamma = gamma
        print(f"gamma parameter has been set to: {self.get_gamma()}")
        if self.gamma == 0.0:
            print("Note that setting gamma to 0.0 nullifies H_0!")
        print("")

        self.set_H_0()

    def set_alpha(self, alpha: float = 0.0) -> None:
        if alpha <= np.max(self.get_profits()):
            if alpha != 0.0:
                print("Invalid alpha parameter!")
            self.alpha = np.max(self.get_profits()) + 1
            print(f"alpha parameter has been set to default as: {self.get_alpha()}\n")
        else:
            self.alpha = alpha
            print(f"alpha parameter has been set to: {self.get_alpha()}\n")

        self.set_H_P()

    def set_H_0_state(self, H_0_state: str) -> None:
        match H_0_state:
            case "mixed":
                self.H_0_state = "mixed"
                print(f"H_0 set as {self.get_H_0_state()}!\n")

            case "transverse":
                self.H_0_state = "transverse"
                print(f"H_0 set as {self.get_H_0_state()}!\n")

            case "original":
                self.H_0_state = "original"
                print(f"H_0 set as {self.get_H_0_state()}!\n")

            case _:
                self.H_0_state = "transverse"
                print(
                    "Choice of H_0 failed! H_0 has been set to default as transverse!\nPlease try again!"
                )

        self.set_H_0()

    #   Setters
    def set_H_0(self) -> None:
        N = self.get_total_qubits()

        H_0 = 0
        match self.get_H_0_state():
            case "mixed":
                H_0 -= sum(
                    Gates.tensor_sigmax(i, N) * Gates.tensor_sigmax(j, N)
                    for i in range(N)
                    for j in range(i + 1, N)
                )

            case "transverse":
                H_0 -= sum(Gates.tensor_sigmax(i, N) for i in range(N))

            case "original":
                H_0 -= sum(Gates.tensor_bin(i, N) for i in range(N))

            case _:
                pass

        self.H_0 = self.get_gamma() * H_0

    def set_H_P(self) -> None:
        N = self.get_total_qubits()

        i_upper = j_lower = self.get_num_items()
        j_upper = self.get_num_items() + self.get_M()
        H_A1 = sum(
            np.power(2, j - j_lower) * Gates.tensor_bin(j, N)
            for j in range(j_lower, j_upper)
        )  #   starting from the 1st ancillary qubit
        H_A2 = (self.get_capacity() + 1 - np.power(2, self.get_M())) * Gates.tensor_bin(
            j_upper, N
        )
        H_A3 = -sum(self.get_weight(i) * Gates.tensor_bin(i, N) for i in range(i_upper))
        H_A = (H_A1 + H_A2 + H_A3) ** 2

        H_B = -sum(self.get_profit(i) * Gates.tensor_bin(i, N) for i in range(i_upper))

        self.H_P = self.get_alpha() * H_A + H_B

    #   Core Getters
    def get_M(self) -> int:
        return self.M

    def get_total_qubits(self) -> int:
        return self.total_qubits

    def get_num_states(self) -> int:
        return self.num_states

    def get_gamma(self) -> float:
        return self.gamma

    def get_alpha(self) -> float:
        return self.alpha

    def get_H_0_state(self) -> str:
        return self.H_0_state

    #   Getters
    def calculate_total_ancillary_weight(
        self, ancillary_bits: str, modifier_bit=None
    ) -> int:
        if modifier_bit is not None:
            modifier = int(modifier_bit) * (
                self.get_capacity() + 1 - np.power(2, self.get_M())
            )
        else:
            modifier = 0
        ancillary_weight = sum(
            np.power(2, i) for i, val in enumerate(ancillary_bits) if int(val) == 1
        )

        return modifier + ancillary_weight

    def get_H_0(self) -> Qobj:
        return self.H_0

    def get_H_P(self) -> Qobj:
        return self.H_P

    def get_H(self, s: float) -> Qobj:
        return (1 - s) * self.get_H_0() + s * self.get_H_P()

    #   Static functionalities
    @staticmethod
    def gen_ts(num_steps: int) -> np.ndarray:
        return np.linspace(0, 1, num_steps)

    #   Functionalities
    def anneal(self, num_steps: int) -> Result:
        ts = self.gen_ts(num_steps)
        psi0 = Observable.get_ground_eigenstate(self.get_H_0())
        res = mesolve(self.get_H, psi0, ts, e_ops=[])
        print("Quantum annealing complete!\n")

        return res

    def compute_probs(self, res: Result, num_steps) -> list:
        """Using result from anneal"""
        interpolate = np.round(np.linspace(0, len(res.states) - 1, num_steps)).astype(
            int
        )
        state_matrix = np.hstack(
            [psi.full() for psi in list(np.array(res.states)[interpolate])]
        )
        qubit_basis = Basis(self.get_total_qubits())
        probs_lst = np.power(
            np.abs(np.dot(qubit_basis.get_basis_matrix().T.conj(), state_matrix)), 2
        ).T.tolist()
        print("Simulated probabilities computed!\n")

        return probs_lst

    def simulate_spectrum(self, num_steps: int) -> list:
        """Independent from anneal"""
        spectrum_lst = [self.get_H(t).eigenenergies() for t in self.gen_ts(num_steps)]
        print("Spectrum computed!\n")

        return spectrum_lst

    def compute_spectrum(self, res: Result, num_steps: int) -> list:
        """Using result from anneal"""
        interpolate = np.round(np.linspace(0, len(res.states) - 1, num_steps)).astype(
            int
        )
        times = np.array(res.times)[interpolate]
        states = np.array(res.states)[interpolate]

        energies = [expect(self.get_H(t), psi) for t, psi in zip(times, states)]

        return energies


class MakeGraphCoffey(MakeGraph):
    def __init__(self) -> None:
        super().__init__()

    #   Static functionalities
    @staticmethod
    def tabulate_probs(coffey: Coffey, probs: list) -> list:
        final_probs = [
            (format(idx, f"0{coffey.get_total_qubits()}b"), prob)
            for idx, prob in enumerate(probs[-1])
        ]

        data = []
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

            data.append(
                [
                    f"| {item_bits} {ancillary_bits + modifier_bit} >",
                    prob,
                    item_weight,
                    ancillary_weight,
                    total_profit,
                ]
            )

        return data

    #   Functionalities
    def display_probs(self, coffey: Coffey) -> None:
        table_data = self.tabulate_probs(coffey, self.get_probs())[:5]

        for idx, val in enumerate(table_data):
            table_data[idx][1] = f"{val[1]:.4f}"

        # Create figure and axes with dynamic size
        num_rows = len(table_data)
        num_columns = len(table_data[0])
        row_height = 0.3
        col_width = max(len(str(x)) for row in table_data for x in row) * 0.2

        fig_width = col_width * num_columns
        fig_height = row_height * num_rows
        _, ax = plt.subplots(figsize=(fig_width, fig_height), dpi=300)
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
        table_data = self.tabulate_probs(coffey, self.get_probs())

        table_data = list(filter(lambda x: x[2] == x[3], table_data))
        subset_prob = np.sum(np.array(table_data)[:, 1].astype(float))

        for idx, val in enumerate(table_data):
            table_data[idx][
                1
            ] = f"{val[1] / subset_prob if subset_prob != 0 else 0:.4f} ({val[1]:.4f})"

        # Create figure and axes with dynamic size
        num_rows = len(table_data)
        num_columns = len(table_data[0])
        row_height = 0.3
        col_width = max(len(str(x)) for row in table_data for x in row) * 0.15

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

        for (i, _), cell in table.get_celld().items():
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
        probs_ts = coffey.gen_ts(len(self.get_probs()))
        actual_spectrum = self.get_computed_spectrum()
        low_spectrum = np.array(
            [ev[:num_energies] for ev in self.get_simulated_spectrum()]
        )
        spectrum_ts = coffey.gen_ts(len(self.get_simulated_spectrum()))

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
            probs_ts, [1 - sum(prob) for prob in self.get_probs()], ls="", marker="."
        )
        ax3.set(
            xlabel="Time units",
            ylabel=r"$1 - \sum P$",
            title=r"Total probability difference from $1$",
        )
        ax3.grid(True, alpha=0.25)

        ax2 = fig.add_subplot(gs[:, 1])
        for idx in range(low_spectrum.shape[1]):
            ax2.plot(spectrum_ts, low_spectrum[:, idx], c="grey", alpha=0.75)
        ax2.plot(spectrum_ts, actual_spectrum, c="blue")
        ax2.set(
            xlabel="Time units",
            ylabel="Energy eigenvalues",
            title="Evolution of energy eigenvalues against time",
        )
        ax2.grid(True, alpha=0.25)

        plt.tight_layout()
        plt.show()
