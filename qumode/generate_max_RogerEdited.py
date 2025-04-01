import glob
import os
import pickle
import re
from pathlib import Path

import numpy as np
from src.quantum_knapsack import *


def simulate(knapsack, penalty_scale, initial_h_strength, energy_scale, time_scale, max_iterations) -> float:
    solution: SolutionAnalyzer = SolutionAnalyzer(knapsack)
    coffey: Mapping = Coffey(knapsack, penalty_scale)
    basis: QumodeBasis = StandardBasis(coffey.dimension)
    coffey.initialize(basis)
    solver: QuantumAnnealer = QuantumAnnealer(basis, coffey)
    solver.initialize(AnnealingParameters(initial_h_strength, energy_scale, time_scale, max_iterations))
    solver.solve()
    result = solver.get_result(max_iterations)

    solution_probability = []
    for prob in result.probabilities:
        solution_probability.append(solution.get_solution_probability(prob))

    return solution_probability

profits = np.array([4, 1])
weights = np.array([3, 3])
capacity = 3
PROBLEM_INSTANCE = Knapsack(profits, weights, capacity)
print(PROBLEM_INSTANCE)

target_dir_2 = os.path.join(".", "beta_specific_instance_max")
if not os.path.exists(target_dir_2):
    os.makedirs(target_dir_2)
    print(f"Created directory: {target_dir_2}\n")

file_pattern_2 = os.path.join(target_dir_2, "knapsack-probs-*.pkl")

def extract_numbers(fn):
    filenames = glob.glob(fn)
    pattern = r"knapsack-probs-(\d+(?:\.\d+)?)_(\d+(?:\.\d+)?).pkl"
    return [(float(re.search(pattern, file).group(1)), float(re.search(pattern, file).group(2))) for file in filenames if re.search(pattern, file)]

def main():
    for step in map(lambda x: 2 ** x, range(6, 7)):
        folder_data = extract_numbers(file_pattern_2)
        for alpha in np.logspace(-1, 6, 8 * step - step + 1)[::-1]:
            for beta in np.logspace(-3, 4, 8 * step - step + 1):
                print(f"Generating probability for alpha = {alpha} and beta = {beta}")

                PENALTY_SCALE = np.max(PROBLEM_INSTANCE.profits).astype(float) + 1.0
                TIME_SCALE = 0.01
                ITERATIONS = 1000
                instance_probs = simulate(PROBLEM_INSTANCE, PENALTY_SCALE, beta, alpha, TIME_SCALE * ITERATIONS,
                                            ITERATIONS)

                file_path_probs = os.path.join(target_dir_2, f"knapsack-probs-{alpha}_{beta}.pkl")
                with open(file_path_probs, "wb") as file:
                    pickle.dump(instance_probs[-1], file)
                print(f"Probabilities have been saved to {file_path_probs}")

                print(f"{instance_probs[-1]:.4f}\n")

if __name__ == "__main__":
    main()
