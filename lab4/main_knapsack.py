import numpy as np
import time
import matplotlib.pyplot as plt
from greedy import KnapsackSolver
from simulated_annealing import SimulateAnnealingKnapsackSolver
from generate_knapsack import generate_knapsack_tests

def main(num_tests=5):
    greedy_results = []
    sa_results = []

    tests = generate_knapsack_tests(num_tests)

    for i, test in enumerate(tests):
        data = test['data']
        max_weight = test['Price']  
        solver = KnapsackSolver(data, max_weight)

        # Greedy
        start_time = time.time()
        g_indexes, g_weight, g_value = solver.greedy_search()
        greedy_time = time.time() - start_time

        # Simulated Annealing
        solver = SimulateAnnealingKnapsackSolver(data, max_weight)
        start_time = time.time()
        sa_value, sa_weight, sa_indexes, sa_iters = solver.simulated_annealing(max_iter=5000, t0=1000, tn=0.01, k=1)
        sa_time = time.time() - start_time

        greedy_results.append((g_value, greedy_time))
        sa_results.append((sa_value, sa_time))

        print(f"Test {i+1}:")
        print(f"  Greedy -> Value: {g_value}, Weight: {g_weight}, Time: {greedy_time:.6f} s")
        print(f"  SA     -> Value: {sa_value}, Weight: {sa_weight}, Time: {sa_time:.6f} s, Iter: {sa_iters}")
        print("-"*50)



if __name__ == "__main__":
    main(num_tests=5)