# main.py

import os
import time
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import pandas as pd

from greedy import GreedyTSP
from simulated_annealing import SimulatedAnnealingTSP


def run_greedy(x, y, exp_id):
    start = time.time()
    tsp = GreedyTSP(x, y)
    path, way, dist = tsp.solve(x, y)
    end = time.time()

    # Visualization
    plt.figure(figsize=(6, 6))
    plt.scatter(x, y, s=100, c='red', zorder=2)
    for i, j in path:
        plt.plot([x[i], x[j]], [y[i], y[j]], 'b-', zorder=1)
    for idx, (xi, yi) in enumerate(zip(x, y)):
        plt.text(xi + 0.01, yi + 0.01, str(idx), fontsize=8)
    plt.title(f"Greedy TSP Tour (n={len(x)})")
    plt.xlim(-0.1, 1.1)
    plt.ylim(-0.1, 1.1)
    plt.grid(True)

    fname = f"lab4/output/greedy_exp{exp_id}.png"
    plt.savefig(fname, dpi=200)
    plt.close()
    print(f"[Greedy] Exp {exp_id} saved to {fname}")

    return dist, end - start


def run_sa(x, y, exp_id):
    start = time.time()
    sa = SimulatedAnnealingTSP(x, y, t_start=10, t_end=1e-3, alpha=0.99, epochs=20000)
    best_state, best_E, arg, energy, temperature, snapshots = sa.solve(log_interval=500)
    end = time.time()

    # --- GIF creation ---
    fig, (ax_tsp, ax_energy, ax_temp) = plt.subplots(1, 3, figsize=(18, 6))

    def update(frame):
        ax_tsp.clear()
        ax_energy.clear()
        ax_temp.clear()

        state = snapshots[frame].flatten()
        path_x = np.append(x[state], x[state[0]])
        path_y = np.append(y[state], y[state[0]])
        ax_tsp.plot(path_x, path_y, 'b-', zorder=1)
        ax_tsp.scatter(x, y, c='red', s=50, zorder=2)
        for idx, (xi, yi) in enumerate(zip(x, y)):
            ax_tsp.text(xi + 0.01, yi + 0.01, str(idx), fontsize=8)
        ax_tsp.set_title(f"TSP Path (iter {arg[frame]})")
        ax_tsp.set_xlim(0, 1)
        ax_tsp.set_ylim(0, 1)

        ax_energy.plot(arg[:frame+1], energy[:frame+1], color="blue")
        ax_energy.set_xlabel("Iteration")
        ax_energy.set_ylabel("Energy")
        ax_energy.set_title("Energy Progression")

        ax_temp.plot(arg[:frame+1], temperature[:frame+1], color="red")
        ax_temp.set_xlabel("Iteration")
        ax_temp.set_ylabel("Temperature")
        ax_temp.set_title("Temperature Progression")

    ani = animation.FuncAnimation(fig, update, frames=len(snapshots), interval=100, repeat=False)
    fname = f"lab4/output/sa_exp{exp_id}.gif"
    ani.save(fname, writer="pillow")
    plt.close(fig)
    print(f"[SA] Exp {exp_id} saved to {fname}")

    return best_E, end - start


def main():
    os.makedirs("lab4/output", exist_ok=True)

    experiments = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]  # different TSP sizes
    results = []

    for exp_id, n_cities in enumerate(experiments, start=1):
        print(f"\n=== Experiment {exp_id}: n_cities={n_cities} ===")
        x = np.random.rand(n_cities)
        y = np.random.rand(n_cities)

        greedy_dist, greedy_time = run_greedy(x, y, exp_id)
        sa_dist, sa_time = run_sa(x, y, exp_id)

        results.append({
            "Experiment": exp_id,
            "Cities": n_cities,
            "Greedy_Dist": greedy_dist,
            "Greedy_Time(s)": round(greedy_time, 3),
            "SA_Dist": sa_dist,
            "SA_Time(s)": round(sa_time, 3),
        })

    # Save results table
    df = pd.DataFrame(results)
    df.to_csv("lab4/output/results.csv", index=False)
    print("\n=== Results ===")
    print(df)


if __name__ == "__main__":
    main()
