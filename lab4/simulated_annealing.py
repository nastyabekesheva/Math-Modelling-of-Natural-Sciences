import numpy as np

class SimulatedAnnealingTSP:
    def __init__(self, x, y, t_start, t_end, alpha, epochs):
        self.x = np.array(x)
        self.y = np.array(y)
        self.cities = np.column_stack([self.x, self.y])  # shape (N,2)
        self.N = len(x)
        self.t_start = t_start
        self.t_end = t_end
        self.alpha = alpha
        self.epochs = epochs

    def energy(self, state):
        state = state.flatten()
        state_prev = state
        state_next = np.roll(state, -1)
        diffs = self.cities[state_next, :] - self.cities[state_prev, :]
        sq_dists = diffs[:, 0]**2 + diffs[:, 1]**2
        return np.sum(np.sqrt(sq_dists))

    def state(self, state):
        n = len(state)
        i = np.random.randint(0, n)
        j = np.random.randint(0, n)

        new_state = state.copy()
        if i > j:
            new_state[j:i+1] = new_state[j:i+1][::-1]
        else:
            new_state[i:j+1] = new_state[i:j+1][::-1]
        return new_state

    def solve(self, log_interval=500):
        arg = []
        energy = []
        temperature = []
        snapshots = []   # store states for GIF

        n = self.N
        state = np.random.permutation(n).reshape(-1, 1)

        E = self.energy(state)         
        T = self.t_start                          

        best_state = state.copy()          
        best_E = E  

        for k in range(1, self.epochs + 1):
            new_state = self.state(state)
            new_E = self.energy(new_state)

            if new_E < E:
                E = new_E
                state = new_state.copy()
            else:
                p = np.exp(-(new_E - E) / T)
                if np.random.rand() < p:
                    E = new_E
                    state = new_state.copy()
            
            if E < best_E:
                best_E = E
                best_state = state.copy()

            T = self.t_start * self.alpha / k

            if k % log_interval == 0:
                arg.append(k)
                energy.append(E)
                temperature.append(T)
                snapshots.append(state.copy())

            if T <= self.t_end:
                break

        return best_state, best_E, np.array(arg), np.array(energy), np.array(temperature), snapshots

import numpy as np

class SimulateAnnealingKnapsackSolver:
    def __init__(self, data, max_weight):
        self.data = data
        self.max_weight = max_weight
        self.N = data.shape[0]
        self.weights = data[:, 0]
        self.values = data[:, 1]

    def greedy_search(self):
        indexed_data = np.hstack((np.arange(1, self.N + 1).reshape(-1, 1), self.data))
        sorted_data = indexed_data[indexed_data[:, 2].argsort()[::-1]]

        total_weight = 0
        total_value = 0
        indexes = []

        for i in range(self.N):
            if total_weight + sorted_data[i, 1] <= self.max_weight:
                indexes.append(int(sorted_data[i, 0]))
                total_weight += sorted_data[i, 1]
                total_value += sorted_data[i, 2]

        indexes.sort()
        return indexes, total_weight, total_value

    def simulated_annealing(self, t0=1000, tn=0.01, max_iter=10000, k=1):
        sol = np.zeros(self.N, dtype=int)
        best_sol = sol.copy()
        best_value = self._objective(sol)
        t = t0
        iter_count = 0

        while iter_count < max_iter and t > tn:
            iter_count += 1
            sol_candidate = sol.copy()
            
            # Flip k random items
            flip_indices = np.random.randint(0, self.N, size=k)
            sol_candidate[flip_indices] = 1 - sol_candidate[flip_indices]

            f_candidate = self._objective(sol_candidate)
            d_energy = self._objective(sol) - f_candidate

            if d_energy <= 0:
                if f_candidate > best_value:
                    best_sol = sol_candidate.copy()
                    best_value = f_candidate
                sol = sol_candidate
            else:
                p = np.exp(-d_energy / t)
                if f_candidate != 0 and p > np.random.rand():
                    sol = sol_candidate

            t = t0 / iter_count  # cooling schedule

        indexes = list(np.where(best_sol == 1)[0] + 1)  # 1-based indices
        total_weight = int(np.sum(self.weights[best_sol == 1]))

        return best_value, total_weight, indexes, iter_count

    def _objective(self, sol):
        total_weight = np.sum(self.weights[sol == 1])
        if total_weight > self.max_weight:
            return 0
        return int(np.sum(self.values[sol == 1]))

