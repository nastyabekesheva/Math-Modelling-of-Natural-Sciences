import random
import numpy as np

class GreedyTSP:
    def __init__(self, x, y):
        self.x = np.array(x)
        self.y = np.array(y)
        self.N = len(x)

    def distance_matrix(self):
        A = np.zeros((self.N, self.N))

        for i in range(self.N):
            A[i, i] = np.inf
            for j in range(i + 1, self.N):
                A[i, j] = np.sqrt((self.x[i] - self.x[j]) ** 2 + (self.y[i] - self.y[j]) ** 2)
                A[j, i] = A[i, j]

        return A

    def solve(self, x, y):
        node = 0
        dist = 0
        way = np.array([node])

        A = self.distance_matrix()

        for _ in range(1, self.N):
            tmp = A[node, :].copy()
            tmp[way] = np.inf

            min_dist = np.min(tmp)
            node = np.argmin(tmp)

            dist += min_dist
            way = np.append(way, node)

        dist += A[way[-1], way[0]]
        way = np.append(way, way[0])

        path = []
        for i in range(len(way) - 1):
            path.append([way[i], way[i+1]])

        path = np.array(path)

        return path, way, dist
    
class KnapsackSolver:
    def __init__(self, data, max_weight):
        self.data = data
        self.max_weight = max_weight
        self.N = data.shape[0]

    def greedy_search(self):
        # Add 1-based index column
        indexed_data = np.hstack((np.arange(1, self.N + 1).reshape(-1, 1), self.data))
        
        # Sort by value descending
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

