import numpy as np

def generate_knapsack_tests(N=50):
    tests = []
    
    for _ in range(N):
        M = np.random.randint(1, 201)  # max number of items
        data = np.zeros((M, 2), dtype=int)
        data[:, 0] = np.random.randint(25, 43, size=M)  # weights
        data[:, 1] = np.random.randint(35, 59, size=M)  # values
        
        vec = np.random.randint(0, 2, size=M)  # 0 or 1 for selected items
        Price = np.sum(data[:, 0] * vec)  # total weight of selected items
        
        tests.append({
            'Price': Price,
            'data': data,
            'vec': vec
        })
    
    return tests

