import numpy as np

def relax(ax, ay, cx, cy, b, d, n, m, U, eps=1e-6, alfa=1.84, max_iter=200):
    """
    Successive Over-Relaxation (SOR) solver for Helmholtz-like equation.
    """
    err = 1.0
    it = 0
    while err > eps and it < max_iter:
        err = 0.0
        it += 1
        for i in range(1, n-1):
            for j in range(1, m-1):
                t = (ax[i,j]*U[i-1,j] + cx[i,j]*U[i+1,j] +
                     ay[i,j]*U[i,j-1] + cy[i,j]*U[i,j+1] - d[i,j]) / b[i,j]
                deviation = abs(U[i,j] - t)
                if deviation > err:
                    err = deviation
                U[i,j] = t*alfa + U[i,j]*(1-alfa)
    return U