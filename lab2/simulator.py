import os
import numpy as np
import matplotlib.pyplot as plt
from sor import relax

def main_simulation(nx=101, ny=101, Lx=1.0, Ly=1.0,
                    sources=[(None, None)], amplitudes=[1e-4], KK=[10],
                    dt=1e-2, max_step=500,
                    boundary="neumann", outdir="output"):

    os.makedirs(outdir, exist_ok=True)

    # Grid setup
    hx = Lx / (nx - 1)
    hy = Ly / (ny - 1)
    x = np.linspace(0, Lx, nx)
    y = np.linspace(0, Ly, ny)

    # Initial conditions
    U = np.zeros((nx, ny))
    U0 = U.copy()
    U1 = U.copy()
    d = np.zeros((nx, ny))
    dU = np.zeros((nx, ny))
    U1[1:nx-1, 1:ny-1] = U0[1:nx-1, 1:ny-1] + dt * dU[1:nx-1, 1:ny-1]

    # Coefficients
    ax = 0.5 * np.ones((nx, ny)) / (hx**2)
    cx = ax.copy()
    ay = 0.5 * np.ones((nx, ny)) / (hy**2)
    cy = ay.copy()
    b = ax + cx + ay + cy + 1.0 / (dt**2)

    t = [0.0, dt]
    index = None
    jndex = None
    U_store = np.zeros((nx, ny, max_step))

    # Map sources to grid indices if None
    sources_idx = []
    for s in sources:
        i = nx // 2 if s[0] is None else int(s[0])
        j = ny // 2 if s[1] is None else int(s[1])
        sources_idx.append((i,j))

    for k in range(max_step):
        t.append(t[-1] + dt)

        # Build RHS
        for i in range(1, nx-1):
            for j in range(1, ny-1):
                d[i,j] = (-2.0 * U1[i,j] + U0[i,j]) / (dt**2) - 0.5 * (
                    (U0[i-1,j] - 2.0*U0[i,j] + U0[i+1,j]) / (hx**2) +
                    (U0[i,j-1] - 2.0*U0[i,j] + U0[i,j+1]) / (hy**2)
                )

        # Apply sources
        for idx, (i,j) in enumerate(sources_idx):
            ax[i,j] = ay[i,j] = cx[i,j] = cy[i,j] = 0.0
            b[i,j] = 1.0
            d[i,j] = amplitudes[idx] * np.sin(np.pi * KK[idx] * k * dt)

        # Boundary conditions
        if boundary.lower() == "neumann":  # free edges
            b[:,1] -= ay[:,1]; ay[:,1] = 0.0
            b[:,ny-2] -= cy[:,ny-2]; cy[:,ny-2] = 0.0
            b[1,:] -= ax[1,:]; ax[1,:] = 0.0
            b[nx-2,:] -= cx[nx-2,:]; cx[nx-2,:] = 0.0
        elif boundary.lower() == "dirichlet":  # fixed edges
            U[0,:] = 0; U[-1,:] = 0
            U[:,0] = 0; U[:,-1] = 0

        # Relaxation step
        U = relax(ax, ay, cx, cy, b, d, nx, ny, U, eps=1e-6, max_iter=200)

        # Edge correction for Neumann
        if boundary.lower() == "neumann":
            U[:,0] = U[:,1]
            U[:,-1] = U[:,-2]
            U[0,:] = U[1,:]
            U[-1,:] = U[-2,:]
            # Corners averaging
            U[0,0] = 0.5 * (U[1,0] + U[0,1])
            U[-1,-1] = 0.5 * (U[-2,-1] + U[-1,-2])
            U[0,-1] = 0.5 * (U[0,-2] + U[1,-1])
            U[-1,0] = 0.5 * (U[-1,1] + U[-2,0])

        # Store field
        U_store[:,:,k] = U.copy()
        U0, U1 = U1, U.copy()

        # Track resonance for first source
        i0,j0 = sources_idx[0]
        if abs(d[i0,j0] + amplitudes[0]) < amplitudes[0]/20.0:
            index = k
        if abs(d[i0,j0] - amplitudes[0]) < amplitudes[0]/20.0:
            jndex = k

        if (k % 100) == 0:
            print(f"{k:5d} step")

    # Defaults if not found
    if index is None or jndex is None:
        index = 0
        jndex = max(1, max_step//2)

    # Chladni-like mask
    FIGURE = np.abs(U_store[:,:,index] - U_store[:,:,jndex]) < (amplitudes[0]/5.0)

    # Save arrays
    np.savez(os.path.join(outdir, "results.npz"),
             U_store=U_store, x=x, y=y,
             params=dict(nx=nx, ny=ny, Lx=Lx, Ly=Ly,
                         amplitudes=amplitudes, KK=KK,
                         dt=dt, max_step=max_step,
                         boundary=boundary),
             index=index, jndex=jndex)

    # Save Chladni mask
    plt.figure(figsize=(6,6))
    plt.imshow(FIGURE.T, origin='lower', extent=(x[0], x[-1], y[0], y[-1]), aspect='equal')
    plt.title(f"Chladni-like figure ({boundary})")
    plt.xlabel("x"); plt.ylabel("y")
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "figure.png"), dpi=150)
    plt.close()

    # Save snapshots
    num_snapshots = min(6, max_step)
    steps = np.linspace(0, max_step-1, num_snapshots, dtype=int)
    for s in steps:
        fig, ax3 = plt.subplots(figsize=(5,4))
        im = ax3.imshow(U_store[:,:,s].T, origin='lower',
                        extent=(x[0], x[-1], y[0], y[-1]), aspect='auto')
        ax3.set_title(f"Surface snapshot k={s}")
        plt.colorbar(im, ax=ax3)
        fname = os.path.join(outdir, f"surface_step_{s:03d}.png")
        plt.tight_layout()
        plt.savefig(fname, dpi=150)
        plt.close()

    print("Saved results to", outdir)
