from simulator import main_simulation

# Base output folder
base_outdir = "lab2/output"

# -------------------------
# 4.1 Single source, free edges (Neumann)
# -------------------------
# Center source, amplitude=1e-4
print("Running 4.1: single center source, free edges")
main_simulation(nx=101, ny=101,
                sources=[(None, None)],
                amplitudes=[1e-4],
                KK=[10],            # 10π
                dt=1e-2,
                max_step=500,
                boundary="neumann",
                outdir=f"{base_outdir}/free_center_K10pi")

main_simulation(nx=101, ny=101,
                sources=[(None, None)],
                amplitudes=[1e-4],
                KK=[3],             # 3π
                dt=1e-2,
                max_step=500,
                boundary="neumann",
                outdir=f"{base_outdir}/free_center_K3pi")

main_simulation(nx=101, ny=101,
                sources=[(None, None)],
                amplitudes=[1e-4],
                KK=[25],            # 25π
                dt=1e-2,
                max_step=500,
                boundary="neumann",
                outdir=f"{base_outdir}/free_center_K25pi")


# -------------------------
# 4.2 Two sources, free edges
# -------------------------
print("Running 4.2: two sources on left and right edges, free edges")
sources_two = [(0, 50), (100, 50)]  # Left and right middle points
main_simulation(nx=101, ny=101,
                sources=sources_two,
                amplitudes=[1e-4, 1e-4],
                KK=[3, 3],          # 3π
                dt=1e-2,
                max_step=500,
                boundary="neumann",
                outdir=f"{base_outdir}/two_sources_K3pi")

main_simulation(nx=101, ny=101,
                sources=sources_two,
                amplitudes=[1e-4, 1e-4],
                KK=[6, 6],          # 6π
                dt=1e-2,
                max_step=500,
                boundary="neumann",
                outdir=f"{base_outdir}/two_sources_K6pi")


# -------------------------
# 4.3 Single source, fixed edges (Dirichlet)
# -------------------------
print("Running 4.3: single center source, fixed edges")
main_simulation(nx=101, ny=101,
                sources=[(None, None)],
                amplitudes=[1e-4],
                KK=[10],
                dt=1e-2,
                max_step=500,
                boundary="dirichlet",
                outdir=f"{base_outdir}/fixed_center_K10pi")

main_simulation(nx=101, ny=101,
                sources=[(None, None)],
                amplitudes=[1e-4],
                KK=[25],
                dt=1e-2,
                max_step=500,
                boundary="dirichlet",
                outdir=f"{base_outdir}/fixed_center_K25pi")

print("All simulations completed. Results saved in lab2_output/")
