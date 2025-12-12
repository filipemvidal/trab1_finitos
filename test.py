import numpy as np
from frameDeflection import calculate_frame_deflection as cfd

nodes = [
    [0, 0],
    [0, 15],
    [0, 30],
    [0, 45],
    [0, 60],
    [10, 60],
    [20, 60]
]

x = np.array([i for [i, _] in nodes])
y = np.array([j for [_, j] in nodes])

nel = 6

el = 30 * 10**6  # elastic modulus
area = 2         # cross-sectional area
xi = 2/3

bcdof = [0, 1, 2]  # dofs constrained (transverse, axial, slope at node 1)
bcval = [0, 0, 0]

applied_forces = {19: -60}

NNEL = 2  # number of nodes per element
NDOF = 3   # number of dofs per node

cfd_results = cfd(nel, NNEL, NDOF, x, y, el, area, xi, bcdof, bcval, applied_forces)

print("\nDOF   Displacement (m)")
for dof in sorted(cfd_results.keys()):
    print(f"{dof:3d}   {cfd_results[dof]:.6e}")