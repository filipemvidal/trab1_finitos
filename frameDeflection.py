"""
Purpose
    Find the deflection of a frame.

Variable descriptions
    x and y = global x and y coordinates of each node
    k = element stiffness matrix
    kk = system stiffness matrix
    ff = system force vector
    index = a vector containing system dofs associated with each element
    bcdof = a vector containing dofs associated with boundary conditions
    bcval = a vector containing boundary condition values associated with
            the dofs in 'bcdof'
"""

import numpy as np
from femUtils import *


# Problem parameters
nel = 6            # number of elements
nnel = 2           # number of nodes per element
ndof = 3           # number of dofs per node
nnode = (nnel - 1) * nel + 1  # total number of nodes in system
sdof = nnode * ndof            # total system dofs

# Node coordinates
x = np.zeros(nnode)
y = np.zeros(nnode)

x[0] = 0;   y[0] = 0    # x, y coord. values of node 1 in terms of the global axis
x[1] = 0;   y[1] = 15   # x, y coord. values of node 2 in terms of the global axis
x[2] = 0;   y[2] = 30   # x, y coord. values of node 3 in terms of the global axis
x[3] = 0;   y[3] = 45   # x, y coord. values of node 4 in terms of the global axis
x[4] = 0;   y[4] = 60   # x, y coord. values of node 5 in terms of the global axis
x[5] = 10;  y[5] = 60   # x, y coord. values of node 6 in terms of the global axis
x[6] = 20;  y[6] = 60   # x, y coord. values of node 7 in terms of the global axis

# Material and geometric properties
el = 30e6          # elastic modulus
area = 2           # cross-sectional area
xi = 2/3           # moment of inertia of cross-section
rho = 1            # mass density per volume (dummy value for static analysis)

# Boundary conditions
bcdof = np.array([0, 1, 2], dtype=int)  # DOFs at node 1 are constrained (0-indexed)
bcval = np.array([0, 0, 0])             # constrained values are 0

# Initialize system matrices
ff = np.zeros(sdof)          # system force vector
kk = np.zeros((sdof, sdof))  # system stiffness matrix

# Applied load
ff[19] = -60  # load applied at node 7 in the negative y direction (0-indexed: DOF 20 -> index 19)

# Assembly loop
for iel in range(nel):  # loop for the total number of elements
    
    # Extract system dofs associated with element
    index = feeldof1(iel, nnel, ndof)
    
    # Node numbers for element 'iel'
    node1 = iel      # starting node number for element 'iel'
    node2 = iel + 1  # ending node number for element 'iel'
    
    # Coordinates of nodes
    x1 = x[node1]
    y1 = y[node1]
    x2 = x[node2]
    y2 = y[node2]
    
    # Length of element 'iel'
    leng = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
    
    # Compute the angle between the local and global axes
    if (x2 - x1) == 0:
        if y2 > y1:
            beta = np.pi / 2
        else:
            beta = -np.pi / 2
    else:
        beta = np.arctan((y2 - y1) / (x2 - x1))
    
    # Compute element stiffness matrix
    k, m = feframe2(el, xi, leng, area, rho, beta, 1)
    
    # Assemble element matrix into system matrix
    kk = feasmbl1(kk, k, index)

# Apply boundary conditions
kk, ff = feaplyc2(kk, ff, bcdof, bcval)

# Solve the matrix equation
fsol = np.linalg.solve(kk, ff)

# Print results
num = np.arange(1, sdof + 1)
store = np.column_stack((num, fsol))

print("DOF    Displacement")
print("---    ------------")
for i in range(len(store)):
    print(f"{store[i, 0]:3.0f}    {store[i, 1]:12.6e}")
