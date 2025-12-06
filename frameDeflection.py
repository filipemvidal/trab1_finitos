import numpy as np
from femUtils import *

def calculate_frame_deflection(
        nel:int, nnel:int, ndof:int, x:np.ndarray, y:np.ndarray, 
        el:float, area:float, xi:float, 
        bcdof:list, bcval:list, applied_forces:dict):
    """
    Purpose:
        Calculate the deflection of a frame structure using the Finite Element Method.
        This function performs a finite element analysis on a frame structure to compute
        nodal displacements under applied loads and boundary conditions. It assembles the
        global stiffness matrix, applies boundary conditions, and solves the system of
        linear equations.
    
    Parameters:
        nel : int
            Number of elements in the frame structure.
        nnel : int
            Number of nodes per element.
        ndof : int
            Number of degrees of freedom per node.
        x : numpy.ndarray
            Array containing x-coordinates of all nodes.
        y : numpy.ndarray
            Array containing y-coordinates of all nodes.
        el : float
            Young's modulus (elastic modulus) of the material.
        area : float
            Cross-sectional area of the frame elements.
        xi : float
            Moment of inertia of the cross-section.
        bcdof : list or numpy.ndarray
            List of degrees of freedom where boundary conditions are applied.
        bcval : list or numpy.ndarray
            List of prescribed values for the boundary conditions.
        applied_forces : dict
            Dictionary mapping degree of freedom indices (int) to force values (float) 
            or moments (float).
            Example: {dof_index: value}

    Returns:
        store : dict
            Dictionary mapping degree of freedom indices (int) to computed displacements (float).
    """

    # Problem parameters
    nnode = (nnel - 1) * nel + 1  # total number of nodes in system
    sdof = nnode * ndof            # total system dofs

    # Initialize system matrices
    ff = np.zeros(sdof)          # system force vector
    kk = np.zeros((sdof, sdof))  # system stiffness matrix

    # Applied load
    for(dof, value) in applied_forces.items():
        ff[dof] = value

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
        k = feframe2(el, xi, leng, area, beta)

        # Assemble element matrix into system matrix
        kk = feasmbl1(kk, k, index)

    # Apply boundary conditions
    kk, ff = feaplyc2(kk, ff, bcdof, bcval)

    # Solve the matrix equation
    fsol = np.linalg.solve(kk, ff)

    tolerance = 1e-10
    fsol[np.abs(fsol) < tolerance] = 0.0

    # Print results
    num = np.arange(0, sdof)
    store = {dof: displacement for dof, displacement in zip(num, fsol)}

    return store

def equivalent_nodal_loads(q, L):
    """
    Purpose:
        Calculate equivalent nodal loads for a uniformly distributed load on a frame element.
    Parameters:
        q : float
            Magnitude of the uniformly distributed load (force per unit length).
        L : float
            Length of the frame element.
    Returns:
        f_local : numpy.ndarray
            Equivalent nodal load vector in the local coordinate system.
    """
    
    f_local = np.array([
        0.0,
        q * L / 2,
        q * L**2 / 12,
        0.0,
        q * L / 2,
        -q * L**2 / 12
    ])

    return f_local