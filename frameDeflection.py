import numpy as np
from femUtils import *


def calculate_frame_deflection(
    nel: int,
    nnel: int,
    ndof: int,
    x: np.ndarray,
    y: np.ndarray,
    el: float,
    area: float,
    xi: float,
    bcdof: list,
    bcval: list,
    applied_forces: dict,
):
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
    sdof = nnode * ndof  # total system dofs

    # Initialize system matrices
    ff = np.zeros(sdof)  # system force vector
    kk = np.zeros((sdof, sdof))  # system stiffness matrix

    # Applied load
    for dof, value in applied_forces.items():
        ff[dof] = value

    # Assembly loop
    for iel in range(0, nel):  # loop for the total number of elements

        # Extract system dofs associated with element
        index = eldof(iel, nnel, ndof)

        # Node numbers for element 'iel'
        node1 = iel  # starting node number for element 'iel'
        node2 = iel + 1  # ending node number for element 'iel'

        # Coordinates of nodes
        x1 = x[node1]
        y1 = y[node1]
        x2 = x[node2]
        y2 = y[node2]

        # Length of element 'iel'
        leng = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

        # Compute the angle between the local and global axes
        if (x2 - x1) == 0:
            if y2 > y1:
                beta = np.pi / 2
            else:
                beta = -np.pi / 2
        else:
            beta = np.arctan((y2 - y1) / (x2 - x1))

        # Compute element stiffness matrix
        k = stiffMat(el, xi, leng, area, beta)

        # Assemble element matrix into system matrix
        kk = assembly(kk, k, index)

    # Apply boundary conditions
    kk, ff = applyCond(kk, ff, bcdof, bcval)

    # Solve the matrix equation
    fsol = np.linalg.solve(kk, ff)

    # Store displacements in a dictionary
    num = np.arange(0, sdof)
    store = {dof: displacement for dof, displacement in zip(num, fsol)}

    return store


def equivalent_nodal_loads(q, L, beta=0.0):
    """
    Purpose:
        Calculate equivalent nodal loads for a uniformly distributed load on a frame element.
    Parameters:
        q : float
            Magnitude of the uniformly distributed load (force per unit length).
        L : float
            Length of the frame element.
        horizontal : bool, optional
            If True, the load is applied horizontally; if False, vertically. Default is False.
    Returns:
        f_local : numpy.ndarray
            Equivalent nodal load vector in the local coordinate system.
    """

    # Convert angle to radians
    beta = np.radians(beta)

    # Rotation matrix
    r = np.array(
        [
            [np.cos(beta), np.sin(beta), 0, 0, 0, 0],
            [-np.sin(beta), np.cos(beta), 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0],
            [0, 0, 0, np.cos(beta), np.sin(beta), 0],
            [0, 0, 0, -np.sin(beta), np.cos(beta), 0],
            [0, 0, 0, 0, 0, 1],
        ]
    )

    f_local = np.array([0.0, q * L / 2, -q * L / 12, 0.0, q * L / 2, q * L / 12])

    # Equivalent nodal load vector in the global coordinate system.
    f_global = r @ f_local

    return f_global

def interpolate_displacement(
    store: dict,
    x: np.ndarray,
    y: np.ndarray,
    nel: int,
    nnel: int,
    ndof: int,
    x_point: float,
    y_point: float,
):
    """
    Purpose:
        Calculate the displacement at a specific point in the frame structure.
        The point can be at a node or within an element. Uses shape functions
        for interpolation within elements.
    Parameters:
        store : dict
            Dictionary mapping degree of freedom indices to computed displacements.
        x : numpy.ndarray
            Array containing x-coordinates of all nodes.
        y : numpy.ndarray
            Array containing y-coordinates of all nodes.
        nel : int
            Number of elements in the frame structure.
        nnel : int
            Number of nodes per element.
        ndof : int
            Number of degrees of freedom per node.
        x_point : float
            X-coordinate of the point where displacement is required.
        y_point : float
            Y-coordinate of the point where displacement is required.
    Returns:
        displacement : dict
            Dictionary with keys 'u' (horizontal displacement), 'v' (vertical displacement),
            and 'theta' (rotation) at the specified point.
    """
    tolerance = 1e-6

    # Check if point is at a node
    for node_idx in range(len(x)):
        if abs(x[node_idx] - x_point) < tolerance and abs(y[node_idx] - y_point) < tolerance:
            u = store[node_idx * ndof]
            v = store[node_idx * ndof + 1]
            theta = store[node_idx * ndof + 2]
            return {'u': u, 'v': v, 'theta': theta}

    # Check if point is within an element
    for iel in range(nel):
        node1 = iel
        node2 = iel + 1

        x1, y1 = x[node1], y[node1]
        x2, y2 = x[node2], y[node2]

        # Length of element
        L = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)

        # Distance from node1 to the point
        d = np.sqrt((x_point - x1)**2 + (y_point - y1)**2)

        # Check if point is on the element (collinear and within bounds)
        if L > tolerance:
            # Check collinearity using cross product
            cross = abs((x_point - x1) * (y2 - y1) - (y_point - y1) * (x2 - x1))
            if cross < tolerance and d < L + tolerance:
                # Normalized position along element (0 to 1)
                xi = d / L

                # Get nodal displacements
                u1 = store[node1 * ndof]
                v1 = store[node1 * ndof + 1]
                theta1 = store[node1 * ndof + 2]
                u2 = store[node2 * ndof]
                v2 = store[node2 * ndof + 1]
                theta2 = store[node2 * ndof + 2]

                # Linear shape functions for axial displacement
                N1 = 1 - xi
                N2 = xi

                # Hermite shape functions for transverse displacement
                H1 = 1 - 3*xi**2 + 2*xi**3
                H2 = xi - 2*xi**2 + xi**3
                H3 = 3*xi**2 - 2*xi**3
                H4 = -xi**2 + xi**3

                # Interpolate displacements in global coordinate system
                u_global = N1 * u1 + N2 * u2
                v_global = H1 * v1 + H2 * L * theta1 + H3 * v2 + H4 * L * theta2
                theta_global = N1 * theta1 + N2 * theta2

                return {'u': u_global, 'v': v_global, 'theta': theta_global}

    raise ValueError(f"Point ({x_point}, {y_point}) is not on the structure.")