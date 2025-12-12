import numpy as np

def feeldof1(iel, nnel, ndof):
    """
    Purpose:
        Compute system dofs associated with each element in one-
        dimensional problem

    Synopsis:
        index = feeldof1(iel, nnel, ndof)

    Variable Description:
        index - system dof vector associated with element "iel"
        iel - element number whose system dofs are to be determined (0-based)
        nnel - number of nodes per element
        ndof - number of dofs per node
    """
    
    edof = nnel * ndof
    start = iel * (nnel - 1) * ndof
    
    index = []
    for i in range(edof):
        index.append(start + i)  # 0-based indexing
    
    return index

def feframe2(el, xi, leng, area, beta):
    """
    Purpose:
        Stiffness and mass matrices for the 2-d frame element
        nodal dof {u_1 v_1 theta_1 u_2 v_2 theta_2}

    Synopsis:
        k, m = feframe2(el, xi, leng, area, rho, beta, ipt) 

    Variable Description:
        k - element stiffness matrix (size of 6x6)   
        m - element mass matrix (size of 6x6)
        el - elastic modulus 
        xi - second moment of inertia of cross-section
        leng - element length
        area - area of beam cross-section
        rho - mass density (mass per unit volume)
        beta - angle between the local and global axes
               is positive if the local axis is in the ccw direction from
               the global axis
        ipt = 1 - consistent mass matrix
            = 2 - lumped mass matrix
            = 3 - diagonal mass matrix
    """
    
    # Stiffness matrix at the local axis
    a = el * area / leng
    c = el * xi / (leng ** 3)
    
    kl = np.array([
        [a,      0,           0,            -a,     0,          0],
        [0,      12*c,        6*leng*c,      0,    -12*c,       6*leng*c],
        [0,      6*leng*c,    4*leng**2*c,   0,    -6*leng*c,   2*leng**2*c],
        [-a,     0,           0,             a,     0,          0],
        [0,     -12*c,       -6*leng*c,      0,     12*c,      -6*leng*c],
        [0,      6*leng*c,    2*leng**2*c,   0,    -6*leng*c,   4*leng**2*c]
    ])
    
    # Rotation matrix
    r = np.array([
        [np.cos(beta),  np.sin(beta),  0,   0,              0,             0],
        [-np.sin(beta), np.cos(beta),  0,   0,              0,             0],
        [0,             0,             1,   0,              0,             0],
        [0,             0,             0,   np.cos(beta),   np.sin(beta),  0],
        [0,             0,             0,  -np.sin(beta),   np.cos(beta),  0],
        [0,             0,             0,   0,              0,             1]
    ])
    
    # Stiffness matrix at the global axis
    k = r.T @ kl @ r
    
    return k

def feasmbl1(kk, k, index):
    """
    Purpose:
        Assembly of element matrices into the system matrix

    Synopsis:
        kk = feasmbl1(kk, k, index)

    Variable Description:
        kk - system matrix
        k  - element matrix
        index - d.o.f. vector associated with an element (0-based)
    """
    
    edof = len(index)
    for i in range(edof):
        ii = index[i]  # Already 0-based
        for j in range(edof):
            jj = index[j]  # Already 0-based
            kk[ii, jj] = kk[ii, jj] + k[i, j]
    
    return kk

def feaplyc2(kk, ff, bcdof, bcval):
    """
    Purpose:
        Apply constraints to matrix equation [kk]{x}={ff}

    Synopsis:
        kk, ff = feaplyc2(kk, ff, bcdof, bcval)

    Variable Description:
        kk - system matrix before applying constraints 
        ff - system vector before applying constraints
        bcdof - a vector containing constrained d.o.f (0-based)
        bcval - a vector containing constrained value 

        For example, there are constraints at d.o.f=2 and 10
        and their constrained values are 0.0 and 2.5, 
        respectively. Then, bcdof[0]=2 and bcdof[1]=10; and
        bcval[0]=1.0 and bcval[1]=2.5.
    """
    
    n = len(bcdof)
    sdof = kk.shape[0]
    
    for i in range(n):
        c = bcdof[i]  # Already 0-based
        for j in range(sdof):
            kk[c, j] = 0
        
        kk[c, c] = 1
        ff[c] = bcval[i]
    
    return kk, ff