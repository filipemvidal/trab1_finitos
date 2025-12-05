import numpy as np


def feasmbl1(kk, k, index):
    """
    Purpose:
        Assembly of element matrices into the system matrix

    Synopsis:
        kk = feasmbl1(kk, k, index)

    Variable Description:
        kk - system matrix
        k  - element matrix
        index - d.o.f. vector associated with an element
    """
    
    edof = len(index)
    for i in range(edof):
        ii = index[i]
        for j in range(edof):
            jj = index[j]
            kk[ii, jj] = kk[ii, jj] + k[i, j]
    
    return kk
