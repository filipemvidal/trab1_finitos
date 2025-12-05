import numpy as np


def feaplyc2(kk, ff, bcdof, bcval):
    """
    Purpose:
        Apply constraints to matrix equation [kk]{x}={ff}

    Synopsis:
        kk, ff = feaplyc2(kk, ff, bcdof, bcval)

    Variable Description:
        kk - system matrix before applying constraints 
        ff - system vector before applying constraints
        bcdof - a vector containing constrained d.o.f
        bcval - a vector containing constrained value 

        For example, there are constraints at d.o.f=2 and 10
        and their constrained values are 0.0 and 2.5, 
        respectively.  Then, bcdof[0]=2 and bcdof[1]=10; and
        bcval[0]=0.0 and bcval[1]=2.5.
    """
    
    n = len(bcdof)
    sdof = kk.shape[0]
    
    for i in range(n):
        c = bcdof[i]
        for j in range(sdof):
            kk[c, j] = 0
        
        kk[c, c] = 1
        ff[c] = bcval[i]
    
    return kk, ff
