#!/usr/bin/env python

import numpy as np

###make a grid in [c,d], c and d are vectors, with m points in each dimension
###X is array of vector to form the grid

def cartesian(X,final=None):
    X=[np.asarray(i) for i in X]
    dtype=X[0].dtype
    n=np.prod([i.size for i in X])
    if final is None:
        final=np.zeros([n,len(X)],dtype=dtype)
    
    m=n/X[0].size
    final[:,0]=np.repeat(X[0],m)
    if X[1:]:
        cartesian(X[1:],final=final[0:m,1:])
        for j in xrange(1,X[0].size):
            final[j*m:(j+1)*m,1:]=final[0:m,1:]
    return final

def grid (c,d,m):
    k=len(c)
    X=[]
    for i in xrange(k):
        X.append(np.linspace(c[i],d[i],m))
    return cartesian(X)
