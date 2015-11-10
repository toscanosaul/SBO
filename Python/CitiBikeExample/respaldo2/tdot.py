# dot(A, A.T) is a common operation, for example as part of computing
# covariances or square distances. The answer is symmetric, so Matlab/Octave
# special cases the code for (A*A') to roughly halve computation time:
#     http://www.walkingrandomly.com/?p=4912
# Numpy's dot doesn't currently (2013-04-25) optimize this case, and routines
# like scipy.cov are ~2x slower than they should be for large matrices.
#
# Here is a demonstration of a fast tdot(A) = dot(A,A.T). It would be great if a
# polished version of this could make it into numpy. It would need to be more
# carefully checked, and maybe deal with singles and maybe complex numbers.
# Ideally incorporating a check in dot() itself would notice when the second
# argument is the transpose of the first.
#
# As an aside, I wonder why there isn't a nice wrapper of DSYRK with the
# existing BLAS wrappers in numpy? Why aren't wrappers for *all* the standard
# BLAS routines included in numpy?
#
# Iain Murray, April 2013. iain contactable via iainmurray.net
# http://homepages.inf.ed.ac.uk/imurray2/code/tdot/tdot.py

# 2013-09-23 James Hensman tells me that the BLAS functions symm, syrk, syr2k,
# hemm, herk and her2k are now wrapped in scipy.linalg, as of scipy version 0.13
# Using those wrappers could make this code somewhat simpler, at the expense of
# depending on scipy.


import numpy as np
import ctypes
from ctypes import byref, c_char, c_int, c_double
try:
    _blaslib = ctypes.cdll.LoadLibrary(np.core._dotblas.__file__)
except:
    raise Exception("Numpy isn't configured to use a fast BLAS library.")

def tdot(mat, out=None):
    """returns np.dot(mat, mat.T), but faster for large 2D arrays of doubles."""
    if (mat.dtype != 'float64') or (len(mat.shape) != 2):
        return np.dot(mat, mat.T)
    nn = mat.shape[0]
    if not out:
        out = np.zeros((nn,nn))
    else:
        assert(out.dtype == 'float64')
        assert(out.shape == (nn,nn))
        # FIXME: should allow non-contiguous out, and copy output into it:
        assert(8 in out.strides)
        # zeroing needed because of dumb way I copy across triangular answer
        out[:] = 0.0

    ## Call to DSYRK from BLAS
    # If already in Fortran order (rare), and has the right sorts of strides I
    # could avoid the copy. I also thought swapping to cblas API would allow use
    # of C order. However, I tried that and had errors with large matrices:
    # http://homepages.inf.ed.ac.uk/imurray2/code/tdot/tdot_broken.py
    mat = mat.copy(order='F')
    TRANS = c_char('n')
    N = c_int(mat.shape[0])
    K = c_int(mat.shape[1])
    LDA = c_int(mat.shape[0])
    UPLO = c_char('l')
    ALPHA = c_double(1.0)
    A = mat.ctypes.data_as(ctypes.c_void_p)
    BETA = c_double(0.0)
    C = out.ctypes.data_as(ctypes.c_void_p)
    LDC = c_int(np.max(out.strides) / 8)
    _blaslib.dsyrk_(byref(UPLO), byref(TRANS), byref(N), byref(K),
            byref(ALPHA), A, byref(LDA), byref(BETA), C, byref(LDC))

    # Copy triangular answer across (there will be a better way):
    # Wouldn't need to zero over provided out array if did this better.
    out += out.T
    out[[range(nn),range(nn)]] /= 2.0  # correct the diagonal

    return out

#def cov(m, y=None, rowvar=1, bias=0, ddof=None):
#    """drop-in replacement for scipy.cov AKA numpy.lib.function_base.cov"""
#    TODO -- could make a fast cov, using tdot


if __name__ == '__main__':
    from numpy.random import rand
    import timeit
    
    a = rand(4000, 4000)
    repeats = 1
    print(timeit.Timer("numpy.dot(a, a.T)",
            "import numpy; from __main__ import a").timeit(repeats))
    print(timeit.Timer("tdot(a)",
            "from __main__ import a, tdot").timeit(repeats))
    ans1 = np.dot(a,a.T)
    ans2 = tdot(a)
    err = np.max(np.abs(ans1 - ans2) / ans1)
    print('max error = %g' % err)
