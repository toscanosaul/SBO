import numpy as np
import scipy.weave
from scipy.spatial.distance import cdist


def dist2(ls, x1, x2=None):
    # Assumes NxD and MxD matrices.
    # Compute the squared distance matrix, given length scales.

    if x2 is None:
        # Find distance with self for x1.

        # Rescale.
        xx1 = x1 / ls
        xx2 = xx1

    else:
        # Rescale.
        xx1 = x1 / ls
        xx2 = x2 / ls

    r2 = cdist(xx1, xx2, 'sqeuclidean')

    return r2


def grad_dist2(ls, x1, x2=None):
    if x2 is None:
        x2 = x1

    # Rescale.
    x1 = x1 / ls
    x2 = x2 / ls

    N = x1.shape[0]
    M = x2.shape[0]
    D = x1.shape[1]
    gX = np.zeros((x1.shape[0], x2.shape[0], x1.shape[1]))

    code = \
        """
        for (int i=0; i<N; i++)
          for (int j=0; j<M; j++)
            for (int d=0; d<D; d++)
              gX(i,j,d) = (2/ls(d))*(x1(i,d) - x2(j,d));
        """
    try:
        scipy.weave.inline(code, ['x1', 'x2', 'gX', 'ls', 'M', 'N', 'D'], \
                           type_converters=scipy.weave.converters.blitz, \
                           compiler='gcc')
    except:
        # The C code weave above is 10x faster than this:
        for i in xrange(0, x1.shape[0]):
            gX[i, :, :] = 2 * (x1[i, :] - x2[:, :]) * (1 / ls)

    return gX


def dist_Mahalanobis(U, x1, x2=None):
    W = np.dot(U, U.T)