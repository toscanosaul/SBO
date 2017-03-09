import numpy as np
from scipy import linalg
import scipy.linalg as spla

class SBO_stats(object):

    def __init__(self, kernel, n1, n2, mean, possible_values_w=None):
        self.kernel = kernel
        self.n1 = n1
        self.n2 = n2
        self.mean = mean
        self.possible_values_w=possible_values_w


    def setup(self, XW, noise=None):
        if noise is None:
            noise = np.zeros(XW.shape[0])

        cov = self.kernel.cov(XW) + np.diag(noise)
        chol = spla.cholesky(cov, lower=True)
        self.L = chol


    def aN_grad(self, x, Xhist, yHist, gradient=True, onlyGradient=False,
                logproductExpectations=None):
        """
        Computes a_{n} and it can compute its derivative. It evaluates a_{n},
        when grad and onlyGradient are False; it evaluates the a_{n} and computes its
        derivative when grad is True and onlyGradient is False, and computes only its
        gradient when gradient and onlyGradient are both True.

        Args:
            x: a_{n} is evaluated at x.
            L: Cholesky decomposition of the matrix A, where A is the covariance
               matrix of the past obsevations (x,w).
            n: Step of the algorithm.
            dataObj: Data object (it contains all the history).
            gradient: True if we want to compute the gradient; False otherwise.
            onlyGradient: True if we only want to compute the gradient; False otherwise.
            logproductExpectations: Vector with the logarithm of the product of the
                                    expectations of np.exp(-alpha2[j]*((z-W[i,j])**2))
                                    where W[i,:] is a point in the history.
                                    --Only with the SEK--
        """
        muStart = self.mean.value

        y2 = yHist - muStart
        tmp_n = len(y2)

        B = np.zeros(tmp_n)
        L=self.L


        for i in xrange(tmp_n):
            B[i] = self.kernel.B_function(
                        x,
                        Xhist[i:i+1, :],
                        possible_values_w=self.possible_values_w
                    )

        inv1 = linalg.solve_triangular(L, y2, lower=True)

        if onlyGradient:
            gradXB = self.gradXBforAn(x, Xhist)
            temp4 = linalg.solve_triangular(L, gradXB.transpose(), lower=True)
            gradAn = np.dot(inv1.transpose(), temp4)
            return gradAn

        inv2 = linalg.solve_triangular(L, B.transpose(), lower=True)
        aN = muStart + np.dot(inv2.transpose(), inv1)
        if gradient == True:
            gradXB = self.gradXBforAn(x, Xhist)
            temp4 = linalg.solve_triangular(L, gradXB.transpose(), lower=True)
            gradAn = np.dot(inv1.transpose(), temp4)
            return aN, gradAn
        else:
            return aN


    def gradXBforAn(self, x, X):
        """Computes the gradient of B(x,i) for i in {1,...,n+nTraining}
           where nTraining is the number of training points

           Args:
              x: Argument of B
              n: Current iteration of the algorithm
              B: Vector {B(x,i)} for i in {1,...,n}
              kern: kernel
              X: Past observations X[i,:] for i in {1,..,n+nTraining}
              n1: Dimension of x
              nT: Number of training points
        """
        nT = X.shape[0]
        gradXB = np.zeros((self.n1, nT))

        for i in range(nT):
            gradXB[:,i] = self.kernel.grad_x_b_function_an(
                    x,
                    X[i:i+1, :],
                    possible_values_w=self.possible_values_w
            )

        return gradXB