import numpy as np
from scipy import linalg
from scipy.stats import norm
from AffineBreakPoints import *
import scipy.linalg as spla


class SBO(object):

    def __init__(self, n1, n2, kernel, candidate_points, mu, possible_values_w=None):
        self.n1=n1
        self.n2=n2
     #   self.B_function = kernel.B_function
        self.possible_values_w = possible_values_w
        self.kernel = kernel
        self._points = candidate_points
        self.mean = mu.value

    def setup(self, XW, y, noise=None):
        if noise is None:
            noise = np.zeros(XW.shape[0])

        cov = self.kernel.cov(XW) + np.diag(noise)
        chol = spla.cholesky(cov, lower=True)
        self.L = chol

        tempN = XW.shape[0]

        Bhist = np.zeros((self._points.shape[0], tempN))

        for j in xrange(0, tempN):
            temp = self.kernel.B_function(
                self._points,
                XW[j:j+1, :],
                possible_values_w=self.possible_values_w
            )

            Bhist[:, j] = temp[:, 0]

        m2 = self._points.shape[0]
        scratch = np.zeros((m2, tempN))

        for j in xrange(m2):
            scratch[j, :] = linalg.solve_triangular(chol, Bhist[j, :].transpose(), lower=True)

        self.scratch = scratch

        muStartt = self.mean

        temp2t = linalg.solve_triangular(chol, (Bhist).T, lower=True)
        temp1t = linalg.solve_triangular(chol, y - muStartt, lower=True)
        a = muStartt + np.dot(temp2t.T, temp1t)

        self.a = a
        self.temp2 = temp2t




    def aANDb(self, x, xNew, wNew, L, temp2, past, kernel, n1, n2):
        """
        Output: A tuple with:
            -b:Vector of posterior variances of G(x)=E[f(x,w,z)] if
               we choose (xNew,wNew) at this iteration. The variances
               are evaluated at all the points of x.
            -gamma: Vector of Sigma_{0}(x_{i},w_{i},xNew,wNew) where
                    (x_{i},w_{i}) are the past observations.
            -BN: Vector B(x_{p},n+1), where x_{p} is a point
                 in the discretization of the domain of x.
            -temp1: Solution to the system Ly=gamma, where L
                    is the Cholesky decomposition of A.
            -aux4: Square of the norm of temp1.

        Args:
            -n: Iteration of the algorithm
            -x: nxdim(x) matrix where b is evaluated.
            -(xNew,wNew): The VOI will be evaluated at this point.
            -L: Cholesky decomposition of the matrix A, where A is the covariance
                matrix of the past obsevations (x,w).
            -temp2:temp2=inv(L)*B.T, where B is a matrix such that B(i,j) is
                   \int\Sigma_{0}(x_{i},w,x_{j},w_{j})dp(w)
                   where points x_{p} is a point of the discretization of
                   the space of x; and (x_{j},w_{j}) is a past observation.
            -past: Past observations.
            -kernel: kernel.
            -B: Computes B(x,XW)=\int\Sigma_{0}(x,w,XW[0:n1],XW[n1:n1+n2])dp(w).
                Its arguments are:
                    -x: Vector of points where B is evaluated
                    -XW: Point (x,w)
                    -n1: Dimension of x
                    -n2: Dimension of w
        """
        x = np.array(x)
        m = x.shape[0]
    #    tempN = self._numberTraining + n
        BN = np.zeros([m, 1])
       # n2 = self.n2
        # print "BN"

        point_temp = np.concatenate((xNew, wNew)).reshape((1, len(xNew)+len(wNew)))
        BN[:, 0] = self.kernel.B_function(x, point_temp, possible_values_w=self.possible_values_w)[:,0]  # B(x,n+1)

        # print BN[:,0]
        # if np.all(BN[:,0]==0):
        # print "aaaaaa"
        # print np.concatenate((xNew,wNew))
        # print x
        # B(x,np.concatenate((xNew,wNew)),self.n1,n2,kernel)
        # asdf

#        n1 = self.n1
 #       n2 = self.n2
        new = np.concatenate((xNew, wNew)).reshape((1, n1 + n2))

        gamma = np.transpose(kernel.cross_cov(new, past))

        temp1 = linalg.solve_triangular(L, gamma, lower=True)

        b = (BN - np.dot(temp2.T, temp1))

        aux4 = np.dot(temp1.T, temp1)

        b2 = kernel.cov(new) - aux4
        b2 = np.clip(b2, 0, np.inf)

        try:
            b = b / (np.sqrt(b2))

        except Exception as e:
            print "use a different point x"
            b = np.zeros((len(b), 1))

        return b, gamma, BN, temp1, aux4


    def evalVOI(self, pointNew, a, b, c, keep, keep1, M , BN, L, inv,
                aux4, kern, XW, scratch=None, grad=False, onlyGradient=False):
        """
        Output:
            Evaluates the VOI and it can compute its derivative. It evaluates
            the VOI, when grad and onlyGradient are False; it evaluates the
            VOI and computes its derivative when grad is True and onlyGradient
            is False, and computes only its gradient when gradient and
            onlyGradient are both True.

        Args:
            -n: Iteration of the algorithm.
            -pointNew: The VOI will be evaluated at this point.
            -a: Vector of the means of the GP on g(x)=E(f(x,w,z)).
                The means are evaluated on the discretization of
                the space of x.
            -b: Vector of posterior variances of G(x)=E[f(x,w,z)] if
                we choose (xNew,wNew) at this iteration. The variances
                are evaluated at all the points of x.
            -c: Vector returned by AffineBreakPoints.
            -keep: Indexes returned by AffineBreakPointsPrep. They represent
                   the new order of the elements of a and b.
            -keep1: Indexes returned by AffineBreakPoints. Those are the
                    indexes of the elements keeped.
            -M: Number of points keeped.
            -gamma: Vector of Sigma_{0}(x_{i},w_{i},xNew,wNew) where
                    (x_{i},w_{i}) are the past observations.
            -BN: Vector B(x_{p},n+1), where x_{p} is a point
                 in the discretization of the domain of x.
            -L: Cholesky decomposition of the matrix A, where A is the covariance
                matrix of the past obsevations (x,w).
            -inv: Solution to the system Ly=gamma, where L
                  is the Cholesky decomposition of A.
            -aux4: Square of the norm of inv.
            -kern: Kernel.
            -XW: Past observations.
            -scratch: Matrix where scratch[i,:] is the solution of the linear system
                      Ly=B[j,:].transpose() (See above for the definition of B and L)
            -grad: True if we want to compute the gradient; False otherwise.
            -onlyGradient: True if we only want to compute the gradient; False otherwise.
        """

        n1 = self.n1
        n2 = self.n2
        corr = True

        if grad == False:
            h = hvoi(b, c, keep1)  ##Vn
            return h
        bPrev = b
        a = a[keep1]
        b = b[keep1]
        keep = keep[keep1]  # indices conserved

        if M <= 1 and onlyGradient == False:
            h = hvoi(bPrev, c, keep1)
            return h, np.zeros(n1 + n2)

        if M <= 1 and onlyGradient == True:
            return np.zeros(n1 + n2)

        cPrev = c
        c = c[keep1 + 1]
        c2 = np.abs(c[0:M - 1])
        evalC = norm.pdf(c2)

      #  nTraining = self._numberTraining
      #  tempN = nTraining + n


        tempN = XW.shape[0]

        gradXSigma0, gradWSigma0 = self.kernel.grad_new_point(pointNew, XW)

        if corr is False:
            print "to do"
           # gradXB = self._gradXBfunc(pointNew, kern, BN, keep, self._points,
           #                           n1, n2)
           # gradWB = self._gradWBfunc(pointNew, kern, BN, keep, self._points)
#
 #           gradientGamma = np.concatenate((gradXSigma0, gradWSigma0),
  #                                         1).transpose()
        else:
            gradXB = self.kernel.grad_x_b_function(
                self._points[keep],
                pointNew,
                possible_values_w=self.possible_values_w
            )

            gradientGamma = gradXSigma0.transpose()

        inv3 = inv
        beta1 = (kern.cov(pointNew) - aux4)
        gradient = np.zeros(M)

        if corr is False:
            result = np.zeros(n1 + n2)
        else:
            result = np.zeros(n1)

        for i in xrange(n1):
            inv2 = linalg.solve_triangular(L, gradientGamma[i,0:tempN].transpose(), lower=True)
            aux5 = np.dot(inv2.T, inv3)
            for j in xrange(M):
                tmp = np.dot(inv2.T, scratch[j, :])
                tmp = (beta1 ** (-.5)) * (gradXB[j, i] - tmp)
                beta2 = BN[keep[j], :] - np.dot(scratch[j, :].T, inv3)
                tmp2 = (.5) * (beta1 ** (-1.5)) * beta2 * (2.0 * aux5)
                gradient[j] = tmp + tmp2
            result[i] = np.dot(np.diff(gradient), evalC)

        if corr:
            if onlyGradient:
                return result
            h = hvoi(bPrev, cPrev, keep1)
            return h, result

      #  for i in xrange(n2):
      #      inv2 = linalg.solve_triangular(L, gradientGamma[i + n1,0:tempN].transpose(), lower=True)
      #      aux5 = np.dot(inv2.T, inv3)
      #      for j in xrange(M):
      #          tmp = np.dot(inv2.T, scratch[j, :])
      #          tmp = (beta1 ** (-.5)) * (gradWB[j, i] - tmp)
      #          beta2 = BN[keep[j], :] - np.dot(scratch[j, :].T, inv3)
      #          tmp2 = (.5) * (beta1 ** (-1.5)) * (2.0 * aux5) * beta2
      #          gradient[j] = tmp + tmp2
      #      result[i + n1] = np.dot(np.diff(gradient), evalC)

      #  if onlyGradient:
      #      return result
      #  h = hvoi(bPrev, cPrev, keep1)
      #  return h, result

    def VOIfunc(self, pointNew, grad, XW, onlyGradient=False):
        """
        Output:
            Evaluates the VOI and it can compute its derivative. It evaluates
            the VOI, when grad and onlyGradient are False; it evaluates the
            VOI and computes its derivative when grad is True and onlyGradient
            is False, and computes only its gradient when gradient and
            onlyGradient are both True.

        Args:
            -n: Iteration of the algorithm.
            -pointNew: The VOI will be evaluated at this point.
            -grad: True if we want to compute the gradient; False otherwise.
            -L: Cholesky decomposition of the matrix A, where A is the covariance
                matrix of the past obsevations (x,w).
            -temp2: temp2=inv(L)*B.T, where B is a matrix such that B(i,j) is
                   \int\Sigma_{0}(x_{i},w,x_{j},w_{j})dp(w)
                   where points x_{p} is a point of the discretization of
                   the space of x; and (x_{j},w_{j}) is a past observation.
            -a: Vector of the means of the GP on g(x)=E(f(x,w,z)).
                The means are evaluated on the discretization of
                the space of x.
            -scratch: Matrix where scratch[i,:] is the solution of the linear system
                      Ly=B[j,:].transpose() (See above for the definition of B and L)
            -kern: Kernel.
            -XW: Past observations.
            -B: Computes B(x,XW)=\int\Sigma_{0}(x,w,XW[0:n1],XW[n1:n1+n2])dp(w).
                Its arguments are:
                    -x: Vector of points where B is evaluated
                    -XW: Point (x,w)
                    -n1: Dimension of x
                    -n2: Dimension of w
            -onlyGradient: True if we only want to compute the gradient; False otherwise.
        """
        L= self.L
        scratch = self.scratch
        a = self.a
        temp2 = self.temp2


        n1 = self.n1
        pointNew = pointNew.reshape([1, n1 + self.n2])

        b, gamma, BN, temp1, aux4 = self.aANDb(
            self._points,
            pointNew[0, 0:n1],
            pointNew[0, n1:n1 + self.n2],
            L,
            temp2=temp2,
            past=XW,
            kernel=self.kernel,
            n1=self.n1,
            n2=self.n2
        )
        #   print "primer"
        #  print a,b,BN
        b = b[:, 0]


        a, b, keep = AffineBreakPointsPrep(a, b)
        keep1, c = AffineBreakPoints(a, b)
        keep1 = keep1.astype(np.int64)
        M = len(keep1)
     #   nTraining = self._numberTraining
      #  tempN = nTraining + n
        tempN = XW.shape[0]
        keep2 = keep[keep1]
        if grad:
            scratch1 = np.zeros((M, tempN))
            for j in xrange(M):
                scratch1[j, :] = scratch[keep2[j], :]

        if onlyGradient:
            return self.evalVOI(
                pointNew,
                a,
                b,
                c,
                keep,
                keep1,
                M,
                BN,
                L,
                scratch=scratch1,
                inv=temp1,
                aux4=aux4,
                grad=True,
                onlyGradient=onlyGradient,
                kern=self.kernel,
                XW=XW
            )
        if grad == False:
            return self.evalVOI(
                pointNew,
                a,
                b,
                c,
                keep,
                keep1,
                M,
                BN,
                L,
                aux4=aux4,
                inv=temp1,
                kern=self.kernel,
                XW=XW
            )

        return self.evalVOI(
            pointNew,
            a,
            b,
            c,
            keep,
            keep1,
            M,
            BN,
            L,
            aux4=aux4,
            inv=temp1,
            scratch=scratch1,
            grad=True,
            kern=self.kernel,
            XW=XW
        )


##evaluate the function h of the paper
##b has been modified in affineBreakPointsPrep
def hvoi (b,c,keep):
    M=len(keep)
    if M>1:
        c=c[keep+1]
        c2=-np.abs(c[0:M-1])
        tmp=norm.pdf(c2)+c2*norm.cdf(c2)
        return np.sum(np.diff(b[keep])*tmp)
    else:
        return 0
