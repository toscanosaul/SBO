import numpy as np

from abc import ABCMeta, abstractmethod
from params import Param as Hyperparameter
import priors
import kernel_utils
from scipy.spatial.distance import cdist

SQRT_3 = np.sqrt(3.0)
SQRT_5 = np.sqrt(5.0)


class AbstractKernel(object):
    __metaclass__ = ABCMeta

    @property
    def hypers(self):
        return None

    @abstractmethod
    def cov(self, inputs):
        pass

    @abstractmethod
    def diag_cov(self, inputs):
        pass

    @abstractmethod
    def cross_cov(self, inputs_1, inputs_2):
        pass

    @abstractmethod
    def cross_cov_grad_data(self, inputs_1, inputs_2):
        pass



class Matern52(AbstractKernel):
    def __init__(self, num_dims, length_scale=None, name='Matern52'):
        self.name = name
        self.num_dims = num_dims
        #TO DO: ADD DIVISION_INPUT IN THE CONSTRUCTOR
        self.n1 = 4
        self.division_input = [0, 4]

        default_ls = Hyperparameter(
            initial_value = np.ones(self.num_dims),
            prior = priors.Tophat(0.01, 8e11),
            name = 'ls'
        )

        self.ls = length_scale if length_scale is not None else default_ls

        assert self.ls.value.shape[0] == self.num_dims

    @property
    def hypers(self):
        return self.ls

    def cov(self, inputs):
        return self.cross_cov(inputs, inputs)

    def diag_cov(self, inputs):
        inputs = inputs[:,self.division_input[0]:self.division_input[1]]
        return np.ones(inputs.shape[0])

    def cross_cov(self, inputs_1, inputs_2):
        inputs_1 = inputs_1[:, self.division_input[0]:self.division_input[1]]
        inputs_2 = inputs_2[:, self.division_input[0]:self.division_input[1]]
        r2  = np.abs(kernel_utils.dist2(self.ls.value, inputs_1, inputs_2))
        r   = np.sqrt(r2)
        cov = (1.0 + SQRT_5*r + (5.0/3.0)*r2) * np.exp(-SQRT_5*r)

        return cov

    def compute_distance(self, inputs_1, inputs_2):

        r2 = np.abs(kernel_utils.dist2(self.ls.value, inputs_1, inputs_2))
        return r2

    def gradient(self, inputs):
        inputs = inputs[:, self.division_input[0]:self.division_input[1]]
        n1 = self.num_dims-1
        X = inputs
        derivate_respect_to_r = self.gradient_respect_distance(inputs)
        r = np.sqrt(self.compute_distance(X, X))
        grad = []
        N = X.shape[0]

        for i in range(n1):
            x_i = X[:, i:i + 1]
            x_2_i = X[:, i:i + 1]
            x_dist = cdist(x_i, x_2_i, 'sqeuclidean')
            product_1 = derivate_respect_to_r * (1.0/r) * x_dist
            product_2 = - 1.0/(self.ls.value[i] ** 3)
            derivative_K_respect_to_alpha_i = product_1 * product_2

            for j in range(N):
                derivative_K_respect_to_alpha_i[j, j] = 0

            grad.append(derivative_K_respect_to_alpha_i)
        return grad

    def gradient_respect_distance(self, inputs):
        r2 = self.compute_distance(inputs, inputs)
        r = np.sqrt(r2)

        part_1 = ((1.0 + SQRT_5*r + (5.0/3.0)*r2) * np.exp (-SQRT_5* r) * (-SQRT_5))
        part_2 = (np.exp (-SQRT_5* r) * (SQRT_5 + (10.0/3.0) * r))
        derivate_respect_to_r = part_1 + part_2
        return derivate_respect_to_r

    def gradient_respect_distance_cross(self, inputs_1, inputs_2):
        r2 = self.compute_distance(inputs_1, inputs_2)
        r = np.sqrt(r2)

        part_1 = ((1.0 + SQRT_5*r + (5.0/3.0)*r2) * np.exp (-SQRT_5* r) * (-SQRT_5))
        part_2 = (np.exp (-SQRT_5* r) * (SQRT_5 + (10.0/3.0) * r))
        derivate_respect_to_r = part_1 + part_2
        return derivate_respect_to_r


    def grad_new_point(self, new, inputs):
        """Computes the vector of the gradients of Sigma_{0}(new,XW[i,:]) for
                all the past observations XW[i,]. Sigma_{0} is the covariance of
                the GP on F.
        """
        ##TO DO: GENERALIZE FOR THE CASE WHEN W isn't discrete and does depend on x?
        inputs = inputs[:,self.division_input[0]:self.division_input[1]]
        new = new[0:1,self.division_input[0]:self.division_input[1]]

        N_points = inputs.shape[0]
        gradXSigma0 = np.zeros([N_points + 1, self.n1])
        gradWSigma0 = np.zeros([N_points + 1, 1])

        derivate_respect_to_r = self.gradient_respect_distance_cross(new, inputs)
        r2 = self.compute_distance(new, inputs)
        r = np.sqrt(r2)

        for i in range(self.n1):
            factor = (self.ls.value[i] ** 2) * (new[0:1,i] - inputs[:,i]) / (r)
            gradXSigma0[0:N_points,i] = derivate_respect_to_r * factor

        return gradXSigma0, gradWSigma0

    def cross_cov_grad_data(self, inputs_1, inputs_2):
        # NOTE: This is the gradient wrt the inputs of inputs_2
        # The gradient wrt the inputs of inputs_1 is -1 times this
        r2      = np.abs(kernel_utils.dist2(self.ls.value, inputs_1, inputs_2))
        r       = np.sqrt(r2)
        grad_r2 = (5.0/6.0)*np.exp(-SQRT_5*r)*(1 + SQRT_5*r)

        return grad_r2[:,:,np.newaxis] * kernel_utils.grad_dist2(self.ls.value, inputs_1, inputs_2)


class multi_task(AbstractKernel):
    def __init__(self, num_dims, nFolds, length_scale=None, name='multitask'):
        self.name = name
        self.num_dims = num_dims
        self.nFolds = nFolds
        #TO DO: ADD DIVISION_INPUT IN THE CONSTRUCTOR
        self.division_input = 4
        self.n1 = 4

        default_ls = Hyperparameter(
            initial_value=np.ones(self.num_dims),
            prior=priors.Tophat(-10, 10),
            name='ls'
        )

        self.ls = length_scale if length_scale is not None else default_ls

    @property
    def hypers(self):
        return self.ls

    def cov(self, inputs):
        return self.cross_cov(inputs, inputs)

    def diag_cov(self, inputs):
        inputs = inputs[:, self.division_input]
        return np.ones(inputs.shape[0])

    def cross_cov(self, inputs_1, inputs_2):
       # print inputs_2
       # print self.division_input
        inputs_1 = inputs_1[:, self.division_input]
        inputs_2 = inputs_2[:, self.division_input]
        nP = len(inputs_1)
        C = np.zeros((nP, nP))
        s, t = np.meshgrid(inputs_1, inputs_2)
        s = s.astype(int)
        t = t.astype(int)
        count = 0
        L = np.zeros((self.nFolds, self.nFolds))
        for i in range(self.nFolds):
            for j in range(i + 1):
                L[i, j] = np.exp(self.ls.value[count + j])
            count += i + 1

        covM = np.dot(L, np.transpose(L))
        T = covM[s, t].transpose()
        return T

    def gradient(self, inputs):
        folds = inputs[:,  self.division_input]

        derivative_cov_folds = {}
        count = 0
        nFolds = self.nFolds
        L = np.zeros((self.nFolds, self.nFolds))
        for i in range(self.nFolds):
            for j in range(i + 1):
                L[i, j] = np.exp(self.ls.value[count + j])
            count += i + 1

        L_cov_folds = L

        N = len(folds)
        grad = []
        count = 0
        for i in range(nFolds):
            for j in range(i + 1):
                tmp_der = np.zeros((nFolds, nFolds))
                tmp_der[i, j] = L_cov_folds[i, j]
                tmp_der_mat = (np.dot(tmp_der, L_cov_folds.transpose()))
                tmp_der_mat += tmp_der_mat.transpose()
                derivative_cov_folds[count + j] = tmp_der_mat
            count += i + 1

        for k in range(np.sum(range(nFolds + 1))):
            der_covariance_folds = np.zeros((N, N))
            for i in range(N):
                for j in range(i + 1):
                    der_covariance_folds[i, j] = derivative_cov_folds[k][folds[i], folds[j]]
                    der_covariance_folds[j, i] = der_covariance_folds[i, j]
            der_K_respect_to_l = der_covariance_folds
            grad.append(der_K_respect_to_l)
        return grad

    def grad_new_point(self, new, inputs):
        """Computes the vector of the gradients of Sigma_{0}(new,XW[i,:]) for
                all the past observations XW[i,]. Sigma_{0} is the covariance of
                the GP on F.
        """
        ##TO DO: GENERALIZE FOR THE CASE WHEN W isn't discrete and does depend on x?

        N_points = inputs.shape[0]
        gradXSigma0 = np.zeros([N_points + 1, self.n1])
        gradWSigma0 = np.zeros([N_points + 1, 1])

        return gradXSigma0, gradWSigma0




    def cross_cov_grad_data(self, inputs_1, inputs_2):
        # NOTE: This is the gradient wrt the inputs of inputs_2
        # The gradient wrt the inputs of inputs_1 is -1 times this
        r2 = np.abs(kernel_utils.dist2(self.ls.value, inputs_1, inputs_2))
        r = np.sqrt(r2)
        grad_r2 = (5.0 / 6.0) * np.exp(-SQRT_5 * r) * (1 + SQRT_5 * r)

        return grad_r2[:, :, np.newaxis] * kernel_utils.grad_dist2(
            self.ls.value, inputs_1, inputs_2)



class ProductKernel(AbstractKernel):
    # TODO: If all kernel values are positive then we can do things in log-space

    def __init__(self, n1, *kernels):
        self.kernels = kernels

        params = [kernel.ls.value for kernel in kernels]
        default_ls = Hyperparameter(
            initial_value=np.concatenate(params),
            prior=priors.ProductOfPriors(15, [kernel.ls.prior for kernel in kernels]),
            name='ls'
        )

        self.ls = default_ls
        self.n1 = n1
        self.dim_x = 4

    @property
    def hypers(self):
        return self.ls

    def cov(self, inputs):
        values = self.ls.value
        self.kernels[0].hypers.set_value(values[0:self.n1])
        self.kernels[1].hypers.set_value(values[self.n1:])
        return reduce(lambda K1, K2: K1 * K2,
                      [kernel.cov(inputs) for kernel in self.kernels])

    def diag_cov(self, inputs):
        values = self.ls.value
        self.kernels[0].hypers.set_value(values[0:self.n1])
        self.kernels[1].hypers.set_value(values[self.n1:])
        return reduce(lambda K1, K2: K1 * K2,
                      [kernel.diag_cov(inputs) for kernel in self.kernels])

    def cross_cov(self, inputs_1, inputs_2):
        values = self.ls.value
        self.kernels[0].hypers.set_value(values[0:self.n1])
        self.kernels[1].hypers.set_value(values[self.n1:])
        return reduce(lambda K1, K2: K1 * K2,
                      [kernel.cross_cov(inputs_1, inputs_2) for kernel in
                       self.kernels])

    def gradient(self, inputs):
        grad_1 = self.kernels[0].gradient(inputs)
        kernel_1 = self.kernels[0].cov(inputs)

        kernel_2 = self.kernels[1].cov(inputs)
        grad_2 = self.kernels[1].gradient(inputs)

        part_1 = [kernel_2 * j for j in grad_1]
        part_2 = [kernel_1 * j for j in grad_2]

        grad = part_1 + part_2

        return grad


    # This is the gradient wrt **inputs_2**
    def cross_cov_grad_data(self, inputs_1, inputs_2):
        vals = np.array(
            [kernel.cross_cov(inputs_1, inputs_2) for kernel in self.kernels])
        vprod = reduce(lambda x, y: x * y, vals)
        grads = np.array(
            [kernel.cross_cov_grad_data(inputs_1, inputs_2) for kernel in
             self.kernels])
        V = vals == 0

        return (((vprod[:, :, np.newaxis] * grads) / (vals + V)[:, :, :,
                                                     np.newaxis]) + (
                V[:, :, :, np.newaxis] * grads)).sum(0)

    def B_function(self, x, XW, discrete_uniform=True, possible_values_w=None):
        """
        -B: Computes B(x,XW)=\int\Sigma_{0}(x,w,XW[0:n1],XW[n1:n1+n2])dp(w).
            Its arguments are:
                -x: Vector of points where B is evaluated
                -XW: Point (x,w)
                -n1: Dimension of x
                -n2: Dimension of w
        """

        cov = 0.0
        N = 1.0

        if discrete_uniform:
            N = len(possible_values_w)
            for value_w in possible_values_w:
                w_Values = value_w * np.ones([x.shape[0],1])
                x_values = np.concatenate([x, w_Values], 1)
                cov += self.cross_cov(x_values,XW)

        return cov / float(N)

    def grad_new_point(self, new, inputs):
        """Computes the vector of the gradients of Sigma_{0}(new,XW[i,:]) for
                all the past observations XW[i,]. Sigma_{0} is the covariance of
                the GP on F.
        """
        ##to do: generalize to the general case of x,w
        N_points = inputs.shape[0]
        gradXSigma0 = np.zeros([N_points + 1, self.n1])
        gradWSigma0 = np.zeros([N_points + 1, 1])

        first_grad = self.kernels[1].grad_new_point(new, inputs)
        first_grad = (first_grad[0][0:N_points, :], first_grad[1][0:N_points, :])

        kern_1 = self.kernels[0].cross_cov(new, inputs)[0,:]
        kern_2 = self.kernels[1].cross_cov(new, inputs)[0,:]

        second_grad = self.kernels[0].grad_new_point(new, inputs)
        second_grad = (second_grad[0][0:N_points, :], second_grad[1][0:N_points, :])

        for i in range(self.dim_x):

            prod_1 = (kern_1 *first_grad[0][:,i], kern_1 * first_grad[1][:,0])

            prod_2 = (kern_2 * second_grad[0][:,i], kern_2 * second_grad[1][:,0])
            gradXSigma0[0:N_points, i] = prod_1[0] + prod_2[0]
            gradWSigma0[0:N_points, 0] = prod_1[1] + prod_2[1]

        return gradXSigma0, gradWSigma0

    def grad_x_b_function(self, x, XW, discrete_uniform=True, possible_values_w=None):
        """Computes the vector of gradients with respect to x_{n+1} of
            B(x_{p},n+1)=\int\Sigma_{0}(x_{p},w,x_{n+1},w_{n+1})dp(w),
            where x_{p} is a point in the discretization of the domain of x.
        """

        cov = 0.0
        N = 1.0
        if discrete_uniform:
            N = len(possible_values_w)
            for value_w in possible_values_w:
                w_Values = value_w * np.ones([x.shape[0],1])
                x_values = np.concatenate([x, w_Values], 1)
                cov += self.grad_new_point(XW, x_values)[0]

        return cov / float(N)
