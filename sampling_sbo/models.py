import sys
import logging
import numpy        as np
import numpy.random as npr
import scipy.linalg as spla
import scipy.stats  as sps

from params import Param as Hyperparameter
from kernels import Matern52, multi_task, ProductKernel
from sampling import Standard_Slice_Sample
import priors
from scipy import linalg
from scipy.optimize import fmin_l_bfgs_b

from abc import ABCMeta, abstractmethod
from joblib import Parallel, delayed
import multiprocessing as mp
from utilities import kernOptWrapper


try:
    module = sys.modules['__main__'].__file__
    log    = logging.getLogger(module)
except:
    log    = logging.getLogger()
    print 'Not running from main.'


class AbstractModel(object):
    __metaclass__ = ABCMeta

    @abstractmethod
    def to_dict(self):
        pass

  #  @abstractmethod
  #  def from_dict(self):
  #      pass

  #  @abstractmethod
  #  def fit(self, inputs, values, meta=None, hypers=None):
  #      pass

    @abstractmethod
    def log_likelihood(self):
        pass

#    @abstractmethod
#    def predict(self, pred, full_cov=False, compute_grad=False):
#        pass

DEFAULT_MCMC_ITERS = 10
DEFAULT_BURNIN = 100


class K_Folds(AbstractModel):

    def __init__(self, num_dims, **options):
        self.num_dims = num_dims
        self.num_params = num_dims-1 + np.cumsum(np.arange(6))[-1] + 1
       # self._set_likelihood(options)

        log.debug('GP received initialization options: %s' % (options))

        self.mcmc_iters = int(options.get("mcmc_iters", DEFAULT_MCMC_ITERS))
        self.burnin = int(options.get("burnin", DEFAULT_BURNIN))
        self.thinning = int(options.get("thinning", 0))

        self.observed_inputs = options.get('X_data', None)
        self.observed_values = options.get('y', None)
        self.noise = options.get('noise', None)
        self.noiseless = options.get('noiseless', True)


        self.log_values_multikernel = options.get('log_multiKernel', np.ones(15))
        self.values_matern = options.get('matern', np.ones(4))

        self.params = None

        self._caching = bool(options.get("caching", True))
        self._cache_list = []  # Cached computations for re-use.
        self._hypers_list = []  # Hyperparameter dicts for each state.

        self._samplers = {}

        self._kernel = None
        self._kernel_with_noise = None

        self.num_states = 0
        self.chain_length = 0

        self.max_cache_mb = 256  # TODO -- make in config
        self.max_cache_bytes = self.max_cache_mb * 1024 * 1024

        self._build()

    def _build(self):

        multi = multi_task(15, self.num_dims )
        matern = Matern52(self.num_dims)

        multi.ls.value = self.log_values_multikernel
        matern.hypers.value = self.values_matern

        params = [multi, matern]
        self._kernel = ProductKernel(15, *params)



        # Build the mean function (just a constant mean for now)
        self.mean = Hyperparameter(
            initial_value = np.mean(self.observed_values),
            prior         = priors.Tophat(-1000, 1000),
            name          = 'mean'
        )

        # Get the hyperparameters to sample
        ls                      = self._kernel.hypers

        self.params = {
            'mean'       : self.mean,
            'ls'         : ls,
        }

        self._samplers['mean'] = Standard_Slice_Sample(
            self.params['mean'],
            compwise=False,
            thinning=self.thinning
        )

        self._samplers['ls'] = Standard_Slice_Sample(
            self.params['ls'],
            compwise=True,
            thinning=self.thinning
        )

    def log_likelihood(self):
        """
        GP Marginal likelihood

        Notes
        -----
        This is called by the samplers when fitting the hyperparameters.
        """
        cov = self._kernel.cov(self.observed_inputs) + np.diag(self.noise)

        chol = spla.cholesky(cov, lower=True)
        solve = spla.cho_solve((chol, True),
                               self.observed_values - self.mean.value)

        # Uses the identity that log det A = log prod diag chol A = sum log diag chol A
        return -np.sum(np.log(np.diag(chol))) - 0.5 * np.dot(
            self.observed_values - self.mean.value,
            solve
        )

    def grad_log_likelihood(self):
        grad = np.zeros(self.num_params)

        grad_cov = self._kernel.gradient(self.observed_inputs)

        for i in range(self.num_params-1):
            grad[i] = self._compute_gradient_llh(grad_cov[i])

        grad[self.num_params-1] = self._compute_gradient_mean()
        return grad

    def _compute_gradient_mean(self):
        cov = self._kernel.cov(self.observed_inputs) + np.diag(self.noise)
        chol = spla.cholesky(cov, lower=True)
        solve = spla.cho_solve((chol, True),
                               -1.0 * np.ones(len(self.noise)))
        return -1.0 * np.dot((self.observed_values - self.mean.value), solve)

    def _compute_gradient_llh(self, gradK):

        cov = self._kernel.cov(self.observed_inputs) + np.diag(self.noise)

        chol = spla.cholesky(cov, lower=True)
        solve = spla.cho_solve((chol, True),
                               self.observed_values - self.mean.value)
        solve = solve.reshape((len(solve),1))
        solve_1 = spla.cho_solve((chol, True), gradK)

        product = np.dot(np.dot(solve, solve.transpose()), gradK)

        sol = 0.5 * np.trace(product - solve_1)

        return sol

    def minus_likelihood(self, param):
        self._update_params_kernel(param)
        return -1.0 * self.log_likelihood()

    def gradient(self, param):
        self._update_params_kernel(param)
        return -1.0 * self.grad_log_likelihood()

    def do_optimization(self, start):
        return fmin_l_bfgs_b(self.minus_likelihood, start, self.gradient)

    def mle_parameters(self, start=None, n_restarts=1):

        if start is None:
            ls = self.params['ls'].value
            mean = self.params['mean'].value

            ls = ls.reshape([1, len(ls)])
            mean = np.array([mean]).reshape([1, 1])

            default_point = np.concatenate([ls, mean], 1)
            if n_restarts == 1:
                starting_points = default_point
            else:
                starting_points = self._get_starting_points(n_restarts-1)
                starting_points = np.concatenate(
                    [default_point, starting_points], 0
                )
        if n_restarts == 1:
            opt = fmin_l_bfgs_b(self.minus_likelihood, starting_points[0,:], self.gradient)
        else:
            n_jobs = mp.cpu_count()

            results = Parallel(n_jobs=n_jobs)(
                delayed(kernOptWrapper)(
                    self,
                    starting_points[i, :]
                ) for i in range(n_restarts))

            opt_values = []
            for i in range(n_restarts):
                try:
                    opt_values.append(results[i])
                except Exception as e:
                    print "opt failed"

            j = np.argmin([o[1] for o in opt_values])
            opt = opt_values[j]

        self.params['ls'].value = opt[0][0:self.num_params-1]
        self.params['mean'].value = opt[0][self.num_params-1]

        return opt

    def _get_starting_points(self, n):

        mu = np.array([np.mean(self.observed_values)])

        new_sampled = self._kernel.ls.sample_from_prior(n)

        new_sampled = np.concatenate([new_sampled, mu*np.ones((n,1))],1)

        return new_sampled


    def _update_params_kernel(self, param):
        self.params['mean'].value = param[self.num_params-1]

        for i in range(self.num_params-1):
            self.params['ls'].value[i] = param[i]

    def _burn_samples(self, num_samples):
        for i in xrange(num_samples):
            for sampler in self._samplers:
                self._samplers[sampler].sample(self)

            self.chain_length += 1

    def _collect_samples(self, num_samples):
        hypers_list = []
        for i in xrange(num_samples):
            for sampler in self._samplers:
                print sampler
                self._samplers[sampler].sample(self)
            hypers_list.append(self.to_dict()['hypers'])
            self.chain_length += 1
        print "likelihood :%f" % self.log_likelihood()
        return hypers_list

    def to_dict(self):
        """return a dictionary that saves the values of the hypers and the chain length"""
        gp_dict = {'hypers' : {}}
        for name, hyper in self.params.iteritems():
            gp_dict['hypers'][name] = hyper.value

        gp_dict['chain length'] = self.chain_length

        return gp_dict
