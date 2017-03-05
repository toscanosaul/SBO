import sys
import logging
import numpy        as np
import numpy.random as npr
import scipy.linalg as spla
import scipy
import scipy.stats  as sps

from params import Param as Hyperparameter
from kernels import Matern52, multi_task, ProductKernel
from sampling import Standard_Slice_Sample, Slice_Sample_Surrogate
import priors
from scipy import linalg
from scipy.optimize import fmin_l_bfgs_b

from abc import ABCMeta, abstractmethod
from joblib import Parallel, delayed
import multiprocessing as mp
from utilities import kernOptWrapper
from objective_function import Objective
import matplotlib.pyplot as plt


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

        self.evaluation_f = options.get('evaluation_f', None)
        self.nEvals = int(options.get('nEvals', 1))
        self.dim_x = int(options.get('dim_x'))
        self.dim_w = int(options.get('dim_w', 0))
        self.domain = options.get('domain')
        self.type_domain = options.get('type_domain', None)



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

        self.objective_function = Objective(
            f=self.evaluation_f,
            nEvals=self.nEvals,
            dim_x=self.dim_x,
            dim_w=self.dim_w,
            domain=self.domain,
            type_domain=self.type_domain
        )

        multi = multi_task(15, self.num_dims )
        matern = Matern52(self.num_dims)

        multi.ls.value = self.log_values_multikernel
        matern.hypers.value = self.values_matern

        params = [multi, matern]
        self._kernel = ProductKernel(15, *params)



        # Build the mean function (just a constant mean for now)
        if self.observed_values is not None:
            initial_value_mean = np.mean(self.observed_values)
        else:
            initial_value_mean = 0

        self.mean = Hyperparameter(
            initial_value = initial_value_mean,
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

        self.params_mle ={
            'mean': None,
            'ls': None,
        }


     #   self._samplers['ls'] = Standard_Slice_Sample(
     #       self.params['ls'],
     #       compwise=True,
     #       thinning=self.thinning
     #   )

        self._samplers['ls'] = Slice_Sample_Surrogate(
            self.params['ls'],
            compwise=True,
            thinning=self.thinning
        )



    def get_training_data(self, n, signature='1'):
        XW, evaluations = self.objective_function.generate_training_points(n)

        self.observed_inputs = XW
        self.observed_values = evaluations[:,0]
        self.noise = evaluations[:,1]

        self.params['mean'] = np.mean(self.observed_values)

        np.savetxt("observed_inputs"+signature+".txt", self.observed_inputs)
        np.savetxt("observed_values" + signature + ".txt", self.observed_values)
        np.savetxt("noise" + signature + ".txt", self.noise)

    def evaluate_function(self, x):
        eval = self.objective_function.evaluate_function(x)
        return eval

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

    def log_likelihood_data_given_latent(self, f):

        return - 0.5 * np.sum((self.observed_values - self.mean.value - f)**2 / self.noise) \
               - 0.5 * np.sum(np.log(self.noise))

    def log_likelihood_g_given_latent(self, f, g, aux_var=None):
        if aux_var is None:
            aux_var = np.diag(self.noise)

        return -0.5 * np.sum((g-f)**2 / aux_var.diagonal()) - \
               0.5 * np.sum(np.log(aux_var.diagonal()))

    def log_likelihood_latent_given_params(self, f):
        cov = self._kernel.cov(self.observed_inputs)
        chol = spla.cholesky(cov, lower=True)
        solve = spla.cho_solve((chol, True), f)

        return -0.5 * np.dot(f, solve) - np.sum(np.log(np.diag(chol)))

    def jacobian_likelihood(self):
        L_r_theta_prime = self.compute_chol_r()

        return -np.sum(np.log(np.diag(L_r_theta_prime)))

    def grad_log_likelihood(self):
        grad = np.zeros(self.num_params)
        grad_cov = self._kernel.gradient(self.observed_inputs)

        for i in range(self.num_params-1):
            grad[i] = self._compute_gradient_llh(grad_cov[i])

        grad[self.num_params-1] = self._compute_gradient_mean()
        return grad

    def sample_f_given_g_theta(self, eta):
        L_r_theta = self.compute_chol_r()
        m_theta = self.compute_m_theta(L_r_theta, g)
        return np.dot(L_r_theta, eta) + m_theta

    def sample_f_given_theta(self):
        cov = self._kernel.cov(self.observed_inputs)
        chol = spla.cholesky(cov, lower=True)
        z = np.random.normal(0, 1, cov.shape[0])
        return np.dot(chol, z)

    def sample_g_given_theta_f(self, f, aux_std=None):
        if aux_std is None:
            aux_std = np.sqrt(self.noise)
        g = f + np.random.normal(0, 1 , len(aux_std)) * aux_std
        return g

    def compute_m_theta(self, L_r_theta, g, aux_var=None):
        if aux_var is None:
            aux_var = self.noise
        part_1 = g / aux_var

        return np.dot(np.dot(L_r_theta, L_r_theta.transpose()), part_1)

    def compute_chol_r(self):
        cov = self._kernel.cov(self.observed_inputs)
        cov_inv = np.linalg.inv(cov)

        R = np.linalg.inv(cov_inv + np.diag(1.0 / self.noise))

        chol = spla.cholesky(R, lower=True)

        return chol

    def get_eta(self, f, g):
        L_r = self.compute_chol_r()
        y = f - self.compute_m_theta(L_r, g)

        eta = scipy.linalg.solve_triangular(L_r, y, lower=True)

        return eta

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
            jobs = []
            pool = mp.Pool(processes=n_jobs)

            for i in range(n_restarts):
                job = pool.apply_async(
                    kernOptWrapper, args=(self, starting_points[i, :],)
                )
                jobs.append(job)
       #     results = Parallel(n_jobs=n_jobs)(
       #         delayed(kernOptWrapper)(
       #             self,
       #             starting_points[i, :]
       #         ) for i in range(n_restarts))
            pool.close()
            pool.join()
            opt_values = []
            for i in range(n_restarts):
                try:
                    opt_values.append(jobs[i].get())
                except Exception as e:
                    print "opt failed"

            j = np.argmin([o[1] for o in opt_values])
            opt = opt_values[j]

        self.params['ls'].value = opt[0][0:self.num_params-1]
        self.params['mean'].value = opt[0][self.num_params-1]

        self.params_mle['ls'] = opt[0][0:self.num_params-1]
        self.params_mle['mean'] = opt[0][self.num_params-1]

        return opt

    def cross_validation_mle_parameters(self, XW, y, noise=None, n_restarts=1):
        training_data_sets = {}
        test_points = {}
        N = len(y)
        for i in range(N):
            selector = [x for x in range(N) if x != i]
            XW_tmp = XW[selector, :]
            y_tmp = y[selector]
            noise_tmp = None
            if noise is not None:
                noise_tmp = noise[selector]
            training_data_sets[i] = [XW_tmp, y_tmp, noise_tmp]
            test_points[i] = XW[i:i + 1, :]

        n_jobs = mp.cpu_count()

        jobs = {}

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

        try:
            pool = mp.Pool(processes=n_jobs)
            for i in range(N):
                jobs[i] = []

                for j in range(n_restarts):
                    job = pool.apply_async(
                        kernOptWrapper, args=(
                            self,
                            starting_points[j, :],
                            training_data_sets[i],
                        )
                    )
                    jobs[i].append(job)
            pool.close()
            pool.join()
        except KeyboardInterrupt:
            print "Ctrl+c received, terminating and joining pool."
            pool.terminate()
            pool.join()

        opt_values = {}
        for i in range(N):
            opt_values[i] = []
            for j in range(n_restarts):
                try:
                    opt_values[i].append(jobs[i][j].get())
                except Exception as e:
                    print "opt failed"

        opt_solutions = {}
        number_correct = 0
        means = np.zeros(N)
        standard_dev = np.zeros(N)
        for i in range(N):
            if len(opt_values[i]):
                j = np.argmin([o[1] for o in opt_values[i]])
                opt_solutions[i] = opt_values[i][j]
                self.params['ls'].value = opt_solutions[i][0][0:self.num_params - 1]
                self.params['mean'].value = opt_solutions[i][0][self.num_params - 1]
                mean, var = self.params_posterior_gp(
                    x=XW[i:i+1,:],
                    X=training_data_sets[i][0],
                    y=training_data_sets[i][1],
                    noise=training_data_sets[i][2]
                )
                if noise is None:
                    var = var
                else:
                    var = var + noise[i]
                means[i] = mean
                standard_dev[i] = np.sqrt(var)
                in_interval_1 = y[i] <= mean + 2.0*np.sqrt(var)
                in_interval_2 = y[i] >= mean - 2.0*np.sqrt(var)
                in_interval = in_interval_1 and in_interval_2
                if in_interval:
                    number_correct += 1
            else:
                "it failed for %d"%i

        print means
        print standard_dev
        plt.errorbar(np.arange(N), means, yerr=2.0 * standard_dev, fmt='o')
        plt.scatter(np.arange(N), y, color='r')
        plt.savefig("diagnostic_kernel.png")

        return number_correct, len(y), means, standard_dev

    def params_posterior_gp(self, x, X, y, noise):
        if noise is None:
            noise = np.zeros(len(y))

        cov = self._kernel.cov(X) + np.diag(noise)
        chol = spla.cholesky(cov, lower=True)
        solve = spla.cho_solve((chol, True), y - self.mean.value)

        vec_cov = self._kernel.cross_cov(x, X)

        mu_n = self.params['mean'].value + np.dot(vec_cov, solve)

        solve_2 = spla.cho_solve((chol, True), vec_cov.transpose())
        cov_n = self._kernel.cov(x) - np.dot(vec_cov, solve_2)

        return mu_n, cov_n

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
