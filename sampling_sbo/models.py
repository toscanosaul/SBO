import sys
import logging
import numpy        as np
import numpy.random as npr
import scipy.linalg as spla
import scipy.stats  as sps

from params import Param as Hyperparameter
from kernels import Matern52, multi_task, ProductKernel
from sampling import slice_sample
import priors

from abc import ABCMeta, abstractmethod

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

    @abstractmethod
    def from_dict(self):
        pass

    @abstractmethod
    def fit(self, inputs, values, meta=None, hypers=None):
        pass

    @abstractmethod
    def log_likelihood(self):
        pass

    @abstractmethod
    def predict(self, pred, full_cov=False, compute_grad=False):
        pass


DEFAULT_MCMC_ITERS = 10
DEFAULT_BURNIN = 100

class K_Folds(AbstractModel):

    def __init__(self, num_dims, **data, **options):
        self.num_dims = num_dims

        self._set_likelihood(options)

        log.debug('GP received initialization options: %s' % (options))

        self.mcmc_iters = int(options.get("mcmc_iters", DEFAULT_MCMC_ITERS))
        self.burnin = int(options.get("burnin", DEFAULT_BURNIN))
        self.thinning = int(options.get("thinning", 0))

        self.inputs = data.get('X_data', None)
        self.values = data.get('y', None)

        self.values_multikernel = data.get('multiKernel', np.ones(15))
        self.values_matern = data.get('matern', np.ones(4))



        self.params = None

        self._caching = bool(options.get("caching", True))
        self._cache_list = []  # Cached computations for re-use.
        self._hypers_list = []  # Hyperparameter dicts for each state.

        self._samplers = []

        self._kernel = None
        self._kernel_with_noise = None

        self.num_states = 0
        self.chain_length = 0

        self.max_cache_mb = 256  # TODO -- make in config
        self.max_cache_bytes = self.max_cache_mb * 1024 * 1024

        self._build()

    def _build(self):

        multi = multi_task(15, 5)
        matern = Matern52(5)

        multi.ls.value = log_covs


        z.hypers.value = np.exp(-log_mu)

        # Build the component kernels
        input_kernel           = Matern52(self.num_dims)
        stability_noise_kernel = Noise(self.num_dims) # Even if noiseless we use some noise for stability
        scaled_input_kernel    = Scale(input_kernel)
        sum_kernel             = SumKernel(scaled_input_kernel, stability_noise_kernel)
        noise_kernel           = Noise(self.num_dims)

        # The final kernel applies the transformation.
        self._kernel = TransformKernel(sum_kernel, transformer)

        # Finally make a noisy version if necessary
        if not self.noiseless:
            self._kernel_with_noise = SumKernel(self._kernel, noise_kernel)

        # Build the mean function (just a constant mean for now)
        self.mean = Hyperparameter(
            initial_value = 0.0,
            prior         = priors.Gaussian(0.0,1.0),
            name          = 'mean'
        )

        # Get the hyperparameters to sample
        ls                      = input_kernel.hypers
        amp2                    = scaled_input_kernel.hypers
        beta_alpha, beta_beta = beta_warp.hypers

        self.params = {
            'mean'       : self.mean,
            'amp2'       : amp2,
            'ls'         : ls,
            'beta_alpha' : beta_alpha,
            'beta_beta'  : beta_beta
        }

        # Build the samplers
        if self.noiseless:
            self._samplers.append(SliceSampler(self.mean, amp2, compwise=False, thinning=self.thinning))
        else:
            noise = noise_kernel.hypers
            self.params.update({'noise' : noise})
            self._samplers.append(SliceSampler(self.mean, amp2, noise, compwise=False, thinning=self.thinning))

        self._samplers.append(SliceSampler(ls, beta_alpha, beta_beta, compwise=True, thinning=self.thinning))