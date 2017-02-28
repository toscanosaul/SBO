from abc import ABCMeta, abstractmethod
import numpy as np
import numpy.random as npr
import scipy.stats as sps
from operator import add  # same as lambda x,y:x+y I think

class AbstractPrior(object):
    __metaclass__ = ABCMeta

    @abstractmethod
    def logprob(self, x):
        pass

        # Some of these are "improper priors" and I cannot sample from them
        # In this case the sample method will just return None
        # (or could raise an exception)
        # In any case the sampling should only be used for debugging
        # Unless we want to initialize the hypers by sampling from the prior?
        # def sample(self, n_samples):
        #     # raise Exception("Sampling not implemented for composed prior")
        #     return None


class Tophat(AbstractPrior):
    def __init__(self, xmin, xmax):
        self.xmin = xmin
        self.xmax = xmax
        if not (xmax > xmin):
            raise Exception("xmax must be greater than xmin")

    def logprob(self, x):
        if np.any(x < self.xmin) or np.any(x > self.xmax):
            return -np.inf
        else:
            return 0.  # More correct is -np.log(self.xmax-self.xmin), but constants don't matter

    def sample(self, n_samples):
        return self.xmin + npr.rand(n_samples) * (self.xmax - self.xmin)


# This is the Horseshoe prior for a scalar entity
# The multivariate Horseshoe distribution is not properly implemented right now
# None of these are, really. We should fix that up at some point, you might
# have to tell it the size in the constructor (e.g. with kwarg: dims=1)
# (Not that we ever really want the multivariate one as a prior, do we?)
# (I think more often we'd just want to vectorize the univariate one)
class Horseshoe(AbstractPrior):
    def __init__(self, scale):
        self.scale = scale

    # THIS IS INEXACT
    def logprob(self, x):
        if np.any(x == 0.0):
            return np.inf  # POSITIVE infinity (this is the "spike")
        # We don't actually have an analytical form for this
        # But we have a bound between 2 and 4, so I just use 3.....
        # (or am I wrong and for the univariate case we have it analytically?)
        return np.sum(np.log(np.log(1 + 3.0 * (self.scale / x) ** 2)))

    def sample(self, n_samples):
        # Sample from standard half-cauchy distribution
        lamda = np.abs(npr.standard_cauchy(size=n_samples))

        # I think scale is the thing called Tau^2 in the paper.
        return npr.randn() * lamda * self.scale
        # return npr.multivariate_normal()


class Lognormal(AbstractPrior):
    def __init__(self, scale, mean=0):
        self.scale = scale
        self.mean = mean

    def logprob(self, x):
        return np.sum(sps.lognorm.logpdf(x, self.scale, loc=self.mean))

    def sample(self, n_samples):
        return npr.lognormal(mean=self.mean, sigma=self.scale, size=n_samples)


class LognormalTophat(AbstractPrior):
    def __init__(self, scale, xmin, xmax, mean=0):
        self.scale = scale
        self.mean = mean
        self.xmin = xmin
        self.xmax = xmax

        if not (xmax > xmin):
            raise Exception("xmax must be greater than xmin")

    def logprob(self, x):
        if np.any(x < self.xmin) or np.any(x > self.xmax):
            return -np.inf
        else:
            return np.sum(sps.lognorm.logpdf(x, self.scale, loc=self.mean))

    def sample(self, n_samples):
        raise Exception('Sampling of LognormalTophat is not implemented.')


# Let X~lognormal and Y=X^2. This is distribution of Y.
class LognormalOnSquare(Lognormal):
    def logprob(self, y):
        if np.any(y < 0):  # Need this here or else sqrt(y) may occur with y < 0
            return -np.inf

        x = np.sqrt(y)
        dy_dx = 2 * x  # this is the Jacobean or inverse Jacobean, whatever
        # p_y(y) = p_x(sqrt(x)) / (dy/dx)
        # log p_y(y) = log p_x(x) - log(dy/dx)
        return Lognormal.logprob(self, x) - np.log(dy_dx)

    def sample(self, n_samples):
        return Lognormal.sample(self, n_samples) ** 2


class LogLogistic(AbstractPrior):
    def __init__(self, shape, scale=1):
        self.shape = shape
        self.scale = scale

    def logprob(self, x):
        return np.sum(sps.fisk.logpdf(x, self.shape, scale=self.scale))


class Exponential(AbstractPrior):
    def __init__(self, mean):
        self.mean = mean

    def logprob(self, x):
        return np.sum(sps.expon.logpdf(x, scale=self.mean))

    def sample(self, n_samples):
        return npr.exponential(scale=self.mean, size=n_samples)


class Gaussian(AbstractPrior):
    def __init__(self, mu, sigma):
        self.mu = mu
        self.sigma = sigma

    def logprob(self, x):
        return np.sum(sps.norm.logpdf(x, loc=self.mu, scale=self.sigma))

    def sample(self, n_samples):
        return self.mu + npr.randn(n_samples) * self.sigma


class MultivariateNormal(AbstractPrior):
    def __init__(self, mu, cov):
        self.mu = mu
        self.cov = cov

        if mu.size != cov.shape[0] or cov.shape[0] != cov.shape[1]:
            raise Exception(
                "mu should be a vector and cov a matrix, of matching sizes")

    def logprob(self, x):
        return sps.multivariate_normal.logpdf(x, mean=self.mu, cov=self.cov)

    def sample(self, n_samples):
        return npr.multivariate_normal(self.mu, self.cov,
                                       size=n_samples).T.squeeze()


class NoPrior(AbstractPrior):
    def __init__(self):
        pass

    def logprob(self, x):
        return 0.0


# This class takes in another prior in its constructor
# And gives you the nonnegative version (actually the positive version, to be numerically safe)
class NonNegative(AbstractPrior):
    def __init__(self, prior):
        self.prior = prior

        if hasattr(prior, 'sample'):
            self.sample = lambda n_samples: np.abs(self.prior.sample(n_samples))

    def logprob(self, x):
        if np.any(x <= 0):
            return -np.inf
        else:
            return self.prior.logprob(x)  # + np.log(2.0)
            # Above: the log(2) makes it correct, but we don't ever care about it I think


# This class allows you to compose a list priors
# (meaning, take the product of their PDFs)
# The resulting distribution is "improper" -- i.e. not normalized
class ProductOfPriors(AbstractPrior):
    def __init__(self, priors):
        self.priors = priors

    def logprob(self, x):
        lp = 0.0
        for prior in self.priors:
            lp += prior.logprob(x)
        return lp


# class Binomial(AbstractPrior):
#     def __init__(self, p, n):
#         self.p = p
#         self.n = n

#     def logprob(self, k):
#         pos = k
#         neg = self.n-k

#         with np.errstate(divide='ignore'):  # suppress warnings about log(0)
#             return np.sum( pos[pos>0]*np.log(self.p[pos>0]) ) + np.sum( neg[neg>0]*np.log(1-self.p[neg>0]) )

#     def sample(self, n_samples):
#         return np.sum(npr.rand(n, n_samples) < p, axis=0)

# class Bernoulli(Binomial):
#     def __init__(self, p):
#         super(Bernoulli, self).__init__(p, 1)


def ParseFromOptions(options):
    parsed = dict()
    for p in options:
        prior_class = eval(options[p]['distribution'])
        args = options[p]['parameters']

        # If they give a list, just stick them in order
        # If they give something else (hopefully a dict of some sort), pass them in as kwargs
        if isinstance(args, list):
            parsed[p] = prior_class(*args)
        elif isinstance(args,
                        dict):  # use isinstance() not type() so that defaultdict, etc are allowed
            parsed[p] = prior_class(**args)
        else:
            raise Exception("Prior parameters must be list or dict type")

    return parsed
