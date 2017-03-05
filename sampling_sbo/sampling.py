import numpy as np
import numpy.random as npr
from abc import ABCMeta, abstractmethod

from params import Param as Hyperparameter
from params import set_params_from_array
from params import params_to_array
from slice_sampling_versions import slice_sample
from slice_sampling_versions import slice_sample_surrogate


class AbstractSampler(object):
    __metaclass__ = ABCMeta

    def __init__(self, *params_to_sample, **sampler_options):
        self.params          = params_to_sample
        self.sampler_options = sampler_options
        self.current_ll      = None

        # Note: thinning is currently implemented such that each sampler does its thinning
        # We could also do a different type of thinning, implemented in SamplerCollection,
        # where all samplers produce a sample, and then you thin (ABABAB rather than AAABBB)
     #   self.thinning_overrideable = not sampler_options.has_key('thinning') # Thinning can be overrided if True
        self.thinning              = sampler_options.get('thinning', 0)

    @abstractmethod
    def logprob(self, x, model, **kwargs):
        pass

    @abstractmethod
    def sample(self, model):
        pass

class Standard_Slice_Sample(AbstractSampler):

    def logprob(self, x, model, **kwargs):
        """compute the log probability of observations x

        This includes the model likelihood as well as any prior
        probability of the parameters

        Returns
        -------
        lp : float
            the log probability
        """
        # set values of the parameers in self.params to be x
        set_params_from_array(self.params, x)

        lp = 0.0
        # sum the log probabilities of the parameter priors
        for param in self.params:
            lp += param.prior_logprob()

            if np.isnan(lp):  # Positive infinity should be ok, right?
                print 'Param diagnostics:'
               # param.print_diagnostics()
                print 'Prior logprob: %f' % param.prior_logprob()
                raise Exception("Prior returned %f logprob" % lp)

        if not np.isfinite(lp):
            return lp

        # include the log probability from the model
        lp += model.log_likelihood()

        if np.isnan(lp):
            raise Exception("Likelihood returned %f logprob" % lp)

        return lp

    def sample(self, model):
        """generate a new sample of parameters for the model

        Notes
        -----
        The parameters are stored as self.params which is a list of Params objects.
        The values of the parameters are updated on each call.  Pesumably the value of
        the parameter affects the model (this is not required, but it would be a bit
        pointless othewise)

        """
        # turn self.params into a 1d numpy array
        params_array = params_to_array(self.params)
        for i in xrange(self.thinning + 1):
            # get a new value for the parameter array via slice sampling
            params_array, current_ll = slice_sample(
                params_array,
                self.logprob,
                model,
                **self.sampler_options
            )

            set_params_from_array(
                self.params,
                params_array
            )  # Can this be untabbed safely?

        self.current_ll = current_ll  # for diagnostics


class Slice_Sample_Surrogate(AbstractSampler):

    def logprob(self, x, model, eta, g):
        """compute the log probability of observations x

        This includes the model likelihood as well as any prior
        probability of the parameters

        Returns
        -------
        lp : float
            the log probability
        """

        L_r = model.compute_chol_r()
        f = np.dot(L_r, eta) + model.compute_m_theta(L_r, g)

        # set values of the parameers in self.params to be x
        set_params_from_array(self.params, x)

        lp = 0.0
        # sum the log probabilities of the parameter priors
        for param in self.params:
            lp += param.prior_logprob()

            if np.isnan(lp):  # Positive infinity should be ok, right?
                print 'Param diagnostics:'
               # param.print_diagnostics()
                print 'Prior logprob: %f' % param.prior_logprob()
                raise Exception("Prior returned %f logprob" % lp)


        if not np.isfinite(lp):
            return lp

        # include the log probability from the model
        lp += model.log_likelihood_data_given_latent(
            f=f
        )



        lp += model.log_likelihood_g_given_latent(f, g)


        lp += model.log_likelihood_latent_given_params(f)

        lp += model.jacobian_likelihood()


        if np.isnan(lp):
            raise Exception("Likelihood returned %f logprob" % lp)



        return lp

    def sample(self, model):
        """generate a new sample of parameters for the model

        Notes
        -----
        The parameters are stored as self.params which is a list of Params objects.
        The values of the parameters are updated on each call.  Pesumably the value of
        the parameter affects the model (this is not required, but it would be a bit
        pointless othewise)

        """
        # turn self.params into a 1d numpy array

        init_f = model.sample_f_given_theta()
        init_f = model.observed_values - model.mean.value
        params_array = params_to_array(self.params)

        g = model.sample_g_given_theta_f(init_f)
        eta = model.get_eta(init_f, g)

        for i in xrange(self.thinning + 1):
            # get a new value for the parameter array via slice sampling
            params_array, current_ll = slice_sample(
                params_array,
                self.logprob,
                model,
                eta,
                g,
                **self.sampler_options
            )

            set_params_from_array(
                self.params,
                params_array
            )  # Can this be untabbed safely?

        self.current_ll = current_ll  # for diagnostics