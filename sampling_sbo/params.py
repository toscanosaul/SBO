import copy

import numpy as np

import priors
#from compression import compress_array


def set_params_from_array(params_iterable, params_array):
    """Update the params in params_iterable with the new values stored in params_array"""
    index = 0
    for param in params_iterable:
        if param.size() == 1 and not param.isArray:
            param.value = params_array[
                index]  # Pull it out of the array, becomes a float
        else:
            param.value = params_array[index:index + param.size()]
        index += param.size()


def params_to_array(params_iterable):
    """Put the params in params_iterable into a 1D numpy array for sampling"""
    return np.hstack([param.value for param in params_iterable])
    # Not sure if copying is really needed


def params_to_dict(params_iterable):
    params_dict = {}
    for param in params_iterable:
        params_dict[param.name] = param.value

    return params_dict


#def params_to_compressed_dict(params_iterable):
#    params_dict = {}
#    for param in params_iterable:
#        params_dict[param.name] = compress_array(param.value)

#    return params_dict


# TODO: get rid of this isArray stuff and just always make it a numpy array
# or else make it check things more carefully
class Param(object):
    """A class to represent a parameter
    """

    def __init__(self, initial_value, prior=priors.NoPrior(), name="Unnamed"):
        self.initial_value = copy.copy(initial_value)
        self.value = initial_value
        self.name = name
        self.prior = prior
        self.isArray = hasattr(initial_value,
                               "shape") and initial_value.shape != ()
        # If the initial value is in a numpy array, keep it as a numpy array later on
        # If the initial value is a single number (float), respect that later on

    def set_value(self, new_value):
        self.value = new_value

    def reset_value(self):
        self.value = self.initial_value

    def get_value(self, i):
        if i < 0 or i >= self.size():
            raise Exception("Param %s: %d out of bounds, size=%d" % (
            self.name, i, self.size()))
        if self.isArray:
            return self.value[i]
        else:
            return self.value

    def size(self):
        try:
            return self.value.size
        except:
            return 1

    def prior_logprob(self):
        return self.prior.logprob(self.value)

    # For MCMC diagnostics -- or maybe will be used for initialization at some point
    def sample_from_prior(self, nSamples):
        if hasattr(self.prior, 'sample'):
            return self.prior.sample(nSamples)
        else:
            raise Exception(
                "Param %s has prior %s, which does not allow sampling" % (
                self.name, self.prior.__class__.__name__))

        # self.value = np.squeeze(self.value) # sampler often creates extra dimension

        try:  # If it is a numpy array of size 1, cast it to a float (this might not be needed)
            self.value = float(self.value)
        except:
            pass

    def print_diagnostics(self):
        if self.size() == 1:
            print '    %s: %s' % (self.name, self.value)
        else:
            print '    %s: min=%s, max=%s (size=%d)' % (
            self.name, self.value.min(), self.value.max(), self.size())