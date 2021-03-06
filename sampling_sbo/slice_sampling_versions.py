import numpy        as np
import numpy.random as npr


def slice_sample(init_x, logprob, *logprob_args, **slice_sample_args):
    """generate a new sample from a probability density using slice sampling

    Parameters
    ----------
    init_x : array
    logprob : callable, `lprob = logprob(x, *logprob_args)`
        A functions which returns the log probability at a given
        location
    *logprob_args :
        additional arguments are passed to logprob

    TODO: this function has too many levels and is hard to read.  It would be clearer
    as a class or just moving the sub-functions to another location

    Returns
    -------
    new_x : float
        the sampled position
    new_llh : float
        the log likelihood at the new position (I'm not sure about this?)

    Notes
    -----
    http://en.wikipedia.org/wiki/Slice_sampling
    """
    sigma = slice_sample_args.get('sigma', 1.0)
    step_out = slice_sample_args.get('step_out', True)
    max_steps_out = slice_sample_args.get('max_steps_out', 1000)
    compwise = slice_sample_args.get('compwise', True)
    doubling_step = slice_sample_args.get('doubling_step', True)
    verbose = slice_sample_args.get('verbose', False)

    if type(init_x) == float or isinstance(init_x, np.number):
        init_x = np.array([init_x])
        scalar = True
    else:
        scalar = False

    dims = init_x.shape[0]
    if compwise:
        ordering = range(dims)
        npr.shuffle(ordering)
        new_x = init_x.copy()
        for d in ordering:
            direction = np.zeros((dims))
            direction[d] = 1.0
            new_x, new_llh = direction_slice(
                direction,
                new_x,
                logprob,
                sigma,
                step_out,
                doubling_step,
                max_steps_out,
                verbose,
                *logprob_args
            )
    else:
        direction = npr.randn(dims)
        direction = direction / np.sqrt(np.sum(direction ** 2))
        new_x, new_llh = direction_slice(
            direction,
            init_x,
            logprob,
            sigma,
            step_out,
            doubling_step,
            max_steps_out,
            verbose,
            *logprob_args
        )

    if scalar:
        return float(new_x[0]), new_llh
    else:
        return new_x, new_llh


def slice_sample_surrogate(init_x, logprob, *logprob_args, **slice_sample_args):

    sigma = slice_sample_args.get('sigma', 1.0)
    step_out = slice_sample_args.get('step_out', True)
    max_steps_out = slice_sample_args.get('max_steps_out', 1000)
    compwise = slice_sample_args.get('compwise', True)
    doubling_step = slice_sample_args.get('doubling_step', True)
    verbose = slice_sample_args.get('verbose', False)

    if type(init_x) == float or isinstance(init_x, np.number):
        init_x = np.array([init_x])
        scalar = True
    else:
        scalar = False

    ###TO DO: version that's not compwise
    dims = init_x.shape[0]

    ordering = range(dims)
    npr.shuffle(ordering)
    new_x = init_x.copy()
    for d in ordering:
        direction = np.zeros((dims))
        direction[d] = 1.0
        new_x, new_llh = direction_slice(
            direction,
            new_x,
            logprob,
            sigma,
            step_out,
            doubling_step,
            max_steps_out,
            verbose,
            *logprob_args
        )

    if scalar:
        return float(new_x[0]), new_llh
    else:
        return new_x, new_llh


def acceptable(z, llh_s, L, U, sigma, dir_logprob):
    while (U - L) > 1.1 * sigma:
        middle = 0.5 * (L + U)
        splits = (middle > 0 and z >= middle) or (
        middle <= 0 and z < middle)
        if z < middle:
            U = middle
        else:
            L = middle
        # Probably these could be cached from the stepping out.
        if splits and llh_s >= dir_logprob(U) and llh_s >= dir_logprob(
                L):
            return False
    return True


def find_x_interval(
        doubling_step,
        dir_logprob,
        llh_s,
        lower,
        upper,
        sigma,
        l_steps_out,
        u_steps_out,
        max_steps_out
):
    if doubling_step:
        while (dir_logprob(lower) > llh_s or dir_logprob(upper) > llh_s) and (l_steps_out + u_steps_out) < max_steps_out:
            if npr.rand() < 0.5:
                l_steps_out += 1
                lower -= (upper - lower)
            else:
                u_steps_out += 1
                upper += (upper - lower)
    else:
        while dir_logprob(
                lower) > llh_s and l_steps_out < max_steps_out:
            l_steps_out += 1
            lower -= sigma
        while dir_logprob(
                upper) > llh_s and u_steps_out < max_steps_out:
            u_steps_out += 1
            upper += sigma

    return lower, upper


def find_sample(
    dir_logprob,
    direction,
    init_x,
    llh_s,
    logprob,
    start_lower,
    start_upper,
    verbose,
    l_steps_out,
    u_steps_out,
    sigma,
    *logprob_args
):
    upper = start_upper
    lower = start_lower
    steps_in =0
  #  print "prob"
   # print llh_s
    while True:

        steps_in += 1
        new_z = (upper - lower) * npr.rand() + lower
        new_llh = dir_logprob(new_z)
       # print new_llh
        if np.isnan(new_llh):
            print new_z, direction * new_z + init_x, new_llh, llh_s, init_x, logprob(
                init_x, *logprob_args)
            raise Exception("Slice sampler got a NaN")
        if new_llh > llh_s and acceptable(new_z, llh_s, start_lower,
                                          start_upper, sigma, dir_logprob):
            break
        elif new_z < 0:
            lower = new_z
        elif new_z > 0:
            upper = new_z
        else:
            raise Exception("Slice sampler shrank to zero!")

    if verbose:
        print "Steps Out:", l_steps_out, u_steps_out, " Steps In:", steps_in

    return new_z * direction + init_x, new_llh

def direction_slice(
        direction,
        init_x,
        logprob,
        sigma,
        step_out,
        doubling_step,
        max_steps_out,
        verbose,
        *logprob_args
):
    def dir_logprob(z):
        return logprob(direction * z + init_x, *logprob_args)

    upper = sigma * npr.rand()
    lower = upper - sigma

    llh_s = np.log(npr.rand()) + dir_logprob(0.0)

    l_steps_out = 0
    u_steps_out = 0
    if step_out:
        lower, upper = find_x_interval(
            doubling_step,
            dir_logprob,
            llh_s,
            lower,
            upper,
            sigma,
            l_steps_out,
            u_steps_out,
            max_steps_out
        )

    start_upper = upper
    start_lower = lower

    new_point, new_llh = find_sample(
        dir_logprob,
        direction,
        init_x,
        llh_s,
        logprob,
        start_lower,
        start_upper,
        verbose,
        l_steps_out,
        u_steps_out,
        sigma,
        logprob_args
    )

    return new_point, new_llh

