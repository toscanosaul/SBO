

def kernOptWrapper(m, start , *args):
    """
    This function just wraps the optimization procedure of a kernel
    object so that optimize() pickleable (necessary for multiprocessing).

    Args:
        m: kernel object
    """

    if args:
        args = args[0]
        m.observed_inputs = args[0]
        m.observed_values = args[1]
        m.noise = args[2]
    result = m.do_optimization(start=start)
    return result


