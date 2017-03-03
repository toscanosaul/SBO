

def kernOptWrapper(m, start):
    """
    This function just wraps the optimization procedure of a kernel
    object so that optimize() pickleable (necessary for multiprocessing).

    Args:
        m: kernel object
    """
    result = m.do_optimization(start=start)
    return result