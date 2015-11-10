#!/usr/bin/env python
#from optimization import *



###m is a SEK object
def kernOptWrapper(m,**kwargs):
    """
    This function just wraps the optimization procedure of a kernel
    object so that optimize() pickleable (necessary for multiprocessing).
    """
    m.optimizeKernel(**kwargs)
    return m.optRuns[-1]

###m is a SBO object
def VOIOptWrapper(m,**kwargs):
    """
    This function just wraps the optimization procedure of a kernel
    object so that optimize() pickleable (necessary for multiprocessing).
    """
    m.optimizeVOI(**kwargs)
    return m.optRuns[-1]

def AnOptWrapper(m,**kwargs):
    """
    This function just wraps the optimization procedure of a kernel
    object so that optimize() pickleable (necessary for multiprocessing).
    """
    m.optimizeAn(**kwargs)
    return m.optRuns[-1]