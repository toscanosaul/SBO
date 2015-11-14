#!/usr/bin/env python

"""
We define classes needed to run the SBO algorithm:

-objective: This class consits of functions and elements related to the
            objective function f(x,w,z) and distribution of w.
            
-Miscellaneous: Consists of path where the results are saved; the random
                seed used to run the program; and specifies if the program
                is run in parallel.

-opt: This class consists of functions and variables used in the gradient ascent
      method when optimizing the VOI and the expectation of the GP. This functions
      allow us to define the termination condition for gradient ascent. Moreover,
      we may want to transform the space of x to a more convenient space, e.g.
      dimension reduction.

-data: This class consists of all the history.

"""

class objective:
    """
    Args:
        -fobj: The simulator or objective function.
        -dimSeparation: Dimension of w.
        -noisyF: Estimator of the conditional expectation given w,
                 F(x,w)=E[f(x,w,z)|w].
        -numberEstimateF: Number of samples used to estimate F.
        -sampleFromX: Chooses a point x at random.
        -simulatorW: Simulates a random vector w.
        -estimationObjective: Estimates the expectation of fobj.
    """
    def __init__(self, fobj,dimSeparation,noisyF,numberEstimateF,SampleFromX,
                 simulatorW,estimationObjective):
        self.fobj=fobj
        self.dimSeparation=dimSeparation
        self.noisyF=noisyF
        self.numberEstimateF=numberEstimateF
        self.sampleFromX=SampleFromX
        self.simulatorW=simulatorW
        self.estimationObjective=estimationObjective

class Miscellaneous:
    def __init__(self,randomSeed,parallel,folder,create):
        """
        Args:
            -randomSeed: Random seed used to run the problem. Only needed for the
                         name of the files with the results.
            -parallel: True if we want to run the multistart gradient ascent algorithm
                       and train the kernel of the GP in parallel; otherwise, it's false. 
            -folder: Path where the files with the results are saved. 
            -create: True if we want to create new files for the results; it's false
                     otherwise. If we want to add more results to our previos results,
                     this variable should be false.
        """
        self.rs=randomSeed
        self.parallel=parallel
        self.folder=folder
        self.create=create

class opt:
    def __init__(self,numberParallel,dimXsteepest,transformationDomainX,
                 transformationDomainW,projectGradient,functionGradientAscentVn,
                 functionGradientAscentAn,functionConditionOpt,xtol):
        """
        Args:
        -numberParallel: Number of starting points for the multistart gradient
                         ascent algorithm.
        -dimXsteepest: Dimension of x when the VOI and a_{n} are optimized. We
                       may want to reduce the dimension of the original problem.
        -transformationDomainX: Transforms the point x given by the steepest ascent
                                method to the right domain of x.
        -transformationDomainW: Transforms the point w given by the steepest ascent
                                method to the right domain of w.
        -projectGradient: Project a point x to the domain of the problem at each step
                          of the gradient ascent method if needed.
        -functionGradientAscentVn: Function used for the gradient ascent method. It
                                   evaluates the VOI, when grad and onlyGradient are
                                   False; it evaluates the VOI and computes its
                                   derivative when grad is True and onlyGradient is False,
                                   and computes only its gradient when gradient and
                                   onlyGradient are both True. 
        -functionGradientAscentAn: Function used for the gradient ascent method. It evaluates
                                   a_{n}, when grad and onlyGradient are False; it evaluates
                                   a_{n} and computes its derivative when grad is True and
                                   onlyGradient is False, and computes only its gradient when
                                   gradient and onlyGradient are both True.
        -functionConditionOpt: Gives the stopping rule for the steepest ascent method, e.g. the
                               function could be the Euclidean norm. 
        -xtol: Tolerance of x for the convergence of the steepest ascent method.
        """
        self.numberParallel=numberParallel
        self.dimXsteepest=dimXsteepest
        self.transformationDomainX=transformationDomainX
        self.transformationDomainW=transformationDomainW
        self.projectGradient=projectGradient
        self.functionGradientAscentVn=functionGradientAscentVn
        self.functionGradientAscentAn=functionGradientAscentAn
        self.functionConditionOpt=functionConditionOpt
        self.xtol=xtol

class data:
    def __init__(self,Xhist,yHist,varHist):
        """
        Arguments:
            -Xhist: Past vectors (x,w).
            -yHist: Past outputs.
            -varHist: Past variances of the outputs.
        """
        self.Xhist=Xhist
        self.yHist=yHist
        self.varHist=varHist