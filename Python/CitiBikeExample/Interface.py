#!/usr/bin/env python

"""
-fobj: The simulator or objective function.
-dimSeparation: Dimension of w.
-noisyF: Estimator of the conditional expectation given w, F(x,w)=E[f(x,w,z)|w].
-numberEstimateF: Observations used to estimate F.
-sampleFromX: Chooses a point x at random.
-simulatorW: Simulates a random vector w.
-estimationObjective: Estimates the expectation of fobj.

"""

class objective:
    def __init__(self, fobj,dimSeparation,noisyF,numberEstimateF,SampleFromX,
                 simulatorW,estimationObjective):
        self.fobj=fobj
        self.dimSeparation=dimSeparation
        self.noisyF=noisyF
        self.numberEstimateF=numberEstimateF
        self.SampleFromX=SampleFromX
        self.simulatorW=simulatorW
        self.estimationObjective=estimationObjective


"""
-randomSeed: Random seed used to run the problem. Only needed for the name of the
	     files with the results.
-parallel: True if we want to run the multistart gradient ascent algorithm and
	   train the kernel of the GP in parallel; otherwise, it's false. 
-folder: Direction where the files with the results are saved. 
-create: True if we want to create new files for the results; it's false
		 otherwise. If we want to add more results to our previos results,
		 this variable should be false.

"""

class Miscellaneous:
    def __init__(self,randomSeed,parallel,folder,create):
        self.rs=randomSeed
        self.parallel=parallel
        self.folder=folder
        self.create=create