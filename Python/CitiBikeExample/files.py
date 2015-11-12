#!/usr/bin/env python

import os
import numpy as np

def createNewFilesFunc(path,rs):
    f=open(os.path.join(path,'%d'%rs+"hyperparameters.txt"),'w')
    f.close()
    f=open(os.path.join(path,'%d'%rs+"XWHist.txt"),'w')
    f.close()
    f=open(os.path.join(path,'%d'%rs+"yhist.txt"),'w')
    f.close()
    f=open(os.path.join(path,'%d'%rs+"varHist.txt"),'w')
    f.close()
    f=open(os.path.join(path,'%d'%rs+"optimalSolutions.txt"),'w')
    f.close()
    f=open(os.path.join(path,'%d'%rs+"optimalValues.txt"),'w')
    f.close()
    f=open(os.path.join(path,'%d'%rs+"optVOIgrad.txt"),'w')
    f.close()
    f=open(os.path.join(path,'%d'%rs+"optAngrad.txt"),'w')
    f.close()
    
def writeTraining(ALGObj):
    with open(os.path.join(ALGObj.path,'%d'%ALGObj.rs+"XWHist.txt"), "a") as f:
        np.savetxt(f,ALGObj._XWhist)
    with open(os.path.join(ALGObj.path,'%d'%ALGObj.rs+"yhist.txt"), "a") as f:
        np.savetxt(f,ALGObj._yHist)
    with open(os.path.join(ALGObj.path,'%d'%ALGObj.rs+"varHist.txt"), "a") as f:
        np.savetxt(f,ALGObj._varianceObservations)
        
        
def writeNewPointSBO(ALGObj,optim):
    temp=optim.xOpt
    gradOpt=optim.gradOpt
    numberIterations=optim.nIterations
    gradOpt=np.sqrt(np.sum(gradOpt**2))
    gradOpt=np.array([gradOpt,numberIterations])
    xTrans=ALGObj.transformationDomainX(optim.xOpt[0:1,0:ALGObj.dimXsteepest])
    wTrans=ALGObj.transformationDomainW(optim.xOpt[0:1,ALGObj.dimXsteepest:ALGObj.dimXsteepest+ALGObj._dimW])
    temp=np.concatenate((xTrans,wTrans),1)
    ALGObj._XWhist=np.vstack([ALGObj._XWhist,temp])
    ALGObj._VOI._PointsHist=ALGObj._XWhist
    ALGObj._VOI._GP._Xhist=ALGObj._XWhist
    y,var=ALGObj._infSource(temp,ALGObj._numberSamples)
    ALGObj._yHist=np.vstack([ALGObj._yHist,y])
    ALGObj._VOI._yHist=ALGObj._yHist
    ALGObj._VOI._GP._yHist=ALGObj._yHist
    ALGObj._varianceObservations=np.append(ALGObj._varianceObservations,var)
    ALGObj._VOI._noiseHist=ALGObj._varianceObservations
    ALGObj._VOI._GP._noiseHist=ALGObj._varianceObservations
    with open(os.path.join(ALGObj.path,'%d'%ALGObj.randomSeed+"varHist.txt"), "a") as f:
        var=np.array(var).reshape(1)
        np.savetxt(f,var)
    with open(os.path.join(ALGObj.path,'%d'%ALGObj.randomSeed+"yhist.txt"), "a") as f:
        y=np.array(y).reshape(1)
        np.savetxt(f,y)
    with open(os.path.join(ALGObj.path,'%d'%ALGObj.randomSeed+"XWHist.txt"), "a") as f:
        np.savetxt(f,temp)
    with open(os.path.join(ALGObj.path,'%d'%ALGObj.randomSeed+"optVOIgrad.txt"), "a") as f:
        np.savetxt(f,gradOpt)
    self.optRuns=[]
    self.optPointsArray=[]
    
def writeSolution(ALGObj,optim):
    temp=optim.xOpt
    tempGrad=optim.gradOpt
    tempGrad=np.sqrt(np.sum(tempGrad**2))
    tempGrad=np.array([tempGrad,optim.nIterations])
    xTrans=ALGObj.transformationDomainX(optim.xOpt[0:1,0:ALGObj.dimXsteepest])
    ALGObj._solutions.append(xTrans)
    with open(os.path.join(ALGObj.path,'%d'%ALGObj.randomSeed+"optimalSolutions.txt"), "a") as f:
        np.savetxt(f,xTrans)
    with open(os.path.join(ALGObj.path,'%d'%ALGObj.randomSeed+"optimalValues.txt"), "a") as f:
        result,var=ALGObj.estimationObjective(xTrans[0,:])
        res=np.append(result,var)
        np.savetxt(f,res)
    with open(os.path.join(ALGObj.path,'%d'%ALGObj.randomSeed+"optAngrad.txt"), "a") as f:
        np.savetxt(f,tempGrad)
    self.optRuns=[]
    self.optPointsArray=[]