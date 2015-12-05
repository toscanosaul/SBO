import sys
sys.path.append("..")
import numpy as np
from simulationPoissonProcessNonHomogeneous import *
from math import *
from matplotlib import pyplot as plt
import scipy.stats as stats
from scipy.stats import norm
import statsmodels.api as sm
import multiprocessing as mp
import os
from scipy.stats import poisson
from BGO.Source import *



n1=4
n2=1


nDays=153
######

"""
We define the variables needed for the queuing simulation. 
"""

g=unhappyPeople

nSets=4

fil="poissonDays.txt"
fil=os.path.join("NonHomegeneousPP",fil)
poissonParameters=np.loadtxt(fil)

###readData

poissonArray=[[] for i in xrange(nDays)]
exponentialTimes=[[] for i in xrange(nDays)]

for i in xrange(nDays):
    fil="daySparse"+"%d"%i+"ExponentialTimesNonHom.txt"
    fil2=os.path.join("NonHomogeneousPP2",fil)
    poissonArray[i].append(np.loadtxt(fil2))
    
    fil="daySparse"+"%d"%i+"PoissonParametersNonHom.txt"
    fil2=os.path.join("NonHomogeneousPP2",fil)
    exponentialTimes[i].append(np.loadtxt(fil2))

numberStations=329
Avertices=[[]]
for j in range(numberStations):
    for k in range(numberStations):
	Avertices[0].append((j,k))

#A,lamb=generateSets(nSets,fil)

#parameterSetsPoisson=np.zeros(n2)
#for j in xrange(n2):
#    parameterSetsPoisson[j]=np.sum(lamb[j])


#exponentialTimes=np.loadtxt("2014-05"+"ExponentialTimes.txt")
with open ('json.json') as data_file:
    data=json.load(data_file)

f = open(str(4)+"-cluster.txt", 'r')
cluster=eval(f.read())
f.close()

bikeData=np.loadtxt("bikesStationsOrdinalIDnumberDocks.txt",skiprows=1)

TimeHours=4.0
numberBikes=6000

poissonParameters*=TimeHours


###upper bounds for X
upperX=np.zeros(n1)
temBikes=bikeData[:,2]
for i in xrange(n1):
    temp=cluster[i]
    indsTemp=np.array([a[0] for a in temp])
    upperX[i]=np.sum(temBikes[indsTemp])


"""
We define the objective object.
"""

def simulatorW(n,ind=False):
    """Simulate n vectors w
      
       Args:
          n: Number of vectors simulated
    """
    wPrior=np.zeros((n,n2))
    indexes=np.random.randint(0,nDays,n)
    for i in range(n):
	for j in range(n2):
	    wPrior[i,j]=np.random.poisson(poissonParameters[indexes[i]],1)
    if ind:
	return wPrior,indexes
    else:
	return wPrior
    
def g2(x,w,day,i):
    return g(TimeHours,w,x,nSets,
                         data,cluster,bikeData,poissonParameters,nDays,
			 Avertices,poissonArray,exponentialTimes,day,i)
    
def estimationObjective(x,nProcesses,N=1000):
    """Estimate g(x)=E(f(x,w,z))
      
       Args:
          x
          N: number of samples used to estimate g(x)
    """
    estimator=N
    W,indexes=simulatorW(estimator,True)
    result=np.zeros(estimator)
    rseed=np.random.randint(1,4294967290,size=N)
    pool = mp.Pool(nProcesses)
    jobs = []
    for j in range(estimator):
        job = pool.apply_async(g2, args=(x,W[j,:],indexes[j],rseed[j],))
        jobs.append(job)
    pool.close()  # signal that no more data coming in
    pool.join()  # wait for all the tasks to complete
    
    for i in range(estimator):
        result[i]=jobs[i].get()
    
    return np.mean(result),float(np.var(result))/estimator


def estimationObjective2(x,N=100):
    """Estimate g(x)=E(f(x,w,z))
      
       Args:
          x
          N: number of samples used to estimate g(x)
    """
    estimator=N
    W,indexes=simulatorW(estimator,True)
    result=np.zeros(estimator)
    rseed=np.random.randint(1,4294967290,size=N)
    
    
    
    for i in range(estimator):
        result[i]=g2(x,W[i,:],indexes[i],rseed[i])
    
    return np.mean(result),float(np.var(result))/estimator

x=(numberBikes/float(n1))*np.ones((1,n1))

nTemp=int(sys.argv[1])
N=nTemp*10


print estimationObjective2(x[0,:],N)/estimationObjective(x[0,:],nTemp,N)