#!/usr/bin/env python

import numpy as np
from simulationPoissonProcessNonHomogeneous import *

n1=4
n2=1
numberSamplesForF=15

nDays=153
"""
We define the variables needed for the queuing simulation. 
"""

g=unhappyPeople  #Simulator

#fil="2014-05PoissonParameters.txt"
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





def noisyF(XW,n,ind=None):
    """Estimate F(x,w)=E(f(x,w,z)|w)
      
       Args:
          XW: Vector (x,w)
          n: Number of samples to estimate F
    """
    simulations=np.zeros(n)
    x=XW[0,0:n1]
    w=XW[0,n1:n1+n2]


    for i in xrange(n):
        simulations[i]=g(TimeHours,w,x,nSets,
                         data,cluster,bikeData,poissonParameters,nDays,
			 Avertices,poissonArray,exponentialTimes,randomSeed=ind)

    
    return np.mean(simulations),float(np.var(simulations))/n

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
aux1=(numberBikes/float(n1))*np.ones((1,n1))
#for i in range(nDays):
#    wPrior=np.zeros((1,1))
#    wPrior[0,0]=np.random.poisson(poissonParameters[i],1)
#    print wPrior,poissonParameters[i]



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

def computeProbability(w,parLambda,nDays):
    probs=poisson.pmf(w,mu=np.array(parLambda))
    probs*=(1.0/nDays)
    return np.sum(probs)

L=650
M=8950
wTemp=np.array(range(L,M))
probsTemp=np.zeros(M-L)
for i in range(M-L):
    probsTemp[i]=computeProbability(wTemp[i],poissonParameters,nDays)

probInf1=0
upLim=4000
for j in range(upLim):
    probInf1+=np.sum(poisson.pmf(j,mu=np.array(poissonParameters))/nDays)

print probInf1
probsTempInf1=probsTemp[0:(upLim-L)]/probInf1

probsTempInf1=probsTemp[(upLim-L):(M-L)]/(1-probInf1)

print np.sum(probsTempInf1)


