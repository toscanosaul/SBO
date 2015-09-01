#########STATISTICAL TESTS#########################
import sys
sys.path.append("..")
import statsmodels.api as sm
import numpy as np
from matplotlib import pyplot as plt
from simulationPoissonProcess import *
import os
import multiprocessing as mp

#######Q-Q PLOTS
variances=[]

randomSeed=int(sys.argv[1])
np.random.seed(randomSeed)

g=unhappyPeople

n1=4
n2=4
TimeHours=4.0
trainingPoints=5
numberBikes=600
dimensionKernel=n1+n2
numberSamplesForF=10
fil="2014-05PoissonParameters.txt"
nSets=4
A,lamb=generateSets(nSets,fil)
pointsVOI=np.loadtxt("pointsPoisson.txt")
#####
parameterSetsPoisson=np.zeros(n2)
for j in xrange(n2):
    parameterSetsPoisson[j]=np.sum(lamb[j])
####


#####work this
def simulatorW(n):
    wPrior=np.zeros((n,n2))
    for i in range(n2):
        wPrior[:,i]=np.random.poisson(parameterSetsPoisson[i],n)
    return wPrior

####Prior Data
randomIndexes=np.random.random_integers(0,pointsVOI.shape[0]-1,trainingPoints)
Xtrain=pointsVOI[randomIndexes,:]
Wtrain=simulatorW(trainingPoints)
XWtrain=np.concatenate((Xtrain,Wtrain),1)

def noisyF(XW,n,seed):
    np.random.seed(seed)
    simulations=np.zeros(n)
    x=XW[0,0:n1]
    w=XW[0,n1:n1+n2]
    for i in xrange(n):
        simulations[i]=g(TimeHours,w,x,nSets,lamb,A,"2014-05")
    return np.mean(simulations),float(np.var(simulations))/n

path=os.path.join(“Results”,’%d’%randomSeed+"StatisticalAnalysisCitiBikeSimulation”)

if not os.path.exists(path):
    os.makedirs(path)

def QQ(k,XW,r):
    y=np.zeros(k)
    var2=np.zeros(k)
    jobs = []
    pool = mp.Pool()
    seeds=np.random.randint(1,10e8,k)
    for i in xrange(k):
        print i
        job = pool.apply_async(noisyF,(XW.reshape((1,n1+n2)),numberSamplesForF,seeds[i]))
        jobs.append(job)
        print i
    pool.close()  # signal that no more data coming in
    pool.join()  # wait for all the tasks to complete
    for j in range(k):
        temp=jobs[j].get()
        y[j]=temp[0]
        var2[j]=temp[1]
    mean=np.mean(y)
    var=np.var(y)
    y=(y-mean)/np.sqrt(var)
    fig=plt.figure()
    sm.qqplot(y, line='45')
    plt.savefig(os.path.join(path,'%d'%r+"QQ.pdf"))
    plt.close(fig)
    fig=plt.figure()
    n, bins, patches=plt.hist(y,10, normed=True, histtype='stepfilled')
    
    plt.savefig(os.path.join(path,'%d'%r+"Histogram.pdf"))
    plt.close(fig)
    save=np.zeros(2)
    variances.append(np.mean(var2))
    save[0]=np.mean(var2)
    save[1]=np.var(var2)/k
    f=open(os.path.join(path,'%d'%r+"variancesObservations.txt"),'w')
    np.savetxt(f,save)
    f.close()
    return var2


k=100
for i in xrange(trainingPoints):
    QQ(k,XWtrain[i,:],i)

#variances=np.array(variances)
#f=open("2ALLvariancesObservations.txt",'w')
#np.savetxt(f,variances)
#f.close()
#fig=plt.figure()
#n, bins, patches=plt.hist(variances,10, normed=True, histtype='stepfilled')
#plt.savefig("HistogramVariances.pdf")
#plt.close(fig)


    