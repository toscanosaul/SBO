#!/usr/bin/env python

import numpy as np
from math import *
from geopy.distance import vincenty
import json
from numpy import linalg as LA
from scipy.stats import poisson

nBikes=6000
nStations=329
distancesBikeStations=np.loadtxt("distanceBikeStations.txt")

###Simulate N(T,(i,j)) where (i,j) is in A, and N(T,(i,j)) is the poisson process related
###to the station (i,j); and A is a subset of stations. Return the number of times generated.
###Precondition: N is N(T,A)=sum(i,j in A) N(T,(i,j)), lamb is a vector with the parameters of the
###             poisson processes N(T,(i,j)) (lamb[i] is related to the ith entry of A)
def PoissonProcess(T,lamb,A,N):
    n=len(A) ##cardinality of A
    prob=np.ones(n)
    lambSum=(np.sum(lamb))
    for i in xrange(n):
        prob[i]=(float(lamb[i])/(lambSum))
    X=np.random.multinomial(N,prob,size=1)[0]
    nArrivals=np.sum(X)
    TIME=[]
    for i in xrange(n):
        if (X[i]>0):
            unif=np.random.uniform(0,1,X[i])
            temp=np.sort(float(T)*unif)
            TIME.append([A[i],temp])
    return TIME,nArrivals ##Each element of the list corresponds to the times of the poisson process of each pair (i,j)

##Generates subsets A(i,j) based on the flow between the bike station, with their parameters lamb
##n sets generated
##fil is the name of the file with the parameters of the poisson processes
def generateSets(n,fil):
    poiss=np.loadtxt(fil)
    n1=poiss.shape[0]
    A=[[] for i in range(n)]
    lamb=[[] for i in range(n)]
    for i in xrange(n1):
        for j in range(i,n1):
            ind=((i-1)*n1+j)%n
            if (poiss[i,j]>0):
                A[ind].append((i,j))
                lamb[ind].append(poiss[i,j])
            if (i!=j and poiss[j,i]>0):
                A[ind].append((j,i))
                lamb[ind].append(poiss[j,i])
    return A,lamb

##Simulate N(T,A)=sum(i,j in A) N(T,(i,j))
##Precondition: lambd are the respective parameters of the elements in A
def SimulateNt (A,lamb,T):
    la=np.sum(lamb)
    la=la*T
    res=np.random.poisson(la)
    return res

###Starts the Initial configuration of the citibike problem, where X is a vector with the initial configuration of bikes,
###and m are the number of groups according to K-means algorithm.
###returns a matrix with the number of docks and bikes available
def startInitialConfiguration (X,m):
    with open ('json.json') as data_file:
        data=json.load(data_file)
    bikeData=np.loadtxt("bikesStationsOrdinalIDnumberDocks.txt",skiprows=1)
    f = open(str(m)+"-cluster.txt", 'r')
    cluster=eval(f.read())
    f.close()
    A=np.zeros((nStations,2))
    f= open(str(m)+"-initialConfiguration.txt", 'w')
    f.write("docks"+","+"bickes"+","+"ID"+","+"total"+","+"bikes/total"+","+"latitude"+","+"longitude"+","+"streets")
    for i in range(m):
        temp=cluster[i]
        setBikes=X[i]/len(temp)
        res=X[i]%len(temp)
        for j in range(len(temp)):
            x=temp[j]
            ind=x[0]
            lat=str(data["stationBeanList"][ind]["latitude"])
            longt=str(data["stationBeanList"][ind]["longitude"])
            street=data["stationBeanList"][ind]["stAddress1"]
            ID=bikeData[ind,1]
            nBikes=setBikes
            if j<res:
                nBikes+=1
            docks=bikeData[ind,2]-nBikes
            total=nBikes+docks
            A[ind,0]=docks
            A[ind,1]=nBikes
            f.write("\n")
            f.write(str(docks)+","+str(nBikes)+","+str(ID)+","+str(total)+","+str(float(nBikes)/(total))+","+lat+","+longt+","+street)
    f.close()
    return A

##Find the closest bike station to currentBikeStation with available docks
##state is the situation of the bike stations
def findBikeStation(state,currentBikeStation):
    dist=distancesBikeStations[currentBikeStation,:]
    sort=[i[0] for i in sorted(enumerate(dist), key=lambda x:x[1])]
    k=1
    while True:
        ind=sort[k]
        if state[ind,0]>0:
            return ind
        else:
            k+=1
    return 0

##Counts the number of people who doesn't find a bike or a dock
##Precondition: N is th evector N(T,A_{i});
##X is the initial configuration of the bikes: we divide the bike stations
##in m groups according to K-means algorithm. The bikes are distributed uniformly
##in each group.
##lamb is a list with vectors of the parameters of the poisson processes N(T,(i,j))
##A is a list with all the sets considered
##date is respect to the data used , and it's a string: yyyy-mm
def unhappyPeople (T,N,X,m,lamb,A,date):
    unHappy=0
    state=startInitialConfiguration(X,m)
    exponentialTimes=np.loadtxt(date+"ExponentialTimes.txt")
    nSets=len(lamb)
    times=[]
    nTimes=0
    for i in range(nSets):
        temp=PoissonProcess(T,lamb[i],A[i],N[i])
        nTimes+=temp[1]
        times.extend(temp[0])
    Times=np.zeros((nTimes,3))
    k=0
    for i in range(len(times)):
        for j in range(len(times[i][1])):
            Times[k,0]=times[i][1][j]
            Times[k,1]=times[i][0][0]
            Times[k,2]=times[i][0][1]
            k+=1
    Times=Times[Times[:,0].argsort()]
    currentTime=0
    dropTimes=[]
    for i in xrange(nTimes):
        currentTime=Times[i,0]
        while (dropTimes and currentTime>dropTimes[0][0]):
            if state[dropTimes[0][1],0]>0:
                state[dropTimes[0][1],0]=state[dropTimes[0][1],0]-1
                state[dropTimes[0][1],1]+=1
                dropTimes.pop(0)
            else:
                unHappy+=1
                j=findBikeStation(state,dropTimes[0][1])
                state[j,0]=state[j,0]-1
                state[j,1]=state[j,1]+1
                dropTimes.pop(0)
        bikePickUp=Times[i,1]
        bikeDrop=Times[i,2]
        timeUsed=np.random.exponential(exponentialTimes[bikePickUp,bikeDrop])
        dropTimes.append((currentTime+timeUsed,bikeDrop))
        dropTimes=sorted(dropTimes, key=lambda x:x[0])
        if state[bikePickUp,1]==0:
            unHappy+=1
            continue
        state[bikePickUp,1]=state[bikePickUp,1]-1
        state[bikePickUp,0]=state[bikePickUp,0]+1
    return -unHappy
    


if __name__ == '__main__':
    np.random.seed(3)
    nSets=4
    nStations=329
    m=4
    date="2014-05"
    fil="2014-05PoissonParameters.txt"
    T=4.0 ##the rate of the PP is per hour, so this is from 7:00 to 11:00
    A,lamb=generateSets(nSets,fil)
    N=np.zeros(nSets)
    for i in range(nSets):
        N[i]=SimulateNt(A[i],lamb[i],T)
    print unhappyPeople (T,N,np.array([1500,1500,1500,1500]),m,lamb,A,date)
    #X=PoissonProcess(T,lamb[0],A[0],N[0])
