import numpy as np

###m is the length of the vectors for the sum
###all the numbers are saved in result


if __name__ == "__main__":
    result=[]
    np.random.seed(1)
    nBikes=6000
    nPoints=1000
    s=nBikes*np.random.dirichlet(np.ones(4),nPoints)
    tpm=np.floor(s[:,0:3])
    col4=nBikes-np.sum(tpm,1)
    col4=col4.reshape((len(col4),1))
    s=np.concatenate((tpm,col4),1)
    temp=s
    n1=4
    f = open(str(4)+"-cluster.txt", 'r')
    cluster=eval(f.read())
    f.close()
    ####
    bikeData=np.loadtxt("bikesStationsOrdinalIDnumberDocks.txt",skiprows=1)
    upperX=np.zeros(n1)
    temBikes=bikeData[:,2]
    for i in xrange(n1):
        temp=cluster[i]
        indsTemp=np.array([a[0] for a in temp])
        upperX[i]=np.sum(temBikes[indsTemp])
    ####
    for j in range(nPoints):
        ind=np.where(temp[:,j]-upperX<0)[0]
        temp[ind,j]=upperX[ind]
        if len(ind)>0:
            res=np.sum(temp[ind,j]-upperX[ind])
            while (res>0):
                ind2=np.where(temp[:,j]-upperX>0)[0]
                
                partialBikes=res/len(ind2)
                resBikes=res%len(ind2)
                putBikes=0
                for k in ind2:
                    tmp=min(upperX[k],partialBikes+temp[k,j])
                    putBikes+=tmp-temp[k,j]
                    temp[k,j]=tmp
                if putBikes==0:
                    for k in ind2:
                        tmp=min(upperX[k],res+temp[k,j])
                        putBikes+=tmp-temp[k,j]
                        temp[k,j]=tmp
                        if putBikes==res:
                            break
                    break
                pendingBikes=partialBikes*len(ind2)-putBikes
                res=resBikes+pendingBikes
    f=open("NewpointsPoisson1000.txt","w")
    np.savetxt(f,temp)
    f.close()
