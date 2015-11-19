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
    f=open("pointsPoisson1000.txt","w")
    np.savetxt(f,s)
    f.close()
