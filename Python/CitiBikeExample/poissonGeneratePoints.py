import numpy as np

###m is the length of the vectors for the sum
###all the numbers are saved in result


if __name__ == "__main__":
    result=[]
    nBikes=600
    bound=100
    l=np.linspace(bound,nBikes,nBikes-bound+1)
    a,b,c= np.meshgrid(l,l,l)
    sum1=a+b+c
    ind=np.where(sum1<=nBikes-bound)
    a1=a[ind]
    a2=b[ind]
    a3=c[ind]
    n1=len(a1)
    print n1
    a1=a1.reshape((n1,1))
    a2=a2.reshape((n1,1))
    a3=a3.reshape((n1,1))
    a4=nBikes-a1-a2-a3
    t=np.concatenate((a1,a2,a3,a4),1)
    f=open("pointsPoisson.txt","w")
    np.savetxt(f,t)
    f.close()
