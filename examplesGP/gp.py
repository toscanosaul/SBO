import numpy as np
import sys




nTemp3=1000
Aparam=float(sys.argv[1])
alphah=Aparam/float(nTemp3)

alphad=(1.0/float(nTemp3))-alphah
betah=0.5

def sigmah(xw,xw2):
    return alphah*np.exp(-betah*np.sqrt(np.sum((xw-xw2)**2)))
    
#xw is matrix where each row is a point where h is evaluated
def covarianceh(xw):
    cov=np.zeros((xw.shape[0],xw.shape[0]))
    for i in xrange(xw.shape[0]):
        for j in xrange(i,xw.shape[0]):
            cov[i,j]=sigmah(xw[i,:],xw[j,:])
            cov[j,i]=cov[i,j]
    return cov

def h2(xw,L=None):
    if L is None:
        cov=covarianceh(xw)
        L=np.linalg.cholesky(cov)

    
    Z=np.random.normal(0,1,xw.shape[0])

    return np.dot(L,Z)

ngrid=50
domainX=np.linspace(0,1,ngrid)
domainW=np.linspace(0,1,ngrid)
z=np.zeros((ngrid*ngrid,2))

for i in range(ngrid):
    for j in range(ngrid):
        z[i*ngrid+j,0]=domainX[i]
        z[i*ngrid+j,1]=domainW[j]
        
output=h2(z)

noisy=np.random.normal(0,np.sqrt(alphad),ngrid*ngrid*ngrid)

def getindex(x):
    dx=1.0/(ngrid-1)
    i=round(x/dx)
    return i

#k is already index
def evalf(x,w,k):
    i=getindex(x)
    j=getindex(w)
    h1=output[i*ngrid+j]
    return h1+noisy[i*ngrid*ngrid+j*ngrid+k]

def simulateZ(n):
    l=np.random.randint(0,ngrid,n)
    return l
    
n1=1
n2=1

def noisyF(XW,n):
    """Estimate F(x,w)=E(f(x,w,z)|w)
      
       Args:
          XW: Vector (x,w)
          n: Number of samples to estimate F
    """
    
    x=XW[0,0:n1]
    w=XW[0,n1:n1+n2]
    z=simulateZ(n)
    res=np.zeros(n)
    for i in xrange(n):
        res[i]=evalf(x,w,z[i])
    return np.mean(res),alphad/n





#print alphad
XW=np.array([[0.5,0.5]])
print noisyF(XW,5)
#print noisyF(XW)


lowerW=0
upperW=1

def simulatorW(n):
    """Simulate n vectors w
      
       Args:
          n: Number of vectors simulated
    """
    return np.random.randint(0,ngrid,n)


    
def evalf2(x,j,k):
    i=getindex(x)
    h1=output[i*ngrid+j]
    return h1+noisy[i*ngrid*ngrid+j*ngrid+k]


def estimationObjective(x,N=100):
    """Estimate g(x)=E(f(x,w,z))
      
       Args:
          x
          N: number of samples used to estimate g(x)
    """
    w=simulatorW(N)
    z=simulateZ(N)

    results=np.zeros(N)

    for i in xrange(N):
        results[i]=evalf2(x,w[i],z[i])


    return np.mean(results),(alphad+alphah)/N

print estimationObjective(0.8)



