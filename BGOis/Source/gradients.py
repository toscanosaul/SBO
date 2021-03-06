
import numpy as np


#####SBO
def gradXBforAnSEK(x,n,B,kerns,X,n1,nT,W=None,n2=None):
    """Computes the gradient of B(x,i) for i in {1,...,n+nTraining}
       where nTraining is the number of training points
      
       Args:
          x: Argument of B
          n: Current iteration of the algorithm
          B: Vector {B(x,i)} for i in {1,...,n}
          kern: kernel
          X: Past observations X[i,:] for i in {1,..,n+nTraining}
          n1: Dimension of x
          nT: Number of training points
    """
    gradXB=np.zeros((n1,n+nT))
    
  #  alpha1=0.5*((kern.alpha[0:n1])**2)/(kern.scaleAlpha[0:n1])**2
    for i in xrange(n+nT):
        w=int(W[i])
        kern=kerns[w]
        alpha1=0.5*((kern.alpha[0:n1])**2)/((kern.scaleAlpha[0:n1])**2)

        gradXB[:,i]=B[i]*(-2.0*alpha1*(x-X[i,:]))
    return gradXB

def gradXBforAnMattern52(x,n,B,kerns,X,n1,nT,W,n2=None):
    """Computes the gradient of B(x,i) for i in {1,...,n+nTraining}
       where nTraining is the number of training points (only for
       multiple I.S.)
      
       Args:
          x: Argument of B
          n: Current iteration of the algorithm
          B: Vector {B(x,i)} for i in {1,...,n}
          kern: kernel
          X: Past observations X[i,:] for i in {1,..,n+nTraining}
          n1: Dimension of x
          nT: Number of training points
    """
    IS=len(kerns)
    gradXB=np.zeros((n1,n+nT))
    
   
    
    for i in xrange(n+nT):
        w=int(W[i])
        kern=kerns[w]
        alpha1=((kern.alpha[0:n1])**2)/(kern.scaleAlpha[0:n1])**2
        r=(np.sum(alpha1*((X[i,:]-x)**2)))
        a=(5.0/3.0)*(np.exp(-np.sqrt(5*r)))*(-1.0-np.sqrt(5*r))\
                 *(alpha1*(x-X[i,:]))
        gradXB[:,i]=kern.variance*a
    return gradXB/float(IS)


def gradXBSEK(new,kerns,BN,keep,points,n1,n2=None):
    """Computes the vector of gradients with respect to x_{n+1} of
        B(x_{p},n+1)=\int\Sigma_{0}(x_{p},w,x_{n+1},w_{n+1})dp(w),
        where x_{p} is a point in the discretization of the domain of x.
        
       Args:
          new: Point (x_{n+1},w_{n+1})
          kern: Kernel
          keep: Indexes of the points keeped of the discretization of the domain of x,
                after using AffineBreakPoints
          BN: Vector B(x_{p},n+1), where x_{p} is a point in the discretization of
              the domain of x.
          points: Discretization of the domain of x
          n1: Dimension of x
    """
    kern=kerns[int(new[0,n1])]
    alpha1=0.5*((kern.alpha[0:n1])**2)/(kern.scaleAlpha[0:n1])**2
    
    xNew=new[0,0:n1].reshape((1,n1))
    gradXBarray=np.zeros([len(keep),n1])
    M=len(keep)
    for i in xrange(n1):
        for j in xrange(M):
            gradXBarray[j,i]=-2.0*alpha1[i]*BN[keep[j],0]*(xNew[0,i]-points[keep[j],i])
    return gradXBarray


def gradXBMattern52(new,kerns,BN,keep,points,n1,n2):
    """Computes the vector of gradients with respect to x_{n+1} of
        B(x_{p},n+1)=\int\Sigma_{0}(x_{p},w,x_{n+1},w_{n+1})dp(w),
        where x_{p} is a point in the discretization of the domain of x.
        
       Args:
          new: Point (x_{n+1},w_{n+1})
          kern: Kernel
          keep: Indexes of the points keeped of the discretization of the domain of x,
                after using AffineBreakPoints
          BN: Vector B(x_{p},n+1), where x_{p} is a point in the discretization of
              the domain of x.
          points: Discretization of the domain of x
          n1: Dimension of x
    """
    kern=kerns[int(new[0,n1])]
    alpha1=((kern.alpha[0:n1])**2)/(kern.scaleAlpha[0:n1])**2

    xNew=new[0,0:n1].reshape((1,n1))
    wNew=new[0,n1:n1+n2].reshape((1,n2))
    gradXBarray=np.zeros([len(keep),n1])
    M=len(keep)
    
    IS=len(kerns)
    for i in range(M):
        temp=(np.sum(alpha1*((points[keep[i],:]-xNew)**2)))
        r=temp
        sum1=(5.0/3.0)*(np.exp(-np.sqrt(5*r)))*(-1.0-np.sqrt(5*r))\
                 *(alpha1*(xNew-points[keep[i],:]))
        gradXBarray[i,:]=sum1

    return kern.variance*gradXBarray/float(IS)

def gradXWSigmaOfuncSEK(n,new,kerns,Xtrain2,Wtrain2,n1,n2,nT,gamma):
    """Computes the vector of the gradients of Sigma_{0}(new,XW[i,:]) for
        all the past observations XW[i,]. Sigma_{0} is the covariance of
        the GP on F.
        
       Args:
          n: Number of iteration
          new: Point where Sigma_{0} is evaluated
          kern: Kernel
          Xtrain2: Past observations of X
          Wtrain2: Past observations of W
          N: Number of observations
          n1: Dimension of x
          n2: Dimension of w
          nT: Number of training points
    """
    gradXSigma0=np.zeros([n+nT,n1])
    tempN=n+nT
    
    
    past=np.concatenate((Xtrain2,Wtrain2),1)
    wNew=int(new[0,n1:n1+n2])
    kern=kerns[wNew]
    
   # gamma=np.transpose(kern.A(new,past))
    alpha1=0.5*((kern.alpha[0:n1])**2)/(kern.scaleAlpha[0:n1])**2
   # gradWSigma0=np.zeros([n+nT+1,n2])

  #  alpha2=0.5*((kern.alpha[n1:n1+n2])**2)/(kern.scaleAlpha[n1:n1+n2])**2
    xNew=new[0,0:n1]
    
    for i in xrange(n+nT):
        
        if wNew==int(Wtrain2[i,0]):
            gradXSigma0[i,:]=-2.0*gamma[i]*alpha1*(xNew-Xtrain2[i,:])
    return gradXSigma0


def gradXWSigmaOfuncMattern52(n,new,kerns,Xtrain2,Wtrain2,n1,n2,nT,gamma=None):
    """Computes the vector of the gradients of Sigma_{0}(new,XW[i,:]) for
        all the past observations XW[i,]. Sigma_{0} is the covariance of
        the GP on F.
        
       Args:
          n: Number of iteration
          new: Point where Sigma_{0} is evaluated
          kern: Kernel
          Xtrain2: Past observations of X
          Wtrain2: Past observations of W
          N: Number of observations
          n1: Dimension of x
          n2: Dimension of w
          nT: Number of training points
    """
    
    gradXSigma0=np.zeros([n+nT,n1])
    tempN=n+nT
    past=np.concatenate((Xtrain2,Wtrain2),1)
    wNew=int(new[0,n1:n1+n2])
    kern=kerns[wNew]
    #gamma=np.transpose(kern.A(new,past))
    alpha1=((kern.alpha[0:n1])**2)/(kern.scaleAlpha[0:n1])**2
  #  gradWSigma0=np.zeros([n+nT+1,n2])

  #  alpha2=((kern.alpha[n1:n1+n2])**2)/(kern.scaleAlpha[n1:n1+n2])**2
    xNew=new[0,0:n1]
    wNew=new[0,n1:n1+n2]

    for i in xrange(n+nT):
        if wNew==int(Wtrain2[i,0]):
            temp=(np.sum(alpha1*((xNew-Xtrain2[i,:])**2)))
            r=temp
            a=(5.0/3.0)*(np.exp(-np.sqrt(5*r)))*(-1.0-np.sqrt(5*r))
            gradXSigma0[i,:]=kern.variance*a*alpha1*(xNew-Xtrain2[i,:])
          #  gradWSigma0[i,:]=kern.variance*a*alpha2*(wNew-Wtrain2[i,:])
    return gradXSigma0

####KG
def gradXKernelSEK(x,n,kern,trainingPoints,X,n1):
    alpha=0.5*((kern.alpha)**2)/(kern.scaleAlpha)**2
    tempN=n+trainingPoints
    gradX=np.zeros((tempN,n1))
    for j in xrange(n1):
        for i in xrange(tempN):
            aux=kern.K(x,X[i,:].reshape((1,n1)))
            gradX[i,j]=aux*(-2.0*alpha[j]*(x[0,j]-X[i,j]))
    return gradX


def gradXKernel2SEK(x,Btemp,points,nD,mD,kern):
    alpha=0.5*((kern.alpha)**2)/(kern.scaleAlpha)**2
    temp=np.zeros((nD,mD))
    for i in xrange(nD):
        temp[i,:]=(-2.0*alpha[i])*(x[0,i]-points[:,i])
    return temp*Btemp[:,0]