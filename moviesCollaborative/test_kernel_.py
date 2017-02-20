#!/usr/bin/env python

#!/usr/bin/env python

import sys
sys.path.append("..")
import numpy as np
from math import *
from matplotlib import pyplot as plt
import scipy.stats as stats
from scipy.stats import norm,poisson
import statsmodels.api as sm
import multiprocessing as mp
import os
from scipy.stats import poisson
import json
from BGO.Source import *
import time
from pmf import cross_validation,PMF
from scipy.spatial.distance import cdist


SQRT_3 = np.sqrt(3.0)
SQRT_5 = np.sqrt(5.0)


import numpy as np
from scipy import linalg
from scipy.optimize import fmin_l_bfgs_b
from scipy.stats import multivariate_normal
import multiprocessing as mp
from scipy import array, linalg, dot
from math import *

def tripleProduct(v,L,w):
    temp=linalg.solve_triangular(L,w,lower=True)
    alp=linalg.solve_triangular(L.T,v.T,lower=False)
    res=np.dot(alp,temp)
    return res

##Computes:(L.L^T)^-1).v
def inverseComp(L,v):
    temp=linalg.solve_triangular(L,v,lower=True)
    alp=linalg.solve_triangular(L.T,temp,lower=False)
    return alp

lowerX=[0.01,0.1,1,1]
upperX=[1.01,2.1,21,201]



nGrid=[6,6,11,6]
n1=4

domainX=[]
for i in range(n1):
    domainX.append(np.linspace(lowerX[i],upperX[i],nGrid[i]))
    
domain=[[a,b,c,d] for a in domainX[0] for b in domainX[1] for c in domainX[2] for d in domainX[3]]


randomSeed=123
#np.random.seed(randomSeed)

"""
We define the objective object.
"""
num_user=943
num_item=1682

train=[]
validate=[]

data_all=[]

for i in range(1,6):
    data=np.loadtxt("ml-100k/u%d.base"%i)
    test=np.loadtxt("ml-100k/u%d.test"%i)
    train.append(data)
    validate.append(test)
    data_all.append(np.concatenate((data,test),axis=0))
    
XWtrain = np.loadtxt("XWdata.txt")
yTrain = np.loadtxt("ydata.txt").reshape((XWtrain.shape[0],1))
XWtrain_2 = np.loadtxt("XWdata_2.txt")
yTrain2 = np.loadtxt("ydata_2.txt").reshape((XWtrain_2.shape[0],1))
XWtrain = np.concatenate((XWtrain,XWtrain_2))
yTrain = np.concatenate((yTrain,yTrain2))

dim = 4
scaleAlpha = np.zeros(dim)
std = np.zeros(dim)
for index in range(dim):
    differences=np.zeros((XWtrain.shape[0],XWtrain.shape[0]))
    for i in range(XWtrain.shape[0]):
        for j in range(XWtrain.shape[0]):
            differences[i,j] = np.abs(XWtrain[i,index]-XWtrain[j,index])
    scaleAlpha[index] = np.mean(differences)
    std[index] = np.std(differences)


alpha=np.zeros(dim)

for i in range(dim):
    alpha[i]=np.random.normal(1.0/scaleAlpha[i],np.abs((1.0/(scaleAlpha[i]+std[i])-(1.0/scaleAlpha[i]))),1)
variance=np.random.rand(1,15)
variance_squared = np.random.rand(1,1)[0]

def Kfunction_2( X, X2=None,alpha=None,var_obs=None,covM=None,distances=False, matrix=True, n1=n1,nFolds=5,scaleAlpha=1.0):
    """
    Computes the covariance matrix cov(X[i,:],X2[j,:]).

    Args:
        X: Matrix where each row is a point.
        X2: Matrix where each row is a point.
        alpha: It's the scaled alpha.
        Variance: Sigma hyperparameter.
        cholesky: Decomposition of the matrix of the IS. Lower triangular
                  matrix. 

    """
   # covM=np.dot(cholesky,cholesky.transpose())



    if matrix is False:
        count =0
        L =np.zeros((nFolds,nFolds))
        for i in range(nFolds):
    
            for j in range(i+1):
                L[i,j] = covM[count+j]
            count += i+1
            
        covM = np.dot(L, np.transpose(L))
            

    z = X[:,n1]
    if X2 is None:
        X = X[:,0:n1]
        X=X*(alpha)/scaleAlpha
        X2=X
        z2=z
    else:
        X = X[:,0:n1]
        X=X*(alpha)/scaleAlpha
        z2=X2[:,n1]
        
        X2=X2[:,0:n1]
        X2=X2*(alpha)/scaleAlpha

    r2=np.abs(cdist(X,X2,'sqeuclidean'))
    r=np.sqrt(r2)
    cov=(1.0 + SQRT_5*r + (5.0/3.0)*r2) * np.exp (-SQRT_5* r)

    if distances:
        return distance*cov,r,r2

    nP=len(z)
    C=np.zeros((nP,nP))
    s,t=np.meshgrid(z,z2)
    s=s.astype(int)
    t=t.astype(int)

    T=covM[s,t].transpose()


    result = T * cov
    
    
    n, D = result.shape
    
    if np.array_equal(X,X2) and np.array_equal(z,z2):
        result += np.identity(n) * var_obs
    else:
        for i in range(n):
            for j in range(D):
                if np.array_equal(X[i,:],X2[j,:]) and z[i] == z2[j]:
                    result[i,j] += var_obs

  
    


    return result, cov

def A_function_2(X,X2=None,alpha=None,var_obs=None,covM=None,nFolds=5, n1=n1, noise=None):
    count =0
    L =np.zeros((nFolds,nFolds))
    for i in range(nFolds):

        for j in range(i+1):
            L[i,j] = covM[count+j]
        count += i+1

    covM = np.dot(L, np.transpose(L))
    
    
    if noise is None:
        K_,cov_K=Kfunction_2(X,X2,alpha=alpha,var_obs=var_obs,covM=covM)
    else:
        K_,cov_K=Kfunction_2(X,X2,alpha=alpha,variance=variance_squared,covM=covM,var_obs=var_obs)+np.diag(noise)
    return K_, L,cov_K
    
def logLikelihood_function_2(X,y,mu=0,var_obs=None,covM=None,nFolds=5, n1=n1, alpha=None, gradient=False):
    
    """
         
         We assume that alpha is in the log-space
         We assume that var_obs=noise in in the log-space. So, noise follows  a N(0,np.exp(var_obs)**2)
         We assume that the entries of covM are in the log-space (those are the entries of L, where L*L^T=covariance_matrix_folds).
        
    """

    K, L_cov_folds, cov_K=A_function_2(X,alpha=np.exp(alpha),var_obs=np.exp(2.0*var_obs),covM=np.exp(covM),nFolds=nFolds, n1=n1)
  

    
    N=X.shape[0]

    L=np.linalg.cholesky(K)
    Y=y-mu
    alp= linalg.solve_triangular(L,Y,lower=True)
    DET=np.sum(np.log(np.diag(L)))
    product_alpha=np.dot(alp.transpose(),alp)
    
    logLike=-0.5*product_alpha-DET
    
    if gradient==False:
        return logLike
    
         
    X3= X[:,0:n1]*np.exp(alpha)
    X2=X3
    cov_folds = np.dot(L_cov_folds, L_cov_folds.transpose())
    
    cov_folds_data = np.zeros((N,N))
    for i in range(N):
        for j in range(N):
            cov_folds_data[i,j] = cov_folds[X[i,n1],X[j,n1]]
            

            
    r2=np.abs(cdist(X3,X2,'sqeuclidean'))
    r=np.sqrt(r2)
    
    
    derivate_respect_to_r = ((1.0 + SQRT_5*r + (5.0/3.0)*r2) * np.exp (-SQRT_5* r) * (-SQRT_5)) + (np.exp (-SQRT_5* r) * (SQRT_5 + (10.0/3.0) * r))
    
    derivate_respect_to_r = cov_folds_data * derivate_respect_to_r
    


    gradient=np.zeros(n1+1+np.sum(range(nFolds+1))+1)
    
    alp_2 = linalg.solve_triangular(L.transpose(),alp,lower=False)
    
    product_alpha = np.dot(alp_2, alp_2.transpose())
    
    
    for i in range(n1):
        x_i = X[:,i:i+1]
        x_2_i = X[:,i:i+1]
        x_dist = cdist(x_i,x_2_i,'sqeuclidean')
        exp_alpha_i = np.exp(2.0 * alpha[i])
        derivative_K_respect_to_log_alpha_i = derivate_respect_to_r * (x_dist) * (exp_alpha_i) * (1.0/ r)
   
        
        for j in range(N):
            derivative_K_respect_to_log_alpha_i[j,j] = 0
            
     
    
        
        product_1 = np.dot(product_alpha,derivative_K_respect_to_log_alpha_i)
       
    
        tmp_1 = linalg.solve_triangular(L,derivative_K_respect_to_log_alpha_i,lower=True)
        tmp_2 = linalg.solve_triangular(L.transpose(),tmp_1,lower=False)

        gradient[i] = 0.5* np.trace(product_1 - tmp_2)
    
   
        

    
     
    derivative_K_respect_to_noise =  2.0 * np.exp(2.0* var_obs) * np.identity(N)
    product_1 = np.dot(product_alpha,derivative_K_respect_to_noise)
    tmp_1 = linalg.solve_triangular(L,derivative_K_respect_to_noise,lower=True)
    tmp_2 = linalg.solve_triangular(L.transpose(),tmp_1,lower=False)
    gradient[n1] = 0.5 * np.trace(product_1 - tmp_2)
    
    ###Gradient respect ot the entries of covM
    
    folds = X[:,n1]
    
    derivative_cov_folds = {}
    count=0
    for i in range(nFolds):
        for j in range(i+1):
            tmp_der = np.zeros((nFolds, nFolds))
            tmp_der[i,j] = L_cov_folds[i,j]
            tmp_der_mat = (np.dot(tmp_der, L_cov_folds.transpose()))
            tmp_der_mat += tmp_der_mat.transpose()
            derivative_cov_folds[count+j] = tmp_der_mat
        count += i+1
    
    for k in range(np.sum(range(nFolds+1))):
        der_covariance_folds = np.zeros((N,N))
        for i in range(N):
            for j in range(i+1):
                der_covariance_folds[i,j] = derivative_cov_folds[k][folds[i],folds[j]]
                der_covariance_folds[j,i] = der_covariance_folds[i,j]
        der_K_respect_to_l = der_covariance_folds * cov_K
        
        product_1 = np.dot(product_alpha,der_K_respect_to_l)
        tmp_1 = linalg.solve_triangular(L,der_K_respect_to_l,lower=True)
        tmp_2 = linalg.solve_triangular(L.transpose(),tmp_1,lower=False)
        gradient[n1+k+1] = 0.5 * np.trace(product_1 - tmp_2)
        
    
    tmp_1 = linalg.solve_triangular(L,np.ones((L.shape[0],1)),lower=True)
    tmp_2 = linalg.solve_triangular(L.transpose(),tmp_1,lower=False)
    gradient[-1] = np.dot(Y.transpose(), tmp_2)
    return logLike,gradient
    
    
     
    
    
def gradientLogLikelihood(X,y,var_obs=None,covM=None,alpha=None,nFolds=5, n1=n1):

    
    return  logLikelihood_function_2(X,y,var_obs=var_obs,covM=covM,nFolds=5, n1=n1, alpha=alpha, gradient=True)[1]

from scipy.optimize import fmin_l_bfgs_b

def minus_log_likelihood(params,X=XWtrain,y=yTrain, nFolds=5,n1=n1,gradient=True):
    alpha = params[0:n1]
    var_obs = params[n1]
    covM = params[n1+1:-1]
    mu=params[-1]
    return -1.0 *logLikelihood_function_2(X,y,mu=mu,var_obs=var_obs,covM=covM,nFolds=5, n1=n1, alpha=alpha)

def gradient_minus_log_likelihood(params,X=XWtrain,y=yTrain,nFolds=5, n1=n1):
    alpha = params[0:n1]
    var_obs = params[n1]
    covM = params[n1+1:-1]
    mu=params[-1]

    return -1.0 * logLikelihood_function_2(X,y,mu=mu,var_obs=var_obs,covM=covM,nFolds=nFolds, n1=n1, alpha=alpha, gradient=True)[1]

def optimizeKernel(minus_likelihood, X, y ,gradient,scaleAlpha=scaleAlpha,std=std,nFolds=5,start=None):
    """
    Optimize the minus log-likelihood using the optimizer method and starting in start.

    Args:
        start: starting point of the algorithm.
        optimizer: Name of the optimization algorithm that we want to use;
                   e.g. 'bfgs'.

    """
    
    if start is None:
        alpha=np.zeros(dim)
        for i in range(dim):
            alpha[i]=np.abs(np.random.normal(1.0/scaleAlpha[i],np.abs((1.0/(scaleAlpha[i]+std[i])-(1.0/scaleAlpha[i]))),1))
        log_alpha = np.log(alpha)
        var_obs=np.array([np.log(0.1)])
        
        variance=np.log(np.abs(np.random.rand(1,np.sum(range(nFolds+1)))))[0]
        
        mu = np.array([0.0])
        start=np.concatenate((log_alpha,var_obs,variance,mu))
    
    
    opt = fmin_l_bfgs_b(minus_likelihood, start, gradient,args=[X,y,])
    
    return opt

nFolds=5
def get_optimal_parameters(opt,n1=n1,nFolds=nFolds):
    alpha = np.exp(opt[0:n1])
    noise_std = np.exp(opt[n1])
    L_cov_folds = np.exp(opt[n1+1:])

    return alpha, noise_std, L_cov_folds

def predictions(alpha,noise,L_cov_folds,x,X,y,n1=4,nFolds=5):
    K, L_cov_folds_, cov_K=A_function_2(X,alpha=alpha,var_obs=noise**2,covM=L_cov_folds,nFolds=nFolds, n1=n1)
    k_chol = np.linalg.cholesky(K)

    tmp_1 =linalg.solve_triangular(k_chol,y,lower=True)
    tmp_2 = linalg.solve_triangular(k_chol.transpose(),tmp_1,lower=False)
    
    vec_cov = A_function_2(x,X2=X,alpha=alpha,var_obs=noise**2,covM=L_cov_folds,nFolds=nFolds, n1=n1)
    vec_cov = vec_cov[0]
    
    mu_n = np.dot(vec_cov,tmp_2)
    
    tmp_1_var =linalg.solve_triangular(k_chol,vec_cov.transpose(),lower=True)
    tmp_2_var = linalg.solve_triangular(k_chol.transpose(),tmp_1_var,lower=False)
    tmp_3 =np.dot(vec_cov,tmp_2_var)
    var_n= A_function_2(x,x,alpha=alpha,var_obs=noise**2,covM=L_cov_folds,nFolds=nFolds, n1=n1)[0]-tmp_3
    
    
    
    return mu_n, var_n
    

N=XWtrain.shape[0]

import multiprocessing as mp
numProcesses=78
jobs={}

numStarts=5
dim=4

starting_points=[]
for j in range(numStarts*N):
    alpha=np.zeros(dim)
    for i in range(dim):
        alpha[i]=np.abs(np.random.normal(1.0/scaleAlpha[i],np.abs((1.0/(scaleAlpha[i]+std[i])-(1.0/scaleAlpha[i]))),1))
    log_alpha = np.log(alpha)
    var_obs=np.array([np.log(np.random.rand()/10.0)])
    variance=np.random.normal(0,1,np.sum(range(nFolds+1)))
    mu=np.array([-100.0])
    start=np.concatenate((log_alpha,var_obs,variance,mu))
    starting_points.append(start)
    
training_data_sets ={}
test_points = {}
for i in range(N):
    selector = [x for x in range(N) if x != i]
    XWtrain_tmp = XWtrain[selector,:]
    yTrain_tmp = yTrain[selector,0:1]
    training_data_sets[i] = [XWtrain_tmp,yTrain_tmp]
    test_points[i] = XWtrain[i:i+1,:]
    



try:
    pool = mp.Pool(processes=numProcesses)
    for i in range(N):
        jobs[i] = []
        for j in range(numStarts):
            st=starting_points[j+i*numStarts]
            job = pool.apply_async(optimizeKernel, args=(minus_log_likelihood,training_data_sets[i][0],training_data_sets[i][1],gradient_minus_log_likelihood,0,0,5,starting_points[j+i*numStarts],))
            jobs[i].append(job)
            
        
    pool.close()
    pool.join()
except KeyboardInterrupt:
    print "Ctrl+c received, terminating and joining pool."
    pool.terminate()
    pool.join()

opt_values ={}
for i in range(N):
    opt_values[i]=[]
    for j in range(numStarts):
        try:
            opt_values[i].append(jobs[i][j].get())
        except Exception as e:
            print "opt failed"

solutions ={}
for i in range(N):
    if len(opt_values[i]):
        j = np.argmin([o[1][0][0] for o in opt_values[i]])
        temp = opt_values[i][j]
        solutions[i] = temp
        
import pickle
with open('optimal_solutions_mu', 'wb') as handle:
    pickle.dump(solutions, handle, protocol=pickle.HIGHEST_PROTOCOL)



    
    