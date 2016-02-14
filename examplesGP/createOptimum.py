#!/usr/bin/env python

import numpy as np
from math import *
import os
import scipy.io
import os
from mpl_toolkits.mplot3d import axes3d
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter

font = {'family' : 'normal',
    'weight' : 'bold',
    'size'   : 15}

repetitions=2
end=1001


samplesIteration=[1]
#samplesIteration=[1]
numberIterations=50
numberPrior=30
A=[2,4,8,16]
#A=[2]
varianceb=[(2.0**k) for k in xrange(-4,10)]#+[2.0**r for r in range(1,5)]
#varianceb=[1.0/(2.0**k) for k in xrange(2)]

numberdifIteration=len(samplesIteration)
numberAs=len(A)
numberofvariance=len(varianceb)
numberSamples=100

directory="results"

differences=np.zeros((numberofvariance,numberdifIteration,numberAs,numberIterations+1))
varDifferences=np.zeros((numberofvariance,numberdifIteration,numberAs,numberIterations+1))
differences2=np.zeros((numberofvariance,numberdifIteration,numberAs,numberIterations+1))
varDifferences2=np.zeros((numberofvariance,numberdifIteration,numberAs,numberIterations+1))
differences3=np.zeros((numberofvariance,numberdifIteration,numberAs,numberIterations+1))
varDifferences3=np.zeros((numberofvariance,numberdifIteration,numberAs,numberIterations+1))
differences4=np.zeros((numberofvariance,numberdifIteration,numberAs,numberIterations+1))
varDifferences4=np.zeros((numberofvariance,numberdifIteration,numberAs,numberIterations+1))
differences5=np.zeros((numberofvariance,numberdifIteration,numberAs,numberIterations+1))
varDifferences5=np.zeros((numberofvariance,numberdifIteration,numberAs,numberIterations+1))
cont=np.zeros((numberofvariance,numberdifIteration,numberAs,1))
numberElements=np.zeros((numberofvariance,numberdifIteration,numberAs,numberIterations+1))
load=True



if load is True:
    for r in xrange(numberofvariance):
        betah=varianceb[r]
        for s in xrange(numberdifIteration):
            n=samplesIteration[s]
            for j in xrange(numberAs):
                Aparam=1.0/(A[j]*n)
                
		ngrid=50
  		valuesOutput=np.zeros(ngrid)
                for i in range(repetitions,end):
			try:
				output=np.loadtxt(os.path.join('%d'%i+"functions","function"+"betah"+'%f'%betah+"Aparam"+'%f'%Aparam+'%d'%n+".txt"))
				for k1 in xrange(ngrid):
  					results=np.zeros(ngrid)
  					for k2 in xrange(ngrid):
            					results[k2]=output[k1*ngrid+k2]
					valuesOutput[k1]=np.mean(results)
                        	f=open(os.path.join("%d"%i+"functions","rangeof"+"betah"+'%f'%betah+"Aparam"+'%f'%Aparam+'%d'%n+".txt"),'w')
				rang=np.zeros(2)
				rang[0]=np.min(valuesOutput)
				rang[1]=np.max(valuesOutput)
  				np.savetxt(f,rang)
  				f.close()
			except:
				print "file not found"
				

                z1=np.zeros([end-repetitions,numberIterations+2])
                z2=np.zeros([end-repetitions,numberIterations+2])
                z3=np.zeros([end-repetitions,numberIterations+2])
                z4=np.zeros([end-repetitions,numberIterations+2])
                z5=np.zeros([end-repetitions,numberIterations+2])
                y=np.zeros([end-repetitions,numberIterations+2])
		
                rangey=np.zeros([end-repetitions,2])
                for i in range(repetitions,end):
                    try:
                        temp=np.loadtxt(os.path.join("%d"%i+directory,
                                                     "function"+"betah"+'%f'%betah+"Aparam"+'%f'%Aparam+'%d'%n+"Results"+
                                                     '%d'%n+"AveragingSamples"+'%d'%+numberPrior+"TrainingPoints",
                                                     "KG","%d"%1+"run","%d"%1+"optimalValues.txt"))
			rangetemp=np.loadtxt(os.path.join("%d"%i+"functions","rangeof"+"betah"+'%f'%betah+"Aparam"+'%f'%Aparam+'%d'%n+".txt"))
			rangey[i-repetitions,:]=rangetemp
                        if len(temp)>=(numberIterations+1)*2 :
                            for l in range(numberIterations+1):
                                y[i-repetitions,l]=temp[2*l]
			
                            y[i-repetitions,numberIterations+1]=1
                    except:
                        continue
	            
    
    
                y2=np.zeros([end-repetitions,numberIterations+2])
                for i in range(repetitions,end):
                    try:
                        temp=np.loadtxt(os.path.join("%d"%i+directory,
                                                     "function"+"betah"+'%f'%betah+"Aparam"+'%f'%Aparam+'%d'%n+"Results"+
                                                     '%d'%n+"AveragingSamples"+'%d'%+numberPrior+"TrainingPoints",
                                                     "SBO","%d"%1+"run","%d"%1+"optimalValues.txt"))
                        
                        if len(temp)>=(numberIterations+1)*2 :
                            for l in range(numberIterations+1):
                                y2[i-repetitions,l]=temp[2*l]
                            y2[i-repetitions,numberIterations+1]=1
                 
                    except:
                        continue
                diff=np.zeros([0,numberIterations+1])
                diff2=np.zeros([0,numberIterations+1])
                diff3=np.zeros([0,numberIterations+1])
                diff4=np.zeros([0,numberIterations+1])
                diff5=np.zeros([0,numberIterations+1])
                for i in range(repetitions,end):
                    if (y2[i-repetitions,numberIterations+1]==1 and y[i-repetitions,numberIterations+1]==1):
                        temp=y2[i-repetitions:i-repetitions+1,0:numberIterations+1]-y[i-repetitions:i-repetitions+1,0:numberIterations+1]
			temp=temp/np.abs(y[i-repetitions:i-repetitions+1,0:numberIterations+1])
                        diff=np.concatenate((diff,temp),0)
                        diff2=np.concatenate((diff2,np.log(temp)),0)
			minf=rangey[i-repetitions,0]
			maxf=rangey[i-repetitions,1]
                        tempy2=y2[i-repetitions:i-repetitions+1,0:numberIterations+1]
                        tempy1=y[i-repetitions:i-repetitions+1,0:numberIterations+1]
			temp3=(maxf-tempy1)/(maxf-tempy2)
			temp4=np.log(temp3)
			temp5=(tempy2-tempy1)/(maxf-minf)
                        diff3=np.concatenate((diff3,temp3),0)
                        diff4=np.concatenate((diff4,temp4),0)
                        diff5=np.concatenate((diff5,temp5),0)
                        cont[r,s,j,0]+=1
		print diff.shape
                for i in xrange(numberIterations+1):
		    numberElements[r,s,j,i]=diff.shape[0]
                    differences[r,s,j,i]=np.mean(diff[:,i])
                    varDifferences[r,s,j,i]=np.var(diff[:,i])

                    differences2[r,s,j,i]=np.mean(diff2[:,i])
                    varDifferences2[r,s,j,i]=np.var(diff2[:,i])

                    differences3[r,s,j,i]=np.mean(diff3[:,i])
                    varDifferences3[r,s,j,i]=np.var(diff3[:,i])

                    differences4[r,s,j,i]=np.mean(diff4[:,i])
                    varDifferences4[r,s,j,i]=np.var(diff4[:,i])
                    differences5[r,s,j,i]=np.mean(diff5[:,i])
                    varDifferences5[r,s,j,i]=np.var(diff5[:,i])


write=load



if write is True:
    matfile = '100300manyfunctionsAvsbetabetaweighteddifferences.mat'
    scipy.io.savemat(matfile, mdict={'out': differences}, oned_as='row')
    
    matfile = '100300manyfunctionsnewAvsbetalargebetavarDiffweighted.mat'
    scipy.io.savemat(matfile, mdict={'out': varDifferences}, oned_as='row')
    
    matfile = '100300manyfunctionsAvsbetabetaweighteddifferences2.mat'
    scipy.io.savemat(matfile, mdict={'out': differences2}, oned_as='row')
    
    matfile = '100300manyfunctionsnewAvsbetalargebetavarDiffweighted2.mat'
    scipy.io.savemat(matfile, mdict={'out': varDifferences2}, oned_as='row')
    matfile = '100300manyfunctionsAvsbetabetaweighteddifferences3.mat'
    scipy.io.savemat(matfile, mdict={'out': differences3}, oned_as='row')
    
    matfile = '100300manyfunctionsnewAvsbetalargebetavarDiffweighted3.mat'
    scipy.io.savemat(matfile, mdict={'out': varDifferences3}, oned_as='row')
    matfile = '100300manyfunctionsAvsbetabetaweighteddifferences4.mat'
    scipy.io.savemat(matfile, mdict={'out': differences4}, oned_as='row')
    
    matfile = '100300manyfunctionsnewAvsbetalargebetavarDiffweighted4.mat'
    scipy.io.savemat(matfile, mdict={'out': varDifferences4}, oned_as='row')
    matfile = '100300manyfunctionsAvsbetabetaweighteddifferences5.mat'
    scipy.io.savemat(matfile, mdict={'out': differences5}, oned_as='row')
    
    matfile = '100300manyfunctionsnewAvsbetalargebetavarDiffweighted5.mat'
    scipy.io.savemat(matfile, mdict={'out': varDifferences5}, oned_as='row')
    matfile = '100200manyfunctionsnewAvsbetalargebetaNUMBERELEMENTSDiffweighted.mat'
    scipy.io.savemat(matfile, mdict={'out': numberElements}, oned_as='row')
