#!/usr/bin/env python

from SquaredExponentialKernel import *
import unittest
import numpy as np
from math import *
import sys
import misc


class TestSEK(unittest.TestCase):
    def setUp(self):
        X = np.random.uniform(-3.,3.,(20,1))
        Y = np.sin(X) + np.random.randn(20,1)*0.05
        Y=Y[:,0]
        noise=(0.05)*0.05*np.ones(20)
        self.SEK=SEK(1,X=X,y=Y,noise=None)
        
        np.random.seed(10)
        X = np.random.uniform(-3.,3.,(50,2))
        Y = np.sin(X[:,0:1]) * np.sin(X[:,1:2])+np.random.randn(50,1)*0.05
        Y=Y[:,0]
        noise=(0.05)*0.05*np.ones(20)
        self.SEK2=SEK(2,X=X,y=Y,noise=None)
        
    
    def testTrain(self):
        
        #noise=np.random.randn(20,1)*0.05
        
      #  misc.kernOptWrapper(self.SEK)
        
       # print self.SEK.K(X)
        self.SEK.train(numStarts=10)
        print "solution0"
        print self.SEK.alpha, self.SEK.variance, self.SEK.mu
        
        
        self.SEK2.train(numStarts=10)
        print "solution"
        print self.SEK2.alpha, self.SEK2.variance, self.SEK2.mu
        
        
if __name__ == '__main__':
   # from sys import argv
    #seed=int(argv[1])
   # print argv[1]
   # x=argv[1]
    i=int(sys.argv[1])
    del sys.argv[1]
    np.random.seed(i)
    unittest.main()