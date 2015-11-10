#!/usr/bin/env python

from LBFGS import *
import unittest
import numpy as np
from math import *
from scipy.optimize import fmin_l_bfgs_b

def f4(x):
    return (x[0])*(x[1])*(1-(x[0]**2)-(x[1]**2))

def Df4(x):
    return np.array([x[1]-3.0*x[1]*(x[0]**2)-x[1]**3,x[0]-3.0*x[0]*(x[1]**2)-x[0]**3])

def f3(x):
    return (x[0]**2)*(x[1]**2)

def Df3(x):
    return np.array([2.0*x[0]*(x[1]**2),2.0*x[1]*(x[0]**2)])


class TestBFGS(unittest.TestCase):
    def setUp(self):
        m=20000
        def f(x):
            return x**2
        def Df(x):
            return 2.0*x
        dimension=1
        self.opt=LBFGS(m,f,Df,dimension)
        
        def f2(x):
            return -2+x[0]**2+x[1]**2
        
        def Df2(x):
            return np.array([2.0*x[0],2.0*x[1]])
        

        
        def f4(x):
            return (x[0])*(x[1])*(1-(x[0]**2)-(x[1]**2))
        
        def Df4(x):
            return np.array([x[1]-3.0*x[1]*(x[0]**2)-x[1]**3,x[0]-3.0*x[0]*(x[1]**2)-x[0]**3])
        
        self.opt2=LBFGS(m,f2,Df2,2,1)
        
        self.opt3=LBFGS(m,f3,Df3,2,100000)
        
        self.opt4=LBFGS(m,f4,Df4,2,1000)
        
    
    def testBFGS(self):
        self.assertAlmostEqual(self.opt.BFGS(100),0)
        print self.opt.BFGS(100)
        print "ou"
        print self.opt2.BFGS(np.array([11,24]))
        
        print fmin_l_bfgs_b(f3,np.array([11,24]),Df3)
        print self.opt3.BFGS(np.array([12,43]))
        
        print fmin_l_bfgs_b(f4,np.array([0.5,1.3]),Df4)
        print self.opt4.BFGS(np.array([0.5,1.3]))
        
if __name__ == '__main__':
    unittest.main()