#!/usr/bin/env python

import numpy as np
import os

directory="Results"
repetitions=100

a=[[] for i in range(3)]
for i in range(1,repetitions+1):
    try:
        temp=np.loadtxt(os.path.join(directory,"%d"%i+"results.txt"))
        a[0].append(temp[0])
        a[1].append(temp[1])
        a[2].append(temp[2])
    except:
        continue

print "results"

for i in range(3):
    print np.mean(a[i])
    print np.std(a[i])
