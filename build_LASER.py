#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb  4 11:09:49 2019

@author: zznj4199
"""

import numpy as np
import sys
dataset = sys.argv[1]
dim = 1024
X = np.fromfile("data/"+dataset+"/LASER.raw", dtype=np.float32, count=-1)                                                                          
X.resize(X.shape[0] // dim, dim)
#%%

of = open('data/'+dataset+'/LASER_embedding.txt','w')
ids_f = open('data/formatted/'+dataset+'/all_ids.txt','r')
ids = ids_f.readlines()
of.write(str(len(X))+' '+str(1024)+'\n')
for i in range(len(X)):
    of.write(str(ids[i]).strip() + " " + " ".join(str(elem) for elem in np.ravel(X[i])) + '\n')
of.close()
