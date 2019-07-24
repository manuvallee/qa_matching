#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 10 16:52:58 2018

@author: zznj4199
"""
import sys
import scipy
import scipy.stats
file1 = sys.argv[1]
file2 = sys.argv[2]


f1 = open(file1+'.ap','r')
f2 = open(file2+'.ap','r')

d = []
d1=[]
d2=[]
for l1, l2 in zip(f1, f2):
    d1.append(l1)
    d2.append(l2)
    d.append(float(l1)-float(l2))

p = scipy.stats.wilcoxon(d,zero_method='pratt') 
print(p)
#%%
    
