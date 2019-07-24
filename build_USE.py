#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 24 14:19:57 2018

@author: zznj4199
"""
from utils import *
import argparse

def get_args():
    parser = argparse.ArgumentParser()
    home = os.path.expanduser("~")
    data_dir = os.path.join("data")
    parser.add_argument('-d', "--dataset")
    return parser.parse_args()


def get_args():
    parser = argparse.ArgumentParser()
    home = os.path.expanduser("~")
    data_dir = os.path.join("data")
    parser.add_argument('-d', "--dataset")
    parser.add_argument('-n', "--compute_ner",default=False, action = 'store_true')
    return parser.parse_args()

def compute_USE(dataset): 
    f = open('data/formatted/'+ dataset + '/all_qa.txt','r')
    embedding = []
    sentences = []
    ids = []
    for line in f:
        line = line.strip().split('\t')
        sentences.append(' '.join(line[1:]))
        ids.append(line[0])
      
    '''
    Computes the embedding with the transformer architecture (useTF = True)
    '''    
    embedding = embed_sentences(sentences, useTF = True)
    o = open('data/'+ dataset + '/USE_embedding.txt','w')
    o.write(str(len(ids)) + ' 512\n')
    for i in range(len(ids)):
        o.write(ids[i] + ' ' + " ".join(str(elem) for elem in np.ravel(embedding[i])) + '\n')
    o.close()

#%%
def main():
    args = get_args()
    compute_USE(args.dataset)

if __name__ == '__main__':
    main()
    
