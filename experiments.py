#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 22 13:55:12 2018

@author: zznj4199
"""
from utils import *    
from keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.utils import class_weight
import os
import argparse
import numpy as np
from utils import *
from methods import *
from eval import *
import logging
import gc
import warnings
from keras import backend as be


def get_args():
    parser = argparse.ArgumentParser()
    home = os.path.expanduser("~")
    data_dir = os.path.join("data","preprocessed")
    out_dir = os.path.join("out")
    parser.add_argument('-p', "--data_path", default=data_dir)
    parser.add_argument('-d', "--dataset")
    parser.add_argument('-o', "--out_dir", default=out_dir)
    parser.add_argument('-m', "--method", default='pointwise')
    parser.add_argument('-a','--arch',default='concat')
    parser.add_argument('-r','--representation',default='avg_emb')
    parser.add_argument('-b', "--balanced", default=False, action='store_true')
    parser.add_argument('-n', "--number_epochs", type=int, default = 50)
    parser.add_argument('--n_runs',type = int, default = 1)
    parser.add_argument("-v", "--verbose", default=False, action='store_true')
    parser.add_argument('-s', '--training_samples',type = int)
    parser.add_argument('-l','--load_weights')
    parser.add_argument('-w','--save_weights', default = 'weights.hdf5')
    parser.add_argument('-f',"--finetune", default=False, action='store_true')
    parser.add_argument('--run', type = int,  default = 0)
    parser.add_argument('-e','--embedding', default='default')
    parser.add_argument('--margin',type = float, default = 0.01)
    parser.add_argument("--ner", default=False, action='store_true')
    return parser.parse_args()


def run_method(args):
    if args.method == 'wordcount':
        scores = wordcount(args)
    if args.method == 'bm25':
        scores = bm25(args)
    if args.method == 'avg_emb':
        scores = avg_embedding(args)
    if args.method == 'hungarian':
        scores = hungarian(args)
    if args.method == 'use':
        scores = USE(args)
    if args.method == 'bert':
        scores = BERT(args)
        
        
    if args.method == 'pointwise':
        scores = pointwise(args)
    if args.method == 'pairwise':
        scores = pairwise(args)
    return scores
    


#%%
def run(args):

    pairs, labels, ids = load_dataset(args, 'test')
    # Compute the scores with designated method
    map, mrr, acc = np.zeros(args.n_runs),np.zeros(args.n_runs), np.zeros(args.n_runs)
    # Write the results
    for i in range(args.n_runs):
        args.run = i
        scores = run_method(args)
        filename = os.path.join(args.out_dir,args.dataset,args.method + '_' +  str(args.arch)+'_'+str(args.representation)+'_'+str(args.embedding)+'_'+str(args.pretraining)+'_'+str(args.finetune)+'_'+str(args.training_samples)+'_'+str(args.run))
        write_scores(ids, scores, labels, filename)
        map[i], mrr[i], acc[i], ap = eval_short(filename+'.scores', ignore_noanswer = False)
        with open(args.resfile,'a') as f:
            f.write(';'.join((args.dataset, args.method, args.arch, str(args.representation), str(args.embedding),str(args.pretraining),str(args.finetune),str(args.training_samples), str(i), str(map[i]),str(mrr[i]),str(acc[i])))+'\n')
        f.close()
        
    x,y,z, ap = eval_short(filename+'.scores', ignore_noanswer = False)
    with open(filename + '.ap', 'w') as file_handler:
        for item in ap:
            file_handler.write("{}\n".format(item))
    f.close()
    
    
    # Memory leaks
    gc.collect()
    be.clear_session()
    return np.mean(map*100), np.std(map)*100, np.mean(mrr), np.std(mrr), np.mean(acc),np.std(acc)

#%% Remove Logging
logger = logging.getLogger()
logger.handlers = [] 
handler = logging.StreamHandler()
handler.setLevel(logging.CRITICAL)
def warn(*args, **kwargs):
    pass
warnings.warn = warn
tf.logging.set_verbosity(tf.logging.WARN)


# configures TensorFlow to not try to grab all the GPU memory
config = tf.ConfigProto(allow_soft_placement=True)
config.gpu_options.allow_growth = True
session = tf.Session(config=config)
K.set_session(session)


args = get_args()


args.verbose = True

#%% Set Margins
margin={}
margin['wikiqa']=0.01
margin['semeval']=0.02
margin['insuranceqa']=0.2
margin['trecqa']=0.1
margin['yahoo']=0.5
margin['squad']=0.05

#%% Unsupervised without pre-training
args.resfile='out/all_res.csv'

args.pretraining = False
args.finetune = False
for args.dataset in ('wikiqa','semeval','trecqa','yahoo'):
    for args.method in ('wordcount', 'bm25','avg_emb', 'hungarian', 'use'):
            if args.method in ('avg_emb', 'hungarian'):
                for args.embedding in ('default', args.dataset):
                    run(args)
            else:
                args.embedding = 'default'
                run(args)


#%% Pre-training
args.dataset = 'squad'
args.number_epochs = 100
args.n_runs = 1
args.margin = margin[args.dataset]
args.pretraining = False
args.training_samples = None
args.resfile='out/squad.csv'
#for args.method in ('pointwise', 'pairwise'):
for args.method in ('pointwise','pairwise'):
    for args.arch in (('cos','default')):
        for args.representation in(('use','avg_emb')):
            if args.representation == 'avg_emb':
                for args.embedding in ('wikiqa','semeval','trecqa','yahoo','default'):
                    args.save_weights = os.path.join('out',args.dataset,args.method+'_'+args.arch+'_'+args.representation+'_'+args.embedding+'.hdf5')
                    run(args)
            else:
                args.embedding='default'
                args.save_weights = os.path.join('out',args.dataset,args.method+'_'+args.arch+'_'+args.representation+'.hdf5')
                run(args)
    
#%% Running with pre-training, without fine-tuning
args.resfile='out/all_res.csv'
args.n_runs = 1
args.training_samples = None
args.finetune = False
args.specific_embedding = False
args.pretraining=True

for args.dataset in ('wikiqa','semeval','trecqa','yahoo'):
    args.margin = margin[args.dataset]
    for args.method in ('pointwise', 'pairwise'):
        for args.arch in (('cos','default')):
            for args.representation in(('use','avg_emb')):
                if args.representation == 'avg_emb':
                    for args.embedding in ('default', args.dataset):
                        args.load_weights = os.path.join('out','squad',args.method+'_'+args.arch+'_'+args.representation+'_'+args.embedding+'.hdf5')
                        args.save_weights = os.path.join(args.out_dir,args.dataset,args.method + '_' +  str(args.arch)+'_'+str(args.representation)+'_'+str(args.embedding)+'_'+str(args.pretraining)+'_'+str(args.finetune)+'_'+str(args.training_samples)+'.hdf5')
                        run(args)
                else:
                    args.embedding = 'default'
                    args.load_weights= os.path.join('out','squad',args.method+'_'+args.arch+'_'+args.representation+'.hdf5')
                    args.save_weights = os.path.join('out',args.dataset,args.method+'_'+args.arch+'_'+args.representation+'_'+str(args.finetune)+'.hdf5')
                    run(args)
            
#%% Running with pre_training, with fine-tuning
args.n_runs = 10
n_questions = [  4 ,  8,  16,  32 , 64 ,128,256 ,512, None]
args.finetune = True
for args.dataset in ('wikiqa','semeval','trecqa','yahoo'):
    args.margin = margin[args.dataset]
    for args.method in ('pointwise', 'pairwise'):
        for args.arch in (('cos','default')):
            for args.representation in(('use','avg_emb')):
                for args.training_samples in n_questions:
                    if args.representation == 'avg_emb':
                        for args.embedding in ('default', args.dataset):
                            args.load_weights = os.path.join('out','squad',args.method+'_'+args.arch+'_'+args.representation+'_'+args.embedding+'.hdf5')
                            args.save_weights = os.path.join(args.out_dir,args.dataset,args.method + '_' +  str(args.arch)+'_'+str(args.representation)+'_'+str(args.embedding)+'_'+str(args.pretraining)+'_'+str(args.finetune)+'_'+str(args.training_samples)+'.hdf5')
                            run(args)
                    else:
                        args.embedding = 'default'
                        args.load_weights= os.path.join('out','squad',args.method+'_'+args.arch+'_'+args.representation+'.hdf5')
                        args.save_weights = os.path.join('out',args.dataset,args.method+'_'+args.arch+'_'+args.representation+'_'+str(args.finetune)+'.hdf5')
                        run(args)


#%% Running without pre-training
args.pretraining=False
args.load_weights = None
for args.dataset in ('wikiqa','semeval','trecqa','yahoo'):
    args.margin = margin[args.dataset]
    for args.method in ('pointwise', 'pairwise'):
        for args.arch in (('cos','default')):
            for args.representation in(('use','avg_emb')):
                for args.training_samples in n_questions:
                    if args.representation == 'avg_emb':
                        for args.embedding in ('default', args.dataset):
                            args.save_weights = os.path.join(args.out_dir,args.dataset,args.method + '_' +  str(args.arch)+'_'+str(args.representation)+'_'+str(args.embedding)+'_'+str(args.pretraining)+'_'+str(args.finetune)+'_'+str(args.training_samples)+'.hdf5')
                            run(args)
                    else:
                        args.embedding = 'default'
                        args.save_weights = os.path.join('out',args.dataset,args.method+'_'+args.arch+'_'+args.representation+'_'+str(args.pretraining)+'_'+str(args.finetune)+'_'+str(args.training_samples)+'.hdf5')
                        run(args)


        
        