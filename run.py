#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Main file to run a single experiment

"""


import os
import argparse
import numpy as np
import logging
import gc
import warnings
import tensorflow as tf
import keras.backend as K
from utils import *
from methods import *
from eval import *

def get_args():
    parser = argparse.ArgumentParser()
    data_dir = os.path.join("data","preprocessed")
    out_dir = os.path.join("out")
    parser.add_argument('-p', "--data_path", default=data_dir, help =' Path of the data to use, i.e., preprocessed or just formatted')
    parser.add_argument('-d', "--dataset", help = 'Dataset to run the experiment on, e.g., wikiqa')
    parser.add_argument('-o', "--out_dir", default=out_dir)
    parser.add_argument('-m', "--method", default='pointwise', help = 'Training method, pointwise or pairwise')
    parser.add_argument('-a','--arch',default='concat', help = 'Architecture: concat or siamese')
    parser.add_argument('-r','--representation',default='avg_emb', help = 'USE or avg_emb representation')
    parser.add_argument('-b', "--balanced", default=False, action='store_true', help = 'Balance the pos/neg samples ratio to 1 for pointwise training')
    parser.add_argument('-n', "--number_epochs", type=int, default = 50)
    parser.add_argument('--n_runs',type = int, default = 1, help = 'number of runs for the training')
    parser.add_argument("-v", "--verbose", default=False, action='store_true')
    parser.add_argument('-s', '--training_samples',type = int,  help = 'Number of questions used for training')
    parser.add_argument('-l','--load_weights', help = 'Load weights from a pre-trained model')
    parser.add_argument('-w','--save_weights', default = 'weights.hdf5', help = 'Save weights of the best model')
    parser.add_argument('-f',"--finetune", default=False, action='store_true')
    parser.add_argument('--run', type = int,  default = 0, help =  'Run ID')
    parser.add_argument('-e','--embedding', default='default', help = 'Load a specific embedding')
    parser.add_argument('--margin',type = float, default = 0.01, help = 'Margin for the pairwise triplet-loss')
#    parser.add_argument("--ner", default=False, action='store_true')
    return parser.parse_args()


'''
This method runs a given method and its arguments'
'''
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
    
  
def main():
    #Remove INFO and WARNING messages
    logger = logging.getLogger()
    logger.handlers = [] 
    handler = logging.StreamHandler()
    handler.setLevel(logging.CRITICAL)
    def warn(*args, **kwargs):
        pass
    warnings.warn = warn
    tf.logging.set_verbosity(tf.logging.WARN)
    
    # Configures TensorFlow to not grab all the GPU memory
    config = tf.ConfigProto(allow_soft_placement=True)
    config.gpu_options.allow_growth = True
    session = tf.Session(config=config)
    K.set_session(session)

    
    args = get_args()
    # Load the test dataset
    pairs, labels, ids = load_dataset(args, 'test')
    
    # Compute the scores with designated method
    map, mrr, acc = np.zeros(args.n_runs),np.zeros(args.n_runs), np.zeros(args.n_runs)
    
    for i in range(args.n_runs):
        # args.run is important because it tells which data subset to use
        args.run = i
        # Perform the run
        scores = run_method(args)
        # Set the filename where the scores are saved
        args.pretraining = args.load_weights is not None
        filename = os.path.join(args.out_dir,args.dataset,args.method + '_' +  str(args.arch).lower()+'_'+str(args.representation).lower()+'_'+str(args.embedding)+'_'+str(args.pretraining)+'_'+str(args.finetune)+'_'+str(args.training_samples)+'_'+str(args.run))
        # Write the scores to the file
        write_scores(ids, scores, labels, filename)
        # Comute the MAP, MRR, P1, and the individual average precision per question
        map[i], mrr[i], acc[i], ap = eval_short(filename+'.scores', ignore_noanswer = False)
        with open(filename + '.ap', 'w') as file_handler:
            for item in ap:
                file_handler.write("{}\n".format(item))
                
    print(args.dataset + ';' +args.method + ';' +  str(args.embedding).lower() + '; %5.2f ; %5.2f; %5.2f ; %5.2f; %5.2f ; %5.2f' %(np.mean(map*100), np.std(map)*100, np.mean(mrr), np.std(mrr), np.mean(acc),np.std(acc)))
    # Clean the memory
    gc.collect()
    

if __name__ == '__main__':
    main()
    

