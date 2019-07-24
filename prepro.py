#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov  5 14:48:48 2018

@author: zznj4199
"""


from utils import *    

from sklearn.utils import class_weight
import os
import argparse
import numpy as np
from utils import *
from methods import *
from eval import *
import logging
import gc
import xmltodict
import sys
import re
import unicodedata
from nltk.corpus import stopwords
import string
import random
from tqdm import tqdm
import json
import pickle
import random
from gensim.summarization.bm25 import BM25, get_bm25_weights


def get_args():
    parser = argparse.ArgumentParser()
    home = os.path.expanduser("~")
    data_dir = os.path.join("data")
    parser.add_argument('-d', "--dataset")
    parser.add_argument('-n', "--compute_ner",default=False, action = 'store_true')
    return parser.parse_args()


def format(args):
    if args.dataset == 'wikiqa':
        path = 'data/orig/wikiqa/'
        opath = 'data/formatted/wikiqa/'
        for split in ('train','dev','test'):
            questions = []
            answers = []
            pairs = []
            f = open(path + 'WikiQA-' + split + '.tsv','r')
            q = open(opath + split + '_questions','w')
            a = open(opath + split + '_answers','w')
            p = open(opath + split + '_pairs','w')
            qid = ''
            for line in f:
                line = line.strip().split('\t')
                q_text = line[1]
                a_text = line[5]          
                questions.append((line[0], q_text))
                answers.append((line[4], a_text))
                pairs.append((line[0],line[4],line[6]))
            # remove duplicate questions
            del questions[0]
            questions = (list(set(questions)))
            del answers[0]
            del pairs[0]
        
            valid_qid=[]
            for i in range(len(pairs)):
                qid = pairs[i][0]
                label = pairs[i][2]
                if label=='1':
                    valid_qid.append(qid)
                
        
        
            for i in range(len(questions)):
                if questions[i][0] in valid_qid:
                    q.write('\t'.join(questions[i])+'\n')
            for i in range(len(pairs)):
                if (pairs[i][0] in valid_qid):
                    a.write('\t'.join(answers[i]) + '\n')
                    p.write('\t'.join(pairs[i])+ '\n')
        q.close()
        a.close()
        p.close()
        
        
        
    if args.dataset == 'semeval':
        path = 'data/orig/semeval/'
        opath = 'data/formatted/semeval/'
        for split in ('train-part1', 'train-part2','dev','test'):
            questions = []
            answers = []
            pairs = []
            filename = path + 'SemEval2016-Task3-CQA-QL-' + split + '-subtaskA-with-multiline.xml'
            fd =  open(filename)
            doc = xmltodict.parse(fd.read())
            data=doc['xml']['Thread']
            q = open(opath + split + '_questions','w')
            a = open(opath + split + '_answers','w')
            p = open(opath + split + '_pairs','w')
            
            
            questions = []
            answers = []
            pairs = []
            for i in range(len(data)):
                question = data[i]['RelQuestion']['RelQClean']
                qid = data[i]['RelQuestion']['@RELQ_ID']
                questions.append((qid,question))
                for j in range(len(data[i]['RelComment'])) : 
                    comment = data[i]['RelComment'][j]['RelCClean']
                    label = data[i]['RelComment'][j]['@RELC_RELEVANCE2RELQ']
                    aid = data[i]['RelComment'][j]['@RELC_ID']
                    if label == 'Good':
                        label = '1'
                    else:
                        label = '0'
                    
                    answers.append((aid,comment))
                    pairs.append((qid,aid,label))
                    
            for i in range(len(questions)):
                q.write('\t'.join(questions[i])+'\n')
            for i in range(len(pairs)):
                a.write('\t'.join(answers[i]) + '\n')
                p.write('\t'.join(pairs[i])+ '\n')
            q.close()
            a.close()
            p.close()
        os.system(' cat data/formatted/semeval/train-part1_questions data/formatted/semeval/train-part2_questions > data/formatted/semeval/train_questions')
        os.system(' cat data/formatted/semeval/train-part1_answers data/formatted/semeval/train-part2_questions > data/formatted/semeval/train_answers')
        os.system(' cat data/formatted/semeval/train-part1_pairs data/formatted/semeval/train-part2_pairs > data/formatted/semeval/train_pairs')
            
    if args.dataset == 'insuranceqa':
        def remove_stopwords(w_list):
            stop_words = set(stopwords.words('english'))
            puncts = set(list(string.punctuation))
            filtered_words = [word for word in w_list if word not in stop_words and word not in puncts]
        
            return filtered_words
        
        
        
        vocab={}
        vfile = open('data/orig/insuranceqa/vocabulary','r')
        for line in vfile:
            line = line.split('\t')
            vocab[line[0]] = line[1]
        
        
        # load answers
        afile = open('data/orig/insuranceqa/answers.label.token_idx','r')
        oafile = open('data/formatted/insuranceqa/answers','w')
        answers = {}
        all_text=[]
        for line in afile:
            line = line.split('\t')
            id = line[0]
            tmp = line[1].strip().split(' ')
            answer=[]
            for word in tmp:
                answer.append(vocab[word].strip())
            all_text.append(remove_stopwords(answer))
            answers[id] = ' '.join(answer)
            oafile.write(id + '\t' + ' '.join(answer) + '\n')
            
        
        
        bm25 = BM25(all_text)
        average_idf = sum(map(lambda k: float(bm25.idf[k]), bm25.idf.keys())) / len(bm25.idf.keys())
        
        

        # Load train
        tfile = open('data/orig/insuranceqa/question.train.token_idx.label','r')
        qfile = open('data/formatted/insuranceqa/train_questions','w')
        #afile = open('data/formatted/insuranceqa/train_answers','w')
        pfile = open('data/formatted/insuranceqa/train_pairs','w')
        pairs = []
        n = 10
        for i,line in tqdm(enumerate(tfile)):
            line = line.split('\t')
            qwords = line[0].strip().split(' ')
            question = []
            
            for word in qwords:
                question.append(vocab[word].strip())   
            question = ' '.join(question)
            qfile.write('q_train_' + str(i) + '\t' + question + '\n')   
            
            answer_ids = np.asarray(line[1].strip().split(' ')).astype(int)  
            for j,aid in enumerate(answer_ids):      
                pfile.write('q_train_' + str(i) + '\t' + str(aid) + '\t 1' + '\n')
        
            scores = np.asarray(bm25.get_scores(remove_stopwords(question.split()), average_idf))           
        
            #Pick n best candidate answers 
            idx = (-scores).argsort()[:n]
            k = 0
            while k < n:
                r = idx[k]+1             
                if r not in answer_ids:
                    pfile.write('q_train_' + str(i) + '\t' + str(r) + '\t 0' + '\n')
                k +=1
                    
        qfile.close()

        # Load dev, test
        
        for split in('test1', 'test2', 'dev'):   
            tfile = open('data/orig/insuranceqa/question.'+split+'.label.token_idx.pool','r')
                
            qfile = open('data/formatted/insuranceqa/'+split+'_questions','w')
        
            pfile = open('data/formatted/insuranceqa/'+split+'_pairs','w')
            pairs = []
            for i,line in enumerate(tfile):
                line = line.split('\t')
                qwords = line[1].strip().split(' ')
                question = []
                for word in qwords:
                    question.append(vocab[word].strip())   
                question = ' '.join(question)
                qfile.write('q_'+split+'_' + str(i) + '\t' + question + '\n')
                
                
                goodanswer_ids = np.asarray(line[0].strip().split(' ')).astype(int)
                answer_ids = np.asarray(line[2].strip().split(' ')).astype(int)
                
                if split == 'dev':
                    for aid in(goodanswer_ids):       
                        #+ '\t D' + str(i) + '\t - \t D' + str(i) + '-' +str(j) + '\t' +  answers[int(a)-1] + '\t 1\n' )
                        #afile.write(str(aid) + '\t' + answers[aid] +'\n' )
                        pfile.write('q_'+split+'_' + str(i) + '\t' + str(aid) + '\t 1' + '\n')
        
                scores = np.asarray(bm25.get_scores(remove_stopwords(question.split()), average_idf))           
                #Pick n best candidate answers 
                idx = (-scores[answer_ids-1]).argsort()[:n]
                for k in range(n):
                    r = answer_ids[idx[k]]
                    if r in goodanswer_ids:
                        pfile.write('q_'+split+'_' + str(i) + '\t' + str(r) + '\t 1' + '\n')
                    else:
                        pfile.write('q_'+split+'_' + str(i) + '\t' + str(r) + '\t 0' + '\n')
                        
            qfile.close()
            pfile.close()
        os.system('mv data/formatted/insuranceqa/test1_questions data/formatted/insuranceqa/test_questions')
        os.system('mv data/formatted/insuranceqa/test1_pairs data/formatted/insuranceqa/test_pairs')
        os.system('ln -sf answers data/formatted/insuranceqa/train_answers')
        os.system('ln -sf answers data/formatted/insuranceqa/dev_answers')
        os.system('ln -sf answers data/formatted/insuranceqa/test_answers')

#
#            
#    if args.dataset == 'trecqa_clean':
#        path = 'data/orig/trecqa_clean/'
#        opath = 'data/formatted/trecqa_clean/'
#        for split in ('train','dev','test'):
#            questions = []
#            answers = []
#            pairs = []
#            f = open(path + split + '-filtered.tsv','r')
#            q = open(opath + split + '_questions','w')
#            a = open(opath + split + '_answers','w')
#            p = open(opath + split + '_pairs','w')
#            qid = ''
#            for line in f:
#                line = line.strip().split('\t')
#                q_text = line[1]
#                a_text = line[5]          
#                questions.append(('q-'+line[0], q_text))
#                answers.append(('a-'+line[0]+'_'+line[4], a_text))
#                pairs.append(('q-'+line[0],'a-'+line[0]+'_'+line[4],line[6]))
#            # remove duplicate questions
#            del questions[0]
#            questions = (list(set(questions)))
#            del answers[0]
#            del pairs[0]
#        
#            valid_qid=[]
#            for i in range(len(pairs)):
#                qid = pairs[i][0]
#                label = pairs[i][2]
#                if label=='1':
#                    valid_qid.append(qid)
#                
#        
#        
#            for i in range(len(questions)):
#                if questions[i][0] in valid_qid:
#                    q.write('\t'.join(questions[i])+'\n')
#            for i in range(len(pairs)):
#                if (pairs[i][0] in valid_qid):
#                    a.write('\t'.join(answers[i]) + '\n')
#                    p.write('\t'.join(pairs[i])+ '\n')
#        q.close()
#        a.close()
#        p.close()            
            
    if args.dataset == 'trecqa':
        path = 'data/orig/trecqa/'
        opath = 'data/formatted/trecqa/'
        for split in ('train','dev','test'):
            questions = []
            answers = []
            pairs = []
            f = open(path + split + '-filtered.tsv','r')
            q = open(opath + split + '_questions','w')
            a = open(opath + split + '_answers','w')
            p = open(opath + split + '_pairs','w')
            qid = ''
            for line in f:
                line = line.strip().split('\t')
                q_text = line[1]
                a_text = line[5]          
                questions.append((line[0], q_text))
                answers.append((line[4], a_text))
                pairs.append((line[0],line[4],line[6]))
            # remove duplicate questions
            del questions[0]
            questions = (list(set(questions)))
            del answers[0]
            del pairs[0]
        
            valid_qid=[]
            for i in range(len(pairs)):
                qid = pairs[i][0]
                label = pairs[i][2]
                if label=='1':
                    valid_qid.append(qid)
                
        
        
            for i in range(len(questions)):
                if questions[i][0] in valid_qid:
                    q.write('\t'.join(questions[i])+'\n')
            for i in range(len(pairs)):
                if (pairs[i][0] in valid_qid):
                    a.write('\t'.join(answers[i]) + '\n')
                    p.write('\t'.join(pairs[i])+ '\n')
        q.close()
        a.close()
        p.close()
        
   
      
    if args.dataset == 'yahoo':
               
        path = 'data/orig/yahoo/'
        opath = 'data/formatted/yahoo/'
        for split in ('train','dev','test'):
            filename = path + 'cqa_questions_yadeep_min4_10k.cqa.'+split+'.xml'
            fd =  open(filename, encoding="latin-1")
            doc = xmltodict.parse(fd.read())
            data=doc['root']['question']
            
            q = open(opath + split + '_questions','w')
            a = open(opath + split + '_answers','w')
            p = open(opath + split + '_pairs','w')
        
            questions = []
            answers = []
            pairs = []
            for i in range(len(data)):
                subject = data[i]['text']
                qid = data[i]['docid']
                questions.append((qid,subject))
                for j in range(len(data[i]['answers']['answer'])):
                    comment = data[i]['answers']['answer'][j]['text']
                    aid = data[i]['answers']['answer'][j]['docid']
                    label = data[i]['answers']['answer'][j]['gold']
                    if label == 'true':
                        label='1'
                    else:
                        label='0'
                    answers.append((aid,comment))
                    pairs.append((qid,aid,label))
            for i in range(len(questions)):
                q.write('\t'.join(questions[i])+'\n')
            for i in range(len(pairs)):
                a.write('\t'.join(answers[i]) + '\n')
                p.write('\t'.join(pairs[i])+ '\n')
            q.close()
            a.close()
            p.close()
            
    if args.dataset == 'squad':
        
        ipath = 'data/orig/squad/'
        opath = 'data/formatted/squad/'
        answers = []
        questions = []
        pairs = []
        
        for split in ('train', 'dev'):
            answers = []
            questions = []
            pairs = []
            
            ifilen = ipath + split + '-v1.1.json'
            json_data=open(ifilen).read()
            data = json.loads(json_data)
            data = data['data']
            q = open(opath + split + '_questions','w')
            a = open(opath + split + '_answers','w')
            p = open(opath + split + '_pairs','w')
        
        
            questions_text = []
            for i, para in enumerate(tqdm(data)):
            
                for j, answer in enumerate(para['paragraphs']):
                    context = answer['context'].replace("\n","").replace("\r","")
                    aid = split + '-'+str(i)+'-'+str(j)
                    answers.append((aid,context))
                    for question in answer['qas']:
                        qid = question['id']
                        q_text = question['question']
                        questions_text.append(q_text)
                        if question['answer'] == False:
                            label = '0'
                        else:
                            label = '1'    
                        questions.append((qid,q_text))
                        pairs.append((qid,aid,label))
                        
            # Remove duplicate question 
            questions_final = []
            questions_text = set(questions_text)
            corr_ids={}
            for quest in tqdm(questions_text):
                found = 0
                for j in range(len(questions)):
                    if questions[j][1] == quest:
                        found +=1
                        if found == 1:
                            qid = questions[j][0] 
                            questions_final.append(questions[j])
                        
                        corr_ids[questions[j][0]]=qid
                                                   
            # Update QIDs
            for i in range(len(pairs)):
                pairs[i]=(corr_ids[pairs[i][0]],pairs[i][1], pairs[i][2])
            
                
            
            for i in range(len(questions_final)):
                q.write('\t'.join(questions_final[i])+'\n')
            for i in range(len(pairs)):
                p.write('\t'.join(pairs[i])+ '\n')
            for i in range(len(answers)):
                a.write('\t'.join(answers[i]) + '\n')
            q.close()
            a.close()
            p.close()


    # Create all_qa.txt file for USE and LASER embeddings
    os.system('cat data/formatted/'+args.dataset+'/*_[q,a]* > data/formatted/'+args.dataset+'/all_qa.txt')
        
def prepro(args):

    reCombining = re.compile(u'[\u0300-\u036f\u1dc0-\u1dff\u20d0-\u20ff\ufe20-\ufe2f\xb0\xa7\xa8]',re.U)


    def remove_diacritics(s):
        " Decomposes string, then removes combining characters "
        return reCombining.sub('',unicodedata.normalize('NFD',(s)) )

    def normtext(_inputstr_):
        _str_=_inputstr_.lower()
        _str_= re.sub('http://([^ "]*)','_url_',_str_)
        _str_= re.sub('https://([^ "]*)','_url_',_str_)
        _str_= re.sub('\[img([^ \]]*)\]','_img_',_str_)
        _str_= re.sub('@([^ "]*)','_atsomeone_',_str_)
        _str_= re.sub('\d+\.\d+\.\d+\.\d+','_ip_',_str_)
        _str_= re.sub('\;\)',' _smileypos_ ',_str_)
        _str_= re.sub('\:\)',' _smileypos_ ',_str_)
        _str_= re.sub('\;\(',' _smileyneg_ ',_str_)
        _str_= re.sub('\:\(',' _smileyneg_ ',_str_)
        _str_=re.sub('\_(\_)*','_',_str_)
        _str_=re.sub('\&\#39\;',' \'',_str_)
        # les . ou , dans les nombres sont remplaces par des _
        _str_= re.sub('(?P<nbbefore>\d+)\.(?P<nbafter>\d+)','\g<nbbefore>_\g<nbafter>',_str_)
        _str_= re.sub('(?P<nbbefore>\d+)\,(?P<nbafter>\d+)','\g<nbbefore>_\g<nbafter>',_str_)
        _str_=re.sub('\.(\.)*',' . ',_str_)
        _str_=re.sub('\?(\?)*',' ? ',_str_)
        _str_=re.sub('\!(\!)*',' ! ',_str_)
        _str_=re.sub('\,(\,)*',' , ',_str_)
        _str_=re.sub('\;(\;)*',' ; ',_str_)
        _str_=re.sub('\:(\:)*',' : ',_str_)
        _str_=re.sub('\=(\=)*',' = ',_str_)
        _str_=re.sub('\<(\<)*',' < ',_str_)
        _str_=re.sub('\>(\>)*',' > ',_str_)
        _str_=re.sub('\((\()*',' ( ',_str_)
        _str_=re.sub('\)(\))*',' ) ',_str_)
        _str_=re.sub('\"(\")*',' " ',_str_)
        _str_=re.sub('\â€œ(\â€œ)*',' " ',_str_)
        _str_=re.sub('\â€(\â€)*',' " ',_str_)
        _str_=re.sub('\-(\-)*','-',_str_)
        _str_=re.sub('\*(\*)*',' ',_str_)
        _str_=re.sub('\~(\~)*','~',_str_)
        _str_=re.sub('\'','\' ',_str_)

        return _str_
    
    def remove_stopwords(s):
        stop_words = set(stopwords.words('english'))
        puncts = set(list(string.punctuation))
        filtered_words = [word for word in s.split(' ') if word not in stop_words and word not in puncts]

        return filtered_words

     
            
            
    for split in ('train_', 'dev_', 'test_'):
    # Copy pairs file
        os.system("cp " + os.path.join('data','formatted',args.dataset,split+"pairs ") + 
                                       os.path.join('data','preprocessed',args.dataset,split+"pairs"))
        for dtype in ('questions', 'answers'):
            out = open(os.path.join('data','preprocessed',args.dataset, split + dtype),'w')
            lines = open(os.path.join('data','formatted',args.dataset,split+ dtype)).read().split("\n")
            rawtxt=[]
            for i in range(len(lines)-1):
                txt = lines[i].split('\t')
                rawtxt.append(txt[1])
                normtxt = remove_diacritics(normtext(txt[1]))
                remove_sw=True
                if remove_sw:
                    normtxt = ' '.join(remove_stopwords(normtxt))
                    
                out.write(txt[0] + ' ' + normtxt + '\n')
            out.close()
            if args.compute_ner:
                ner_vec = compute_ner(rawtxt)
                np.savetxt(os.path.join('data','preprocessed',args.dataset, split + dtype + '_ner'),np.asarray(ner_vec),fmt='%i')
    if args.dataset !=  'squad':   
        #%% subsample data
        '''
        Generate 10 subsets
        '''
        for i in range(10):
            print(i)
            # With increasing number of questions
            for n in np.logspace(2,9,8,  base=2, dtype = int):
                subsample_dataset(args,n,i)
                generate_triplets(args,n,i)
        # And generate the triplets for the full dataset
        generate_triplets(args,0,0)
    else:
        generate_triplets(args,0,0)
        

def main():
    #Remove INFO messages
    logger = logging.getLogger()
    logger.handlers = [] # This is the key thing for the question!
    handler = logging.StreamHandler()
    handler.setLevel(logging.CRITICAL)
    
    args = get_args()
    for directory in(os.path.join('data',args.dataset),os.path.join('data','formatted',args.dataset),os.path.join('data','preprocessed',args.dataset),os.path.join('data','out',args.dataset)):
        if not os.path.exists(directory):
            os.makedirs(directory)
    format(args)
    prepro(args)

    gc.collect()
if __name__ == '__main__':
    main()
    
