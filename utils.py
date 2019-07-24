#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 30 11:41:30 2018

@author: zznj4199
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 13 16:39:41 2018

@author: zznj4199
"""
import os
import numpy as np
from tqdm import tqdm

from nltk.tokenize import RegexpTokenizer
 
from gensim.models import KeyedVectors

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

import tensorflow as tf
import tensorflow_hub as hub

from stanfordcorenlp import StanfordCoreNLP
from nltk.corpus import stopwords


'''
Data loaders
'''
def load_dataset(args, split, triplets = False):
    base = os.path.join(args.data_path, args.dataset)
    questions = load_questions(base, split)
    answers = load_answers(base,split)
    if triplets==True:
        triplet, ids = load_triplets(base, split, questions, answers, args.training_samples, args.run)
        return triplet, ids
    else:
        pairs, labels, ids = load_pairs(base, split, questions, answers, args.training_samples, args.run)
        return pairs, labels, ids
def load_questions(base, split):
    qf = open(base + '/'+split+'_questions')
    questions={}
    for line in qf:
        line = line.split(' ')
        questions[line[0]]=' '.join(line[1:]).strip()
    return questions
    
def load_answers(base, split):
    af = open(base + '/'+split+'_answers') 
    answers={}
    for line in af:
        line = line.split(' ')
        answers[line[0]]=' '.join(line[1:]).strip()
    return answers
  
'''
Return the text QA pairs for a given dataset, as well as the labels and pairs ids
'''
def load_pairs(base, split, questions, answers, training_samples = None, run = 0):
    if training_samples is not '' and training_samples is not None and split =='train':
        pf = open(base + '/'+split+'_pairs_n'+str(training_samples)+'_'+str(run))
    else: 
        pf = open(base + '/'+split+'_pairs')
    pairs=[]
    ids = []
    labels = []
    for line in pf:
        line = line.split('\t')
        pairs.append([questions.get(line[0], "empty"),answers.get(line[1],"empty")])
        labels.append(int(line[2].strip()))
        ids.append([line[0], line[1]])
    return pairs, np.ravel(labels), ids

def load_triplets(base, split, questions, answers, training_samples = None, run = 0):
    if training_samples is not None and training_samples is not '' and split =='train':
        pf = open(base + '/'+split+'_triplets_n'+str(training_samples)+'_'+str(run))
    else: 
        pf = open(base + '/'+split+'_triplets')
    triplets=[]
    ids = []
    for line in pf:
        line = line.strip().split('\t')
        triplets.append([questions.get(line[0], "empty"),answers.get(line[1],"empty"), answers.get(line[2],'empty')])
        ids.append([line[0], line[1], line[2]])
    return triplets, ids




def pairs2text(pairs):
    q = []
    a= []
    for i in range(len(pairs)):
        q.append(pairs[i][0])
        a.append(pairs[i][1])
    return q,a

def triplets2text(triplets):
    a = []
    p = []
    n = []
    for i in range(len(triplets)):
        a.append(triplets[i][0])
        p.append(triplets[i][1])
        n.append(triplets[i][2])
    return a, p, n


'''
Compute the average word embedding representation of the Q/A pairs
'''
def avg_emb_sent(pairs, embedding, triplets = False):   
    if triplets:
        s_a_words=[]
        s_p_words=[]
        s_n_words=[]
        for i in range(0,len(pairs)):
            a_words = [embedding[word] for word in pairs[i][0].split() if word in embedding.vocab]
            p_words = [embedding[word] for word in pairs[i][1].split() if word in embedding.vocab]
            n_words = [embedding[word] for word in pairs[i][2].split() if word in embedding.vocab]
            if len(a_words)<2:
                a_words = np.zeros((1,300)) 
            if len(p_words)<2:
                p_words = np.zeros((1,300))
            if len(n_words)<2:
                n_words = np.zeros((1,300))               
            s_a_words.append(np.asarray(a_words).mean(axis=0)*np.ones(300))
            s_p_words.append(np.asarray(p_words).mean(axis=0)*np.ones(300))
            s_n_words.append(np.asarray(n_words).mean(axis=0)*np.ones(300))
        s_a_words=np.array(s_a_words)
        s_p_words=np.array(s_p_words)
        s_n_words=np.array(s_n_words)
        return   s_a_words , s_p_words, s_n_words
                
    else:      
        s_q_words=[]
        s_a_words=[]
        for i in range(0,len(pairs)):
            q_words = [embedding[word] for word in pairs[i][0].split() if word in embedding.vocab]
            a_words = [embedding[word] for word in pairs[i][1].split() if word in embedding.vocab]
            if len(a_words)<1:
                a_words = np.zeros((1,300))        
            if len(q_words)<1:
                q_words = np.zeros((1,300))
            s_q_words.append(np.asarray(q_words).mean(axis=0)*np.ones(300))
            s_a_words.append(np.asarray(a_words).mean(axis=0)*np.ones(300))
        s_q_words=np.array(s_q_words)
        s_a_words=np.array(s_a_words)
        return   s_q_words , s_a_words




'''
Perform the USE sentence embedding with a given model (transforer or DAN)
'''
def embed_sentences(sentences, useTF = True):
# Import the Universal Sentence Encoder's TF Hub module
    #@param ["https://tfhub.dev/google/universal-sentence-encoder/2", "https://tfhub.dev/google/universal-sentence-encoder-large/2"]
    with tf.Graph().as_default():
        if (useTF):
            module_url = 'https://tfhub.dev/google/universal-sentence-encoder-large/3'
        else:
            module_url = "https://tfhub.dev/google/universal-sentence-encoder/2"
        embed = hub.Module(module_url)
        messages = tf.placeholder(dtype=tf.string, shape=[None])
        output = embed(messages)
     
        with tf.Session() as session:
            session.run([tf.global_variables_initializer(), tf.tables_initializer()])
            embedding = []
            for i in tqdm(range(len(sentences))):
                embedding.append(session.run(output, feed_dict={messages: [sentences[i]]}))
    return embedding


'''
Write the scores
'''
def write_scores(ids, scores, labels, filename):
    file=open( filename + '.scores','w')
    for i in range(0,len(ids)):
        if labels[i]==1:
            line=ids[i][0] + ' '  + ids[i][1] + ' 0 ' + str((scores[i])) + ' true \n'
        else:
            line=ids[i][0] + ' ' + ids[i][1] + ' 0 ' + str((scores[i])) + ' false \n'
        file.write(line)
    file.close()
    
    
'''
Not implement
Writes the scores as well as the sentences 
'''    
def write_scores_debug(pairs, labels, scores, filename):
    file=open('data/'+ filename + '.scores','w')
    for i in range(0,len(pairs)):
        line=pairs[i][0] + '\t' + pairs[i][1] + '\t' + str((scores[i])) + '\t' + labels[i] +' \n'
        file.write(line)
    file.close()


'''
Write the scores of the triplets and returns the scores
'''
def write_scores_triplet(ntext,ids_triplet,ids_pairs,Xp_pos,Xp_neg):
    scores = []
    h_sim={}
    f = open(ntext,'w')
    for i in range(len(ids_triplet)):
        anchorid=ids_triplet[i][0]
        posid=ids_triplet[i][1]
        negid=ids_triplet[i][2]
        if (anchorid,posid) not in h_sim:
            h_sim[(anchorid,posid)]=Xp_pos[i]
            f.write(anchorid+' '+posid+ ' 0 ' + str(h_sim[(anchorid,posid)])+' true\n')
                    
        if (anchorid,negid) not in h_sim:
            h_sim[(anchorid,negid)]=Xp_neg[i]
            f.write(anchorid+' '+negid+' 0 '+str(h_sim[(anchorid,negid)])+' false\n')   
            
    
    for i in range(len(ids_pairs)):
        score = h_sim.get((ids_pairs[i][0],ids_pairs[i][1]), "0")
        scores.append(score)
        
    return np.asarray(scores)


'''
Return the USE representation for a list of ids
'''  
def embed_USE(ids, embedding,triplets = False):
    if triplets:
        a = np.zeros((len(ids),embedding.vector_size))
        p = np.zeros((len(ids),embedding.vector_size))
        n = np.zeros((len(ids),embedding.vector_size))
        for i in range(len(ids)):
            a[i,:]=embedding[ids[i][0]]
            p[i,:]=embedding[ids[i][1]]
            n[i,:]=embedding[ids[i][2]]
        return a, p, n
        
    else:
        q = np.zeros((len(ids),embedding.vector_size))
        a = np.zeros((len(ids),embedding.vector_size))
        for i in range(len(ids)):
            q[i,:]=embedding[ids[i][0]]
            a[i,:]=embedding[ids[i][1]]
        return q, a
        

'''
Returns the USE/LASER vector representation of the train, dev and test splits
'''
def prepare_data_supervised_USE(args):   
    pairs_train, labels_train, ids_train = load_dataset(args, 'train')
    pairs_dev, labels_dev, ids_dev = load_dataset(args, 'dev')
    pairs_test, labels_test, ids_test = load_dataset(args, 'test')    
    embedding = KeyedVectors.load_word2vec_format(os.path.join('data',args.dataset,args.representation.upper()+'_embedding.txt'), binary=False)
    train_q, train_a = embed_USE(ids_train, embedding)
    dev_q, dev_a = embed_USE(ids_dev, embedding)
    test_q, test_a = embed_USE(ids_test, embedding)
    return train_q, train_a, labels_train, dev_q, dev_a, labels_dev, test_q, test_a, labels_test




'''
Sample the dataset with n questions and 
identify the subset with i
'''
def subsample_dataset(args, n = 1024, k = 0):
    path = os.path.join('data','preprocessed',args.dataset)
    split = 'train'
    questions = load_questions(path, split)
    answers = load_answers(path,split)
    pairs, labels, ids = load_pairs(path, split, questions, answers)
    ids = np.asarray(ids)
    
    # Build unique Qid array
    q_ids = np.asarray(list(set(ids[:,0])))
    # Select n Qids
    q_ids = q_ids[np.random.choice(len(q_ids),n, replace = False)]

    # Retrieve QA pairs corresponding to QID
    out_pairs = np.empty((0,3))
    for i in range(len(q_ids)):
        qid = q_ids[i]    
        indx = np.where(ids == qid)[0]
        res = (np.concatenate((ids[indx],labels[indx,np.newaxis]),axis = 1))
        out_pairs = np.concatenate((out_pairs,res))
    
    # Write
    p = open(os.path.join(path,'train_pairs_n' + str(n)) + '_' + str(k),'w')  
    for i in range(len(out_pairs)):
        p.write('\t'.join(out_pairs[i,:])+'\n')
    p.close()
    
    
    
    
'''
Generate the triplets from the formatted files
n is the number of questions
k is the run_id (questions subset)
'''
def generate_triplets(args,n = 1024, k = 0):
    dataset = args.dataset
    split = 'train'
    h_anchorpos={}
    h_anchorneg={}
    if n>0:
        samples = '_n'+str(n)
        for line in open(os.path.join('data','preprocessed',dataset,split+'_pairs'+samples+'_'+str(k))):
            l_line=line.rstrip().split('\t')
            anchorid=l_line[0]
            senid=l_line[1]
            label=int(l_line[2])
            h_anchorpos.setdefault(anchorid,[])
            h_anchorneg.setdefault(anchorid,[])
            if label>0:
                h_anchorpos[anchorid].append(senid)
            else:
                h_anchorneg[anchorid].append(senid)
        f = open(os.path.join('data','preprocessed',dataset,split+'_triplets'+samples+'_'+str(k)),'w')
    else:
        for line in open(os.path.join('data','preprocessed',dataset,split+'_pairs')):
            l_line=line.rstrip().split('\t')
            anchorid=l_line[0]
            senid=l_line[1]
            label=int(l_line[2])
            h_anchorpos.setdefault(anchorid,[])
            h_anchorneg.setdefault(anchorid,[])
            if label>0:
                h_anchorpos[anchorid].append(senid)
            else:
                h_anchorneg[anchorid].append(senid)
        f = open(os.path.join('data','preprocessed',dataset,split+'_triplets'),'w')


    for anchor in h_anchorpos:
        for pos in h_anchorpos[anchor]:
            for neg in h_anchorneg[anchor]:
                f.write('\t'.join((anchor,pos,neg))+'\n')
    f.close()
                
    for split in ('dev','test'):
        h_anchorpos={}
        h_anchorneg={}
        
        print(split)
        for line in open(os.path.join('data','preprocessed',dataset,split+'_pairs')):
            l_line=line.rstrip().split('\t')
            anchorid=l_line[0]
            senid=l_line[1]
            label=int(l_line[2])
            h_anchorpos.setdefault(anchorid,[])
            h_anchorneg.setdefault(anchorid,[])
            if label>0:
                h_anchorpos[anchorid].append(senid)
            else:
                h_anchorneg[anchorid].append(senid)
        
        f = open(os.path.join('data','preprocessed',dataset,split+'_triplets'),'w')
        for anchor in h_anchorpos:
            for pos in h_anchorpos[anchor]:
                for neg in h_anchorneg[anchor]:
                    f.write('\t'.join((anchor,pos,neg))+'\n')
        f.close()
        
        
########################################################## Experiments ###########################################################


def prepare_data_supervised_BERT(args):   
    pairs_train, labels_train, ids_train = load_dataset(args, 'train')
    pairs_dev, labels_dev, ids_dev = load_dataset(args, 'dev')
    pairs_test, labels_test, ids_test = load_dataset(args, 'test')    
    embedding = KeyedVectors.load_word2vec_format(os.path.join('data',args.dataset,'BERT_embedding.txt'), binary=False)
    train_q, train_a = embed_USE(ids_train, embedding)
    dev_q, dev_a = embed_USE(ids_dev, embedding)
    test_q, test_a = embed_USE(ids_test, embedding)
    return train_q, train_a, labels_train, dev_q, dev_a, labels_dev, test_q, test_a, labels_test




'''
Return w2v sequences
'''
def prepare_data_supervised(args, q_seq_len = 10, a_seq_len = 50):   
    MAX_NB_WORDS = 50000
    
    pairs_train, labels_train, ids_train = load_dataset(args, 'train')
    pairs_dev, labels_dev, ids_dev = load_dataset(args, 'dev')
    pairs_test, labels_test, ids_test = load_dataset(args, 'test')


    questions_train, answers_train = pairs2text(pairs_train)
    questions_dev, answers_dev = pairs2text(pairs_dev)
    questions_test, answers_test = pairs2text(pairs_test)

    tokenizer = Tokenizer(num_words=MAX_NB_WORDS, lower=True, char_level=False)
    tokenizer.fit_on_texts(questions_train + answers_train + questions_dev + answers_dev + questions_test + answers_test)  #leaky
    word_seq_train_q = tokenizer.texts_to_sequences(questions_train)
    word_seq_train_a = tokenizer.texts_to_sequences(answers_train)
    word_seq_dev_q = tokenizer.texts_to_sequences(questions_dev)
    word_seq_dev_a = tokenizer.texts_to_sequences(answers_dev)
    word_seq_test_q = tokenizer.texts_to_sequences(questions_test)
    word_seq_test_a = tokenizer.texts_to_sequences(answers_test)
    word_index = tokenizer.word_index

    word_seq_train_q = pad_sequences(word_seq_train_q, maxlen=max(q_seq_len, a_seq_len))
    word_seq_train_a = pad_sequences(word_seq_train_a, maxlen=max(q_seq_len, a_seq_len))
    word_seq_dev_q = pad_sequences(word_seq_dev_q, maxlen=max(q_seq_len, a_seq_len))
    word_seq_dev_a = pad_sequences(word_seq_dev_a, maxlen=max(q_seq_len, a_seq_len))
    word_seq_test_q = pad_sequences(word_seq_test_q, maxlen=max(q_seq_len, a_seq_len))
    word_seq_test_a = pad_sequences(word_seq_test_a, maxlen=max(q_seq_len, a_seq_len))

    embedding = KeyedVectors.load_word2vec_format(os.path.join('data',args.embedding,'embedding.bin'), binary=True)

    nb_words = len(word_index)+1
    words_not_found = []
    embed_dim=300
    embedding_matrix = np.zeros((nb_words, embed_dim))
    for word, i in word_index.items():     
        if word in embedding.vocab:
            # words not found in embedding index will be all-zeros.
            embedding_matrix[i] = embedding[word]
        else:
            words_not_found.append(word)
            #print('word not found: ' + word)
    return word_seq_train_q, word_seq_train_a,labels_train, word_seq_dev_q, word_seq_dev_a,labels_dev, word_seq_test_q, word_seq_test_a, labels_test, embedding_matrix

def prepare_data_supervised_triplets(args, a_seq_len = 10, p_seq_len = 50, n_seq_len = 50):   
    MAX_NB_WORDS = 50000
    
    triplet_train, ids_train = load_dataset(args, 'train', triplets = True)
    triplet_dev, ids_dev = load_dataset(args, 'dev', triplets = True)
    triplet_test, ids_test = load_dataset(args, 'test', triplets = True)
    
    anchors_train, pos_train, neg_train = triplets2text(triplet_train)
    anchors_dev, pos_dev, neg_dev = triplets2text(triplet_dev)
    anchors_test, pos_test, neg_test = triplets2text(triplet_test)

    tokenizer = Tokenizer(num_words=MAX_NB_WORDS, lower=True, char_level=False)
    tokenizer.fit_on_texts(anchors_train + pos_train + neg_train + anchors_dev + pos_dev + neg_dev + anchors_test + pos_test + neg_test)  #leaky
    word_seq_train_a = tokenizer.texts_to_sequences(anchors_train)
    word_seq_train_p = tokenizer.texts_to_sequences(pos_train)
    word_seq_train_n = tokenizer.texts_to_sequences(neg_train)
    
    word_seq_dev_a = tokenizer.texts_to_sequences(anchors_dev)
    word_seq_dev_p = tokenizer.texts_to_sequences(pos_dev)
    word_seq_dev_n = tokenizer.texts_to_sequences(neg_dev)
    
    word_seq_test_a = tokenizer.texts_to_sequences(anchors_test)
    word_seq_test_p = tokenizer.texts_to_sequences(pos_test)
    word_seq_test_n = tokenizer.texts_to_sequences(neg_test)

    word_index = tokenizer.word_index
    
    
    word_seq_train_a = pad_sequences(word_seq_train_a, maxlen=max(a_seq_len, p_seq_len))
    word_seq_train_p = pad_sequences(word_seq_train_p, maxlen=max(a_seq_len, p_seq_len))
    word_seq_train_n = pad_sequences(word_seq_train_n, maxlen=max(a_seq_len, p_seq_len))
    
    word_seq_dev_a = pad_sequences(word_seq_dev_a, maxlen=max(p_seq_len, p_seq_len))
    word_seq_dev_p = pad_sequences(word_seq_dev_p, maxlen=max(p_seq_len, p_seq_len))
    word_seq_dev_n = pad_sequences(word_seq_dev_n, maxlen=max(p_seq_len, p_seq_len))
    
    word_seq_test_a = pad_sequences(word_seq_test_a, maxlen=max(p_seq_len, a_seq_len))
    word_seq_test_p = pad_sequences(word_seq_test_p, maxlen=max(p_seq_len, a_seq_len))
    word_seq_test_n = pad_sequences(word_seq_test_n, maxlen=max(p_seq_len, a_seq_len))

    if args.specific_embedding:
        embedding = KeyedVectors.load_word2vec_format(os.path.join('data',args.dataset,'embedding.bin'), binary=True)
    else:
        embedding = KeyedVectors.load_word2vec_format(os.path.join('data','GoogleNews-vectors-negative300.bin'), binary=True)
    
    nb_words = len(word_index)+1
    words_not_found = []
    embed_dim=300
    embedding_matrix = np.zeros((nb_words, embed_dim))
    for word, i in word_index.items():     
        if word in embedding.vocab:
            # words not found in embedding index will be all-zeros.
            embedding_matrix[i] = embedding[word]
        else:
            words_not_found.append(word)
    return word_seq_train_a, word_seq_train_p, word_seq_train_n, word_seq_dev_a, word_seq_dev_p, word_seq_dev_n, word_seq_test_a, word_seq_test_p, word_seq_test_n, embedding_matrix


def compute_ner(sentences):
    ner_vec=[]
    nlp = StanfordCoreNLP(r'/home/zznj4199/sources/stanford-corenlp-full-2018-02-27/')
    for i in tqdm(range(len(sentences))):     
        res = nlp.ner(sentences[i])
        out=np.zeros(23)
        for i in range(0,len(res)):
            tmp = res[i][1]
            if tmp=='PERSON':
                out[0]=1
            elif tmp=='LOCATION':
                out[1]=1
            elif tmp=='ORGANIZATION':
                out[2]=1
            elif tmp=='MISC':
                out[3]=1
            elif tmp=='MONEY':
                out[4]=1
            elif tmp=='NUMBER':
                out[5]=1
            elif tmp=='ORDINAL':
                out[6]=1
            elif tmp=='PERCENT':
                out[7]=1
            elif tmp=='DATE':
                out[8]=1
            elif tmp=='TIME':
                out[9]=1
            elif tmp =='DURATION':
                out[10]=1
            elif tmp=='SET':
                out[11]=1
            elif tmp =='EMAIL':
                out[12]=1
            elif tmp=='URL':
                out[13]=1
            elif tmp == 'CITY':
                out[14]=1
            elif tmp == 'STATE_OR_PROVINCE':
                out[15]=1
            elif tmp == 'COUNTRY':
                out[16]=1
            elif tmp =='NATIONALITY':
                out[17]=1
            elif tmp == 'RELIGION':
                out[18]=1
            elif tmp == 'TITLE':
                out[19]=1
            elif tmp == 'IDEOLOGY':
                out[20]=1
            elif tmp =='CRIMINAL_CHARGE':
                out[21]=1
            elif tmp == 'CAUSE_OF_DEATH':
                out[22]=1
        ner_vec.append(out)
    nlp.close()
    return ner_vec

def compute_class(sentences):
    types=[]
    for i in range(len(sentences)):
        if 'how' in sentences[i]:
            qt=(1, 0, 0, 0, 0)
        elif 'what' in sentences[i]:
            qt=(0, 1, 0, 0, 0)
        elif 'when' in sentences[i]:
            qt=(0, 0, 1, 0, 0)
        elif 'where' in sentences[i]:
            qt=(0, 0, 0, 1, 0)
        elif 'who' in sentences[i]:
            qt=(0, 0, 0, 0, 1)
        else:
            qt=(0, 0, 0, 0, 0)
        types.append(qt)
    return np.asarray(types).astype(float)

