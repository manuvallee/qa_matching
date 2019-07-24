
#!/usr/bin/env python
# coding=utf-8

import tensorflow as tf
import bottle
from bottle import route, run
import threading
import json
import numpy as np
import os
import string
from time import sleep
from nltk.corpus import stopwords
from scipy.optimize import linear_sum_assignment
from gensim.models import KeyedVectors
import flask
from flask import request


app = flask.Flask(__name__)
app.config["DEBUG"] = True
embedding = KeyedVectors.load_word2vec_format(os.path.join('data','wikiqa','embedding.bin'), binary=True)


@app.route('/', methods=['GET'])
def home():
    return " "

app.run()



@app.route('/sim', methods=['GET'])
def sim():
    
    #return(request.args.get('question'))
    return str(hungarian(request.args.get('question'),request.args.get('answer'),embedding))

    # the code below is executed if the request method
    # was GET or the credentials were invalid



def hungarian(question, answer, embedding):
    #embedding = KeyedVectors.load_word2vec_format(os.path.join('data',args.embedding,'embedding.bin'), binary=True)
    stop_words = set(stopwords.words('english'))
    puncts = set(list(string.punctuation))
    q_words = [embedding[word] for word in question.split() if word in embedding.vocab and word not in stop_words and word not in puncts]
    a_words = [embedding[word] for word in answer.split() if word in embedding.vocab and word not in stop_words and word not in puncts]
    if len(a_words)==0:
        a_words = np.ones((1,300))
    if len(q_words)==0:
        q_words = np.ones((1,300))
    mat=np.zeros((len(a_words),len(q_words)))
    for j in range(0,len(a_words)):
        for k in range(0,len(q_words)):
            mat[j,k] = (np.dot(a_words[j], q_words[k]))#/(np.linalg.norm(a_words[j])*np.linalg.norm(q_words[k]))
    cost = np.max(mat)-mat
    assign = linear_sum_assignment(cost)
    score = (sum(mat[assign]))
    return(score)
