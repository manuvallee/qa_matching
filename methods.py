from nltk.corpus import stopwords
import string
import numpy as np
from matplotlib import pyplot as plt
from utils import *
from scipy.optimize import linear_sum_assignment
from gensim.models import KeyedVectors
import os
from utils import *
from models import *
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.models import Model, load_model
from keras.utils import plot_model
from gensim.summarization.bm25 import BM25, get_bm25_weights
#import ot




################################################## Unsupervised methods ###############################################
def wordcount(args):
    pairs, labels, ids = load_dataset(args,'test') 
    puncts = set(list(string.punctuation))
    stop_words = set(stopwords.words())

    scores=[]
    for i in range(0,len(pairs)):
        q_words = [word for word in pairs[i][0].split() if word not in puncts and word not in stop_words]
        a_words = [word for word in pairs[i][1].split() if word not in puncts and word not in stop_words]
        scores.append(sum([word in q_words for word in a_words])/(len(a_words)+1)+np.random.rand(1,1)*0.00001)  # *-1 because argsort goes ascending
    return np.ravel(scores)

def bm25(args):
    pairs, labels, ids = load_dataset(args,'test')
    scores = []
    all_text = []

    for i in tqdm(range(len(pairs))):
        all_text.append(pairs[i][1].split())
    bm25 = BM25(all_text)
    average_idf = sum(float(val) for val in bm25.idf.values()) / len(bm25.idf)
    scores = []
     
    for i in range(len(pairs)):
        scores.append(bm25.get_score(pairs[i][0].split(), i, average_idf)+np.random.rand(1,1)*0.00001)
    scores = np.asarray(scores)
    
    return np.ravel(scores)

'''
Cosine on averaged word embeddings
'''
def avg_embedding(args):
    pairs, labels, ids = load_dataset(args,'test') 
    # Load the embedding dictionnary
    embedding = KeyedVectors.load_word2vec_format(os.path.join('data',args.embedding,'embedding.bin'), binary=True)
    # Compute the averaged word embedding for the question and answer sequences
    s_q_words,s_a_words = avg_emb_sent(pairs, embedding)
    scores = []
    for i in range(len(s_a_words)):
        # Cosine similarity
        scores.append(np.dot(s_q_words[i],s_a_words[i])/(np.linalg.norm(s_q_words[i])*np.linalg.norm(s_a_words[i])))
    return(np.ravel(scores))
    
#'''
#Experiment with Earth Moving Distance (Optimal transport) requires the ot library
#'''
#def emd(args):
#    pairs, labels, ids = load_dataset(args,'test') 
#    embedding = KeyedVectors.load_word2vec_format(os.path.join('data',args.embedding,'embedding.bin'), binary=True)
#    scores = []
#    for i in range(0,len(pairs)):
#        q_words = [embedding[word] for word in pairs[i][0].split() if word in embedding.vocab]
#        a_words = [embedding[word] for word in pairs[i][1].split() if word in embedding.vocab]
#        if len(a_words)==0:
#            a_words = np.ones((1,300))
#        if len(q_words)==0:
#            q_words = np.ones((1,300))
#        C2= -ot.dist(np.asarray(q_words),np.asarray(a_words))
#        scores.append(C2.sum())
#    return(np.ravel(scores)
   
    

'''
Hungarian method
'''
def hungarian(args):
    pairs, labels, ids = load_dataset(args,'test') 
    embedding = KeyedVectors.load_word2vec_format(os.path.join('data',args.embedding,'embedding.bin'), binary=True)
    scores = []
    for i in range(0,len(pairs)):
        q_words = [embedding[word] for word in pairs[i][0].split() if word in embedding.vocab]
        a_words = [embedding[word] for word in pairs[i][1].split() if word in embedding.vocab]
        
        # In case the sequence contains no words in the embedding dictionnary, fill a vector with 0
        if len(a_words)==0:
            a_words = np.ones((1,300))
        if len(q_words)==0:
            q_words = np.ones((1,300))
        # mat is the similarity matrix
        mat=np.zeros((len(a_words),len(q_words)))
        for j in range(0,len(a_words)):
            for k in range(0,len(q_words)):
                # NB: The similarity is given by the dot product, without normalisation (better performances)
                mat[j,k] = (np.dot(a_words[j], q_words[k]))#/(np.linalg.norm(a_words[j])*np.linalg.norm(q_words[k]))
        # The hungarian method looks for the minimal cost, hence we transform the similarity into cost
        cost = np.max(mat)-mat
        # assign is the optimal assignment matrix, computed with the scipy library
        assign = linear_sum_assignment(cost)
        # sum the vector resulting from the hadamard product
        score = sum(mat[assign])
        scores.append(score)
    return(np.ravel(scores))
'''
Cosine on USE embeddings
'''
def USE(args):
    pairs, labels, ids = load_dataset(args,'test')
    scores = []
    embedding = KeyedVectors.load_word2vec_format(os.path.join('data',args.dataset,'USE_embedding.txt'), binary=False)
    for i in range(len(ids)):
        scores.append(np.dot(embedding[ids[i][0]],embedding[ids[i][1]])/(np.linalg.norm(embedding[ids[i][0]])*np.linalg.norm(embedding[ids[i][1]])))
    return np.ravel(scores)
'''
Cosine on BERT embeddings
'''
def BERT(args):
    pairs, labels, ids = load_dataset(args,'test')
    scores = []
    embedding = KeyedVectors.load_word2vec_format(os.path.join('data',args.dataset,'BERT_embedding.txt'), binary=False)
    for i in range(len(ids)):
        scores.append(np.dot(embedding[ids[i][0]],embedding[ids[i][1]])/(np.linalg.norm(embedding[ids[i][0]])*np.linalg.norm(embedding[ids[i][1]])))
    return np.ravel(scores)


######################################################### Supervised #####################################################

'''
pointwise training
'''
def pointwise(args):
    batch_size = 32
    # Load the train, dev, test sets with a given representation (one embedding vector per sequence)
    if args.representation != 'avg_emb':
        train_q, train_a, train_labels, dev_q, dev_a, dev_labels, test_q, test_a, test_labels = prepare_data_supervised_USE(args)
    else:  
        train_pairs, train_labels, ids = load_dataset(args,'train')
        dev_pairs, dev_labels, ids = load_dataset(args,'dev')
        test_pairs, test_labels, ids = load_dataset(args,'test')
        
        embedding = KeyedVectors.load_word2vec_format(os.path.join('data',args.embedding,'embedding.bin'), binary=True)
        
        train_q,train_a = avg_emb_sent(train_pairs, embedding)
        dev_q,dev_a = avg_emb_sent(dev_pairs, embedding)
        test_q,test_a = avg_emb_sent(test_pairs, embedding)
    
#    if args.ner:
#        train_aner = np.loadtxt(os.path.join('data','preprocessed',args.dataset,'train_answers_ner'))
#        dev_aner = np.loadtxt(os.path.join('data','preprocessed',args.dataset,'dev_answers_ner'))
#        test_aner = np.loadtxt(os.path.join('data','preprocessed',args.dataset,'test_answers_ner'))
#        train_a = np.concatenate((train_a,train_aner),axis = 1)
#        dev_a = np.concatenate((dev_a,dev_aner),axis = 1)
#        test_a = np.concatenate((test_a,test_aner),axis = 1)
#        
#        train_qner = np.loadtxt(os.path.join('data','preprocessed',args.dataset,'train_questions_ner'))
#        dev_qner = np.loadtxt(os.path.join('data','preprocessed',args.dataset,'dev_questions_ner'))
#        test_qner = np.loadtxt(os.path.join('data','preprocessed',args.dataset,'test_questions_ner'))        
#        train_q = np.concatenate((train_q,np.zeros(np.shape(train_qner))),axis = 1)
#        dev_q = np.concatenate((dev_q,np.zeros(np.shape(dev_qner))),axis = 1)
#        test_q = np.concatenate((test_q,np.zeros(np.shape(test_qner))),axis = 1)


    # Early stopping
    earlystop = EarlyStopping(monitor='val_loss', min_delta=0.0001, patience=5, verbose=args.verbose, mode='auto')
    # Save best model only
    checkpointer = ModelCheckpoint(filepath=args.save_weights, verbose=args.verbose, save_best_only=True, save_weights_only = True)
    callbacks_list = [earlystop, checkpointer]
    
    # Build the model
    if args.arch=='cos':
        net=nn_cos(np.shape(train_q)[1])
    else:    
        net=nn(np.shape(train_q)[1])
    
    # If the model is not pre-trained    
    if args.load_weights is None:
        # Perform the training
        net.fit([train_q, train_a], train_labels,batch_size,
                nb_epoch=int(args.number_epochs), verbose=args.verbose, shuffle = True,
                validation_data=([dev_q, dev_a],dev_labels),
                callbacks = callbacks_list)
        # Load the best weights (safety training several models at the same time)
        net.load_weights(args.save_weights)
        
    # If the model is pre-trained 
    else:
        print ('Loading weights')
        net.load_weights(args.load_weights)
        if args.finetune:
            net.fit([train_q, train_a], train_labels,batch_size = batch_size,
                    nb_epoch=int(args.number_epochs), verbose=args.verbose, shuffle = True,
                    validation_data=([dev_q, dev_a],dev_labels),
                    callbacks = callbacks_list)
            net.load_weights(args.save_weights)
    # The scores are the model predictions        
    scores = np.ravel(net.predict([test_q, test_a]))   
    
    return scores

def pairwise(args):
    margin = args.margin
    def local_triplet_loss(y_true, y_pred):
        scorepos=y_pred[:,0]
        scoreneg=y_pred[:,1]
        triplet_loss=K.maximum(-scorepos+scoreneg+margin,0)
        #valid_triplets = tf.to_float(tf.greater(triplet_loss, 1e-16))
        #num_positive_triplets = tf.reduce_sum(valid_triplets)
        #loss=K.sum(triplet_loss)/(num_positive_triplets+1e-16)
        #return(loss) # loss normalisee par le nombre de triplet qui contribue
        return(K.sum(triplet_loss)) # loss non normalisee


    def custom_object_triplet_loss():
        return{'local_triplet_loss':local_triplet_loss}
    
    # If USE representation
    if args.representation != 'avg_emb':
        train_triplets,  train_ids = load_dataset(args,'train', triplets = True)
        dev_triplets,  dev_ids = load_dataset(args,'dev',triplets = True)
        test_triplets,  test_ids = load_dataset(args,'test',triplets = True)        
        pairs, labels, pairs_ids = load_dataset(args, 'test')        
        embedding = KeyedVectors.load_word2vec_format(os.path.join('data',args.dataset,args.representation.upper()+'_embedding.txt'), binary=False)
        train_a,train_p, train_n = embed_USE(train_ids, embedding, triplets = True)
        dev_a,dev_p, dev_n = embed_USE(dev_ids, embedding, triplets = True)
        test_a,test_p, test_n = embed_USE(test_ids, embedding, triplets = True)
    # If average embedding representation
    else:
        embedding = KeyedVectors.load_word2vec_format(os.path.join('data',args.embedding,'embedding.bin'), binary=True)      
        train_triplets,  train_ids = load_dataset(args,'train', triplets = True)
        dev_triplets,  dev_ids = load_dataset(args,'dev',triplets = True)
        test_triplets,  test_ids = load_dataset(args,'test',triplets = True)    
        pairs, labels, pairs_ids = load_dataset(args, 'test')     
        train_a,train_p, train_n = avg_emb_sent(train_triplets, embedding, triplets = True)
        dev_a,dev_p, dev_n = avg_emb_sent(dev_triplets, embedding, triplets = True)
        test_a,test_p, test_n = avg_emb_sent(test_triplets, embedding, triplets = True)

    X=[train_a,train_p,train_n]
    X_dev = [dev_a, dev_p, dev_n]
    X_test = [test_a, test_p, test_n]
    
    y=np.ones((train_a.shape[0],2))
    ydev_val=np.ones((dev_a.shape[0],2))    
    
    # Build the model
    if args.arch == 'cos':
        net=nn_triplet_cos(np.shape(train_a)[1])
    else:
        net=nn_triplet(np.shape(train_a)[1])
    net.compile(optimizer='adam',loss=local_triplet_loss)
    print('compile ok')
    
    

    monitor=ModelCheckpoint(args.save_weights,monitor='val_loss',verbose=args.verbose,save_best_only=True, save_weights_only = True)
    earlystop = EarlyStopping(monitor='val_loss', min_delta=0.0001, patience=5, verbose=args.verbose, mode='auto')
    if args.load_weights is None:    
        net.fit(X,y, batch_size=32, epochs=args.number_epochs, validation_data=(X_dev,ydev_val), shuffle=True,callbacks=[monitor, earlystop],verbose=args.verbose)  
        net.load_weights(args.save_weights)
    else:
        print ('Loading weights')
        net.load_weights(args.load_weights)  
        if args.finetune:
            net.fit(X,y, batch_size=32, epochs=args.number_epochs, validation_data=(X_dev,ydev_val), shuffle=True,callbacks=[monitor, earlystop],verbose=args.verbose)  
            net.load_weights(args.save_weights)
            
    # prediction test
    Xp=net.predict(X_test)
    Xp_pos=Xp[:,0]
    Xp_neg=Xp[:,1]
    scores = write_scores_triplet('na',test_ids,pairs_ids,Xp_pos,Xp_neg)
    return scores

################################################################ Experimental ###########################################################

def nn_bert(args):
    batch_size = 32
    train_q, train_a, train_labels, dev_q, dev_a, dev_labels, test_q, test_a, test_labels = prepare_data_supervised_BERT(args)
    if args.ner:
        train_aner = np.loadtxt(os.path.join('data','preprocessed',args.dataset,'train_answers_ner'))
        dev_aner = np.loadtxt(os.path.join('data','preprocessed',args.dataset,'dev_answers_ner'))
        test_aner = np.loadtxt(os.path.join('data','preprocessed',args.dataset,'test_answers_ner'))
        train_a = np.concatenate((train_a,train_aner),axis = 1)
        dev_a = np.concatenate((dev_a,dev_aner),axis = 1)
        test_a = np.concatenate((test_a,test_aner),axis = 1)
        
        #train_qner = np.loadtxt(os.path.join('data','preprocessed',args.dataset,'train_questions_ner'))
        #dev_qner = np.loadtxt(os.path.join('data','preprocessed',args.dataset,'dev_questions_ner'))
        #test_qner = np.loadtxt(os.path.join('data','preprocessed',args.dataset,'test_questions_ner'))        
        #train_q = np.concatenate((train_q,np.zeros(np.shape(train_aner))),axis = 1)
        #dev_q = np.concatenate((dev_q,np.zeros(np.shape(dev_aner))),axis = 1)
        #test_q = np.concatenate((test_q,np.zeros(np.shape(test_aner))),axis = 1)
    
    n_units = 1500
    earlystop = EarlyStopping(monitor='val_loss', min_delta=0.0001, patience=5, verbose=args.verbose, mode='auto')
    checkpointer = ModelCheckpoint(filepath=args.save_weights, verbose=args.verbose, save_best_only=True, save_weights_only = True)
    
    callbacks_list = [earlystop, checkpointer]
    if args.arch=='cos':
        net=nn_cos(np.shape(train_q)[1],n_units)
    else:    
        net=nn(np.shape(train_q)[1],n_units )
    plot_model(net, to_file='nn_use.png')
    if args.load_weights is None:
        net.fit([train_q, train_a], train_labels,32,
                nb_epoch=int(args.number_epochs), verbose=args.verbose, shuffle = True,
                validation_data=([dev_q, dev_a],dev_labels),
                callbacks = callbacks_list)
        net.load_weights(args.save_weights)
        
    else:
        print ('Loading weights')
        net.load_weights(args.load_weights)  
        if args.finetune:
            net.fit([train_q, train_a], train_labels,batch_size = batch_size,
                    nb_epoch=int(args.number_epochs), verbose=args.verbose, shuffle = True,
                    validation_data=([dev_q, dev_a],dev_labels),
                    callbacks = callbacks_list)
            net.load_weights(args.save_weights)
            
    scores = np.ravel(net.predict([test_q, test_a]))   
    
    return scores




def cnn_emb(args):
    nb_filter = 4000
    filter_width = 4
    q_seq_len = 20
    a_seq_len = 50
    margin = args.margin
       
    word_seq_train_q, word_seq_train_a,labels_train, word_seq_dev_q, word_seq_dev_a,labels_dev, word_seq_test_q, word_seq_test_a, labels_test, embedding_matrix = prepare_data_supervised(args, q_seq_len = q_seq_len, a_seq_len = a_seq_len)

    earlystop = EarlyStopping(monitor='val_loss', min_delta=0.0001, patience=5, verbose=1, mode='auto')
    checkpointer = ModelCheckpoint(filepath=args.save_weights, verbose=1, save_best_only=True)
    callbacks_list = [earlystop, checkpointer]
    net=cnn_cos(left_seq_len = q_seq_len, right_seq_len = a_seq_len , embed_dimensions = 300, embedding_matrix = embedding_matrix,  nb_filter = 4000 , filter_width = 7 , out_dim_param = 200, drop_out = 0.4)
    net.compile(optimizer = 'adam', loss='binary_crossentropy')
    net.fit([word_seq_train_q, word_seq_train_a], labels_train,
        nb_epoch=args.number_epochs, verbose=1, batch_size=32,shuffle = True,
        validation_data=([word_seq_dev_q, word_seq_dev_a],labels_dev), 
        callbacks = callbacks_list)
    scores = np.ravel(net.predict([word_seq_test_q, word_seq_test_a]))
    return scores
    
#%%ABCNN

    
def abcnn(args):
    nb_filter = 4000
    filter_width = 4
    q_seq_len = 20
    a_seq_len = 50

       
    word_seq_train_q, word_seq_train_a,labels_train, word_seq_dev_q, word_seq_dev_a,labels_dev, word_seq_test_q, word_seq_test_a, labels_test, embedding_matrix = prepare_data_supervised(args, q_seq_len = q_seq_len, a_seq_len = a_seq_len)

    earlystop = EarlyStopping(monitor='val_loss', min_delta=0.0001, patience=5, verbose=1, mode='auto')
    checkpointer = ModelCheckpoint(filepath=args.save_weights, verbose=1, save_best_only=True)
    callbacks_list = [earlystop, checkpointer]
    net=abcnn(q_seq_len = q_seq_len, a_seq_len = a_seq_len , embedding_matrix = embedding_matrix,  nb_filter = 4000 , filter_width = 7 )
    net.compile(optimizer = 'adam', loss='binary_crossentropy')
    net.fit([word_seq_train_q, word_seq_train_a], labels_train,
        nb_epoch=args.number_epochs, verbose=1, batch_size=32,shuffle = True,
        validation_data=([word_seq_dev_q, word_seq_dev_a],labels_dev), 
        callbacks = callbacks_list)
    scores = np.ravel(net.predict([word_seq_test_q, word_seq_test_a]))
    return scores    
#%%  Triplets

def triplet_cnn_emb(args):
    def local_triplet_loss(y_true, y_pred):
        scorepos=y_pred[:,0]
        scoreneg=y_pred[:,1]
        triplet_loss=K.maximum(-scorepos+scoreneg+margin,0)
        #valid_triplets = tf.to_float(tf.greater(triplet_loss, 1e-16))
        #num_positive_triplets = tf.reduce_sum(valid_triplets)
        #loss=K.sum(triplet_loss)/(num_positive_triplets+1e-16)
        #return(loss) # loss normalisee par le nombre de triplet qui contribue
        return(K.sum(triplet_loss)) # loss non normalisee


    def custom_object_triplet_loss():
        return{'local_triplet_loss':local_triplet_loss}
    
    nb_filter = 4000
    filter_width = 7
    a_seq_len = 20
    p_seq_len = 40
    margin = args.margin
    
    word_seq_train_a, word_seq_train_p, word_seq_train_n, word_seq_dev_a, word_seq_dev_p, word_seq_dev_n, word_seq_test_a, word_seq_test_p, word_seq_test_n, embedding_matrix = prepare_data_supervised_triplets(args, a_seq_len = a_seq_len, p_seq_len = p_seq_len, n_seq_len = 50)
    test_triplets,  test_ids = load_dataset(args,'test',triplets = True)
    pairs, labels, pairs_ids = load_dataset(args, 'test')   
    earlystop = EarlyStopping(monitor='val_loss', min_delta=0.0001, patience=5, verbose=1, mode='auto')
    monitor=ModelCheckpoint(args.save_weights,monitor='val_loss',verbose=args.verbose,save_best_only=True, save_weights_only = True)
    net=cnn_triplet_cos(left_seq_len = a_seq_len, right_seq_len = p_seq_len , embed_dimensions = 300, embedding_matrix = embedding_matrix,  nb_filter = 4000 , filter_width = 7 , out_dim_param = 200, drop_out = 0.4)
    net.compile(optimizer = 'adam', loss=local_triplet_loss)
    net.summary()
    labels_train=np.ones((word_seq_train_a.shape[0],2))
    labels_dev=np.ones((word_seq_dev_a.shape[0],2))  
    
    if args.load_weights is None:    
        net.fit([word_seq_train_a, word_seq_train_p, word_seq_train_n],labels_train, batch_size=32, epochs=args.number_epochs, validation_data=([word_seq_dev_a, word_seq_dev_p, word_seq_dev_n],labels_dev), shuffle=True,callbacks=[monitor, earlystop],verbose=args.verbose)  
        net.load_weights(args.save_weights)
    else:
        print ('Loading weights')
        net.load_weights(args.load_weights)  
        if args.finetune:
            net.fit([word_seq_train_a, word_seq_train_p, word_seq_train_n],labels_train, batch_size=32, epochs=args.number_epochs, validation_data=([word_seq_dev_a, word_seq_dev_p, word_seq_dev_n],labels_dev), shuffle=True,callbacks=[monitor, earlystop],verbose=args.verbose)  
            net.load_weights(args.save_weights)
            
    # prediction test
    Xp=net.predict([word_seq_test_a, word_seq_test_p, word_seq_test_n])
    Xp_pos=Xp[:,0]
    Xp_neg=Xp[:,1]
    scores = write_scores_triplet('na',test_ids,pairs_ids,Xp_pos,Xp_neg)
    return scores
   