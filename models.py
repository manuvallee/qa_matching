import keras
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras import optimizers
from keras import backend as K
from keras import regularizers
from keras.models import Model
from keras.utils import plot_model
from keras.callbacks import EarlyStopping
from keras.optimizers import SGD,Adam, Adagrad
from keras.preprocessing import sequence
from keras import backend as K
from keras.layers import *
import numpy as np
import tensorflow as tf

def abs_diff(X):
    s = X[0]
    for i in range(1, len(X)):
        s -= X[i]
    s = K.abs(s)
    return s


##################################################### Pointwise ############################################""


'''
Concat architecture
'''
def nn(embed_dim = 512):    
    #Inputs
    input_shape1 = (embed_dim,)
    input_shape2 = (embed_dim,)
    x1 = Input(input_shape1)
    x2 = Input(input_shape2)
    
    # Add handcrafted operation give +2 pts 
    x3 = Multiply()([x1,x2])
#    x4 = Subtract()([x1, x2])
    x4 = Lambda(abs_diff)([x1, x2])
    
    x =  Concatenate()([x1, x2, x3, x4])
    # There are n = concat_dim hidden units
    x = Dense(K.int_shape(x)[1], activation = 'relu')(x)
    x = Dropout(0.3) (x)
    
    prediction = Dense(1,activation='sigmoid')(x)
    
    nn = Model(input=[x1,x2],output=prediction)
    
    optim = optimizers.Adam()
    nn.compile(loss="binary_crossentropy",optimizer=optim)
    print(nn.summary())
    return nn


'''
Siamese architecture
'''
def nn_cos(embed_dim = 512):   
    #Inputs
    input_shape1 = (embed_dim,)
    input_shape2 = (embed_dim,)
    x1 = Input(input_shape1)
    x2 = Input(input_shape2)
    # There are n = embed_dim hidden units
    dense1 = Dense(K.int_shape(x1)[1],activation = 'relu')
    x11 = dense1(x1)
    x11=Dropout(0.3)(x11)
    x21 = dense1(x2)
    x21 = Dropout(0.3)(x21)
    prediction = dot([x11,x21],axes=(1,1),normalize=True)
    #prediction = MatchScore(x11, x21, mode='euclidean')
    nn = Model(input=[x1,x2],output=prediction)
    
    # Compile
    optim = optimizers.Adam()#Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
    nn.compile(loss="binary_crossentropy",optimizer=optim)
    print(nn.summary())
    return nn

# Combining siamese with concat architecture degrades the performance
#def nn_both(embed_dim = 512):   
#    #Inputs
#    input_shape1 = (embed_dim,)
#    input_shape2 = (embed_dim,)
#    x1 = Input(input_shape1)
#    x2 = Input(input_shape2)
#    dense1 = Dense(K.int_shape(x1)[1],activation = 'relu')
#    x11 = dense1(x1)
#    x11=Dropout(0.3)(x11)
#    x21 = dense1(x2)
#    x21 = Dropout(0.3)(x21)
#    x3 = Multiply()([x21,x11])
#    #x4 = Subtract()([x21, x11])
#    x4 = Lambda(abs_diff)([x1, x2])
#    x =  Concatenate()([x21, x11, x3, x4])
#    x = Dense(K.int_shape(x)[1], activation = 'relu')(x)
#    x = Dropout(0.3) (x)
#    prediction = Dense(1,activation='sigmoid')(x)
#    #prediction = MatchScore(x11, x21, mode='euclidean')
#    nn = Model(input=[x1,x2],output=prediction)
#    
#    # Compile
#    optim = optimizers.Adam()#Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
#    nn.compile(loss="binary_crossentropy",optimizer=optim)
#    print(nn.summary())
#    return nn


#%% Triplets
def nn_triplet(embed_dim = 300):
    drop_out = 0.3    
    anchor_input = Input((embed_dim,))
    pos_input = Input((embed_dim,))
    neg_input = Input((embed_dim,))
    
    pos_mult = Multiply()([anchor_input, pos_input])
    #pos_dist = Subtract()([anchor_input, pos_input]) substract and abs substract gives similar performances
    pos_dist = Lambda(abs_diff)([anchor_input, pos_input])
    
    neg_mult = Multiply()([anchor_input, neg_input])
    #neg_dist = Subtract()([anchor_input, neg_input])
    neg_dist = Lambda(abs_diff)([anchor_input, neg_input])
    
    pos = Concatenate()([anchor_input,pos_input, pos_mult, pos_dist])
    neg = Concatenate()([anchor_input,neg_input, neg_mult, neg_dist])
    
    dense1 = Dense(K.int_shape(pos)[1], activation = 'relu')
    dense2 = Dense(1,activation = 'sigmoid')
    
    pos = dense1(pos)
    pos = Dropout(drop_out)(pos)
    pos = dense2(pos)
    
    neg = dense1(neg)
    neg = Dropout(drop_out)(neg)
    neg = dense2(neg)
    
    out = Concatenate()([pos,neg])
       
    # Compile
    optim = optimizers.Adam()#lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)   
    
    nn_triplet = Model(input=[anchor_input,pos_input, neg_input],output=out)
    #plot_model(dan, to_file='model.png')
    return nn_triplet
    

#retourne une paire de similarite (simPos,simNeg) du triplet, ou la similarite=dense(anchor,test) avec un embeddings global de chaque phrase
def nn_triplet_cos(embed_dim):
    drop_out = 0.3
    anchor_input = Input((embed_dim,))
    pos_input = Input((embed_dim,))
    neg_input = Input((embed_dim,))

    dense1 = Dense(K.int_shape(pos_input)[1], activation='relu')

    anchor_output=dense1(anchor_input)
    pos_output=dense1(pos_input)
    neg_output=dense1(neg_input)

    anchor_output=Dropout(drop_out)(anchor_output)
    pos_output=Dropout(drop_out)(pos_output)
    neg_output=Dropout(drop_out)(neg_output)

    cospos=dot([anchor_output,pos_output],axes=(1,1),normalize=True)
    cosneg=dot([anchor_output,neg_output],axes=(1,1),normalize=True)

    triplet_embed=concatenate([cospos,cosneg],axis=-1)
    nn = Model([anchor_input,pos_input,neg_input], [triplet_embed])
    print(nn.summary())

    # l'output du model n'est pas une liste de 3 tenseurs, mais un seul tenseur concatene, pour les values de label, il faut y faire attention
    return nn


###############################################  Experimental #####################################################


def compute_cos_match_score(l_r):
    l, r = l_r
    return K.batch_dot(
        K.l2_normalize(l, axis=-1),
        K.l2_normalize(r, axis=-1),
        axes=[2, 2]
    )


def compute_euclidean_match_score(l_r):
    l, r = l_r
    denominator = 1. + K.sqrt(
        -2 * K.batch_dot(l, r, axes=[2, 2]) +
        K.expand_dims(K.sum(K.square(l), axis=2), 2) +
        K.expand_dims(K.sum(K.square(r), axis=2), 1)
    )
    denominator = K.maximum(denominator, K.epsilon())
    return 1. / denominator


def MatchScore(l, r, mode="euclidean"):
    if mode == "euclidean":
        return merge(
            [l, r],
            mode=compute_euclidean_match_score,
            output_shape=lambda shapes: (None, shapes[0][1], shapes[1][1])
        )
    elif mode == "cos":
        return merge(
            [l, r],
            mode=compute_cos_match_score,
            output_shape=lambda shapes: (None, shapes[0][1], shapes[1][1])
        )
    elif mode == "dot":
        return merge([l, r], mode="dot")
    else:
        raise ValueError("Unknown match score mode %s" % mode)


def ABCNN(
        q_seq_len, a_seq_len, embedding_matrix, nb_filter, filter_widths,
        depth=2, dropout=0.4, abcnn_1=True, abcnn_2=True, collect_sentence_representations=True, mode="cos", batch_normalize=True
):
    assert depth >= 1, "Need at least one layer to build ABCNN"
    assert not (depth == 1 and abcnn_2), "Cannot build ABCNN-2 with only one layer!"
    if type(filter_widths) == int:
        filter_widths = [filter_widths] * depth
    assert len(filter_widths) == depth

    print("Using %s match score" % mode)

    nb_words = len(embedding_matrix)
    nb_filter = 50
    filter_width = 4
    embed_dim = 300 
    
    left_sentence_representations = []
    right_sentence_representations = []


    q_input_shape = (q_seq_len,)
    a_input_shape = (a_seq_len,)
    left_input = Input(q_input_shape)
    right_input = Input(a_input_shape)
    left_embed = Embedding(nb_words, embed_dim, weights=[embedding_matrix], input_length=q_seq_len, trainable=False) (left_input)
    right_embed = Embedding(nb_words, embed_dim, weights=[embedding_matrix], input_length=a_seq_len, trainable=False) (right_input)

    if batch_normalize:
        left_embed = BatchNormalization()(left_embed)
        right_embed = BatchNormalization()(right_embed)

    filter_width = filter_widths.pop(0)
    if abcnn_1:
        match_score = MatchScore(left_embed, right_embed, mode=mode)

        # compute attention
        attention_left = TimeDistributed(
            Dense(embed_dim, activation="relu"), input_shape=(q_seq_len, a_seq_len))(match_score)
        match_score_t = Permute((2, 1))(match_score)
        attention_right = TimeDistributed(
            Dense(embed_dim, activation="relu"), input_shape=(q_seq_len, a_seq_len))(match_score_t)

        left_reshape = Reshape((1, attention_left._keras_shape[1], attention_left._keras_shape[2]))
        right_reshape = Reshape((1, attention_right._keras_shape[1], attention_right._keras_shape[2]))

        attention_left = left_reshape(attention_left)
        left_embed = left_reshape(left_embed)

        attention_right = right_reshape(attention_right)
        right_embed = right_reshape(right_embed)

        # concat attention
        # (samples, channels, rows, cols)
        left_embed = merge([left_embed, attention_left], mode="concat", concat_axis=1)
        right_embed = merge([right_embed, attention_right], mode="concat", concat_axis=1)

        # Padding so we have wide convolution
        left_embed_padded = ZeroPadding2D((filter_width - 1, 0))(left_embed)
        right_embed_padded = ZeroPadding2D((filter_width - 1, 0))(right_embed)

        # 2D convolutions so we have the ability to treat channels. Effectively, we are still doing 1-D convolutions.
        conv_left = Convolution2D(
            nb_filter=nb_filter, nb_row=filter_width, nb_col=embed_dim, activation="tanh", border_mode="valid",
            dim_ordering="th"
        )(left_embed_padded)

        # Reshape and Permute to get back to 1-D
        conv_left = (Reshape((conv_left._keras_shape[1], conv_left._keras_shape[2])))(conv_left)
        conv_left = Permute((2, 1))(conv_left)

        conv_right = Convolution2D(
            nb_filter=nb_filter, nb_row=filter_width, nb_col=embed_dim, activation="tanh",
            border_mode="valid",
            dim_ordering="th"
        )(right_embed_padded)

        # Reshape and Permute to get back to 1-D
        conv_right = (Reshape((conv_right._keras_shape[1], conv_right._keras_shape[2])))(conv_right)
        conv_right = Permute((2, 1))(conv_right)

    else:
        # Padding so we have wide convolution
        left_embed_padded = ZeroPadding1D(filter_width - 1)(left_embed)
        right_embed_padded = ZeroPadding1D(filter_width - 1)(right_embed)
        conv_left = Convolution1D(nb_filter, filter_width, activation="tanh", border_mode="valid")(left_embed_padded)
        conv_right = Convolution1D(nb_filter, filter_width, activation="tanh", border_mode="valid")(right_embed_padded)

    if batch_normalize:
        conv_left = BatchNormalization()(conv_left)
        conv_right = BatchNormalization()(conv_right)

    conv_left = Dropout(dropout)(conv_left)
    conv_right = Dropout(dropout)(conv_right)

    pool_left = AveragePooling1D(pool_length=filter_width, stride=1, border_mode="valid")(conv_left)
    pool_right = AveragePooling1D(pool_length=filter_width, stride=1, border_mode="valid")(conv_right)

    #assert pool_left._keras_shape[1] == max_seq_len, "%s != %s" % (pool_left._keras_shape[1], max_seq_len)
    #assert pool_right._keras_shape[1] == max_seq_len, "%s != %s" % (pool_right._keras_shape[1], max_seq_len)

    if collect_sentence_representations or depth == 1:  # always collect last layers global representation
        left_sentence_representations.append(GlobalAveragePooling1D()(conv_left))
        right_sentence_representations.append(GlobalAveragePooling1D()(conv_right))

    # ###################### #
    # ### END OF ABCNN-1 ### #
    # ###################### #

    for i in range(depth - 1):
        filter_width = filter_widths.pop(0)
        pool_left = ZeroPadding1D(filter_width - 1)(pool_left)
        pool_right = ZeroPadding1D(filter_width - 1)(pool_right)
        # Wide convolution
        conv_left = Convolution1D(nb_filter, filter_width, activation="tanh", border_mode="valid")(pool_left)
        conv_right = Convolution1D(nb_filter, filter_width, activation="tanh", border_mode="valid")(pool_right)

        if abcnn_2:
            conv_match_score = MatchScore(conv_left, conv_right, mode=mode)

            # compute attention
            conv_attention_left = Lambda(lambda match: K.sum(match, axis=-1), output_shape=(conv_match_score._keras_shape[1],))(conv_match_score)
            conv_attention_right = Lambda(lambda match: K.sum(match, axis=-2), output_shape=(conv_match_score._keras_shape[2],))(conv_match_score)

            conv_attention_left = Permute((2, 1))(RepeatVector(nb_filter)(conv_attention_left))
            conv_attention_right = Permute((2, 1))(RepeatVector(nb_filter)(conv_attention_right))

            # apply attention  TODO is "multiply each value by the sum of it's respective attention row/column" correct?
            conv_left = merge([conv_left, conv_attention_left], mode="mul")
            conv_right = merge([conv_right, conv_attention_right], mode="mul")

        if batch_normalize:
            conv_left = BatchNormalization()(conv_left)
            conv_right = BatchNormalization()(conv_right)

        conv_left = Dropout(dropout)(conv_left)
        conv_right = Dropout(dropout)(conv_right)

        pool_left = AveragePooling1D(pool_length=filter_width, stride=1, border_mode="valid")(conv_left)
        pool_right = AveragePooling1D(pool_length=filter_width, stride=1, border_mode="valid")(conv_right)

        #assert pool_left._keras_shape[1] == max_seq_len
        #assert pool_right._keras_shape[1] == max_seq_len

        if collect_sentence_representations or (i == (depth - 2)):  # always collect last layers global representation
            left_sentence_representations.append(GlobalAveragePooling1D()(conv_left))
            right_sentence_representations.append(GlobalAveragePooling1D()(conv_right))

    # ###################### #
    # ### END OF ABCNN-2 ### #
    # ###################### #

    # Merge collected sentence representations if necessary
    left_sentence_rep = left_sentence_representations.pop(-1)
    if left_sentence_representations:
        left_sentence_rep = merge([left_sentence_rep] + left_sentence_representations, mode="concat")

    right_sentence_rep = right_sentence_representations.pop(-1)
    if right_sentence_representations:
        right_sentence_rep = merge([right_sentence_rep] + right_sentence_representations, mode="concat")

    global_representation = merge([left_sentence_rep, right_sentence_rep], mode="concat")
    global_representation = Dropout(dropout)(global_representation)

    # Add logistic regression on top.
    classify = Dense(1, activation="sigmoid")(global_representation)
    
    model =  Model([left_input, right_input], output=classify)
    opt = keras.optimizers.Adam()
    model.compile(optimizer=opt, loss="binary_crossentropy", metrics=["acc"])
    return model

