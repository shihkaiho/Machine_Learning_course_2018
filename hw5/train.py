import numpy as np
from os.path import join as path
from gensim.models import Word2Vec
from keras.preprocessing.text import text_to_word_sequence
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential, load_model
from keras.layers.recurrent import LSTM
from keras.layers import Dense, Embedding
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint
from keras import regularizers
import pickle
import time
import sys
import pprint
import os
label_data_path = sys.argv[1]
#label_data_path = 'training_label.txt'
nolabel_data_path = sys.argv[2]
#nolabel_data_path = 'training_nolabel.txt'
word_dim = 350
sentence_len = 43
min_count = 10

def read_label_data():
    print('read label data...')
    with open(label_data_path,'r',encoding='utf8') as f:
        data = f.read()
    data = data.split('\n')
    label_data = []
    label = []
    for i in range(200000):
        temp = data[i].split('+++$+++')
        temp_string = text_to_word_sequence(temp[1],filters='"#$%&()*+-/:;<=>@[\]^_`{|}~')
#        temp_string = text_to_word_sequence(temp[1],filters='!"#$%&()*+,-./:;<=>?@[\]^_`{|}~')                                       
        temp_string2 = []
        prime_flag = False
        for word in temp_string:
            if prime_flag:
                if len(temp_string2)==0:
                    prime_flag = False
                    continue
                temp_string2[-1] = temp_string2[-1] + "'" + word
                prime_flag = False
                continue
            if word == "'":
                prime_flag = True
                continue
            if any(char.isdigit() for char in word):
                continue
            temp_string2.append(word)
        if len(temp_string2) != 0:
            label.append(int(temp[0]))
            label_data.append(temp_string2)
    label = np.array(label)
    with open(path('temp','label_data.pickle'), "wb") as fp:
        pickle.dump(label_data, fp)
    np.save(path('temp','label.npy'),label) 
    return
  
def read_nolabel_data():
    print('read no label data...')
    with open(nolabel_data_path,'r',encoding='utf8') as f:
        data = f.read()
    data = data.split('\n')
    nolabel_data = []
    for i in range(1178614):
        temp_string = text_to_word_sequence(data[i],filters='"#$%&()*+-/:;<=>@[\]^_`{|}~')
#        temp_string = text_to_word_sequence(data[i],filters='!"#$%&()*+,-./:;<=>?@[\]^_`{|}~')                                       
        temp_string2 = []
        prime_flag = False
        for word in temp_string:
            if prime_flag:
                if len(temp_string2)==0:
                    prime_flag = False
                    continue
                temp_string2[-1] = temp_string2[-1] + "'" + word
                prime_flag = False
                continue
            if word == "'":
                prime_flag = True
                continue
            if any(char.isdigit() for char in word):
                continue
            temp_string2.append(word)
        if len(temp_string2) != 0:
            nolabel_data.append(temp_string2)
    with open(path('temp','nolabel_data.pickle'), "wb") as fp:
        pickle.dump(nolabel_data, fp)
    return
   
def get_word2vec_model():
    print('get word2vec model...')
    with open(path('temp','label_data.pickle'), "rb") as fp:
        label_data = pickle.load(fp)
    with open(path('temp','nolabel_data.pickle'), "rb") as fp:
        nolabel_data = pickle.load(fp)
    model = Word2Vec(label_data+nolabel_data,
                     size=word_dim,
                     window=5,
                     min_count=min_count,
                     workers=4)
    model.save(path('temp',"word2vec_model_{0}".format(word_dim)))
    return

def get_keras_embedding():
    model = Word2Vec.load(path('temp',"word2vec_model_{0}".format(word_dim)))
    vocab_list = sorted([(k, model.wv[k], v.index) for k, v in model.wv.vocab.items()],key=lambda x:x[2])
    #0 for padding
    embeddings_matrix = np.zeros((len(model.wv.vocab.items()) + 2, model.vector_size))
    word2idx = {"__PAD__": 0, "__unknown__":1}
    unknown_vec = np.zeros(word_dim)
    for i in range(len(vocab_list)):
        word = vocab_list[i][0]
        word2idx[word] = i + 2
        embeddings_matrix[i + 2] = vocab_list[i][1]
        unknown_vec += vocab_list[i][1]
    unknown_vec /= len(model.wv.vocab.items())
    embeddings_matrix[1] = unknown_vec
    embedding_layer = Embedding(input_dim=embeddings_matrix.shape[0],
                                output_dim=embeddings_matrix.shape[1],
                                weights=[embeddings_matrix],
                                trainable=False)
    with open('word2idx.pickle', "wb") as fp:
        pickle.dump(word2idx, fp)
    return embedding_layer

def get_keras_model():
    embedding_layer = get_keras_embedding()
    model = Sequential()
    model.add(embedding_layer)
    model.add(LSTM(units=word_dim))
#    model.add(Dense(units=32, activation='relu',kernel_regularizer=regularizers.l2(0.1)))
#    model.add(Dropout(0.5))
#    model.add(Dense(units=32, activation='relu',kernel_regularizer=regularizers.l2(0.1)))
    model.add(Dense(units=128, activation='relu',kernel_regularizer=regularizers.l2(0.1)))
#    model.add(Dropout(0.5))
#    model.add(Dense(units=64, activation='relu'))
#    model.add(Dropout(0.5))
#    model.add(Dense(units=32, activation='relu'))
    model.add(Dense(units=1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy',
                  optimizer=Adam(lr=1e-3),
                  metrics=['accuracy'])
    model.summary()
    return model

def data_analysis(sentences_num):
    max_len = 0
    min_len = 1000
    average_len = 0
    for sentence in sentences_num:
        sentence_len = len(sentence)
        if sentence_len>max_len:
            max_len = sentence_len
        if sentence_len<min_len:
            min_len = sentence_len
        average_len += sentence_len
    average_len /= len(sentences_num)
    print('max len: {0}'.format(max_len))
    print('min len: {0}'.format(min_len))
    print('average len: {0}'.format(average_len))
#    temp = input('type to contunue...')
    
def hash_word():
    with open(path('temp','label_data.pickle'), "rb") as fp:
        label_data = pickle.load(fp)
    with open(path('word2idx.pickle'), "rb") as fp:
        word2idx = pickle.load(fp)
    sentences_num = []
    for sentence in label_data:
        sentence_num = []
        for word in sentence:
            if (word in word2idx):
                sentence_num.append(word2idx[word])
            else:
                sentence_num.append(word2idx["__unknown__"])
        sentences_num.append(sentence_num)
    return sentences_num

def get_data():
    sentences_num = hash_word()
    data_analysis(sentences_num)
    sentences_num = pad_sequences(sentences_num,
                                  maxlen=sentence_len,
                                  dtype='int32',
                                  padding='post',
                                  truncating='post',
                                  value=0)
    sentences_num = np.array(sentences_num)
    return sentences_num
    
def semi_supervise():
    model = load_model(path('temp','model_best.h5'))
    print('read no label data...')
    with open(path('temp','nolabel_data.pickle'), "rb") as fp:
        nolabel_data = pickle.load(fp)
    with open(path('temp','word2idx.pickle'), "rb") as fp:
        word2idx = pickle.load(fp)
    sentences_num = []
    print('hash no label data')
    for sentence in nolabel_data:
        sentence_num = []
        for word in sentence:
            if (word in word2idx):
                sentence_num.append(word2idx[word])
            else:
                sentence_num.append(word2idx["__unknown__"])
        sentences_num.append(sentence_num)
    sentences_num = pad_sequences(sentences_num,
                                  maxlen=sentence_len,
                                  dtype='int32',
                                  padding='post',
                                  truncating='post',
                                  value=0)
    sentences_num = np.array(sentences_num)
    print('predict...')
    t_start = time.time()
    predict = model.predict(sentences_num, verbose=1)
    print('time elasped: {0:.5f}'.format(time.time()-t_start))
    new_positive = np.where(predict>0.8)
    new_positive = sentences_num[new_positive[0]]
    new_negative = np.where(predict<0.2)
    new_negative = sentences_num[new_negative[0]]
    new_label = np.hstack( ( np.ones(new_positive.shape[0]),
                           np.zeros(new_negative.shape[0])) )
    new_data = np.vstack((new_positive,new_negative))
    seed = np.arange(new_data.shape[0])
    np.random.shuffle(seed)
    new_label = new_label[seed]
    new_data = new_data[seed]
    
    return new_label, new_data
#%%
    
if __name__=='__main__':
    os.system('mkdir temp')
    read_label_data()
    read_nolabel_data()
    get_word2vec_model()
    #%%
    model = get_keras_model()
    model.save_weights(path('temp','my_model_weights.h5'))
    sentences_num = get_data()
    os.system('rm model_best.h5')
    checkpoint = ModelCheckpoint(filepath=path('temp','model_best.h5'),
                                 monitor='val_acc',
                                 save_best_only=True
                                 )
    
    label = np.load(path('temp','label.npy'))
    
    train_X = sentences_num[:-20000]
    train_Y = label[:-20000]
    
    val_X = sentences_num[-20000:]
    val_Y = label[-20000:]
    
    history = model.fit(x=train_X,
                        y=train_Y,
                        batch_size=256,
                        epochs=10,
                        validation_data=(val_X,val_Y),
                        callbacks=[checkpoint],
                        shuffle=True)
    #%%
    model_structure = history.history
    model_structure['word_dim'] = word_dim
    model_structure['min_count'] = min_count
    model_structure['sentence_len'] = sentence_len
    
    with open(path('temp','log.txt'), "w") as fp:
        temp = sys.stdout
        sys.stdout = fp
        model.summary()
        pprint.pprint(model_structure)
        sys.stdout = temp
    #%%
    checkpoint = ModelCheckpoint(filepath=path('temp','model_best2.h5'),
                                 monitor='val_acc',
                                 save_best_only=True
                                 )
    os.system('rm -rf temp')
    model.save('model.h5')
#    new_label, new_data = semi_supervise()
#    train_Y = np.hstack((train_Y,new_label))
#    train_X = np.vstack((train_X,new_data))
#    model.load_weights(path('..','temp','my_model_weights.h5'))
#    history = model.fit(x=train_X,
#                        y=train_Y,
#                        batch_size=512,
#                        epochs=10,
#                        validation_data=(val_X,val_Y),
#                        callbacks=[checkpoint],
#                        shuffle=True)
        
    
    
    
    
    
    