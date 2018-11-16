from os.path import join as path
from keras.models import load_model
from keras.preprocessing.text import text_to_word_sequence
import pickle
from keras.preprocessing.sequence import pad_sequences
import numpy as np
import time
import pandas as pd
import os
import sys
model_path = 'model.h5'
word2idx_path = 'word2idx.pickle'
test_data_path = sys.argv[1]
#test_data_path = 'testing_data.txt'
output_path = sys.argv[2]
#output_path = 'test.csv'


word_dim = 350
sentence_len = 43
min_count = 10

model = load_model(model_path)
print('read test data...')
with open(test_data_path, 'r', encoding='utf8') as f:
    data = f.read()
data = data.split('\n')
test_data = []
for i in range(1,200001):
    temp = data[i].split(',', 1)
    temp_string = text_to_word_sequence(temp[1],filters='"#$%&()*+-/:;<=>@[\]^_`{|}~')
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
    test_data.append(temp_string2)
with open(word2idx_path, "rb") as fp:
    word2idx = pickle.load(fp)
sentences_num = []
for sentence in test_data:
    sentence_num = []
    for word in sentence:
        if word in word2idx:
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
predict = model.predict(sentences_num,verbose=1)
print('time elasped: {0:.5f}'.format(time.time()-t_start))
predict = (predict>0.5).astype(np.int16).flatten()
predict = np.hstack((np.arange(predict.shape[0]).reshape(-1,1), predict.reshape(-1,1)))
predict_df = pd.DataFrame(data = predict, columns = ['id', 'label'])
predict_df.to_csv(output_path, index = False)