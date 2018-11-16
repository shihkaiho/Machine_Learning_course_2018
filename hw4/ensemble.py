import numpy as np
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint
from os.path import join as path
from PIL import Image
import os
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import np_utils
import pandas as pd
#%%
model = Sequential()

model.add(ZeroPadding2D((1,1),input_shape=(48,48,1)))
model.add(Convolution2D(28, 3, 3, activation='relu'))
model.add(MaxPooling2D((2,2), strides=(2,2)))

model.add(Convolution2D(56, 3, 3, activation='relu', border_mode='same'))
model.add(Convolution2D(56, 3, 3, activation='relu', border_mode='same'))
model.add(MaxPooling2D((2,2), strides=(2,2)))

model.add(Convolution2D(108, 3, 3, activation='relu', border_mode='same'))
model.add(Convolution2D(108, 3, 3, activation='relu', border_mode='same'))
model.add(MaxPooling2D((2,2), strides=(2,2)))


model.add(Flatten())
model.add(Dense(169, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(169, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(7, activation='softmax'))

model.summary()
#%%
train_X = np.load(path('..','data_set','train_X.npy'))
train_Y = np.load(path('..','data_set','train_Y.npy'))
val_X = train_X[:3000]
val_Y = train_Y[:3000]
train_X = train_X[3000:]
train_Y = train_Y[3000:]
data_num = train_X.shape[0]
weights = np.ones(data_num)
alpha = []
train_Y_single = train_Y.argmax(axis=1)
#%%
learningRate = 4e-4
model.compile(loss='categorical_crossentropy',\
              optimizer = Adam(lr=learningRate),\
              metrics=['accuracy'])
#%%
adaboost_num = 4
for i in range(adaboost_num):
    save_best = ModelCheckpoint('model{0}_save_best.h5'.format(i),\
                             monitor = 'val_acc',\
                             save_best_only = True)
    history = model.fit(train_X,train_Y,batch_size=256,
                        epochs=20,
                        validation_data=(val_X,val_Y),
                        sample_weight=weights,
                        callbacks=[save_best])
    loss, accuracy = model.evaluate(train_X,train_Y)
    err = 1. - accuracy
    alpha.append(np.log(np.sqrt((1-err)/err)))
    predict = model.predict(train_X).argmax(axis=1)
    answer = np.logical_and(predict,train_Y_single)
    true = np.where(answer==True)
    false = np.where(answer==False)
    weights[true] = weights[true]*np.exp(-alpha[i])
    weights[false] = weights[false]*np.exp(alpha[i])
    model.save('model{0}.h5'.format(i))
    
#%%
model = []
for i in range(adaboost_num):
    model.append(load_model('model{0}.h5'.format(i)))
predict = []
for i in range(adaboost_num):
    predict.append(model[i].predict(train_X).argmax(axis=1))
for i in range(adaboost_num):
    predict[i] = np_utils.to_categorical(predict[i],7)
final_ans = 0.
for i in range(adaboost_num):
    final_ans += alpha[i]*predict[i]
final_ans = final_ans.argmax(axis=1)
answer_check = np.logical_and(final_ans,train_Y_single)
accuracy = np.where(answer_check==True)[0].shape[0]/answer_check.shape[0]
print('adaboost accuracy:',accuracy)
test_X = np.load(path('..','data_set','test_X.npy'))
data_num = test_X.shape[0]
predict = []
for i in range(adaboost_num):
    predict.append(model[i].predict(test_X).argmax(axis=1))
for i in range(adaboost_num):
    predict[i] = np_utils.to_categorical(predict[i],7)
final_ans = 0.
for i in range(adaboost_num):
    final_ans += alpha[i]*predict[i]
result = final_ans.argmax(axis=1)
id = np.arange(data_num)
output = pd.DataFrame({'id': id,'label':result})
output.to_csv('adaboost.csv', index = False)

model0 = load_model('model0.h5')
result = model0.predict(test_X).argmax(axis=1)
id = np.arange(data_num)
output = pd.DataFrame({'id': id,'label':result})
output.to_csv('no_adaboost.csv', index = False)

loss, accuracy = model0.evaluate(train_X,train_Y)
print('original model accuracy:',accuracy)






