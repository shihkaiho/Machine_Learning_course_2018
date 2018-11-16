import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint
from os.path import join as path
from keras.preprocessing.image import ImageDataGenerator
import pandas as pd
from keras.utils import np_utils
import time
import sys
train_file_name = sys.argv[1]
def read_data():
    print('strat reading ...')
    tStart = time.time()
    RawData = pd.read_csv(train_file_name)
    train_Y = RawData['label'].values.astype(np.float64).reshape(-1,1)
    train_Y = np_utils.to_categorical(train_Y,7)
    data_num = train_Y.shape[0]
    dim = np.fromstring(RawData['feature'].values[0],\
                        dtype=np.float64, sep=' ').shape[0]
    dim2D = int(np.sqrt(dim))
    train_X = np.empty((data_num, dim2D, dim2D, 1))
    for i in range(data_num):
        train_X[i,:,:,0] = np.fromstring(RawData['feature'].values[i],\
                                       dtype=np.float64,\
                                       sep=' ').reshape(dim2D,dim2D)/255.
    print('end of reading')
    print('time elapsed: {0:5f}s'.format(time.time()-tStart))
    return train_X, train_Y
train_X, train_Y = read_data()

#%%
model = Sequential()

model.add(ZeroPadding2D((1,1),input_shape=(48,48,1)))
model.add(Convolution2D(32, 3, 3, activation='relu'))
model.add(Convolution2D(32, 3, 3, activation='relu', border_mode='same'))
model.add(MaxPooling2D((2,2), strides=(2,2)))

model.add(Convolution2D(64, 3, 3, activation='relu', border_mode='same'))
model.add(Convolution2D(64, 3, 3, activation='relu', border_mode='same'))
model.add(MaxPooling2D((2,2), strides=(2,2)))

model.add(Convolution2D(128, 3, 3, activation='relu', border_mode='same'))
model.add(Convolution2D(128, 3, 3, activation='relu', border_mode='same'))
model.add(Convolution2D(128, 3, 3, activation='relu', border_mode='same'))
model.add(MaxPooling2D((2,2), strides=(2,2)))


model.add(Flatten())
model.add(Dense(196, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(196, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(7, activation='softmax'))

model.summary()
#%%
train_X = np.load(path('..','data_set','train_X.npy'))
train_Y = np.load(path('..','data_set','train_Y.npy'))
datagen2 = ImageDataGenerator(
    featurewise_center=True,
    featurewise_std_normalization=True,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True)
datagen2.fit(train_X)

datagen1 = ImageDataGenerator(
    featurewise_center=True,
    featurewise_std_normalization=True,
    rotation_range=10,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True)
datagen1.fit(train_X)

datagen3 = ImageDataGenerator(
    featurewise_center=True,
    featurewise_std_normalization=True,
    rotation_range=30,
    width_shift_range=0.3,
    height_shift_range=0.3,
    horizontal_flip=True)
datagen3.fit(train_X)
#%%
bs = 156
ep = 13
vs = 0.1
learningRate = 3e-4
model.compile(loss='categorical_crossentropy',\
              optimizer = Adam(lr=learningRate),\
              metrics=['accuracy'])
checkpoint = ModelCheckpoint('model.h5',\
                             monitor = 'val_acc',\
                             save_best_only = True)

model.fit_generator(datagen2.flow(train_X, train_Y, batch_size=64), steps_per_epoch=len(train_X) / 64, epochs=10)
model.fit_generator(datagen1.flow(train_X, train_Y, batch_size=64), steps_per_epoch=len(train_X) / 64, epochs=10)
model.fit_generator(datagen3.flow(train_X, train_Y, batch_size=64), steps_per_epoch=len(train_X) / 64, epochs=10)
model.fit(train_X,train_Y,batch_size = 128,\
          epochs = 6,\
          validation_split = 0.1)

model.fit_generator(datagen3.flow(train_X, train_Y, batch_size=128), steps_per_epoch=len(train_X) / 128, epochs=10)
model.fit_generator(datagen2.flow(train_X, train_Y, batch_size=128), steps_per_epoch=len(train_X) / 128, epochs=10)
model.fit_generator(datagen1.flow(train_X, train_Y, batch_size=128), steps_per_epoch=len(train_X) / 128, epochs=10)
history = model.fit(train_X,train_Y,batch_size = 512,
                    epochs = 1,
                    validation_split = 0.2,
                    shuffle = True,
                    callbacks = [checkpoint])

model.save(path('models','CNN.h5'))