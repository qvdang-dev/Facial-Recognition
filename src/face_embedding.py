import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import ZeroPadding2D, Convolution2D, MaxPool2D
from tensorflow.keras.layers import Dense, Dropout, Softmax, Flatten, Activation, BatchNormalization
import tensorflow.keras.backend as K

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

model = Sequential()
model.add(ZeroPadding2D((1,1), input_shape = (224, 224, 3)))
model.add(Convolution2D(64, (3,3), activation='relu'))
model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(64, (3,3), activation='relu'))
model.add(MaxPool2D((2,2), strides=(2,2)))

model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(128, (3,3), activation='relu' ))
model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(128, (3,3), activation='relu' ))
model.add(MaxPool2D((2,2), strides=(2,2)))

model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(256, (3,3), activation='relu' ))
model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(256, (3,3), activation='relu' ))
model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(256, (3,3), activation='relu' ))
model.add(MaxPool2D((2,2), strides=(2,2)))

model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(512, (3,3), activation='relu' ))
model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(512, (3,3), activation='relu' ))
model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(512, (3,3), activation='relu' ))
model.add(MaxPool2D((2,2), strides=(2,2)))

model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(512, (3,3), activation='relu' ))
model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(512, (3,3), activation='relu' ))
model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(512, (3,3), activation='relu' ))
model.add(MaxPool2D((2,2), strides=(2,2)))

model.add(Convolution2D(4096, (7, 7), activation='relu'))
model.add(Dropout(0.5))

model.add(Convolution2D(4096, (1,1), activation='relu'))
model.add(Dropout(0.5))

model.add(Convolution2D(2622, (1,1), activation='relu'))
model.add(Flatten())
model.add(Activation('softmax'))

model.load_weights('../cnn_models/vgg_face_weights.h5')

vgg_face = Model(inputs =model.layers[0].input, outputs = model.layers[-2].output)
print(model.summary())
print(tf.__version__)