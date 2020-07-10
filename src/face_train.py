import pickle
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Dense,Activation,BatchNormalization, Dropout
from sklearn.model_selection import train_test_split

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

X = []
y = []
with open('train.pickle', 'rb') as f:
    X, y = pickle.load(f)

X = np.array(X, dtype= np.float64)
y = np.array(y, dtype= np.float64)

x_train, x_test, y_train, y_test = train_test_split(X,y, test_size = 0.2)

model = Sequential()
model.add(Dense(units=100, input_dim=x_train.shape[1], kernel_initializer='glorot_uniform'))
model.add(BatchNormalization())
model.add(Activation('tanh'))
model.add(Dropout(0.3))
model.add(Dense(units=10,kernel_initializer='glorot_uniform'))
model.add(BatchNormalization())
model.add(Activation('tanh'))
model.add(Dropout(0.2))
model.add(Dense(units=5, kernel_initializer='he_uniform'))
model.add(BatchNormalization())
model.add(Activation('softmax'))
model.compile(loss = tf.keras.losses.SparseCategoricalCrossentropy(), optimizer='nadam', metrics=['accuracy'])
model.fit(x_train, y_train, epochs=100, validation_data=(x_test, y_test))
tf.keras.models.save_model(model, '../trained_model/model_00.h5')
print('done')


