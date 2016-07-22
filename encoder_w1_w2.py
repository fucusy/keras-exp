#!/usr/bin/python2.7
from keras.datasets.mnist import load_data
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Activation


(X_train, y_train), (X_test, y_test) = load_data()

dim = 784
compress_dim = 2

X_train = X_train.reshape((X_train.shape[0], dim)) 
X_test = X_test.reshape((X_test.shape[0], dim))


W1 = np.random.random((dim, compress_dim))

X_train_c = X_train.dot(W1)
X_test_c = X_test.dot(W1)




# define your model
model = Sequential()
model.add(Dense(dim, input_shape = (compress_dim,)))
model.add(Activation('tanh'))
model.compile(optimizer='sgd', loss="mean_squared_error")

# train config
epoch = 5



# train
model.fit(X_train_c, X_train, validation_data=(X_test_c, X_test), nb_epoch=epoch)
