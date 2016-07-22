#!/usr/bin/python2.7
"""
Created on Fri Jul 22 10:53:21 2016

@author: liuzheng
"""
import os
import numpy as np
np.random.seed(1337)  # for reproducibility

from keras.datasets import mnist
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.optimizers import SGD, Adam, RMSprop, Adagrad, Adadelta
from keras.utils import np_utils

import skimage.io as skio
import skimage.transform as sktr

nb_classes = 10

dim = 784
compress_to_dim = 100

# the data, shuffled and split between train and test sets
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# convert class vectors to binary class matrices
Y_train = np_utils.to_categorical(y_train, nb_classes)
Y_test = np_utils.to_categorical(y_test, nb_classes)

X_train = X_train.reshape(60000, dim)
X_test = X_test.reshape(10000, dim)
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255
print(X_train.shape[0], 'train samples')
print(X_test.shape[0], 'test samples')

W1 = np.random.random((dim, compress_to_dim))

xx_train = X_train.dot(W1)
xx_train = np.tan(xx_train)

xx_test = X_test.dot(W1)
xx_test = np.tan(xx_test)

print list(X_train[0])



# config model
optimizer = 'adam'
loss = 'mean_squared_error'

# create model
model = Sequential()
model.add(Dense(dim, input_dim=compress_to_dim))
model.add(Activation('relu'))

model.summary()


if optimizer == 'sgd':
    opt = SGD(lr=1e-3, decay=1e-6, momentum=0.9, nesterov=True)
elif optimizer == 'adam':
    opt = Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
elif optimizer == 'adagrad':
    opt = Adagrad(lr=0.01, epsilon=1e-08)
elif optimizer == 'adadelta':
    opt = Adadelta(lr=1.0, rho=0.95, epsilon=1e-08)
elif optimizer == 'rmsprop':
    opt = RMSprop(lr=0.001, rho=0.9, epsilon=1e-08)

model.compile(loss='mean_squared_error',
              optimizer=opt,)

# train config
batch_size = 16
nb_classes = 10
nb_epoch = 100


# training
history = model.fit(xx_train, X_train,
                    batch_size=batch_size, nb_epoch=nb_epoch,
                    verbose=1, validation_data=(xx_test, X_test))
#score = model.evaluate(xx_test, xx_test, verbose=0)
from keras import backend as K

# with a Sequential model
decoder = K.function([model.layers[0].input, K.learning_phase()],
                              [model.layers[1].output])

dir = "data"
if not os.path.exists(dir):
    os.makedirs(dir)

for i in range(xx_test.shape[0]):
    tmp = np.zeros((1, compress_to_dim))
    tmp[0, :] = xx_test[i, :]
    recon = decoder([tmp, 0])[0]
    recon = np.reshape(recon, [28, 28])
    origin = np.reshape(X_test[i], [28, 28])

    skio.imsave("data/%d_ori.bmp" % i, origin)
    skio.imsave("data/%d_new.bmp" % i, recon)

    if i > 100:
        print "done %d img" % i
        break
