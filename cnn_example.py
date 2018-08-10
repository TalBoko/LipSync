from __future__ import absolute_import
from __future__ import print_function
import numpy as np

np.random.seed(2222)  # for reproducibility

from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.utils import np_utils
from keras.regularizers import l2, activity_l2

nb_classes = 10


def load_data(filename, setSize):
    data = np.empty((setSize, 784), dtype=np.float32)
    label = np.empty((setSize, 1), dtype=np.int32)
    j = 0
    for line in open(filename, "rb"):
        lineArray = line.split(',')
        label[j, :] = lineArray.pop(0)
        data[j, :] = lineArray
        j = j + 1
    return data, label
'''
model = Sequential()
model.add(Conv1D(64, 3, activation='relu', input_shape=(seq_length, 100)))
model.add(Conv1D(64, 3, activation='relu'))
model.add(MaxPooling1D(3))
model.add(Conv1D(128, 3, activation='relu'))
model.add(Conv1D(128, 3, activation='relu'))
model.add(GlobalAveragePooling1D())
model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])

model.fit(x_train, y_train, batch_size=16, epochs=10)
score = model.evaluate(x_test, y_test, batch_size=16)'''

def get_model():
    nn = Sequential()

    nn.add(Convolution2D(32, 3, 3,
                         border_mode='valid',
                         input_shape=(1, 28, 28)))
    nn.add(Activation('relu'))
    nn.add(MaxPooling2D(pool_size=(2, 2)))
    nn.add(Dropout(0.5))
    nn.add(Convolution2D(64, 3, 3))
    nn.add(Activation('relu'))
    nn.add(MaxPooling2D(pool_size=(2, 2)))

    nn.add(Flatten())
    nn.add(Dense(128))
    nn.add(Activation('relu'))
    #nn.add(Dropout(0.5))
    nn.add(Dense(nb_classes))
    nn.add(Activation('softmax'))

    nn.compile(loss='categorical_crossentropy', optimizer='adadelta', metrics=['accuracy'])

    return nn


batch_size = 128
epochs = 10

X_train, y_train = load_data("/home/osboxes/Documents/data/train.txt", 60000)
X_test, y_test = load_data("/home/osboxes/Documents/data/validate1.txt", 10000)

X_train = X_train.reshape(X_train.shape[0], 1, 28, 28)
X_test = X_test.reshape(X_test.shape[0], 1, 28, 28)
X_train /= 255
X_test /= 255

Y_train = np_utils.to_categorical(y_train, nb_classes)
Y_test = np_utils.to_categorical(y_test, nb_classes)

nn = get_model()
nn.fit(X_train, Y_train, batch_size=batch_size, nb_epoch=epochs, verbose=1, validation_data=(X_test, Y_test))
