
from keras.layers.core import Dropout, Activation, Dense,Flatten#,TimeDistributedDense
from keras.layers.wrappers import TimeDistributed
from keras.layers.recurrent import GRU, LSTM
from keras.layers import Conv1D, GlobalAveragePooling1D, MaxPooling1D

from keras.models import Sequential#, Graph
from keras.optimizers import Adagrad, SGD,Adadelta
from keras import callbacks
import numpy as np

SEED = 7

class MyModel:

    def __init__(self, input_len,num_features, num_labels):
        self.num_of_features = num_features
        self.output_dim = num_labels
        self.input_len = input_len
        self._model = Sequential()  # build an empty model
        np.random.seed(SEED)

    def build_model(self, mode, net_arc, drop_out_rates,weights_path=None):

        if mode.upper() == 'GRU':
            if len(net_arc) == 0:
                self._model.add(GRU(input_shape=(self.input_len,self.num_of_features), output_dim=self.output_dim, return_sequences=True))
            else:
                self._model.add(GRU(input_shape=(self.input_len,self.num_of_features), output_dim=net_arc[0], return_sequences=True))
                self._model.add(Dropout(drop_out_rates[0]))
                for i in range(1, len(net_arc)):
                    self._model.add(GRU(input_dim=net_arc[i - 1], output_dim=net_arc[i], return_sequences=True))
                    self._model.add(Dropout(drop_out_rates[i - 1]))
                self._model.add(TimeDistributed(Dense(activation='softmax', output_dim=self.output_dim)))

        if mode.upper() == 'GRU_1':
            if len(net_arc) == 0:
                self._model.add(GRU(input_shape=(self.input_len,self.num_of_features), output_dim=self.output_dim, return_sequences=True))
            else:
                self._model.add(GRU(input_shape=(self.input_len,self.num_of_features), output_dim=net_arc[0], return_sequences=True))
                self._model.add(Dropout(drop_out_rates[0]))
                for i in range(1, len(net_arc)):
                    self._model.add(GRU(input_dim=net_arc[i - 1], output_dim=net_arc[i], return_sequences=True))
                    self._model.add(Dropout(drop_out_rates[i - 1]))
                self._model.add(Flatten())
                self._model.add(Dense(units=self.output_dim,activation='softmax'))

        elif mode.upper() == 'LSTM':
            if len(net_arc) == 0:
                self._model.add(LSTM(input_shape=(self.input_len,self.num_of_features), output_dim=self.output_dim, return_sequences=True))
            else:
                self._model.add(LSTM(input_shape=(self.input_len,self.num_of_features), output_dim=net_arc[0], return_sequences=True))
                self._model.add(Dropout(drop_out_rates[0]))
                for i in range(1, len(net_arc)):
                    self._model.add(LSTM(input_dim=net_arc[i - 1], output_dim=net_arc[i], return_sequences=True))
                    self._model.add(Dropout(drop_out_rates[i - 1]))
                self._model.add(TimeDistributed(Dense(activation='softmax', output_dim=self.output_dim)))

#learn about conv 1d
        elif mode.upper() == 'CONV1':
            if True:
                self._model.add(
                    Conv1D(filters=64, kernel_size=3, strides=1, input_shape=(self.input_len,self.num_of_features), activation='relu'))#border ='valid'
                self._model.add(MaxPooling1D(2))
                self._model.add(Conv1D(128, 3, activation='relu'))
                self._model.add(MaxPooling1D(2))
                self._model.add(Flatten())
                self._model.add(Dropout(0.1))
                self._model.add(Dense(units=200, activation='relu'))
                self._model.add(Dense(units=self.output_dim, activation='softmax'))
                self._model.summary()
            else:
                self._model.add(
                    Conv1D(filters=64, kernel_size=3, strides=1, input_shape=(self.input_len, self.num_of_features),
                           border_mode='valid', activation='relu'))
                self._model.add(MaxPooling1D(pool_size=2))
                self._model.add(Flatten())
                self._model.add(Dense(units=200, activation='relu'))
                self._model.add(Dense(units=self.output_dim, activation='softmax'))
            '''
            if len(net_arc) == 0:
                self._model.add(Conv1D(filters=64,kernel_size=3,strides=1,input_shape=(self.input_len,self.num_of_features), border_mode='valid', activation='relu'))
                self._model.add(MaxPooling1D(pool_size=2))
            else:
                self._model.add(Conv1D(filters=64,kernel_size=3,input_shape=(self.input_len,self.num_of_features),border_mode='valid',activation='relu'))
                self._model.add(MaxPooling1D(pool_size=2))
                #self._model.add(Dropout(drop_out_rates[0]))
                for i in range(1, len(net_arc)-1):
                    self._model.add(Conv1D(filters=10,kernel_size=5,border_mode='valid',activation='relu'))
                    self._model.add(MaxPooling1D(pool_size=2))
                
                #self._model.add(Conv1D(64, 3, activation='relu'))
                #self._model.add(MaxPooling1D(3))
                self._model.add(Conv1D(128, 3, activation='relu'))
                self._model.add(GlobalAveragePooling1D())
                self._model.add(Dropout(0.1))

            self._model.add(Flatten())
            self._model.add(Dense(units=200, activation='relu'))
            self._model.add(Dense(units=self.output_dim, activation='softmax'))
        '''
        if weights_path:
            self.load_weights(weights_path)
        # compile the model
        #optimizer = Adadelta()
        optimizer = Adagrad()

        #optimizer = SGD(lr=0.01, momentum=0.7, nesterov=False)
        self._model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=["accuracy"])
        return self._model

    def fit(self, x_train, y_train, model_path, early_stopping_patience, val_percentage, batch_size, num_epochs):

        # preparing the callbacks
        check_pointer = callbacks.ModelCheckpoint(filepath=model_path, verbose=0, save_best_only=True)
        early_stop = callbacks.EarlyStopping(patience=early_stopping_patience, verbose=1)

        # training
        self._model.fit(x_train, y_train, validation_split=val_percentage, batch_size=batch_size, epochs=num_epochs,
                    callbacks = [check_pointer, early_stop], shuffle=False)#callbacks=[check_pointer, early_stop],

    def evaluate(self, x_test, y_test):
        outs = self._model.evaluate(x_test, y_test, verbose=1)
        return outs

    def predict(self, x_test):
        y_hat = self._model.predict(x_test, batch_size=1, verbose=0)
        return y_hat

    def load_weights(self, filename):
        self._model.load_weights(filename)

'''
model = MyModel(2,2)
model.build_model("GRU",[150],[0.01])
x_train = np.array([[[1,1]],[[0,0]],[[1,1]],[[1,1]],[[0,0]],[[1,1]],[[0,0]],[[1,1]],[[1,1]],[[1,1]]])#,[[2,2]],[[2,2]],[[1,1]],[[1,1]],[[2,2]],[[2,2]],[[2,2]],[[2,2]],[[2,2]],[[2,2]]])
y_train = np.array([[[1,0]],[[0,1]],[[1,0]],[[1,0]],[[0,1]],[[1,0]],[[0,1]],[[1,0]],[[1,0]],[[1,0]]])#,[[0,1]],[[0,1]],[[1,0]],[[1,0]],[[0,1]],[[0,1]],[[0,1]],[[0,1]],[[0,1]],[[0,1]]])


model.fit(x_train,y_train,"model.txt",15,0.2,4,20)
x = x_train[:2]

out = model.predict(x_train[:3])
print (out)
out = model.evaluate(x_train[:3],y_train[:3])
print (out)
'''