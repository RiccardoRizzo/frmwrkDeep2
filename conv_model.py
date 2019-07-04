from keras.layers import Conv1D, MaxPooling1D, Dense, Flatten, Dropout, Embedding, LSTM, Activation
from keras.models import Sequential
from keras.regularizers import l2
from keras import optimizers


def build_conv_lstm(inputShape, filters, kernel_len, lstm_hidden_size):
    """
    Il modello ha un input shape prefissato (151,4) relativo 
    alla rappresentazione delle sequenze adottata.
    
    Modello formato dai seguenti strati:
        - convoluzione di 
            **filters numero** di kernel, di dimensione
            **kernel_len** con **padding=same**, 
            quindi l'uscita e' della stessa dimensione dell'ingresso.
        - attivazione ReLU
        - MaxPooling1D senza parametri
        - Droput con probabilita' 50%
        
        - LSTM con 50 unita' , return_sequences=True, kernel_regularized=l2(1e-3), recurrent dropout=0.1
        - Dropout=50%
        
        - Flatten
        
        - strato fully connected (Dense) di 150 unita', kernel_regularized=l2(1e-3),
          attivazione relu
        - Dropout 50%
        - strato fully connected di dimensione 1 con kernel_regularized=l2(1e-3) 
          e attivazione sigmoide
        
    ottimizzazione **Adam** con **learning rate=0.0003** e funzione di loss
    **binary_crossentropy**.
    
    """
    beta = 1e-3
    model = Sequential()
    
    #model.add(Conv1D(filters=50, kernel_size=3, input_shape=(151, 4), kernel_regularizer=l2(beta), padding='same'))
    model.add(Conv1D(filters, kernel_len, input_shape=inputShape, kernel_regularizer=l2(beta), padding='same'))
    model.add(Activation('relu'))    
    model.add(MaxPooling1D())
    model.add(Dropout(0.5))
    
    model.add(LSTM(units=50, return_sequences=True, kernel_regularizer=l2(beta), recurrent_dropout=0.1))
    model.add(Dropout(0.5))
    
    model.add(Flatten())    
    #model.add(Dense(1, kernel_regularizer=l2(beta), activation='sigmoid'))
    model.add(Dense(150, kernel_regularizer=l2(beta), activation='relu'))

    model.add(Dropout(0.5))
    model.add(Dense(1, kernel_regularizer=l2(beta), activation='sigmoid'))

    optim = optimizers.Adam(lr=0.0003)
    model.compile(optimizer=optim, loss='binary_crossentropy')

    return model


def build_conv_conv(inputShape, filters1, kernel_len1, filters2 ):
    """
    Il modello ha un input shape prefissato (151,4) relativo 
    alla rappresentazione delle sequenze adottata.
    
    Modello formato dai seguenti strati:
        - convoluzione di 
            **filters numero** di kernel, di dimensione
            **kernel_len** con **padding=same**, 
            quindi l'uscita e' della stessa dimensione dell'ingresso.
        - attivazione ReLU
        - MaxPooling1D senza parametri
        - Droput con probabilita' 50%
        
        - convoluzione di 
            **filters numero** di kernel, di dimensione
            **kernel_len** con **padding=same**, 
            quindi l'uscita e' della stessa dimensione dell'ingresso.
        - attivazione ReLU
        - MaxPooling1D senza parametri
        - Droput con probabilita' 50%

        
        - Flatten
        
        - strato fully connected (Dense) di 150 unita', kernel_regularized=l2(1e-3),
          attivazione relu
        - Dropout 50%
        - strato fully connected di dimensione 1 con kernel_regularized=l2(1e-3) 
          e attivazione sigmoide
        
    ottimizzazione **Adam** con **learning rate=0.0003** e funzione di loss
    **binary_crossentropy**.
    
    """
    beta = 1e-3
    model = Sequential()
    
    #model.add(Conv1D(filters=50, kernel_size=3, input_shape=(151, 4), kernel_regularizer=l2(beta), padding='same'))
    model.add(Conv1D(filters1, kernel_len1, input_shape=inputShape, kernel_regularizer=l2(beta), padding='same'))
    model.add(Activation('relu'))    
    model.add(MaxPooling1D())
    model.add(Dropout(0.5))
    
    model.add(Conv1D(filters2, kernel_len1, kernel_regularizer=l2(beta), padding='same'))
    model.add(Activation('relu'))    
    model.add(MaxPooling1D())
    model.add(Dropout(0.5))
    
    
    model.add(Flatten())    
    #model.add(Dense(1, kernel_regularizer=l2(beta), activation='sigmoid'))
    model.add(Dense(1024, kernel_regularizer=l2(beta), activation='relu'))

    model.add(Dropout(0.5))
    model.add(Dense(1, kernel_regularizer=l2(beta), activation='sigmoid'))

    optim = optimizers.Adam(lr=0.0003)
    model.compile(optimizer=optim, loss='binary_crossentropy')

    return model


def build_conv_2conv(inputShape, filters1, kernel_len1, filters2 ):
    """
    Il modello ha un input shape prefissato (151,4) relativo 
    alla rappresentazione delle sequenze adottata.
    
    Modello formato dai seguenti strati:
        - convoluzione di 
            **filters numero** di kernel, di dimensione
            **kernel_len** con **padding=same**, 
            quindi l'uscita e' della stessa dimensione dell'ingresso.
        - attivazione ReLU
        - MaxPooling1D senza parametri
        - Droput con probabilita' 50%
        
        - convoluzione di 
            **filters numero** di kernel, di dimensione
            **kernel_len** con **padding=same**, 
            quindi l'uscita e' della stessa dimensione dell'ingresso.
        - attivazione ReLU
        - MaxPooling1D senza parametri
        - Droput con probabilita' 50%

        
        - Flatten
        
        - strato fully connected (Dense) di 150 unita', kernel_regularized=l2(1e-3),
          attivazione relu
        - Dropout 50%
        - strato fully connected di dimensione 1 con kernel_regularized=l2(1e-3) 
          e attivazione sigmoide
        
    ottimizzazione **Adam** con **learning rate=0.0003** e funzione di loss
    **binary_crossentropy**.
    
    """
    beta = 1e-3
    model = Sequential()
    
    #model.add(Conv1D(filters=50, kernel_size=3, input_shape=(151, 4), kernel_regularizer=l2(beta), padding='same'))
    model.add(Conv1D(filters1, kernel_len1, input_shape=inputShape, kernel_regularizer=l2(beta), padding='same'))
    model.add(Activation('relu'))    
    model.add(MaxPooling1D())
    model.add(Dropout(0.5))
    
    model.add(Conv1D(filters2, 2*kernel_len1, kernel_regularizer=l2(beta), padding='same'))
    model.add(Activation('relu'))    
    model.add(MaxPooling1D())
    model.add(Dropout(0.5))
    
    
    model.add(Flatten())    
    #model.add(Dense(1, kernel_regularizer=l2(beta), activation='sigmoid'))
    model.add(Dense(1024, kernel_regularizer=l2(beta), activation='relu'))

    model.add(Dropout(0.5))
    model.add(Dense(1, kernel_regularizer=l2(beta), activation='sigmoid'))

    optim = optimizers.Adam(lr=0.0003)
    model.compile(optimizer=optim, loss='binary_crossentropy')

    return model

def predict_classes(self, x, batch_size=32, verbose=1):
	'''Generate class predictions for the input samples batch by batch.
		# Arguments
			x: input data, as a Numpy array or list of Numpy arrays (if the model has multiple inputs).
			batch_size: integer.
			verbose: verbosity mode, 0 or 1.
		# Returns
			A numpy array of class predictions.
	'''
	proba = self.predict(x, batch_size=batch_size, verbose=verbose)
	if proba.shape[-1] > 1:
		return proba.argmax(axis=-1)
	else:
		return (proba > 0.5).astype('int32')


def predict_classes_prob(self, x, batch_size=32, verbose=1):
    """
    Restutuisce la pobabilitia' delle classi
    :param self:
    :param x:
    :param batch_size:
    :param verbose:
    :return:
    """
    return self.predict(x, batch_size=batch_size, verbose=verbose)
