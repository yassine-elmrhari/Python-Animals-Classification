# -*- coding: utf-8 -*-
"""

@author: Yassine
"""

#############################################LIBRAIRIES###############################################
import numpy as np
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Dense
from keras.layers import Flatten
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Dropout
from keras.layers import BatchNormalization
import keras
import tensorflow as tf
from matplotlib import pyplot


#################################PARAMETRES############################################################
config = tf.compat.v1.ConfigProto(
    device_count={'GPU': 1},
    intra_op_parallelism_threads=1,
    allow_soft_placement=True
)

config.gpu_options.allow_growth = True
config.gpu_options.per_process_gpu_memory_fraction = 0.6

session = tf.compat.v1.Session(config=config)

tf.compat.v1.keras.backend.set_session(session)

Nbr_classes = 10

#############################################UNPICKLING FASE############################################
#Function to unpickle data:
def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

#Unpickling training data
for i in range(1,6):
    globals()['dict{}'.format(i)] = eval("cifar-10-batches-py/data_batch_{}')".format(i))

#Unpickling test data
test = unpickle("cifar-10-batches-py/test_batch")

############################################GENERATING X VALUES#######################################
#FOR TRAINING
for i in range(1,6):
        globals()['data{}'.format(i)] = eval("dict{}[b'data']".format(i))

X_Train = np.concatenate((data1, data2, data3, data4, data5))
X_Train = X_Train.reshape(X_Train.shape[0], 32, 32, 3) 
X_Train = X_Train.astype('float32')/255 

#FOR TESTING
X_Test = test[b'data']
X_Test = X_Test.reshape(X_Test.shape[0], 32, 32, 3) 
X_Test = X_Test.astype('float32')/255 

############################################GENERATING Y VALUES#######################################
#FOR TRAINING
for i in range(1,6):
        globals()['label{}'.format(i)] = eval("dict{}[b'labels']".format(i))

Y_Train = label1 + label2 + label3 + label4 + label5
Y_Train = keras.utils.to_categorical(Y_Train,Nbr_classes) #Optimisation des Y labels pour le CNN

#FOR TESTING
Y_Test = test[b'labels']
Y_Test = keras.utils.to_categorical(Y_Test,Nbr_classes) #Optimisiation des Y labels pour le CNN

############################################Neural Network Initialisation#############################
model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same', input_shape=(32, 32, 3)))
model.add(BatchNormalization())
model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
model.add(BatchNormalization())
model.add(MaxPooling2D((2, 2)))
model.add(Dropout(0.2))
model.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
model.add(BatchNormalization())
model.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
model.add(BatchNormalization())
model.add(MaxPooling2D((2, 2)))
model.add(Dropout(0.3))
model.add(Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
model.add(BatchNormalization())
model.add(Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
model.add(BatchNormalization())
model.add(MaxPooling2D((2, 2)))
model.add(Dropout(0.4))
model.add(Flatten())
model.add(Dense(128, activation='relu', kernel_initializer='he_uniform'))
model.add(BatchNormalization())
model.add(Dropout(0.5))
model.add(Dense(10, activation='softmax'))

# compile model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])


####################################################################################################################
#Generateur de donnees
datagen = ImageDataGenerator(width_shift_range=0.1, height_shift_range=0.1, horizontal_flip=True)

#Preparation de l'initiateur
it_train = datagen.flow(X_Train, Y_Train, batch_size=64)

#Apprentissage
res = model.fit_generator(it_train, epochs=300, validation_data=(X_Test, Y_Test), verbose=1)

#Evaluation
_, acc = model.evaluate(X_Test, Y_Test, verbose=1)
print('> %.3f' % (acc * 100.0))

#Generation des courbes
def summarize_diagnostics(history):
	# plot loss
    pyplot.subplot(211)
    pyplot.title('Cross Entropy Loss')
    pyplot.plot(history.history['loss'], color='blue', label='train')
    pyplot.plot(history.history['val_loss'], color='orange', label='test')
    # plot accuracy
    pyplot.subplot(212)
    pyplot.title('Classification Accuracy')
    pyplot.plot(history.history['acc'], color='blue', label='train')
    pyplot.plot(history.history['val_acc'], color='orange', label='test')
    pyplot.tight_layout(pad=2.0)


summarize_diagnostics(res)

#Sommaire
model.summary()

#Sauvegarde du modele
model.save('modNN1.h5')


