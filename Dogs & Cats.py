# -*- coding: utf-8 -*-

from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense

# Initialising the CNN
classifier = Sequential()

# Convolution 
classifier.add(Convolution2D(32, 3, 3, input_shape = (64, 64, 3), activation = 'relu')) 
# In theanos backend 2nd array dimension, 1st channel(color = 3)
# Pooling
classifier.add(MaxPooling2D(pool_size = (2, 2)))

# Adding a 2nd Convolution layer
classifier.add(Convolution2D(32, 3, 3, activation = 'relu')) # to improve we can double up
# 32 to 64 to increase the accuracy or even to 256 but that would take a lot of time so better run it on GPU
classifier.add(MaxPooling2D(pool_size = (2, 2)))
# Flattening
classifier.add(Flatten())
# Full Connection
classifier.add(Dense(activation = 'relu', units = 128))
classifier.add(Dense(activation = 'sigmoid', units = 1))
# Compiling the CNN
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metric = ['accuracy'])

#Fitting the CNN to images
# Image augmnetation preprocesses images to avoid overfitting
from keras.preprocessing.image import ImageDataGenerator
train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)

training_set = train_datagen.flow_from_directory('dataset/training_set',
                                                target_size=(64, 64),   # Dimension expected by the CNN
                                                batch_size=32,
                                                class_mode='binary')

test_set = test_datagen.flow_from_directory('dataset/test_set',
                                            target_size=(64, 64),
                                            batch_size=32,
                                            class_mode='binary')

classifier.fit_generator(training_set,
                        steps_per_epoch=8000,  # number of  training Images
                        epochs=25,
                        validation_data=test_set,
                        validation_steps=2000)  # number of test images
''' you need change the epoch steps with respect to the images you have in training and test set
