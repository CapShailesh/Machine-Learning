#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 19 10:59:29 2020

@author: lenovo
"""

#Importing the libraries

import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator
print(tf.__version__)


#Part 1 - Data Preprocessing / Image Augmentation

#Preprocessing the Training set
train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)

training_set = train_datagen.flow_from_directory('dataset/training_set',
                                                 target_size = (64, 64),
                                                 batch_size = 32,
                                                 class_mode = 'binary')

#Preprocessing the Test set
test_datagen = ImageDataGenerator(rescale = 1./255)

test_set = test_datagen.flow_from_directory('dataset/test_set',
                                            target_size = (64, 64),
                                            batch_size = 32,
                                            class_mode = 'binary')

#Building the CNN

#Initialising the CNN
cnn = tf.keras.models.Sequential()

#Step 1 - Convolution
cnn.add(tf.keras.layers.Conv2D(filters=32,
                               kernel_size=3,
                               input_shape=[64, 64, 3],
                               activation = 'relu'))


#Step 2 - Pooling
cnn.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2))

#Adding a second convolutional layer
cnn.add(tf.keras.layers.Conv2D(filters=32,
                               kernel_size=3,
                               activation = 'relu'))


cnn.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2))


#Step 3 - Flattening
cnn.add(tf.keras.layers.Flatten())

#Step 4 - Full Connection
cnn.add(tf.keras.layers.Dense(units=128,
                              activation='relu',
                              ))

#Step 5 - Output Layer
cnn.add(tf.keras.layers.Dense(units=1,
                              activation='sigmoid',
                              ))
#Part 3 - Training the CNN

#Compiling the CNN
cnn.compile(optimizer='adam',
            loss='binary_crossentropy',
            metrics= ['accuracy'])

#Training the CNN on the Training set and evaluating it on the Test set
cnn.fit(x=training_set,
        validation_data=test_set,
        epochs=25)



#Part 4 - Making a single prediction

import numpy as np
from keras.preprocessing import image
test_image = image.load_img('dataset/single_prediction/cat_or_dog_5.jpeg',
                            target_size=(64, 64))

test_image = image.img_to_array(test_image)

test_image = np.expand_dims(test_image,
                            axis=0)

result = cnn.predict(test_image)

print(training_set.class_indices)

if result[0][0] == 0:
    prediction = 'Cat'
else:
    prediction = 'Dog'
    
print(prediction)

#Output of Training
'''
Epoch 25/25
250/250 [==============================] - 47s 189ms/step - 
loss: 0.2394 - accuracy: 0.9019 - val_loss: 0.5150 - val_accuracy: 0.7995
'''




