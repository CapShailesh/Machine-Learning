#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 16 16:22:19 2020

@author: lenovo
"""

#import libraries
import os.path
import numpy as np
import tensorflow as tf
import ktrain
from ktrain import text

#import dataset
dataset = tf.keras.utils.get_file(fname="aclImdb_v1.tar.gz",
                                  origin="http://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz",
                                  extract=True)
IMDB_DATADIR = os.path.join(os.path.dirname(dataset), 'aclImdb')

#spliting dataset
(x_train, y_train), (x_test, y_test), preproc = text.texts_from_folder(datadir=IMDB_DATADIR,
                                                                       classes=['pos', 'neg'], 
                                                                       maxlen=500,
                                                                       train_test_names=['train', 'test'], 
                                                                       preprocess_mode='bert')

#Buildung the Bert Model
model = text.text_classifier(name='bert',
                             train_data=(x_train, y_train), 
                             preproc=preproc)



#Training The Bert Model
learner = ktrain.get_learner(model=model,
                            train_data=(x_train, y_train),
                            val_data=(x_test, y_test),
                            batch_size=6)


learner.fit_onecycle(lr=2e-5,
                     epochs=1)

'''
begin training using onecycle policy with max lr of 2e-05...
4167/4167 [==============================] - 3294s 790ms/step -
 loss: 0.2498 - accuracy: 0.8968 - val_loss: 0.1590 - val_accuracy: 0.9401
<tensorflow.python.keras.callbacks.History at 0x7f523c3e2048>
'''





