#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 27 18:31:08 2018

@author: adlythebaud
"""

from sklearn.model_selection import train_test_split
from sklearn import datasets
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import numpy as np

# 1. import tensorflow and keras.
import tensorflow as tf
from tensorflow import keras

# get the clothing images (MNIST) from keras datasets
fashion_mnist = keras.datasets.fashion_mnist

# split into training and testing images and labels
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

# store class names
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

# show an image
def showImage(image):    
    plt.figure()
    plt.imshow(image)
    plt.colorbar()
    plt.grid(False)

#showImage(train_images[857])
    
# missing values? There are none. But what would we do next time?

#show a couple images to make sure everything is in correct format.
#plt.figure(figsize=(10,10))

#for i in range(25):
#    plt.subplot(5,5,i+1)
#    plt.xticks([])
#    plt.yticks([])
#    plt.grid(False)
#    plt.imshow(train_images[i], cmap=plt.cm.binary)
#    plt.xlabel(class_names[train_labels[i]])

# normalize pixel data from 0-255 to 0-1
#train_images = train_images / 255.0
#train_labels = train_labels / 255.0

# set up the model and layers
# this is our Multi-Layer Perceptron (per instructions I was following along)
model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),
    keras.layers.Dense(128, activation="relu"),
    keras.layers.Dense(10, activation="relu")
])

model.compile(optimizer=tf.train.AdamOptimizer(), 
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])    

# fit data to model
model.fit(train_images, train_labels, epochs=5)

# compare how model does on test data set
test_loss, test_acc = model.evaluate(test_images, test_labels)
print('Test accuracy:', test_acc)

# make predictions:
predictions = model.predict(test_images)
