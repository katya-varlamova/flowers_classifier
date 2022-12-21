import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import numpy as np
from os import listdir
from os.path import join
import cv2
import pandas
import os
import random
data = "train/"
folders = os.listdir(data)

with open("order", 'w') as f:
    for fol in folders:
        f.write(fol + ',')
    f.write('\n')
    
image_names = []
train_labels = []
train_images = []

size = 64,64
for folder in folders:
    for file in os.listdir(os.path.join(data,folder)):
        if file.endswith("jpg"):
            image_names.append(os.path.join(data,folder,file))
            train_labels.append(folder)
            img = cv2.imread(os.path.join(data,folder,file))
            im = cv2.resize(img,size)
            train_images.append(im)
        else:
            continue
train = np.array(train_images)

# Reduce the RGB values between 0 and 1

train = train.astype('float32') / 255.0
# Extract the labels

label_dummies = pandas.get_dummies(train_labels)

labels =  label_dummies.values.argmax(1)


# Shuffle the labels and images randomly for better results

union_list = list(zip(train, labels))
random.shuffle(union_list)
train,labels = zip(*union_list)

# Convert the shuffled list to numpy array type

train = np.array(train)
labels = np.array(labels)

# Develop a sequential model using tensorflow keras

model = keras.Sequential([
    keras.layers.Flatten(input_shape=(64,64,3)),
    keras.layers.Dense(128, activation=tf.nn.tanh),
    keras.layers.Dense(5, activation=tf.nn.softmax)
])

model.compile(optimizer=tf.optimizers.Adam(), 
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])


model.fit(train, labels, epochs=5)
model.save('my_model.h5')
