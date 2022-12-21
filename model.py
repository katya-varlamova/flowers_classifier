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
## singleton code for classification of one image
## it works!!
MODEL_FILENAME = 'my_model.h5'
ORDER_FILENAME = 'order'

class ModelWrapper:
    __instance = None
    def __init__(self):
        if ModelWrapper.__instance is not None:
            raise Exception("This class is a singleton!")
        ModelWrapper.__instance = self
        self.model = tf.keras.models.load_model(MODEL_FILENAME)
        with open(ORDER_FILENAME) as f:
            self.order_tags = f.readline().split(',')[:5]
            
    @staticmethod
    def get_shared():
        if ModelWrapper.__instance is None:
            ModelWrapper()
        return ModelWrapper.__instance
    
    def predict(self, filename):
        size = 64,64
        img = cv2.imread(filename)
        im = np.array([cv2.resize(img, size)])
        im = im.astype('float32') / 255.0
        poss = self.model.predict(im)
        ind = np.argmax(poss)
        return self.order_tags[ind]

## how to use it
if __name__ == "__main__":
    model = ModelWrapper.get_shared()
    print(model.predict("test/Image_1.jpg"))


# old code with classification of several images

##model = tf.keras.models.load_model('my_model.h5')
##size = 64,64
##image_names = []
##images = []
##data = "test1/"
##for file in os.listdir(data):
##    if file.endswith("jpg"):
##        img = cv2.imread(os.path.join(data,file))
##        im = cv2.resize(img, size)
##        images.append(im)
##    else:
##        continue
##test = np.array(images)
##
##
##test = test.astype('float32') / 255.0
##
##print(model.predict(test))


