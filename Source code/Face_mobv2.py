# -*- coding: utf-8 -*-
"""
Created on Tue Oct 19 07:15:29 2021

@author: sas11
"""
from tensorflow import keras
import tensorflow as tf # Imports tensorflow



from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Flatten, Dropout, BatchNormalization
from tensorflow.keras.layers import Conv2D, MaxPooling2D,LSTM
from tensorflow.keras import regularizers, optimizers
from tensorflow.keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from tensorflow.keras.utils import plot_model
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import os
import cv2
from glob import glob
from tensorflow.keras.metrics import Recall,Precision,AUC,TruePositives,TrueNegatives,FalseNegatives,FalsePositives
from tensorflow.keras.utils import plot_model

img_shape = (96, 96, 3)
data=[]
data_dir='dataset'
Name = "Face_Mobv2"
lebel = []
rel_dirname = os.path.dirname(__file__)
    
for dirname in os.listdir(os.path.join(rel_dirname, data_dir)):
        for filename in glob(os.path.join(rel_dirname, data_dir+'/'+dirname+'/*.jpg')):
             img = image.load_img(os.path.join(rel_dirname, filename),target_size=img_shape)
             img = image.img_to_array(img)
             img = img/255.0
             data.append(img)
             lebel.append(dirname)
X = np.array(data)
lebel = np.array(lebel)
y=to_categorical(lebel)
for i in range (7):
    print(str(i)+" class has: "+str(np.count_nonzero(y[:,i])))
    
print(X.shape)
print(y.shape)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=48, test_size=0.2)

base_model = tf.keras.applications.MobileNetV2(input_shape=img_shape,
                                               include_top=False,
                                               weights='imagenet',
                                               classifier_activation='softmax')
base_model.trainable = True

print("Number of layers in the base model: ", len(base_model.layers))
fine_tune_at = 50
for layer in base_model.layers[:fine_tune_at]:
  layer.trainable =  False

model = tf.keras.Sequential([
  base_model,
  tf.keras.layers.GlobalAveragePooling2D(),
  tf.keras.layers.Dense(7, activation='softmax')
])


model.summary()
model.compile(optimizer= keras.optimizers.Adam(),
              loss=keras.losses.CategoricalCrossentropy(from_logits=True), 
              metrics=['acc',Recall(),Precision(),AUC(),
                       TruePositives(),TrueNegatives(),FalseNegatives(),FalsePositives()])
plot_model(model, to_file=Name+'.png',show_shapes= True , show_layer_names=True)
history = model.fit(X_train, y_train, epochs=100, validation_data=(X_test, y_test), batch_size=64)

plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='lower left')
plt.savefig(Name+'acc.png')
plt.show()

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.savefig(Name+'loss.png')
plt.show()



model.save(Name+'.h5')

pd.DataFrame.from_dict(history.history).to_csv(Name+'.csv',index=False)
