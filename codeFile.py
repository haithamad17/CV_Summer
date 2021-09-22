# -*- coding: utf-8 -*-
"""
Created on Wed Aug 07 10:08:28 2021

@author: HAITHAM
"""
import cv2
import numpy as np 
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from keras.utils import np_utils
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, MaxPooling2D
from sklearn.metrics import classification_report
from tensorflow import keras
import os
from PIL import Image
import pandas as pd

class myCallBack(tf.keras.callbacks.Callback):
    def endOfEpoch(self,epoch,logs={}):
        if(logs.get('accuracy')>0.95 and logs.get('val_accuracy')>0.85):
            print("Stopped. Reached 90% accuracy")
            self.model.stop_training = True
            
fPath = 'C:\\Users\\HAITHAM\\Desktop\\PlantVillage'
training = []
labels = list(os.listdir('C:\\Users\\HAITHAM\\Desktop\\PlantVillage'))
print("Labels are: \n")
print(labels)
list_diseases = os.listdir('C:\\Users\\HAITHAM\\Desktop\\PlantVillage')
results2 = []
#Printinga table contains each label with quantity of images in it
for disease in list_diseases:
    dies_name_count = {}
    count_disease = len(os.listdir('C:\\Users\\HAITHAM\\Desktop\\PlantVillage'))
    dies_name_count['disease'] = disease
    dies_name_count['count_images'] = count_disease
    results2.append(dies_name_count)
results = pd.DataFrame(results2)
print(results)
im = Image.open('C:\\Users\\HAITHAM\\Desktop\\PlantVillage\\Tomato_Leaf_Mold/00694db7-3327-45e0-b4da-a8bb7ab6a4b7___Crnl_L.Mold 6923.JPG')

h,w = im.size
X = []
y = []

imgDim = 64

os.listdir(fPath)
for l in labels:
    img_path = os.path.join(fPath, l)
    class_num = labels.index(l)
    
    
    for img in os.listdir(img_path):
        try:
            img_array = cv2.imread(os.path.join(img_path, img))
            new_array = cv2.resize(img_array, (imgDim, imgDim))
            training.append([new_array, class_num])
        except:
            continue

for features, label in training:
    y.append(label)
    X.append(features)
    
X = np.array(X).reshape(-1, imgDim, imgDim, 3)
        
X = X.astype('float32')
X /= 255
Y = np_utils.to_categorical(y, 15)

#Train and test splitting data

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=0,shuffle=True)

model = Sequential([
    Conv2D(16, (3,3), activation='relu',input_shape=(64,64,3)),
    MaxPooling2D((2,2),strides=2),
    Conv2D(16, (3,3), activation='relu'),
    MaxPooling2D((2,2),strides=2),
    Flatten(),
    Dense(64,activation='relu'),
    Dense(15, activation='softmax')
    
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])        
callbacks = myCallBack()
#testing the splitted data
history = model.fit(X_train, y_train, batch_size= 16, epochs=20, verbose=1, validation_data=(X_test, y_test),callbacks=[callbacks])
model.save('./plantDieasesPrediction')

#plotting the accuracy and loss
plt.figure(0)
plt.plot(history.history['accuracy'], label="Training accuracy")
plt.plot(history.history['val_accuracy'], label="val accuracy")
plt.title("Accuracy")
plt.xlabel("epochs")
plt.ylabel("accuracy")
plt.legend()
plt.figure(1)
plt.plot(history.history['loss'], label="training loss")
plt.plot(history.history['val_loss'], label="val loss")
plt.title("Loss")
plt.xlabel("epochs")
plt.ylabel("Loss")
plt.legend()
plt.show()

score = model.evaluate(X_test, y_test, verbose = 1)
print("Test Score: ", score[0])
print("Test accuracy: ", score[1])

preds = np.round(model.predict(X_test),0)


pred = model.predict(X_test, batch_size = 32)
pred = np.argmax(pred, axis=1)
y_target = np.argmax(y_test, axis=1)
print(classification_report(y_target, pred, target_names = labels))

"""
-----------------------------------------------------------------------------
"""



#Insert you input image path to predict it's disease
test_img_path = 'C:\\Users\\HAITHAM\\Desktop\\test\\tomato-septoria-leaf-spot-grabowski.JPG'
img = keras.preprocessing.image.load_img(
    test_img_path, target_size=(64, 64)
)
img_array = keras.preprocessing.image.img_to_array(img)
img_array = tf.expand_dims(img_array, 0) # Create a batch

predictions = model.predict(img_array)
score = tf.nn.softmax(predictions[0])

print(
    "This image most likely belongs to {} "
    .format(labels[np.argmax(score)])
)