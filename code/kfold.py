from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score
from keras.models import load_model
from sklearn.metrics import confusion_matrix
from sklearn.metrics import cohen_kappa_score
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense, Dropout
from keras.models import load_model,Model
import cv2
import os
import keras
import pandas as pd
import numpy as np
from keras.callbacks import EarlyStopping
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
#model=load_model("cnn_pre")
from sklearn.model_selection import KFold

dic={}
string1="ABCDEFGHIJKLMNOPQRSTUVWXYZ"
for i in range(0,len(string1)):
    dic[string1[i]]=i
train_x=[]
train_y=[]
for i in os.listdir("/Users/currentwire/Documents/train_set"):
  count=0
  if not i.startswith("."):

    for j in os.listdir("/Users/currentwire/Documents/train_set"+"/"+str(i)):
        count=count+1
        if count <100:
            ima=cv2.imread("/Users/currentwire/Documents/train_set"+"/"+str(i)+"/"+str(j))
            hsv_image = cv2.cvtColor(ima,cv2.COLOR_RGB2GRAY)
            hsv_image=cv2.resize(hsv_image,(28,28))
            train_x.append(hsv_image)
            train_y.append(dic[str(i)])

test_x=[]
test_y=[]
for i in os.listdir("/Users/currentwire/Documents/test_set"):
  if not i.startswith("."):
    count=0
    #print(i)
    for j in os.listdir("/Users/currentwire/Documents/test_set"+"/"+str(i)):
     count=count+1
     if count <50:
      ima=cv2.imread("/Users/currentwire/Documents/test_set"+"/"+str(i)+"/"+str(j))
      hsv_image = cv2.cvtColor(ima,cv2.COLOR_RGB2GRAY)
      hsv_image=cv2.resize(hsv_image,(28,28))
      test_x.append(hsv_image/255)
      test_y.append(dic[str(i)])
model = load_model("this_one")
y = keras.utils.np_utils.to_categorical(np.array(train_y))
train=np.array(train_x)/255
x = np.expand_dims(train, axis = -1)
y1 = keras.utils.np_utils.to_categorical(np.array(test_y))
x1 = np.expand_dims(np.array(test_x), axis = -1)
acc_per_fold=[]
loss_per_fold=[]
inputs = np.concatenate((x, x1), axis=0)
targets = np.concatenate((y, y1), axis=0)

# Define the K-fold Cross Validator
kfold = KFold(n_splits=5, shuffle=True)

# K-fold Cross Validation model evaluation
fold_no = 1
for train, test in kfold.split(inputs, targets):

  # Define the model architecture
  classifier = Sequential()

  # Step 1 - Convolutio Layer
  classifier.add(Conv2D(32, 3, 3, input_shape=(28, 28, 1), activation='relu', name="3x3_32_28_28"))

  # step 2 - Pooling
  classifier.add(MaxPooling2D(pool_size=(2, 2), name="Pool_size_2x2"))

  # Adding second convolution layer
  classifier.add(Conv2D(32, 3, 3, activation='relu', name="3x3_32"))
  classifier.add(MaxPooling2D(pool_size=(2, 2), name="Pool_size_2x2_1"))

  # Adding 3rd Concolution Layer
  classifier.add(Conv2D(64, 3, 3, activation='relu', name="3x3_64"))
  # classifier.add(Conv2D(64, 3,  3, activation = 'relu'))

  classifier.add(MaxPooling2D(pool_size=(2, 2), name="Pool_size_2x2_2"))

  # classifier.add(MaxPooling2D(pool_size =(2,2)))
  # Step 3 - Flattening
  classifier.add(Flatten())

  # Step 4 - Full Connection
  classifier.add(Dense(256, activation='relu'))
  classifier.add(Dropout(0.5))
  classifier.add(Dense(128, activation='relu'))
  classifier.add(Dropout(0.5))
  classifier.add(Dense(26, activation='softmax'))
  for i in range(4):
      classifier.layers[i].set_weights(model.layers[i].get_weights())

  # Compile the model
  classifier.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
  es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=10)

  # Generate a print
  print('------------------------------------------------------------------------')
  print(f'Training for fold {fold_no} ...')

  # Fit data to model
  history = classifier.fit(inputs[train], targets[train],
              epochs=100,
              verbose=1,
              validation_split=0.2,
              callbacks=[es])

  # Generate generalization metrics
  scores = classifier.evaluate(inputs[test], targets[test], verbose=0)
  print(f'Score for fold {fold_no}: {classifier.metrics_names[0]} of {scores[0]}; {classifier.metrics_names[1]} of {scores[1]*100}%')
  acc_per_fold.append(scores[1] * 100)
  loss_per_fold.append(scores[0])

  # Increase fold number
  fold_no = fold_no + 1

