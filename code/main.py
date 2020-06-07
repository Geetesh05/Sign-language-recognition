import os
import cv2
#os.chdir('/content/drive/My Drive')
import string
import keras
import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense, Dropout
from keras.models import load_model,Model
from keras import optimizers
from keras.layers import LSTM
from keras.layers import TimeDistributed
from keras.layers import Conv1D
from keras.regularizers import l2
from cnnlstm import cnnlstm
from lstm import lstm
from cnn import cnn
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
#select a model
#cnn(x,y,x1,y1)
#lstm(train,y,np.array(test_x),y1)
cnnlstm(x,y,x1,y1)
