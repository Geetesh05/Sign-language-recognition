import os
import cv2
import keras
import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense, Dropout
from keras.models import load_model,Model
from keras import optimizers
from keras.layers import LSTM
from keras.layers import TimeDistributed
from keras.layers import Conv1D
from keras.callbacks import EarlyStopping
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
def cnnlstm(x,y,x1,y1):
    model = Sequential()
    model.add(TimeDistributed(Conv1D(32, 3,activation='relu'), input_shape=(28,28,1)))
    model.add(TimeDistributed(Dropout(0.2)))
    model.add(TimeDistributed(Conv1D(64, 3, activation='relu')))
    model.add(TimeDistributed(Dropout(0.2)))
    model.add(TimeDistributed(Conv1D(128, 3, activation='relu')))
    model.add(TimeDistributed(Dropout(0.2)))
    model.add(TimeDistributed(Flatten()))
    model.add(LSTM(64))
    model.add(Dropout(0.2))
    model.add(Dense(26, activation = 'softmax'))
    es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=10)
    model.compile(optimizer='adam', loss='categorical_crossentropy',metrics=['accuracy'])
    model.fit(x,y,epochs=100,validation_data=(x1,y1),callbacks=[es])
    model.save("model_cnnlstm")
    yhat_classes = model.predict_classes(x1, verbose=0)
    yhat_classes = keras.utils.np_utils.to_categorical(yhat_classes)
    accuracy = accuracy_score(y1, yhat_classes)
    print('Accuracy: %f' % accuracy)
    # precision tp / (tp + fp)
    precision = precision_score(y1, yhat_classes, average="macro")
    print('Precision: %f' % precision)
    # recall: tp / (tp + fn)
    recall = recall_score(y1, yhat_classes, average="macro")
    print('Recall: %f' % recall)
    # f1: 2 tp / (2 tp + fp + fn)
    f1 = f1_score(y1, yhat_classes, average="macro")
    print('F1 score: %f' % f1)
    conf_matrix = confusion_matrix(np.argmax(y1, axis=1), np.argmax(y_pred_classes, axis=1), normalize='true')
    df_conf = pd.DataFrame(conf_matrix, index=[i for i in string1],
                           columns=[i for i in string1])

    plt.figure(figsize=(10, 10))
    sns.heatmap(df_conf, annot=True, cmap='Blues', fmt='.2', cbar=False)
    plt.savefig("confuse_cnnlstm.jpg")