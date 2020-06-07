import pandas as pd
import cv2
import numpy as np
import keras
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from keras.models import Sequential


def string_to_array(string):
        pixels = [int(x) for x in string[1:]]
        pixels = np.array(pixels)
        pixels = pixels.reshape((28, 28))
        return pixels


def normalize(array):
        return array / 255


def make_xy(df):
        x = []
        y = []

        for row in df.iterrows():
                alphabets = row[1]['label']
                array = string_to_array(row[1])
                x.append(array)
                y.append(alphabets)

        x = np.array(x)
        y = np.array(y)
        y = keras.utils.np_utils.to_categorical(y)
        x = normalize(x)
        x = np.expand_dims(x, axis=-1)

        return x, y

def train(x,y,x1,y1):
        classifier = Sequential()
        classifier.add(Conv2D(32, 3, 3, input_shape=(28, 28, 1), activation='relu', name="3x3_32_28_28"))
        classifier.add(MaxPooling2D(pool_size=(2, 2), name="Pool_size_2x2"))
        classifier.add(Conv2D(32, 3, 3, activation='relu', name="3x3_32"))
        classifier.add(MaxPooling2D(pool_size=(2, 2), name="Pool_size_2x2_1"))
        classifier.add(Conv2D(64, 3, 3, activation='relu', name="3x3_64"))
        classifier.add(MaxPooling2D(pool_size=(2, 2), name="Pool_size_2x2_2"))
        classifier.add(Flatten())
        classifier.add(Dense(256, activation='relu'))
        classifier.add(Dropout(0.5))
        classifier.add(Dense(25, activation='softmax'))
        classifier.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        history = classifier.fit(x, y, epochs=100, validation_data=(x1, y1))


image=pd.read_csv("sign_mnist_train.csv")
image2=pd.read_csv("sign_mnist_test.csv")
x, y = make_xy(image)
x1,y1=make_xy(image2)
train(x,y,x1,y1)