import os

import numpy as np
import cv2
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D

from tensorflow import keras

from utils import constants

def load_image_color(path):
    return cv2.imread(path)


def create_cnn():
    cnn3 = Sequential()

    cnn3.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(224, 224, 3)))
    cnn3.add(MaxPooling2D((2, 2)))
    cnn3.add(Dropout(0.2))

    cnn3.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
    cnn3.add(MaxPooling2D(pool_size=(2, 2)))
    cnn3.add(Dropout(0.2))

    cnn3.add(Conv2D(256, kernel_size=(3, 3), activation='relu'))
    cnn3.add(MaxPooling2D(pool_size=(2, 2)))
    cnn3.add(Dropout(0.2))

    cnn3.add(Conv2D(256, kernel_size=(3, 3), activation='relu'))
    cnn3.add(MaxPooling2D(pool_size=(2, 2)))
    cnn3.add(Dropout(0.2))

    cnn3.add(Conv2D(256, kernel_size=(3, 3), activation='relu'))
    cnn3.add(MaxPooling2D(pool_size=(2, 2)))
    cnn3.add(Dropout(0.2))

    cnn3.add(Flatten())

    cnn3.add(Dense(256, activation='relu'))
    cnn3.add(Dropout(0.5))

    cnn3.add(Dense(6, activation='softmax'))

    cnn3.compile(loss=keras.losses.categorical_crossentropy,
                 optimizer=keras.optimizers.Adam(),
                 metrics=['accuracy'])
    return cnn3


def train_cnn():
    imgs = []
    labels = []

    temp = []
    temp_labels = []
    for img_name in os.listdir("dogs/train/blue_terrier/"):
        img_path = os.path.join("dogs/train/blue_terrier/", img_name)
        img = load_image_color(img_path)
        img = cv2.resize(img, constants.IMG_DIMENSIONS, interpolation=cv2.INTER_CUBIC)
        temp.append(img)
        temp_labels.append(5)
    imgs.extend(temp)
    labels.extend(temp_labels)

    temp = []
    temp_labels = []
    for img_name in os.listdir("dogs/train/rottweiler/"):
        img_path = os.path.join("dogs/train/rottweiler/", img_name)
        img = load_image_color(img_path)
        img = cv2.resize(img, constants.IMG_DIMENSIONS, interpolation=cv2.INTER_CUBIC)
        imgs.append(img)
        labels.append(4)
    imgs.extend(temp)
    labels.extend(temp_labels)

    temp = []
    temp_labels = []
    for img_name in os.listdir("dogs/train/basenji/"):
        img_path = os.path.join("dogs/train/basenji/", img_name)
        img = load_image_color(img_path)
        img = cv2.resize(img, constants.IMG_DIMENSIONS, interpolation=cv2.INTER_CUBIC)
        imgs.append(img)
        labels.append(3)
    imgs.extend(temp)
    labels.extend(temp_labels)

    temp = []
    temp_labels = []
    for img_name in os.listdir("dogs/train/leonberg/"):
        img_path = os.path.join("dogs/train/leonberg/", img_name)
        img = load_image_color(img_path)
        img = cv2.resize(img, constants.IMG_DIMENSIONS, interpolation=cv2.INTER_CUBIC)
        imgs.append(img)
        labels.append(2)
    imgs.extend(temp)
    labels.extend(temp_labels)

    temp = []
    temp_labels = []
    for img_name in os.listdir("dogs/train/samoyed/"):
        img_path = os.path.join("dogs/train/samoyed/", img_name)
        img = load_image_color(img_path)
        img = cv2.resize(img, constants.IMG_DIMENSIONS, interpolation=cv2.INTER_CUBIC)
        imgs.append(img)
        labels.append(1)
    imgs.extend(temp)
    labels.extend(temp_labels)

    temp = []
    temp_labels = []
    for img_name in os.listdir(constants.TRAIN_NEGATIVE):
        img_path = os.path.join(constants.TRAIN_NEGATIVE, img_name)
        img = load_image_color(img_path)
        img = cv2.resize(img, constants.IMG_DIMENSIONS, interpolation=cv2.INTER_CUBIC)
        imgs.append(img)
        labels.append(0)
    imgs.extend(temp)
    labels.extend(temp_labels)

    imgs = np.array(imgs)
    labels = np.array(labels)
    print(imgs.shape)

    x_train = imgs.reshape(imgs.shape[0], 224, 224, 3)
    x_train = x_train.astype('float32')
    x_train /= 255

    y_train = to_categorical(np.array(labels))
    x_train, x_valid, y_train, y_valid = train_test_split(x_train, y_train, test_size=0.1, random_state=42)
    print('Train shape: ', x_train.shape, y_train.shape)
    print('Test shape: ', x_valid.shape, y_valid.shape)

    cnn = create_cnn()

    cnn.fit(x_train, y_train, validation_data=(x_valid, y_valid), epochs=9, batch_size=32, verbose=1, shuffle=True)
    cnn.save("cnn.hdf5")

    pred_classes = np.argmax(cnn.predict(x_valid), axis=1)
    test_cases_num = len(x_valid)
    matches_num = np.sum(y_valid == pred_classes)
    success = float(matches_num) / test_cases_num
    success_percentage = success * 100
    print("Number of test cases: %d" % test_cases_num)
    print("Number of matches: %d" % matches_num)
    print("Success: %f%%" % success_percentage)