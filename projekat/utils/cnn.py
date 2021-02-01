import os
import cv2
import numpy as np

from sklearn.model_selection import train_test_split
from tensorflow import keras
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from tensorflow.python.keras.callbacks import ModelCheckpoint

from projekat.utils import constants, util


def create_cnn():
    cnn = Sequential()

    cnn.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(224, 224, 3)))
    cnn.add(MaxPooling2D((2, 2)))
    cnn.add(Dropout(0.25))

    cnn.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
    cnn.add(MaxPooling2D(pool_size=(2, 2)))
    cnn.add(Dropout(0.25))

    cnn.add(Conv2D(256, kernel_size=(3, 3), activation='relu'))
    cnn.add(MaxPooling2D(pool_size=(2, 2)))
    cnn.add(Dropout(0.25))

    cnn.add(Conv2D(256, kernel_size=(3, 3), activation='relu'))
    cnn.add(MaxPooling2D(pool_size=(2, 2)))
    cnn.add(Dropout(0.25))

    cnn.add(Conv2D(256, kernel_size=(3, 3), activation='relu'))
    cnn.add(MaxPooling2D(pool_size=(2, 2)))
    cnn.add(Dropout(0.25))

    cnn.add(Conv2D(256, kernel_size=(3, 3), activation='relu'))
    cnn.add(MaxPooling2D(pool_size=(2, 2)))
    cnn.add(Dropout(0.25))

    cnn.add(Flatten())

    cnn.add(Dense(256, activation='relu'))
    cnn.add(Dropout(0.5))

    cnn.add(Dense(6, activation='softmax'))

    cnn.compile(loss=keras.losses.categorical_crossentropy,
                 optimizer=keras.optimizers.Adam(),
                 metrics=['accuracy'])
    return cnn


def train_cnn():
    images = []
    labels = []

    temp = []
    temp_labels = []
    for img_name in os.listdir("dogs/train/blue_terrier/"):
        img_path = os.path.join("dogs/train/blue_terrier/", img_name)
        img = util.load_image(img_path)
        img = cv2.resize(img, constants.IMG_DIMENSIONS, interpolation=cv2.INTER_CUBIC)
        temp.append(img)
        temp_labels.append(5)
    images.extend(temp)
    labels.extend(temp_labels)

    temp = []
    temp_labels = []
    for img_name in os.listdir("dogs/train/rottweiler/"):
        img_path = os.path.join("dogs/train/rottweiler/", img_name)
        img = util.load_image(img_path)
        img = cv2.resize(img, constants.IMG_DIMENSIONS, interpolation=cv2.INTER_CUBIC)
        images.append(img)
        labels.append(4)
    images.extend(temp)
    labels.extend(temp_labels)

    temp = []
    temp_labels = []
    for img_name in os.listdir("dogs/train/basenji/"):
        img_path = os.path.join("dogs/train/basenji/", img_name)
        img = util.load_image(img_path)
        img = cv2.resize(img, constants.IMG_DIMENSIONS, interpolation=cv2.INTER_CUBIC)
        images.append(img)
        labels.append(3)
    images.extend(temp)
    labels.extend(temp_labels)

    temp = []
    temp_labels = []
    for img_name in os.listdir("dogs/train/leonberg/"):
        img_path = os.path.join("dogs/train/leonberg/", img_name)
        img = util.load_image(img_path)
        img = cv2.resize(img, constants.IMG_DIMENSIONS, interpolation=cv2.INTER_CUBIC)
        images.append(img)
        labels.append(2)
    images.extend(temp)
    labels.extend(temp_labels)

    temp = []
    temp_labels = []
    for img_name in os.listdir("dogs/train/samoyed/"):
        img_path = os.path.join("dogs/train/samoyed/", img_name)
        img = util.load_image(img_path)
        img = cv2.resize(img, constants.IMG_DIMENSIONS, interpolation=cv2.INTER_CUBIC)
        images.append(img)
        labels.append(1)
    images.extend(temp)
    labels.extend(temp_labels)

    temp = []
    temp_labels = []
    for img_name in os.listdir(constants.TRAIN_NEGATIVE):
        img_path = os.path.join(constants.TRAIN_NEGATIVE, img_name)
        img = util.load_image(img_path)
        img = cv2.resize(img, constants.IMG_DIMENSIONS, interpolation=cv2.INTER_CUBIC)
        images.append(img)
        labels.append(0)
    images.extend(temp)
    labels.extend(temp_labels)

    images = np.array(images)
    labels = np.array(labels)

    x_train = images.reshape(images.shape[0], 224, 224, 3)
    x_train = x_train.astype('float32')
    x_train /= 255

    y_train = to_categorical(np.array(labels))
    x_train, x_valid, y_train, y_valid = train_test_split(x_train, y_train, test_size=0.1, random_state=42)

    print('Train shape: ', x_train.shape, y_train.shape)
    print('Test shape: ', x_valid.shape, y_valid.shape)

    cnn = create_cnn()

    checkpointer = ModelCheckpoint(filepath="cnn_best.hdf5", verbose=1, save_best_only=True)

    cnn.fit(x_train, y_train, validation_data=(x_valid, y_valid), epochs=30, batch_size=32, callbacks=[checkpointer], verbose=1, shuffle=True)
    cnn.save("cnn.hdf5")
