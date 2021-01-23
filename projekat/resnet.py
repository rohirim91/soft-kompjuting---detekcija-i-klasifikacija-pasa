import numpy as np
from keras.preprocessing import image
from keras_applications import resnet50
from keras_preprocessing.image import ImageDataGenerator
from tqdm import tqdm
from keras.applications.resnet50 import preprocess_input
from tensorflow.keras.applications import resnet50, ResNet50

ResNet50_model = ResNet50(input_shape=(224, 224,3), include_top=True, weights="imagenet")


def path_to_tensor(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    return np.expand_dims(img, axis=0)


def paths_to_tensor(img_paths):
    list_of_tensors = [path_to_tensor(img_path) for img_path in tqdm(img_paths)]
    return np.vstack(list_of_tensors)


def ResNet50_predict_labels(img_path):
    img = preprocess_input(path_to_tensor(img_path))
    return ResNet50_model.predict(img)


def dog_detector(img_path):
    prediction = ResNet50_predict_labels(img_path)
    cnum = np.argmax(prediction)
    print((cnum <= 268) & (cnum >= 151))
    return resnet50.decode_predictions(prediction, top=1)[0][0][1]


def is_dog(img):
    img = np.expand_dims(img, axis=0)
    prediction = ResNet50_model.predict(img)
    cnum = np.argmax(prediction)
    prediction = resnet50.decode_predictions(prediction, top=1)[0][0]
    return [(cnum <= 268) & (cnum >= 151), prediction[2]]