import cv2
import numpy as np

from projekat import resnet
from projekat.utils import constants


def load_image(path):
    return cv2.imread(path)


def classify_rect(cnn, image):
    image = np.array([image])
    image = image.reshape(image.shape[0], 224, 224, 3)
    image = image.astype('float32')
    image /= 255

    prediction = cnn.predict(image)
    return np.max(prediction), np.argmax(prediction, axis=1)[0]


def process_image_rects(cnn, image, img_h, img_w, rects, min_score=0.9, resnet_enabled=False):
    best_rects = []
    best_scores = []
    best_classes = []

    for (x, y, w, h) in rects:
        if w / img_w < 0.15 or h / img_h < 0.15 \
                or w / img_w > 0.9 or h / img_h > 0.9 \
                or w / h < 0.2 or h / w < 0.2:
            continue
        roi = image[y:y + h, x:x + w]
        roi = cv2.resize(roi, constants.IMG_DIMENSIONS, interpolation=cv2.INTER_CUBIC)

        if resnet_enabled:
            prediction = resnet.is_dog(roi)
        else:
            prediction = classify_rect(cnn, roi)

        if prediction[0] > min_score and (prediction[1] != 0 and prediction[1]):
            best_rects.append([x, y, x + w, y + h])
            best_scores.append(prediction[0])
            best_classes.append(prediction[1])

    return np.array(best_rects), np.array(best_scores), np.array(best_classes)


def process_whole_image(cnn, image):
    image = cv2.resize(image, constants.IMG_DIMENSIONS, interpolation=cv2.INTER_CUBIC)
    prediction = classify_rect(cnn, image)
    return prediction[0], prediction[1]
