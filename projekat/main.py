import numpy as np
import cv2
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
from utils.nms import non_max_suppression
from utils import constants

labele = ["samoyed", "leonberg", "basenji", "rottweiler", "kerry blue"]
img_w = 0
img_h = 0

test_dir = 'dogs/test/'

pos_imgs = []
neg_imgs = []
test_imgs = []


def load_image(path):
    return cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2GRAY)


def load_image_color(path):
    return cv2.imread(path)


def load_image_color_rgb(path):
    return cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)

# transformisemo u oblik pogodan za scikit-learn
def reshape_data(input_data):
    nsamples, nx, ny = input_data.shape
    return input_data.reshape((nsamples, nx * ny))


def classify_rect(image):
    image = np.array([image])
    image = image.reshape(image.shape[0], 224, 224, 3)
    image = image.astype('float32')
    image /= 255

    prediction = clf_cnn.predict(image)
    return np.max(prediction), np.argmax(prediction, axis=1)[0]


def process_image_sliding(image, min_score=0.9):
    best_rects = []
    best_scores = []
    best_classes = []

    min_dim = min(img_h, img_w)
    max_dim = max(img_h, img_w)
    dim_ratio = min_dim / max_dim

    sizes = [[int(img_w * 0.20), int(img_h * 0.20)],
             [int(img_w * 0.30), int(img_h * 0.30)],
             [int(img_w * 0.40), int(img_h * 0.40)],
             [int(img_w * 0.50), int(img_h * 0.50)],
             [int(img_w * 0.60), int(img_h * 0.60)],
             [int(img_w * 0.90), int(img_h * 0.90)],
             [int(img_h * 0.20 * dim_ratio), int(img_h * 0.20)],
             [int(img_h * 0.30 * dim_ratio), int(img_h * 0.30)],
             [int(img_h * 0.40 * dim_ratio), int(img_h * 0.40)],
             [int(img_h * 0.50 * dim_ratio), int(img_h * 0.50)],
             [int(img_h * 0.60 * dim_ratio), int(img_h * 0.60)],
             [int(img_h * 0.90 * dim_ratio), int(img_h * 0.90)]]
    step_size_h = int(img_h * 0.05)
    step_size_w = int(img_w * 0.05)

    for size_x, size_y in sizes:
        for y in range(0, img_h, step_size_h):
            for x in range(0, img_w, step_size_w):
                roi = image[y:y + size_y, x:x + size_x]
                if roi.shape[0:2] == (size_y, size_x):
                    roi = cv2.resize(roi, constants.IMG_DIMENSIONS, interpolation=cv2.INTER_CUBIC)
                    prediction = classify_rect(roi)
                    if prediction[0] > min_score and prediction[1] != 0:
                        best_rects.append([x, y, x + size_x, y + size_y])
                        best_scores.append(prediction[0])
                        best_classes.append(prediction[1])

    return np.array(best_rects), np.array(best_scores), np.array(best_classes)


def process_image(image, rects, min_score=0.9996):
    best_rects = []
    best_scores = []
    best_classes = []

    for (x, y, w, h) in rects:
        # necemo: mnogo malo, izrazito neravnomernih dimenzija
        if w / img_w < 0.15 or h / img_h < 0.15 \
                or w / img_w > 0.9 or h / img_h > 0.9 \
                or w / h < 0.2 or h / w < 0.2:
            continue
        roi = image[y:y + h, x:x + w]
        roi = cv2.resize(roi, constants.IMG_DIMENSIONS, interpolation=cv2.INTER_CUBIC)
        prediction = classify_rect(roi)
        if prediction[0] > min_score and prediction[1] != 0:
            best_rects.append([x, y, x + w, y + h])
            best_scores.append(prediction[0])
            best_classes.append(prediction[1])

    return np.array(best_rects), np.array(best_scores), np.array(best_classes)


def process_whole_image(image):
    image = cv2.resize(image, constants.IMG_DIMENSIONS, interpolation=cv2.INTER_CUBIC)
    prediction = classify_rect(image)
    return prediction[0], prediction[1]

if __name__ == '__main__':
    segmentation_enabled = True
    selective_enabled = False

    # ucitamo model i sliku
    #train_cnn()

    clf_cnn = load_model("cnn_161_944.hdf5")

    itest = load_image_color('dogs/test/blue-test4.png')
    (img_h, img_w) = itest.shape[:2]
    ratio = img_w / img_h

    # garantujemo da slika nece biti minijaturna tokom obrade
    if (img_w < 448):
        itest = cv2.resize(itest, (448, int(448 / ratio)), interpolation=cv2.INTER_CUBIC)
    if (img_h < 448):
        itest = cv2.resize(itest, (int(448 * ratio), 448), interpolation=cv2.INTER_CUBIC)
    (img_h, img_w) = itest.shape[:2]

    if segmentation_enabled:
        if selective_enabled:
            # selective search nam da predloge objekata
            ss = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()
            ss.setBaseImage(itest)
            ss.switchToSelectiveSearchQuality()
            rects = ss.process()

            windows, scores, classes = process_image(itest, rects)
        else:
            windows, scores, classes = process_image_sliding(itest)

        # odbacimo preklapajuce predloge
        pick = non_max_suppression(windows, probs=scores, overlapThresh=0.03)
        windows = windows[pick]
        classes = classes[pick]
        scores = scores[pick]
        print(scores)
        print(classes)

        # oznacavamo predloge na slici
        i = 0
        for startX, startY, endX, endY in windows:
            cv2.putText(itest, labele[classes[i] - 1] + "{:6.3f}".format(scores[i]),
                        (startX + 5, startY + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                        (36, 255, 12), 2)
            cv2.rectangle(itest, (startX, startY), (endX, endY), (0, 255, 0), 2)
            i += 1
    else:
        score, class_idx = process_whole_image(itest)[1:2]
        cv2.putText(itest, labele[class_idx] + "{:6.3f}".format(score),
                    (0 + 5, 0 + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    (36, 255, 12), 2)
        cv2.rectangle(itest, (0, 0), (img_w, img_h), (0, 255, 0), 2)

    # prikazujemo sliku korisniku
    itest = cv2.cvtColor(itest, cv2.COLOR_BGR2RGB)
    plt.imshow(itest)
    plt.show()
