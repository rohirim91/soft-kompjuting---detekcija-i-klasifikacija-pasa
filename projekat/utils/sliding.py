import cv2
import numpy as np
from utils import constants, util


def process_image_sliding(cnn, image, img_h, img_w, min_score=0.9):
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
                    prediction = util.classify_rect(cnn, roi)
                    if prediction[0] > min_score and prediction[1] != 0:
                        best_rects.append([x, y, x + size_x, y + size_y])
                        best_scores.append(prediction[0])
                        best_classes.append(prediction[1])

    return np.array(best_rects), np.array(best_scores), np.array(best_classes)
