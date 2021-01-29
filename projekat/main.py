import cv2
import matplotlib.pyplot as plt

from tensorflow.keras.models import load_model
from utils import nms, util, sliding

labels = ["samoyed", "leonberg", "basenji", "rottweiler", "kerry blue"]


def main():
    segmentation_enabled = True
    selective_enabled = True
    cnn = load_model("cnn_161_944.hdf5")

    test_image = util.load_image('dogs/test/blue-test4.png')
    (img_h, img_w) = test_image.shape[:2]
    ratio = img_w / img_h

    if img_w < 448:
        test_image = cv2.resize(test_image, (448, int(448 / ratio)), interpolation=cv2.INTER_CUBIC)
    if img_h < 448:
        test_image = cv2.resize(test_image, (int(448 * ratio), 448), interpolation=cv2.INTER_CUBIC)
    (img_h, img_w) = test_image.shape[:2]

    if segmentation_enabled:
        if selective_enabled:
            ss = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()
            ss.setBaseImage(test_image)
            ss.switchToSelectiveSearchQuality()
            rects = ss.process()
            windows, scores, classes = util.process_image_rects(cnn, test_image, img_h, img_w, rects)
        else:
            windows, scores, classes = sliding.process_image_sliding(cnn, test_image, img_h, img_w)

        pick = nms.non_max_suppression(windows, probs=scores, overlap_thresh=0.03)
        windows = windows[pick]
        classes = classes[pick]
        scores = scores[pick]

        i = 0
        for startX, startY, endX, endY in windows:
            cv2.putText(test_image, labels[classes[i] - 1] + "{:6.3f}".format(scores[i]),
                        (startX + 5, startY + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                        (36, 255, 12), 2)
            cv2.rectangle(test_image, (startX, startY), (endX, endY), (0, 255, 0), 2)
            i += 1
    else:
        score, class_idx = util.process_whole_image(cnn, test_image)[1:2]
        cv2.putText(test_image, labels[class_idx] + "{:6.3f}".format(score),
                    (0 + 5, 0 + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    (36, 255, 12), 2)
        cv2.rectangle(test_image, (0, 0), (img_w, img_h), (0, 255, 0), 2)

    test_image = cv2.cvtColor(test_image, cv2.COLOR_BGR2RGB)
    plt.imshow(test_image)
    plt.show()


main()