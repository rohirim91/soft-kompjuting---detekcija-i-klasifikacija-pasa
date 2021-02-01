import os
import cv2
from projekat.utils import constants
from bs4 import BeautifulSoup
from imutils import paths
from projekat.utils import compute_iou
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator


def init_dataset():
    for dir_path in (constants.TRAIN_POSITIVE, constants.TRAIN_NEGATIVE):
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)

    img_paths = list(paths.list_images(constants.ORIG_IMAGES))
    total_positive = 0
    total_negative = 0

    for img_path in img_paths:

        filename = img_path.split(os.path.sep)[-1]
        filename = filename[:filename.rfind(".")]
        annotation = os.path.sep.join([constants.ORIG_ANNOTATIONS, "{}.xml".format(filename)])

        content = open(annotation).read()
        soup = BeautifulSoup(content, "html.parser")
        gt_boxes = []
        width = int(soup.find("width").string)
        height = int(soup.find("height").string)

        for obj in soup.find_all("object"):
            x_min = int(obj.find("xmin").string)
            y_min = int(obj.find("ymin").string)
            x_max = int(obj.find("xmax").string)
            y_max = int(obj.find("ymax").string)

            x_min = max(0, x_min)
            y_min = max(0, y_min)
            x_max = min(width, x_max)
            y_max = min(height, y_max)

            gt_boxes.append((x_min, y_min, x_max, y_max))

        img = cv2.imread(img_path)
        ss = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()
        ss.setBaseImage(img)
        ss.switchToSelectiveSearchFast()
        rects = ss.process()
        proposed_rects = []

        for (x, y, width, height) in rects:
            proposed_rects.append((x, y, x + width, y + height))

        positive_rois = 0
        negative_rois = 0

        for proposed_rect in proposed_rects[:constants.MAX_PROPOSALS]:
            (prop_start_x, prop_start_y, prop_end_x, prop_end_y) = proposed_rect

            for gt_box in gt_boxes:
                iou = compute_iou(gt_box, proposed_rect)
                (gt_start_x, gt_start_y, gt_end_x, gt_end_y) = gt_box

                roi = None
                output_path = None

                if iou > constants.IOU_POSITIVE and positive_rois <= constants.MAX_POSITIVE:
                    roi = img[prop_start_y:prop_end_y, prop_start_x:prop_end_x]
                    output_path = os.path.sep.join([constants.TRAIN_POSITIVE, "{}.png".format(total_positive)])
                    positive_rois += 1
                    total_positive += 1

                full_overlap = prop_start_x >= gt_start_x
                full_overlap = full_overlap and prop_start_y >= gt_start_y
                full_overlap = full_overlap and prop_end_x <= gt_end_x
                full_overlap = full_overlap and prop_end_y <= gt_end_y

                if not full_overlap and iou < constants.IOU_NEGATIVE and negative_rois <= constants.MAX_NEGATIVE:
                    roi = img[prop_start_y:prop_end_y, prop_start_x:prop_end_x]
                    output_path = os.path.sep.join([constants.TRAIN_NEGATIVE, "{}.png".format(total_negative)])
                    negative_rois += 1
                    total_negative += 1

                if roi is not None and output_path is not None:
                    roi = cv2.resize(roi, constants.IMG_DIMENSIONS, interpolation=cv2.INTER_CUBIC)
                    cv2.imwrite(output_path, roi)


def apply_gaussian():
    for img_name in os.listdir("input"):
        img_path = os.path.join("input", img_name)
        img = cv2.imread(img_path)
        img = cv2.GaussianBlur(img, (9, 9), 0)
        cv2.imwrite('output/' + img_name, img)


def apply_augmentation():
    data_generator = ImageDataGenerator(rotation_range=15,
                                        brightness_range=(0.7, 1.4),
                                        shear_range=0.1,
                                        zoom_range=[0.95, 1.25],
                                        horizontal_flip=True)

    train_generator = data_generator.flow_from_directory("input",
                                                         batch_size=20,
                                                         class_mode=None,
                                                         target_size=constants.IMG_DIMENSIONS,
                                                         save_to_dir="output",
                                                         save_prefix='aug-',
                                                         save_format='jpeg')

    for img in train_generator:
        pass
