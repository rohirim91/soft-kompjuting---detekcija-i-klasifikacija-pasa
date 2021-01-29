import os

ORIG_IMAGES = os.path.sep.join(["dogs", "input"])
ORIG_ANNOTATIONS = os.path.sep.join(["dogs", "annotations"])

TRAIN_POSITIVE = os.path.sep.join([os.path.sep.join(["dogs", "train"]), "pos"])
TRAIN_NEGATIVE = os.path.sep.join([os.path.sep.join(["dogs", "train"]), "neg"])

MAX_POSITIVE = 30
MAX_NEGATIVE = 10

IOU_POSITIVE = 0.7
IOU_NEGATIVE = 0.05

MAX_PROPOSALS = 2000
IMG_DIMENSIONS = (224, 224)
