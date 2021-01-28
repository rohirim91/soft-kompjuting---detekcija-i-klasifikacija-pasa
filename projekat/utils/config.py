import os

ORIG_IMAGES = os.path.sep.join(["dogs", "images"])
ORIG_ANNOTATIONS = os.path.sep.join(["dogs", "annotations"])

POSITIVE_PATH = os.path.sep.join([os.path.sep.join(["dogs", "train"]), "pos"])
NEGATIVE_PATH = os.path.sep.join([os.path.sep.join(["dogs", "train"]), "neg"])

MAX_POSITIVE = 30
MAX_NEGATIVE = 10

MAX_PROPOSALS = 2000
INPUT_DIMENSIONS = (224, 224)
