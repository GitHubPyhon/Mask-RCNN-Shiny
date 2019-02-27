#!/usr/bin/env python3

import os
import cv2

from mrcnn.config import coco_config
from mrcnn.model import MaskRCNN, apply_magic
from mrcnn.utils import download_trained_weights


ORIGINAL_IMAGE = 'demo/demo.jpg'

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

COCO_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")

if not os.path.isfile(COCO_MODEL_PATH):
    download_trained_weights(COCO_MODEL_PATH)


# COCO data set object names
model = MaskRCNN(mode="inference", config=coco_config)

model.load_weights(COCO_MODEL_PATH, by_name=True)


# Use OpenCV to read
image = cv2.imread(ORIGINAL_IMAGE)

# Use cvtColor to accomplish image transformation from RGB image to gray image
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


results = model.detect([image])

r = results[0]

apply_magic(
    image, gray_image, r['rois'], r['masks'], r['class_ids']
)

cv2.imwrite('save_image.jpg', image)
