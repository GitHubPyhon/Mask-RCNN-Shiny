#!/usr/bin/env python3

import os
import cv2
import numpy as np

from mrcnn.config import Config
from mrcnn.model import MaskRCNN


# Input the original image name
original_image = 'demo/1.jpg'


# Use OpenCV to read
image = cv2.imread(original_image)

# Use cvtColor to accomplish image transformation from RGB image to gray image
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(ROOT_DIR, "logs")
COCO_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")


class CocoConfig(Config):
    """Configuration for training on MS COCO.
    Derives from the base Config class and overrides values specific
    to the COCO dataset.
    """
    # Give the configuration a recognizable name
    NAME = "coco"

    # We use a GPU with 12GB memory, which can fit two images.
    # Adjust down if you use a smaller GPU.
    IMAGES_PER_GPU = 1

    # Uncomment to train on 8 GPUs (default is 1)
    # GPU_COUNT = 8

    # Number of classes (including background)
    NUM_CLASSES = 1 + 80  # COCO has 80 classes


config = CocoConfig()


# COCO data set object names
model = MaskRCNN(
    mode="inference", model_dir=MODEL_DIR, config=config
)
model.load_weights(COCO_MODEL_PATH, by_name=True)
class_names = [
    'BG', 'person', 'bicycle', 'car', 'motorcycle', 'airplane',
    'bus', 'train', 'truck', 'boat', 'traffic light',
    'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird',
    'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear',
    'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie',
    'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
    'kite', 'baseball bat', 'baseball glove', 'skateboard',
    'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
    'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
    'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
    'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed',
    'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote',
    'keyboard', 'cell phone', 'microwave', 'oven', 'toaster',
    'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors',
    'teddy bear', 'hair drier', 'toothbrush'
]


# This function is used to change the colorful background information to
# grayscale. image[:,:,0] is the Blue channel,image[:,:,1] is the Green channel
# image[:,:,2] is the Red channel mask == 0 means that this pixel is not belong
# to the object. np.where function means that if the pixel belong to background
# change it to gray_image. Since the gray_image is 2D, for each pixel in
# background, we should set 3 channels to the same value to keep the grayscale.


def apply_mask(image, mask):
    image[:, :, 0] = np.where(
        mask == 0,
        gray_image[:, :],
        image[:, :, 0]
    )
    image[:, :, 1] = np.where(
        mask == 0,
        gray_image[:, :],
        image[:, :, 1]
    )
    image[:, :, 2] = np.where(
        mask == 0,
        gray_image[:, :],
        image[:, :, 2]
    )
    return image


def display_instances(image, boxes, masks, ids, names, obj='person', count=1):

    # n_instances saves the amount of all objects
    n_instances = boxes.shape[0]

    objects = []
    final_mask = None

    for i in range(n_instances):
        if not np.any(boxes[i]):
            continue

        # compute the square of each object
        y1, x1, y2, x2 = boxes[i]
        square = (y2 - y1) * (x2 - x1)

        label = names[ids[i]]

        objects.append([label, square])

        if final_mask is not None:
            final_mask |= masks[:, :, i]
        else:
            final_mask = masks[:, :, i]

    apply_mask(image, final_mask)

    return True


results = model.detect([image], verbose=0)
r = results[0]
display_instances(
    image, r['rois'], r['masks'], r['class_ids'], class_names
)

cv2.imwrite('save_image.jpg', image)
