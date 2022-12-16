import math
import random
import numpy as np
from PIL import Image, ImageFile
import matplotlib.pyplot as plt


def object_det_mix_up_(image1, image2, mixup_ratio):
    '''
    image1, image2: images to be mixed up, type=ndarray
    mixup_ratio: ratio in which two images are mixed up
    Returns a mixed-up image with new set of smoothed labels
    '''

    height = max(image1.shape[0], image2.shape[0])
    width = max(image1.shape[1], image2.shape[1])
    mix_img = np.zeros((height, width, 3), dtype=np.float32)
    mix_img[:image1.shape[0], :image1.shape[1], :] = image1.astype(np.float32) \
                                                     * mixup_ratio
    mix_img[:image2.shape[0], :image2.shape[1], :] += image2.astype(np.float32) \
                                                      * (1 - mixup_ratio)
    return mix_img


def horizontal_flip(image, boxes):
    '''
    Flips the image and its bounding boxes horizontally
    '''
    _, width, _ = image.shape
    if random.randrange(2):
        image = image[:, ::-1]
        boxes = boxes.copy()
        boxes[:, 0::2] = width - boxes[:, 2::-2]
    return image, boxes


def random_crop(image, boxes, labels, ratios=None):
    '''
    Performs random crop on image and its bounding boxes
    '''


    height, width, _ = image.shape

    if len(boxes) == 0:
        return image, boxes, labels, ratios

    while True:
        mode = random.choice((
            None,
            (0.1, None),
            (0.3, None),
            (0.5, None),
            (0.7, None),
            (0.9, None),
            (None, None),
        ))

        if mode is None:
            return image, boxes, labels, ratios

        min_iou, max_iou = mode
        if min_iou is None:
            min_iou = float('-inf')
        if max_iou is None:
            max_iou = float('inf')

        for _ in range(50):
            scale = random.uniform(0.3, 1.)
            min_ratio = max(0.5, scale * scale)
            max_ratio = min(2, 1. / scale / scale)
            ratio = math.sqrt(random.uniform(min_ratio, max_ratio))
            w = int(scale * ratio * width)
            h = int((scale / ratio) * height)

            l = random.randrange(width - w)
            t = random.randrange(height - h)
            roi = np.array((l, t, l + w, t + h))

            iou = iou(boxes, roi[np.newaxis])

            if not (min_iou <= iou.min() and iou.max() <= max_iou):
                continue

            image_t = image[roi[1]:roi[3], roi[0]:roi[2]]

            centers = (boxes[:, :2] + boxes[:, 2:]) / 2
            mask = np.logical_and(roi[:2] < centers, centers < roi[2:]) \
                .all(axis=1)
            boxes_t = boxes[mask].copy()
            labels_t = labels[mask].copy()
            if ratios is not None:
                ratios_t = ratios[mask].copy()
            else:
                ratios_t = None

            if len(boxes_t) == 0:
                continue

            boxes_t[:, :2] = np.maximum(boxes_t[:, :2], roi[:2])
            boxes_t[:, :2] -= roi[:2]
            boxes_t[:, 2:] = np.minimum(boxes_t[:, 2:], roi[2:])
            boxes_t[:, 2:] -= roi[:2]

            return image_t, boxes_t, labels_t, ratios_t



if __name__ == '__main__':
    img1 = 'PASCAL_VOC/images/000013.jpg'
    img2 = 'PASCAL_VOC/images/000098.jpg'
    mixup_ratio = 0.5
    img1 = np.array(Image.open(img1).convert("RGB"))
    img2 = np.array(Image.open(img2).convert("RGB"))
    mix_img = object_det_mix_up_(img1, img2, mixup_ratio)
    # height, width, _ = mix_img.shape
    plt.imshow(img1)
    plt.show()
    plt.imshow(img2)
    plt.show()

    plt.imshow(mix_img)
    plt.show()
