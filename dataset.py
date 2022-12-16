"""
Creates a Pytorch dataset to load the Pascal VOC & MS COCO datasets
"""

import config
import numpy as np
import os
import pandas as pd
import torch

from PIL import Image, ImageFile
from torch.utils.data import Dataset, DataLoader
from utils import (
    cells_to_bboxes,
    iou_width_height as iou,
    non_max_suppression as nms,
    plot_image
)

ImageFile.LOAD_TRUNCATED_IMAGES = True

class YOLODataset(Dataset):
    """
    takes image path and label path and assign each image in image directory a target label from label directory.
    target tensor must assign 3 different anchors for each 3 different output scales.
    """


    def __init__(
        self,
        csv_file,
        img_dir,
        label_dir,
        anchors,  # (3,3,2) : for all 3 scales - 3 anchors for each scale , each anchor is w ,h
        # image_size=416,
        S=[13, 26, 52],
        C=20,
        transform=None,
    ):
        self.annotations = pd.read_csv(csv_file)  # train or test.csv
        self.img_dir = img_dir
        self.label_dir = label_dir
        # self.image_size = image_size
        self.transform = transform
        self.S = S
        self.anchors = torch.tensor(anchors[0] + anchors[1] + anchors[2]) # tensor shape : [9,2]
        self.num_anchors = self.anchors.shape[0]  # 9
        self.num_anchors_per_scale = self.num_anchors // 3   # 3
        self.C = C   # number of classes
        self.ignore_iou_thresh = 0.5  # the anchor box with higher iou w.r.t. groun truth is responxibe for outputing bbox

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index):
        label_path = os.path.join(self.label_dir, self.annotations.iloc[index, 1])
        # [class , x,y,w,h] -> [x,y,w,h, class] : for augmentation this format is acceptable
        # [[0.641, 0.5705705705705706, 0.718, 0.8408408408408409, 6.0]]
        bboxes = np.roll(np.loadtxt(fname=label_path, delimiter=" ", ndmin=2), 4, axis=1).tolist()
        img_path = os.path.join(self.img_dir, self.annotations.iloc[index, 0])
        image = np.array(Image.open(img_path).convert("RGB"))

        if self.transform:
            augmentations = self.transform(image=image, bboxes=bboxes)
            image = augmentations["image"]
            bboxes = augmentations["bboxes"]

        # Below assumes 3 scale predictions (as paper) and same num of anchors per scale
        targets = [torch.zeros((self.num_anchors // 3, S, S, 6)) for S in self.S]  # [3, 13 , 13 , 6] 6: p, x,y,w,h, c
        # which anchor is responsible for each particular cell for each scale :
        for box in bboxes:

            # tensor([0.1020, 0.3021, 0.7510, 0.0174, 0.0273, 0.0672, 0.0010, 0.0046, 0.0080])
            iou_anchors = iou(torch.tensor(box[2:4]), self.anchors)  #  # torch.Size([9]) : iou calculates iou from w , h
            # gives indexes :  tensor([2, 1, 0, 5, 4, 3, 8, 7, 6])
            anchor_indices = iou_anchors.argsort(descending=True, dim=0) # sort ious for  9 anchors
            x, y, width, height, class_label = box
            has_anchor = [False] * 3  # each scale should have one anchor
            for anchor_idx in anchor_indices:
                # anchor in which scale :
                scale_idx = anchor_idx // self.num_anchors_per_scale
                # which anchor on that scale :
                anchor_on_scale = anchor_idx % self.num_anchors_per_scale
                # How many cells in that scale : 13 or 26 or 52
                S = self.S[scale_idx]
                i, j = int(S * y), int(S * x)  # which cell
                # assign the selected anchor
                anchor_taken = targets[scale_idx][anchor_on_scale, i, j, 0]
                if not anchor_taken and not has_anchor[scale_idx]:
                    targets[scale_idx][anchor_on_scale, i, j, 0] = 1
                    # coordinates on that particular cell both between [0,1] :
                    x_cell, y_cell = S * x - j, S * y - i
                    # width and cell relative to the cell
                    width_cell, height_cell = (
                        width * S,
                        height * S,
                    )  # can be greater than 1 since it's relative to cell
                    box_coordinates = torch.tensor(
                        [x_cell, y_cell, width_cell, height_cell]
                    )
                    targets[scale_idx][anchor_on_scale, i, j, 1:5] = box_coordinates
                    targets[scale_idx][anchor_on_scale, i, j, 5] = int(class_label)
                    has_anchor[scale_idx] = True

                elif not anchor_taken and iou_anchors[anchor_idx] > self.ignore_iou_thresh:
                    targets[scale_idx][anchor_on_scale, i, j, 0] = -1  # ignore prediction

        return image, tuple(targets)


def test():
    anchors = config.ANCHORS

    transform = config.test_transforms

    dataset = YOLODataset(
        "PASCAL_VOC/train.csv",
        "PASCAL_VOC/images/",
        "PASCAL_VOC/labels/",
        S=[13, 26, 52],
        anchors=anchors,
        transform=transform,
    )
    S = [13, 26, 52]

    # anchors w.r.t a cell
    scaled_anchors = torch.tensor(anchors) / (
        1 / torch.tensor(S).unsqueeze(1).unsqueeze(1).repeat(1, 3, 2) ) # scaled_anchors :  torch.Size([3, 3, 2])

    loader = DataLoader(dataset=dataset, batch_size=1, shuffle=True)
    for x, y in loader:
        boxes = []

        for i in range(y[0].shape[1]): # 3 times
            anchor = scaled_anchors[i]
            print(anchor.shape)
            print(y[i].shape)
            boxes += cells_to_bboxes(
                y[i], is_preds=False, S=y[i].shape[2], anchors=anchor
            )[0]
        boxes = nms(boxes, iou_threshold=1, threshold=0.7, box_format="midpoint")
        print(boxes)
        plot_image(x[0].permute(1, 2, 0).to("cpu"), boxes)


if __name__ == "__main__":
    test()
