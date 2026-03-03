# Copyright (c) OpenMMLab. All rights reserved.
from mmdet.registry import DATASETS

from .coco import CocoDataset


@DATASETS.register_module()
class KittiDataset(CocoDataset):
    """Dataset alias for Kitti data stored in COCO detection format."""

