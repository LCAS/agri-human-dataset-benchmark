# Copyright (c) OpenMMLab. All rights reserved.
from mmdet.registry import DATASETS

from .coco import CocoDataset


@DATASETS.register_module()
class FisheyeDataset(CocoDataset):
    """Dataset alias for fisheye data stored in COCO detection format."""

