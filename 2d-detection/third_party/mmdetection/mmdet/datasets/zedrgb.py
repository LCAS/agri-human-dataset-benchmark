# Copyright (c) OpenMMLab. All rights reserved.
from mmdet.registry import DATASETS

from .coco import CocoDataset


@DATASETS.register_module()
class ZedrgbDataset(CocoDataset):
    """Dataset alias for zedrgb data stored in COCO detection format."""

