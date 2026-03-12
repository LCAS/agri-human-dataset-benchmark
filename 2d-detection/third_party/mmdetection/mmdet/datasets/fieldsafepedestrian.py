# Copyright (c) OpenMMLab. All rights reserved.
from mmdet.registry import DATASETS

from .coco import CocoDataset


@DATASETS.register_module()
class FieldsafepedestrianDataset(CocoDataset):
    """Dataset alias for fieldsafepedestrian data stored in COCO detection format."""

