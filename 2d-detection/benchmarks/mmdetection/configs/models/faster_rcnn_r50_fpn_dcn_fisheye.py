_base_ = [
    '../../../../third_party/mmdetection/configs/_base_/models/faster-rcnn_r50_fpn.py',
    '../datasets/fisheye_detection.py',
    '../../../../third_party/mmdetection/configs/_base_/schedules/schedule_1x.py',
    '../../../../third_party/mmdetection/configs/_base_/default_runtime.py',
]

model = dict(
    backbone=dict(
        dcn=dict(type='DCN', deform_groups=1, fallback_on_stride=False),
        stage_with_dcn=(False, True, True, True)),
    roi_head=dict(
        bbox_head=dict(num_classes=1)))
