_base_ = [
    '../../../../third_party/mmdetection/configs/_base_/models/faster-rcnn_r50_fpn.py',
    '../datasets/kitti_filtered_detection.py',
    '../../../../third_party/mmdetection/configs/_base_/schedules/schedule_1x.py',
    '../../../../third_party/mmdetection/configs/_base_/default_runtime.py',
]

model = dict(
    roi_head=dict(
        bbox_head=dict(num_classes=1)))
