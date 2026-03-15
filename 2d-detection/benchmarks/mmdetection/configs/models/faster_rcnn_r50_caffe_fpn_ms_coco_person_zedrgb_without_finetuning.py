_base_ = [
    '../../../../third_party/mmdetection/configs/faster_rcnn/faster-rcnn_r50-caffe_fpn_ms-1x_coco-person.py',
    '../datasets/zedrgb_detection.py',
]

load_from = 'https://download.openmmlab.com/mmdetection/v2.0/faster_rcnn/faster_rcnn_r50_fpn_1x_coco-person/faster_rcnn_r50_fpn_1x_coco-person_20201216_175929-d022e227.pth'
