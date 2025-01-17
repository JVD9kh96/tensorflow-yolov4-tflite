#! /usr/bin/env python
# coding=utf-8
from easydict import EasyDict as edict


__C                           = edict()
# Consumers can get config by: from config import cfg

cfg                           = __C

# YOLO options
__C.YOLO                      = edict()

__C.YOLO.CLASSES              = "./data/classes/coco.names"
__C.YOLO.ANCHORS              = [12,16, 19,36, 40,28, 36,75, 76,55, 72,146, 142,110, 192,243, 459,401]
__C.YOLO.ANCHORS_V3           = [10,13, 16,30, 33,23, 30,61, 62,45, 59,119, 116,90, 156,198, 373,326]
__C.YOLO.ANCHORS_TINY         = [23,27, 37,58, 81,82, 81,82, 135,169, 344,319]
__C.YOLO.STRIDES              = [8, 16, 32]
__C.YOLO.STRIDES_TINY         = [16, 32]
__C.YOLO.XYSCALE              = [1.2, 1.1, 1.05]
__C.YOLO.XYSCALE_TINY         = [1.05, 1.05]
__C.YOLO.ANCHOR_PER_SCALE     = 3
__C.YOLO.IOU_LOSS_THRESH      = 0.5


# Train options
__C.TRAIN                     = edict()

__C.TRAIN.ANNOT_PATH          = "./data/dataset/val2017.txt"
__C.TRAIN.BATCH_SIZE          = 2
# __C.TRAIN.INPUT_SIZE            = [320, 352, 384, 416, 448, 480, 512, 544, 576, 608]
__C.TRAIN.INPUT_SIZE            = 416
__C.TRAIN.DATA_AUG              = True
__C.TRAIN.LR_INIT               = 1e-3
__C.TRAIN.LR_END                = 1e-6
__C.TRAIN.WARMUP_EPOCHS         = 2
__C.TRAIN.FISRT_STAGE_EPOCHS    = 20
__C.TRAIN.SECOND_STAGE_EPOCHS   = 30
__C.TRAIN.OPTIMIZER             = 'Adam' # 'Adam', 'SGD', 'Adagrad', 'Adadelta', 'RMSprop', 'Nadam', 'Adamax'
__C.TRAIN.WEIGHT_DECAY          = 0.0005


# TEST options
__C.TEST                      = edict()

__C.TEST.ANNOT_PATH           = "./data/dataset/val2017.txt"
__C.TEST.BATCH_SIZE           = 2
__C.TEST.INPUT_SIZE           = 416
__C.TEST.DATA_AUG             = False
__C.TEST.DECTECTED_IMAGE_PATH = "./data/detection/"
__C.TEST.SCORE_THRESHOLD      = 0.25
__C.TEST.IOU_THRESHOLD        = 0.5

# Augmentation options
__C.AUG                       = edict()
__C.AUG.HORIZONTAL            = True
__C.AUG.VERTICAL              = False
__C.AUG.CROP                  = True
__C.AUG.TRANSLATE             = True

# Scale jittering
__C.SCALE                     = edict()
__C.SCALE.JITTER              = True
__C.SCALE.FACTOR              = [-0.25, 0.25]
__C.SCALE.FREQ                = 10

# Adversarial Attack
__C.ADV                       = edict()
__C.ADV.PROB                  = 0.2
__C.ADV.LR_INIT               = 0.001
__C.ADV.LR_FINAL              = 1e-6
__C.ADV.ENABLE                = False

