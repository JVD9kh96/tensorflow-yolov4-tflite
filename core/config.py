
cfg =   {'YOLO':{
                'CLASSES'             : "data/classes/voc.names",
                'ANCHORS'             : [12,16, 19,36, 40,28, 36,75, 76,55, 72,146, 142,110, 192,243, 459,401],
                'ANCHORS_V3'          : [10,13, 16,30, 33,23, 30,61, 62,45, 59,119, 116,90, 156,198, 373,326],
                'ANCHORS_TINY'        : [23,27, 37,58, 81,82, 81,82, 135,169, 344,319],
                'STRIDES'             : [8, 16, 32],
                'STRIDES_TINY'        : [16, 32],
                'XYSCALE'             : [1.2, 1.1, 1.05],
                'XYSCALE_TINY'        : [1.05, 1.05],
                'ANCHOR_PER_SCALE'    : 3,
                'IOU_LOSS_THRESH'     : 0.5,
                },
        'TRAIN':{
                'ANNOT_PATH'          : "data/dataset/yymnist_train.txt",
                'BATCH_SIZE'          : 1,
                'INPUT_SIZE'          : 416,
                'DATA_AUG'            : True,
                'LR_INIT'             : 1.3e-3,
                'LR_END'              : 1e-6,
                'WARMUP_EPOCHS'       : 5,
                'FISRT_STAGE_EPOCHS'  : 0,
                'SECOND_STAGE_EPOCHS ': 150
        },
        'TEST':{
                'ANNOT_PATH'           : "data/dataset/yymnist_test.txt",
                'BATCH_SIZE'           : 1,
                'INPUT_SIZE'           : 416,
                'DATA_AUG'             : False,
                'DECTECTED_IMAGE_PATH' : "./data/detection/",
                'SCORE_THRESHOLD'      : 0.25,
                'IOU_THRESHOLD'        : 0.5,
        }
        }
