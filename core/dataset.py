#! /usr/bin/env python
# coding=utf-8

import os
# import cv2
import random
# import numpy as np
import tensorflow as tf
import core.utils as utils
from core.config import cfg
import tensorflow_addons as tfa


class Dataset(object):
    """implement Dataset here"""

    def __init__(self, FLAGS, is_training: bool, dataset_type: str = "converted_coco"):
        self.tiny = FLAGS.tiny
        self.strides, self.anchors, NUM_CLASS, XYSCALE = utils.load_config(FLAGS)
        self.dataset_type = dataset_type

        self.annot_path = (
            cfg.TRAIN.ANNOT_PATH if is_training else cfg.TEST.ANNOT_PATH
        )
        self.input_sizes = (
            cfg.TRAIN.INPUT_SIZE if is_training else cfg.TEST.INPUT_SIZE
        )
        self.batch_size = (
            cfg.TRAIN.BATCH_SIZE if is_training else cfg.TEST.BATCH_SIZE
        )
        self.data_aug = cfg.TRAIN.DATA_AUG if is_training else cfg.TEST.DATA_AUG

        self.train_input_sizes = cfg.TRAIN.INPUT_SIZE
        self.classes = utils.read_class_names(cfg.YOLO.CLASSES)
        self.num_classes = len(self.classes)
        self.anchor_per_scale = cfg.YOLO.ANCHOR_PER_SCALE
        self.max_bbox_per_scale = 150

        self.annotations = self.load_annotations()
        self.num_samples = len(self.annotations)
        self.num_batchs = int(tf.math.ceil(self.num_samples / self.batch_size))
        self.batch_count = 0

    def load_annotations(self):
        with open(self.annot_path, "r") as f:
            txt = f.readlines()
            if self.dataset_type == "converted_coco":
                annotations = [
                    line.strip()
                    for line in txt
                    if len(line.strip().split()[1:]) != 0
                ]
            elif self.dataset_type == "yolo":
                annotations = []
                for line in txt:
                    image_path = line.strip()
                    root, _ = os.path.splitext(image_path)
                    with open(root + ".txt") as fd:
                        boxes = fd.readlines()
                        string = ""
                        for box in boxes:
                            box = box.strip()
                            box = box.split()
                            class_num = int(box[0])
                            center_x = float(box[1])
                            center_y = float(box[2])
                            half_width = float(box[3]) / 2
                            half_height = float(box[4]) / 2
                            string += " {},{},{},{},{}".format(
                                center_x - half_width,
                                center_y - half_height,
                                center_x + half_width,
                                center_y + half_height,
                                class_num,
                            )
                        annotations.append(image_path + string)

        tf.random.shuffle(annotations)
        return annotations

    def __iter__(self):
        return self

    def __next__(self):
        with tf.device("/cpu:0"):
            # self.train_input_size = random.choice(self.train_input_sizes)
            self.train_input_size = cfg.TRAIN.INPUT_SIZE
            self.train_output_sizes = self.train_input_size // self.strides

            batch_image = tf.zeros(
                (
                    self.batch_size,
                    self.train_input_size,
                    self.train_input_size,
                    3,
                ),
                dtype=tf.dtypes.float32,
            )
            batch_image = tf.Variable(batch_image)

            batch_label_sbbox = tf.zeros(
                (
                    self.batch_size,
                    self.train_output_sizes[0],
                    self.train_output_sizes[0],
                    self.anchor_per_scale,
                    5 + self.num_classes,
                ),
                dtype=tf.dtypes.float32,
            )
            batch_label_sbbox = tf.Variable(batch_label_sbbox)
            
            batch_label_mbbox = tf.zeros(
                (
                    self.batch_size,
                    self.train_output_sizes[1],
                    self.train_output_sizes[1],
                    self.anchor_per_scale,
                    5 + self.num_classes,
                ),
                dtype=tf.dtypes.float32,
            )
            batch_label_mbbox = tf.Variable(batch_label_mbbox)
            
            batch_label_lbbox = tf.zeros(
                (
                    self.batch_size,
                    self.train_output_sizes[2],
                    self.train_output_sizes[2],
                    self.anchor_per_scale,
                    5 + self.num_classes,
                ),
                dtype=tf.dtypes.float32,
            )
            batch_label_lbbox = tf.Variable(batch_label_lbbox)

            batch_sbboxes = tf.zeros(
                (self.batch_size, self.max_bbox_per_scale, 4), dtype=tf.dtypes.float32
            )
            batch_sbboxes = tf.Variable(batch_sbboxes)
            
            batch_mbboxes = tf.zeros(
                (self.batch_size, self.max_bbox_per_scale, 4), dtype=tf.dtypes.float32
            )
            batch_mbboxes = tf.Variable(batch_mbboxes)
            
            batch_lbboxes = tf.zeros(
                (self.batch_size, self.max_bbox_per_scale, 4), dtype=tf.dtypes.float32
            )
            batch_lbboxes = tf.Variable(batch_lbboxes)
            
            num = 0
            if self.batch_count < self.num_batchs:
                while num < self.batch_size:
                    index = self.batch_count * self.batch_size + num
                    if index >= self.num_samples:
                        index -= self.num_samples
                    annotation = self.annotations[index]
                    image, bboxes = self.parse_annotation(annotation)
                    (
                        label_sbbox,
                        label_mbbox,
                        label_lbbox,
                        sbboxes,
                        mbboxes,
                        lbboxes,
                    ) = self.preprocess_true_boxes(bboxes)

                    batch_image[num, :, :, :] = image
                    batch_label_sbbox[num, :, :, :, :] = label_sbbox
                    batch_label_mbbox[num, :, :, :, :] = label_mbbox
                    batch_label_lbbox[num, :, :, :, :] = label_lbbox
                    batch_sbboxes[num, :, :] = sbboxes
                    batch_mbboxes[num, :, :] = mbboxes
                    batch_lbboxes[num, :, :] = lbboxes
                    num += 1
                self.batch_count += 1
                batch_smaller_target = batch_label_sbbox, batch_sbboxes
                batch_medium_target = batch_label_mbbox, batch_mbboxes
                batch_larger_target = batch_label_lbbox, batch_lbboxes

                return (
                    batch_image,
                    (
                        batch_smaller_target,
                        batch_medium_target,
                        batch_larger_target,
                    ),
                )
            else:
                self.batch_count = 0
                tf.random.shuffle(self.annotations)
                raise StopIteration

    def random_horizontal_flip(self, image, bboxes):
        if random.random() < 0.5:
            _, w, _ = image.shape
            image = image[:, ::-1, :]
            bboxes[:, 0].assign(w - bboxes[:, 2])
            bboxes[:, 2].assign(w - bboxes[:, 0])

        return image, bboxes

    def random_crop(self, image, bboxes):
        if random.random() < 0.5:
            h, w, _ = image.shape
            max_bbox = tf.concat(
                [
                    tf.math.reduce_min(bboxes[:, 0:2], axis=0),
                    tf.math.reduce_max(bboxes[:, 2:4], axis=0),
                ],
                axis=-1,
            )

            max_l_trans = max_bbox[0]
            max_u_trans = max_bbox[1]
            max_r_trans = w - max_bbox[2]
            max_d_trans = h - max_bbox[3]

            crop_xmin = max(
                0, int(max_bbox[0] - random.uniform(0, max_l_trans))
            )
            crop_ymin = max(
                0, int(max_bbox[1] - random.uniform(0, max_u_trans))
            )
            crop_xmax = max(
                w, int(max_bbox[2] + random.uniform(0, max_r_trans))
            )
            crop_ymax = max(
                h, int(max_bbox[3] + random.uniform(0, max_d_trans))
            )

            image = image[crop_ymin:crop_ymax, crop_xmin:crop_xmax]

            bboxes[:, 0].assign(bboxes[:, 0] - crop_xmin)
            bboxes[:, 2].assign(bboxes[:, 2] - crop_xmin)
            bboxes[:, 1].assign(bboxes[:, 1] - crop_ymin)
            bboxes[:, 3].assign(bboxes[:, 3] - crop_ymin)

        return image, bboxes

    def random_translate(self, image, bboxes):
        if random.random() < 0.5:
            h, w, _ = image.shape
            max_bbox = tf.concat(
                [
                    tf.math.reduce_min(bboxes[:, 0:2], axis=0),
                    tf.math.reduce_max(bboxes[:, 2:4], axis=0),
                ],
                axis=-1,
            )

            max_l_trans = max_bbox[0]
            max_u_trans = max_bbox[1]
            max_r_trans = w - max_bbox[2]
            max_d_trans = h - max_bbox[3]

            tx = random.uniform(-(max_l_trans - 1), (max_r_trans - 1))
            ty = random.uniform(-(max_u_trans - 1), (max_d_trans - 1))

#             M = tf.constant([[1, 0, tx], [0, 1, ty]])
#             image = cv2.warpAffine(image, M, (w, h))
#             image = tf.keras.preprocessing.image.apply_affine_transform(
#                     image, theta=0, tx=tx, ty=ty, shear=0, zx=1, zy=1, row_axis=0, col_axis=1,
#                     channel_axis=2, fill_mode='constant', cval=0.0, order=1
#             )
            image = tfa.image.translate_xy(image, [tx, ty], 0.0)
            image = tf.Variable(image)

            bboxes[:, 0].assign(bboxes[:, 0] + tx)
            bboxes[:, 2].assign(bboxes[:, 2] + tx)
            bboxes[:, 1].assign(bboxes[:, 1] + ty)
            bboxes[:, 3].assign(bboxes[:, 3] + ty)

        return image, bboxes

    def parse_annotation(self, annotation):
        line = annotation.split()
        image_path = line[0]
        if not os.path.exists(image_path):
            raise KeyError("%s does not exist ... " % image_path)
        image = tf.io.decode_jpeg(tf.io.read_file(image_path))
        if self.dataset_type == "converted_coco":
            bboxes = tf.constant(
                [list(map(int, box.split(","))) for box in line[1:]],
                dtype=tf.float64
            )
        elif self.dataset_type == "yolo":
            height, width, _ = image.shape
            bboxes = tf.constant(
                [list(map(float, box.split(","))) for box in line[1:]],
                dtype=tf.float64
            )
            bboxes = bboxes * tf.constant([width, height, width, height, 1], dtype=tf.int64)
        bboxes = tf.Variable(bboxes)
        
        if self.data_aug:
            image, bboxes = self.random_horizontal_flip(
                tf.Variable(image), tf.Variable(bboxes)
            )
            image, bboxes = self.random_crop(tf.Variable(image), tf.Variable(bboxes))
            image, bboxes = self.random_translate(
                tf.Variable(image), tf.Variable(bboxes)
            )

        image, bboxes = utils.image_preprocess(
            tf.Variable(image),
            [self.train_input_size, self.train_input_size],
            tf.Variable(bboxes),
        )
        return image, bboxes


    def preprocess_true_boxes(self, bboxes):
        label = [
            tf.zeros(
                (
                    self.train_output_sizes[i],
                    self.train_output_sizes[i],
                    self.anchor_per_scale,
                    5 + self.num_classes,
                )
            )
            for i in range(3)
        ]
        bboxes_xywh = [tf.zeros((self.max_bbox_per_scale, 4)) for _ in range(3)]
        bbox_count = tf.zeros((3,))

        for i in range(bboxes.shape[0]):
            bbox = bboxes[i]            
            bbox_coor = bbox[:4]
            bbox_class_ind = bbox[4]
            
            onehot = tf.zeros(self.num_classes, dtype=tf.float64)
            onehot = tf.Variable(onehot)
            
            onehot[int(bbox_class_ind)] = 1.0
            uniform_distribution = tf.fill(
                self.num_classes, 1.0 / self.num_classes
            )
            deta = 0.01
            smooth_onehot = onehot * (1 - deta) + deta * uniform_distribution

            bbox_xywh = tf.concat(
                [
                    (bbox_coor[2:] + bbox_coor[:2]) * 0.5,
                    bbox_coor[2:] - bbox_coor[:2],
                ],
                axis=-1,
            )
            bbox_xywh_scaled = (
                1.0 * bbox_xywh[tf.newaxis, :] / self.strides[:, tf.newaxis]
            )

            iou = []
            exist_positive = False
            for i in range(3):
                anchors_xywh = tf.zeros((self.anchor_per_scale, 4))
                anchors_xywh[:, 0:2] = (
                    tf.math.floor(bbox_xywh_scaled[i, 0:2]).astype(tf.int32) + 0.5
                )
                anchors_xywh[:, 2:4] = self.anchors[i]

                iou_scale = utils.bbox_iou(
                    bbox_xywh_scaled[i][tf.newaxis, :], anchors_xywh
                )
                iou.append(iou_scale)
                iou_mask = iou_scale > 0.3

                if tf.math.reduce_any(iou_mask):
                    xind, yind = tf.math.floor(bbox_xywh_scaled[i, 0:2]).astype(
                        tf.int32
                    )

                    label[i][yind, xind, iou_mask, :] = 0
                    label[i][yind, xind, iou_mask, 0:4] = bbox_xywh
                    label[i][yind, xind, iou_mask, 4:5] = 1.0
                    label[i][yind, xind, iou_mask, 5:] = smooth_onehot

                    bbox_ind = int(bbox_count[i] % self.max_bbox_per_scale)
                    bboxes_xywh[i][bbox_ind, :4] = bbox_xywh
                    bbox_count[i] += 1

                    exist_positive = True

            if not exist_positive:
                best_anchor_ind = tf.math.argmax(tf.constant(iou).reshape(-1), axis=-1)
                best_detect = int(best_anchor_ind / self.anchor_per_scale)
                best_anchor = int(best_anchor_ind % self.anchor_per_scale)
                xind, yind = tf.math.floor(
                    bbox_xywh_scaled[best_detect, 0:2]
                ).astype(tf.int32)

                label[best_detect][yind, xind, best_anchor, :] = 0
                label[best_detect][yind, xind, best_anchor, 0:4] = bbox_xywh
                label[best_detect][yind, xind, best_anchor, 4:5] = 1.0
                label[best_detect][yind, xind, best_anchor, 5:] = smooth_onehot

                bbox_ind = int(
                    bbox_count[best_detect] % self.max_bbox_per_scale
                )
                bboxes_xywh[best_detect][bbox_ind, :4] = bbox_xywh
                bbox_count[best_detect] += 1
        label_sbbox, label_mbbox, label_lbbox = label
        sbboxes, mbboxes, lbboxes = bboxes_xywh
        return label_sbbox, label_mbbox, label_lbbox, sbboxes, mbboxes, lbboxes

    def __len__(self):
        return self.num_batchs
