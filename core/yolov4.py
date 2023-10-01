#! /usr/bin/env python
# coding=utf-8

import tensorflow as tf
import core.utils as utils
import core.common as common
import core.backbone as backbone


def YOLO(input_layer, NUM_CLASS, model='yolov4', is_tiny=False, activation = 'gelu', projection_dim = 128,transformer_layers =[6, 6, 6], attention_heads=[4, 4, 4], spp=0, normal=0, axes = [1, 2], include_head=True):
    if is_tiny:
        if model == 'yolov4':
            return YOLOv4_tiny(input_layer, NUM_CLASS)
        elif model == 'yolov3':
            return YOLOv3_tiny(input_layer, NUM_CLASS)
        elif model == 'yolov4_vit_v1':
            return YOLOv4_vit_v1_tiny(input_layer, NUM_CLASS, activation)
    else:
        if model == 'yolov4':
            return YOLOv4(input_layer, NUM_CLASS)
        elif model == 'yolov3':
            return YOLOv3(input_layer, NUM_CLASS)
        elif model == 'yolov4_vit_v1':
            return YOLOv4_vit_v1(input_layer, NUM_CLASS, activation, projection_dim, transformer_layers, attention_heads, spp, normal)
        elif model == 'yolov4_vit_v1_light':
            return YOLOv4_vit_v1_light(input_layer, NUM_CLASS, activation)
        elif model == 'yolov4_vit_v2':
            return YOLOv4_vit_v2(input_layer, NUM_CLASS, activation, projection_dim, transformer_layers, attention_heads, spp, normal)
        elif model == 'yolov4_vit_v3':
            return YOLOv4_vit_v3(input_layer, NUM_CLASS, activation, projection_dim, transformer_layers, attention_heads, spp, normal)
        elif model == 'yolov3_vit_v3':
            return YOLOv3_vit_v3(input_layer, NUM_CLASS, activation, projection_dim, transformer_layers, attention_heads, spp, normal)
        elif model == 'yolov4_att_v1':
            return YOLOv4_att_v1(input_layer, NUM_CLASS, activation, transformer_layers, attention_heads, normal)
        elif model == 'yolov4_vit_v5':
            return YOLOv4_vit_v5(input_layer, NUM_CLASS, activation, projection_dim, transformer_layers, attention_heads, spp, normal)
        elif model == 'yolov4_att_v2':
            return YOLOv4_att_v2(input_layer, NUM_CLASS, activation=activation, normal = normal)
        elif model == 'yolov4_att_v3':
            return YOLOv4_att_v3(input_layer, NUM_CLASS, activation=activation, attention_axes = axes, normal = normal)
        elif model == 'yolov4_att_v4':
            return YOLOv4_att_v4(input_layer, NUM_CLASS, activation=activation, attention_axes = axes, normal = normal,  include_head=include_head)


def YOLOv3(input_layer, NUM_CLASS):
    route_1, route_2, conv = backbone.darknet53(input_layer)

    conv = common.convolutional(conv, (1, 1, 1024, 512))
    conv = common.convolutional(conv, (3, 3, 512, 1024))
    conv = common.convolutional(conv, (1, 1, 1024, 512))
    conv = common.convolutional(conv, (3, 3, 512, 1024))
    conv = common.convolutional(conv, (1, 1, 1024, 512))

    conv_lobj_branch = common.convolutional(conv, (3, 3, 512, 1024))
    conv_lbbox = common.convolutional(conv_lobj_branch, (1, 1, 1024, 3 * (NUM_CLASS + 5)), activate=False, bn=False)

    conv = common.convolutional(conv, (1, 1, 512, 256))
    conv = common.upsample(conv)

    conv = tf.concat([conv, route_2], axis=-1)

    conv = common.convolutional(conv, (1, 1, 768, 256))
    conv = common.convolutional(conv, (3, 3, 256, 512))
    conv = common.convolutional(conv, (1, 1, 512, 256))
    conv = common.convolutional(conv, (3, 3, 256, 512))
    conv = common.convolutional(conv, (1, 1, 512, 256))

    conv_mobj_branch = common.convolutional(conv, (3, 3, 256, 512))
    conv_mbbox = common.convolutional(conv_mobj_branch, (1, 1, 512, 3 * (NUM_CLASS + 5)), activate=False, bn=False)

    conv = common.convolutional(conv, (1, 1, 256, 128))
    conv = common.upsample(conv)

    conv = tf.concat([conv, route_1], axis=-1)

    conv = common.convolutional(conv, (1, 1, 384, 128))
    conv = common.convolutional(conv, (3, 3, 128, 256))
    conv = common.convolutional(conv, (1, 1, 256, 128))
    conv = common.convolutional(conv, (3, 3, 128, 256))
    conv = common.convolutional(conv, (1, 1, 256, 128))

    conv_sobj_branch = common.convolutional(conv, (3, 3, 128, 256))
    conv_sbbox = common.convolutional(conv_sobj_branch, (1, 1, 256, 3 * (NUM_CLASS + 5)), activate=False, bn=False)

    return [conv_sbbox, conv_mbbox, conv_lbbox]

def YOLOv3_vit_v3(input_layer,
                  NUM_CLASS,
                  activation = 'gelu',
                  projection_dim = 128,
                  transformer_layers =[6, 6, 6],
                  attention_heads=[4, 4, 4],
                  spp = 0,
                  normal = 0):
    if normal == 6:
        normal = 5
        norm = 1
    else:
        norm = 0
    route_1, route_2, conv = backbone.VIT_v3(input_layer,
                                             projection_dim = projection_dim,
                                             transformer_layers =transformer_layers,
                                             attention_heads=attention_heads,
                                             activation=activation,
                                             normal=normal)

    conv = common.convolutional(conv, (1, 1, 1024, 512))
    conv = common.convolutional(conv, (3, 3, 512, 1024))
    conv = common.convolutional(conv, (1, 1, 1024, 512))
    conv = common.convolutional(conv, (3, 3, 512, 1024))
    conv = common.convolutional(conv, (1, 1, 1024, 512))

    conv_lobj_branch = common.convolutional(conv, (3, 3, 512, 1024))
    conv_lbbox = common.convolutional(conv_lobj_branch, (1, 1, 1024, 3 * (NUM_CLASS + 5)), activate=False, bn=False)

    conv = common.convolutional(conv, (1, 1, 512, 256))
    conv = common.upsample(conv)

    conv = tf.concat([conv, route_2], axis=-1)

    conv = common.convolutional(conv, (1, 1, 768, 256))
    conv = common.convolutional(conv, (3, 3, 256, 512))
    conv = common.convolutional(conv, (1, 1, 512, 256))
    conv = common.convolutional(conv, (3, 3, 256, 512))
    conv = common.convolutional(conv, (1, 1, 512, 256))

    conv_mobj_branch = common.convolutional(conv, (3, 3, 256, 512))
    conv_mbbox = common.convolutional(conv_mobj_branch, (1, 1, 512, 3 * (NUM_CLASS + 5)), activate=False, bn=False)

    conv = common.convolutional(conv, (1, 1, 256, 128))
    conv = common.upsample(conv)

    conv = tf.concat([conv, route_1], axis=-1)

    conv = common.convolutional(conv, (1, 1, 384, 128))
    conv = common.convolutional(conv, (3, 3, 128, 256))
    conv = common.convolutional(conv, (1, 1, 256, 128))
    conv = common.convolutional(conv, (3, 3, 128, 256))
    conv = common.convolutional(conv, (1, 1, 256, 128))

    conv_sobj_branch = common.convolutional(conv, (3, 3, 128, 256))
    conv_sbbox = common.convolutional(conv_sobj_branch, (1, 1, 256, 3 * (NUM_CLASS + 5)), activate=False, bn=False)

    return [conv_sbbox, conv_mbbox, conv_lbbox]

def YOLOv4(input_layer, NUM_CLASS):
    route_1, route_2, conv = backbone.cspdarknet53(input_layer)
    #x1 = common.convolutional(conv, (1, 1, 1024, 512))
    #x2 = common.convolutional(x1, (3, 1, 512, 1024))
    #x3 = common.convolutional(x2, (1, 1, 1024, 512))
    #mxp1 = tf.keras.layers.MaxPool2D(pool_size = (5, 5), strides = 1, padding = 'same')(x3)
    #mxp2 = tf.keras.layers.MaxPool2D(pool_size=(9, 9), strides = 1, padding = 'same')(x3)
    #mxp3 = tf.keras.layers.MaxPool2D(pool_size = (13, 13), strides = 1, padding = 'same')(x3)
    #spp = tf.keras.layers.concatenate([mxp1, mxp2, mxp3, x3], axis = -1)
    #x4 = common.convolutional(spp, (1, 1, 2048, 512))
    #x5 = common.convolutional(x4, (3, 1, 512, 1024))
    #x6 = common.convolutional(x5, (1, 1, 1024, 512))
    #conv = x6
    route = conv
    conv = common.convolutional(conv, (1, 1, 512, 256))
    conv = common.upsample(conv)
    route_2 = common.convolutional(route_2, (1, 1, 512, 256))
    conv = tf.concat([route_2, conv], axis=-1)

    conv = common.convolutional(conv, (1, 1, 512, 256))
    conv = common.convolutional(conv, (3, 3, 256, 512))
    conv = common.convolutional(conv, (1, 1, 512, 256))
    conv = common.convolutional(conv, (3, 3, 256, 512))
    conv = common.convolutional(conv, (1, 1, 512, 256))

    route_2 = conv
    conv = common.convolutional(conv, (1, 1, 256, 128))
    conv = common.upsample(conv)
    route_1 = common.convolutional(route_1, (1, 1, 256, 128))
    conv = tf.concat([route_1, conv], axis=-1)

    conv = common.convolutional(conv, (1, 1, 256, 128))
    conv = common.convolutional(conv, (3, 3, 128, 256))
    conv = common.convolutional(conv, (1, 1, 256, 128))
    conv = common.convolutional(conv, (3, 3, 128, 256))
    conv = common.convolutional(conv, (1, 1, 256, 128))

    route_1 = conv
    conv = common.convolutional(conv, (3, 3, 128, 256))
    conv_sbbox = common.convolutional(conv, (1, 1, 256, 3 * (NUM_CLASS + 5)), activate=False, bn=False)

    conv = common.convolutional(route_1, (3, 3, 128, 256), downsample=True)
    conv = tf.concat([conv, route_2], axis=-1)

    conv = common.convolutional(conv, (1, 1, 512, 256))
    conv = common.convolutional(conv, (3, 3, 256, 512))
    conv = common.convolutional(conv, (1, 1, 512, 256))
    conv = common.convolutional(conv, (3, 3, 256, 512))
    conv = common.convolutional(conv, (1, 1, 512, 256))

    route_2 = conv
    conv = common.convolutional(conv, (3, 3, 256, 512))
    conv_mbbox = common.convolutional(conv, (1, 1, 512, 3 * (NUM_CLASS + 5)), activate=False, bn=False)

    conv = common.convolutional(route_2, (3, 3, 256, 512), downsample=True)
    conv = tf.concat([conv, route], axis=-1)

    conv = common.convolutional(conv, (1, 1, 1024, 512))
    conv = common.convolutional(conv, (3, 3, 512, 1024))
    conv = common.convolutional(conv, (1, 1, 1024, 512))
    conv = common.convolutional(conv, (3, 3, 512, 1024))
    conv = common.convolutional(conv, (1, 1, 1024, 512))

    conv = common.convolutional(conv, (3, 3, 512, 1024))
    conv_lbbox = common.convolutional(conv, (1, 1, 1024, 3 * (NUM_CLASS + 5)), activate=False, bn=False)

    return [conv_sbbox, conv_mbbox, conv_lbbox]

def YOLOv4_att_v1(input_layer, NUM_CLASS,
                  activation = 'gelu',
                  transformer_layers =[1, 1, 1],
                  attention_heads=[4, 4, 4],
                  normal = 2):
    if normal > 2:
        normal = normal - 3
    
    route_1, route_2, conv = backbone.cspdarknet53_att_v1(input_layer, 
                                                          transformer_layers ,
                                                          attention_heads,
                                                          activation,
                                                          normal)

    route = conv
    conv = common.convolutional(conv, (1, 1, 512, 256))
    conv = common.upsample(conv)
    route_2 = common.convolutional(route_2, (1, 1, 512, 256))
    conv = tf.concat([route_2, conv], axis=-1)

    conv = common.convolutional(conv, (1, 1, 512, 256))
    conv = common.convolutional(conv, (3, 3, 256, 512))
    conv = common.convolutional(conv, (1, 1, 512, 256))
    conv = common.convolutional(conv, (3, 3, 256, 512))
    conv = common.convolutional(conv, (1, 1, 512, 256))

    route_2 = conv
    conv = common.convolutional(conv, (1, 1, 256, 128))
    conv = common.upsample(conv)
    route_1 = common.convolutional(route_1, (1, 1, 256, 128))
    conv = tf.concat([route_1, conv], axis=-1)

    conv = common.convolutional(conv, (1, 1, 256, 128))
    conv = common.convolutional(conv, (3, 3, 128, 256))
    conv = common.convolutional(conv, (1, 1, 256, 128))
    conv = common.convolutional(conv, (3, 3, 128, 256))
    conv = common.convolutional(conv, (1, 1, 256, 128))

    route_1 = conv
    conv = common.convolutional(conv, (3, 3, 128, 256))
    conv_sbbox = common.convolutional(conv, (1, 1, 256, 3 * (NUM_CLASS + 5)), activate=False, bn=False)

    conv = common.convolutional(route_1, (3, 3, 128, 256), downsample=True)
    conv = tf.concat([conv, route_2], axis=-1)

    conv = common.convolutional(conv, (1, 1, 512, 256))
    conv = common.convolutional(conv, (3, 3, 256, 512))
    conv = common.convolutional(conv, (1, 1, 512, 256))
    conv = common.convolutional(conv, (3, 3, 256, 512))
    conv = common.convolutional(conv, (1, 1, 512, 256))

    route_2 = conv
    conv = common.convolutional(conv, (3, 3, 256, 512))
    conv_mbbox = common.convolutional(conv, (1, 1, 512, 3 * (NUM_CLASS + 5)), activate=False, bn=False)

    conv = common.convolutional(route_2, (3, 3, 256, 512), downsample=True)
    conv = tf.concat([conv, route], axis=-1)

    conv = common.convolutional(conv, (1, 1, 1024, 512))
    conv = common.convolutional(conv, (3, 3, 512, 1024))
    conv = common.convolutional(conv, (1, 1, 1024, 512))
    conv = common.convolutional(conv, (3, 3, 512, 1024))
    conv = common.convolutional(conv, (1, 1, 1024, 512))

    conv = common.convolutional(conv, (3, 3, 512, 1024))
    conv_lbbox = common.convolutional(conv, (1, 1, 1024, 3 * (NUM_CLASS + 5)), activate=False, bn=False)

    return [conv_sbbox, conv_mbbox, conv_lbbox]

def YOLOv4_att_v2(input_layer,
                  NUM_CLASS,
                  activation = 'mish',
                  normal = 2):
    if normal > 2:
        normal = normal - 3
    if normal == 0:
        normal = 'batch'
    elif normal == 1:
        normal = 'group'
    elif normal == 2:
        normal = 'layer'

    route_1, route_2, conv = backbone.cspdarkernet53(input_layer, 
                                                          attention_axes = 1,
                                                          activation=activation,
                                                          normalization = normal)

    route = conv
    conv = common.convolutional(conv, (1, 1, 512, 256))
    conv = common.upsample(conv)
    route_2 = common.convolutional(route_2, (1, 1, 512, 256))
    conv = tf.concat([route_2, conv], axis=-1)

    conv = common.convolutional(conv, (1, 1, 512, 256))
    conv = common.convolutional(conv, (3, 3, 256, 512))
    conv = common.convolutional(conv, (1, 1, 512, 256))
    conv = common.convolutional(conv, (3, 3, 256, 512))
    conv = common.convolutional(conv, (1, 1, 512, 256))

    route_2 = conv
    conv = common.convolutional(conv, (1, 1, 256, 128))
    conv = common.upsample(conv)
    route_1 = common.convolutional(route_1, (1, 1, 256, 128))
    conv = tf.concat([route_1, conv], axis=-1)

    conv = common.convolutional(conv, (1, 1, 256, 128))
    conv = common.convolutional(conv, (3, 3, 128, 256))
    conv = common.convolutional(conv, (1, 1, 256, 128))
    conv = common.convolutional(conv, (3, 3, 128, 256))
    conv = common.convolutional(conv, (1, 1, 256, 128))

    route_1 = conv
    conv = common.convolutional(conv, (3, 3, 128, 256))
    conv_sbbox = common.convolutional(conv, (1, 1, 256, 3 * (NUM_CLASS + 5)), activate=False, bn=False)

    conv = common.convolutional(route_1, (3, 3, 128, 256), downsample=True)
    conv = tf.concat([conv, route_2], axis=-1)

    conv = common.convolutional(conv, (1, 1, 512, 256))
    conv = common.convolutional(conv, (3, 3, 256, 512))
    conv = common.convolutional(conv, (1, 1, 512, 256))
    conv = common.convolutional(conv, (3, 3, 256, 512))
    conv = common.convolutional(conv, (1, 1, 512, 256))

    route_2 = conv
    conv = common.convolutional(conv, (3, 3, 256, 512))
    conv_mbbox = common.convolutional(conv, (1, 1, 512, 3 * (NUM_CLASS + 5)), activate=False, bn=False)

    conv = common.convolutional(route_2, (3, 3, 256, 512), downsample=True)
    conv = tf.concat([conv, route], axis=-1)

    conv = common.convolutional(conv, (1, 1, 1024, 512))
    conv = common.convolutional(conv, (3, 3, 512, 1024))
    conv = common.convolutional(conv, (1, 1, 1024, 512))
    conv = common.convolutional(conv, (3, 3, 512, 1024))
    conv = common.convolutional(conv, (1, 1, 1024, 512))

    conv = common.convolutional(conv, (3, 3, 512, 1024))
    conv_lbbox = common.convolutional(conv, (1, 1, 1024, 3 * (NUM_CLASS + 5)), activate=False, bn=False)

    return [conv_sbbox, conv_mbbox, conv_lbbox]


def YOLOv4_att_v3(input_layer,
                  NUM_CLASS,
                  activation = 'mish',
                  normal = 2,
                  attention_axes = [1, 2]):
    if normal > 2:
        normal = normal - 3
    if normal == 0:
        normal = 'batch'
    elif normal == 1:
        normal = 'group'
    elif normal == 2:
        normal = 'layer'

    route_1, route_2, conv = backbone.darkernet53(input_layer, 
                                                          attention_axes = attention_axes,
                                                          activation=activation,
                                                          normalization = normal)

    route = conv
    conv = common.convolutional(conv, (1, 1, 512, 256))
    conv = common.upsample(conv)
    route_2 = common.convolutional(route_2, (1, 1, 512, 256))
    conv = tf.concat([route_2, conv], axis=-1)

    conv = common.convolutional(conv, (1, 1, 512, 256))
    conv = common.convolutional(conv, (3, 3, 256, 512))
    conv = common.convolutional(conv, (1, 1, 512, 256))
    conv = common.convolutional(conv, (3, 3, 256, 512))
    conv = common.convolutional(conv, (1, 1, 512, 256))

    route_2 = conv
    conv = common.convolutional(conv, (1, 1, 256, 128))
    conv = common.upsample(conv)
    route_1 = common.convolutional(route_1, (1, 1, 256, 128))
    conv = tf.concat([route_1, conv], axis=-1)

    conv = common.convolutional(conv, (1, 1, 256, 128))
    conv = common.convolutional(conv, (3, 3, 128, 256))
    conv = common.convolutional(conv, (1, 1, 256, 128))
    conv = common.convolutional(conv, (3, 3, 128, 256))
    conv = common.convolutional(conv, (1, 1, 256, 128))

    route_1 = conv
    conv = common.convolutional(conv, (3, 3, 128, 256))
    conv_sbbox = common.convolutional(conv, (1, 1, 256, 3 * (NUM_CLASS + 5)), activate=False, bn=False)

    conv = common.convolutional(route_1, (3, 3, 128, 256), downsample=True)
    conv = tf.concat([conv, route_2], axis=-1)

    conv = common.convolutional(conv, (1, 1, 512, 256))
    conv = common.convolutional(conv, (3, 3, 256, 512))
    conv = common.convolutional(conv, (1, 1, 512, 256))
    conv = common.convolutional(conv, (3, 3, 256, 512))
    conv = common.convolutional(conv, (1, 1, 512, 256))

    route_2 = conv
    conv = common.convolutional(conv, (3, 3, 256, 512))
    conv_mbbox = common.convolutional(conv, (1, 1, 512, 3 * (NUM_CLASS + 5)), activate=False, bn=False)

    conv = common.convolutional(route_2, (3, 3, 256, 512), downsample=True)
    conv = tf.concat([conv, route], axis=-1)

    conv = common.convolutional(conv, (1, 1, 1024, 512))
    conv = common.convolutional(conv, (3, 3, 512, 1024))
    conv = common.convolutional(conv, (1, 1, 1024, 512))
    conv = common.convolutional(conv, (3, 3, 512, 1024))
    conv = common.convolutional(conv, (1, 1, 1024, 512))

    conv = common.convolutional(conv, (3, 3, 512, 1024))
    conv_lbbox = common.convolutional(conv, (1, 1, 1024, 3 * (NUM_CLASS + 5)), activate=False, bn=False)

    return [conv_sbbox, conv_mbbox, conv_lbbox]


def YOLOv4_att_v4(input_layer,
                  NUM_CLASS,
                  activation = 'mish',
                  normal = 2,
                  attention_axes = [1, 2],
                  include_head=True):
    
    if normal > 2:
        normal = normal - 3
    if normal == 0:
        normal = 'batch'
    elif normal == 1:
        normal = 'group'
    elif normal == 2:
        normal = 'layer'

    route_1, route_2, conv = backbone.cspdarkerattnet53(input_layer, 
                                                          attention_axes = attention_axes, 
                                                          activation=activation,
                                                          normalization = normal)

    route = conv
    conv = common.convolutional(conv, (1, 1, 512, 256))
    conv = common.upsample(conv)
    route_2 = common.convolutional(route_2, (1, 1, 512, 256))
    conv = tf.concat([route_2, conv], axis=-1)

    conv = common.convolutional(conv, (1, 1, 512, 256))
    conv = common.convolutional(conv, (3, 3, 256, 512))
    conv = common.convolutional(conv, (1, 1, 512, 256))
    conv = common.convolutional(conv, (3, 3, 256, 512))
    conv = common.convolutional(conv, (1, 1, 512, 256))

    route_2 = conv
    conv = common.convolutional(conv, (1, 1, 256, 128))
    conv = common.upsample(conv)
    route_1 = common.convolutional(route_1, (1, 1, 256, 128))
    conv = tf.concat([route_1, conv], axis=-1)

    conv = common.convolutional(conv, (1, 1, 256, 128))
    conv = common.convolutional(conv, (3, 3, 128, 256))
    conv = common.convolutional(conv, (1, 1, 256, 128))
    conv = common.convolutional(conv, (3, 3, 128, 256))
    conv = common.convolutional(conv, (1, 1, 256, 128))

    route_1 = conv
    conv = common.convolutional(conv, (3, 3, 128, 256))
    if include_head:
        conv_sbbox = common.convolutional(conv, (1, 1, 256, 3 * (NUM_CLASS + 5)), activate=False, bn=False)
    else:
        conv_sbbox = conv

    conv = common.convolutional(route_1, (3, 3, 128, 256), downsample=True)
    conv = tf.concat([conv, route_2], axis=-1)

    conv = common.convolutional(conv, (1, 1, 512, 256))
    conv = common.convolutional(conv, (3, 3, 256, 512))
    conv = common.convolutional(conv, (1, 1, 512, 256))
    conv = common.convolutional(conv, (3, 3, 256, 512))
    conv = common.convolutional(conv, (1, 1, 512, 256))

    route_2 = conv
    conv = common.convolutional(conv, (3, 3, 256, 512))
    if include_head:
        conv_mbbox = common.convolutional(conv, (1, 1, 512, 3 * (NUM_CLASS + 5)), activate=False, bn=False)
    else:
        conv_mbbox = conv

    conv = common.convolutional(route_2, (3, 3, 256, 512), downsample=True)
    conv = tf.concat([conv, route], axis=-1)

    conv = common.convolutional(conv, (1, 1, 1024, 512))
    conv = common.convolutional(conv, (3, 3, 512, 1024))
    conv = common.convolutional(conv, (1, 1, 1024, 512))
    conv = common.convolutional(conv, (3, 3, 512, 1024))
    conv = common.convolutional(conv, (1, 1, 1024, 512))

    conv = common.convolutional(conv, (3, 3, 512, 1024))
    if include_head:
        conv_lbbox = common.convolutional(conv, (1, 1, 1024, 3 * (NUM_CLASS + 5)), activate=False, bn=False)
    else:
        conv_lbbox = conv
    return [conv_sbbox, conv_mbbox, conv_lbbox]


def Yolov4_neck(route_1,
                route_2,
                conv ,
                NUM_CLASS,
                include_head=True,
                dropblock=False,
                dropblock_keep_prob=1,
                dtype=tf.float32):

    route = conv
    conv = common.convolutional(conv, (1, 1, 512, 256))
    conv = common.upsample(conv, dtype)
    route_2 = common.convolutional(route_2, (1, 1, 512, 256))
    conv = tf.concat([route_2, conv], axis=-1)

    conv = common.convolutional(conv, (1, 1, 512, 256))
    conv = common.convolutional(conv, (3, 3, 256, 512))
    conv = common.convolutional(conv, (1, 1, 512, 256))
    conv = common.convolutional(conv, (3, 3, 256, 512))
    conv = common.convolutional(conv, (1, 1, 512, 256))

    route_2 = conv
    conv = common.convolutional(conv, (1, 1, 256, 128))
    conv = common.upsample(conv, dtype)
    route_1 = common.convolutional(route_1, (1, 1, 256, 128))
    conv = tf.concat([route_1, conv], axis=-1)

    conv = common.convolutional(conv, (1, 1, 256, 128))
    conv = common.convolutional(conv, (3, 3, 128, 256))
    conv = common.convolutional(conv, (1, 1, 256, 128))
    conv = common.convolutional(conv, (3, 3, 128, 256), dropblock=True, dropblock_keep_prob=dropblock_keep_prob)
    conv = common.convolutional(conv, (1, 1, 256, 128), dropblock=True, dropblock_keep_prob=dropblock_keep_prob)

    route_1 = conv
    conv = common.convolutional(conv, (3, 3, 128, 256), dropblock=True, dropblock_keep_prob=dropblock_keep_prob)
    if include_head:
        conv_sbbox = common.convolutional(conv, (1, 1, 256, 3 * (NUM_CLASS + 5)), activate=False, bn=False)
    else:
        conv_sbbox = conv

    conv = common.convolutional(route_1, (3, 3, 128, 256), downsample=True)
    conv = tf.concat([conv, route_2], axis=-1)

    conv = common.convolutional(conv, (1, 1, 512, 256))
    conv = common.convolutional(conv, (3, 3, 256, 512))
    conv = common.convolutional(conv, (1, 1, 512, 256))
    conv = common.convolutional(conv, (3, 3, 256, 512), dropblock=True, dropblock_keep_prob=dropblock_keep_prob)
    conv = common.convolutional(conv, (1, 1, 512, 256), dropblock=True, dropblock_keep_prob=dropblock_keep_prob)

    route_2 = conv
    conv = common.convolutional(conv, (3, 3, 256, 512), dropblock=True, dropblock_keep_prob=dropblock_keep_prob)
    if include_head:
        conv_mbbox = common.convolutional(conv, (1, 1, 512, 3 * (NUM_CLASS + 5)), activate=False, bn=False)
    else:
        conv_mbbox = conv

    conv = common.convolutional(route_2, (3, 3, 256, 512), downsample=True)
    conv = tf.concat([conv, route], axis=-1)

    conv = common.convolutional(conv, (1, 1, 1024, 512))
    conv = common.convolutional(conv, (3, 3, 512, 1024))
    conv = common.convolutional(conv, (1, 1, 1024, 512))
    conv = common.convolutional(conv, (3, 3, 512, 1024), dropblock=True, dropblock_keep_prob=dropblock_keep_prob)
    conv = common.convolutional(conv, (1, 1, 1024, 512), dropblock=True, dropblock_keep_prob=dropblock_keep_prob)

    conv = common.convolutional(conv, (3, 3, 512, 1024), dropblock=True, dropblock_keep_prob=dropblock_keep_prob)
    if include_head:
        conv_lbbox = common.convolutional(conv, (1, 1, 1024, 3 * (NUM_CLASS + 5)), activate=False, bn=False)
    else:
        conv_lbbox = conv
    return [conv_sbbox, conv_mbbox, conv_lbbox]

def sepYolov4_neck(route_1,
                route_2,
                conv ,
                NUM_CLASS,
                include_head=True,
                dropblock=False,
                dropblock_keep_prob=1,
                dtype=tf.float32):

    route = conv
    conv = common.convolutional(conv, (1, 1, 512, 256))
    conv = common.upsample(conv, dtype)
    route_2 = common.convolutional(route_2, (1, 1, 512, 256))
    conv = tf.concat([route_2, conv], axis=-1)

    conv = common.convolutional(conv, (1, 1, 512, 256))
    conv = common.sepconvolutional(conv, (3, 3, 256, 512))
    conv = common.convolutional(conv, (1, 1, 512, 256))
    conv = common.sepconvolutional(conv, (3, 3, 256, 512))
    conv = common.convolutional(conv, (1, 1, 512, 256))

    route_2 = conv
    conv = common.convolutional(conv, (1, 1, 256, 128))
    conv = common.upsample(conv, dtype)
    route_1 = common.convolutional(route_1, (1, 1, 256, 128))
    conv = tf.concat([route_1, conv], axis=-1)

    conv = common.convolutional(conv, (1, 1, 256, 128))
    conv = common.sepconvolutional(conv, (3, 3, 128, 256))
    conv = common.convolutional(conv, (1, 1, 256, 128))
    conv = common.sepconvolutional(conv, (3, 3, 128, 256), dropblock=True, dropblock_keep_prob=dropblock_keep_prob)
    conv = common.convolutional(conv, (1, 1, 256, 128), dropblock=True, dropblock_keep_prob=dropblock_keep_prob)

    route_1 = conv
    conv = common.sepconvolutional(conv, (3, 3, 128, 256), dropblock=True, dropblock_keep_prob=dropblock_keep_prob)
    if include_head:
        conv_sbbox = common.convolutional(conv, (1, 1, 256, 3 * (NUM_CLASS + 5)), activate=False, bn=False)
    else:
        conv_sbbox = conv

    conv = common.sepconvolutional(route_1, (3, 3, 128, 256), downsample=True)
    conv = tf.concat([conv, route_2], axis=-1)

    conv = common.convolutional(conv, (1, 1, 512, 256))
    conv = common.sepconvolutional(conv, (3, 3, 256, 512))
    conv = common.convolutional(conv, (1, 1, 512, 256))
    conv = common.sepconvolutional(conv, (3, 3, 256, 512), dropblock=True, dropblock_keep_prob=dropblock_keep_prob)
    conv = common.convolutional(conv, (1, 1, 512, 256), dropblock=True, dropblock_keep_prob=dropblock_keep_prob)

    route_2 = conv
    conv = common.sepconvolutional(conv, (3, 3, 256, 512), dropblock=True, dropblock_keep_prob=dropblock_keep_prob)
    if include_head:
        conv_mbbox = common.convolutional(conv, (1, 1, 512, 3 * (NUM_CLASS + 5)), activate=False, bn=False)
    else:
        conv_mbbox = conv

    conv = common.sepconvolutional(route_2, (3, 3, 256, 512), downsample=True)
    conv = tf.concat([conv, route], axis=-1)

    conv = common.convolutional(conv, (1, 1, 1024, 512))
    conv = common.sepconvolutional(conv, (3, 3, 512, 1024))
    conv = common.convolutional(conv, (1, 1, 1024, 512))
    conv = common.sepconvolutional(conv, (3, 3, 512, 1024), dropblock=True, dropblock_keep_prob=dropblock_keep_prob)
    conv = common.convolutional(conv, (1, 1, 1024, 512), dropblock=True, dropblock_keep_prob=dropblock_keep_prob)

    conv = common.sepconvolutional(conv, (3, 3, 512, 1024), dropblock=True, dropblock_keep_prob=dropblock_keep_prob)
    if include_head:
        conv_lbbox = common.convolutional(conv, (1, 1, 1024, 3 * (NUM_CLASS + 5)), activate=False, bn=False)
    else:
        conv_lbbox = conv
    return [conv_sbbox, conv_mbbox, conv_lbbox]

def Yolov4_neck_small(route_1,
                route_2,
                conv ,
                NUM_CLASS,
                include_head=True,
                dropblock=False,
                dropblock_keep_prob=1,
                dtype=tf.float32):

    route = conv
    conv = common.convolutional(conv, (1, 1, 256, 128))
    conv = common.upsample(conv, dtype)
    route_2 = common.convolutional(route_2, (1, 1, 256, 128))
    conv = tf.concat([route_2, conv], axis=-1)

    conv = common.convolutional(conv, (1, 1, 256, 128))
    conv = common.convolutional(conv, (3, 3, 128, 256))
    conv = common.convolutional(conv, (1, 1, 256, 128))
    conv = common.convolutional(conv, (3, 3, 128, 256))
    conv = common.convolutional(conv, (1, 1, 256, 128))

    route_2 = conv
    conv = common.convolutional(conv, (1, 1, 128, 64))
    conv = common.upsample(conv, dtype)
    route_1 = common.convolutional(route_1, (1, 1, 128, 64))
    conv = tf.concat([route_1, conv], axis=-1)

    conv = common.convolutional(conv, (1, 1, 128, 64))
    conv = common.convolutional(conv, (3, 3, 64, 128))
    conv = common.convolutional(conv, (1, 1, 128, 64))
    conv = common.convolutional(conv, (3, 3, 64, 128), dropblock=True, dropblock_keep_prob=dropblock_keep_prob)
    conv = common.convolutional(conv, (1, 1, 128, 64), dropblock=True, dropblock_keep_prob=dropblock_keep_prob)

    route_1 = conv
    conv = common.convolutional(conv, (3, 3, 64, 128), dropblock=True, dropblock_keep_prob=dropblock_keep_prob)
    if include_head:
        conv_sbbox = common.convolutional(conv, (1, 1, 128, 3 * (NUM_CLASS + 5)), activate=False, bn=False)
    else:
        conv_sbbox = conv

    conv = common.convolutional(route_1, (3, 3, 64, 128), downsample=True)
    conv = tf.concat([conv, route_2], axis=-1)

    conv = common.convolutional(conv, (1, 1, 256, 128))
    conv = common.convolutional(conv, (3, 3, 128, 256))
    conv = common.convolutional(conv, (1, 1, 256, 128))
    conv = common.convolutional(conv, (3, 3, 128, 256), dropblock=True, dropblock_keep_prob=dropblock_keep_prob)
    conv = common.convolutional(conv, (1, 1, 256, 128), dropblock=True, dropblock_keep_prob=dropblock_keep_prob)

    route_2 = conv
    conv = common.convolutional(conv, (3, 3, 128, 256), dropblock=True, dropblock_keep_prob=dropblock_keep_prob)
    if include_head:
        conv_mbbox = common.convolutional(conv, (1, 1, 256, 3 * (NUM_CLASS + 5)), activate=False, bn=False)
    else:
        conv_mbbox = conv

    conv = common.convolutional(route_2, (3, 3, 128, 256), downsample=True)
    conv = tf.concat([conv, route], axis=-1)

    conv = common.convolutional(conv, (1, 1, 512, 256))
    conv = common.convolutional(conv, (3, 3, 256, 512))
    conv = common.convolutional(conv, (1, 1, 512, 256))
    conv = common.convolutional(conv, (3, 3, 256, 512), dropblock=True, dropblock_keep_prob=dropblock_keep_prob)
    conv = common.convolutional(conv, (1, 1, 512, 256), dropblock=True, dropblock_keep_prob=dropblock_keep_prob)

    conv = common.convolutional(conv, (3, 3, 256, 512), dropblock=True, dropblock_keep_prob=dropblock_keep_prob)
    if include_head:
        conv_lbbox = common.convolutional(conv, (1, 1, 512, 3 * (NUM_CLASS + 5)), activate=False, bn=False)
    else:
        conv_lbbox = conv
    return [conv_sbbox, conv_mbbox, conv_lbbox]

def YOLOv4_vit_v1(input_layer,
                  NUM_CLASS,
                  activation = 'gelu',
                  projection_dim = 128,
                  transformer_layers =[6, 6, 6],
                  attention_heads=[4, 4, 4],
                  spp = 0,
                  normal = 0):
    
    route_1, route_2, conv = backbone.VIT_v1(input_layer,
                                             projection_dim = projection_dim,
                                             transformer_layers =transformer_layers,
                                             attention_heads=attention_heads,
                                             activation=activation,
                                             normal=normal)
    if spp:
            x1 = common.convolutional(conv, (1, 1, 1024, 512))
            x2 = common.convolutional(x1, (3, 1, 512, 1024))
            x3 = common.convolutional(x2, (1, 1, 1024, 512))
            mxp1 = tf.keras.layers.MaxPool2D(pool_size = (5, 5), strides = 1, padding = 'same')(x3)
            mxp2 = tf.keras.layers.MaxPool2D(pool_size=(9, 9), strides = 1, padding = 'same')(x3)
            mxp3 = tf.keras.layers.MaxPool2D(pool_size = (13, 13), strides = 1, padding = 'same')(x3)
            spp = tf.keras.layers.concatenate([mxp1, mxp2, mxp3, x3], axis = -1)
            x4 = common.convolutional(spp, (1, 1, 2048, 512))
            x5 = common.convolutional(x4, (3, 1, 512, 1024))
            x6 = common.convolutional(x5, (1, 1, 1024, 512))
            conv = x6
    

    route = conv
    conv = common.convolutional(conv, (1, 1, 512, 256))
    conv = common.upsample(conv)
    route_2 = common.convolutional(route_2, (1, 1, 512, 256))
    conv = tf.concat([route_2, conv], axis=-1)

    conv = common.convolutional(conv, (1, 1, 512, 256))
    conv = common.convolutional(conv, (3, 3, 256, 512))
    conv = common.convolutional(conv, (1, 1, 512, 256))
    conv = common.convolutional(conv, (3, 3, 256, 512))
    conv = common.convolutional(conv, (1, 1, 512, 256))

    route_2 = conv
    conv = common.convolutional(conv, (1, 1, 256, 128))
    conv = common.upsample(conv)
    route_1 = common.convolutional(route_1, (1, 1, 256, 128))
    conv = tf.concat([route_1, conv], axis=-1)

    conv = common.convolutional(conv, (1, 1, 256, 128))
    conv = common.convolutional(conv, (3, 3, 128, 256))
    conv = common.convolutional(conv, (1, 1, 256, 128))
    conv = common.convolutional(conv, (3, 3, 128, 256))
    conv = common.convolutional(conv, (1, 1, 256, 128))

    route_1 = conv
    conv = common.convolutional(conv, (3, 3, 128, 256))
    conv_sbbox = common.convolutional(conv, (1, 1, 256, 3 * (NUM_CLASS + 5)), activate=False, bn=False)

    conv = common.convolutional(route_1, (3, 3, 128, 256), downsample=True)
    conv = tf.concat([conv, route_2], axis=-1)

    conv = common.convolutional(conv, (1, 1, 512, 256))
    conv = common.convolutional(conv, (3, 3, 256, 512))
    conv = common.convolutional(conv, (1, 1, 512, 256))
    conv = common.convolutional(conv, (3, 3, 256, 512))
    conv = common.convolutional(conv, (1, 1, 512, 256))

    route_2 = conv
    conv = common.convolutional(conv, (3, 3, 256, 512))
    conv_mbbox = common.convolutional(conv, (1, 1, 512, 3 * (NUM_CLASS + 5)), activate=False, bn=False)

    conv = common.convolutional(route_2, (3, 3, 256, 512), downsample=True)
    conv = tf.concat([conv, route], axis=-1)

    conv = common.convolutional(conv, (1, 1, 1024, 512))
    conv = common.convolutional(conv, (3, 3, 512, 1024))
    conv = common.convolutional(conv, (1, 1, 1024, 512))
    conv = common.convolutional(conv, (3, 3, 512, 1024))
    conv = common.convolutional(conv, (1, 1, 1024, 512))

    conv = common.convolutional(conv, (3, 3, 512, 1024))
    conv_lbbox = common.convolutional(conv, (1, 1, 1024, 3 * (NUM_CLASS + 5)), activate=False, bn=False)

    return [conv_sbbox, conv_mbbox, conv_lbbox]


def YOLOv4_vit_v2(input_layer,
                  NUM_CLASS,
                  activation = 'gelu',
                  projection_dim = 128,
                  transformer_layers =[6, 6, 6],
                  attention_heads=[4, 4, 4],
                  spp=0,
                  normal=0):
    
    route_1, route_2, conv = backbone.VIT_v2(input_layer,
                                             projection_dim = projection_dim,
                                             transformer_layers =transformer_layers,
                                             attention_heads=attention_heads,
                                             activation = activation,
                                             normal = normal)
    
    conv    = common.convolutional(conv, (3, 3, 128, 256))
    conv    = common.convolutional(conv, (3, 3, 128, 512), downsample=True)
    conv    = common.convolutional(conv, (3, 3, 128, 1024), downsample=True)
    route_2 = common.convolutional(route_2, (3, 3, 128, 256), downsample=True)
    route_2 = common.convolutional(route_2, (3, 3, 128, 512))

    
    if spp:
            x1 = common.convolutional(conv, (1, 1, 1024, 512))
            x2 = common.convolutional(x1, (3, 1, 512, 1024))
            x3 = common.convolutional(x2, (1, 1, 1024, 512))
            mxp1 = tf.keras.layers.MaxPool2D(pool_size = (5, 5), strides = 1, padding = 'same')(x3)
            mxp2 = tf.keras.layers.MaxPool2D(pool_size=(9, 9), strides = 1, padding = 'same')(x3)
            mxp3 = tf.keras.layers.MaxPool2D(pool_size = (13, 13), strides = 1, padding = 'same')(x3)
            spp = tf.keras.layers.concatenate([mxp1, mxp2, mxp3, x3], axis = -1)
            x4 = common.convolutional(spp, (1, 1, 2048, 512))
            x5 = common.convolutional(x4, (3, 1, 512, 1024))
            x6 = common.convolutional(x5, (1, 1, 1024, 512))
            conv = x6
    

    route = conv
    conv = common.convolutional(conv, (1, 1, 512, 256))
    conv = common.upsample(conv)
    route_2 = common.convolutional(route_2, (1, 1, 512, 256))
    conv = tf.concat([route_2, conv], axis=-1)

    conv = common.convolutional(conv, (1, 1, 512, 256))
    conv = common.convolutional(conv, (3, 3, 256, 512))
    conv = common.convolutional(conv, (1, 1, 512, 256))
    conv = common.convolutional(conv, (3, 3, 256, 512))
    conv = common.convolutional(conv, (1, 1, 512, 256))

    route_2 = conv
    conv = common.convolutional(conv, (1, 1, 256, 128))
    conv = common.upsample(conv)
    route_1 = common.convolutional(route_1, (1, 1, 256, 128))
    conv = tf.concat([route_1, conv], axis=-1)

    conv = common.convolutional(conv, (1, 1, 256, 128))
    conv = common.convolutional(conv, (3, 3, 128, 256))
    conv = common.convolutional(conv, (1, 1, 256, 128))
    conv = common.convolutional(conv, (3, 3, 128, 256))
    conv = common.convolutional(conv, (1, 1, 256, 128))

    route_1 = conv
    conv = common.convolutional(conv, (3, 3, 128, 256))
    conv_sbbox = common.convolutional(conv, (1, 1, 256, 3 * (NUM_CLASS + 5)), activate=False, bn=False)

    conv = common.convolutional(route_1, (3, 3, 128, 256), downsample=True)
    conv = tf.concat([conv, route_2], axis=-1)

    conv = common.convolutional(conv, (1, 1, 512, 256))
    conv = common.convolutional(conv, (3, 3, 256, 512))
    conv = common.convolutional(conv, (1, 1, 512, 256))
    conv = common.convolutional(conv, (3, 3, 256, 512))
    conv = common.convolutional(conv, (1, 1, 512, 256))

    route_2 = conv
    conv = common.convolutional(conv, (3, 3, 256, 512))
    conv_mbbox = common.convolutional(conv, (1, 1, 512, 3 * (NUM_CLASS + 5)), activate=False, bn=False)

    conv = common.convolutional(route_2, (3, 3, 256, 512), downsample=True)
    conv = tf.concat([conv, route], axis=-1)

    conv = common.convolutional(conv, (1, 1, 1024, 512))
    conv = common.convolutional(conv, (3, 3, 512, 1024))
    conv = common.convolutional(conv, (1, 1, 1024, 512))
    conv = common.convolutional(conv, (3, 3, 512, 1024))
    conv = common.convolutional(conv, (1, 1, 1024, 512))

    conv = common.convolutional(conv, (3, 3, 512, 1024))
    conv_lbbox = common.convolutional(conv, (1, 1, 1024, 3 * (NUM_CLASS + 5)), activate=False, bn=False)

    return [conv_sbbox, conv_mbbox, conv_lbbox]

def YOLOv4_vit_v3(input_layer,
                  NUM_CLASS,
                  activation = 'gelu',
                  projection_dim = 128,
                  transformer_layers =[6, 6, 6],
                  attention_heads=[4, 4, 4],
                  spp = 0,
                  normal = 0):
    if normal == 6:
        normal = 5
        norm = 1
    else:
        norm = 0
    route_1, route_2, conv = backbone.VIT_v3(input_layer,
                                             projection_dim = projection_dim,
                                             transformer_layers =transformer_layers,
                                             attention_heads=attention_heads,
                                             activation=activation,
                                             normal=normal)
    if spp:
            x1 = common.convolutional(conv, (1, 1, 1024, 512), norm = norm)
            x2 = common.convolutional(x1, (3, 1, 512, 1024), norm = norm)
            x3 = common.convolutional(x2, (1, 1, 1024, 512), norm = norm)
            mxp1 = tf.keras.layers.MaxPool2D(pool_size = (5, 5), strides = 1, padding = 'same')(x3)
            mxp2 = tf.keras.layers.MaxPool2D(pool_size=(9, 9), strides = 1, padding = 'same')(x3)
            mxp3 = tf.keras.layers.MaxPool2D(pool_size = (13, 13), strides = 1, padding = 'same')(x3)
            spp = tf.keras.layers.concatenate([mxp1, mxp2, mxp3, x3], axis = -1)
            x4 = common.convolutional(spp, (1, 1, 2048, 512), norm = norm)
            x5 = common.convolutional(x4, (3, 1, 512, 1024), norm = norm)
            x6 = common.convolutional(x5, (1, 1, 1024, 512), norm = norm)
            conv = x6
    

    route = conv
    conv = common.convolutional(conv, (1, 1, 512, 256), norm = norm)
    conv = common.upsample(conv)
    route_2 = common.convolutional(route_2, (1, 1, 512, 256), norm = norm)
    conv = tf.concat([route_2, conv], axis=-1)

    conv = common.convolutional(conv, (1, 1, 512, 256), norm = norm)
    conv = common.convolutional(conv, (3, 3, 256, 512), norm = norm)
    conv = common.convolutional(conv, (1, 1, 512, 256), norm = norm)
    conv = common.convolutional(conv, (3, 3, 256, 512), norm = norm)
    conv = common.convolutional(conv, (1, 1, 512, 256), norm = norm)

    route_2 = conv
    conv = common.convolutional(conv, (1, 1, 256, 128), norm = norm)
    conv = common.upsample(conv)
    route_1 = common.convolutional(route_1, (1, 1, 256, 128), norm = norm)
    conv = tf.concat([route_1, conv], axis=-1)

    conv = common.convolutional(conv, (1, 1, 256, 128), norm = norm)
    conv = common.convolutional(conv, (3, 3, 128, 256), norm = norm)
    conv = common.convolutional(conv, (1, 1, 256, 128), norm = norm)
    conv = common.convolutional(conv, (3, 3, 128, 256), norm = norm)
    conv = common.convolutional(conv, (1, 1, 256, 128), norm = norm)

    route_1 = conv
    conv = common.convolutional(conv, (3, 3, 128, 256), norm = norm)
    conv_sbbox = common.convolutional(conv, (1, 1, 256, 3 * (NUM_CLASS + 5)), activate=False, bn=False)

    conv = common.convolutional(route_1, (3, 3, 128, 256), downsample=True, norm = norm)
    conv = tf.concat([conv, route_2], axis=-1)

    conv = common.convolutional(conv, (1, 1, 512, 256), norm = norm)
    conv = common.convolutional(conv, (3, 3, 256, 512), norm = norm)
    conv = common.convolutional(conv, (1, 1, 512, 256), norm = norm)
    conv = common.convolutional(conv, (3, 3, 256, 512), norm = norm)
    conv = common.convolutional(conv, (1, 1, 512, 256), norm = norm)

    route_2 = conv
    conv = common.convolutional(conv, (3, 3, 256, 512), norm = norm)
    conv_mbbox = common.convolutional(conv, (1, 1, 512, 3 * (NUM_CLASS + 5)), activate=False, bn=False)

    conv = common.convolutional(route_2, (3, 3, 256, 512), downsample=True, norm = norm)
    conv = tf.concat([conv, route], axis=-1)

    conv = common.convolutional(conv, (1, 1, 1024, 512), norm = norm)
    conv = common.convolutional(conv, (3, 3, 512, 1024), norm = norm)
    conv = common.convolutional(conv, (1, 1, 1024, 512), norm = norm)
    conv = common.convolutional(conv, (3, 3, 512, 1024), norm = norm)
    conv = common.convolutional(conv, (1, 1, 1024, 512), norm = norm)

    conv = common.convolutional(conv, (3, 3, 512, 1024), norm = norm)
    conv_lbbox = common.convolutional(conv, (1, 1, 1024, 3 * (NUM_CLASS + 5)), activate=False, bn=False)

    return [conv_sbbox, conv_mbbox, conv_lbbox]

def YOLOv4_vit_v5(input_layer,
                  NUM_CLASS,
                  activation = 'gelu',
                  projection_dim = 128,
                  transformer_layers =[6, 6, 6],
                  attention_heads=[4, 4, 4],
                  spp = 0,
                  normal = 0):
    if normal == 6:
        normal = 5
        norm = 1
    else:
        norm = 0
    route_1, route_2, conv = backbone.VIT_v5(input_layer,
                                             projection_dim = projection_dim,
                                             transformer_layers =transformer_layers,
                                             attention_heads=attention_heads,
                                             activation=activation,
                                             normal=normal)
    if spp:
            x1 = common.convolutional(conv, (1, 1, 1024, 512), norm = norm)
            x2 = common.convolutional(x1, (3, 1, 512, 1024), norm = norm)
            x3 = common.convolutional(x2, (1, 1, 1024, 512), norm = norm)
            mxp1 = tf.keras.layers.MaxPool2D(pool_size = (5, 5), strides = 1, padding = 'same')(x3)
            mxp2 = tf.keras.layers.MaxPool2D(pool_size=(9, 9), strides = 1, padding = 'same')(x3)
            mxp3 = tf.keras.layers.MaxPool2D(pool_size = (13, 13), strides = 1, padding = 'same')(x3)
            spp = tf.keras.layers.concatenate([mxp1, mxp2, mxp3, x3], axis = -1)
            x4 = common.convolutional(spp, (1, 1, 2048, 512), norm = norm)
            x5 = common.convolutional(x4, (3, 1, 512, 1024), norm = norm)
            x6 = common.convolutional(x5, (1, 1, 1024, 512), norm = norm)
            conv = x6
    

    route = conv
    conv = common.convolutional(conv, (1, 1, 512, 256), norm = norm)
    conv = common.upsample(conv)
    route_2 = common.convolutional(route_2, (1, 1, 512, 256), norm = norm)
    conv = tf.concat([route_2, conv], axis=-1)

    conv = common.convolutional(conv, (1, 1, 512, 256), norm = norm)
    conv = common.convolutional(conv, (3, 3, 256, 512), norm = norm)
    conv = common.convolutional(conv, (1, 1, 512, 256), norm = norm)
    conv = common.convolutional(conv, (3, 3, 256, 512), norm = norm)
    conv = common.convolutional(conv, (1, 1, 512, 256), norm = norm)

    route_2 = conv
    conv = common.convolutional(conv, (1, 1, 256, 128), norm = norm)
    conv = common.upsample(conv)
    route_1 = common.convolutional(route_1, (1, 1, 256, 128), norm = norm)
    conv = tf.concat([route_1, conv], axis=-1)

    conv = common.convolutional(conv, (1, 1, 256, 128), norm = norm)
    conv = common.convolutional(conv, (3, 3, 128, 256), norm = norm)
    conv = common.convolutional(conv, (1, 1, 256, 128), norm = norm)
    conv = common.convolutional(conv, (3, 3, 128, 256), norm = norm)
    conv = common.convolutional(conv, (1, 1, 256, 128), norm = norm)

    route_1 = conv
    conv = common.convolutional(conv, (3, 3, 128, 256), norm = norm)
    conv_sbbox = common.convolutional(conv, (1, 1, 256, 3 * (NUM_CLASS + 5)), activate=False, bn=False)

    conv = common.convolutional(route_1, (3, 3, 128, 256), downsample=True, norm = norm)
    conv = tf.concat([conv, route_2], axis=-1)

    conv = common.convolutional(conv, (1, 1, 512, 256), norm = norm)
    conv = common.convolutional(conv, (3, 3, 256, 512), norm = norm)
    conv = common.convolutional(conv, (1, 1, 512, 256), norm = norm)
    conv = common.convolutional(conv, (3, 3, 256, 512), norm = norm)
    conv = common.convolutional(conv, (1, 1, 512, 256), norm = norm)

    route_2 = conv
    conv = common.convolutional(conv, (3, 3, 256, 512), norm = norm)
    conv_mbbox = common.convolutional(conv, (1, 1, 512, 3 * (NUM_CLASS + 5)), activate=False, bn=False)

    conv = common.convolutional(route_2, (3, 3, 256, 512), downsample=True, norm = norm)
    conv = tf.concat([conv, route], axis=-1)

    conv = common.convolutional(conv, (1, 1, 1024, 512), norm = norm)
    conv = common.convolutional(conv, (3, 3, 512, 1024), norm = norm)
    conv = common.convolutional(conv, (1, 1, 1024, 512), norm = norm)
    conv = common.convolutional(conv, (3, 3, 512, 1024), norm = norm)
    conv = common.convolutional(conv, (1, 1, 1024, 512), norm = norm)

    conv = common.convolutional(conv, (3, 3, 512, 1024), norm = norm)
    conv_lbbox = common.convolutional(conv, (1, 1, 1024, 3 * (NUM_CLASS + 5)), activate=False, bn=False)

    return [conv_sbbox, conv_mbbox, conv_lbbox]


def YOLOv4_vit_v1_light(input_layer, NUM_CLASS, activation = 'gelu'):
    route_1, route_2, conv = backbone.VIT_v1(input_layer,
                                  transformer_layers =[4, 4, 4], activation = activation)

    route = conv
    conv = common.convolutional(conv, (1, 1, 512, 256))
    conv = common.upsample(conv)
    route_2 = common.convolutional(route_2, (1, 1, 512, 256))
    conv = tf.concat([route_2, conv], axis=-1)

    conv = common.convolutional(conv, (1, 1, 512, 256))
    conv = common.convolutional(conv, (3, 3, 256, 512))
    conv = common.convolutional(conv, (1, 1, 512, 256))
    conv = common.convolutional(conv, (3, 3, 256, 512))
    conv = common.convolutional(conv, (1, 1, 512, 256))

    route_2 = conv
    conv = common.convolutional(conv, (1, 1, 256, 128))
    conv = common.upsample(conv)
    route_1 = common.convolutional(route_1, (1, 1, 256, 128))
    conv = tf.concat([route_1, conv], axis=-1)

    conv = common.convolutional(conv, (1, 1, 256, 128))
    conv = common.convolutional(conv, (3, 3, 128, 256))
    conv = common.convolutional(conv, (1, 1, 256, 128))
    conv = common.convolutional(conv, (3, 3, 128, 256))
    conv = common.convolutional(conv, (1, 1, 256, 128))

    route_1 = conv
    conv = common.convolutional(conv, (3, 3, 128, 256))
    conv_sbbox = common.convolutional(conv, (1, 1, 256, 3 * (NUM_CLASS + 5)), activate=False, bn=False)

    conv = common.convolutional(route_1, (3, 3, 128, 256), downsample=True)
    conv = tf.concat([conv, route_2], axis=-1)

    conv = common.convolutional(conv, (1, 1, 512, 256))
    conv = common.convolutional(conv, (3, 3, 256, 512))
    conv = common.convolutional(conv, (1, 1, 512, 256))
    conv = common.convolutional(conv, (3, 3, 256, 512))
    conv = common.convolutional(conv, (1, 1, 512, 256))

    route_2 = conv
    conv = common.convolutional(conv, (3, 3, 256, 512))
    conv_mbbox = common.convolutional(conv, (1, 1, 512, 3 * (NUM_CLASS + 5)), activate=False, bn=False)

    conv = common.convolutional(route_2, (3, 3, 256, 512), downsample=True)
    conv = tf.concat([conv, route], axis=-1)

    conv = common.convolutional(conv, (1, 1, 1024, 512))
    conv = common.convolutional(conv, (3, 3, 512, 1024))
    conv = common.convolutional(conv, (1, 1, 1024, 512))
    conv = common.convolutional(conv, (3, 3, 512, 1024))
    conv = common.convolutional(conv, (1, 1, 1024, 512))

    conv = common.convolutional(conv, (3, 3, 512, 1024))
    conv_lbbox = common.convolutional(conv, (1, 1, 1024, 3 * (NUM_CLASS + 5)), activate=False, bn=False)

    return [conv_sbbox, conv_mbbox, conv_lbbox]

def YOLOv4_tiny(input_layer, NUM_CLASS):
    route_1, conv = backbone.cspdarknet53_tiny(input_layer)

    conv = common.convolutional(conv, (1, 1, 512, 256))

    conv_lobj_branch = common.convolutional(conv, (3, 3, 256, 512))
    conv_lbbox = common.convolutional(conv_lobj_branch, (1, 1, 512, 3 * (NUM_CLASS + 5)), activate=False, bn=False)

    conv = common.convolutional(conv, (1, 1, 256, 128))
    conv = common.upsample(conv)
    conv = tf.concat([conv, route_1], axis=-1)

    conv_mobj_branch = common.convolutional(conv, (3, 3, 128, 256))
    conv_mbbox = common.convolutional(conv_mobj_branch, (1, 1, 256, 3 * (NUM_CLASS + 5)), activate=False, bn=False)

    return [conv_mbbox, conv_lbbox]

def YOLOv4_vit_v1_tiny(input_layer, NUM_CLASS):
    route_1, conv = backbone.VIT_v1_tiny(input_layer)

    conv = common.convolutional(conv, (1, 1, 512, 256))

    conv_lobj_branch = common.convolutional(conv, (3, 3, 256, 512))
    conv_lbbox = common.convolutional(conv_lobj_branch, (1, 1, 512, 3 * (NUM_CLASS + 5)), activate=False, bn=False)

    conv = common.convolutional(conv, (1, 1, 256, 128))
    conv = common.upsample(conv)
    conv = tf.concat([conv, route_1], axis=-1)

    conv_mobj_branch = common.convolutional(conv, (3, 3, 128, 256))
    conv_mbbox = common.convolutional(conv_mobj_branch, (1, 1, 256, 3 * (NUM_CLASS + 5)), activate=False, bn=False)

    return [conv_mbbox, conv_lbbox]

def YOLOv3_tiny(input_layer, NUM_CLASS):
    route_1, conv = backbone.darknet53_tiny(input_layer)

    conv = common.convolutional(conv, (1, 1, 1024, 256))

    conv_lobj_branch = common.convolutional(conv, (3, 3, 256, 512))
    conv_lbbox = common.convolutional(conv_lobj_branch, (1, 1, 512, 3 * (NUM_CLASS + 5)), activate=False, bn=False)

    conv = common.convolutional(conv, (1, 1, 256, 128))
    conv = common.upsample(conv)
    conv = tf.concat([conv, route_1], axis=-1)

    conv_mobj_branch = common.convolutional(conv, (3, 3, 128, 256))
    conv_mbbox = common.convolutional(conv_mobj_branch, (1, 1, 256, 3 * (NUM_CLASS + 5)), activate=False, bn=False)

    return [conv_mbbox, conv_lbbox]

def decode(conv_output, output_size, NUM_CLASS, STRIDES, ANCHORS, i, XYSCALE=[1,1,1], FRAMEWORK='tf'):
    if FRAMEWORK == 'trt':
        return decode_trt(conv_output, output_size, NUM_CLASS, STRIDES, ANCHORS, i=i, XYSCALE=XYSCALE)
    elif FRAMEWORK == 'tflite':
        return decode_tflite(conv_output, output_size, NUM_CLASS, STRIDES, ANCHORS, i=i, XYSCALE=XYSCALE)
    else:
        return decode_tf(conv_output, output_size, NUM_CLASS, STRIDES, ANCHORS, i=i, XYSCALE=XYSCALE)

def decode_train(conv_output, output_size, NUM_CLASS, STRIDES, ANCHORS, i=0, XYSCALE=[1, 1, 1]):
    conv_output = tf.reshape(conv_output,
                             (tf.shape(conv_output)[0], output_size, output_size, 3, 5 + NUM_CLASS))

    conv_raw_dxdy, conv_raw_dwdh, conv_raw_conf, conv_raw_prob = tf.split(conv_output, (2, 2, 1, NUM_CLASS),
                                                                          axis=-1)

    xy_grid = tf.meshgrid(tf.range(output_size), tf.range(output_size))
    xy_grid = tf.expand_dims(tf.stack(xy_grid, axis=-1), axis=2)  # [gx, gy, 1, 2]
    xy_grid = tf.tile(tf.expand_dims(xy_grid, axis=0), [tf.shape(conv_output)[0], 1, 1, 3, 1])

    xy_grid = tf.cast(xy_grid, conv_output.dtype)

    pred_xy = ((tf.sigmoid(conv_raw_dxdy) * XYSCALE[i]) - 0.5 * (XYSCALE[i] - 1) + xy_grid) * \
              STRIDES[i]
    pred_wh = (tf.exp(conv_raw_dwdh) * ANCHORS[i])
    pred_xywh = tf.concat([pred_xy, pred_wh], axis=-1)

    pred_conf = tf.sigmoid(conv_raw_conf)
    pred_prob = tf.sigmoid(conv_raw_prob)

    return tf.concat([pred_xywh, pred_conf, pred_prob], axis=-1)

def decode_tf(conv_output, output_size, NUM_CLASS, STRIDES, ANCHORS, i=0, XYSCALE=[1, 1, 1]):
    batch_size = tf.shape(conv_output)[0]
    conv_output = tf.reshape(conv_output,
                             (batch_size, output_size, output_size, 3, 5 + NUM_CLASS))

    conv_raw_dxdy, conv_raw_dwdh, conv_raw_conf, conv_raw_prob = tf.split(conv_output, (2, 2, 1, NUM_CLASS),
                                                                          axis=-1)

    xy_grid = tf.meshgrid(tf.range(output_size), tf.range(output_size))
    xy_grid = tf.expand_dims(tf.stack(xy_grid, axis=-1), axis=2)  # [gx, gy, 1, 2]
    xy_grid = tf.tile(tf.expand_dims(xy_grid, axis=0), [batch_size, 1, 1, 3, 1])

    xy_grid = tf.cast(xy_grid, tf.float32)

    pred_xy = ((tf.sigmoid(conv_raw_dxdy) * XYSCALE[i]) - 0.5 * (XYSCALE[i] - 1) + xy_grid) * \
              STRIDES[i]
    pred_wh = (tf.exp(conv_raw_dwdh) * ANCHORS[i])
    pred_xywh = tf.concat([pred_xy, pred_wh], axis=-1)

    pred_conf = tf.sigmoid(conv_raw_conf)
    pred_prob = tf.sigmoid(conv_raw_prob)

    pred_prob = pred_conf * pred_prob
    pred_prob = tf.reshape(pred_prob, (batch_size, -1, NUM_CLASS))
    pred_xywh = tf.reshape(pred_xywh, (batch_size, -1, 4))

    return pred_xywh, pred_prob
    # return tf.concat([pred_xywh, pred_conf, pred_prob], axis=-1)

def decode_tflite(conv_output, output_size, NUM_CLASS, STRIDES, ANCHORS, i=0, XYSCALE=[1,1,1]):
    conv_raw_dxdy_0, conv_raw_dwdh_0, conv_raw_score_0,\
    conv_raw_dxdy_1, conv_raw_dwdh_1, conv_raw_score_1,\
    conv_raw_dxdy_2, conv_raw_dwdh_2, conv_raw_score_2 = tf.split(conv_output, (2, 2, 1+NUM_CLASS, 2, 2, 1+NUM_CLASS,
                                                                                2, 2, 1+NUM_CLASS), axis=-1)

    conv_raw_score = [conv_raw_score_0, conv_raw_score_1, conv_raw_score_2]
    for idx, score in enumerate(conv_raw_score):
        score = tf.sigmoid(score)
        score = score[:, :, :, 0:1] * score[:, :, :, 1:]
        conv_raw_score[idx] = tf.reshape(score, (1, -1, NUM_CLASS))
    pred_prob = tf.concat(conv_raw_score, axis=1)

    conv_raw_dwdh = [conv_raw_dwdh_0, conv_raw_dwdh_1, conv_raw_dwdh_2]
    for idx, dwdh in enumerate(conv_raw_dwdh):
        dwdh = tf.exp(dwdh) * ANCHORS[i][idx]
        conv_raw_dwdh[idx] = tf.reshape(dwdh, (1, -1, 2))
    pred_wh = tf.concat(conv_raw_dwdh, axis=1)

    xy_grid = tf.meshgrid(tf.range(output_size), tf.range(output_size))
    xy_grid = tf.stack(xy_grid, axis=-1)  # [gx, gy, 2]
    xy_grid = tf.expand_dims(xy_grid, axis=0)
    xy_grid = tf.cast(xy_grid, tf.float32)

    conv_raw_dxdy = [conv_raw_dxdy_0, conv_raw_dxdy_1, conv_raw_dxdy_2]
    for idx, dxdy in enumerate(conv_raw_dxdy):
        dxdy = ((tf.sigmoid(dxdy) * XYSCALE[i]) - 0.5 * (XYSCALE[i] - 1) + xy_grid) * \
              STRIDES[i]
        conv_raw_dxdy[idx] = tf.reshape(dxdy, (1, -1, 2))
    pred_xy = tf.concat(conv_raw_dxdy, axis=1)
    pred_xywh = tf.concat([pred_xy, pred_wh], axis=-1)
    return pred_xywh, pred_prob
    # return tf.concat([pred_xywh, pred_conf, pred_prob], axis=-1)

def decode_trt(conv_output, output_size, NUM_CLASS, STRIDES, ANCHORS, i=0, XYSCALE=[1,1,1]):
    batch_size = tf.shape(conv_output)[0]
    conv_output = tf.reshape(conv_output, (batch_size, output_size, output_size, 3, 5 + NUM_CLASS))

    conv_raw_dxdy, conv_raw_dwdh, conv_raw_conf, conv_raw_prob = tf.split(conv_output, (2, 2, 1, NUM_CLASS), axis=-1)

    xy_grid = tf.meshgrid(tf.range(output_size), tf.range(output_size))
    xy_grid = tf.expand_dims(tf.stack(xy_grid, axis=-1), axis=2)  # [gx, gy, 1, 2]
    xy_grid = tf.tile(tf.expand_dims(xy_grid, axis=0), [batch_size, 1, 1, 3, 1])

    # x = tf.tile(tf.expand_dims(tf.range(output_size, dtype=tf.float32), axis=0), [output_size, 1])
    # y = tf.tile(tf.expand_dims(tf.range(output_size, dtype=tf.float32), axis=1), [1, output_size])
    # xy_grid = tf.expand_dims(tf.stack([x, y], axis=-1), axis=2)  # [gx, gy, 1, 2]
    # xy_grid = tf.tile(tf.expand_dims(xy_grid, axis=0), [tf.shape(conv_output)[0], 1, 1, 3, 1])

    xy_grid = tf.cast(xy_grid, tf.float32)

    # pred_xy = ((tf.sigmoid(conv_raw_dxdy) * XYSCALE[i]) - 0.5 * (XYSCALE[i] - 1) + xy_grid) * \
    #           STRIDES[i]
    pred_xy = (tf.reshape(tf.sigmoid(conv_raw_dxdy), (-1, 2)) * XYSCALE[i] - 0.5 * (XYSCALE[i] - 1) + tf.reshape(xy_grid, (-1, 2))) * STRIDES[i]
    pred_xy = tf.reshape(pred_xy, (batch_size, output_size, output_size, 3, 2))
    pred_wh = (tf.exp(conv_raw_dwdh) * ANCHORS[i])
    pred_xywh = tf.concat([pred_xy, pred_wh], axis=-1)

    pred_conf = tf.sigmoid(conv_raw_conf)
    pred_prob = tf.sigmoid(conv_raw_prob)

    pred_prob = pred_conf * pred_prob

    pred_prob = tf.reshape(pred_prob, (batch_size, -1, NUM_CLASS))
    pred_xywh = tf.reshape(pred_xywh, (batch_size, -1, 4))
    return pred_xywh, pred_prob
    # return tf.concat([pred_xywh, pred_conf, pred_prob], axis=-1)


def filter_boxes(box_xywh, scores, score_threshold=0.4, input_shape = tf.constant([416,416])):
    scores_max = tf.math.reduce_max(scores, axis=-1)

    mask = scores_max >= score_threshold
    class_boxes = tf.boolean_mask(box_xywh, mask)
    pred_conf = tf.boolean_mask(scores, mask)
    class_boxes = tf.reshape(class_boxes, [tf.shape(scores)[0], -1, tf.shape(class_boxes)[-1]])
    pred_conf = tf.reshape(pred_conf, [tf.shape(scores)[0], -1, tf.shape(pred_conf)[-1]])

    box_xy, box_wh = tf.split(class_boxes, (2, 2), axis=-1)

    input_shape = tf.cast(input_shape, dtype=tf.float32)

    box_yx = box_xy[..., ::-1]
    box_hw = box_wh[..., ::-1]

    box_mins = (box_yx - (box_hw / 2.)) / input_shape
    box_maxes = (box_yx + (box_hw / 2.)) / input_shape
    boxes = tf.concat([
        box_mins[..., 0:1],  # y_min
        box_mins[..., 1:2],  # x_min
        box_maxes[..., 0:1],  # y_max
        box_maxes[..., 1:2]  # x_max
    ], axis=-1)
    # return tf.concat([boxes, pred_conf], axis=-1)
    return (boxes, pred_conf)


def compute_loss(pred, conv, label, bboxes, STRIDES, NUM_CLASS, IOU_LOSS_THRESH, i=0):
    conv_shape  = tf.shape(conv)
    batch_size  = conv_shape[0]
    output_size = conv_shape[1]
    input_size  = STRIDES[i] * tf.cast(output_size, conv.dtype)
    conv = tf.reshape(conv, (batch_size, output_size, output_size, 3, 5 + NUM_CLASS))

    conv_raw_conf = conv[:, :, :, :, 4:5]
    conv_raw_prob = conv[:, :, :, :, 5:]

    pred_xywh     = pred[:, :, :, :, 0:4]
    pred_conf     = pred[:, :, :, :, 4:5]

    label_xywh    = label[:, :, :, :, 0:4]
    respond_bbox  = label[:, :, :, :, 4:5]
    label_prob    = label[:, :, :, :, 5:]

    giou = tf.expand_dims(utils.bbox_giou(pred_xywh, label_xywh), axis=-1)
    input_size = tf.cast(input_size, conv.dtype)

    bbox_loss_scale = 2.0 - 1.0 * label_xywh[:, :, :, :, 2:3] * label_xywh[:, :, :, :, 3:4] / (input_size ** 2)
    giou_loss = respond_bbox * bbox_loss_scale * (1- giou)

    iou = utils.bbox_iou(pred_xywh[:, :, :, :, tf.newaxis, :], bboxes[:, tf.newaxis, tf.newaxis, tf.newaxis, :, :])
    max_iou = tf.expand_dims(tf.reduce_max(iou, axis=-1), axis=-1)

    respond_bgd = (1.0 - respond_bbox) * tf.cast( max_iou < IOU_LOSS_THRESH, conv.dtype)

    conf_focal = tf.pow(respond_bbox - pred_conf, 2)

    conf_loss = conf_focal * (
            respond_bbox * tf.nn.sigmoid_cross_entropy_with_logits(labels=respond_bbox, logits=conv_raw_conf)
            +
            respond_bgd * tf.nn.sigmoid_cross_entropy_with_logits(labels=respond_bbox, logits=conv_raw_conf)
    )

    prob_loss = respond_bbox * tf.nn.sigmoid_cross_entropy_with_logits(labels=label_prob, logits=conv_raw_prob)

    giou_loss = (tf.reduce_sum(giou_loss, axis=[1,2,3,4]))
    conf_loss = (tf.reduce_sum(conf_loss, axis=[1,2,3,4]))
    prob_loss = (tf.reduce_sum(prob_loss, axis=[1,2,3,4]))

    return giou_loss, conf_loss, prob_loss





