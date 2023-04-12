#! /usr/bin/env python
# coding=utf-8
import math
import tensorflow as tf
from core.config import cfg

# import tensorflow_addons as tfa
class BatchNormalization(tf.keras.layers.BatchNormalization):
    """
    "Frozen state" and "inference mode" are two separate concepts.
    `layer.trainable = False` is to freeze the layer, so the layer will use
    stored moving `var` and `mean` in the "inference mode", and both `gama`
    and `beta` will not be updated !
    """
    def call(self, x, training=False):
        if not training:
            training = tf.constant(False)
        training = tf.logical_and(training, self.trainable)
        return super().call(x, training)

def convolutional(input_layer, filters_shape, downsample=False, activate=True, bn=True, activate_type='leaky'):
    if downsample:
        input_layer = tf.keras.layers.ZeroPadding2D(((1, 0), (1, 0)))(input_layer)
        padding = 'valid'
        strides = 2
    else:
        strides = 1
        padding = 'same'

    conv = tf.keras.layers.Conv2D(filters=filters_shape[-1], kernel_size = filters_shape[0], strides=strides, padding=padding,
                                  use_bias=not bn, kernel_regularizer=tf.keras.regularizers.l2(cfg.TRAIN.WEIGHT_DECAY),
                                  kernel_initializer=tf.random_normal_initializer(stddev=0.01),
                                  bias_initializer=tf.constant_initializer(0.))(input_layer)

    if bn: conv = BatchNormalization()(conv)
    if activate == True:
        if activate_type == "leaky":
            conv = tf.nn.leaky_relu(conv, alpha=0.1)
        elif activate_type == "mish":
            conv = mish(conv)
    return conv

def mish(x):
    return x * tf.math.tanh(tf.math.softplus(x))
    # return tf.keras.layers.Lambda(lambda x: x*tf.tanh(tf.math.log(1+tf.exp(x))))(x)

def residual_block(input_layer, input_channel, filter_num1, filter_num2, activate_type='leaky'):
    short_cut = input_layer
    conv = convolutional(input_layer, filters_shape=(1, 1, input_channel, filter_num1), activate_type=activate_type)
    conv = convolutional(conv       , filters_shape=(3, 3, filter_num1,   filter_num2), activate_type=activate_type)

    residual_output = short_cut + conv
    return residual_output

# def block_tiny(input_layer, input_channel, filter_num1, activate_type='leaky'):
#     conv = convolutional(input_layer, filters_shape=(3, 3, input_channel, filter_num1), activate_type=activate_type)
#     short_cut = input_layer
#     conv = convolutional(conv, filters_shape=(3, 3, input_channel, filter_num1), activate_type=activate_type)
#
#     input_data = tf.concat([conv, short_cut], axis=-1)
#     return residual_output

def route_group(input_layer, groups, group_id):
    convs = tf.split(input_layer, num_or_size_splits=groups, axis=-1)
    return convs[group_id]

def upsample(input_layer):
    return tf.image.resize(input_layer, (tf.shape(input_layer)[1] * 2, tf.shape(input_layer)[2] * 2), method='bilinear')

def autopad(k, p=None, d=1):  # kernel, padding, dilation
    # Pad to 'same' shape outputs
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]  # actual kernel-size
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p

class Conv(tf.keras.layers.Layer):
    def __init__(self, filters, kernel_size=3, strides=1, padding=None, dilation_rate=1, groups=1, activation=None, bn=True):
        super(Conv, self).__init__()
        self.filters     = filters
        self.kernel_size = kernel_size 
        self.strides     = strides
        self.dilatation  = dilation_rate
        self.padding     = autopad(kernel_size, padding, self.dilatation)
        self.groups      = groups
        self.bn          = bn
        self.linear      = lambda x: x
        self.activation  = activation if activation is not None else self.linear
    def build(self, input_shape):
        self.conv = tf.keras.layers.Conv2D(filters=self.filters,
                                            kernel_size=self.kernel_size,
                                            strides=self.strides,
                                            padding='valid',
                                            dilation_rate=self.dilatation,
                                            groups=self.groups,
                                            activation=None,
                                            use_bias=not self.bn,
                                            kernel_initializer='glorot_uniform',
                                            bias_initializer='zeros',
                                            kernel_regularizer=tf.keras.regularizers.l2(cfg.TRAIN.WEIGHT_DECAY))
        self.pad = tf.keras.layers.ZeroPadding2D(padding=self.padding)
        if self.bn:
            self.norm = BatchNormalization()
            
    def call(self, x):
        x = self.pad(x)
        x = self.conv(x)
        if self.bn:
            x = self.norm(x)
        x = self.activation(x)
        return x

class DwConv(tf.keras.layers.Layer):
    # Depth-wise convolution
    def __init__(self, filters, kernel_size=3, strides=1, padding=None, dilation_rate=1, groups=1, activation=None, bn=True, depth_multiplier=1):
        super(DwConv, self).__init__()
        self.filters          = filters
        self.kernel_size      = kernel_size 
        self.strides          = strides
        self.dilatation       = dilation_rate
        self.padding          = autopad(kernel_size, padding, self.dilatation)
        self.groups           = groups
        self.bn               = bn
        self.linear           = lambda x: x
        self.activation       = activation if activation is not None else self.linear
        self.depth_multiplier = depth_multiplier

    def build(self, input_shape):
        self.dwconv = tf.keras.layers.DepthwiseConv2D(
                                                        kernel_size=self.kernel_size,
                                                        strides=self.strides,
                                                        padding='valid',
                                                        depth_multiplier=self.depth_multiplier,
                                                        dilation_rate=self.dilatation,
                                                        activation=None,
                                                        use_bias=not self.bn,
                                                        depthwise_initializer='glorot_uniform',
                                                        bias_initializer='zeros',
                                                        depthwise_regularizer=tf.keras.regularizers.l2(cfg.TRAIN.WEIGHT_DECAY))
        self.pad = tf.keras.layers.ZeroPadding2D(padding=self.padding)
        if self.bn:
            self.norm = BatchNormalization()
    def call(self, x):
        x = self.pad(x)
        x = self.dwconv(x)
        if self.bn:
            x = self.norm(x)
        x = self.activation(x)
        return x
    
class Bottleneck(tf.keras.layers.Layer):
    # Standard bottleneck
    def __init__(self, filters, shortcut=True, groups=1, kernel_size=(3, 3), e=0.5, activation=tf.nn.silu):
        super(Bottleneck, self).__init__()
        self.filters1    = int(filters * e)
        self.filters2    = filters
        self.groups      = groups
        self.shortcut    = shortcut
        self.activation  = activation
        self.kernel_size = kernel_size
        
    def build(self, input_shape):
        self.cnv1 = Conv(filters=self.filters1,
                         kernel_size=self.kernel_size[0], 
                         strides=1,
                         padding=None,
                         dilation_rate=1,
                         groups=self.groups,
                         activation=self.activation,
                         bn=True)
        self.cnv2 = Conv(filters=self.filters2,
                         kernel_size=self.kernel_size[1], 
                         strides=1,
                         padding=None,
                         dilation_rate=1,
                         groups=self.groups,
                         activation=self.activation,
                         bn=True)
        self.add  = input_shape[-1] == self.filters2 and self.shortcut
    def call(self, x):
        return x + self.cnv2(self.cnv1(x)) if self.add else self.cnv2(self.cnv1(x))

    
class C2f(tf.keras.layers.Layer):
    def __init__(self, filters, n=1, shortcut=False, kernel_size=(3, 3), groups=1, e=0.5, activation=tf.nn.silu):
        super(C2f, self).__init__()
        self.filters1    = int(filters * e)
        self.filters2    = filters
        self.shortcut    = shortcut
        self.kernel_size = kernel_size
        self.activation  = activation
        self.groups      = groups
        self.n           = n
        self.e           = e
    
    def build(self, input_shape):
        self.cnv1 = Conv(filters= 2*self.filters1,
                         kernel_size=self.kernel_size[0], 
                         strides=1,
                         padding=None,
                         dilation_rate=1,
                         groups=self.groups,
                         activation=self.activation,
                         bn=True)
        self.cnv2 = Conv(filters=self.filters2,
                         kernel_size=self.kernel_size[1], 
                         strides=1,
                         padding=None,
                         dilation_rate=1,
                         groups=self.groups,
                         activation=self.activation,
                         bn=True)
        
        self.m = [Bottleneck(filters=self.f1,
                             shortcut=self.shortcut,
                             groups=self.groups,
                             kernel_size=self.kernel_size,
                             e=self.e) for _ in range(self.n)]

    def call(self, x):
        y = tf.split(self.cnv1(x), 2, axis=3)
        y.extend(m(y[-1]) for m in self.m)
        return self.cnv2(tf.concat(y, axis=3))

    
class MaxPool(tf.keras.layers.Layer):
    # Spatial Pyramid Pooling - Fast (SPPF) layer for YOLOv5 by Glenn Jocher
    def __init__(self, pool_size=3, strides=1, padding=5):  # equivalent to SPP(k=(5, 9, 13))
        super(MaxPool, self).__init__()
        self.pool_size   = pool_size 
        self.strides     = strides
        self.padding     = padding
    
    def build(self, input_shape):
        self.maxpool = tf.keras.layers.MaxPooling2D(pool_size=self.pool_size,
                                                    strides=self.strides,
                                                    padding='valid')
        self.pad     = tf.keras.layers.ZeroPadding2D(padding=self.padding)
    
    def call(self, x):
        x = self.pad(x)
        x = self.maxpool(x)
        return x


class SPPF(tf.keras.layers.Layer):
    # Spatial Pyramid Pooling - Fast (SPPF) layer for YOLOv5 by Glenn Jocher
    def __init__(self, filters, kernel_size=1, strides=1, dilation_rate=1, groups=1, activation=tf.nn.silu, pool_size=5):  # equivalent to SPP(k=(5, 9, 13))
        super(SPPF, self).__init__()
        self.filters     = filters
        self.kernel_size = kernel_size 
        self.strides     = strides
        self.dilatation  = dilation_rate
        self.groups      = groups
        self.activation  = activation
        self.pool_size   = pool_size
    
    def build(self, input_shape):
        self.filters1 = input_shape[-1] // 2
        self.filters2 = self.filters

        self.cnv1   = Conv(filters= self.filters1,
                            kernel_size=self.kernel_size, 
                            strides=1,
                            padding=None,
                            dilation_rate=1,
                            groups=self.groups,
                            activation=self.activation,
                            bn=True)
        
        self.cnv2   = Conv(filters=self.filters2,
                            kernel_size=self.kernel_size, 
                            strides=1,
                            padding=None,
                            dilation_rate=1,
                            groups=self.groups,
                            activation=self.activation,
                            bn=True)
        self.maxpool = MaxPool(pool_size=self.pool_size,
                               strides=self.strides,
                               padding=self.pool_size//2)
        
    def call(self, x):
        x  = self.cnv1(x)
        y1 = self.maxpool(x)
        y2 = self.maxpool(y1)
        return self.cnv2(tf.concat([x, y1, y2, self.maxpool(y2)], axis=-1))
