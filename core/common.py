#! /usr/bin/env python
# coding=utf-8

import tensorflow as tf
from tensorflow.keras import layers
# import tensorflow_addons as tfa
from tensorflow.keras import regularizers
from tensorflow.python.keras import backend as K

class Dropblock(tf.keras.layers.Layer):
  """DropBlock: a regularization method for convolutional neural networks.
    DropBlock is a form of structured dropout, where units in a contiguous
    region of a feature map are dropped together. DropBlock works better than
    dropout on convolutional layers due to the fact that activation units in
    convolutional layers are spatially correlated.
    See https://arxiv.org/pdf/1810.12890.pdf for details.
  """

  def __init__(self,
               dropblock_keep_prob=None,
               dropblock_size=None,
               data_format='channels_last'):
    super(Dropblock, self).__init__()
    self._dropblock_keep_prob = dropblock_keep_prob
    self._dropblock_size = dropblock_size
    self._data_format = data_format

  def call(self, net, training=False):
    """Builds Dropblock layer.
    Args:
      net: `Tensor` input tensor.
      is_training: `bool` if True, the model is in training mode.
    Returns:
      A version of input tensor with DropBlock applied.
    """
    if (not training or self._dropblock_keep_prob is None or
        self._dropblock_keep_prob == 1.0):
      return net

  

    if self._data_format == 'channels_last':
      height = tf.shape(net)[1]
      width = tf.shape(net)[2]
      #_, height, width, _ = net.get_shape().as_list()
    else:
      height = tf.shape(net)[2]
      width = tf.shape(net)[3]
      #_, _, height, width = net.get_shape().as_list()

    total_size = width * height
    dropblock_size = tf.math.minimum(self._dropblock_size, tf.math.minimum(width, height))
    # Seed_drop_rate is the gamma parameter of DropBlcok.
    seed_drop_rate = (
        1.0 - self._dropblock_keep_prob) * tf.cast(total_size, tf.float32) / tf.cast(dropblock_size**2 , tf.float32) / tf.cast(
            (width - self._dropblock_size + 1) *
            (height - self._dropblock_size + 1), tf.float32)

    # Forces the block to be inside the feature map.
    w_i, h_i = tf.meshgrid(tf.range(width), tf.range(height))
    valid_block = tf.logical_and(
        tf.logical_and(w_i >= tf.cast(dropblock_size / 2, tf.int32),
                       w_i < width - (dropblock_size - 1) // 2),
        tf.logical_and(h_i >= tf.cast(dropblock_size / 2, tf.int32),
                       h_i < width - (dropblock_size - 1) // 2))

    if self._data_format == 'channels_last':
      valid_block = tf.reshape(valid_block, [1, height, width, 1])
    else:
      valid_block = tf.reshape(valid_block, [1, 1, height, width])

    randnoise = tf.random.uniform(tf.shape(net), dtype=tf.float32)
    valid_block = tf.cast(valid_block, dtype=tf.float32)
    seed_keep_rate = tf.cast(1 - seed_drop_rate, dtype=tf.float32)
    block_pattern = (1 - valid_block + seed_keep_rate + randnoise) >= 1
    block_pattern = tf.cast(block_pattern, dtype=tf.float32)

    if self._data_format == 'channels_last':
      ksize = [1, self._dropblock_size, self._dropblock_size, 1]
    else:
      ksize = [1, 1, self._dropblock_size, self._dropblock_size]
    block_pattern = -tf.nn.max_pool(
        -block_pattern,
        ksize=ksize,
        strides=[1, 1, 1, 1],
        padding='SAME',
        data_format='NHWC' if self._data_format == 'channels_last' else 'NCHW')

    percent_ones = tf.cast(tf.reduce_sum(block_pattern), tf.float32) / tf.cast(
        tf.size(block_pattern), tf.float32)

    net = net / tf.cast(percent_ones, net.dtype) * tf.cast(
        block_pattern, net.dtype)
    return net

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

def convolutional(input_layer, filters_shape, downsample=False, activate=True, bn=True, activate_type='leaky', norm = 0):
    if downsample:
        input_layer = tf.keras.layers.ZeroPadding2D(((1, 0), (1, 0)))(input_layer)
        padding = 'valid'
        strides = 2
    else:
        strides = 1
        padding = 'same'

    conv = tf.keras.layers.Conv2D(filters=filters_shape[-1], kernel_size = filters_shape[0], strides=strides, padding=padding,
                                  use_bias=not bn, kernel_regularizer=tf.keras.regularizers.l1_l2(l1=0.0005, l2=0.0005),
                                  kernel_initializer=tf.random_normal_initializer(stddev=0.01),
                                  bias_initializer=tf.constant_initializer(0.))(input_layer)

    if bn and norm==0: 
        conv = BatchNormalization()(conv)
        conv_shape = getattr(conv, 'shape')
        block_size = tf.maximum(1, conv_shape[1] // 8)
        conv = Dropblock(dropblock_keep_prob=0.9, dropblock_size=block_size)(conv)
    # elif bn and norm==1:
    #     conv = tfa.layers.GroupNormalization(groups = min(filters_shape[-1], 32))(conv)
     
    if activate == True:
        if activate_type == "leaky":
            conv = tf.nn.leaky_relu(conv, alpha=0.1)
        elif activate_type == "mish":
            conv = mish(conv)
        elif activate_type == 'gelu':
            # conv = tfa.activations.gelu(conv)
            conv = tf.nn.gelu(conv)
    return conv

def mish(x):
    return x * tf.math.tanh(tf.math.softplus(x))
    # return tf.keras.layers.Lambda(lambda x: x*tf.tanh(tf.math.log(1+tf.exp(x))))(x)

class softmax_2d(tf.keras.layers.Layer):
    def __init__(self):
        super(softmax_2d, self).__init__()
    def call(self, images):
        exp = tf.math.exp(images)
        sum_ = tf.reduce_sum(exp, axis = [1, 2])
        return exp/sum_    

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
    return tf.image.resize(input_layer, (input_layer.shape[1] * 2, input_layer.shape[2] * 2), method='bilinear')

def mlp(x, hidden_units, dropout_rate, activation = 'gelu'):
    for units in hidden_units:
        if activation == 'gelu':
            x = layers.Dense(units, activation=tf.nn.gelu)(x)
        elif activation == 'mish':
            x = layers.Dense(units)(x)
            x = mish(x)
        x = layers.Dropout(dropout_rate)(x)
    return x

def transformer(input_layer, projection_dim, transformer_units, num_layers = 4, num_heads = 4, activation = 'gelu', normal = 0):
    encoded_patches = input_layer
    for _ in range(num_layers):
        # Layer normalization 1.
        if normal == 0:
            x1 = layers.BatchNormalization()(encoded_patches)
        # elif normal == 1:
        #     x1 = tfa.layers.GroupNormalization(groups = min(projection_dim, 16))(encoded_patches)
        elif normal == 2:
            x1 = layers.LayerNormalization(epsilon=1e-6)(encoded_patches)
        # Create a multi-head attention layer.
        attention_output = layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=projection_dim, dropout=0.1
        )(x1, x1)
        # Skip connection 1.
        x2 = layers.Add()([attention_output, encoded_patches])
        # Layer normalization 2.
        if normal == 0:
            x3 = layers.BatchNormalization()(x2)
        # elif normal == 1:
        #     x3 = tfa.layers.GroupNormalization(groups = min(projection_dim, 16))(x2)
        elif normal ==2:
            x3 = layers.LayerNormalization(epsilon=1e-6)(x2)
        # MLP.
        x3 = mlp(x3, hidden_units=transformer_units, dropout_rate=0.1, activation = activation)
        # Skip connection 2.
        encoded_patches = layers.Add()([x3, x2])
    return encoded_patches


def kai_attention(key,
                  value,
                  query,
                  heads=32,
                  out_filters=32,
                  axis = 1,
                  activation = 'gelu',
                  kernel_size = 3):
    """
    heads: number of filters in query, key and value 
    out_filters: number of the output in the output channgel
    axis: if 1, the attention will be calculated in height of the image, 
          if 2, the attention will be calculated in the width of the image, 
          if 3, the attention will be calculated in the depth of the image, 
          if [1, 2], the attention will be calculated in height and width, then
             they are summed up
          if [1, 2, 3], the attention will be calculated in the heigth, width
             and the depth of the image, then the results will be summed up.
          if '2d', the attention will be calculated in 2D (hight and width 
             simultaneously) 
    """
    key = tf.keras.layers.Conv2D(filters = heads//2,
                                 kernel_size=(1, 1),
                                 strides = (1, 1),
                                 padding = 'same',
                                 use_bias = False,
                                 kernel_regularizer=tf.keras.regularizers.l1_l2(l1=0.00025, l2=0.00025),
                                 kernel_initializer=tf.random_normal_initializer(stddev=0.01))(key)
    
    value = tf.keras.layers.Conv2D(filters = heads//2,
                                 kernel_size=(1, 1),
                                 strides = (1, 1),
                                 padding = 'same',
                                 use_bias = False,
                                 kernel_regularizer=tf.keras.regularizers.l1_l2(l1=0.00025, l2=0.00025),
                                 kernel_initializer=tf.random_normal_initializer(stddev=0.01))(value)
    
    query = tf.keras.layers.Conv2D(filters = heads//2,
                                 kernel_size=(1, 1),
                                 strides = (1, 1),
                                 padding = 'same',
                                 use_bias = False,
                                 kernel_regularizer=tf.keras.regularizers.l1_l2(l1=0.00025, l2=0.00025),
                                 kernel_initializer=tf.random_normal_initializer(stddev=0.01))(query)
    shape = getattr(value, 'shape')
    dk = tf.cast(shape[1]*shape[2], tf.float32)
    qk = tf.einsum('aijb,ajkb->aikb', query, key)/tf.math.sqrt(dk)

    if axis == 1:
        qk = tf.nn.softmax(qk, axis = 1)
    elif axis ==2:
        qk = tf.nn.softmax(qk, axis = 2)
    elif axis == 3:
        qk = tf.nn.softmax(qk, axis = 3)
    elif axis == [1, 2]:
        qk_1 = tf.nn.softmax(qk, axis = 1)
        qk_2 = tf.nn.softmax(qk, axis = 2)
        qk = tf.keras.layers.Add()([qk_1, qk_2])
    elif axis == [1, 2, 3]:
        qk_1 = tf.nn.softmax(qk, axis = 1)
        qk_2 = tf.nn.softmax(qk, axis = 2)
        qk_3 = tf.nn.softmax(qk, axis = 3)
        qk = tf.keras.layers.Add()([qk_1, qk_2, qk_3])
    elif axis == '2d':
        qk = softmax_2d()(qk)

    attention = tf.einsum('aijb,ajkb->aikb', qk, value)
    attention = tf.keras.layers.Conv2D(filters = out_filters, kernel_size = kernel_size, strides = (1, 1), padding = 'same',
                                        kernel_regularizer=tf.keras.regularizers.l1_l2(l1=0.00025, l2=0.00025),
                                        kernel_initializer=tf.random_normal_initializer(stddev=0.01),
                                        use_bias = False,
                                        activity_regularizer=regularizers.l2(1e-5))(attention)
    if activation == 'mish':
        attention = mish(attention)
    elif activation == 'gelu':
        # attention = tfa.activations.gelu(attention)
        attention = tf.nn.gelu(attention)
    elif activation == 'leaky':
        attention = tf.keras.layers.LeakyReLU(alpha = 0.3)(attention)

    return attention

def transformer_block(inp,
                      out_filt = 128,
                      activation = 'mish',
                      down_sample = False,
                      attention_axes = 1,
                      kernel_size = 3,
                      normalization = 'batch'):
    
    inp = tf.keras.layers.Conv2D(filters = out_filt,
                                 kernel_size = kernel_size,
                                 strides = (1, 1),
                                 kernel_regularizer=tf.keras.regularizers.l1_l2(l1=0.00025, l2=0.00025),
                                 kernel_initializer=tf.random_normal_initializer(stddev=0.01),
                                 use_bias = False,
                                 padding='same')(inp)
    if activation == 'mish':
        inp = mish(inp)
    elif activation == 'gelu':
        # inp = tfa.activations.gelu(inp)
        inp = tf.nn.gelu(inp)
    elif activation == 'leaky':
        inp = tf.keras.layers.LeakyReLU(alpha = 0.3)(inp)

    if normalization == 'batch':
        x1 = tf.keras.layers.BatchNormalization()(inp)
        conv_shape = getattr(x1, 'shape')
        block_size = tf.maximum(1, conv_shape[1] // 8)
        x1 = Dropblock(dropblock_keep_prob=0.9, dropblock_size=block_size)(x1)
    # elif normalization == 'group':
    #     x1 = tfa.layers.GroupNormalization(min(16, inp.shape[-1]))(inp)
    elif normalization == 'layer':
        x1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)(inp)
        conv_shape = getattr(x1, 'shape')
        block_size = tf.maximum(1, conv_shape[1] // 8)
        x1 = Dropblock(dropblock_keep_prob=0.9, dropblock_size=block_size)(x1)
        
    x2 = kai_attention(x1,
                       x1,
                       x1,
                       heads=out_filt,
                       out_filters=out_filt,
                       axis = attention_axes,
                       activation = activation
                       )
    x3 = tf.keras.layers.Add()([x2, inp])
    if normalization == 'batch':
        x4 = tf.keras.layers.BatchNormalization()(x3)
        conv_shape = getattr(x4, 'shape')
        block_size = tf.maximum(1, conv_shape[1] // 8)
        x4 = Dropblock(dropblock_keep_prob=0.9, dropblock_size=block_size)(x4)
    # elif normalization == 'group':
    #     x4 = tfa.layers.GroupNormalization(min(16, x3.shape[-1]))(x3)
    elif normalization == 'layer':
        x4 = tf.keras.layers.LayerNormalization(epsilon=1e-6)(x3)
        conv_shape = getattr(x4, 'shape')
        block_size = tf.maximum(1, conv_shape[1] // 8)
        x4 = Dropblock(dropblock_keep_prob=0.9, dropblock_size=block_size)(x4)
    
    x5 = tf.keras.layers.Conv2D(filters = out_filt//2,
                                kernel_size=(1, 1),
                                strides=(1, 1),
                                padding = 'same',
                                use_bias = False,
                                kernel_regularizer=tf.keras.regularizers.l1_l2(l1=0.00025, l2=0.00025),
                                kernel_initializer=tf.random_normal_initializer(stddev=0.01))(x4)
    if activation == 'mish':
        x6 = mish(x5)
    elif activation == 'gelu':
        # x6 = tfa.activations.gelu(x5)
        x6 = tf.nn.gelu(x5)
    elif activation == 'leaky':
        x6 = tf.keras.layers.LeakyReLU(alpha = 0.3)(x5)
    else:
        x6 = x5
    x7 = tf.keras.layers.Conv2D(filters = out_filt,
                                kernel_size=kernel_size,
                                strides=(1, 1),
                                padding = 'same',
                                use_bias = False,
                                kernel_regularizer=tf.keras.regularizers.l1_l2(l1=0.00025, l2=0.00025),
                                kernel_initializer=tf.random_normal_initializer(stddev=0.01),
                                bias_regularizer=regularizers.l2(1e-4),
                                activity_regularizer=regularizers.l2(1e-5))(x6)
    if activation == 'mish':
        x7 = mish(x7)
    elif activation == 'gelu':
        # x7 = tfa.activations.gelu(x7)
        x7 = tf.nn.gelu(x7)
    elif activation == 'leaky':
        x7 = tf.keras.layers.LeakyReLU(alpha = 0.3)(x7)
    else:
        x7 = x7
              
    x8 = tf.keras.layers.Add()([x7, x3])

    if normalization == 'batch':
        x8 = tf.keras.layers.BatchNormalization()(x8)
        conv_shape = getattr(x8, 'shape')
        block_size = tf.maximum(1, conv_shape[1] // 8)
        x8 = Dropblock(dropblock_keep_prob=0.9, dropblock_size=block_size)(x8) 
    # elif normalization == 'group':
    #     x8 = tfa.layers.GroupNormalization(min(16, x3.shape[-1]))(x8)
    elif normalization == 'layer':
        x8 = tf.keras.layers.LayerNormalization(epsilon=0.001)(x8)
        conv_shape = getattr(x8, 'shape')
        block_size = tf.maximum(1, conv_shape[1] // 8)
        x8 = Dropblock(dropblock_keep_prob=0.9, dropblock_size=block_size)(x8) 

    if down_sample:
        x8 = tf.keras.layers.MaxPooling2D(pool_size = (2, 2), strides = (2, 2))(x8)
    
    return x8
