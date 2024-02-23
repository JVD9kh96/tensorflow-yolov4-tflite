#! /usr/bin/env python
# coding=utf-8

import tensorflow as tf
from tensorflow.keras import layers
# import tensorflow_addons as tfa
from tensorflow.keras import regularizers

from tensorflow.python.keras import backend as K

class shake_shake_branch(tf.keras.layers.Layer):
    def __init__(self):
        super(shake_shake_branch, self).__init__()
        
    def call(self, x, rand_forward, rand_backward, training=False):
        if training:
            x = x * rand_backward + tf.stop_gradient(x * rand_forward -
                                                 x * rand_backward)
        else:
            x = x * (1.0 / 2)
        return x

class shake_shake_add(tf.keras.layers.Layer):
    def __init__(self):
      super(shake_shake_add, self).__init__()
      self.shake1 = shake_shake_branch()
      self.shake2 = shake_shake_branch()
      
        
    def call(self, x, x1, x1p, training=False):
        batch_size = tf.shape(x)[0]
        dtype      = x.dtype
        # Generate random numbers for scaling the branches
        rand_forward = [
          tf.cast(tf.random.uniform(
              [batch_size, 1, 1, 1], minval=0, maxval=1, dtype=tf.float32), dtype=dtype)
          for _ in range(2)
        ]
        rand_backward = [
          tf.cast(tf.random.uniform(
              [batch_size, 1, 1, 1], minval=0, maxval=1, dtype=tf.float32), dtype=dtype)
          for _ in range(2)
        ]
        # Normalize so that all sum to 1
        total_forward  = tf.add_n(rand_forward)
        total_backward = tf.add_n(rand_backward)
        rand_forward   = [samp / total_forward for samp in rand_forward]
        rand_backward  = [samp / total_backward for samp in rand_backward]
        branches        = []
        
        b = [self.shake1(x1, rand_forward[0], rand_backward[0], training=training),
             self.shake2(x1p, rand_forward[1], rand_backward[1], training=training),
             x]
        return tf.add_n(b)

class Dropblock(tf.keras.layers.Layer):
  """DropBlock: a regularization method for convolutional neural networks.
    DropBlock is a form of structured dropout, where units in a contiguous
    region of a feature map are dropped together. DropBlock works better than
    dropout on convolutional layers due to the fact that activation units in
    convolutional layers are spatially correlated.
    See https://arxiv.org/pdf/1810.12890.pdf for details.
  """
  def __init__(self,
               dropblock_keep_prob=0.9,
               dropblock_size=5,
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
    def __init__(self, synchronized=True, **kwargs):
        super(BatchNormalization, self).__init__(synchronized=synchronized, **kwargs)
    def call(self, x, training=False):
        if not training:
            training = tf.constant(False)
        training = tf.logical_and(training, self.trainable)
        return super().call(x, training)

def convolutional(input_layer, filters_shape, downsample=False, activate=True, bn=True, activate_type='leaky', norm = 0, dropblock=False, dropblock_keep_prob=0.9):
    if downsample:
        input_layer = tf.keras.layers.ZeroPadding2D(((1, 0), (1, 0)))(input_layer)
        padding = 'valid'
        strides = 2
    else:
        strides = 1
        padding = 'same'

    conv = tf.keras.layers.Conv2D(filters=filters_shape[-1], kernel_size = filters_shape[0], strides=strides, padding=padding,
                                  use_bias=not bn, kernel_regularizer=tf.keras.regularizers.l2(0.0005),
                                  kernel_initializer=tf.random_normal_initializer(stddev=0.01),
                                  bias_initializer=tf.constant_initializer(0.))(input_layer)

    if bn and norm==0: 
        conv = BatchNormalization()(conv)
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
    if dropblock:
        conv = Dropblock(dropblock_keep_prob=dropblock_keep_prob)(conv)
    return conv

class Mish(tf.keras.layers.Layer):
    def call(self, x, training=False):
        return x * tf.math.tanh(tf.math.softplus(x))
mish = Mish()
# def mish(x):
#     return x * tf.math.tanh(tf.math.softplus(x))
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

def upsample(input_layer, dtype=None):
    if dtype is None:
        dtype = input_layer.dtype
    return tf.cast(tf.image.resize(input_layer, (input_layer.shape[1] * 2, input_layer.shape[2] * 2), method='bilinear'), dtype)

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
            x1 = BatchNormalization()(encoded_patches)
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
            x3 = BatchNormalization()(x2)
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
                  kernel_size = 3,
                  normalization = 'batch',
                  dropblock = False, 
                  dropblock_keep_prob = 0.9):
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
    shortcut = value 
    k1 = tf.keras.layers.Conv2D(filters = heads//4,
                                 kernel_size=(1, 1),
                                 strides = (1, 1),
                                 padding = 'same',
                                 use_bias = False,
                                 kernel_regularizer=tf.keras.regularizers.l2(0.0005),
                                 kernel_initializer=tf.random_normal_initializer(stddev=0.01))(key)
    if normalization == 'batch':
        k1 = BatchNormalization()(k1)
    # elif normalization == 'group':
    #     key = tfa.layers.GroupNormalization(min(16, inp.shape[-1]))(key)
    elif normalization == 'layer':
        k1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)(k1)
        
    if activation == 'mish':
        k1 = mish(k1)
    elif activation == 'gelu':
        # key = tfa.activations.gelu(key)
        k1 = tf.nn.gelu(k1)
    elif activation == 'leaky':
        k1 = tf.keras.layers.LeakyReLU(alpha = 0.3)(k1)
    
    k2 = tf.keras.layers.Conv2D(filters = heads//4,
                                 kernel_size=(1, 1),
                                 strides = (1, 1),
                                 padding = 'same',
                                 use_bias = False,
                                 kernel_regularizer=tf.keras.regularizers.l2(0.0005),
                                 kernel_initializer=tf.random_normal_initializer(stddev=0.01))(key)
    if normalization == 'batch':
        k2 = BatchNormalization()(k2)
    # elif normalization == 'group':
    #     key = tfa.layers.GroupNormalization(min(16, inp.shape[-1]))(key)
    elif normalization == 'layer':
        k2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)(k2)
        
    if activation == 'mish':
        k2 = mish(k2)
    elif activation == 'gelu':
        # key = tfa.activations.gelu(key)
        k2 = tf.nn.gelu(k2)
    elif activation == 'leaky':
        k2 = tf.keras.layers.LeakyReLU(alpha = 0.3)(k2)
    
    if dropblock:
        k2 = DropBlock(dropblock_keep_prob=dropblock_keep_prob)(k2)
    
    k3 = tf.keras.layers.Conv2D(filters = heads//4,
                                 kernel_size=(1, 1),
                                 strides = (1, 1),
                                 padding = 'same',
                                 use_bias = False,
                                 kernel_regularizer=tf.keras.regularizers.l2(0.0005),
                                 kernel_initializer=tf.random_normal_initializer(stddev=0.01))(key)
    if normalization == 'batch':
        k3 = BatchNormalization()(k3)
    # elif normalization == 'group':
    #     key = tfa.layers.GroupNormalization(min(16, inp.shape[-1]))(key)
    elif normalization == 'layer':
        k3 = tf.keras.layers.LayerNormalization(epsilon=1e-6)(k3)
        
    if activation == 'mish':
        k3 = mish(k3)
    elif activation == 'gelu':
        # key = tfa.activations.gelu(key)
        k3 = tf.nn.gelu(k3)
    elif activation == 'leaky':
        k3 = tf.keras.layers.LeakyReLU(alpha = 0.3)(k3)
    
#     key = shake_shake_add()(k1, k2, k3)
    k_split = tf.keras.layers.Lambda(lambda x: tf.split(x, 4, axis=-1))(k1)
    k11, k12, k13, k14 = k_split[0], k_split[1], k_split[2], k_split[3]
    k1 = tf.keras.layers.Concatenate(axis=-1)((tf.keras.layers.Conv2D(filters = heads//16,
                             kernel_size=(kernel_size, kernel_size),
                             strides = (1, 1),
                             padding = 'same',
                             use_bias = False,
                             kernel_regularizer=tf.keras.regularizers.l2(0.0005),
                             kernel_initializer=tf.random_normal_initializer(stddev=0.01))(k11),
                    tf.keras.layers.Conv2D(filters = heads//16,
                             kernel_size=(kernel_size, kernel_size),
                             strides = (1, 1),
                             padding = 'same',
                             use_bias = False,
                             kernel_regularizer=tf.keras.regularizers.l2(0.0005),
                             kernel_initializer=tf.random_normal_initializer(stddev=0.01))(k12),
                    tf.keras.layers.Conv2D(filters = heads//16,
                             kernel_size=(kernel_size, kernel_size),
                             strides = (1, 1),
                             padding = 'same',
                             use_bias = False,
                             kernel_regularizer=tf.keras.regularizers.l2(0.0005),
                             kernel_initializer=tf.random_normal_initializer(stddev=0.01))(k13),
                    tf.keras.layers.Conv2D(filters = heads//16,
                             kernel_size=(kernel_size, kernel_size),
                             strides = (1, 1),
                             padding = 'same',
                             use_bias = False,
                             kernel_regularizer=tf.keras.regularizers.l2(0.0005),
                             kernel_initializer=tf.random_normal_initializer(stddev=0.01))(k14)))
                    
#     k1 = tf.keras.layers.Conv2D(filters = heads//2,
#                              kernel_size=(kernel_size, kernel_size),
#                              strides = (1, 1),
#                              padding = 'same',
#                              use_bias = False,
#                              kernel_regularizer=tf.keras.regularizers.l2(0.0005),
#                              kernel_initializer=tf.random_normal_initializer(stddev=0.01))(k1)    
        
    v1 = tf.keras.layers.Conv2D(filters = heads//4,
                                 kernel_size=(1, 1),
                                 strides = (1, 1),
                                 padding = 'same',
                                 use_bias = False,
                                 kernel_regularizer=tf.keras.regularizers.l2(0.0005),
                                 kernel_initializer=tf.random_normal_initializer(stddev=0.01))(value)
    
    if normalization == 'batch':
        v1 = BatchNormalization()(v1)
    # elif normalization == 'group':
    #     key = tfa.layers.GroupNormalization(min(16, inp.shape[-1]))(key)
    elif normalization == 'layer':
        v1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)(v1)
        
    if activation == 'mish':
        v1 = mish(v1)
    elif activation == 'gelu':
        # key = tfa.activations.gelu(key)
        v1 = tf.nn.gelu(v1)
    elif activation == 'leaky':
        v1 = tf.keras.layers.LeakyReLU(alpha = 0.3)(v1)
    
    if dropblock:
        v1 = DropBlock(dropblock_keep_prob=dropblock_keep_prob)(v1)
    
    v2 = tf.keras.layers.Conv2D(filters = heads//4,
                                 kernel_size=(1, 1),
                                 strides = (1, 1),
                                 padding = 'same',
                                 use_bias = False,
                                 kernel_regularizer=tf.keras.regularizers.l2(0.0005),
                                 kernel_initializer=tf.random_normal_initializer(stddev=0.01))(value)
    
    if normalization == 'batch':
        v2 = BatchNormalization()(v2)
    # elif normalization == 'group':
    #     key = tfa.layers.GroupNormalization(min(16, inp.shape[-1]))(key)
    elif normalization == 'layer':
        v2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)(v2)
        
    if activation == 'mish':
        v2 = mish(v2)
    elif activation == 'gelu':
        # key = tfa.activations.gelu(key)
        v2 = tf.nn.gelu(v2)
    elif activation == 'leaky':
        v2 = tf.keras.layers.LeakyReLU(alpha = 0.3)(v2)
    
    if dropblock:
        v2 = DropBlock(dropblock_keep_prob=dropblock_keep_prob)(v2)
     
    v3 = tf.keras.layers.Conv2D(filters = heads//4,
                                 kernel_size=(1, 1),
                                 strides = (1, 1),
                                 padding = 'same',
                                 use_bias = False,
                                 kernel_regularizer=tf.keras.regularizers.l2(0.0005),
                                 kernel_initializer=tf.random_normal_initializer(stddev=0.01))(value)
    
    if normalization == 'batch':
        v3 = BatchNormalization()(v3)
    # elif normalization == 'group':
    #     key = tfa.layers.GroupNormalization(min(16, inp.shape[-1]))(key)
    elif normalization == 'layer':
        v3 = tf.keras.layers.LayerNormalization(epsilon=1e-6)(v3)
        
    if activation == 'mish':
        v3 = mish(v3)
    elif activation == 'gelu':
        # key = tfa.activations.gelu(key)
        v3 = tf.nn.gelu(v3)
    elif activation == 'leaky':
        v3 = tf.keras.layers.LeakyReLU(alpha = 0.3)(v3)
    
    if dropblock:
        v3 = DropBlock(dropblock_keep_prob=dropblock_keep_prob)(v3)
    
#     value = shake_shake_add()(v1, v2, v3)
    
#     v1 = tf.keras.layers.Conv2D(filters = heads//2,
#                              kernel_size=(kernel_size, kernel_size),
#                              strides = (1, 1),
#                              padding = 'same',
#                              use_bias = False,
#                              kernel_regularizer=tf.keras.regularizers.l2(0.0005),
#                              kernel_initializer=tf.random_normal_initializer(stddev=0.01))(v1)
    
    v_split = tf.keras.layers.Lambda(lambda x: tf.split(x, 4, axis=-1))(v1)
    v11, v12, v13, v14 = v_split[0], v_split[1], v_split[2], v_split[3]
    v1 = tf.keras.layers.Concatenate(axis=-1)((tf.keras.layers.Conv2D(filters = heads//16,
                             kernel_size=(kernel_size, kernel_size),
                             strides = (1, 1),
                             padding = 'same',
                             use_bias = False,
                             kernel_regularizer=tf.keras.regularizers.l2(0.0005),
                             kernel_initializer=tf.random_normal_initializer(stddev=0.01))(v11),
                    tf.keras.layers.Conv2D(filters = heads//16,
                             kernel_size=(kernel_size, kernel_size),
                             strides = (1, 1),
                             padding = 'same',
                             use_bias = False,
                             kernel_regularizer=tf.keras.regularizers.l2(0.0005),
                             kernel_initializer=tf.random_normal_initializer(stddev=0.01))(v12),
                    tf.keras.layers.Conv2D(filters = heads//16,
                             kernel_size=(kernel_size, kernel_size),
                             strides = (1, 1),
                             padding = 'same',
                             use_bias = False,
                             kernel_regularizer=tf.keras.regularizers.l2(0.0005),
                             kernel_initializer=tf.random_normal_initializer(stddev=0.01))(v13),
                    tf.keras.layers.Conv2D(filters = heads//16,
                             kernel_size=(kernel_size, kernel_size),
                             strides = (1, 1),
                             padding = 'same',
                             use_bias = False,
                             kernel_regularizer=tf.keras.regularizers.l2(0.0005),
                             kernel_initializer=tf.random_normal_initializer(stddev=0.01))(v14)))
    if normalization == 'batch':
        v1 = BatchNormalization()(v1)
    # elif normalization == 'group':
    #     key = tfa.layers.GroupNormalization(min(16, inp.shape[-1]))(key)
    elif normalization == 'layer':
        v1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)(v1)
        
    if activation == 'mish':
        v1 = mish(v1)
    elif activation == 'gelu':
        # key = tfa.activations.gelu(key)
        v1 = tf.nn.gelu(v1)
    elif activation == 'leaky':
        v1 = tf.keras.layers.LeakyReLU(alpha = 0.3)(v1)
    if dropblock:
        v1 = DropBlock()(value)
    
    q1 = tf.keras.layers.Conv2D(filters = heads//4,
                                 kernel_size=(1, 1),
                                 strides = (1, 1),
                                 padding = 'same',
                                 use_bias = False,
                                 kernel_regularizer=tf.keras.regularizers.l2(0.0005),
                                 kernel_initializer=tf.random_normal_initializer(stddev=0.01))(query)
    if normalization == 'batch':
        q1 = BatchNormalization()(q1)
    # elif normalization == 'group':
    #     key = tfa.layers.GroupNormalization(min(16, inp.shape[-1]))(key)
    elif normalization == 'layer':
        q1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)(q1)
        
    if activation == 'mish':
        q1 = mish(q1)
    elif activation == 'gelu':
        # query = tfa.activations.gelu(query)
        q1 = tf.nn.gelu(q1)
    elif activation == 'leaky':
        q1 = tf.keras.layers.LeakyReLU(alpha = 0.3)(q1)
    if dropblock:
        q1 = DropBlock(dropblock_keep_prob=dropblock_keep_prob)(q1)
    
    q2 = tf.keras.layers.Conv2D(filters = heads//4,
                                 kernel_size=(1, 1),
                                 strides = (1, 1),
                                 padding = 'same',
                                 use_bias = False,
                                 kernel_regularizer=tf.keras.regularizers.l2(0.0005),
                                 kernel_initializer=tf.random_normal_initializer(stddev=0.01))(query)
    if normalization == 'batch':
        q2 = BatchNormalization()(q2)
    # elif normalization == 'group':
    #     key = tfa.layers.GroupNormalization(min(16, inp.shape[-1]))(key)
    elif normalization == 'layer':
        q2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)(q2)
        
    if activation == 'mish':
        q2 = mish(q2)
    elif activation == 'gelu':
        # query = tfa.activations.gelu(query)
        q2 = tf.nn.gelu(q2)
    elif activation == 'leaky':
        q2 = tf.keras.layers.LeakyReLU(alpha = 0.3)(q2)
    if dropblock:
        q2 = DropBlock(dropblock_keep_prob=dropblock_keep_prob)(q2)
    
    q3 = tf.keras.layers.Conv2D(filters = heads//4,
                                 kernel_size=(1, 1),
                                 strides = (1, 1),
                                 padding = 'same',
                                 use_bias = False,
                                 kernel_regularizer=tf.keras.regularizers.l2(0.0005),
                                 kernel_initializer=tf.random_normal_initializer(stddev=0.01))(query)
    if normalization == 'batch':
        q3 = BatchNormalization()(q3)
    # elif normalization == 'group':
    #     key = tfa.layers.GroupNormalization(min(16, inp.shape[-1]))(key)
    elif normalization == 'layer':
        q3 = tf.keras.layers.LayerNormalization(epsilon=1e-6)(q3)
        
    if activation == 'mish':
        q3 = mish(q3)
    elif activation == 'gelu':
        # query = tfa.activations.gelu(query)
        q3 = tf.nn.gelu(q3)
    elif activation == 'leaky':
        q3 = tf.keras.layers.LeakyReLU(alpha = 0.3)(q3)
    if dropblock:
        q3 = DropBlock(dropblock_keep_prob=dropblock_keep_prob)(q3)
    
#     query = shake_shake_add()(q1, q2, q3)
    
#     q1 = tf.keras.layers.Conv2D(filters = heads//2,
#                              kernel_size=(kernel_size, kernel_size),
#                              strides = (1, 1),
#                              padding = 'same',
#                              use_bias = False,
#                              kernel_regularizer=tf.keras.regularizers.l2(0.0005),
#                              kernel_initializer=tf.random_normal_initializer(stddev=0.01))(q1)
    q_split = tfkeras.layers.Lambda(lambda x: tf.split(x, 4, axis=-1))(q1)
    q11, q12, q13, q14 = q_split[0], q_split[1], q_split[2], q_split[3]
    q1 = tf.keras.layers.Concatenate(axis=-1)((tf.keras.layers.Conv2D(filters = heads//16,
                             kernel_size=(kernel_size, kernel_size),
                             strides = (1, 1),
                             padding = 'same',
                             use_bias = False,
                             kernel_regularizer=tf.keras.regularizers.l2(0.0005),
                             kernel_initializer=tf.random_normal_initializer(stddev=0.01))(q11),
                    tf.keras.layers.Conv2D(filters = heads//16,
                             kernel_size=(kernel_size, kernel_size),
                             strides = (1, 1),
                             padding = 'same',
                             use_bias = False,
                             kernel_regularizer=tf.keras.regularizers.l2(0.0005),
                             kernel_initializer=tf.random_normal_initializer(stddev=0.01))(q12),
                    tf.keras.layers.Conv2D(filters = heads//16,
                             kernel_size=(kernel_size, kernel_size),
                             strides = (1, 1),
                             padding = 'same',
                             use_bias = False,
                             kernel_regularizer=tf.keras.regularizers.l2(0.0005),
                             kernel_initializer=tf.random_normal_initializer(stddev=0.01))(q13),
                    tf.keras.layers.Conv2D(filters = heads//16,
                             kernel_size=(kernel_size, kernel_size),
                             strides = (1, 1),
                             padding = 'same',
                             use_bias = False,
                             kernel_regularizer=tf.keras.regularizers.l2(0.0005),
                             kernel_initializer=tf.random_normal_initializer(stddev=0.01))(q14)))
    
    shape = getattr(value, 'shape')
    dtype = getattr(value, 'dtype')
    dk    = tf.cast(shape[1]*shape[2], dtype=dtype)
#     qk    = tf.einsum('aijb,ajkb->aikb', query, key)/tf.math.sqrt(dk)
#     qk1    = tf.multiply(q1, k1)
#     qk2    = tf.multiply(q2, k2)
#     qk3    = tf.multiply(q3, k3)
    q = shake_shake_add()(q1, q2, q3)
    k = shake_shake_add()(k1, k2, k3)
    v = shake_shake_add()(v1, v2, v3)
    qk    = tf.keras.layers.Lambda(lambda x: multiply(x[0], x[1]))((q, k))
    
    if normalization == 'batch':
        qk = BatchNormalization()(qk)
    # elif normalization == 'group':
    #     x1 = tfa.layers.GroupNormalization(min(16, inp.shape[-1]))(inp)
    elif normalization == 'layer':
        qk = tf.keras.layers.LayerNormalization(epsilon=1e-6)(qk)
    qk = tf.keras.layers.Lambda(lambda x:tf.nn.sigmoid(x))(qk)
#     if normalization == 'batch':
#         qk1 = BatchNormalization()(qk1)
#     # elif normalization == 'group':
#     #     x1 = tfa.layers.GroupNormalization(min(16, inp.shape[-1]))(inp)
#     elif normalization == 'layer':
#         qk1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)(qk1)
    
#     if normalization == 'batch':
#         qk2 = BatchNormalization()(qk2)
#     # elif normalization == 'group':
#     #     x1 = tfa.layers.GroupNormalization(min(16, inp.shape[-1]))(inp)
#     elif normalization == 'layer':
#         qk2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)(qk2)
    
#     if normalization == 'batch':
#         qk3 = BatchNormalization()(qk3)
#     # elif normalization == 'group':
#     #     x1 = tfa.layers.GroupNormalization(min(16, inp.shape[-1]))(inp)
#     elif normalization == 'layer':
#         qk3 = tf.keras.layers.LayerNormalization(epsilon=1e-6)(qk3)  
#     if axis == 1:
#         qk = tf.nn.softmax(qk, axis = 1)
#     elif axis ==2:
#         qk = tf.nn.softmax(qk, axis = 2)
#     elif axis == 3:
#         qk = tf.nn.softmax(qk, axis = 3)
#     elif axis == [1, 2]:
#         qk_1 = tf.nn.softmax(qk, axis = 1)
#         qk_2 = tf.nn.softmax(qk, axis = 2)
#         qk = tf.keras.layers.Add()([qk_1, qk_2])
#     elif axis == [1, 2, 3]:
#         qk_1 = tf.nn.softmax(qk, axis = 1)
#         qk_2 = tf.nn.softmax(qk, axis = 2)
#         qk_3 = tf.nn.softmax(qk, axis = 3)
#         qk = tf.keras.layers.Add()([qk_1, qk_2, qk_3])
#     elif axis == '2d':
#         qk = softmax_2d()(qk)
#     qk1        = tf.nn.sigmoid(qk1)
#     qk2        = tf.nn.sigmoid(qk2)
#     qk3        = tf.nn.sigmoid(qk3)
    
#     a1         = tf.math.multiply(qk1 , v1)
#     a2         = tf.math.multiply(qk2 , v2)
#     a3         = tf.math.multiply(qk3 , v3)
#     attention  = shake_shake_add()(a1, a2, a3)
    attention  = tf.keras.layers.Lambda(lambda x: x[0]*x[1])((qk , v))
    attention  = tf.keras.layers.Conv2D(filters = out_filters//2, kernel_size = (1, 1), strides = (1, 1), padding = 'same',
                                        kernel_regularizer=tf.keras.regularizers.l2(0.0005),
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
    
    if dropblock:
        attention = DropBlock(dropblock_keep_prob=dropblock_keep_prob)(attention)
    
#     attention = tf.math.multiply(qk , value)
#     a1 = tf.keras.layers.Conv2D(filters = out_filters//2, kernel_size = (1, 1), strides = (1, 1), padding = 'same',
#                                         kernel_regularizer=tf.keras.regularizers.l2(0.0005),
#                                         kernel_initializer=tf.random_normal_initializer(stddev=0.01),
#                                         use_bias = False,
#                                         activity_regularizer=regularizers.l2(1e-5))(a1)
#     if activation == 'mish':
#         a1 = mish(a1)
#     elif activation == 'gelu':
#         # attention = tfa.activations.gelu(attention)
#         a1 = tf.nn.gelu(a1)
#     elif activation == 'leaky':
#         a1 = tf.keras.layers.LeakyReLU(alpha = 0.3)(a1)
    
#     if dropblock:
#         a1 = DropBlock(dropblock_keep_prob=dropblock_keep_prob)(a1)
        
#     a2 = tf.keras.layers.Conv2D(filters = out_filters//2, kernel_size = (1, 1), strides = (1, 1), padding = 'same',
#                                         kernel_regularizer=tf.keras.regularizers.l2(0.0005),
#                                         kernel_initializer=tf.random_normal_initializer(stddev=0.01),
#                                         use_bias = False,
#                                         activity_regularizer=regularizers.l2(1e-5))(a2)
#     if activation == 'mish':
#         a2 = mish(a2)
#     elif activation == 'gelu':
#         # attention = tfa.activations.gelu(attention)
#         a2 = tf.nn.gelu(a2)
#     elif activation == 'leaky':
#         a2 = tf.keras.layers.LeakyReLU(alpha = 0.3)(a2)
    
#     if dropblock:
#         a2 = DropBlock(dropblock_keep_prob=dropblock_keep_prob)(a2)
    
#     a3 = tf.keras.layers.Conv2D(filters = out_filters//2, kernel_size = (1, 1), strides = (1, 1), padding = 'same',
#                                         kernel_regularizer=tf.keras.regularizers.l2(0.0005),
#                                         kernel_initializer=tf.random_normal_initializer(stddev=0.01),
#                                         use_bias = False,
#                                         activity_regularizer=regularizers.l2(1e-5))(a3)
#     if activation == 'mish':
#         a3 = mish(a3)
#     elif activation == 'gelu':
#         # attention = tfa.activations.gelu(attention)
#         a3 = tf.nn.gelu(a3)
#     elif activation == 'leaky':
#         a3 = tf.keras.layers.LeakyReLU(alpha = 0.3)(a3)
    
#     if dropblock:
#         a3 = DropBlock(dropblock_keep_prob=dropblock_keep_prob)(a3)
    
#     attention = shake_shake_add()(a1, a2, a3)
    
    attention = tf.keras.layers.Conv2D(filters = out_filters, kernel_size = kernel_size, strides = (1, 1), padding = 'same',
                                    kernel_regularizer=tf.keras.regularizers.l2(0.0005),
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
    attention = tf.keras.layers.Add()([attention, shortcut])
    if dropblock:
        attention = Dropblock(dropblock_keep_prob=dropblock_keep_prob)(attention)
    return attention

def transformer_block(inp,
                      out_filt = 128,
                      activation = 'mish',
                      down_sample = False,
                      attention_axes = 1,
                      kernel_size = 3,
                      normalization = 'batch',
                      dropblock = False,
                      dropblock_keep_prob = 0.9):
    
#     i1 = tf.keras.layers.Conv2D(filters = out_filt//2,
#                                  kernel_size = (1, 1),
#                                  strides = (1, 1),
#                                  kernel_regularizer=tf.keras.regularizers.l2(0.0005),
#                                  kernel_initializer=tf.random_normal_initializer(stddev=0.01),
#                                  use_bias = False,
#                                  padding='same')(inp)
#     if activation == 'mish':
#         i1 = mish(i1)
#     elif activation == 'gelu':
#         # inp = tfa.activations.gelu(inp)
#         i1 = tf.nn.gelu(i1)
#     elif activation == 'leaky':
#         i1 = tf.keras.layers.LeakyReLU(alpha = 0.3)(i1)

#     if normalization == 'batch':
#         i1 = BatchNormalization()(i1)
#     # elif normalization == 'group':
#     #     x1 = tfa.layers.GroupNormalization(min(16, inp.shape[-1]))(inp)
#     elif normalization == 'layer':
#         i1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)(i1)
    
    inp = tf.keras.layers.Conv2D(filters = out_filt,
                                 kernel_size = kernel_size,
                                 strides = (1, 1),
                                 kernel_regularizer=tf.keras.regularizers.l2(0.0005),
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
        x1 = BatchNormalization()(inp)
    # elif normalization == 'group':
    #     x1 = tfa.layers.GroupNormalization(min(16, inp.shape[-1]))(inp)
    elif normalization == 'layer':
        x1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)(inp)
    if dropblock:
        x1 = DropBlock(dropblock_keep_prob=dropblock_keep_prob)(x1)
    
    x2 = kai_attention(x1,
                       x1,
                       x1,
                       heads=out_filt,
                       out_filters=out_filt,
                       axis = attention_axes,
                       activation = activation,
                       normalization =  normalization,
                       dropblock = dropblock)
    
#     if normalization == 'batch':
#         x1p = BatchNormalization()(inp)
#     # elif normalization == 'group':
#     #     x1 = tfa.layers.GroupNormalization(min(16, inp.shape[-1]))(inp)
#     elif normalization == 'layer':
#         x1p = tf.keras.layers.LayerNormalization(epsilon=1e-6)(inp)
#     if dropblock:
#         x1p = DropBlock(dropblock_keep_prob=dropblock_keep_prob)(x1p)
    
#     x2p = kai_attention(x1p,
#                        x1p,
#                        x1p,
#                        heads=out_filt,
#                        out_filters=out_filt,
#                        axis = attention_axes,
#                        activation = activation,
#                        normalization =  normalization,
#                        dropblock = dropblock)
    
#     x3 = shake_shake_add()(inp, x2, x2p)
    x3 = tf.keras.layers.Add()([inp, x2])
    if normalization == 'batch':
        x4 = BatchNormalization()(x3)
    # elif normalization == 'group':
    #     x4 = tfa.layers.GroupNormalization(min(16, x3.shape[-1]))(x3)
    elif normalization == 'layer':
        x4 = tf.keras.layers.LayerNormalization(epsilon=1e-6)(x3)
    
    x5 = tf.keras.layers.Conv2D(filters = out_filt//2,
                                kernel_size=(1, 1),
                                strides=(1, 1),
                                padding = 'same',
                                use_bias = False,
                                kernel_regularizer=tf.keras.regularizers.l2(0.0005),
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
    if dropblock:
        x6 = DropBlock(dropblock_keep_prob=dropblock_keep_prob)(x6)
        
    x7 = tf.keras.layers.Conv2D(filters = out_filt,
                                kernel_size=kernel_size,
                                strides=(1, 1),
                                padding = 'same',
                                use_bias = False,
                                kernel_regularizer=tf.keras.regularizers.l2(0.0005),
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
    if dropblock:
        x7 = DropBlock(dropblock_keep_prob=dropblock_keep_prob)(x7)
    
#     if normalization == 'batch':
#         x4p = BatchNormalization()(x3)
#     # elif normalization == 'group':
#     #     x4 = tfa.layers.GroupNormalization(min(16, x3.shape[-1]))(x3)
#     elif normalization == 'layer':
#         x4p = tf.keras.layers.LayerNormalization(epsilon=1e-6)(x3)
    
#     x5p = tf.keras.layers.Conv2D(filters = out_filt//2,
#                                 kernel_size=(1, 1),
#                                 strides=(1, 1),
#                                 padding = 'same',
#                                 use_bias = False,
#                                 kernel_regularizer=tf.keras.regularizers.l2(0.0005),
#                                 kernel_initializer=tf.random_normal_initializer(stddev=0.01))(x4p)
#     if activation == 'mish':
#         x6p = mish(x5p)
#     elif activation == 'gelu':
#         # x6 = tfa.activations.gelu(x5)
#         x6p = tf.nn.gelu(x5p)
#     elif activation == 'leaky':
#         x6p = tf.keras.layers.LeakyReLU(alpha = 0.3)(x5p)
#     else:
#         x6p = x5p
#     if dropblock:
#         x6p = DropBlock(dropblock_keep_prob=dropblock_keep_prob)(x6p)
        
#     x7p = tf.keras.layers.Conv2D(filters = out_filt,
#                                 kernel_size=kernel_size,
#                                 strides=(1, 1),
#                                 padding = 'same',
#                                 use_bias = False,
#                                 kernel_regularizer=tf.keras.regularizers.l2(0.0005),
#                                 kernel_initializer=tf.random_normal_initializer(stddev=0.01),
#                                 bias_regularizer=regularizers.l2(1e-4),
#                                 activity_regularizer=regularizers.l2(1e-5))(x6p)
#     if activation == 'mish':
#         x7p = mish(x7p)
#     elif activation == 'gelu':
#         # x7 = tfa.activations.gelu(x7)
#         x7p = tf.nn.gelu(x7p)
#     elif activation == 'leaky':
#         x7p = tf.keras.layers.LeakyReLU(alpha = 0.3)(x7p)
#     else:
#         x7p = x7p
#     if dropblock:
#         x7p = DropBlock(dropblock_keep_prob=dropblock_keep_prob)(x7p)
    
#     x8 = shake_shake_add()(x3, x7, x7p)
    x8 = tf.keras.layers.Add()([x3, x7])

    if normalization == 'batch':
        x8 = BatchNormalization()(x8)
    # elif normalization == 'group':
    #     x8 = tfa.layers.GroupNormalization(min(16, x3.shape[-1]))(x8)
    elif normalization == 'layer':
        x8 = tf.keras.layers.LayerNormalization(epsilon=0.001)(x8)

    if down_sample:
        x8 = tf.keras.layers.MaxPooling2D(pool_size = (2, 2), strides = (2, 2))(x8)
    if dropblock:
        x8 = Dropblock(dropblock_keep_prob=dropblock_keep_prob)(x8)
        
    return x8
