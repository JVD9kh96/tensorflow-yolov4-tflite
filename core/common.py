#! /usr/bin/env python
# coding=utf-8

import tensorflow as tf
from tensorflow.keras import layers
# import tensorflow_addons as tfa
from tensorflow.keras import regularizers


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
                  kernel_size = 3,
                  normalization = 'batch'):
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
    key = tf.keras.layers.Conv2D(filters = heads//2,
                                 kernel_size=(1, 1),
                                 strides = (1, 1),
                                 padding = 'same',
                                 use_bias = False,
                                 kernel_regularizer=tf.keras.regularizers.l2(0.0005),
                                 kernel_initializer=tf.random_normal_initializer(stddev=0.01))(key)
    if normalization == 'batch':
        key = tf.keras.layers.BatchNormalization()(key)
    # elif normalization == 'group':
    #     key = tfa.layers.GroupNormalization(min(16, inp.shape[-1]))(key)
    elif normalization == 'layer':
        key = tf.keras.layers.LayerNormalization(epsilon=1e-6)(key)
        
    if activation == 'mish':
        key = mish(key)
    elif activation == 'gelu':
        # key = tfa.activations.gelu(key)
        key = tf.nn.gelu(key)
    elif activation == 'leaky':
        key = tf.keras.layers.LeakyReLU(alpha = 0.3)(key)
        
    key = tf.keras.layers.Conv2D(filters = heads,
                             kernel_size=(out_filters, out_filters),
                             strides = (1, 1),
                             padding = 'same',
                             use_bias = False,
                             kernel_regularizer=tf.keras.regularizers.l2(0.0005),
                             kernel_initializer=tf.random_normal_initializer(stddev=0.01))(key)    
        
    value = tf.keras.layers.Conv2D(filters = heads//2,
                                 kernel_size=(1, 1),
                                 strides = (1, 1),
                                 padding = 'same',
                                 use_bias = False,
                                 kernel_regularizer=tf.keras.regularizers.l2(0.0005),
                                 kernel_initializer=tf.random_normal_initializer(stddev=0.01))(value)
    
    if normalization == 'batch':
        value = tf.keras.layers.BatchNormalization()(value)
    # elif normalization == 'group':
    #     key = tfa.layers.GroupNormalization(min(16, inp.shape[-1]))(key)
    elif normalization == 'layer':
        value = tf.keras.layers.LayerNormalization(epsilon=1e-6)(value)
        
    if activation == 'mish':
        value = mish(value)
    elif activation == 'gelu':
        # key = tfa.activations.gelu(key)
        value = tf.nn.gelu(value)
    elif activation == 'leaky':
        value = tf.keras.layers.LeakyReLU(alpha = 0.3)(value)
        
    value = tf.keras.layers.Conv2D(filters = heads,
                             kernel_size=(out_filters, out_filters),
                             strides = (1, 1),
                             padding = 'same',
                             use_bias = False,
                             kernel_regularizer=tf.keras.regularizers.l2(0.0005),
                             kernel_initializer=tf.random_normal_initializer(stddev=0.01))(value)
        
    query = tf.keras.layers.Conv2D(filters = heads//2,
                                 kernel_size=(1, 1),
                                 strides = (1, 1),
                                 padding = 'same',
                                 use_bias = False,
                                 kernel_regularizer=tf.keras.regularizers.l2(0.0005),
                                 kernel_initializer=tf.random_normal_initializer(stddev=0.01))(query)
    if normalization == 'batch':
        query = tf.keras.layers.BatchNormalization()(query)
    # elif normalization == 'group':
    #     key = tfa.layers.GroupNormalization(min(16, inp.shape[-1]))(key)
    elif normalization == 'layer':
        query = tf.keras.layers.LayerNormalization(epsilon=1e-6)(query)
        
    if activation == 'mish':
        query = mish(query)
    elif activation == 'gelu':
        # query = tfa.activations.gelu(query)
        query = tf.nn.gelu(query)
    elif activation == 'leaky':
        query = tf.keras.layers.LeakyReLU(alpha = 0.3)(query)
        
    query = tf.keras.layers.Conv2D(filters = heads,
                             kernel_size=(out_filters, out_filters),
                             strides = (1, 1),
                             padding = 'same',
                             use_bias = False,
                             kernel_regularizer=tf.keras.regularizers.l2(0.0005),
                             kernel_initializer=tf.random_normal_initializer(stddev=0.01))(query)
    
    shape = getattr(value, 'shape')
    dtype = getattr(value, 'dtype')
    dk = tf.cast(shape[1]*shape[2], dtype=dtype)
    qk = tf.einsum('aijb,ajkb->aikb', query, key)/tf.math.sqrt(dk)
    
    
    if normalization == 'batch':
        qk = tf.keras.layers.BatchNormalization()(qk)
    # elif normalization == 'group':
    #     x1 = tfa.layers.GroupNormalization(min(16, inp.shape[-1]))(inp)
    elif normalization == 'layer':
        qk = tf.keras.layers.LayerNormalization(epsilon=1e-6)(qk)
        
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
    qk        = tf.nn.sigmoid(qk)
    attention = tf.math.multiply(qk , value)
    attention = tf.keras.layers.Conv2D(filters = out_filters//2, kernel_size = (1, 1), strides = (1, 1), padding = 'same',
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
    attention = attention + shortcut
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
        x1 = tf.keras.layers.BatchNormalization()(inp)
    # elif normalization == 'group':
    #     x1 = tfa.layers.GroupNormalization(min(16, inp.shape[-1]))(inp)
    elif normalization == 'layer':
        x1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)(inp)
        
    x2 = kai_attention(x1,
                       x1,
                       x1,
                       heads=out_filt,
                       out_filters=out_filt,
                       axis = attention_axes,
                       activation = activation,
                       normalization =  normalization)
    
    x3 = tf.keras.layers.Add()([x2, inp])
    if normalization == 'batch':
        x4 = tf.keras.layers.BatchNormalization()(x3)
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
              
    x8 = tf.keras.layers.Add()([x7, x3])

    if normalization == 'batch':
        x8 = tf.keras.layers.BatchNormalization()(x8)
    # elif normalization == 'group':
    #     x8 = tfa.layers.GroupNormalization(min(16, x3.shape[-1]))(x8)
    elif normalization == 'layer':
        x8 = tf.keras.layers.LayerNormalization(epsilon=0.001)(x8)

    if down_sample:
        x8 = tf.keras.layers.MaxPooling2D(pool_size = (2, 2), strides = (2, 2))(x8)
    
    return x8
