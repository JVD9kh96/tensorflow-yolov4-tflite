#! /usr/bin/env python
# coding=utf-8

import tensorflow as tf
import core.common as common
from tensorflow.keras import layers
import numpy as np
from tensorflow import keras
import tensorflow_addons as tfa


def mlp(x, hidden_units, dropout_rate, activation = 'gelu'):
    for units in hidden_units:
        if activation == 'gelu':
            x = layers.Dense(units, activation=tf.nn.gelu)(x)
        elif activation == 'mish':
            x = layers.Dense(units)(x)
            x = mish(x)
        x = layers.Dropout(dropout_rate)(x)
    return x

class Patches(layers.Layer):
    def __init__(self, patch_size):
        super(Patches, self).__init__()
        self.patch_size = patch_size

    def call(self, images):
        batch_size = tf.shape(images)[0]
        patches = tf.image.extract_patches(
            images=images,
            sizes=[1, self.patch_size, self.patch_size, 1],
            strides=[1, self.patch_size, self.patch_size, 1],
            rates=[1, 1, 1, 1],
            padding="VALID",
        )
        patch_dims = patches.shape[-1]
        patches = tf.reshape(patches, [batch_size, -1, patch_dims])
        return patches

class Patches_sp(layers.Layer):
    def __init__(self, patch_size):
        super(Patches_sp, self).__init__()
        self.patch_size = patch_size

    def call(self, images):
        batch_size = tf.shape(images)[0]
        patches = tf.image.extract_patches(
            images=images,
            sizes=[1, self.patch_size[0], self.patch_size[1], 1],
            strides=[1, self.patch_size[0], self.patch_size[1], 1],
            rates=[1, 1, 1, 1],
            padding="VALID",
        )
        patch_dims = patches.shape[-1]
        patches = tf.reshape(patches, [batch_size, -1, patch_dims])
        return patches



class PatchEncoder(layers.Layer):
    def __init__(self, num_patches, projection_dim):
        super(PatchEncoder, self).__init__()
        self.num_patches = num_patches
        self.projection = layers.Dense(units=projection_dim)
        self.position_embedding = layers.Embedding(
            input_dim=num_patches, output_dim=projection_dim
        )

    def call(self, patch):
        positions = tf.range(start=0, limit=self.num_patches, delta=1)
        encoded = self.projection(patch) + self.position_embedding(positions)
        return encoded


def darknet53(input_data):

    input_data = common.convolutional(input_data, (3, 3,  3,  32))
    input_data = common.convolutional(input_data, (3, 3, 32,  64), downsample=True)

    for i in range(1):
        input_data = common.residual_block(input_data,  64,  32, 64)

    input_data = common.convolutional(input_data, (3, 3,  64, 128), downsample=True)

    for i in range(2):
        input_data = common.residual_block(input_data, 128,  64, 128)

    input_data = common.convolutional(input_data, (3, 3, 128, 256), downsample=True)

    for i in range(8):
        input_data = common.residual_block(input_data, 256, 128, 256)

    route_1 = input_data
    input_data = common.convolutional(input_data, (3, 3, 256, 512), downsample=True)

    for i in range(8):
        input_data = common.residual_block(input_data, 512, 256, 512)

    route_2 = input_data
    input_data = common.convolutional(input_data, (3, 3, 512, 1024), downsample=True)

    for i in range(4):
        input_data = common.residual_block(input_data, 1024, 512, 1024)

    return route_1, route_2, input_data

def cspdarknet53(input_data):

    input_data = common.convolutional(input_data, (3, 3,  3,  32), activate_type="mish")
    input_data = common.convolutional(input_data, (3, 3, 32,  64), downsample=True, activate_type="mish")

    route = input_data
    route = common.convolutional(route, (1, 1, 64, 64), activate_type="mish")
    input_data = common.convolutional(input_data, (1, 1, 64, 64), activate_type="mish")
    for i in range(1):
        input_data = common.residual_block(input_data,  64,  32, 64, activate_type="mish")
    input_data = common.convolutional(input_data, (1, 1, 64, 64), activate_type="mish")

    input_data = tf.concat([input_data, route], axis=-1)
    input_data = common.convolutional(input_data, (1, 1, 128, 64), activate_type="mish")
    input_data = common.convolutional(input_data, (3, 3, 64, 128), downsample=True, activate_type="mish")
    route = input_data
    route = common.convolutional(route, (1, 1, 128, 64), activate_type="mish")
    input_data = common.convolutional(input_data, (1, 1, 128, 64), activate_type="mish")
    for i in range(2):
        input_data = common.residual_block(input_data, 64,  64, 64, activate_type="mish")
    input_data = common.convolutional(input_data, (1, 1, 64, 64), activate_type="mish")
    input_data = tf.concat([input_data, route], axis=-1)

    input_data = common.convolutional(input_data, (1, 1, 128, 128), activate_type="mish")
    input_data = common.convolutional(input_data, (3, 3, 128, 256), downsample=True, activate_type="mish")
    route = input_data
    route = common.convolutional(route, (1, 1, 256, 128), activate_type="mish")
    input_data = common.convolutional(input_data, (1, 1, 256, 128), activate_type="mish")
    for i in range(8):
        input_data = common.residual_block(input_data, 128, 128, 128, activate_type="mish")
    input_data = common.convolutional(input_data, (1, 1, 128, 128), activate_type="mish")
    input_data = tf.concat([input_data, route], axis=-1)

    input_data = common.convolutional(input_data, (1, 1, 256, 256), activate_type="mish")
    route_1 = input_data
    input_data = common.convolutional(input_data, (3, 3, 256, 512), downsample=True, activate_type="mish")
    route = input_data
    route = common.convolutional(route, (1, 1, 512, 256), activate_type="mish")
    input_data = common.convolutional(input_data, (1, 1, 512, 256), activate_type="mish")
    for i in range(8):
        input_data = common.residual_block(input_data, 256, 256, 256, activate_type="mish")
    input_data = common.convolutional(input_data, (1, 1, 256, 256), activate_type="mish")
    input_data = tf.concat([input_data, route], axis=-1)

    input_data = common.convolutional(input_data, (1, 1, 512, 512), activate_type="mish")
    route_2 = input_data
    input_data = common.convolutional(input_data, (3, 3, 512, 1024), downsample=True, activate_type="mish")
    route = input_data
    route = common.convolutional(route, (1, 1, 1024, 512), activate_type="mish")
    input_data = common.convolutional(input_data, (1, 1, 1024, 512), activate_type="mish")
    for i in range(4):
        input_data = common.residual_block(input_data, 512, 512, 512, activate_type="mish")
    input_data = common.convolutional(input_data, (1, 1, 512, 512), activate_type="mish")
    input_data = tf.concat([input_data, route], axis=-1)

    input_data = common.convolutional(input_data, (1, 1, 1024, 1024), activate_type="mish")
    input_data = common.convolutional(input_data, (1, 1, 1024, 512))
    input_data = common.convolutional(input_data, (3, 3, 512, 1024))
    input_data = common.convolutional(input_data, (1, 1, 1024, 512))

    input_data = tf.concat([tf.nn.max_pool(input_data, ksize=13, padding='SAME', strides=1), tf.nn.max_pool(input_data, ksize=9, padding='SAME', strides=1)
                            , tf.nn.max_pool(input_data, ksize=5, padding='SAME', strides=1), input_data], axis=-1)
    input_data = common.convolutional(input_data, (1, 1, 2048, 512))
    input_data = common.convolutional(input_data, (3, 3, 512, 1024))
    input_data = common.convolutional(input_data, (1, 1, 1024, 512))

    return route_1, route_2, input_data

def cspdarknet53_att_v1(input_data,
                 att_layers = [1, 1, 1],
                 att_heads = [4, 4, 4],
                 att_activation = 'mish',
                 att_normal = 2):

    input_data = common.convolutional(input_data, (3, 3,  3,  32), activate_type="mish")
    input_data = common.convolutional(input_data, (3, 3, 32,  64), downsample=True, activate_type="mish")

    route = input_data
    route = common.convolutional(route, (1, 1, 64, 64), activate_type="mish")
    input_data = common.convolutional(input_data, (1, 1, 64, 64), activate_type="mish")
    for i in range(1):
        input_data = common.residual_block(input_data,  64,  32, 64, activate_type="mish")
    input_data = common.convolutional(input_data, (1, 1, 64, 64), activate_type="mish")

    input_data = tf.concat([input_data, route], axis=-1)
    input_data = common.convolutional(input_data, (1, 1, 128, 64), activate_type="mish")
    input_data = common.convolutional(input_data, (3, 3, 64, 128), downsample=True, activate_type="mish")
    route = input_data
    route = common.convolutional(route, (1, 1, 128, 64), activate_type="mish")
    input_data = common.convolutional(input_data, (1, 1, 128, 64), activate_type="mish")
    for i in range(2):
        input_data = common.residual_block(input_data, 64,  64, 64, activate_type="mish")
    input_data = common.convolutional(input_data, (1, 1, 64, 64), activate_type="mish")
    input_data = tf.concat([input_data, route], axis=-1)

    input_data = common.convolutional(input_data, (1, 1, 128, 128), activate_type="mish")
    input_data = common.convolutional(input_data, (3, 3, 128, 256), downsample=True, activate_type="mish")
    route = input_data
    route = common.convolutional(route, (1, 1, 256, 128), activate_type="mish")
    input_data = common.convolutional(input_data, (1, 1, 256, 128), activate_type="mish")
    for i in range(8):
        input_data = common.residual_block(input_data, 128, 128, 128, activate_type="mish")
    input_data = common.convolutional(input_data, (1, 1, 128, 128), activate_type="mish")
    input_data = tf.concat([input_data, route], axis=-1)

    input_data = common.convolutional(input_data, (1, 1, 256, 256), activate_type="mish")
    route_1 = input_data
    route_1_shape = getattr(route_1, 'shape')
    patches = Patches(1)(route_1)
    encoded_patches = PatchEncoder((route_1_shape[1]**2), route_1_shape[-1])(patches)
    encoded_patches = common.transformer(encoded_patches, projection_dim = route_1_shape[-1], transformer_units=[route_1_shape[-1]*2, route_1_shape[-1]], num_layers = att_layers[0], num_heads = att_heads[0], activation=att_activation, normal = att_normal)
    encoded_patches = tf.keras.layers.Reshape(route_1.shape[1:])(encoded_patches)
    route_1 = route_1 + encoded_patches

    input_data = common.convolutional(input_data, (3, 3, 256, 512), downsample=True, activate_type="mish")
    route = input_data
    route = common.convolutional(route, (1, 1, 512, 256), activate_type="mish")
    input_data = common.convolutional(input_data, (1, 1, 512, 256), activate_type="mish")
    for i in range(8):
        input_data = common.residual_block(input_data, 256, 256, 256, activate_type="mish")
    input_data = common.convolutional(input_data, (1, 1, 256, 256), activate_type="mish")
    input_data = tf.concat([input_data, route], axis=-1)

    input_data = common.convolutional(input_data, (1, 1, 512, 512), activate_type="mish")
    route_2 = input_data
    route_2_shape = getattr(route_2, 'shape')
    patches = Patches(1)(route_2)
    encoded_patches = PatchEncoder((route_2_shape[1]**2), route_2_shape[-1])(patches)
    encoded_patches = common.transformer(encoded_patches, projection_dim = route_2_shape[-1], transformer_units=[route_2_shape[-1]*2, route_2_shape[-1]], num_layers = att_layers[1], num_heads = att_heads[1], activation=att_activation, normal = att_normal)
    encoded_patches = tf.keras.layers.Reshape(route_2.shape[1:])(encoded_patches)
    route_2 = route_2 + encoded_patches

    input_data = common.convolutional(input_data, (3, 3, 512, 1024), downsample=True, activate_type="mish")
    route = input_data
    route = common.convolutional(route, (1, 1, 1024, 512), activate_type="mish")
    input_data = common.convolutional(input_data, (1, 1, 1024, 512), activate_type="mish")
    for i in range(4):
        input_data = common.residual_block(input_data, 512, 512, 512, activate_type="mish")
    input_data = common.convolutional(input_data, (1, 1, 512, 512), activate_type="mish")
    input_data = tf.concat([input_data, route], axis=-1)

    input_data = common.convolutional(input_data, (1, 1, 1024, 1024), activate_type="mish")
    input_data = common.convolutional(input_data, (1, 1, 1024, 512))
    input_data = common.convolutional(input_data, (3, 3, 512, 1024))
    input_data = common.convolutional(input_data, (1, 1, 1024, 512))

    input_data = tf.concat([tf.nn.max_pool(input_data, ksize=13, padding='SAME', strides=1), tf.nn.max_pool(input_data, ksize=9, padding='SAME', strides=1)
                            , tf.nn.max_pool(input_data, ksize=5, padding='SAME', strides=1), input_data], axis=-1)
    input_data = common.convolutional(input_data, (1, 1, 2048, 512))
    input_data = common.convolutional(input_data, (3, 3, 512, 1024))
    input_data = common.convolutional(input_data, (1, 1, 1024, 512))
    input_data_shape = getattr(input_data, 'shape')
    patches = Patches(1)(input_data)
    encoded_patches = PatchEncoder((input_data_shape[1]**2), input_data_shape[-1])(patches)
    encoded_patches = common.transformer(encoded_patches, projection_dim = input_data_shape[-1], transformer_units=[input_data_shape[-1]*2, input_data_shape[-1]], num_layers = att_layers[2], num_heads = att_heads[1], activation=att_activation, normal = att_normal)
    encoded_patches = tf.keras.layers.Reshape(input_data.shape[1:])(encoded_patches)
    input_data = input_data + encoded_patches
    return route_1, route_2, input_data

def cspdarknet53_tiny(input_data):
    input_data = common.convolutional(input_data, (3, 3, 3, 32), downsample=True)
    input_data = common.convolutional(input_data, (3, 3, 32, 64), downsample=True)
    input_data = common.convolutional(input_data, (3, 3, 64, 64))

    route = input_data
    input_data = common.route_group(input_data, 2, 1)
    input_data = common.convolutional(input_data, (3, 3, 32, 32))
    route_1 = input_data
    input_data = common.convolutional(input_data, (3, 3, 32, 32))
    input_data = tf.concat([input_data, route_1], axis=-1)
    input_data = common.convolutional(input_data, (1, 1, 32, 64))
    input_data = tf.concat([route, input_data], axis=-1)
    input_data = tf.keras.layers.MaxPool2D(2, 2, 'same')(input_data)

    input_data = common.convolutional(input_data, (3, 3, 64, 128))
    route = input_data
    input_data = common.route_group(input_data, 2, 1)
    input_data = common.convolutional(input_data, (3, 3, 64, 64))
    route_1 = input_data
    input_data = common.convolutional(input_data, (3, 3, 64, 64))
    input_data = tf.concat([input_data, route_1], axis=-1)
    input_data = common.convolutional(input_data, (1, 1, 64, 128))
    input_data = tf.concat([route, input_data], axis=-1)
    input_data = tf.keras.layers.MaxPool2D(2, 2, 'same')(input_data)

    input_data = common.convolutional(input_data, (3, 3, 128, 256))
    route = input_data
    input_data = common.route_group(input_data, 2, 1)
    input_data = common.convolutional(input_data, (3, 3, 128, 128))
    route_1 = input_data
    input_data = common.convolutional(input_data, (3, 3, 128, 128))
    input_data = tf.concat([input_data, route_1], axis=-1)
    input_data = common.convolutional(input_data, (1, 1, 128, 256))
    route_1 = input_data
    input_data = tf.concat([route, input_data], axis=-1)
    input_data = tf.keras.layers.MaxPool2D(2, 2, 'same')(input_data)

    input_data = common.convolutional(input_data, (3, 3, 512, 512))

    return route_1, input_data

def darknet53_tiny(input_data):
    input_data = common.convolutional(input_data, (3, 3, 3, 16))
    input_data = tf.keras.layers.MaxPool2D(2, 2, 'same')(input_data)
    input_data = common.convolutional(input_data, (3, 3, 16, 32))
    input_data = tf.keras.layers.MaxPool2D(2, 2, 'same')(input_data)
    input_data = common.convolutional(input_data, (3, 3, 32, 64))
    input_data = tf.keras.layers.MaxPool2D(2, 2, 'same')(input_data)
    input_data = common.convolutional(input_data, (3, 3, 64, 128))
    input_data = tf.keras.layers.MaxPool2D(2, 2, 'same')(input_data)
    input_data = common.convolutional(input_data, (3, 3, 128, 256))
    route_1 = input_data
    input_data = tf.keras.layers.MaxPool2D(2, 2, 'same')(input_data)
    input_data = common.convolutional(input_data, (3, 3, 256, 512))
    input_data = tf.keras.layers.MaxPool2D(2, 1, 'same')(input_data)
    input_data = common.convolutional(input_data, (3, 3, 512, 1024))

    return route_1, input_data



def VIT_v1(inputs, image_size = 416,
                          patch_size=8,
                          projection_dim = 128,
                          transformer_layers =[6, 6, 6],
                          attention_heads=[4, 4, 4],
                          activation='gelu',
                          normal=0):
    temp_norm = normal if normal<3 else normal-3
    num_patches = (image_size // patch_size) ** 2
    transformer_units = [projection_dim * 2, projection_dim] 
    # inputs = layers.Input(shape=(image_size, image_size, 3))
    patches = Patches(patch_size)(inputs)
    encoded_patches = PatchEncoder(num_patches, projection_dim)(patches)    
    
    encoded_patches = common.transformer(encoded_patches, projection_dim, transformer_units, transformer_layers[0], num_heads = attention_heads[0], activation = activation, normal = temp_norm)
    if normal <3:
        encoded_patches = layers.BatchNormalization()(encoded_patches)
    else:
        encoded_patches = tfa.layers.GroupNormalization(groups = min(projection_dim, 16))(encoded_patches)
    route1 = encoded_patches
    route1 = tf.keras.layers.Reshape((image_size//8, image_size//8, projection_dim))(route1)
    dict_size = getattr(encoded_patches, 'shape')
    encoded_patches = tf.expand_dims(encoded_patches, axis = -1)
    encoded_patches = Patches_sp([4, projection_dim])(encoded_patches)
    # if not down_sample[0]:
    encoded_patches = PatchEncoder(dict_size[1]//4, projection_dim*2)(encoded_patches) 
    # else:
    #     encoded_patches = tf.keras.layers.MaxPooling1D(pool_size = 4, strides=4)(encoded_patches)

    encoded_patches = common.transformer(encoded_patches, projection_dim*2, [projection_dim*4, projection_dim*2], transformer_layers[1], num_heads = attention_heads[1], activation = activation, normal = temp_norm)
    if normal <3:
        encoded_patches = layers.BatchNormalization()(encoded_patches)
    else:
        encoded_patches = tfa.layers.GroupNormalization(groups = min(projection_dim, 16))(encoded_patches)
    route2 = encoded_patches
    route2 = tf.keras.layers.Reshape((image_size//16, image_size//16, projection_dim*2))(route2)
    encoded_patches = tf.expand_dims(encoded_patches, axis = -1)
    encoded_patches = Patches_sp([4, projection_dim*2])(encoded_patches)
    encoded_patches = PatchEncoder(dict_size[1]//16, projection_dim*4)(encoded_patches)    
    encoded_patches = common.transformer(encoded_patches, projection_dim*4, [projection_dim*8, projection_dim*4], transformer_layers[2], num_heads = attention_heads[2], activation = activation, normal = temp_norm)
    if normal <3:
        encoded_patches = layers.BatchNormalization()(encoded_patches)
    else:
        encoded_patches = tfa.layers.GroupNormalization(groups = min(projection_dim, 16))(encoded_patches)
    encoded_patches = tf.keras.layers.Reshape((image_size//32, image_size//32, projection_dim*4))(encoded_patches)
    # model = keras.Model(inputs=inputs, outputs=[route1, route2, encoded_patches])
    return route1, route2, encoded_patches

def VIT_v2(inputs, image_size = 416,
                          patch_size=8,
                          projection_dim = 128,
                          transformer_layers =[6, 6, 6],
                          attention_heads=[4, 4, 4],
                          activation='gelu',
                          normal=0):
    temp_norm = normal if normal<3 else normal-3
    num_patches = (image_size // patch_size) ** 2
    transformer_units = [projection_dim * 2, projection_dim] 
    # inputs = layers.Input(shape=(image_size, image_size, 3))
    patches = Patches(patch_size)(inputs)
    encoded_patches = PatchEncoder(num_patches, projection_dim)(patches)    
    encoded_patches = common.transformer(encoded_patches, projection_dim, transformer_units, transformer_layers[0], num_heads = attention_heads[0], activation = activation, normal = temp_norm)
    if normal <3:
        encoded_patches = layers.BatchNormalization()(encoded_patches)
    else:
        encoded_patches = tfa.layers.GroupNormalization(groups = min(projection_dim, 16))(encoded_patches)
    route1 = encoded_patches
    route1 = tf.keras.layers.Reshape((image_size//8, image_size//8, projection_dim))(route1)
    dict_size = getattr(encoded_patches, 'shape')
    #encoded_patches = tf.expand_dims(encoded_patches, axis = -1)
    #encoded_patches = Patches_sp([4, projection_dim])(encoded_patches)
    # if not down_sample[0]:
    # encoded_patches = PatchEncoder(dict_size[1]//4, projection_dim*2)(encoded_patches) 
    # else:
    #     encoded_patches = tf.keras.layers.MaxPooling1D(pool_size = 4, strides=4)(encoded_patches)

    encoded_patches = common.transformer(encoded_patches, projection_dim*2, [projection_dim*2, projection_dim*1], transformer_layers[1], num_heads = attention_heads[1], activation = activation, normal = temp_norm)
    if normal <3:
        encoded_patches = layers.BatchNormalization()(encoded_patches)
    else:
        encoded_patches = tfa.layers.GroupNormalization(groups = min(projection_dim, 16))(encoded_patches)
    route2 = encoded_patches
    route2 = tf.keras.layers.Reshape((image_size//8, image_size//8, projection_dim))(route2)
    #encoded_patches = tf.expand_dims(encoded_patches, axis = -1)
    #encoded_patches = Patches_sp([4, projection_dim*2])(encoded_patches)
    #encoded_patches = PatchEncoder(dict_size[1]//16, projection_dim*4)(encoded_patches)    
    encoded_patches = common.transformer(encoded_patches, projection_dim, [projection_dim*2, projection_dim*1], transformer_layers[2], num_heads = attention_heads[2], activation = activation, normal = temp_norm)
    if normal <3:
        encoded_patches = layers.BatchNormalization()(encoded_patches)
    else:
        encoded_patches = tfa.layers.GroupNormalization(groups = min(projection_dim, 16))(encoded_patches)
    encoded_patches = tf.keras.layers.Reshape((image_size//8, image_size//8, projection_dim))(encoded_patches)
    # model = keras.Model(inputs=inputs, outputs=[route1, route2, encoded_patches])
    return route1, route2, encoded_patches

def VIT_v3(inputs, image_size = 416,
                          patch_size=8,
                          projection_dim = 128,
                          transformer_layers =[6, 6, 6],
                          attention_heads=[4, 4, 4],
                          activation='gelu',
                          normal=0):
    temp_norm = normal if normal<3 else normal-3
    num_patches = (image_size // patch_size) ** 2
    transformer_units = [projection_dim * 2, projection_dim] 
    # inputs = layers.Input(shape=(image_size, image_size, 3))
    patches = Patches(patch_size)(inputs)
    encoded_patches = PatchEncoder(num_patches, projection_dim)(patches)    
    
    encoded_patches = common.transformer(encoded_patches, projection_dim, transformer_units, transformer_layers[0], num_heads = attention_heads[0], activation = activation, normal = temp_norm)
    if normal <3:
        encoded_patches = layers.BatchNormalization()(encoded_patches)
    else:
        encoded_patches = tfa.layers.GroupNormalization(groups = min(projection_dim, 16))(encoded_patches)
    route1 = encoded_patches
    route1 = tf.keras.layers.Reshape((image_size//8, image_size//8, projection_dim))(route1)
    dict_size = getattr(encoded_patches, 'shape')
    route1_shape = getattr(route1, 'shape')
    num_patches = (route1_shape[1] // (2)) ** 2
    patches = Patches(2)(route1)
    encoded_patches = PatchEncoder(num_patches, projection_dim*2)(patches)    
    # if not down_sample[0]:
    # else:
    #     encoded_patches = tf.keras.layers.MaxPooling1D(pool_size = 4, strides=4)(encoded_patches)

    encoded_patches = common.transformer(encoded_patches, projection_dim*2, [projection_dim*4, projection_dim*2], transformer_layers[1], num_heads = attention_heads[1], activation = activation, normal = temp_norm)
    if normal <3:
        encoded_patches = layers.BatchNormalization()(encoded_patches)
    else:
        encoded_patches = tfa.layers.GroupNormalization(groups = min(projection_dim, 16))(encoded_patches)
    route2 = encoded_patches
    route2 = tf.keras.layers.Reshape((image_size//16, image_size//16, projection_dim*2))(route2)
    route2_shape = getattr(route2, 'shape')
    num_patches = (route2_shape[1] // (2)) ** 2
    patches = Patches(2)(route2)
    encoded_patches = PatchEncoder(num_patches, projection_dim*4)(patches)   
    encoded_patches = common.transformer(encoded_patches, projection_dim*4, [projection_dim*8, projection_dim*4], transformer_layers[2], num_heads = attention_heads[2], activation = activation, normal = temp_norm)
    if normal <3:
        encoded_patches = layers.BatchNormalization()(encoded_patches)
    else:
        encoded_patches = tfa.layers.GroupNormalization(groups = min(projection_dim, 16))(encoded_patches)
    encoded_patches = tf.keras.layers.Reshape((image_size//32, image_size//32, projection_dim*4))(encoded_patches)
    # model = keras.Model(inputs=inputs, outputs=[route1, route2, encoded_patches])
    return route1, route2, encoded_patches

def VIT_v5(inputs, image_size = 416,
                          patch_size=2,
                          projection_dim = 128,
                          transformer_layers =[6, 6, 6],
                          attention_heads=[4, 4, 4],
                          activation='gelu',
                          normal=0):
    temp_norm = normal if normal<3 else normal-3
    num_patches = (image_size // patch_size) ** 2
    transformer_units = [8 * 2, 8] 
    # inputs = layers.Input(shape=(image_size, image_size, 3))


    patches = Patches(patch_size)(inputs)
    encoded_patches = PatchEncoder(num_patches, 8)(patches)    
    
    encoded_patches = common.transformer(encoded_patches, 8, transformer_units, 2, num_heads = attention_heads[0], activation = activation, normal = temp_norm)
    if normal <3:
        encoded_patches = layers.BatchNormalization()(encoded_patches)
    else:
        encoded_patches = tfa.layers.GroupNormalization(groups = 8)(encoded_patches)
    encoded_patches = tf.keras.layers.Reshape((image_size//2, image_size//2, encoded_patches.shape[-1]))(encoded_patches)
    patches = Patches(2)(encoded_patches)
    num_patches = (image_size // (patch_size*2)) ** 2
    encoded_patches = PatchEncoder(num_patches, 16)(patches) 
    transformer_units = [16 * 2, 16] 

    encoded_patches = common.transformer(encoded_patches, 16, transformer_units, 4, num_heads = attention_heads[0], activation = activation, normal = temp_norm)
    if normal <3:
        encoded_patches = layers.BatchNormalization()(encoded_patches)
    else:
        encoded_patches = tfa.layers.GroupNormalization(groups = 16)(encoded_patches)
    encoded_patches = tf.keras.layers.Reshape((image_size//4, image_size//4, encoded_patches.shape[-1]))(encoded_patches)
    patches = Patches(2)(encoded_patches)
    num_patches = (image_size // (patch_size*4)) ** 2
    encoded_patches = PatchEncoder(num_patches, 32)(patches) 
    transformer_units = [32 * 2, 32] 
    encoded_patches = common.transformer(encoded_patches, 32, transformer_units, 8, num_heads = attention_heads[0], activation = activation, normal = temp_norm)
    if normal <3:
        encoded_patches = layers.BatchNormalization()(encoded_patches)
    else:
        encoded_patches = tfa.layers.GroupNormalization(groups = 16)(encoded_patches)

    encoded_patches = common.transformer(encoded_patches, projection_dim, transformer_units, transformer_layers[0], num_heads = attention_heads[0], activation = activation, normal = temp_norm)
    if normal <3:
        encoded_patches = layers.BatchNormalization()(encoded_patches)
    else:
        encoded_patches = tfa.layers.GroupNormalization(groups = 16)(encoded_patches)
    route1 = encoded_patches
    route1 = tf.keras.layers.Reshape((image_size//8, image_size//8, encoded_patches.shape[-1]))(route1)
    dict_size = getattr(encoded_patches, 'shape')
    route1_shape = getattr(route1, 'shape')
    num_patches = (route1_shape[1] // (2)) ** 2
    patches = Patches(2)(route1)
    encoded_patches = PatchEncoder(num_patches, projection_dim*2)(patches)    
    encoded_patches = common.transformer(encoded_patches, projection_dim*2, [projection_dim*4, projection_dim*2], transformer_layers[1], num_heads = attention_heads[1], activation = activation, normal = temp_norm)
    if normal <3:
        encoded_patches = layers.BatchNormalization()(encoded_patches)
    else:
        encoded_patches = tfa.layers.GroupNormalization(groups = min(projection_dim, 16))(encoded_patches)
    route2 = encoded_patches
    route2 = tf.keras.layers.Reshape((image_size//16, image_size//16, projection_dim*2))(route2)
    route2_shape = getattr(route2, 'shape')
    num_patches = (route2_shape[1] // (2)) ** 2
    patches = Patches(2)(route2)
    encoded_patches = PatchEncoder(num_patches, projection_dim*4)(patches)   
    encoded_patches = common.transformer(encoded_patches, projection_dim*4, [projection_dim*8, projection_dim*4], transformer_layers[2], num_heads = attention_heads[2], activation = activation, normal = temp_norm)
    if normal <3:
        encoded_patches = layers.BatchNormalization()(encoded_patches)
    else:
        encoded_patches = tfa.layers.GroupNormalization(groups = min(projection_dim, 16))(encoded_patches)
    encoded_patches = tf.keras.layers.Reshape((image_size//32, image_size//32, projection_dim*4))(encoded_patches)
    return route1, route2, encoded_patches

def VIT_v1_tiny(inputs, image_size = 416,
                          patch_size=16,
                          projection_dim = 128,
                          transformer_layers =[6, 6],
                          attention_heads=[4, 4],
                          activation = 'gelu'):
    
    num_patches = (image_size // patch_size) ** 2
    transformer_units = [projection_dim * 2, projection_dim] 
    # inputs = layers.Input(shape=(image_size, image_size, 3))
    patches = Patches(patch_size)(inputs)
    encoded_patches = PatchEncoder(num_patches, projection_dim)(patches)    
    encoded_patches = common.transformer(encoded_patches, projection_dim, transformer_units, transformer_layers[0], num_heads = attention_heads[0], activation = activation)
    encoded_patches = layers.LayerNormalization(epsilon=1e-6)(encoded_patches)
    route1 = encoded_patches
    route1 = tf.keras.layers.Reshape((image_size//16, image_size//16, projection_dim))(route1)
    dict_size = getattr(encoded_patches, 'shape')
    encoded_patches = tf.expand_dims(encoded_patches, axis = -1)
    encoded_patches = Patches_sp([4, projection_dim])(encoded_patches)
    # if not down_sample[0]:
    encoded_patches = PatchEncoder(dict_size[1]//4, projection_dim*2)(encoded_patches) 
    # else:
    #     encoded_patches = tf.keras.layers.MaxPooling1D(pool_size = 4, strides=4)(encoded_patches)

    encoded_patches = common.transformer(encoded_patches, projection_dim*2, [projection_dim*4, projection_dim*2], transformer_layers[1], num_heads = attention_heads[1], activation = activation)
    encoded_patches = layers.LayerNormalization(epsilon=1e-6)(encoded_patches)
    route2 = encoded_patches
    route2 = tf.keras.layers.Reshape((image_size//32, image_size//32, projection_dim*2))(route2)

    # model = keras.Model(inputs=inputs, outputs=[route1, route2, encoded_patches])
    return route1, route2



def cspdarkernet53(input_data,
                   attention_axes = 1,
                   activation = 'mish',
                   normalization = 'group'):

    input_data = common.transformer_block(input_data, out_filt = 32,
                                          activation = activation,
                                          down_sample = False,
                                          attention_axes = attention_axes,
                                          kernel_size = 3,
                                          normalization = normalization)
    
    input_data = common.transformer_block(input_data, out_filt = 64,
                                          activation = activation,
                                          down_sample = True,
                                          attention_axes = attention_axes,
                                          kernel_size = 3,
                                          normalization = normalization)
    route = input_data
    route = common.transformer_block(route, out_filt = 64,
                                          activation = activation,
                                          down_sample = False,
                                          attention_axes = attention_axes,
                                          kernel_size = 1,
                                          normalization = normalization)
    
    input_data = common.transformer_block(input_data, out_filt = 64,
                                          activation = activation,
                                          down_sample = False,
                                          attention_axes = attention_axes,
                                          kernel_size = 1,
                                          normalization = normalization)
    for i in range(1):
        input_data = common.transformer_block(input_data, out_filt = 64,
                                          activation = activation,
                                          down_sample = False,
                                          attention_axes = attention_axes,
                                          kernel_size = 3,
                                          normalization = normalization)
    input_data = common.transformer_block(input_data, out_filt = 64,
                                          activation = activation,
                                          down_sample = False,
                                          attention_axes = attention_axes,
                                          kernel_size = 1,
                                          normalization = normalization)

    input_data = tf.concat([input_data, route], axis=-1)
    input_data = common.transformer_block(input_data, out_filt = 64,
                                          activation = activation,
                                          down_sample = False,
                                          attention_axes = attention_axes,
                                          kernel_size = 1,
                                          normalization = normalization)
    input_data = common.transformer_block(input_data, out_filt = 128,
                                          activation = activation,
                                          down_sample = True,
                                          attention_axes = attention_axes,
                                          kernel_size = 3,
                                          normalization = normalization)
    route = input_data
    route = common.transformer_block(route, out_filt = 64,
                                          activation = activation,
                                          down_sample = False,
                                          attention_axes = attention_axes,
                                          kernel_size = 1,
                                          normalization = normalization)
    
    input_data = common.transformer_block(input_data, out_filt = 64,
                                          activation = activation,
                                          down_sample = False,
                                          attention_axes = attention_axes,
                                          kernel_size = 1,
                                          normalization = normalization)
    for i in range(2):
        input_data = common.transformer_block(input_data, out_filt = 64,
                                          activation = activation,
                                          down_sample = False,
                                          attention_axes = attention_axes,
                                          kernel_size = 3,
                                          normalization = normalization)
    
    input_data = common.transformer_block(input_data, out_filt = 64,
                                          activation = activation,
                                          down_sample = False,
                                          attention_axes = attention_axes,
                                          kernel_size = 1,
                                          normalization = normalization)
    input_data = tf.concat([input_data, route], axis=-1)

    input_data = common.transformer_block(input_data, out_filt = 128,
                                          activation = activation,
                                          down_sample = False,
                                          attention_axes = attention_axes,
                                          kernel_size = 1,
                                          normalization = normalization)
    input_data = common.transformer_block(input_data, out_filt = 256,
                                          activation = activation,
                                          down_sample = True,
                                          attention_axes = attention_axes,
                                          kernel_size = 3,
                                          normalization = normalization)
    route = input_data
    route = common.transformer_block(route, out_filt = 128,
                                          activation = activation,
                                          down_sample = False,
                                          attention_axes = attention_axes,
                                          kernel_size = 1,
                                          normalization = normalization)
    input_data = common.transformer_block(input_data, out_filt = 128,
                                          activation = activation,
                                          down_sample = False,
                                          attention_axes = attention_axes,
                                          kernel_size = 1,
                                          normalization = normalization)
    for i in range(8):
        input_data = common.transformer_block(input_data, out_filt = 128,
                                          activation = activation,
                                          down_sample = False,
                                          attention_axes = attention_axes,
                                          kernel_size = 3,
                                          normalization = normalization)
    input_data = common.transformer_block(input_data, out_filt = 128,
                                          activation = activation,
                                          down_sample = False,
                                          attention_axes = attention_axes,
                                          kernel_size = 1,
                                          normalization = normalization)
    input_data = tf.concat([input_data, route], axis=-1)
    input_data = common.transformer_block(input_data, out_filt = 256,
                                          activation = activation,
                                          down_sample = False,
                                          attention_axes = attention_axes,
                                          kernel_size = 1,
                                          normalization = normalization)
    route_1 = input_data
    input_data = common.transformer_block(input_data, out_filt = 512,
                                          activation = activation,
                                          down_sample = True,
                                          attention_axes = attention_axes,
                                          kernel_size = 3,
                                          normalization = normalization)
    route = input_data
    input_data = common.transformer_block(input_data, out_filt = 256,
                                          activation = activation,
                                          down_sample = False,
                                          attention_axes = attention_axes,
                                          kernel_size = 1,
                                          normalization = normalization)

    for i in range(8):
        input_data = common.transformer_block(input_data, out_filt = 256,
                                          activation = activation,
                                          down_sample = False,
                                          attention_axes = attention_axes,
                                          kernel_size = 3,
                                          normalization = normalization)
    input_data = common.transformer_block(input_data, out_filt = 256,
                                          activation = activation,
                                          down_sample = False,
                                          attention_axes = attention_axes,
                                          kernel_size = 1,
                                          normalization = normalization)
    input_data = tf.concat([input_data, route], axis=-1)

    input_data = common.convolutional(input_data, (1, 1, 512, 512), activate_type="mish")
    input_data = common.transformer_block(input_data, out_filt = 512,
                                          activation = activation,
                                          down_sample = False,
                                          attention_axes = attention_axes,
                                          kernel_size = 1,
                                          normalization = normalization)
    route_2 = input_data
    input_data = common.transformer_block(input_data, out_filt = 1024,
                                          activation = activation,
                                          down_sample = True,
                                          attention_axes = attention_axes,
                                          kernel_size = 3,
                                          normalization = normalization)
    route = input_data
    route = common.transformer_block(route, out_filt = 512,
                                          activation = activation,
                                          down_sample = False,
                                          attention_axes = attention_axes,
                                          kernel_size = 1,
                                          normalization = normalization)
    input_data = common.transformer_block(input_data, out_filt = 512,
                                          activation = activation,
                                          down_sample = False,
                                          attention_axes = attention_axes,
                                          kernel_size = 1,
                                          normalization = normalization)
    for i in range(4):
        input_data = common.transformer_block(input_data, out_filt = 512,
                                          activation = activation,
                                          down_sample = False,
                                          attention_axes = attention_axes,
                                          kernel_size = 3,
                                          normalization = normalization)
    input_data = common.transformer_block(input_data, out_filt = 512,
                                          activation = activation,
                                          down_sample = False,
                                          attention_axes = attention_axes,
                                          kernel_size = 1,
                                          normalization = normalization)
    input_data = tf.concat([input_data, route], axis=-1)
    input_data = common.transformer_block(input_data, out_filt = 1024,
                                          activation = activation,
                                          down_sample = False,
                                          attention_axes = attention_axes,
                                          kernel_size = 1,
                                          normalization = normalization)
    input_data = common.convolutional(input_data, (1, 1, 1024, 512))
    input_data = common.convolutional(input_data, (3, 3, 512, 1024))
    input_data = common.convolutional(input_data, (1, 1, 1024, 512))

    input_data = tf.concat([tf.nn.max_pool(input_data, ksize=13, padding='SAME', strides=1), tf.nn.max_pool(input_data, ksize=9, padding='SAME', strides=1)
                            , tf.nn.max_pool(input_data, ksize=5, padding='SAME', strides=1), input_data], axis=-1)
    input_data = common.convolutional(input_data, (1, 1, 2048, 512))
    input_data = common.convolutional(input_data, (3, 3, 512, 1024))
    input_data = common.convolutional(input_data, (1, 1, 1024, 512))

    return route_1, route_2, input_data


def darkernet53(input_data,
                   attention_axes = 1,
                   activation = 'mish',
                   normalization = 'group'):

    input_data = common.transformer_block(input_data, out_filt = 32,
                                          activation = activation,
                                          down_sample = False,
                                          attention_axes = attention_axes,
                                          kernel_size = 3,
                                          normalization = normalization)
    input_data = common.transformer_block(input_data, out_filt = 64,
                                          activation = activation,
                                          down_sample = True,
                                          attention_axes = attention_axes,
                                          kernel_size = 3,
                                          normalization = normalization)

    for i in range(1):
        input_data = common.transformer_block(input_data, out_filt = 64,
                                          activation = activation,
                                          down_sample = False,
                                          attention_axes = attention_axes,
                                          kernel_size = 3,
                                          normalization = normalization)

    input_data = common.transformer_block(input_data, out_filt = 128,
                                          activation = activation,
                                          down_sample = True,
                                          attention_axes = attention_axes,
                                          kernel_size = 3,
                                          normalization = normalization)
    for i in range(2):
        input_data = common.residual_block(input_data, 128,  64, 128)
        input_data = common.transformer_block(input_data, out_filt = 128,
                                          activation = activation,
                                          down_sample = False,
                                          attention_axes = attention_axes,
                                          kernel_size = 3,
                                          normalization = normalization)

    input_data = common.transformer_block(input_data, out_filt = 256,
                                          activation = activation,
                                          down_sample = True,
                                          attention_axes = attention_axes,
                                          kernel_size = 3,
                                          normalization = normalization)
    for i in range(8):
        input_data = common.transformer_block(input_data, out_filt = 256,
                                          activation = activation,
                                          down_sample = False,
                                          attention_axes = attention_axes,
                                          kernel_size = 3,
                                          normalization = normalization)
    route_1 = input_data
    input_data = common.transformer_block(input_data, out_filt = 256,
                                          activation = activation,
                                          down_sample = True,
                                          attention_axes = attention_axes,
                                          kernel_size = 3,
                                          normalization = normalization)
    for i in range(8):
        input_data = common.transformer_block(input_data, out_filt = 512,
                                          activation = activation,
                                          down_sample = False,
                                          attention_axes = attention_axes,
                                          kernel_size = 3,
                                          normalization = normalization)

    route_2 = input_data
    input_data = common.transformer_block(input_data, out_filt = 1024,
                                          activation = activation,
                                          down_sample = True,
                                          attention_axes = attention_axes,
                                          kernel_size = 3,
                                          normalization = normalization)
    for i in range(4):
        input_data = common.transformer_block(input_data, out_filt = 1024,
                                          activation = activation,
                                          down_sample = False,
                                          attention_axes = attention_axes,
                                          kernel_size = 3,
                                          normalization = normalization)

    return route_1, route_2, input_data

def cspdarkerattnet53(input_data,
                   attention_axes = 1,
                   activation = 'mish',
                   normalization = 'group'):

    
    input_data = common.convolutional(input_data, (3, 3,  3,  32), activate_type="mish")
    input_data = common.convolutional(input_data, (3, 3, 32,  64), downsample=True, activate_type="mish")

    route = input_data
    route = common.convolutional(route, (1, 1, 64, 64), activate_type="mish")
    input_data = common.convolutional(input_data, (1, 1, 64, 64), activate_type="mish")
    for i in range(1):
        input_data = common.transformer_block(input_data, out_filt = 64,
                                          activation = activation,
                                          down_sample = False,
                                          attention_axes = attention_axes,
                                          kernel_size = 3,
                                          normalization = normalization)

    input_data = common.convolutional(input_data, (1, 1, 64, 64), activate_type="mish")

    input_data = tf.concat([input_data, route], axis=-1)
    input_data = common.convolutional(input_data, (1, 1, 128, 64), activate_type="mish")
    input_data = common.convolutional(input_data, (3, 3, 64, 128), downsample=True, activate_type="mish")
    route = input_data
    route = common.convolutional(route, (1, 1, 128, 64), activate_type="mish")
    input_data = common.convolutional(input_data, (1, 1, 128, 64), activate_type="mish")
    for i in range(2):
        input_data = common.transformer_block(input_data, out_filt = 64,
                                          activation = activation,
                                          down_sample = False,
                                          attention_axes = attention_axes,
                                          kernel_size = 3,
                                          normalization = normalization)

    input_data = common.convolutional(input_data, (1, 1, 64, 64), activate_type="mish")
    input_data = tf.concat([input_data, route], axis=-1)

    input_data = common.convolutional(input_data, (1, 1, 128, 128), activate_type="mish")
    input_data = common.convolutional(input_data, (3, 3, 128, 256), downsample=True, activate_type="mish")
    route = input_data
    route = common.convolutional(route, (1, 1, 256, 128), activate_type="mish")
    input_data = common.convolutional(input_data, (1, 1, 256, 128), activate_type="mish")
    for i in range(8):
        input_data = common.transformer_block(input_data, out_filt = 128,
                                          activation = activation,
                                          down_sample = False,
                                          attention_axes = attention_axes,
                                          kernel_size = 3,
                                          normalization = normalization)

    input_data = common.convolutional(input_data, (1, 1, 128, 128), activate_type="mish")
    input_data = tf.concat([input_data, route], axis=-1)

    input_data = common.convolutional(input_data, (1, 1, 256, 256), activate_type="mish")
    route_1 = input_data
    input_data = common.convolutional(input_data, (3, 3, 256, 512), downsample=True, activate_type="mish")
    route = input_data
    route = common.convolutional(route, (1, 1, 512, 256), activate_type="mish")
    input_data = common.convolutional(input_data, (1, 1, 512, 256), activate_type="mish")
    for i in range(8):
        input_data = common.transformer_block(input_data, out_filt = 256,
                                          activation = activation,
                                          down_sample = False,
                                          attention_axes = attention_axes,
                                          kernel_size = 3,
                                          normalization = normalization)

    input_data = common.convolutional(input_data, (1, 1, 256, 256), activate_type="mish")
    input_data = tf.concat([input_data, route], axis=-1)

    input_data = common.convolutional(input_data, (1, 1, 512, 512), activate_type="mish")
    route_2 = input_data
    input_data = common.convolutional(input_data, (3, 3, 512, 1024), downsample=True, activate_type="mish")
    route = input_data
    route = common.convolutional(route, (1, 1, 1024, 512), activate_type="mish")
    input_data = common.convolutional(input_data, (1, 1, 1024, 512), activate_type="mish")
    for i in range(4):
        input_data = common.transformer_block(input_data, out_filt = 512,
                                          activation = activation,
                                          down_sample = False,
                                          attention_axes = attention_axes,
                                          kernel_size = 3,
                                          normalization = normalization)

    input_data = common.convolutional(input_data, (1, 1, 512, 512), activate_type="mish")
    input_data = tf.concat([input_data, route], axis=-1)

    input_data = common.convolutional(input_data, (1, 1, 1024, 1024), activate_type="mish")
    input_data = common.convolutional(input_data, (1, 1, 1024, 512))
    input_data = common.convolutional(input_data, (3, 3, 512, 1024))
    input_data = common.convolutional(input_data, (1, 1, 1024, 512))

#     input_data = tf.concat([tf.nn.max_pool(input_data, ksize=13, padding='SAME', strides=1), tf.nn.max_pool(input_data, ksize=9, padding='SAME', strides=1)
#                             , tf.nn.max_pool(input_data, ksize=5, padding='SAME', strides=1), input_data], axis=-1)
#     input_data = common.convolutional(input_data, (1, 1, 2048, 512))
#     input_data = common.convolutional(input_data, (3, 3, 512, 1024))
#     input_data = common.convolutional(input_data, (1, 1, 1024, 512))

    return route_1, route_2, input_data
