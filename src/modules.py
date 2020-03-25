import tensorflow as tf
import numpy as np

def upsampling_module(x, filters, kernels, strides=2, dropout=None, apply_batchnorm=True):
    x = tf.keras.layers.Conv2DTranspose(filters=filters, kernel_size=kernels, strides=strides, padding='same', use_bias=not apply_batchnorm)(x)

    if apply_batchnorm:
        x = tf.keras.layers.BatchNormalization()(x)

    x = tf.keras.layers.ReLU()(x)

    if dropout:
        x = tf.keras.layers.Dropout(rate=dropout)(x)

    return x


def downsampling_module(x, filters, kernels, strides=2, dropout=None, apply_batchnorm=True):
    x = tf.keras.layers.Conv2D(filters=filters, kernel_size=kernels, strides=strides, padding='same', use_bias=not apply_batchnorm)(x)

    if apply_batchnorm:
        x = tf.keras.layers.BatchNormalization()(x)

    x = tf.keras.layers.LeakyReLU()(x)

    if dropout:
        x = tf.keras.layers.Dropout(rate=dropout)(x)

    return x
