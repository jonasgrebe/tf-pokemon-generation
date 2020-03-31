import tensorflow as tf
import numpy as np

from spectral_norm import SpectralNorm

def upsampling_module(x, filters, kernels, strides=2, dropout=None, batch_norm=True, spectral_norm=False, initializer='glorot_uniform'):
    x = tf.keras.layers.Conv2DTranspose(filters=filters, kernel_size=kernels, strides=strides, padding='same', use_bias=not batch_norm,
                                        kernel_initializer=initializer, kernel_constraint=SpectralNorm() if spectral_norm else None)(x)

    if batch_norm:
        x = tf.keras.layers.BatchNormalization()(x)

    x = tf.keras.layers.LeakyReLU()(x)

    if dropout:
        x = tf.keras.layers.Dropout(dropout)(x)

    return x


def downsampling_module(x, filters, kernels, strides=2, dropout=None, batch_norm=True, spectral_norm=False, initializer='glorot_uniform'):
    x = tf.keras.layers.Conv2D(filters=filters, kernel_size=kernels, strides=strides, padding='same', use_bias=not batch_norm,
                                   kernel_initializer=initializer, kernel_constraint=SpectralNorm() if spectral_norm else None)(x)

    if batch_norm:
        x = tf.keras.layers.BatchNormalization()(x)

    x = tf.keras.layers.LeakyReLU()(x)

    if dropout:
        x = tf.keras.layers.Dropout(rate=dropout)(x)

    return x
