import tensorflow as tf
import numpy as np
from tensorflow.keras import layers

LATENT_SIZE = 100
NUM_OF_CLASSES = 10 + 1

def make_generator():
    model = tf.keras.Sequential()
    model.add(layers.Dense(7*7*256, use_bias=False, input_shape=(LATENT_SIZE,)))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Reshape((7, 7, 256)))
    assert model.output_shape == (None, 7, 7, 256) # Note: None is the batch size

    model.add(layers.Conv2DTranspose(128, (5, 5), strides=(1, 1), padding='same', use_bias=False))
    assert model.output_shape == (None, 7, 7, 128)
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Conv2DTranspose(64, (5, 5), strides=(2, 2), padding='same', use_bias=False))
    assert model.output_shape == (None, 14, 14, 64)
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Conv2DTranspose(1, (5, 5), strides=(2, 2), padding='same', use_bias=False, activation='tanh'))
    assert model.output_shape == (None, 28, 28, 1)

    return model

def make_f():
    xin = layers.Input((28, 28, 1))

    x = layers.Conv2D(64, (5, 5), strides=(2, 2), padding='same')(xin)
    x = layers.LeakyReLU()(x)
    x = layers.Dropout(0.3)(x)
    
    return tf.keras.Model(xin, x)

def make_s():
    model = tf.keras.Sequential()
    model.add(layers.Conv2D(128, (5, 5), strides=(2, 2), padding='same', input_shape=[14, 14, 64]))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3))
    return model

def make_ff():
    xin = layers.Input((7, 7, 128))
    
    x = layers.Flatten()(xin)
    x = layers.Dense(NUM_OF_CLASSES)(x)
    
    return tf.keras.Model(xin, x)