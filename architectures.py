import tensorflow as tf
import numpy as np


#------------------------------------------------------------------------------------
def ResBlock(inputs, dim, ks=3, with_batch_norm=True, activation='relu'):
    x = inputs
    
    if with_batch_norm:
        x = tf.keras.layers.BatchNormalization()(x)
        
    x = tf.keras.layers.Activation(activation)(x)
    x = tf.keras.layers.Conv2D(dim, ks, padding='same')(x)
    
    if with_batch_norm:
        x = tf.keras.layers.BatchNormalization()(x)
        
    x = tf.keras.layers.Activation(activation)(x)
    x = tf.keras.layers.Conv2D(dim, ks, padding='same')(x)
    return inputs + x
#------------------------------------------------------------------------------------

def f0(input_shape):
    xin = tf.keras.layers.Input(input_shape)
    x = tf.keras.layers.Conv2D(64, 5, 1, padding='same', activation="relu")(xin)
    x = tf.keras.layers.MaxPool2D(2)(x)
    x = tf.keras.layers.Conv2D(128, 5, 1, padding='same', activation="relu")(x)
    x = tf.keras.layers.MaxPool2D(2)(x)
    x = tf.keras.layers.Conv2D(256, 5, 1, padding='same', activation="relu")(x)
    x = tf.keras.layers.MaxPool2D(2)(x)
    return tf.keras.Model(xin, x)
    
def f1(input_shape):
    xin = tf.keras.layers.Input(input_shape)
    x = tf.keras.layers.Conv2D(128, 3, 2, padding='same', activation="swish")(xin)
    x = tf.keras.layers.Conv2D(256, 3, 2, padding='same', activation="swish")(x)
    x = tf.keras.layers.Conv2D(256, 3, 2, padding='same', activation="relu")(x)
    return tf.keras.Model(xin, x)

def decoder0(input_shape, channels):
    xin = tf.keras.layers.Input(input_shape)
    x = tf.keras.layers.Conv2DTranspose(256, 3, 2, padding='same', activation="relu")(xin)
    x = tf.keras.layers.Conv2DTranspose(128, 3, 2, padding='same', activation="relu")(x)
    x = tf.keras.layers.Conv2DTranspose(channels, 3, 2, padding='same', activation="tanh")(x)
    return tf.keras.Model(xin, x)

def D0(input_shape):
    xin = tf.keras.layers.Input(input_shape)
    x = tf.keras.layers.Conv2D(256, 3, 1, padding='same')(xin)
    x = ResBlock(x, 256)
    x = ResBlock(x, 256)
    x = ResBlock(x, 256)
    x = ResBlock(x, 256)
    x = ResBlock(x, 256)
    x = tf.keras.layers.Conv2D(256, 3, 2, padding='same')(x)
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(1)(x)
    return tf.keras.Model(xin, x)

def C0(input_shape=None, channels=None):
    xin = tf.keras.layers.Input(input_shape)
    x = tf.keras.layers.Flatten()(xin)
    logits = tf.keras.layers.Dense(1)(x)
    return tf.keras.Model(xin, logits)

#=============================================================================================


SETUPS = [
    [f0, f1, decoder0, D0],
    [f0, f1, C0, D0],
]
