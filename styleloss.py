import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np
import tqdm
import matplotlib.pyplot as plt
import tensorflow_hub as hub

style_layers = ['conv_pw_1_relu',
                'conv_pw_2_relu',
                'conv_pw_3_relu',
                'conv_pw_4_relu',
                'conv_pw_5_relu']

num_style_layers = len(style_layers)

def vgg_layers(layer_names, x_shape):
    vgg = tf.keras.applications.MobileNet(input_shape=x_shape, include_top=False, weights='imagenet')
    vgg.trainable = False

    outputs = [vgg.get_layer(name).output for name in layer_names]

    model = tf.keras.Model([vgg.input], outputs)
    return model

def gram_matrix(input_tensor):
    result = tf.linalg.einsum('bijc,bijd->bcd', input_tensor, input_tensor)
    input_shape = tf.shape(input_tensor)
    num_locations = tf.cast(input_shape[1]*input_shape[2], tf.float32)
    return result/(num_locations)

def makeStyleModel(x_shape):
    style_extractor = vgg_layers(style_layers, x_shape)

    x_in = tf.keras.layers.Input(x_shape)
    x = tf.keras.applications.mobilenet.preprocess_input(x_in)
    
    style_outputs = style_extractor(x)
    style_outputs = [gram_matrix(style_output) for style_output in style_outputs]

    style = tf.keras.Model(x_in, style_outputs)
    
    return style

def getStyleLoss(x, x_target, style):
    style_outputs = style(x)
    style_targets = style(x_target)

    assert len(style_targets) == len(style_outputs)
    style_loss = tf.add_n([tf.reduce_mean((style_outputs[i]-style_targets[i])**2) for i in range(len(style_targets))])
    style_loss /= num_style_layers
    
    return style_loss