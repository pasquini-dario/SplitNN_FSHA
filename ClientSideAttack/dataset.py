import tensorflow as tf
import numpy as np

BUFFER_SIZE = 10000

def parse(x):
    x = x[:,:,None]
    x = x / (255/2) - 1
    x = tf.clip_by_value(x, -1., 1.)
    return x

def make_dataset(X, Y, f):
    x = tf.data.Dataset.from_tensor_slices(X)
    y = tf.data.Dataset.from_tensor_slices(Y)
    x = x.map(f)
    xy = tf.data.Dataset.zip((x, y))
    xy = xy.shuffle(BUFFER_SIZE)
    return xy

def load_mnist():
    mnist = tf.keras.datasets.mnist
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = x_train.astype(np.float32)
    x_test = x_test.astype(np.float32)
    
    xpriv = make_dataset(x_train, y_train, parse)
    xpub = make_dataset(x_test, y_test, parse)
    
    return xpriv, xpub