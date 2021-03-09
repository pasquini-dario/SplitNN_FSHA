import tensorflow as tf
import numpy as np
import tqdm
import sklearn
import matplotlib.pyplot as plt
import tensorflow_datasets as tfds

BUFFER_SIZE = 10000
SIZE = 32

getImagesDS = lambda X, n: np.concatenate([x[0].numpy()[None,] for x in X.take(n)])

def parse(x):
    x = x[:,:,None]
    x = tf.tile(x, (1,1,3))    
    x = tf.image.resize(x, (SIZE, SIZE))
    x = x / (255/2) - 1
    x = tf.clip_by_value(x, -1., 1.)
    return x

def parseC(x):
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



def load_mnist_mangled(class_to_remove):
    mnist = tf.keras.datasets.mnist
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = x_train.astype(np.float32)
    x_test = x_test.astype(np.float32)
    
    # remove class from Xpub
    (x_test, y_test), _ = remove_class(x_test, y_test, class_to_remove)
    # for evaluation
    (x_train_seen, y_train_seen), (x_removed_examples, y_removed_examples) = remove_class(x_train, y_train, class_to_remove)
    
    xpriv = make_dataset(x_train, y_train, parse)
    xpub = make_dataset(x_test, y_test, parse)
    xremoved_examples = make_dataset(x_removed_examples, y_removed_examples, parse)
    
    xpriv_other = make_dataset(x_train_seen, y_train_seen, parse)
    
    return xpriv, xpub, xremoved_examples, xpriv_other


def load_fashion_mnist():
    mnist = tf.keras.datasets.fashion_mnist
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    x_train = x_train.astype(np.float32)
    x_test = x_test.astype(np.float32)
    
    xpriv = make_dataset(x_train, y_train, parse)
    xpub = make_dataset(x_test, y_test, parse)
    
    return xpriv, xpub

def remove_class(X, Y, ctr):
    mask = Y!=ctr
    XY = X[mask], Y[mask]
    mask = Y==ctr
    XYr = X[mask], Y[mask]
    return XY, XYr

def plot(X, label='', norm=True):
    n = len(X)
    X = (X+1) / 2 
    fig, ax = plt.subplots(1, n, figsize=(n*3,3))
    for i in range(n):
        ax[i].imshow(X[i]);  
        ax[i].set(xticks=[], yticks=[], title=label)
