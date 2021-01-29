import tensorflow as tf

# from "NoPeek: Information leakage reduction to share activations in distributed deep learning" Vepakomma et al.

def pairwise_dist(A):
    r = tf.reduce_sum(A*A, 1)
    r = tf.reshape(r, [-1, 1])
    D = tf.maximum(r - 2*tf.matmul(A, tf.transpose(A)) + tf.transpose(r), 1e-7)
    D = tf.sqrt(D)
    return D

def dist_corr(X, Y):
    X = tf.keras.layers.Flatten()(X)
    Y = tf.keras.layers.Flatten()(Y)
    n = tf.cast(tf.shape(X)[0], tf.float32)
    a = pairwise_dist(X)
    b = pairwise_dist(Y)
    A = a - tf.reduce_mean(a, axis=1) -\
    tf.expand_dims(tf.reduce_mean(a,axis=0),axis=1)+\
    tf.reduce_mean(a)
    B = b - tf.reduce_mean(b, axis=1) -\
    tf.expand_dims(tf.reduce_mean(b,axis=0),axis=1)+\
    tf.reduce_mean(b)
    dCovXY = tf.sqrt(tf.reduce_sum(A*B) / (n ** 2))
    dVarXX = tf.sqrt(tf.reduce_sum(A*A) / (n ** 2))
    dVarYY = tf.sqrt(tf.reduce_sum(B*B) / (n ** 2))
    
    dCorXY = dCovXY / tf.sqrt(dVarXX * dVarYY)
    return dCorXY