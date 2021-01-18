import tensorflow as tf
import numpy as np
from tensorflow.keras import layers

from archs import *
from dataset import *

class Client:
    def __init__(self, f, ff):
        self.f = f
        self.ff = ff
        self.isbad = False
                    
    @tf.function
    def step0(self, x, y):
        z = self.f(x, training=True)
        return z, y
    
    @tf.function
    def step1(self, z, y):
        y_ = self.ff(z, training=True)
        loss = self.loss(y, y_)
        
        return loss
    
    def loss(self, y, y_):
        loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)(y, y_);
        return loss
    
    def trainable_variables(self):
        return self.f.trainable_variables + self.ff.trainable_variables
#---------------------------------------------------------------------------------------

gradient_factor = 100000.0

@tf.custom_gradient
def scale_down(x):
    def grad(dy):
        return dy / gradient_factor 
    return tf.identity(x), grad

@tf.custom_gradient
def scale_up(x):
    def grad(dy):
        return dy * gradient_factor
    return tf.identity(x), grad
    
class BadClient(Client):
    def __init__(self, f, ff, G, batch_size, yt):
        super().__init__(f, ff)
        self.isbad = True
        self.G = G
        self.batch_size = batch_size
                
        # target class
        self.yt = yt
        
        # the lasr class is used to poisoning 
        self.ytt = NUM_OF_CLASSES - 1
        
    def sample(self):
        return tf.random.normal([self.batch_size, LATENT_SIZE])
        
    def generate(self):
        zG = self.sample()
        x_ = self.G(zG, training=True)
        return x_
        
    @tf.function
    def step0_G(self, x, y):
        zG = self.sample()
        x_ = self.G(zG, training=True)
        z = self.f(x_, training=True)
        
        # scale-back gradient and backpropagate it to f<-G
        z = scale_up(z)
        
        # target class (for adversarial training)
        y_ = np.ones(self.batch_size) * self.yt
        
        return z, y_
    
    
    @tf.function
    def step1_G(self, z, y):
        
        # force gradient close to zero
        z = scale_down(z)
        
        y_ = self.ff(z, training=True)
        loss = self.loss(y, y_)
        
        return loss
    
    
    @tf.function
    def step0_poison(self, x, y):
        zG = self.sample()
        x_ = self.G(zG, training=True)
        z = self.f(x_, training=True)
        
        y_ = np.ones(self.batch_size) * self.ytt
        
        return z, y_
        
    
    def trainable_variables_G(self):
        # client trains only G
        return self.G.trainable_variables
        