import tensorflow as tf
import tqdm
import numpy as np

from archs import *
from dataset import *

LR = 1e-4

class Server:
    def __init__(self, make_s, clients, trainset, batch_size):
        self.server = make_s()
        self.clients = clients
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=LR)
        
        self.trainset = trainset.shuffle(BUFFER_SIZE).batch(batch_size, drop_remainder=True).repeat(-1)
    
    def train_step(self, step0, step1, client_trainable_variables, x=None, y=None):

        # simulate comunication loop split learning for a training step
        with tf.GradientTape(persistent=True) as tape:
            # initial inference on client-side
            z, y = step0(x, y)

            # intermediate inference on server
            zs  = self.server(z)

            # final inference step + loss client-side
            loss = step1(zs, y)

        # server is always trained; clients decide what to train (as it happens client-side)
        var = self.server.trainable_variables + client_trainable_variables    
        gradients = tape.gradient(loss, var)
        self.optimizer.apply_gradients(zip(gradients, var))
        
        return loss
    
    def __call__(self, iterations):
        LOG = []
        LOG_BAD = []
        
        dds = iter(self.trainset)
        
        # simulate training loop among clients
        for i in tqdm.trange(iterations):
            # for each client sequentially
            for j, client in enumerate(self.clients):
                                
                # attacker's turn:
                if client.isbad:
                    if i % 2 == 0:
                        # train generator
                        log_bad = self.train_step(client.step0_G, client.step1_G, client.trainable_variables_G())
                        LOG_BAD.append(log_bad)
                    else:
                        # poisoning
                        self.train_step(client.step0_poison, client.step1, client.trainable_variables())
                
                # honest cleint's turn:
                else:
                    # get a batch from training set for honest client d
                    x, y = next(dds)

                    # train the classifier
                    log = self.train_step(client.step0, client.step1, client.trainable_variables(), x, y)
                    LOG.append(log)
                    
        return LOG, LOG_BAD