# -*- coding: utf-8 -*-
"""
Created on Thu Jan  3 20:23:41 2019

@author: DART_HSU
"""
import tensorflow as tf
import time
import numpy as np

from modules import *
from process_data import ProcessData as pdata

'''
# Attetion Model
(layer - hyperparameter name)
------------------------
Encoder:
    dense - e_FC1_size
    encoder_rnn_lstm 
        4 layers LSTMs - e_LSTM_size
------------------------
Attention:
    multihead_attention - a_units_size, dropout_rate
    dense - a_FC1_size
-------------------------
Output:
    dense - o_FC1_size
------------------------- 
'''
# hyperparameter:  
inputs_size = 3600 # data input ( demand_taxi shape: 60*60 ) 
timestep = 24 # time step {demand_taxi}
outputs_size = 3600

e_FC1_size = 2800
e_LSTM_size = 480
a_units_size = 480 # units_size: must be divided by 8
a_FC1_size = 1600
o_FC1_size = 3600

dropout_rate = 0.5
# hyperparameter: training 
batch_size = 15 # batch_size: must be multiply of 255
training_iters = 801
gamma = 0.01 # loss function gamma
lr = 0.0001 # learning rate

class Graph():
    
    def __init__(self, is_training=True):
        self.graph = tf.Graph()
        self.is_training = is_training
        
        if self.is_training==False: # Remove dropout layer when it's testing.
            self.dropout_rate = 0.0
        else:
            self.dropout_rate = dropout_rate
            
        with self.graph.as_default():
            self.x = tf.placeholder(tf.float32, [None, inputs_size], name='x')
            self.y = tf.placeholder(tf.float32, [None, outputs_size], name='y')
            
            with tf.variable_scope('encoder'): 
                # 1 layer fully connection
                self.x_encoder = tf.layers.dense(self.x, e_FC1_size, activation=tf.nn.relu)
                
                # 4 layers lstms
                self.x_encoder = encoder_rnn_lstm(self.x_encoder, e_FC1_size, e_LSTM_size, 24)
                
                # 1 layer attention
                self.x_encoder = tf.layers.dropout(self.x_encoder, rate=dropout_rate)
            
            with tf.variable_scope('attention'):
                
                self.x_attention = multihead_attention(queries=self.x_encoder,
                                                        keys=self.x_encoder,
                                                        units_size=a_units_size,
                                                        heads_size=8,
                                                        dropout_rate=dropout_rate,
                                                        is_training=True,
                                                        causality=True,
                                                        reuse=None,
                                                        scope='multihead_attention')
                    
                # 1 layer fully connection
                self.x_attention = tf.layers.dense(self.x_attention, a_FC1_size, activation=tf.nn.relu)
                  
            self.x_output =  tf.layers.dense(self.x_attention, o_FC1_size, activation=tf.nn.relu )
            self.pred = tf.reshape(self.x_output, [-1, outputs_size])
            
            if is_training:  
                # h: gamma
                # lr: learning_rate
                self.loss =  tf.reduce_sum( tf.subtract( tf.square(self.y-self.pred), tf.multiply( gamma, tf.div(tf.square(self.y-self.pred), tf.add(self.y, 1.0)))))
                self.train_op = tf.train.AdamOptimizer(lr).minimize(self.loss)
                
                # tensorboard: loss
                tf.summary.scalar('Loss', self.loss)
                
            with tf.name_scope('Evaluation'):      
                # MAPE : if it not y+1, will be inf 
                self.MAPE = tf.reduce_mean( tf.divide( tf.abs( tf.subtract(self.pred, self.y)), tf.add(self.y, 1.0)))
                # RMSE : if it not y+1, will be inf
                self.RMSE = tf.sqrt( tf.reduce_mean( tf.square( tf.subtract(self.pred, self.y))))

            self.merged = tf.summary.merge_all()
    
if __name__  =='__main__':
    pdata = pdata()
    training_X, training_Y, testing_X, testing_Y = pdata.get_taxi_data_24()
    training_X_batch = np.shape(np.reshape(training_X, (-1, 24, 3600)))[0] # (B, T, N)  255*24*3600
    
    g = Graph()
    config = tf.ConfigProto()
    # config.gpu_options.allow_growth = True
    with tf.Session(graph=g.graph, config=config) as sess:
        # tensorboard writer
        writer = tf.summary.FileWriter('logs/', sess.graph)
        
        tStart = time.time()
        sess.run(tf.global_variables_initializer())
        step = 0
        
        while step < training_iters:
            
            for i in range(int(training_X_batch/batch_size)): 
                # e.g. [0:1*24*5] ->[1*5*24:2*24*5]
                start = i*batch_size*timestep
                end = (i+1)*batch_size*timestep
                sess.run(g.train_op, feed_dict={g.x:training_X[start: end], g.y:training_Y[start: end]})
            
            if step % 20 == 0:
                loss_ = sess.run(g.loss, feed_dict={g.x:training_X, g.y:training_Y})
                print('Iteration:' + str(step) + ', Loss:' + str(loss_) )
                
                MAPE_, RMSE_ = sess.run([g.MAPE, g.RMSE], feed_dict={g.x:training_X, g.y:training_Y})
                print('Tr_MAPE:' + str(MAPE_) + ', Tr_RMSE:' + str(RMSE_) )                
                
                # tensorboard: 
                summary = sess.run(g.merged, feed_dict={g.x:training_X, g.y:training_Y})
                writer.add_summary(summary, step)

            step += 1
        
        g.is_training = False
        prediction, t_MAPE_, t_RMSE_ = sess.run([g.pred, g.MAPE, g.RMSE], feed_dict={g.x:testing_X, g.y:testing_Y})
        print('Te_MAPE:' + str(t_MAPE_) + ', Te_RMSE:' + str(t_RMSE_) )         
        
        # save prediction to  a file
        pdata.prediction_to_csv(prediction, 'C_main')
        
        tEnd = time.time()
        print ("It cost %f sec" % (tEnd - tStart))    