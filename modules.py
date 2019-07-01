# -*- coding: utf-8 -*-
"""
Created on Thu Jan  3 19:32:27 2019

@author: DART_HSU
"""

import tensorflow as tf

def conv2d(inputs, outputs_size, dropout_rate=0.0):
    # [batch_size, 3600] -> [batch_size, 60, 60]  
    inputs = tf.reshape(inputs, [-1, 60, 60, 1])
    
    conv2d_1 = tf.layers.conv2d(inputs,
                                filters=32,
                                kernel_size=(2, 2),
                                strides=(2, 2), 
                                padding='same')
    
    conv2d_2 = tf.layers.conv2d(conv2d_1,
                                filters=64,
                                kernel_size=(2, 2),
                                strides=(2, 2),
                                padding='same')
    
    flt = tf.layers.flatten(conv2d_2)
    
    fc = tf.layers.dense(flt, outputs_size)
    outputs = tf.layers.dropout(inputs=fc, rate=dropout_rate)
   
    return outputs

def encoder_rnn_lstm(inputs,
             inputs_size,
             units_size=200,
             timestep=24):
    
    # transpose the inputs shape (B, T, N)
    inputs = tf.reshape(inputs, [-1, timestep, inputs_size])

    # create a BasicRNNCell
    cells = []
    cells.append(tf.nn.rnn_cell.BasicLSTMCell(units_size))
    for _ in range(3):
        cells.append(tf.nn.rnn_cell.ResidualWrapper(tf.nn.rnn_cell.BasicLSTMCell(units_size)) )
    
    stacked_rnn_cell = tf.nn.rnn_cell.MultiRNNCell(cells)
    
    # 'state' is a tensor of shape [batch_size, cell_state_size]
    # 'outputs' is a tensor of shape [batch_size, max_time, cell_state_size]
    outputs, state = tf.nn.dynamic_rnn(stacked_rnn_cell, inputs, dtype=tf.float32)
                        
    return outputs
    
def decoder_rnn_lstm(inputs,
             inputs_size,
             units_size=600,
             timestep=24):
    
    # transpose the inputs shape (B, T, N)
    inputs = tf.reshape(inputs, [-1, timestep, inputs_size])

    # create a BasicRNNCell
    cells = []
    cells.append(tf.nn.rnn_cell.BasicLSTMCell(units_size))
    for _ in range(3):
        cells.append(tf.nn.rnn_cell.ResidualWrapper(tf.nn.rnn_cell.BasicLSTMCell(units_size)) )
    
    stacked_rnn_cell = tf.nn.rnn_cell.MultiRNNCell(cells)
    
    # 'state' is a tensor of shape [batch_size, cell_state_size]
    # 'outputs' is a tensor of shape [batch_size, max_time, cell_state_size]
    outputs, state = tf.nn.dynamic_rnn(stacked_rnn_cell, inputs, dtype=tf.float32)
                        
    return outputs

'''
reference: https://github.com/Kyubyong/transformer/blob/master/modules.py
'''
def multihead_attention(queries,
                        keys,
                        units_size=200,
                        heads_size=8,
                        dropout_rate=0,
                        is_training=True,
                        is_normalization=True,
                        causality=True,
                        reuse=None,
                        scope="multihead_attention"):

    with tf.variable_scope(scope, reuse=reuse):
        
        # Linear projections
        Q = tf.layers.dense(queries, units_size, activation=tf.nn.relu) # (B, T_q, C)
        K = tf.layers.dense(keys, units_size, activation=tf.nn.relu) # (B, T_k, C)
        V = tf.layers.dense(keys, units_size, activation=tf.nn.relu) # (B, T_k, C)
        
        '''
        tf.concat:
        t1 = [[1, 2, 3], [4, 5, 6]]
        t2 = [[7, 8, 9], [10, 11, 12]]
        tf.concat([t1, t2], 0)  # [[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]]
        tf.concat([t1, t2], 1)  # [[1, 2, 3, 7, 8, 9], [4, 5, 6, 10, 11, 12]]
        '''
        # Split and concat
        Q_ = tf.concat(tf.split(Q, heads_size, axis=2), axis=0) # (head*B, T_q, C/h) 
        K_ = tf.concat(tf.split(K, heads_size, axis=2), axis=0) # (head*B, T_k, C/h) 
        V_ = tf.concat(tf.split(V, heads_size, axis=2), axis=0) # (head*B, T_k, C/h) 
        
        # Multiplication
        outputs = tf.matmul(Q_, tf.transpose(K_, [0, 2, 1])) # (h*B, T_q, T_k)
        
        # Scale
        outputs = outputs / (K_.get_shape().as_list()[-1] ** 0.5)
        
        # Key Masking
        key_masks = tf.sign(tf.abs(tf.reduce_sum(keys, axis=-1))) # (B, T_k)
        key_masks = tf.tile(key_masks, [heads_size, 1]) # (h*B, T_k)
        key_masks = tf.tile(tf.expand_dims(key_masks, 1), [1, tf.shape(queries)[1], 1]) # (h*B, T_q, T_k)
    
        # Activation
        outputs = tf.nn.softmax(outputs) # (h*B, T_q, T_k)
         
        # Query Masking
        query_masks = tf.sign(tf.abs(tf.reduce_sum(queries, axis=-1))) # (B, T_q)
        query_masks = tf.tile(query_masks, [heads_size, 1]) # (h*B, T_q)
        query_masks = tf.tile(tf.expand_dims(query_masks, -1), [1, 1, tf.shape(keys)[1]]) # (h*B, T_q, T_k)
        outputs *= query_masks # broadcasting. (B, T_q, C)
          
        # Dropouts
        outputs = tf.layers.dropout(outputs, rate=dropout_rate, training=tf.convert_to_tensor(is_training))
               
        # Weighted sum
        outputs = tf.matmul(outputs, V_) # ( h*B, T_q, C/h)
        
        # Restore shape
        outputs = tf.concat(tf.split(outputs, heads_size, axis=0), axis=2 ) # (B, T_q, C)
              
        # Residual connection
        outputs += queries
              
        # Normalize
        # equation: gamma * (inputs - mean) / ( (variance + epsilon) ** (.5) )
        if is_normalization:
            outputs = tf.layers.batch_normalization(outputs) # (B, T_q, C)
        
    return outputs