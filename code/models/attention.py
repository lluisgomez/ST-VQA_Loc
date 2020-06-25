#-*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np
import os, sys

rnn_cell = tf.contrib.rnn

class Attention():
    def __init__(self, rnn_size, rnn_layer, batch_size, dim_image, dim_hidden, dim_attention, max_words_q, text_embedding_dim, drop_out_rate, training=True):

        self.rnn_size = rnn_size
        self.rnn_layer = rnn_layer
        self.batch_size = batch_size
        self.dim_image = dim_image
        self.dim_hidden = dim_hidden
        self.dim_att = dim_attention
        self.max_words_q = max_words_q
        self.text_embedding_dim = text_embedding_dim
        self.drop_out_rate = drop_out_rate
        self.training = training

        # Q encoder: RNN body
        self.lstm_1 = rnn_cell.LSTMCell(rnn_size, use_peepholes=True)
        self.lstm_dropout_1 = rnn_cell.DropoutWrapper(self.lstm_1, output_keep_prob = 1 - self.drop_out_rate)
        self.lstm_2 = rnn_cell.LSTMCell(rnn_size, use_peepholes=True)
        self.lstm_dropout_2 = rnn_cell.DropoutWrapper(self.lstm_2, output_keep_prob = 1 - self.drop_out_rate)
        self.stacked_lstm = rnn_cell.MultiRNNCell([self.lstm_dropout_1, self.lstm_dropout_2])


    def build_model(self, image_features, txt_features, question):
        
        state = self.stacked_lstm.zero_state(self.batch_size, tf.float32)
        with tf.variable_scope("embed"):
            for i in range(self.max_words_q):
                ques_emb = question[:,i,:]
                output, state = self.stacked_lstm(ques_emb, state)

        question_emb = tf.reshape(tf.transpose(state, [2, 1, 0, 3]), [self.batch_size, -1])

        image_emb = tf.concat([image_features,txt_features], 3)
        image_emb = self.conv_layer(image_emb, (1, 1, self.dim_image[2], self.dim_hidden), activation='tanh')

        # attention layers
        with tf.variable_scope("att1"):
            prob_att1, comb_emb = self.attention_layer(question_emb, image_emb, activation='softmax')
        with tf.variable_scope("att2"):
            prob_att2, _ = self.attention_layer(comb_emb, image_emb, activation=None)

        # activation in the loss function during training
        if not self.training:
            prob_att2 = tf.nn.sigmoid(prob_att2)

        prob_att = tf.reshape(prob_att2, [-1, 38, 38])
        return prob_att

    def conv_layer(self, inp, kernels_shape, activation=None):
        xavier = tf.contrib.layers.xavier_initializer()
        kernels = np.zeros(kernels_shape, dtype=np.float32)
        biases  = np.zeros((kernels.shape[-1],), dtype=np.float32)
        conv_kernels = tf.Variable(xavier(kernels.shape), name='ConvKernels')
        conv_biases  = tf.Variable(biases, name='ConvBiases')
        x = tf.nn.conv2d(inp, conv_kernels, padding = 'SAME', strides = [1, 1, 1, 1])
        x = tf.nn.bias_add(x, conv_biases)

        if activation == 'tanh':
          x = tf.tanh(x)

        return x
    

    def attention_layer(self, question_emb, image_emb, activation='softmax'):

        question_att = tf.expand_dims(question_emb, 1)
        question_att = tf.tile(question_att, tf.constant([1, self.dim_image[0] * self.dim_image[1], 1]))
        question_att = tf.reshape(question_att, [-1, 38, 38, self.dim_hidden])
        question_att = self.conv_layer(question_att, (1, 1, self.dim_hidden, self.dim_att), activation='tanh')
        
        image_att = self.conv_layer(image_emb, (1, 1, self.dim_hidden, self.dim_att), activation=None)

        output_att = tf.tanh(image_att + question_att)
        output_att = tf.nn.dropout(output_att, 1 - self.drop_out_rate)

        prob_att  = self.conv_layer(output_att, (1, 1, self.dim_att, 1), activation=None)
        prob_att = tf.reshape(prob_att, [self.batch_size, self.dim_image[0] * self.dim_image[1]])

        if activation == 'softmax':
            prob_att = tf.nn.softmax(prob_att)

        image_att = []
        image_emb = tf.reshape(image_emb, [self.batch_size, self.dim_image[0] * self.dim_image[1], self.dim_hidden])
        for b in range(self.batch_size):
            image_att.append(tf.matmul(tf.expand_dims(prob_att[b,:],0), image_emb[b,:,:]))

        image_att = tf.stack(image_att)
        image_att = tf.reduce_sum(image_att, 1)

        comb_emb = tf.add(image_att, question_emb)

        return prob_att, comb_emb
    
