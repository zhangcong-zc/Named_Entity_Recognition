# !/usr/bin/env python 
# -*- coding: UTF-8 -*- 
# @Time: 2020/12/10 17:30
# @Author: Zhang Cong

import tensorflow as tf
from config import Config
from tensorflow.contrib import rnn
from tensorflow.contrib import crf

class Model():

    def __init__(self):
        self.config = Config()                                                                                   # 配置参数
        self.input_x = tf.placeholder(shape=[None, self.config.seq_length], dtype=tf.int32, name='input-x')      # 输入文本
        self.input_y = tf.placeholder(shape=[None, self.config.seq_length], dtype=tf.int32, name='input-y')      # 输入文本对应的true label
        self.input_length = tf.placeholder(shape=[None], dtype=tf.int32, name='input-length')                    # 输入文本的长度
        self.input_keep_prob = tf.placeholder(dtype=tf.float32, name='input-keep-prob')                          # keep-prob

        # Embedding layer
        embedding = tf.get_variable(shape=[self.config.vocab_size, self.config.embedding_dim], dtype=tf.float32, name='embedding')
        embedding_x = tf.nn.embedding_lookup(params=embedding, ids=self.input_x)        # dim:(batch_size, max_length, 300)

        # Bi-LSTM/Bi-GRU
        cell_fw = self.get_rnn(self.config.rnn_type)    # 前向cell
        cell_bw = self.get_rnn(self.config.rnn_type)    # 后向cell
        outputs, states = tf.nn.bidirectional_dynamic_rnn(cell_fw=cell_fw, cell_bw=cell_bw, inputs=embedding_x, dtype=tf.float32)
        outputs = tf.concat(values=outputs, axis=2)     # 将前向cell和后向cell的结果进行concat拼接   dim:(batch_size, max_length, 2*hidden_dim)
        outputs = tf.layers.dropout(inputs=outputs, rate=self.input_keep_prob)

        # 输出层   dim:(batch_size, max_length, num_classes)
        self.logits = tf.layers.dense(inputs=outputs, units=self.config.num_classes, name='logits')

        # 是否使用CRF层
        if self.config.crf:
            log_likelihood, self.transition_params = crf.crf_log_likelihood(inputs=self.logits,
                                                                            tag_indices=self.input_y,
                                                                            sequence_lengths=self.input_length)
            self.loss = -tf.reduce_mean(log_likelihood)
            # 结果输出
            self.predict, self.viterbi_score = crf.crf_decode(potentials=self.logits,
                                                              transition_params=self.transition_params,
                                                              sequence_length=self.input_length)
        else:
            # 损失函数，交叉熵
            cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=self.input_y, logits=self.logits)
            mask = tf.sequence_mask(lengths=self.input_length)
            losses = tf.boolean_mask(cross_entropy, mask=mask)
            self.loss = tf.reduce_mean(losses)
            # 结果输出
            self.predict = tf.argmax(tf.nn.softmax(self.logits), axis=1, name='predict')

        # 优化器
        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.config.learning_rate).minimize(loss=self.loss)


    def get_rnn(self, rnn_type):
        '''
        根据rnn_type创建RNN层
        :param rnn_type: RNN类型
        :return:
        '''
        if rnn_type == 'lstm':
            cell = rnn.LSTMCell(num_units=self.config.hidden_dim)
        else:
            cell = rnn.GRUCell(num_units=self.config.hidden_dim)
        cell = rnn.DropoutWrapper(cell=cell, input_keep_prob=self.input_keep_prob)
        return cell


if __name__ == '__main__':
    Model()