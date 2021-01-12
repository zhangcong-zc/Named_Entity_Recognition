# !/usr/bin/env python 
# -*- coding: UTF-8 -*- 
# @Time: 2020/12/17 10:38
# @Author: Zhang Cong

import tensorflow as tf
from config import Config
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
        embedding_x = tf.expand_dims(embedding_x, axis=1)       # 扩充维度dim:(batch_size, 1, max_length, 300)  卷积操作后seq_length长度不变
        # 卷积层
        conv = tf.layers.conv2d(inputs=embedding_x,
                                filters=self.config.hidden_dim,
                                kernel_size=[1, self.config.kernel_size],
                                strides=1,
                                padding='SAME',
                                activation='relu',
                                use_bias=True,
                                kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                bias_initializer=tf.contrib.layers.xavier_initializer())

        final_output_layer_list = []    # 存储多个block的结果
        for block_i in range(self.config.block_num):
            for dilation in self.config.dilation_size:
                with tf.variable_scope(name_or_scope='atrous-conv-layer-%d' % dilation, reuse=tf.AUTO_REUSE):
                    # weight = tf.get_variable(shape=[self.config.kernel_size, self.config.embedding_dim, self.config.hidden_dim, self.config.hidden_dim],
                    #                          dtype=tf.float32,
                    #                          name='dilation-weight',
                    #                          initializer=tf.contrib.layers.xavier_initializer())
                    # bias = tf.get_variable(shape=[self.config.hidden_dim],
                    #                        dtype=tf.float32,
                    #                        name='dilation-bias',
                    #                        initializer=tf.contrib.layers.xavier_initializer())
                    # conv = tf.nn.atrous_conv2d(value=conv, filters=weight, rate=dilation, padding='SAME')
                    # conv = conv + bias
                    # conv = tf.nn.relu(conv)

                    # 与上面语句效果等价
                    conv = tf.layers.conv2d(inputs=conv,
                                            filters=self.config.hidden_dim,
                                            kernel_size=[self.config.kernel_size, self.config.embedding_dim],
                                            strides=1,
                                            dilation_rate=dilation,
                                            padding='SAME',
                                            activation='relu',
                                            use_bias=True,
                                            kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                            bias_initializer=tf.contrib.layers.xavier_initializer())
            # 存储当前block的输出
            final_output_layer_list.append(conv)
        # 将多个block的输出结果进行拼接
        final_output = tf.concat(final_output_layer_list, axis=-1)
        # drop out
        final_output = tf.nn.dropout(final_output, keep_prob=self.input_keep_prob)
        # 压缩降维，去除维度为1的项 dim:(batch_size, max_length, 3*hidden_dim)
        final_output = tf.squeeze(input=final_output, axis=1)
        # 输出层   dim:(batch_size, max_length, num_classes)
        self.logits = tf.layers.dense(inputs=final_output, units=self.config.num_classes, name='logits')

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


if __name__ == '__main__':
    Model()

