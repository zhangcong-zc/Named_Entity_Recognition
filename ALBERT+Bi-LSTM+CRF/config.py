# !/usr/bin/env python 
# -*- coding: UTF-8 -*- 
# @Time: 2020/12/5 17:30
# @Author: Zhang Cong

# 模型配置参数
class Config():
    def __init__(self):
        self.original_data_path = './data/MSRA/data.txt'
        self.train_data_path = './data/train_data.txt'
        self.test_data_path = './data/test_data.txt'
        self.stopwords_path = './data/stopwords.txt'
        self.label_path = './data/label.txt'
        self.model_save_path = './save_model/'
        self.is_bilstm = True
        self.rnn_type = 'lstm'
        self.crf = True
        self.seq_length = 100
        self.num_classes = 7
        self.batch_size = 32
        self.keep_prob = 0.5
        self.epochs = 100
        self.vocab_size = 5000
        self.hidden_dim = 128
        self.embedding_dim = 300
        self.learning_rate = 1e-5
        self.train_test_split_value = 0.9


        # Albert配置参数
        self.is_training = False
        self.do_lower_case = True
        self.vocab_path = 'albert/albert_small_zh_google/vocab.txt'
        self.bert_config_file = 'albert/albert_small_zh_google/albert_config_small_google.json'
        self.initial_checkpoint = 'albert/albert_small_zh_google/albert_model.ckpt'

