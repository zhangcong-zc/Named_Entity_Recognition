# !/usr/bin/env python 
# -*- coding: UTF-8 -*- 
# @Time: 2020/12/5 17:31
# @Author: Zhang Cong

import os
import jieba
import logging
import numpy as np
from tqdm import tqdm
import tensorflow as tf
from model import Model
from config import Config
from collections import Counter
from albert import utils
# from albert import tokenization

logging.getLogger().setLevel(level=logging.INFO)
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s', level=logging.INFO)

config = Config()
# GPU配置信息
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"                  # 按照PCI_BUS_ID顺序从0开始排列GPU设备
os.environ["CUDA_VISIBLE_DEVICES"] = "0"                        # 设置当前使用的GPU设备仅为0号设备
gpuConfig = tf.ConfigProto()
gpuConfig.allow_soft_placement = True                           #设置为True，当GPU不存在或者程序中出现GPU不能运行的代码时，自动切换到CPU运行
# gpuConfig.gpu_options.allow_growth = True                       #设置为True，程序运行时，会根据程序所需GPU显存情况，分配最小的资源
gpuConfig.gpu_options.per_process_gpu_memory_fraction = 0.8     #程序运行的时，所需的GPU显存资源最大不允许超过rate的设定值

# 模型训练
class Train():
    def __init__(self):
        # 实例化模型结构
        self.model = Model()
        self.sess = tf.Session(config=gpuConfig)
        self.sess.run(tf.global_variables_initializer())

    def train(self):
        train_sentences, train_labels = load_dataset(config.train_data_path)      # 加载训练数据集
        test_sentences, test_labels = load_dataset(config.test_data_path)         # 加载验证数据集
        # 构建词汇映射表
        if not os.path.exists(config.vocab_path):
            build_vocab(train_sentences, config.vocab_path)
        word_to_id = read_vocab(config.vocab_path)      # 读取词汇表及其映射关系
        # 构建类别映射表
        if not os.path.exists(config.label_path):
            build_label(train_labels, config.label_path)
        label_to_id = read_label(config.label_path)     # 读取类别表及其映射关系

        # 构建训练数据集
        train_input_ids, train_input_masks, train_segment_ids, train_label_ids, train_seq_length = data_transform(train_sentences,
                                                                                                                  train_labels,
                                                                                                                  word_to_id,
                                                                                                                  label_to_id,
                                                                                                                  'train')
        test_input_ids, test_input_masks, test_segment_ids, test_label_ids, test_seq_length = data_transform(test_sentences,
                                                                                                             test_labels,
                                                                                                             word_to_id,
                                                                                                             label_to_id,
                                                                                                             'test')
        # 打印训练、测试数据量，数据与标签量是否相等
        logging.info('Train Data: {}'.format(np.array(train_input_ids).shape))
        logging.info('Train Label: {}'.format(np.array(train_label_ids).shape))
        logging.info('Test Data: {}'.format(np.array(test_input_ids).shape))
        logging.info('Test Label: {}'.format(np.array(test_label_ids).shape))

        # 数据集校验，数据和标签数量是否有误
        if (len(train_input_ids) != len(train_label_ids)) or (len(test_input_ids) != len(test_label_ids)):
            logging.info('Data number != Label number')
            exit(0)

        # 配置Saver
        saver = tf.train.Saver()
        if not os.path.exists(config.model_save_path):      # 如不存在相应文件夹，则创建
            os.mkdir(config.model_save_path)

        # 模型训练
        best_f1_score = 0  # 初始best模型的F1值
        for epoch in range(1, config.epochs + 1):
            train_loss_list = []            # 存储每个epoch的loss
            train_precision_list = []       # 存储每个epoch的precision
            train_recall_list = []          # 存储每个epoch的recall
            train_f1_list = []              # 存储每个epoch的f1值
            # 将训练数据进行 batch_size 切分
            batch_train_input_ids, batch_train_input_masks, batch_train_segment_ids, batch_train_label_ids, batch_train_seq_length = \
                creat_batch_data(train_input_ids,
                                 train_input_masks,
                                 train_segment_ids,
                                 train_label_ids,
                                 train_seq_length,
                                 config.batch_size)
            for step, (batch_input_ids, batch_input_masks, batch_segment_ids, batch_label_ids, batch_seq_length) in tqdm(enumerate(zip(batch_train_input_ids,
                                                                                                                         batch_train_input_masks,
                                                                                                                         batch_train_segment_ids,
                                                                                                                         batch_train_label_ids,
                                                                                                                         batch_train_seq_length))):
                feed_dict = {self.model.input_ids: batch_input_ids,
                             self.model.input_masks: batch_input_masks,
                             self.model.segment_ids: batch_segment_ids,
                             self.model.label_ids: batch_label_ids,
                             self.model.input_length: batch_seq_length,
                             self.model.input_keep_prob: config.keep_prob}
                train_predict, train_loss, _ = self.sess.run([self.model.predict, self.model.loss, self.model.optimizer], feed_dict=feed_dict)
                train_loss_list.append(train_loss)
                # 计算准确率
                for predict, input_x, input_y, seq_length in zip(train_predict, batch_input_ids, batch_label_ids, batch_seq_length):
                    train_entity_y = get_all_entity(input_y, input_x, seq_length, word_to_id, label_to_id)
                    train_entity_predict = get_all_entity(predict, input_x, seq_length, word_to_id, label_to_id)
                    train_intersection = [item for item in train_entity_y if item in train_entity_predict]      # 实体交集
                    if len(train_intersection)!=0:
                        train_precision = len(train_intersection)/len(train_entity_predict)
                        train_recall = len(train_intersection)/len(train_entity_y)
                        train_f1 = (2*train_precision*train_recall)/(train_precision+train_recall)

                        train_precision_list.append(train_precision)
                        train_recall_list.append(train_recall)
                        train_f1_list.append(train_f1)

            # 完成一个epoch的训练，输出训练数据的mean accuracy、mean loss
            logging.info('Train Epoch: %d , Loss: %.6f , Precision: %.6f , Recall: %.6f , F1: %.6f' % (epoch,
                                                                                                       float(np.mean(np.array(train_loss_list))),
                                                                                                       float(np.mean(np.array(train_precision_list))),
                                                                                                       float(np.mean(np.array(train_recall_list))),
                                                                                                       float(np.mean(np.array(train_f1_list)))))
            # 模型验证
            test_loss_list = []        # 存储每个epoch的loss
            test_precision_list = []   # 存储每个epoch的precision
            test_recall_list = []      # 存储每个epoch的recall
            test_f1_list = []          # 存储每个epoch的f1值
            batch_test_input_ids, batch_test_input_masks, batch_test_segment_ids, batch_test_label_ids, batch_test_seq_length = \
                creat_batch_data(test_input_ids,
                                 test_input_masks,
                                 test_segment_ids,
                                 test_label_ids,
                                 test_seq_length,
                                 config.batch_size)
            for (batch_input_ids, batch_input_masks, batch_segment_ids, batch_label_ids, batch_seq_length) in tqdm(zip(batch_test_input_ids,
                                                                                                                     batch_test_input_masks,
                                                                                                                     batch_test_segment_ids,
                                                                                                                     batch_test_label_ids,
                                                                                                                     batch_test_seq_length)):
                feed_dict = {self.model.input_ids: batch_input_ids,
                             self.model.input_masks: batch_input_masks,
                             self.model.segment_ids: batch_segment_ids,
                             self.model.label_ids: batch_label_ids,
                             self.model.input_length: batch_seq_length,
                             self.model.input_keep_prob: 1.0}
                test_predict, test_loss = self.sess.run([self.model.predict, self.model.loss], feed_dict=feed_dict)
                test_loss_list.append(test_loss)
                # 计算准确率
                for predict, input_x, input_y, seq_length in zip(test_predict, batch_input_ids, batch_label_ids, batch_seq_length):
                    test_entity_y = get_all_entity(input_y, input_x, seq_length, word_to_id, label_to_id)
                    test_entity_predict = get_all_entity(predict, input_x, seq_length, word_to_id, label_to_id)
                    test_intersection = [item for item in test_entity_y if item in test_entity_predict]     # 实体交集
                    if len(test_intersection)!=0:
                        test_precision = len(test_intersection) / len(test_entity_predict)
                        test_recall = len(test_intersection) / len(test_entity_y)
                        test_f1 = (2 * test_precision * test_recall) / (test_precision + test_recall)

                        test_precision_list.append(test_precision)
                        test_recall_list.append(test_recall)
                        test_f1_list.append(test_f1)

            # 完成一个epoch的训练，输出训练数据的mean accuracy、mean loss
            f1_score = float(np.mean(np.array(test_f1_list)))
            logging.info('Test Epoch: %d , Loss: %.6f , Precision: %.6f , Recall: %.6f , F1: %.6f' % (epoch,
                                                                                                       float(np.mean(np.array(test_loss_list))),
                                                                                                       float(np.mean(np.array(test_precision_list))),
                                                                                                       float(np.mean(np.array(test_recall_list))),
                                                                                                       f1_score))
            # 当前epoch产生的模型F1值超过最好指标时，保存当前模型
            if best_f1_score < f1_score:
                best_f1_score = f1_score
                saver.save(sess=self.sess, save_path=config.model_save_path)
                logging.info('Save Model Success ...')


# 模型预测
class Predict():
    def __init__(self):
        # 实例化并加载模型
        self.model = Model()
        self.sess = tf.Session(config=gpuConfig)
        self.saver = tf.train.Saver()
        self.saver.restore(sess=self.sess, save_path=config.model_save_path)

        # 加载词汇->ID映射表
        self.word_to_id = read_vocab(config.vocab_path)
        # 加载label->ID映射表
        self.label_to_id = read_label(config.label_path)
        # 加载停用词
        # self.stopwords = [word.replace('\n', '').strip() for word in open(config.stopwords_path, encoding='UTF-8')]


    def pre_process(self, sentence):
        '''
        文本数据预处理
        :param sentence: 输入的文本句子
        :return:
        '''
        # 分词，去除停用词
        sentence_seg = list(sentence)
        # 将词汇映射为ID
        sentence_id = []
        for word in sentence_seg:
            if word in self.word_to_id:
                sentence_id.append(self.word_to_id[word])
            else:
                sentence_id.append(self.word_to_id['<UNK>'])
        # 对文本长度进行padding填充
        sentence_length = len(sentence_id)
        if sentence_length > config.seq_length:
            sentence_length = config.seq_length
            sentence_id = sentence_id[: config.seq_length]
        else:
            sentence_id.extend([self.word_to_id['<PAD>']] * (config.seq_length - sentence_length))

        return sentence_id, sentence_length


    def predict(self, sentence):
        '''
        结果预测
        :param sentence:
        :return:
        '''
        # 对句子预处理并进行ID表示
        sentence_id, sentence_length = self.pre_process(sentence)
        feed_dict = {self.model.input_x: [sentence_id],
                     self.model.input_length: [sentence_length],
                     self.model.input_keep_prob: 1.0}
        tag_predict = self.sess.run(self.model.predict, feed_dict=feed_dict)[0]

        word_list = list(sentence)
        input_y = tag_predict[: sentence_length]

        id_to_label = dict(zip(self.label_to_id.values(), self.label_to_id.keys()))
        label_list = [id_to_label[index] for index in input_y]

        PER = get_PER_entity(word_list, label_list)
        LOC = get_LOC_entity(word_list, label_list)
        ORG = get_ORG_entity(word_list, label_list)

        return 'PER: {}\nLOC: {}\nORG: {}'.format(PER, LOC, ORG)


def text_processing(text):
    '''
    文本数据预处理，分词，去除停用词
    :param text: 文本数据sentence
    :return: 以空格为分隔符进行分词/分字结果
    '''
    # 删除（）里的内容
    # text = re.sub('（[^（.]*）', '', text)
    # 只保留中文部分
    text = ''.join([x for x in text if '\u4e00' <= x <= '\u9fa5'])
    # 利用jieba进行分词
    words = list(jieba.cut(text))
    # 不分词
    # words = [x for x in ''.join(text)]
    return ' '.join(words)


def pre_process(data_path, preprocess_path):
    '''
    原始数据预处理
    :param data_path: 原始文本文件路径
    :param preprocess_path: 预处理后的数据存储路径
    :return:
    '''
    # 加载停用词表
    logging.info('Start Preprocess ...')
    preprocess_file = open(preprocess_path, mode='w', encoding='UTF-8')
    # 加载停用词表
    stopwords = [word.replace('\n', '').strip() for word in open(config.stopwords_path, encoding='UTF-8')]
    for line in tqdm(open(data_path, encoding='UTF-8')):
        label_sentence = str(line).strip().replace('\n', '').split('\t')    # 去除收尾空格、结尾换行符\n、使用\t切分
        label = label_sentence[0]
        sentence = label_sentence[1].replace('\t', '').replace('\n', '').replace(' ', '')   # 符号过滤
        sentence = [word for word in text_processing(sentence).split(' ') if word not in stopwords and not word.isdigit()]
        preprocess_file.write(label + '\t' + ' '.join(sentence) + '\n')

    preprocess_file.close()


def load_dataset(data_path):
    '''
    从本地磁盘加载经过预处理的数据集，避免每次都进行预处理操作
    :param data_path: 预处理好的数据集路径
    :return: 句子列表，标签列表
    '''
    sentences = []
    labels = []
    # 加载停用词表
    logging.info('Load Dataset from {}'.format(data_path))
    for line in tqdm(open(data_path, encoding='UTF-8')):
        word_temp = []
        label_temp = []
        for item in line.split(' '):
            item_list = item.replace('\n', '').split('/')
            word_temp.append(item_list[0])
            label_temp.append(item_list[1])

        if len(word_temp) == len(label_temp):
            sentences.append(word_temp)
            labels.append(label_temp)
        else:
            logging.info('Load Data Error ... msg: {}'.format(line))    # 部分数据去除英文和数字后为空，跳过异常
            continue

    return sentences, labels


def build_vocab(input_data, vocab_path):
    '''
    根据数据集构建词汇表，存储到本地备用
    :param input_data: 全部句子集合 [n] n为数据条数
    :param vocab_path: 词表文件存储路径
    :return:
    '''
    logging.info('Build Vocab ...')
    all_data = []       # 全部句子集合
    for content in input_data:
        all_data.extend(content)

    counter = Counter(all_data)     # 词频统计
    count_pairs = counter.most_common(config.vocab_size - 2)    # 对词汇按次数进行降序排序
    words, _ = list(zip(*count_pairs))              # 将(word, count)元祖形式解压，转换为列表list
    # 添加一个 <PAD> 来将所有文本pad为同一长度
    words = ['<UNK>'] + list(words)  # 增加一个OOV标识的编码
    words = ['<PAD>'] + list(words)  # 增加一个PAD标识的编码
    open(vocab_path, mode='w', encoding='UTF-8').write('\n'.join(words) + '\n')


def build_label(input_label, label_path):
    '''
    根据标签集构建标签表，存储到本地备用
    :param input_label: 全部标签集合
    :param label_path: 标签文件存储路径
    :return:
    '''
    logging.info('Build Label ...')
    all_label = set(input_label)
    open(label_path, mode='w', encoding='UTF-8').write('\n'.join(all_label))


def read_vocab(vocab_path):
    """
    读取词汇表，构建 词汇-->ID 映射字典
    :param vocab_path: 词表文件路径
    :return: 词表，word_to_id
    """
    words = [word.replace('\n', '').strip() for word in open(vocab_path, encoding='UTF-8')]
    word_to_id = dict(zip(words, range(len(words))))

    return word_to_id


def read_label(label_path):
    '''
    读取类别表，构建 类别-->ID 映射字典
    :param label_path: 类别文件路径
    :return: 类别表，label_to_id
    '''
    labels = [label.replace('\n', '').strip() for label in open(label_path, encoding='UTF-8')]
    label_to_id = dict(zip(labels, range(len(labels))))

    return label_to_id


def data_transform(input_data, input_label, word_to_id, label_to_id, set_type):
    '''
    数据预处理，将文本和标签映射为ID形式
    :param input_data: 文本数据集合
    :param input_label: 标签集合
    :param word_to_id: 词汇——ID映射表
    :param label_to_id: 标签——ID映射表
    :return: ID形式的文本，ID形式的标签
    '''
    logging.info('Convert {} data to Albert feature ...'.format(set_type))
    input_ids = []
    input_masks = []
    segment_ids = []
    sequence_length = []
    for index, sentence in tqdm(enumerate(input_data)):
        sentence = ''.join(sentence)
        if len(sentence) > config.seq_length:
            sequence_length.append(config.seq_length)
        else:
            sequence_length.append(len(sentence))
        # 获取Albert的特征
        feature = utils.get_single_feature(index, sentence, set_type)
        input_ids.append(feature.input_ids)
        input_masks.append(feature.input_mask)
        segment_ids.append(feature.segment_ids)
    logging.info('Get {} features finished ...'.format(set_type))

    # 将标签转换为ID形式
    logging.info('Label Trans To One-Hot ...')
    label_id = []
    for label in tqdm(input_label):
        tag_temp = []
        for tag in label:
            tag_temp.append(label_to_id[tag])
        # 对label长度进行padding填充
        label_length = len(tag_temp)
        if label_length > config.seq_length:
            tag_temp = tag_temp[: config.seq_length]
        else:
            tag_temp.extend([label_to_id['O']] * (config.seq_length - label_length))
        label_id.append(tag_temp)

    # shuffle
    indices = np.random.permutation(np.arange(len(input_ids)))
    input_ids = np.array(input_ids)[indices]
    input_masks = np.array(input_masks)[indices]
    segment_ids = np.array(segment_ids)[indices]
    label_ids = np.array(label_id)[indices]
    seq_length = np.array(sequence_length)[indices]

    return input_ids, input_masks, segment_ids, label_ids, seq_length


def creat_batch_data(input_ids, input_masks, segment_ids, label_ids, seq_length, batch_size):
    '''
    将数据集以batch_size大小进行切分
    :param input_data: 数据列表
    :param input_label: 标签列表
    :param batch_size: 批大小
    :return:
    '''
    max_length = len(input_ids)            # 数据量
    max_index = max_length // batch_size    # 最大批次
    # shuffle
    indices = np.random.permutation(np.arange(max_length))
    input_ids_shuffle = np.array(input_ids)[indices]
    input_masks_shuffle = np.array(input_masks)[indices]
    segment_ids_shuffle = np.array(segment_ids)[indices]
    label_ids_shuffle = np.array(label_ids)[indices]
    seq_length_shuffle = np.array(seq_length)[indices]

    batch_input_ids, batch_input_masks, batch_segment_ids, batch_label_ids, batch_seq_length = [], [], [], [], []
    for index in range(max_index):
        start = index * batch_size                              # 起始索引
        end = min((index + 1) * batch_size, max_length)         # 结束索引，可能为start + batch_size 或max_length
        batch_input_ids.append(input_ids_shuffle[start: end])
        batch_input_masks.append(input_masks_shuffle[start: end])
        batch_segment_ids.append(segment_ids_shuffle[start: end])
        batch_label_ids.append(label_ids_shuffle[start: end])
        batch_seq_length.append(seq_length_shuffle[start: end])

        if (index + 1) * batch_size > max_length:               # 如果结束索引超过了数据量，则结束
            break

    return batch_input_ids, batch_input_masks, batch_segment_ids, batch_label_ids, batch_seq_length


def get_PER_entity(char_seq, tag_seq):
    '''
    获取人名实体
    :param char_seq: 字符序列
    :param tag_seq: 标记序列
    :return:
    '''
    PER = []
    temp = ''
    for char, tag in zip(char_seq, tag_seq):
        if tag == 'B-PER':
            temp = char
        elif tag == 'I-PER' and len(temp) > 0:
            temp += char
        else:
            if len(temp) > 0:
                PER.append(temp)
            temp = ''
    return PER


def get_LOC_entity(char_seq, tag_seq):
    '''
    获取地址实体
    :param char_seq: 字符序列
    :param tag_seq: 标记序列
    :return:
    '''
    LOC = []
    temp = ''
    for char, tag in zip(char_seq, tag_seq):
        if tag == 'B-LOC':
            temp = char
        elif tag == 'I-LOC' and len(temp) > 0:
            temp += char
        else:
            if len(temp) > 0:
                LOC.append(temp)
            temp = ''
    return LOC


def get_ORG_entity(char_seq, tag_seq):
    '''
    获取机构实体
    :param char_seq: 字符序列
    :param tag_seq: 标记序列
    :return:
    '''
    ORG = []
    temp = ''
    for char, tag in zip(char_seq, tag_seq):
        if tag == 'B-ORG':
            temp = char
        elif tag == 'I-ORG' and len(temp) > 0:
            temp += char
        else:
            if len(temp) > 0:
                ORG.append(temp)
            temp = ''
    return ORG


def get_all_entity(input_y, input_x, seq_length, word_to_id, label_to_id):
    '''
    获取全部实体
    :param input_y: 真实/预测出的ID形式label
    :param input_x: 真实ID形式sentence
    :param seq_length: sentence真实长度
    :param word_to_id: word->ID的映射
    :param label_to_id:label->ID的映射
    :return: entity实体集合
    '''
    input_x = input_x[: seq_length]     # 按照真实长度截取sequence
    input_y = input_y[: seq_length]     # 按照真实长度截取tag

    id_to_word = dict(zip(word_to_id.values(), word_to_id.keys()))
    id_to_label = dict(zip(label_to_id.values(), label_to_id.keys()))

    word_list = [id_to_word[index] for index in input_x]        # ID-> char
    label_list = [id_to_label[index] for index in input_y]      # ID-> label

    PER = get_PER_entity(word_list, label_list)     # 获取人名
    LOC = get_LOC_entity(word_list, label_list)     # 获取地址
    ORG = get_ORG_entity(word_list, label_list)     # 获取机构

    return PER + LOC + ORG


if __name__ == "__main__":
    # 训练
    Train().train()

    # 预测
    # predictor = Predict()
    # while True:
    #     sentence = input('输入：')
    #     result = predictor.predict(sentence)
    #     print(result)