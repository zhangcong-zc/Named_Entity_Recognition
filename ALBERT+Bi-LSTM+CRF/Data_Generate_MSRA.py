# !/usr/bin/env python 
# -*- coding: UTF-8 -*- 
# @Time : 2020/12/5 16:00
# @Author : Zhang Cong

import logging
from tqdm import tqdm

logging.getLogger().setLevel(level=logging.INFO)
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s', level=logging.INFO)


def data_generate(root_path, output_file_path):
    '''
    遍历文件夹，生成数据格式（将原始数据中的/nt、/nr转换成/B-PER、/I-PER等形式，BIO标注）
    :param root_path: 原始数据路径
    :param output_file_path: 筛选并进行格式转换后的路径
    :return:
    '''
    logging.info('Start Generate Data ...')
    output_file = open(output_file_path, mode='w', encoding='UTF-8')
    # 遍历文件夹中的全部文件
    for line in tqdm(open(file=root_path, encoding='UTF-8')):
        line_list = line.split(' ')
        temp_list = []
        for item in line_list:
            item_list = item.split('/')
            if len(item_list)!=2: continue  # 如果长度部位2，则跳过
            word = item_list[0]     # 词汇
            pos = item_list[1]      # 词性o、ns、nt、nr
            if pos == 'o':      # 独立字符
                temp_list += [ it+'/O' for it in list(word)]
            elif pos == 'nt':   # 机构名
                word_list = list(word)
                for i in range(len(word_list)):
                    if i==0:
                        temp_list.append(word_list[i]+'/B-ORG')
                    else:
                        temp_list.append(word_list[i]+'/I-ORG')
            elif pos == 'ns':   # 地名
                word_list = list(word)
                for i in range(len(word_list)):
                    if i == 0:
                        temp_list.append(word_list[i] + '/B-LOC')
                    else:
                        temp_list.append(word_list[i] + '/I-LOC')
            elif pos == 'nr':   # 人名
                word_list = list(word)
                for i in range(len(word_list)):
                    if i == 0:
                        temp_list.append(word_list[i] + '/B-PER')
                    else:
                        temp_list.append(word_list[i] + '/I-PER')

        output_file.write(' '.join(temp_list) + '\n')

    output_file.close()
    logging.info('Generate Data Success ...')


if __name__ == "__main__":
    # 生成训练数据格式
    original_train_data_path = './data/MSRA/train.txt'
    output_train_data_path = './data/train_data.txt'
    data_generate(original_train_data_path, output_train_data_path)

    # 生成测试数据格式
    original_test_data_path = './data/MSRA/test.txt'
    output_test_data_path = './data/test_data.txt'
    data_generate(original_test_data_path, output_test_data_path)