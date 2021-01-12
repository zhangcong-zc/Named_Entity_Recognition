# !/usr/bin/python
# -*- coding:utf-8 -*-

import tensorflow as tf

def get_ORG_entity(char_seq, tag_seq):
    ORG = []
    temp = ''
    for char, tag in zip(char_seq, tag_seq):
        if tag=='B-LOC':
            temp = char
        elif tag=='I-LOC' and len(temp)>0:
            temp += char
        else:
            if len(temp)>0:
                ORG.append(temp)
            temp = ''

    return ORG


a = ['金', '奖', '作', '品', '“', '莲', '中', '珠', '宝', '”', '（', '见', '图', '）', '选', '择', '最', '具', '西', '藏', '特', '色', '的', '藏', '传', '佛', '教', '文', '化', '作', '为', '切', '入', '点', '，', '神', '秘', '经', '文', '的', '图', '形', '、', '寓', '意', '吉', '祥', '的', '鸟', '兽', '纹', '样', '在', '服', '装', '中', '时', '隐', '时', '现', '。']
b = ['O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'B-LOC', 'I-LOC', 'O', 'O', 'O', 'B-LOC', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O']

res = get_ORG_entity(a, b)
print(res)


c = [1,2,3,4,5]
print(c[-4: ])

print(tf.test.gpu_device_name())