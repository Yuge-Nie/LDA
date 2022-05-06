# coding=gbk
import jieba
import os
import re
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
output_path = 'C:/Users/NYG/PycharmProjects/LDA/result'
file_path = 'C:/Users/NYG/PycharmProjects/LDA/data'
stop_file = "C:/Users/NYG/PycharmProjects/LDA/stop_word/stopword.txt"

def preprocess(path):
    files = os.listdir(path)
    try:
        stopword_list = open(stop_file, encoding='utf-8')
    except:
        stopword_list = []
        print("error in stop_file")
    stop_list = []
    for line in stopword_list:
        line = re.sub(u'\n|\\r', '', line)
        stop_list.append(line)
    seg_list = []
    for file in files:
        position = path + '\\' + file
        with open(position, "r", encoding="ANSI") as f:
            data = re.sub(u"[^\u4e00-\u9fa5]", '', f.read())
        seg = " ".join(jieba.lcut(data, use_paddle=True, cut_all=False))
        seg_list.append(seg)
    vec = CountVectorizer(stop_words = stop_list)
    cnt = vec.fit_transform(seg_list)
    return cnt,vec

def word_seg(text):
    """对文本进行基于jieba的分词，返回以空格分隔单词的文本"""
    seg_list = [" ".join(jieba.lcut(e, use_paddle=True, cut_all=False)) for e in text]
    return seg_list

def cut_para(path, min_char):
    with open(path, "r", encoding="ANSI") as f:
        file = f.read()
        n = len(file)
        paras = []
        tmp_char = ''
        for i, char in enumerate(file):
            if char != '\u3000':
                tmp_char += char
            if char == '\n':
                tmp_char = tmp_char[:-1]
                if len(tmp_char) >= min_char:
                    paras.append(tmp_char[0:min_char])
                    tmp_char = ''
                else:
                    continue
            if (i + 1) == n:
                if len(tmp_char) >= min_char:
                    paras.append(tmp_char[0:min_char])
                break
    f.close()
    paras = paras[::int(len(paras)/15)][0:15]
    return paras

def docs_gen(path, char_len, voc):
    files = os.listdir(path)
    try:
        stopword_list = open(stop_file, encoding='utf-8')
    except:
        stopword_list = []
        print("error in stop_file")
    stop_list = []
    for line in stopword_list:
        line = re.sub(u'\n|\\r', '', line)
        stop_list.append(line)
    seg_list = []
    for file in files:
        position = path + '\\' + file
        seg_list.extend(word_seg(cut_para(position, char_len)))
    vec = CountVectorizer(token_pattern = r"(?u)\b\w\w+\b",
                          max_features = 5000,
                          stop_words = stop_list,
                          max_df = 0.5,
                          vocabulary = voc)
    cnt = vec.fit_transform(seg_list)
    return cnt,vec


def print_top_words(model, feature_names, n_top_words):
    tword = []
    for topic_idx, topic in enumerate(model.components_):
        print("Topic #%d:" % topic_idx)
        topic_w = " ".join([feature_names[i] for i in topic.argsort()[:-n_top_words - 1:-1]])
        tword.append(topic_w)
        print(topic_w)
    return tword


def print_res(model, res):
    max_idx = []
    for prob in res:
        prob = prob.tolist()
        max_idx.append(prob.index(max(prob)))
    for topic_idx, _ in enumerate(model.components_):
        para_list = []
        print("Topic #%d:" % topic_idx)
        for i, value in enumerate(max_idx):
            if value == topic_idx:
                para_list.append(i)
        print(para_list)


def print_res2(res1, res2):
    match = 0
    for j, p2 in enumerate(res2):
        dis = []
        for i, p1 in enumerate(res1):
            dis.append(np.sqrt(np.sum((p1 - p2) ** 2)))
        match_idx = dis.index(min(dis))
        print(j, '→', match_idx)
        if (int(j / 15) == match_idx):
            match += 1
    print('正确率: %f %%' % (match / len(res2) * 100))


n_topics = 15

if __name__ == "__main__":
    # cnt1,v1 = docs_gen('LDA/text', 'LDA/cn_stopwords.txt', 3000)
    cnt1, v1 = preprocess(file_path)
    lda = LatentDirichletAllocation(n_components=n_topics,
                                    random_state=0,
                                    max_iter=50)
    res1 = lda.fit_transform(cnt1)
    print_top_words(lda, v1.get_feature_names(), 8)
    print_res(lda, res1)
    cnt2, v2 = docs_gen(file_path, 2000, v1.vocabulary_)
    res2 = lda.transform(cnt2)
    print_res2(res1, res2)