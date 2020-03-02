# -*- "coding: utf-8" -*-

import math
import numpy
import sys
from collections import Counter

class TFIDF(object):

    def __init__(self, corpus, max_words):
        '''
        Args:
            corpus: list  用于计算idf值的语料库 ["word1 word2 ...", ...]
            max_words: int  仅保留整个语料库中词频最高的max_words个词语
        '''
        self.corpus = corpus
        self.max_words = max_words

    def _computeTF(self, sen_word_dic):
        '''
        计算一篇文章中每个词的词频，即TF值
        Args:
            sen_word_dic: dict  字典为一篇文章的词频统计结果 {word1:cnt1, word2:cnt2, ...}
        Returns:
            tf_dic: dict  字典为一篇文章词语tf值的计算结果 {word1:tf1, word2:tf2, ...}
        '''
        total = sum(sen_word_dic.values())  # 一篇文章的词语总数
        tf_dic = {item[0]:(lambda item: item[1]/total)(item) for item in sen_word_dic.items()}  # 计算一篇文章中的词频
        return tf_dic

    def _computeIDF(self, sen_word_dics):
        '''
        计算每个词的逆文档频率
        Args:
            sen_word_dics: list  列表中一个元素为一篇文章词语的统计结果 [{word1:cnt1, word2:cnt2, ...}, ...]
        Returns:
            idf_dic: dict  字典为整个语料库idf值的计算结果 {word1:idf1, word2:idf2, ...}
        '''
        total = len(sen_word_dics)
        idf_dic = {}
        for sen_word_dic in sen_word_dics:
            for word, value in sen_word_dic.items():
                if value > 0:
                    idf_dic[word] = idf_dic.setdefault(word, 0)+1
        idf_dic = {item[0]:(lambda item: math.log(total/item[1]+1))(item) for item in idf_dic.items()}  # 计算一篇文章中的词频
        return idf_dic

    def fit(self):
        '''
        基于语料库计算idf值，根据max_words对最终使用的词语进行截取
        '''
        self.sen_word_dics = [Counter(line.split()) for line in self.corpus]
        self.idf_dic = self._computeIDF(self.sen_word_dics)  # 计算idf值

        corpus_word_dic = Counter((" ".join(self.corpus)).split())
        corpus_word_tuples = sorted(corpus_word_dic.items(), key=lambda x:x[1], reverse=True)  # 将词语按整个预料库中的词频排序
        self.all_features = corpus_word_tuples  # 据此调整停用词

        corpus_word_tuples = corpus_word_tuples[:self.max_words]  # 仅使用前max_words个词语
        self.cur_features = corpus_word_tuples

    def _computeTFIDF(self, sen_word_dics):
        '''
        计算 TF-IDF 值
        Args:
            doc: list  用于计算tf值的文本列表 ["word1 word2 ...", ...]
        Returns:
            tfidf_dics: list  每个元素是一篇文章中每个词语对应的tfidf值 [{word1:tfidf1, word2:tfidf2, ...}, ...]
        '''
        tfidf_dics = []
        for sen_word_dic in sen_word_dics:
            tf_dic = self._computeTF(sen_word_dic)  # 计算tf值
            try:
                tfidf_dic = {item[0]:(lambda item: item[1]*self.idf_dic[item[0]])(item) for item in tf_dic.items()}  # 计算一篇文章中每个词的tfidf值
            except AttributeError as e:
                print(e, file=sys.stderr)
                print("Maybe you should: call TFIDF.fit() function first.", file=sys.stderr)
                exit(1)
            tfidf_dics.append(tfidf_dic)
        return tfidf_dics

    def _transform(self, sen_word_dics):
        '''
        将ifidf值对应到稀疏矩阵上
        Returns:
            X: ndarray  行为文章，列为词语，值为该文章该词语的tfidf值，默认值为0
        '''
        tfidf_dics = self._computeTFIDF(sen_word_dics)
        index_dic = {idx:item[0] for idx, item in enumerate(self.cur_features)}  # 将每个词对应到一个索引上
        X = numpy.zeros((len(tfidf_dics), len(index_dic)))
        for i, tfidf_dic in enumerate(tfidf_dics):
            for j, word in index_dic.items():
                X[i, j] = tfidf_dic.setdefault(word, 0)
        return X
        
    def transform(self, doc):
        doc_word_dics = [Counter(line.split()) for line in doc]
        return self._transform(doc_word_dics)

    def fit_transform(self):
        self.fit()
        return self._transform(self.sen_word_dics)


if __name__ == '__main__':

    corpus = ["我 爱 我 家", "我 爱 亲爱的 故乡", "重整 行囊", "勇 攀 高峰"]
    tfidf = TFIDF(corpus, 5)
    X = tfidf.fit_transform()
    # tfidf.fit()
    # X = tfidf.transform(corpus)
    print(tfidf.all_features)
    print(tfidf.cur_features)
    print(X)
