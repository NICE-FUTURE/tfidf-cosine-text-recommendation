# -*- "coding: utf-8" -*-

import re
import jieba
from tfidf import TFIDF
from cosine import Cosine

def load_data():
    '''
    加载数据
    '''
    with open("news_sohusite_xml.smarty.dat", "r", encoding="utf-8") as f:
        doc = f.read()
    titles = list(map(lambda x:x[14:-15], re.findall(r"<contenttitle>.*</contenttitle>", doc)))
    contents = list(map(lambda x:x[9:-10], re.findall(r"<content>.*</content>", doc)))
    return titles, contents

def split_word(lines):
    '''
    分词
    '''
    with open("stopwords.txt", encoding="utf-8") as f:
        stopwords = f.read().split("\n")
    words_list = []
    for line in lines:
        words = [word for word in jieba.cut(line.strip().replace("\n", "").replace("\r", "").replace("\ue40c", "")) if word not in stopwords]
        words_list.append(" ".join(words))
    return words_list

if __name__ == "__main__":
    # 处理数据
    titles, contents = load_data()
    title_words = split_word(titles)
    # content_words = split_word(contents)

    # tf-idf向量化
    tfidf = TFIDF(title_words, max_words=300)
    # content_model = TFIDF(content_words, max_words=1000)
    title_array = tfidf.fit_transform()

    # 余弦相似度计算
    consine = Cosine(n_recommendation=3)
    indices, similarities = consine.cal_similarity(title_array)

    # 结果展示
    for i in range(3):
        title = titles[i]
        index = indices[i]
        similarity = similarities[i]
        print("与标题《{}》相似的标题:".format(title))
        for idx, sim in zip(index, similarity):
            print("\t\t《{}》:{:.5}".format(titles[idx], sim))
        print()
