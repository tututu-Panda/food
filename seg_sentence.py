from preprocessing import getDataFromCsv, getTestDataFromCsv
import jieba
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer


def stopwordslist(filepath):
    """
    加载停用词
    """
    stopwords = [line.strip() for line in open(filepath, 'r', encoding='utf-8').readlines()]
    return stopwords


def seg_depart(data):
    """
    对文档中的每一行进行中文分词并去除特殊符号
    """
    result = []
    for sentence in data:
        sentence_depart = list(jieba.cut(sentence.strip()))  # 分词
        out_list = []
        for word in sentence_depart:
            out_list.append(word)
        result.append(out_list)

    return result


def getTFIDF(result, stopwords_list):
    """
    得到输入list的TFIDF值
    """
    vectorizer = CountVectorizer(stop_words=stopwords_list)
    transformer = TfidfVectorizer()  # 该类会统计每个词语的tf-idf权值
    tfidf = transformer.fit_transform(result)  # 第一个fit_transform是计算tf-idf，第二个fit_transform是将文本转为词频矩阵
    X = vectorizer.fit_transform(result)
    word = vectorizer.get_feature_names()  # 获取词袋模型中的所有词语
    weight = tfidf.toarray()  # 将tf-idf矩阵抽取出来，元素a[i][j]表示j词在i类文本中的tf-idf权重
    words = {}
    for i in range(len(weight)):  # 打印每类文本的tf-idf词语权重，第一个for遍历所有文本，第二个for便利某一类文本下的词语权重
        for j in range(len(word)):
            if word[j] not in words:
                words.update({word[j]: weight[i][j]})
            else:
                words.update({word[j]: words.get(word[j]) + weight[i][j]})
    return words


def getRankValue(words):
    rankResult = {}
    name = sorted(words, reverse=True)
    value = sorted(words.values(), reverse=True)
    for index, v in enumerate(value):
        if v > 20: rankResult.update({name[index]: v})
    return rankResult

# stopwords = stopwordslist("stopwords.txt")
# #
# posComment, posLabel, negComment, negLabel = getDataFromCsv()
#
# result = seg_depart(negComment)
#
# result_words = getTFIDF(result, stopwords)
# b = sorted(result_words, reverse=True)
# c = sorted(result_words.values(), reverse=True)
# for index, name in enumerate(b):
#     print(name, ":", c[index])
#
# # rankValue = getRankValue(result_words)
#
# test = list()
# test.append("吃了一个发霉的鸭爪，鸭骨架没问题，味道很好")
# words = seg_depart(test)
# result_words = getTFIDF(words, stopwords)
# # rankValue = getRankValue(result_words)
# print(words)
# print(result_words)
