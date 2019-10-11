import jieba


def stopwordslist(filepath):
    """
    加载停用词
    """
    stopwords = [line.strip() for line in open(filepath, 'r', encoding='utf-8').readlines()]
    return stopwords


def seg_depart(data, stopwords_list):
    """
    对文档中的每一行进行中文分词并去除特殊符号
    """
    result = []
    for sentence in data:
        sentence_depart = list(jieba.cut(sentence.strip()))  # 分词
        out_list = []
        for word in sentence_depart:
            if word not in stopwords_list:              # 去除特殊符号
                if word != '\t':
                    out_list.append(word)
        result.append(out_list)
    return result
