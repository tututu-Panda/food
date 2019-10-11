from preprocessing import getDataFromCsv
from sklearn.model_selection import train_test_split
import jieba
from gensim.models.word2vec import Word2Vec


def getData():
    """
    划分数据集6/4
    """
    testSize = 0.4
    # 得到评论与标签信息
    label, comment = getDataFromCsv()
    # 划分训练集与测试集
    X_train, X_test, y_train, y_test = train_test_split(comment, label, test_size=testSize)

    print("总数据大小:", len(label))
    print("训练数据大小:", len(X_train))
    print("测试数据大小:", len(X_test))

    return X_train, X_test, y_train, y_test


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
        sentence_depart = jieba.lcut(sentence.strip())  # 分词
        out_list = []
        for word in sentence_depart:
            if word not in stopwords_list:              # 去除特殊符号
                if word != '\t':
                    out_list.append(word)
        result.append(out_list)
    return result



if __name__ == "__main__":
    # 1. 划分训练集与测试集
    X_train, X_test, y_train, y_test = getData()

    # 2. 将数据分词并去除停用词
    stopwords = stopwordslist("stopwords.txt")
    print(X_train[0:2])
    seg_data = seg_depart(X_train[0:2],stopwords)
    # seg_stop_data = move_stopwords(seg_data, stopwords)
    print("分词后和去除停用词后:", seg_data)

    # 3. 计算训练集和测试集每条评论数据的向量并存入文件

    # 4. 获取训练集和测试集的向量与标签

    # 5. 训练SVM

    # 6. 构建单个句子的向量

    # 7. 对句子进行预测
