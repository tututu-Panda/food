from preprocessing import getDataFromCsv, getTestDataFromCsv
from seg_sentence import stopwordslist, seg_depart

from sklearn.model_selection import train_test_split
import numpy as np
from gensim.models.word2vec import Word2Vec
from sklearn.externals import joblib
from sklearn.svm import SVC


def getData():
    """
    划分数据集
    """
    testSize = 0.2
    # 得到评论与标签信息
    posComment, posLabel, negComment, negLabel = getDataFromCsv()

    # 分词处理
    segPosComment = seg_depart(posComment)
    segNegComment = seg_depart(negComment)

    y = np.concatenate((np.array(posLabel), np.array(negLabel)))
    # print(y.shape)

    # 划分训练集与测试集
    X_train, X_test, y_train, y_test = train_test_split(np.concatenate((segPosComment, segNegComment)), y, test_size=testSize)

    print("总数据大小:", len(y))
    print("训练数据大小:", len(X_train))
    print("测试数据大小:", len(X_test))

    np.save('pre_data/y_train.npy', y_train)
    np.save('pre_data/y_test.npy', y_test)

    return X_train, X_test



def build_sentence_vec(sentence, size, w2v_model):
    """
    对每个句子的所有词向量取平均值，生成一个句子的vector
    """
    # vec = np.zeros(size).reshape((1, size))
    vec = np.zeros(size)
    count = 0
    for word in sentence:
        try:
            vec += w2v_model[word]
            count += 1
        except KeyError:
            continue
    if count != 0:
        vec /= count
    return vec


def get_train_vec(x_train, x_test):
    """
    计算词向量
    """
    n_dim = 200
    w2v_model = Word2Vec(size=n_dim, window=5, sg=0, hs=0, negative=5, min_count=5)    # 建立词向量模型
    w2v_model.build_vocab(x_train)                                                      # 准备模型词汇表

    w2v_model.train(x_train, total_examples=w2v_model.corpus_count, epochs=w2v_model.iter)  # 训练训练集的词向量
    train_vecs = np.concatenate([build_sentence_vec(z, n_dim, w2v_model) for z in x_train])  # 训练集评论向量集合
    np.save('pre_data/train_vecs.npy', train_vecs)

    w2v_model.train(x_test, total_examples=w2v_model.corpus_count, epochs=w2v_model.iter)
    w2v_model.save('pre_data/w2v_model/w2v_model.pkl')
    test_vecs = np.concatenate([build_sentence_vec(z, n_dim, w2v_model) for z in x_test])
    np.save('pre_data/test_vecs.npy', test_vecs)


def get_data():
    """
    获得训练集和测试集的向量以及标签
    """
    train_vecs = np.load("pre_data/train_vecs.npy")
    train_label = np.load("pre_data/y_train.npy")
    test_vecs = np.load("pre_data/test_vecs.npy")
    test_label = np.load("pre_data/y_test.npy")

    return train_vecs, train_label, test_vecs, test_label


def svm_train(train_vecs, train_label, test_vecs, test_label):
    """
    进行svm模型训练
    """
    clf = SVC(verbose=True)               # 定义一个以rbf为内核的SVM
    clf.fit(train_vecs, train_label)                    # 训练模型
    joblib.dump(clf, 'pre_data/svm_model/model.pkl')    # 保存训练好的模型
    print(clf.score(test_vecs, test_label))                 # 输出测试数据的平均准确度
    test = clf.predict(test_vecs)
    print(test.shape)

def predict(ids, comments, stopwords, test_vecs, test_label):
    """
    应用模型进行预测
    """
    n_dim = 200
    w2v_model = Word2Vec.load('pre_data/w2v_model/w2v_model.pkl')
    clf = joblib.load('pre_data/svm_model/model.pkl')
    segComment = seg_depart(comments)

    for comment in segComment:
        print(comment)
        word_vecs = build_sentence_vec(comment, n_dim, w2v_model)
        print(clf.predict(word_vecs))
    #     print(clf.predict(word_vecs))

    print("xdasd", test_vecs)


    return train_vecs


if __name__ == "__main__":
    # 1. 划分训练集与测试集

    # 2. 将数据分词并去除停用词

    # 3. 计算训练集和测试集每条评论数据的向量并存入文件

    # 4. 获取训练集和测试集的向量与标签

    # 5. 训练SVM

    # 6. 构建单个句子的向量

    # # 7. 对句子进行预测

    stopwords = stopwordslist("stopwords.txt")

    x_train, x_test = getData()

    get_train_vec(x_train, x_test)

    train_vecs, train_label, test_vecs, test_label = get_data()


    svm_train(train_vecs, train_label, test_vecs, test_label)

    ids, comments = getTestDataFromCsv()

    predict(ids, comments, stopwords, test_vecs, test_label)


    # n_dim = 200
    # posComment, posLabel, negComment, negLabel = getDataFromCsv()
    # test = posComment + negComment
    # r = seg_depart(test)
    # w2v_model = Word2Vec(size=n_dim, window=5, sg=0, hs=0, negative=5, min_count=5)  # 建立词向量模型
    # w2v_model.build_vocab(r)  # 准备模型词汇表
    # w2v_model.train(r, total_examples=w2v_model.corpus_count, epochs=w2v_model.iter)  # 训练训练集的词向量
    # w2v_model.save('pre_data/w2v_model/w2v_model.pkl')
    # w2v_model = Word2Vec.load('pre_data/w2v_model/w2v_model.pkl')
    #
    # print('-----------------分割线---------------------------')
    # print('与虫最相近的3个字的词')
    # req_count = 5
    # for key in w2v_model.similar_by_word('钢丝', topn=100):
    #     if len(key[0]) == 3:
    #         req_count -= 1
    #         print(key[0], key[1])
    #         if req_count == 0:
    #             break
