import pandas as pd
import numpy as np
import jieba
import fasttext

"""
预处理文件, 将数据转换为文本集与结果集
"""


def stopwordslist(filepath):
    """
    加载停用词
    """
    stopwords = [line.strip() for line in open(filepath, 'r', encoding='utf-8').readlines()]
    return stopwords


def addLabelToCsv(file, stopwords, outFile):
    """
    将文本处理为: 分词的comment \t __label__ 0/1
    """
    csv_data = pd.read_csv(file)
    # csv_data = csv_data.sample(n=4000, random_state=65535, axis=0)

    writeFile = open(outFile, mode='w')
    for index, value in csv_data.iterrows():
        l = value.str.split('\t')
        label = "__label__" + l[0][0]  # 类别
        comment = l[0][1]
        sentence_depart = list(jieba.cut(comment.strip()))  # 分词
        out_list = ""
        for word in sentence_depart:  # 去停用词
            if word not in stopwords:
                out_list += word + " "
        out_list += out_list + "\t" + label
        # print(out_list)
        writeFile.writelines(out_list + "\n")
    writeFile.close()


def segTestComment(file, stopwords, outFile):
    """
    对测试文件进行分词并写入文件
    """
    csv_data = pd.read_csv(file)
    writeFile = open(outFile, mode='w')
    for index, value in csv_data.iterrows():
        id = value['id']
        comment = value['comment']
        sentence_depart = list(jieba.cut(comment.strip()))  # 分词
        out_list = id + "\t"
        for word in sentence_depart:  # 去停用词
            if word not in stopwords:
                out_list += word + " "
        writeFile.writelines(out_list + "\n")
    writeFile.close()


def getTestDataFromCsv(file, stopwords, outFile):
    """
    获取训练数据
    :return:
    """
    csv_data = pd.read_csv(file)
    # csv_data = csv_data.sample(n=10, random_state=234, axis=0)  # 随机采样
    writeFile = open(outFile, mode='w')
    for index, value in csv_data.iterrows():
        comment = value['comment']
        sentence_depart = list(jieba.cut(comment.strip()))  # 分词
        out_list = ""
        for word in sentence_depart:  # 去停用词
            if word not in stopwords:
                out_list += word + " "
        label = "__label__"+str(value['label'])
        out_list += "\t" + label
        writeFile.writelines(out_list + "\n")
    writeFile.close()


def trainModel(trainFile, saveModelFile):
    """
    训练fasttext模型
    """
    # model = fasttext.train_supervised("train_label.txt", lr=0.1, dim=100, epoch=5, word_ngrams=2, loss='softmax')
    model = fasttext.train_supervised(trainFile, lr=0.1, dim=100, epoch=5, word_ngrams=2, loss='softmax')
    # model.save_model("model_file.bin")
    model.save_model(saveModelFile)


def testModel(testFile, modelFile):
    """
    对fasttext模型进行检验
    """
    classifier = fasttext.load_model(modelFile)
    result = classifier.test(testFile)
    print("准确率:", result)


def predictModel(predFile, modelFile, sampleFile):
    """
    预测模型的输出
    """
    classifier = fasttext.load_model(modelFile)
    predList = []
    columns = ["id", "label"]
    with open(predFile, "r") as f:
        for line in f:
            l = line.split("\n")
            id, comment = l[0].split("\t")
            a = classifier.predict(comment)
            label = a[0][0][-1]
            predList.append([id, label])
    dt = pd.DataFrame(predList, columns=columns)
    dt.to_csv(sampleFile, index=0)


# ids, comments = getTestDataFromCsv()
# addLabelToCsv("train.csv", stopwords, 'train_label.txt')
# segTestComment(testFile, stopwords, testOut)

trainInFile = "train/train_copy.csv"
trainLabel = "train/train_label.txt"
# trainLabel = "train/train_label_without_stopwords.txt"

testFile = "test/test_new.csv"
testOutFile = "test/testOut.txt"
# testOutFile = "test/testOut_without_stopwords.txt"

predFile = "test/testOut.txt"
predOutFile = "out.csv"

# saveModelFile = "model_file.bin"
saveModelFile = "model/0.9871_model_file.bin"


stopwords = stopwordslist("stopwords.txt")

trueFile = "true/test_ans.csv"
trueOutFile = "true/tureOut.txt"
# trueOutFile = "true/tureOut_without_stop.txt"

# getTestDataFromCsv(trueFile, stopwords, trueOutFile)
# segTestComment(testFile, stopwords, testOutFile)
# addLabelToCsv(trainInFile, stopwords, trainLabel)
segTestComment(testFile, stopwords, predFile)
trainModel(trainLabel, saveModelFile)
testModel(trueOutFile, saveModelFile)
predictModel(predFile, saveModelFile, predOutFile)


