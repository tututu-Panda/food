import pandas as pd
import numpy as np


"""
预处理文件, 将数据转换为文本集与结果集
"""
def getDataFromCsv():
    csv_data = pd.read_csv("./train.csv")
    csv_data = csv_data.sample(n=2000, random_state=234, axis=0)    # 随机采样
    # print(csv_data)
    negLabel = []
    negComment = []
    posLabel = []
    posComment = []
    for index, value in csv_data.iterrows():
        l = value.str.split('\t')
        if l[0][0] == '1':                          # 负样例
            negLabel.append(l[0][0])
            negComment.append(l[0][1].strip())
        else:
            posLabel.append(l[0][0])                # 正样例
            posComment.append(l[0][1].strip())
    return posComment, posLabel, negComment, negLabel
