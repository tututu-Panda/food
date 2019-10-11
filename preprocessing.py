import pandas as pd
import numpy as np


"""
预处理文件, 将数据转换为文本集与结果集
"""
def getDataFromCsv():
    csv_data = pd.read_csv("./train.csv")
    # print(csv_data)
    label = []
    comment = []
    for index, value in csv_data.iterrows():
        l = value.str.split('\t')
        if l[0][0] == '1':
            label.append(l[0][0])
            comment.append(l[0][1].strip())     # 去除空格
    return label, comment
