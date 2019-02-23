import numpy as np
import pandas as pd

from queue import Queue, PriorityQueue

# from models.basemodel import basemodel
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import svds
from scipy import sparse
import matplotlib.pyplot as plt
from jieba import analyse
# 引入TF-IDF关键词抽取接口
tfidf = analyse.extract_tags
import operator

news_map = {}
def generate_news_map():
    # news_keywords = open("Data/news_keywords.txt", "w", encoding="utf-8")
    with open("Data/news_context.txt") as newsf:
        for line in newsf:
            items = line.strip().split("\t")
            if(len(items)!=4): continue
            keywords = tfidf(items[1]+"。 "+items[2])
            news_map[items[0]]=keywords
            # arr = [items[0],]
            # for keyword in keywords: arr.append(keyword)
            # news_keywords.write('\t'.join(arr) + '\n')
    # news_keywords.close()
# generate_news_map()

def evaluate(test_df):
    read_sum = test_df.shape[0]
    user_row = np.array([test_df.iloc[i, 0] for i in range(read_sum)])
    item_col = np.array([test_df.iloc[i, 1] for i in range(read_sum)])
    read_score = np.array([1 for i in range(read_sum)])
    # 构建稀疏矩阵
    self.test_mat = csr_matrix((read_score, (user_row, item_col)), shape=(self.USER_NUM, self.ITEM_NUM))
    # print(self.test_mat)

    ui_dict = dict()
    for i in range(test_df.shape[0]):
        if test_df.iloc[i, 0] not in ui_dict.keys():
            ui_dict[test_df.iloc[i, 0]] = [test_df.iloc[i, 1]]
        else:
            ui_dict[test_df.iloc[i, 0]].append(test_df.iloc[i, 1])
    # ui_dict() 测试集合里，用户点击的新闻，一个用户对应多个新闻
    
train_df = pd.read_csv("Data/train_data.txt", sep='\t', header=-1)
test_df = pd.read_csv("Data/test_data.txt", sep='\t', header=-1)
# 构建新闻关键词
news_map = {}
with open("Data/news_keywords.txt") as nkf:
    for line in nkf:
        items = line.strip().split('\t')
        if len(items)>1:
            news_map[items[0]] = items[1:]

# 统计用户特征
def get_user_news_map(data_df):
    user_news_map = {}
    for i in range(data_df.shape[0]):
        user_id, news_id = data_df.iloc[i, 0], str(data_df.iloc[i, 1])
        if user_id not in user_news_map:
            user_news_map[user_id] = {}
        for wd in news_map[news_id]:
            if wd in user_news_map[user_id]:
                user_news_map[user_id][wd] += 1
            else:
                user_news_map[user_id][wd] = 1
    return user_news_map

# 构建用户特征关键词画像，每个用户采用20个关键词描述
train_user_kws = get_user_news_map(train_df)
for user_id in train_user_kws:
    train_user_kws[user_id] = sorted(train_user_kws[user_id].items(), key=operator.itemgetter(1), reverse=True)[:20]

# 用户-新闻 点击矩阵
USER_NUM = 10000
ITEM_NUM = 6183
def get_mat(ui_df):
    read_sum = ui_df.shape[0]
    user_row = np.array([ui_df.iloc[i, 0] for i in range(read_sum)])
    item_col = np.array([ui_df.iloc[i, 1] for i in range(read_sum)])
    mat = np.zeros((USER_NUM, ITEM_NUM))
    for i in range(read_sum):
        mat[user_row[i], item_col[i]] += 1
    return mat
ui_mat = get_mat(train_df)  # 用户-新闻 点击矩阵

# 测试集处理，
ui_dict = dict()
for i in range(test_df.shape[0]):
    if test_df.iloc[i, 0] not in ui_dict.keys():
        ui_dict[test_df.iloc[i, 0]] = [test_df.iloc[i, 1]]
    else:
        ui_dict[test_df.iloc[i, 0]].append(test_df.iloc[i, 1])

# 在所有新闻里，预测所有新闻
def predict_topK(user_id):
    pass

eval_user = 0
user_sum = len(ui_dict)
for user_id, itemlist in ui_dict.items():
    eval_user += 1
    if eval_user % 100 == 0:
        print("Eval process: %d / %d" % (eval_user, user_sum))
    if eval_user > user_sum:
        break
    
# print(train_user_kws[0])
# print(train_user_kws[1])
