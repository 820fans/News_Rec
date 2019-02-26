
import numpy as np
import pandas as pd
import re
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import svds
from scipy import sparse
import matplotlib.pyplot as plt
from jieba import analyse
import jieba
import operator
import math
import random
import re
from scipy import spatial

# load Glove Vectors
embedding_path = "/media/yida/Data/Embedding/词向量/sgns.baidubaike.bigram-char"
embeddings_index = {}
EMBEDDING_DIM = 300
with open(embedding_path, encoding='utf-8') as f:
    for i, line in enumerate(f):
        values = line.split()
        words = values[:-EMBEDDING_DIM]
        word = ''.join(words)
        try:
            coefs = np.asarray(values[-EMBEDDING_DIM:], dtype='float32')
            embeddings_index[word] = coefs
        except:
            pass

train_df = pd.read_csv("Data/train_data.txt", sep='\t', header=-1)
test_df = pd.read_csv("Data/test_data.txt", sep='\t', header=-1)
# 构建新闻关键词
news_map = {}
def generate_news_map():
    # news_keywords = open("Data/news_keywords.txt", "w", encoding="utf-8")
    with open("Data/news_context.txt") as newsf:
        for line in newsf:
            items = line.strip().split("\t")
            if(len(items)!=4): continue
            wds = jieba.cut(items[1])
            words = []
            for wd in wds:
                words.append(wd)
            news_map[items[0]] = words
generate_news_map()

def generate_keywords_map():
    kw_map = {}
    with open("Data/news_keywords.txt") as nkf:
        for line in nkf:
            items = line.strip().split('\t')
            if len(items)>1:
                kw_map[items[0]] = []
                for wd in items[1:]:
                    if wd.isnumeric(): continue
                    if re.search('[a-zA-Z]', wd): continue
                    kw_map[items[0]].append(wd)
    return kw_map

# 每个新闻由关键词组成向量
USER_NUM = 10000
ITEM_NUM = 6183
news_emb = {}
news_kws_map = generate_keywords_map()
for item_id in range(ITEM_NUM):
    item_id = str(item_id)
    emb = []
    for wd in news_map[item_id]:
        try:
            wemb = embeddings_index[wd]
            if len(emb) == 0:
                emb = wemb
            else:
                emb = emb + wemb
        except:
            pass
    if len(emb)>0:
        news_emb[item_id] = emb/len(news_map[item_id])


# 统计用户特征
def get_user_news_map(data_df):
    user_news_map = {}
    for i in range(data_df.shape[0]):
        user_id, news_id = data_df.iloc[i, 0], str(data_df.iloc[i, 1])
        if user_id not in user_news_map:
            user_news_map[user_id] = {}
        for wd in news_kws_map[news_id]:
            if wd.isnumeric(): continue
            if re.search('[a-zA-Z]', wd): continue
            if wd in user_news_map[user_id]:
                user_news_map[user_id][wd] += 1
            else:
                user_news_map[user_id][wd] = 1
    return user_news_map

# 构建用户特征关键词画像，每个用户采用20个关键词描述
train_user_kws = get_user_news_map(train_df)
train_user_emb = {}
for user_id in train_user_kws:
    train_user_kws[user_id] = sorted(train_user_kws[user_id].items(), key=operator.itemgetter(1), reverse=True)[:10]
    # print(train_user_kws[user_id])
    train_user_kws[user_id] = [item[0] for item in train_user_kws[user_id]]
    # print(train_user_kws[user_id])
    emb = []
    for wd in train_user_kws[user_id]:
        try:
            wemb = embeddings_index[wd]
            if len(emb) == 0:
                emb = wemb
            else:
                emb = emb + wemb
        except:
            pass
    if len(emb) > 0:
        train_user_emb[user_id] = emb/len(train_user_kws[user_id])

# print(train_user_emb[0])
# exit(100)
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
read_sum = test_df.shape[0]
user_row = np.array([test_df.iloc[i, 0] for i in range(read_sum)])
item_col = np.array([test_df.iloc[i, 1] for i in range(read_sum)])
read_score = np.array([1 for i in range(read_sum)])
# 构建稀疏矩阵
test_mat = csr_matrix((read_score, (user_row, item_col)), shape=(USER_NUM, ITEM_NUM))

ui_dict = dict()
for i in range(test_df.shape[0]):
    if test_df.iloc[i, 0] not in ui_dict.keys():
        ui_dict[test_df.iloc[i, 0]] = [test_df.iloc[i, 1]]
    else:
        ui_dict[test_df.iloc[i, 0]].append(test_df.iloc[i, 1])


def cal_PN(predlist, reclist, n=10):
    p = 0
    for pred in predlist:
        if pred in reclist:
            p += 1
    p /= n
    return p

def cal_AP(predlist, reclist):
    pos = 1
    rel = 1
    ap = 0
    for i in range(len(reclist)):
        if reclist[i] in predlist:
            ap += rel / pos
            rel += 1
        pos += 1
    ap /= len(reclist)
    return ap

def cal_DCG(user, predlist, reclist, n=10):
    pred_rank = [test_mat[user, item] for item in predlist]
    rec_rank = [test_mat[user, item] for item in reclist]
    dcg = pred_rank[0]
    idcg = rec_rank[0]
    for i in range(1, len(pred_rank)):
        dcg += pred_rank[i] / math.log2(i + 1)
    for i in range(1, len(rec_rank)):
        idcg += rec_rank[i] / math.log2(i + 1)
    ndcg = dcg / idcg
    return ndcg

def vecfy(a, b):
    return 1 - spatial.distance.cosine(a, b)

# 在所有新闻里，预测所有新闻
def predict_topK(user_id, K):
    user_rating = ui_mat[user_id, :]
    reclist = dict()
    for item_id in range(ITEM_NUM):
        if user_rating[item_id] == 0:
            str_item_id = str(item_id)
            if str_item_id in news_emb:
                # print(news_map[str(item_id)])
                prediction = vecfy(news_emb[str_item_id], train_user_emb[user_id])
                reclist[item_id] = prediction
    # exit(200)
    # 取topK个项目生成推荐列表
    rec_topK = sorted(reclist.items(), key=lambda e: e[1], reverse=True)
    # print(rec_topK)
    return [rec_topK[i][0] for i in range(K)]
            
eval_user = 0
topn = 10
mAP = 0
nDCG = 0
mPrecision = 0
user_sum = len(ui_dict)
cnt = 0
cnt2 = 0
# cnt1 = 0, cnt2 = 0, cnt3 = 0
for user_id, itemlist in ui_dict.items():
    eval_user += 1
    if eval_user % 100 == 0:
        print("Eval process: %d / %d" % (eval_user, user_sum))
    if eval_user > user_sum:
        break
    if user_id not in train_user_emb:
        # 随机选择
        predlist = [random.randint(0, ITEM_NUM-1) for i in range(topn)]
    else:
        cnt += 1
        predlist = predict_topK(user_id, topn)
    reclist = list(set(itemlist)) # 用户实际点击了哪些新闻
    mPrecision += cal_PN(predlist, reclist)
    # print(cal_PN(predlist, reclist), ";", cal_AP(predlist, reclist))
    mAP += cal_AP(predlist, reclist)
    nDCG +=  cal_DCG(user_id, predlist, reclist)
    if mAP > 0:
        cnt2 += 1

print(cnt2, " users got val, map:", mAP/cnt2)
print(cnt, " users tested")
mPrecision /= eval_user
mAP /= eval_user
nDCG /= eval_user
print("Top%d Rec Result:" % topn)
print("mPrecision: %g  mAP: %g  nDCG: %g" % (mPrecision, mAP, nDCG))

# mPrecision: 0.00258949  mAP: 0.00486149  nDCG: 0.00438426 每个用户有20个标签
# mPrecision: 0.00331302  mAP: 0.00840359  nDCG: 0.0060068  每个用户有10个标签
# mPrecision: 0.00251333  mAP: 0.00395982  nDCG: 0.00417177          5

# 计算所有用户之后（包含冷启动用户） mPrecision: 0.00249284  mAP: 0.00632316  nDCG: 0.00451973
# 为冷启动用户新增随机推荐之后的准确率：mPrecision: 0.00289398  mAP: 0.00645916  nDCG: 0.00487577