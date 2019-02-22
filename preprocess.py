'''
对数据集的预处理，包括统计平均用户点击，新闻被点击次数
抽取用户-新闻对数据，和新闻id-内容数据
根据时间戳分割训练集和测试集
'''
import pandas as pd
import time

def statistic():
    data_df = pd.read_csv("Data/user_click_data.txt", sep='\t', header=-1)
    user_dict = dict()
    news_dict = dict()
    for i in range(data_df.shape[0]):
        user_id = data_df.iloc[i, 0]
        news_id = data_df.iloc[i, 1]
        if user_id not in user_dict:
            user_dict[user_id] = 1
        else:
            user_dict[user_id] += 1
        if news_id not in news_dict:
            news_dict[news_id] = 1
        else:
            news_dict[news_id] += 1
    user_read_mean = sum(user_dict.values()) / len(user_dict)
    news_read_mean = sum(news_dict.values()) / len(news_dict)
    min_userread = min(user_dict.values())
    min_newsread = min(news_dict.values())
    print("Mean user read times: %g" % user_read_mean)
    print("Mean news read times: %g" % news_read_mean)
    print("Min user read times: %d" % min_userread)
    print("Min news read times: %d" % min_newsread)

# 抽取新闻内容
def extract():
    data_df = pd.read_csv("Data/user_click_data.txt", sep='\t', header=-1)
    user_dict = dict()
    news_dict = dict()
    uid = 0
    nid = 0
    # 提取用户和新闻id,存储新闻内容
    with open("Data/news_context.txt") as news_info_file:
        for i in range(data_df.shape[0]):
            user_id = data_df.iloc[i, 0]
            news_id = data_df.iloc[i, 1]
            if user_id not in user_dict:
                user_dict[user_id] = uid
                uid += 1
            if news_id not in news_dict:
                news_dict[news_id] = nid
                context = [str(nid), str(data_df.iloc[i, 3]), str(data_df.iloc[i, 4]), str(data_df.iloc[i, 5])]
                news_info_file.write('\t'.join(context) + '\n')
                nid += 1

    # 存储用户id字典和新闻id字典
    user_list = sorted(user_dict.items(), key=lambda e: e[1], reverse=False)
    news_list = sorted(news_dict.items(), key=lambda e: e[1], reverse=False)
    user_file = open("Data/user_id.txt", 'w', encoding='utf-8')
    for user in user_list:
        user_file.write(str(user[1]) + '\t' + str(user[0]) + '\n')
    user_file.close()
    news_file = open("Data/news_id.txt", 'w', encoding='utf-8')
    for news in news_list:
        news_file.write(str(news[1]) + '\t' + str(news[0]) + '\n')
    news_file.close()

    # 存储用户id-新闻id-时间数据
    user_news_df = data_df.loc[:, 0:2]
    for i in range(data_df.shape[0]):
        user_news_df.iloc[i, 0] = user_dict[user_news_df.iloc[i, 0]]
        user_news_df.iloc[i, 1] = news_dict[user_news_df.iloc[i, 1]]
    user_news_df.to_csv("Data/user_news_id.csv", sep='\t', header=False, index=False)

# 时间戳转日期
def timestamp2datatime(value):
    format = '%Y-%m-%d %H:%M'
    #format = '%Y-%m-%d %H:%M:%S'
    #value 为时间戳值,如:1460073600.0
    value = time.localtime(value)
    dt = time.strftime(format, value)
    return dt

def split_data():
    data_df = pd.read_csv("Data/user_news_id.csv", header=-1, sep='\t')
    train_file = open("Data/train_data.txt", 'w', encoding='utf-8')
    test_file = open("Data/test_data.txt", 'w', encoding='utf-8')
    for i in range(data_df.shape[0]):
        read_time = data_df.iloc[i, 2]
        date = timestamp2datatime(read_time)
        day = date.split(' ')[0].split('-')[2]
        uid = str(data_df.iloc[i, 0])
        nid = str(data_df.iloc[i, 1])
        if int(day) < 20:
            train_file.write(uid + '\t' + nid + '\t' + date + '\n')
        else:
            test_file.write(uid + '\t' + nid + '\t' + date + '\n')
    train_file.close()
    test_file.close()

def cold_start():
    # 统计冷启动用户有多少
    train_df = pd.read_csv("Data/train_data.txt", header=-1, sep='\t')
    test_df = pd.read_csv("Data/test_data.txt", header=-1, sep='\t')
    train_uset = set()
    for i in range(train_df.shape[0]):
        train_uset.add(train_df.iloc[i, 0])
    print("Train users:", str(len(train_uset)))
    test_uset = set()
    for i in range(test_df.shape[0]):
        test_uset.add(test_df.iloc[i, 0])
    print("Test users:", str(len(test_uset)))
    # 寻找出现在test_set里，却没有出现在train_uset里的用户
    cnt = 0
    for test_uid in test_uset:
        if test_uid not in train_uset:
            cnt += 1
    print(str(cnt) + " users not in train_uset")
    print(str(100*cnt/(len(test_uset))) + "% percent in test_data is cold")
# split_data()
# statistic()
cold_start()