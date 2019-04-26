import random
import pickle
import numpy as np
import pandas as pd

random.seed(1234)

workdir = './datasets'
click_f = np.loadtxt(workdir + '/ratings_data.txt', dtype=np.int32)
trust_f = np.loadtxt(workdir + '/trust_data.txt', dtype=np.int32)

click_list = []  # 用户评分记录列表
trust_list = []  # 用户信任列表

u_read_list = []  # 某个用户看过的所有项目集合
u_friend_list = []  # 某个用户他的朋友集合
uf_read_list = []  # 某个用户他的朋友看过的所有项目的集合
i_read_list = []  # 看过某个项目的所有用户集合
i_friend_list = []  # 某个项目类似的项目集合
if_read_list = []  # 看过某个项目类似的项目的用户集合
i_link_list = []  # 两个项目之间的相似度集合
user_count = 0  # 用户数
item_count = 0  # 项目数

for s in click_f:
    uid = s[0]  # 用户ID
    iid = s[1]  # 项目ID
    label = s[2]  # 评分
    if uid > user_count:
        user_count = uid
    if iid > item_count:
        item_count = iid
    click_list.append([uid, iid, label])

pos_list = []
print("数据集总长度：", len(click_list))
for i in range(len(click_list)):
    pos_list.append((click_list[i][0], click_list[i][1], click_list[i][2]))
random.shuffle(pos_list)  # 随机打乱数据
train_set = pos_list[:int(0.8 * len(pos_list))]  # 训练集合占比80%
test_set = pos_list[int(0.8 * len(pos_list)):len(pos_list)]  # 测试集合占比20%
print("训练集长度：", len(train_set))

# 将数据集写成Tensorflow数据文件
with open(workdir + '/dataset.pkl', 'wb') as f:
    """
    pickle模块主要提供了数据持久化功能:
        序列化可使用dumps()函数，逆序列化使用loads()函数，将文件中的数据解析为一个python对象.
    pickle模块提供的常量：
        pickle.HIGHEST_PROTOCOL：整型，最高协议版本.
    　　pickle.DEFAULT_PROTOCOL：序列化中默认的协议版本，可能会低于HIGHEST_PROTOCOL，目前默认协议为3.
    """
    pickle.dump(train_set, f, pickle.HIGHEST_PROTOCOL)
    pickle.dump(test_set, f, pickle.HIGHEST_PROTOCOL)

train_df = pd.DataFrame(train_set, columns=['uid', 'iid', 'label'])  # 将训练数据集转为DataFrame格式
test_df = pd.DataFrame(test_set, columns=['uid', 'iid', 'label'])  # 将测试数据集转为DataFrame格式

click_df = pd.DataFrame(click_list, columns=['uid', 'iid', 'label'])  # 将用户-项目评分数据集转为DataFrame格式
train_df = train_df.sort_values(axis=0, ascending=True, by='uid')  # 根据UID排序

for u in range(user_count + 1):
    hist = train_df[train_df['uid'] == u]
    # hist = hist[hist['label']>3] # 想选择评分大于3的
    u_read = hist['iid'].unique().tolist()
    if not u_read:
        u_read_list.append([0])
    else:
        u_read_list.append(u_read)

train_df = train_df.sort_values(axis=0, ascending=True, by='iid')

for i in range(item_count + 1):
    hist = train_df[train_df['iid'] == i]
    # hist = hist[hist['label']>3] # 想选择评分大于3的
    i_read = hist['uid'].unique().tolist()
    if not i_read:
        i_read_list.append([0])
    else:
        i_read_list.append(i_read)

for s in trust_f:
    uid = s[0]
    fid = s[1]
    if uid > user_count or fid > user_count:
        continue
    trust_list.append([uid, fid])

trust_df = pd.DataFrame(trust_list, columns=['uid', 'fid'])
trust_df = trust_df.sort_values(axis=0, ascending=True, by='uid')

for u in range(user_count + 1):
    hist = trust_df[trust_df['uid'] == u]
    u_friend = hist['fid'].unique().tolist()
    if not u_friend:
        u_friend_list.append([0])
        uf_read_list.append([[0]])
    else:
        u_friend_list.append(u_friend)
        uf_read_f = []
        for f in u_friend:
            uf_read_f.append(u_read_list[f])
        uf_read_list.append(uf_read_f)

for i in range(item_count + 1):
    if len(i_read_list[i]) <= 30:
        i_friend_list.append([0])
        if_read_list.append([[0]])
        i_link_list.append([0])
        continue
    i_friend = []
    for j in range(item_count + 1):
        if len(i_read_list[j]) <= 30:
            sim_ij = 0
        else:
            sim_ij = 0
            for s in i_read_list[i]:
                sim_ij += np.sum(i_read_list[j] == s)
        i_friend.append([j, sim_ij])
    i_friend_cd = sorted(i_friend, key=lambda d: d[1], reverse=True)
    i_friend_i = []
    i_link_i = []
    for k in range(20):
        if i_friend_cd[k][1] > 5:
            i_friend_i.append(i_friend_cd[k][0])
            i_link_i.append(i_friend_cd[k][1])
    if not i_friend_i:
        i_friend_list.append([0])
        if_read_list.append([[0]])
        i_link_list.append([0])
    else:
        i_friend_list.append(i_friend_i)
        i_link_list.append(i_link_i)
        if_read_f = []
        for f in i_friend_i:
            if_read_f.append(i_read_list[f])
        if_read_list.append(if_read_f)

with open(workdir + '/list.pkl', 'wb') as f:
    pickle.dump(u_friend_list, f, pickle.HIGHEST_PROTOCOL)
    pickle.dump(u_read_list, f, pickle.HIGHEST_PROTOCOL)
    pickle.dump(uf_read_list, f, pickle.HIGHEST_PROTOCOL)
    pickle.dump(i_friend_list, f, pickle.HIGHEST_PROTOCOL)
    pickle.dump(i_read_list, f, pickle.HIGHEST_PROTOCOL)
    pickle.dump(if_read_list, f, pickle.HIGHEST_PROTOCOL)
    pickle.dump(i_link_list, f, pickle.HIGHEST_PROTOCOL)
    pickle.dump((user_count, item_count), f, pickle.HIGHEST_PROTOCOL)
