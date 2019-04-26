import numpy as np


class DataInput:
    def __init__(self, data, u_read_list, u_friend_list, uf_read_list, i_read_list, i_friend_list, if_read_list, \
                 i_link_list, batch_size, trunc_len):
        self.batch_size = batch_size
        self.data = data
        self.u_read_list = u_read_list
        self.u_friend_list = u_friend_list
        self.uf_read_list = uf_read_list
        self.i_read_list = i_read_list
        self.i_friend_list = i_friend_list
        self.if_read_list = if_read_list
        self.i_link_list = i_link_list
        self.epoch_size = len(self.data) // self.batch_size
        # self.epoch_size = len(self.data) // self.batch_size if len(self.data) % self.batch_size == 0 else len(self.data) // self.batch_size + 1
        self.trunc_len = trunc_len
        if self.epoch_size * self.batch_size < len(self.data):
            self.epoch_size += 1
        self.i = 0

    def __iter__(self):
        return self

    def __next__(self):
        if self.i == self.epoch_size:
            raise StopIteration

        ts = self.data[self.i * self.batch_size: min((self.i + 1) * self.batch_size, len(self.data))]
        self.i += 1

        iid, uid, label = [], [], []
        u_read, u_friend, uf_read = [], [], []
        u_read_l, u_friend_l, uf_read_l = [], [], []
        i_read, i_friend, if_read = [], [], []
        i_read_l, i_friend_l, if_read_l, i_link = [], [], [], []

        for t in ts:
            uid.append(t[0])
            iid.append(t[1])
            label.append(t[2])

            u_read_u = self.u_read_list[t[0]]  # 某个用户的观测项列表
            u_read.append(u_read_u)  # 某个用户的观测项列表集合
            u_read_l.append(len(u_read_u))  # 某个用户的观测项列表长度
            u_friend_u = self.u_friend_list[t[0]]  # 某个用户的朋友列表
            u_friend.append(u_friend_u)
            u_friend_l.append(min(len(u_friend_u), self.trunc_len))
            uf_read_u = self.uf_read_list[t[0]]
            uf_read.append(uf_read_u)
            uf_read_l_u = []
            for f in range(len(uf_read_u)):
                uf_read_l_u.append(min(len(uf_read_u[f]), self.trunc_len))
            uf_read_l.append(uf_read_l_u)

            i_read_i = self.i_read_list[t[1]]
            i_read.append(i_read_i)
            i_friend_i = self.i_friend_list[t[1]]
            i_friend.append(i_friend_i)
            if_read_i = self.if_read_list[t[1]]
            if_read.append(if_read_i)
            i_link_i = self.i_link_list[t[1]]
            i_link.append(i_link_i)
            if len(i_read_i) <= 1:
                i_read_l.append(0)
            else:
                i_read_l.append(len(i_read_i))
            if len(i_friend_i) <= 1:
                i_friend_l.append(0)
            else:
                i_friend_l.append(min(len(i_friend_i), self.trunc_len))
            if_read_l_i = []
            for f in range(len(if_read_i)):
                if len(if_read_i[f]) <= 1:
                    if_read_l_i.append(0)
                else:
                    if_read_l_i.append(min(len(if_read_i[f]), self.trunc_len))
            if_read_l.append(if_read_l_i)

        data_len = len(iid)

        # padding
        u_read_maxlength = max(u_read_l)
        u_friend_maxlength = min(10, max(u_friend_l))
        uf_read_maxlength = min(self.trunc_len,
                                max(max(uf_read_l)))  # eg: a = [[1,2,3],[8,5],[6,7]] ===> max(max(a)) = max([8,5]) = 8 -----> 按照元素里面元组的第一个元素的排列顺序，输出最大值（如果第一个元素相同，则比较第二个元素，输出最大值）据推理是按ascii码进行排序的
        u_read_input = np.zeros([data_len, u_read_maxlength], dtype=np.int32)
        for i, ru in enumerate(u_read):
            u_read_input[i, :len(ru)] = ru[:len(ru)]
        u_friend_input = np.zeros([data_len, u_friend_maxlength], dtype=np.int32)
        for i, fi in enumerate(u_friend):
            u_friend_input[i, :min(len(fi), u_friend_maxlength)] = fi[:min(len(fi), u_friend_maxlength)]
        uf_read_input = np.zeros([data_len, u_friend_maxlength, u_read_maxlength], dtype=np.int32)
        for i in range(len(uf_read)):
            for j, rj in enumerate(uf_read[i][:u_friend_maxlength]):
                uf_read_input[i, j, :min(len(rj), uf_read_maxlength)] = rj[:min(len(rj), uf_read_maxlength)]
        uf_read_l_input = np.zeros([data_len, u_friend_maxlength], dtype=np.int32)
        for i, fr in enumerate(uf_read_l):
            uf_read_l_input[i, :min(len(fr), u_friend_maxlength)] = fr[:min(len(fr), u_friend_maxlength)]

        i_read_maxlength = max(i_read_l)
        i_friend_maxlength = min(10, max(i_friend_l))
        if_read_maxlength = min(self.trunc_len, max(max(if_read_l)))
        i_read_input = np.zeros([data_len, i_read_maxlength], dtype=np.int32)
        for i, ru in enumerate(i_read):
            i_read_input[i, :len(ru)] = ru[:len(ru)]
        i_friend_input = np.zeros([data_len, i_friend_maxlength], dtype=np.int32)
        for i, fi in enumerate(i_friend):
            i_friend_input[i, :min(len(fi), i_friend_maxlength)] = fi[:min(len(fi), i_friend_maxlength)]
        if_read_input = np.zeros([data_len, i_friend_maxlength, i_read_maxlength], dtype=np.int32)
        for i in range(len(if_read)):
            for j, rj in enumerate(if_read[i][:i_friend_maxlength]):
                if_read_input[i, j, :min(len(rj), if_read_maxlength)] = rj[:min(len(rj), if_read_maxlength)]
        if_read_l_input = np.zeros([data_len, i_friend_maxlength], dtype=np.int32)
        for i, fr in enumerate(if_read_l):
            if_read_l_input[i, :min(len(fr), i_friend_maxlength)] = fr[:min(len(fr), i_friend_maxlength)]
        i_link_input = np.zeros([data_len, i_friend_maxlength, 1], dtype=np.int32)
        for i, li in enumerate(i_link):
            li = np.reshape(np.array(li), [-1, 1])
            i_link_input[i, :min(len(li), i_friend_maxlength)] = li[:min(len(li), i_friend_maxlength)]

        return self.i, (iid, uid, label), u_read_input, u_friend_input, uf_read_input, u_read_l, u_friend_l, uf_read_l_input, \
               i_read_input, i_friend_input, if_read_input, i_link_input, i_read_l, i_friend_l, if_read_l_input
