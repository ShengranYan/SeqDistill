from collections import defaultdict
import numpy as np
import torch
import torch.utils.data as data
from utils import to_one_hot, random_neq


def read_data(fname):
    usernum = 0
    itemnum = 0
    User = defaultdict(list)
    Item = defaultdict(list)
    f = open('data/%s.txt' % fname, 'r')
    for line in f: #转换成用户与对应的物品序列的字典
        u, i = line.rstrip().split(' ')
        u = int(u)
        i = int(i)
        usernum = max(u, usernum)
        itemnum = max(i, itemnum)
        User[u].append(i) #user history
        Item[i].append(u) #item history

    return [User, Item, usernum, itemnum]

def item_freq_by_pos(train_dict, usernum, itemnum, tgt_len):
    item_freq = torch.zeros([tgt_len, itemnum + 1], dtype=torch.float32)
    for u in range(1, usernum + 1):
        for idx, item in enumerate(reversed(train_dict[u])):
            if idx >= tgt_len:
                break
            item_freq[tgt_len - idx - 1][item] += 1.0
    return item_freq

def item_freq(train_dict, usernum, itemnum):
    item_freq = torch.zeros([itemnum + 1,], dtype=torch.float32)
    for u in range(1, usernum + 1):
        for j in range(len(train_dict[u])):
            item_freq[train_dict[u][j]] += 1.0
    return item_freq

def data_partition(fname):
    usernum = 0
    itemnum = 0
    User = defaultdict(list)
    user_train = {}
    user_valid = {}
    user_test = {}
    # assume user/item index starting from 1
    f = open('data/%s.txt' % fname, 'r')
    for line in f:
        u, i = line.rstrip().split(' ')
        u = int(u)
        i = int(i)
        usernum = max(u, usernum)
        itemnum = max(i, itemnum)
        User[u].append(i)

    for user in User:
        nfeedback = len(User[user])
        if nfeedback < 3:
            user_train[user] = User[user]
            user_valid[user] = []
            user_test[user] = []
        else:
            user_train[user] = User[user][:-2]
            user_valid[user] = []
            user_valid[user].append(User[user][-2])
            user_test[user] = []
            user_test[user].append(User[user][-1])
    return [user_train, user_valid, user_test, usernum, itemnum]

def leave_one_out(data_dict):
    """
    :param data_dict:
    :return: data_input, ground_truth
    """
    data_input = {}
    ground_truth = {}

    for user in data_dict:
        nfeedback = len(data_dict[user])
        if nfeedback < 2:
            data_input[user] = data_dict[user]
            ground_truth[user] = []
        else:
            data_input[user] = data_dict[user][:-1]
            ground_truth[user] = []
            ground_truth[user].append(data_dict[user][-1])

    return [data_input, ground_truth]

class SeqData(data.Dataset):
    def __init__(self, user_train, user_num, item_num, max_len, device, is_training=None):
        self.user_train = user_train
        self.user_num = user_num
        self.item_num = item_num
        self.is_training = is_training
        self.max_len = max_len
        self.device = device

    def ng_sample(self):
        assert self.is_training

        self.data_fill = []
        for u, ori_seq in self.user_train.items():
            seq = np.zeros([self.max_len], dtype=np.int32)
            pos = np.zeros([self.max_len], dtype=np.int32)
            neg = np.zeros([self.max_len], dtype=np.int32)
            nxt = ori_seq[-1]
            idx = self.max_len - 1

            ts = set(ori_seq)
            for i in reversed(ori_seq[:-1]):
                seq[idx] = i
                pos[idx] = nxt
                if nxt != 0:
                    neg[idx] = random_neq(1, self.item_num + 1, ts)  # 随机生成一个负物品
                nxt = i
                idx -= 1
                if idx == -1: break
            timeline_musk = torch.BoolTensor(seq != 0)
            self.data_fill.append([u, to_one_hot(seq, self.item_num + 1, self.device),
                to_one_hot(pos, self.item_num + 1, self.device), to_one_hot(neg, self.item_num + 1, self.device), timeline_musk])

    def __len__(self):
        return len(self.user_train)

    def __getitem__(self, item):
        contents = self.data_fill if self.is_training else self.user_train
        return contents[item]

class SynSeqData(data.Dataset):
    """合成数据集的Dataset类，方便梯度传播"""
    def __init__(self, user_train, user_num, item_num, max_len, syn_len, device, is_training=None):
        self.user_train = user_train
        self.user_num = user_num
        self.item_num = item_num
        self.is_training = is_training
        self.max_len = max_len
        self.syn_len = syn_len
        self.ZeroPad = torch.nn.ZeroPad2d(padding=(0, 0, self.max_len - syn_len + 1, 0))
        self.device = device

    def ng_sample(self, sample=False):
        assert self.is_training

        self.data_fill = []
        all_item = torch.tensor([i for i in range(self.item_num + 1)], dtype=torch.float32, device=self.device)
        for u, ori_seq in self.user_train.items():
            padding = to_one_hot(torch.zeros((self.max_len - self.syn_len + 1,), dtype=torch.long, device=self.device), class_num=self.item_num + 1, device=self.device)
            padding_pos = to_one_hot(torch.zeros((self.max_len - self.syn_len,), dtype=torch.long, device=self.device), class_num=self.item_num + 1, device=self.device)
            seq = torch.cat([padding, ori_seq[:-1]], dim=0)
            pos = torch.cat([padding_pos, ori_seq], dim=0)
            neg = torch.zeros((self.max_len,), dtype=torch.long)
            if sample:

                #timeline_musk = torch.zeros((self.max_len,), dtype=torch.int32)
                idx = self.max_len - 1
                idx_seq = torch.matmul(ori_seq.detach().clone(), all_item).long().cpu()
                nxt = idx_seq[-1]
                ts = set(idx_seq)
                for i in reversed(idx_seq[:-1]):
                    if nxt != 0:
                        neg[idx] = random_neq(1, self.item_num + 1, ts)  # 随机生成一个负物品
                    nxt = i
                    idx -= 1
                    if idx == (self.max_len - self.syn_len - 1): break

            # timeline_musk = torch.BoolTensor(neg != 0)
            timeline_musk = torch.zeros((self.max_len, ), dtype=torch.bool)
            timeline_musk[self.max_len - self.syn_len : ] = True

            self.data_fill.append([u, seq, pos, to_one_hot(neg, self.item_num + 1, device=self.device), timeline_musk])

    def __len__(self):
        return len(self.user_train)

    def __getitem__(self, item):
        contents = self.data_fill if self.is_training else self.user_train
        return contents[item]



def neg_sample(user_train, args, item_num):
    data_fill = []
    for u, ori_seq in user_train.items():
        seq = np.zeros([args.maxlen], dtype=np.int32)
        pos = np.zeros([args.maxlen], dtype=np.int32)
        neg = np.zeros([args.maxlen], dtype=np.int32)
        nxt = ori_seq[-1]
        idx = args.maxlen - 1

        ts = set(ori_seq)
        for i in reversed(ori_seq[:-1]):
            seq[idx] = i
            pos[idx] = nxt
            if nxt != 0:
                neg[idx] = random_neq(1, item_num + 1, ts)  # 随机生成一个负物品
            nxt = i
            idx -= 1
            if idx == -1: break
        timeline_musk = torch.BoolTensor(seq != 0)
        data_fill.append([u, torch.LongTensor(seq), torch.LongTensor(pos), torch.LongTensor(neg), timeline_musk])
    return data_fill


class OriDataset(torch.utils.data.Dataset): #TODO: to be finished
    def __init__(self, data_fill, args):
        self.data_fill = data_fill
    def __len__(self):
        return len(self.data_fill)
    def __getitem__(self, item):
        return self.data_fill[item]
