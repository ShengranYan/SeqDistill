import sys
import time
import numpy as np
import torch
import random

from torch.utils.data import DataLoader

from utils import to_one_hot
from data_utils import leave_one_out, SeqData
import torch.utils.data as data

"""2024/03/22检查: 验证时输入的数据没有问题"""
def evaluate_syn_valid_data(model, valid_data, ground_truth, usernum, itemnum, device, args):
    NDCG = 0.0
    HT = 0.0
    valid_user = 0
    if usernum > 10000:
        users = random.sample(range(1, usernum + 1), k=10000)
    else:
        users = range(1, usernum + 1)

    for u in users:
        if len(valid_data[u]) < 1 or len(ground_truth[u]) < 1: continue
        seq = np.zeros([args.maxlen], dtype=np.int32)
        idx = args.maxlen - 1
        for i in reversed(valid_data[u]):
            seq[idx] = i
            idx -= 1
            if idx == -1: break

        rated = set(valid_data[u])
        rated.add(0)
        item_idx = [ground_truth[u][0]]
        for _ in range(100): # choose random 100 item for evaluation
            t = np.random.randint(1, itemnum + 1)
            while t in rated: t = np.random.randint(1, itemnum + 1)
            item_idx.append(t)
        timeline_musk = torch.BoolTensor(seq != 0)
        predictions = -model.predict(u, to_one_hot(seq, itemnum + 1, device).unsqueeze(0), to_one_hot(item_idx, itemnum + 1, device), timeline_musk.unsqueeze(0))
        predictions = predictions[0]
        rank = predictions.argsort().argsort()[0].item()
        valid_user += 1

        if rank < 10:
            NDCG += 1 / np.log2(rank + 2)
            HT += 1
        if valid_user % 100 == 0:
            print('.', end="")
            sys.stdout.flush()

    # for user, history in valid_data.items(): #TODO: 改验证，只抽取10000个用户进行验证
    #     if len(history) < 1 or len(ground_truth[user]) < 1: continue
    #     seq = np.zeros([args.maxlen], dtype=np.int32)
    #     idx = args.maxlen - 1
    #     for i in reversed(history):
    #         seq[idx] = i
    #         idx -= 1
    #         if idx == -1: break
    #     rated = set(history)
    #     rated.add(0)
    #     item_idx = [ground_truth[user][0]]
    #     for _ in range(100):
    #         t = np.random.randint(1, itemnum + 1)
    #         while t in rated: t = np.random.randint(1, itemnum + 1)
    #         item_idx.append(t)
    #     timeline_musk = torch.BoolTensor(seq != 0)
    #     predictions = -model.predict(user, to_one_hot(seq, itemnum + 1, device).unsqueeze(0), to_one_hot(item_idx, itemnum + 1, device), timeline_musk.unsqueeze(0))
    #     predictions = predictions[0]
    #
    #     rank = predictions.argsort().argsort()[0].item()
    #
    #     valid_user += 1
    #
    #     if rank < 10:
    #         NDCG += 1 / np.log2(rank + 2)
    #         HT += 1
    #     if valid_user % 100 == 0:
    #         print('.', end='')
    #         sys.stdout.flush()
    #

    print()
    return NDCG / valid_user, HT / valid_user

def epoch(mode, model, train_dataloader, optimizer, criterion, args):
    loss_avg = 0.0
    #model.to(args.device)
    criterion = criterion.to(args.device)
    if mode == 'train' : model.train()
    else: model.eval()

    for data in train_dataloader:
        u, seq, pos, neg, timeline_musk = data
        seq_, pos_, neg_ = seq.argmax(dim=-1), pos.argmax(dim=-1), neg.argmax(dim=-1)
        pos_logits, neg_logits = model(u, seq, pos, neg, timeline_musk)
        pos_labels, neg_labels = torch.ones(pos_logits.shape, device=args.device), torch.zeros(neg_logits.shape, device=args.device)

        loss = criterion(pos_logits[timeline_musk], pos_labels[timeline_musk]) + criterion(neg_logits[timeline_musk], neg_labels[timeline_musk])
        loss_avg += loss.item()

        for param in model.item_emb.parameters(): loss += args.l2_emb * torch.norm(param)
        if mode == 'train':
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        loss_avg = loss_avg / len(train_dataloader)
    return loss_avg

def evaluate_synset(model, train_dataloader, train_data, valid_data, usernum, itemnum, device, args, is_trained=False):
    """用syn_data训练模型
    train_dataloader: 验证时训练模型的数据
    train_data: 验证时输入给模型的已知序列信息
    valid_data: 验证时要预测的下一个物品
    """
    if not is_trained:

        model = model.to(device)

        for name, param in model.named_parameters():
            try:
                torch.nn.init.xavier_normal_(param.data)
            except:
                pass  # just ignore those failed init layers

        bce_criterion = torch.nn.BCEWithLogitsLoss()
        adam_optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.98))
        epoch_num = args.epoch_eval_train
        print('training random model with synth data...')

        for ep in range(epoch_num):
            loss_avg = epoch('train', model, train_dataloader, adam_optimizer, bce_criterion, args)

    print('evaluating', end='')

    NDCG_valid, HS_valid = evaluate_syn_valid_data(model, train_data, valid_data, usernum, itemnum, device, args)

    return NDCG_valid, HS_valid
