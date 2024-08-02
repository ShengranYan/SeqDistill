import copy
import math
import torch
import torch.nn.functional as F
from utils import gumbel_softmax_fix, match_loss, get_time, set_seed
from data_utils import read_data, data_partition, SynSeqData, neg_sample, OriDataset
from Models.SASRec import SASRec
from evaluate import *
from argParser import args
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

if __name__ == '__main__':

    device = args.device
    RANDOM_SEED = 1
    set_seed()

    [user_train, user_valid, user_test, user_num, item_num] = data_partition(args.dataset)
    train_length = len(user_train)
    num_batch = math.ceil(train_length / args.batch_size) #上取整



    print("Train set size: %d." % train_length)
    print("Dataset: %s \nUser num: %d \nItem num: %d \nModel: %s" % (args.dataset, user_num, item_num, args.model))

    fu = np.load(f'fu_{args.dataset}_{args.co_cluster_num}.npy')
    cluster_center = np.argmax(fu[1:], axis=0) + 1 #获取聚类中心的用户
    co_cluster_label = fu > args.cluster_prob_thre

    """____________________________________co_cluster____________________________________"""
    indices_class = [[] for c in range(args.co_cluster_num)] #储存每一个类别在源数据集中的下标
    for i in range(1, user_num + 1):
        for c in range(args.co_cluster_num):
            if co_cluster_label[i][c]:
                indices_class[c].append(i)

    data_fill = neg_sample(user_train, args, item_num) #负采样

    clusters_raw = [[] for c in range(args.co_cluster_num)] #存在空类

    for c in range(args.co_cluster_num): #把每一类的用户集合起来
        for i in indices_class[c]:
            clusters_raw[c].append(data_fill[i - 1])

    clusters = [ sublist for sublist in clusters_raw if len(sublist) > 1 ]

    for i in range(len(clusters)):
        print(len(clusters[i]))
    cluster_num = len(clusters)
    print('cluster num:', cluster_num)

    def get_cluster(c):
        return clusters[c]

    """____________________________________synthetic dataset____________________________________"""
    syn_data = torch.randn(size=(cluster_num * args.num_per_cluster, args.syn_seq_length, item_num + 1), requires_grad=True, dtype=torch.float32,
                           device=device)
    print('synth data interaction num = ', cluster_num * args.num_per_cluster * args.syn_seq_length)
    torch.nn.init.xavier_uniform_(syn_data.data)

    optimizer_syn = torch.optim.Adam([syn_data, ], lr=args.lr_syn_seq, betas=(0.9, 0.98))
    optimizer_syn.zero_grad()
    criterion = torch.nn.BCEWithLogitsLoss()

    best_NDCG = 0.0

    for e in range(args.num_epochs):

        """____________________________________evaluation____________________________________"""
        if e % args.epoch_num_for_eval == 0 and e != 0:
            syn_seqs_eval = {}
            data_logits_eval = copy.deepcopy(syn_data.softmax(dim=-1).log().detach().clone())
            for i in range(cluster_num * args.num_per_cluster):
                syn_seqs_eval[i + 1] = F.gumbel_softmax(data_logits_eval[i], tau=args.tau, hard=True, dim=-1)

            sample_accuracy = 0
            for i in range(cluster_num * args.num_per_cluster):
                for j in range(args.syn_seq_length):
                    if syn_data[i][j].argmax() == syn_seqs_eval[i + 1][j].argmax():
                        sample_accuracy += 1

            print("sample accuracy eval: ", sample_accuracy / (cluster_num * args.num_per_cluster * args.syn_seq_length))

            syn_dataset_eval = SynSeqData(syn_seqs_eval, cluster_num * args.num_per_cluster, item_num, args.maxlen, args.syn_seq_length, device,True)
            syn_dataset_eval.ng_sample(sample=True)
            syn_dataloader_eval = data.DataLoader(syn_dataset_eval, batch_size=1, shuffle=False)

            """evaluate synthetic data"""
            print("epoch=%d, start evaluating..." % e)
            all_NDCG_valid, all_HS_valid = [], []


            for it_eval in range(args.num_eval):
                eval_model = SASRec(user_num, item_num, args).to(device)
                NDCG_valid, HS_valid = evaluate_synset(eval_model, syn_dataloader_eval, user_train, user_valid, user_num, item_num, device, args)
                all_NDCG_valid.append(NDCG_valid)
                all_HS_valid.append(HS_valid)

            del syn_seqs_eval
            del syn_dataloader_eval
            del data_logits_eval
            print('Evaluate %d random %s, NDCG_valid = %.4f HS_valid = %.4f\n------------------------------' % (
            args.num_eval, args.model, np.mean(all_NDCG_valid), np.mean(all_HS_valid)))

            if best_NDCG < np.mean(all_NDCG_valid):
                best_NDCG = np.mean(all_NDCG_valid)
                #TODO:保存数据集

        model = SASRec(user_num, item_num, args).to(device)

        for name, param in model.named_parameters():
            try:
                torch.nn.init.xavier_normal_(param.data)
            except:
                pass

        model_parameters = list(model.parameters())
        optimizer_model = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.98))
        optimizer_model.zero_grad()
        bce_criterion = torch.nn.BCEWithLogitsLoss()

        for ol in range(args.outer_loops):
            loss = torch.tensor(0.0).to(args.device)
            prev_idx = 0
            for c in range(cluster_num):
                seq_ori = get_cluster(c)
                Seq_dataset = OriDataset(seq_ori, args)
                Seq_dataloader = data.DataLoader(Seq_dataset, batch_size=Seq_dataset.__len__() // args.num_per_cluster, shuffle=False) #TODO: batch_size=cluster_len / args.num_per_cluster

                seq_syn = syn_data[c * args.num_per_cluster : (c + 1) * args.num_per_cluster]

                data_logits = seq_syn.softmax(dim=-1).log()

                syn_seqs = {}
                for i in range(args.num_per_cluster):
                    syn_seqs[i + 1] = F.gumbel_softmax(data_logits[i], tau=args.tau, hard=True, dim=-1)

                syn_dataset = SynSeqData(syn_seqs, cluster_num * args.num_per_cluster, item_num, args.maxlen, args.syn_seq_length, device, True)
                syn_dataset.ng_sample()
                syn_dataloader = data.DataLoader(syn_dataset, batch_size=args.num_per_cluster, shuffle=False)

                for _, (data_ori, data_syn) in enumerate(zip(Seq_dataloader, syn_dataloader)):
                    u_ori, seqs_ori, pos_ori, neg_ori, timeline_musk_ori = data_ori
                    u_syn, seqs_syn, pos_syn, neg_syn, timeline_musk_syn = data_syn

                    pos_logits_ori, neg_logits_ori = model(u_ori, seqs_ori, pos_ori, neg_ori, timeline_musk_ori, one_hot=False)
                    pos_labels_ori = torch.ones(pos_logits_ori.shape, device=device)

                    pos_logits_syn, neg_logits_syn = model(u_syn, seqs_syn, pos_syn, neg_syn, timeline_musk_syn)
                    pos_labels_syn = torch.ones(pos_logits_syn.shape, device=device)

                    loss_ori = criterion(pos_logits_ori[timeline_musk_ori], pos_labels_ori[timeline_musk_ori])

                    loss_syn = criterion(pos_logits_syn[timeline_musk_syn], pos_labels_syn[timeline_musk_syn])

                    grad_ori = torch.autograd.grad(loss_ori, model_parameters)
                    grad_ori = list((_.detach().clone() for _ in grad_ori))

                    grad_syn = torch.autograd.grad(loss_syn, model_parameters, create_graph=True)

                    loss += match_loss(grad_syn, grad_ori, args)

            optimizer_syn.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_([syn_data, ], max_norm=5, norm_type=2)  # 梯度裁剪
            optimizer_syn.step()

            #print("outer loop: %d, loss: %.4f" % (ol, loss.item() / syn_data.shape[0]))

            if ol == args.outer_loops - 1:
                #TODO:验证这一轮利用的模型在合成数据集上的表现
                syn_seqs_eval = {}
                data_logits_eval = copy.deepcopy(syn_data.softmax(dim=-1).log().detach().clone())
                for i in range(cluster_num * args.num_per_cluster):
                    syn_seqs_eval[i + 1] = F.gumbel_softmax(data_logits_eval[i], tau=args.tau, hard=True, dim=-1)
                syn_dataset_eval = SynSeqData(syn_seqs_eval, cluster_num, item_num, args.maxlen, args.syn_seq_length,
                                              device, True)
                syn_dataset_eval.ng_sample(sample=True)
                syn_dataloader_eval = data.DataLoader(syn_dataset_eval, batch_size=1, shuffle=False)

                print("Updating Network...")
                for il in range(args.inner_loops):
                    epoch('train', model, syn_dataloader_eval, optimizer_model, bce_criterion, args)

                NDCG_valid, HS_valid = evaluate_synset(model, syn_dataloader_eval, user_train, user_valid, user_num, item_num,
                                                       device, args, is_trained=True)
                print('eval result in this epoch: NDCG@10: %.4f, HR@10: %.4f' %(NDCG_valid, HS_valid))

                break





