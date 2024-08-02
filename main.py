import os.path
import copy
import sys
import math
import torch
import torch.nn.functional as F
from utils import gumbel_softmax_fix, match_loss, get_time, set_seed
from data_utils import read_data, data_partition, SynSeqData, neg_sample, OriDataset
import torch.utils.data as data
from Models.SASRec import SASRec
from evaluate import *
from argParser import args
#torch.autograd.set_detect_anomaly(True) #用于检测梯度传播中的问题

if __name__ == '__main__':

    device = args.device

    set_seed()

    [user_history, Item_history, user_num, item_num] = read_data(args.dataset) #每个物品都至少出现5次
    [user_train, user_valid, user_test, user_num, item_num] = data_partition(args.dataset)

    RANDOM_SEED = 1

    train_length = len(user_train)
    num_batch = math.ceil(train_length / args.batch_size) #上取整

    print("Train set size: %d." % train_length)
    print("Dataset: %s \nUser num: %d \nItem num: %d \nModel: %s" % (args.dataset, user_num, item_num, args.model))
    print("synthetic Dataset size: %d" % num_batch)

    data_fill = neg_sample(user_train, args, item_num)
    Seq_dataset = OriDataset(data_fill, args)
    Seq_dataloader = data.DataLoader(Seq_dataset, batch_size=args.batch_size, shuffle=False)
    # N(0, 1) shape: [num_batch, syn_len, item_num + 1] 多出一个位置给padding
    #随机初始化
    syn_data = torch.randn(size=(num_batch, args.syn_seq_length, item_num + 1), requires_grad=True, dtype=torch.float32, device=device)
    torch.nn.init.xavier_uniform_(syn_data.data)

    print("%s synth data initialized, start training!" %get_time())
    """开始训练"""
    optimizer_syn = torch.optim.Adam([syn_data, ], lr=args.lr_syn_seq, betas=(0.9, 0.98))
    optimizer_syn.zero_grad()
    criterion = torch.nn.BCEWithLogitsLoss()

    for e in range(args.num_epochs):

        if e % args.epoch_num_for_eval == 0 and e != 0:
            syn_seqs_eval = {}
            data_logits_eval = copy.deepcopy(syn_data.softmax(dim=-1).log().detach().clone())
            for i in range(num_batch):
                syn_seqs_eval[i + 1] = F.gumbel_softmax(data_logits_eval[i], tau=args.tau, hard=True, dim=-1)

            sample_accuracy = 0
            for i in range(num_batch):
                for j in range(args.syn_seq_length):
                    if syn_data[i][j].argmax() == syn_seqs_eval[i + 1][j].argmax():
                        sample_accuracy += 1

            print("sample accuracy eval: ", sample_accuracy / (num_batch * args.syn_seq_length))

            syn_dataset_eval = SynSeqData(syn_seqs_eval, num_batch, item_num, args.maxlen, args.syn_seq_length, device,True)
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

        """train synthetic data"""
        model = SASRec(user_num, item_num, args).to(device)

        for name, param in model.named_parameters():
            try:
                torch.nn.init.xavier_normal_(param.data)
            except:
                pass  # just ignore those failed init layers

        model_parameters = list(model.parameters())
        optimizer_model = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.98))
        optimizer_model.zero_grad()
        bce_criterion = torch.nn.BCEWithLogitsLoss()
        loss_avg = 0.0

        for ol in range(args.outer_loops): #更新合成数据集的轮数
            loss = torch.tensor(0.0).to(args.device)
            syn_seqs = {}
            data_logits = syn_data.softmax(dim=-1).log() # 对syn_data归一化转为概率再取对数转为logits

            for i in range(num_batch):
                syn_seqs[i + 1] = F.gumbel_softmax(data_logits[i], tau=args.tau, hard=True, dim=-1)

            """检查gumbel_softmax采样的正确率"""
            if ol % 100 == 0:
                sample_accuracy = 0
                for i in range(num_batch):
                    for j in range(args.syn_seq_length):
                        if syn_data[i][j].argmax() == syn_seqs[i + 1][j].argmax():
                            sample_accuracy += 1

                print("sample_accuracy: ", sample_accuracy / (num_batch * args.syn_seq_length))

            """synth data"""
            syn_dataset = SynSeqData(syn_seqs, num_batch, item_num, args.maxlen, args.syn_seq_length, device, True)
            syn_dataset.ng_sample()
            syn_dataloader = data.DataLoader(syn_dataset, batch_size=1, shuffle=False)

            model.eval() #禁用dropout
            for _, (data_ori, data_syn) in enumerate(zip(Seq_dataloader, syn_dataloader)):
                u_ori, seqs_ori, pos_ori, neg_ori, timeline_musk_ori = data_ori
                u_syn, seqs_syn, pos_syn, neg_syn, timeline_musk_syn = data_syn
                #seqs_syn第一个物品为0
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
            torch.nn.utils.clip_grad_norm_([syn_data, ], max_norm=5, norm_type=2) #梯度裁剪
            optimizer_syn.step()
            loss_avg += loss.item()

            # print("outer loop: %d, loss: %.4f" %(ol, loss.item() / num_batch))

            if ol == args.outer_loops - 1:
                #TODO:验证这一轮利用的模型在合成数据集上的表现
                syn_seqs_eval = {}
                data_logits_eval = copy.deepcopy(syn_data.softmax(dim=-1).log().detach().clone())
                for i in range(num_batch):
                    syn_seqs_eval[i + 1] = F.gumbel_softmax(data_logits_eval[i], tau=args.tau, hard=True, dim=-1)
                syn_dataset_eval = SynSeqData(syn_seqs_eval, num_batch, item_num, args.maxlen, args.syn_seq_length,
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


        loss_avg /= (args.outer_loops * num_batch)
        print('%s epoch = %04d, loss = %.4f' % (get_time(), e, loss_avg))