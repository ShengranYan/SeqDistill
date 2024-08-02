import torch
from Models.SASRec import SASRec
from data_utils import read_data, data_partition
from argParser import args
import random
from collections import defaultdict
from modeltrain import model_train

def RandomUsers(Users, sample_num=10):
    pairs = list(Users.items())
    random.shuffle(pairs)
    return dict(pairs[:sample_num])

def RandomInteractions(dataset=None, interaction_num=0, sample_ratio=None):
    if dataset in ['ml-1m', 'wikipedia', 'Beauty']:
        f = open('data/%s.txt' % dataset, 'r')
        all_interactions = []
        Users = defaultdict(list)
        for index, line in enumerate(f):
            all_interactions.append((index, line.strip()))
        if sample_ratio is not None:
            interaction_num = int(len(all_interactions) * sample_ratio)
        sampled_indices = sorted(random.sample(range(len(all_interactions)), interaction_num))
        sampled_interactions = [all_interactions[i][1] for i in sampled_indices]
        for interaction in sampled_interactions:
            u, i = interaction.split(' ')
            u = int(u)
            i = int(i)
            Users[u].append(i)
        return Users
    else:
        raise ValueError('Unknown dataset: {}'.format(dataset))

def HeadUsers(AllUsers, sample_num=10):
    user_interaction_count = {user: len(items) for user, items in AllUsers.items()}
    sorted_users = sorted(user_interaction_count.items(), key=lambda x: x[1], reverse=True)[:sample_num]
    HeadUsers = {user: AllUsers[user] for user, _ in sorted_users}
    return HeadUsers

if __name__ == '__main__':
    sample_users = args.co_cluster_num * args.num_per_cluster # for users sample
    interaction_num = args.co_cluster_num * args.num_per_cluster * args.syn_seq_length # for interactions sample
    [Users, user_valid, user_test, user_num, item_num] = data_partition(args.dataset)
    #RandomUsers(user_train, args.dataset, 10)
    user_train = RandomInteractions(args.dataset, interaction_num, 0.2) # random interaction
    head_users = HeadUsers(Users, sample_users) # head users
    random_users = RandomUsers(Users, sample_users) # random users
    """______________________________________________model_train______________________________________________"""
    print('number of selected interactions: {}'.format(interaction_num))
    model_train(head_users, user_valid, user_test, user_num, item_num, args)