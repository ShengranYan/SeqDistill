import matplotlib.pyplot as plt
import numpy as np
from scipy.sparse import csr_matrix
from argParser import args
from data_utils import read_data
from sklearn.cluster import SpectralClustering, SpectralBiclustering


# the likehood function
def L(indexes_of_ones, fu, fi):
    import itertools
    product_ones = 1
    product_zeros = 1

    for (u, i) in itertools.product(range(fu.shape[0]), range(fi.shape[0])):
        if (u, i) in indexes_of_ones:
            product_ones *= (1 - np.exp(-np.inner(fu[u], fi[i])))
        else:
            product_zeros *= np.exp(-np.inner(fu[u], fi[i]))

    return product_ones * product_zeros

# the Penalized log likehood function
def Q(indexes_of_ones, fu, fi, lam):
    return -np.log(L(indexes_of_ones, fu, fi)) + lam * sum(np.linalg.norm(fu, axis=1)**2) + lam * sum(np.linalg.norm(fi, axis=1)**2)


# the Penalized log likehood function (from the sums)
def Q2(indexes_of_ones, fu, fi, lam):
    import itertools
    sum_ones = 0
    sum_zeros = 0

    for (u, i) in itertools.product(range(fu.shape[0]), range(fi.shape[0])):
        if (u, i) in indexes_of_ones:
            sum_ones += np.log(1 - np.exp(-np.inner(fu[u], fi[i])))
        else:
            sum_zeros += np.inner(fu[u], fi[i])

    return - sum_ones + sum_zeros + lam * sum(np.linalg.norm(fu, axis=1) ** 2) + lam * sum(
        np.linalg.norm(fi, axis=1) ** 2)


# the Penalized log likehood function (from the Qfi)
def Qfi(i, indexes_of_ones, fu, fi, lam):
    import itertools
    sum_ones = 0
    sum_zeros = 0

    for u in range(fu.shape[0]):
        if (u, i) in indexes_of_ones:
            sum_ones += np.log(1 - np.exp(-np.inner(fu[u], fi[i]) - 5e-8))
        else:
            sum_zeros += np.inner(fu[u], fi[i])

    return - sum_ones + sum_zeros + lam * np.linalg.norm(fi[i]) ** 2


def Qfu(u, indexes_of_ones, fu, fi, lam):
    import itertools
    sum_ones = 0
    sum_zeros = 0

    for i in range(fi.shape[0]):
        if (u, i) in indexes_of_ones:
            sum_ones += np.log(1 - np.exp(-np.inner(fu[u], fi[i]) - 5e-8))
        else:
            sum_zeros += np.inner(fu[u], fi[i])

    return - sum_ones + sum_zeros + lam * np.linalg.norm(fu[u]) ** 2


def Qfi_ones(item_i_history, fu, fi_i, sfu, lam):
    sum_ones = 0

    for u in item_i_history:
        inner = np.inner(fu[u], fi_i)
        sum_ones += np.log(1 - np.exp(-inner - 5e-8)) + inner

    return - sum_ones + np.inner(fi_i, sfu) + lam * np.linalg.norm(fi_i) ** 2


def Qfu_ones(user_u_history, fu_u, fi, sfi, lam):
    sum_ones = 0

    for i in user_u_history:
        inner = np.inner(fu_u, fi[i])
        sum_ones += np.log(1 - np.exp(-inner - 5e-8)) + inner

    return - sum_ones + np.inner(fu_u, sfi) + lam * np.linalg.norm(fu_u) ** 2

def Q3(indexes_of_ones, fu, fi, lam):
    q = 0

    for i in range(fi.shape[0]):
        q += Qfi(i, indexes_of_ones, fu, fi, lam)

    return q + lam * sum(np.linalg.norm(fu, axis=1) ** 2)

def DQfi_v(features, fu, fi_i, sfu, lam):
    return - (fu[features].T * (1 / (1 - np.exp(-np.dot(fu[features], fi_i) - 5e-8)))).sum(axis=1).T + sfu + 2 * lam * fi_i


def DQfu_v(features, fu_u, fi, sfi, lam):
    return - (fi[features].T * (1 / (1 - np.exp(-np.dot(fi[features], fu_u) - 5e-8)))).sum(axis=1).T + sfi + 2 * lam * fu_u


def fi_next_v(ifi_i, item_i_history, fu, sfu, lam, a):
    d = DQfi_v(item_i_history[ifi_i[0]], fu, ifi_i[1:], sfu, lam)
    return np.maximum(1e-8, ifi_i[1:] - a * d)


def fu_next_v(ufu_u, user_u_history, fi, sfi, lam, a):
    d = DQfu_v(user_u_history[ufu_u[0]], ufu_u[1:], fi, sfi, lam)
    return np.maximum(1e-8, ufu_u[1:] - a * d)



if __name__ == '__main__':

    [user_history, item_history, user_num, item_num] = read_data(args.dataset)
    print('user_num: {}, item_num: {}'.format(user_num, item_num))
    # matrix = np.zeros((user_num + 1, item_num + 1))
    # for k, v in user_history.items():
    #     for item in v:
    #         matrix[k][item] = 1
    #
    # model = SpectralClustering(n_clusters=args.co_cluster_num, random_state=0)
    # model.fit(matrix)
    #
    # labels = model.labels_
    # label_cnt = np.zeros((args.co_cluster_num, 1))
    # for i in range(1, user_num + 1):
    #     label_cnt[labels[i]] += 1
    # for i in range(len(label_cnt)):
    #     print(i, ': ', label_cnt[i])
    # print(labels)
    #np.save('labels.npy', labels)

    # fu_copy = np.load('fu.npy')
    # fu_max_min = np.amax(fu_copy, axis=1).min()
    # fu_stat = np.sum(fu_copy >= fu_max_min, axis=-1)
    # initialize the parameters and the fs

    rnd = np.random.RandomState(seed=123456789)
    k = args.co_cluster_num # co cluster类别的数量
    lam = 600
    max_it = 20
    fu = rnd.rand(user_num + 1, k)
    fi = rnd.rand(item_num + 1, k)
    sfu = np.sum(fu, axis=0)
    sfi = np.sum(fi, axis=0)
    sigma = 0.01
    lsParam = 5e-3
    M_rec = 130


    for it in range(max_it):
        # items
        for i in range(1, fi.shape[0]):
            a = 1
            active = True
            d = DQfi_v(item_history[i], fu, fi[i], sfu, lam)
            old_cost = Qfi_ones(item_history[i], fu, fi[i], sfu, lam)
            old_fi = fi[i, :].copy()

            while active:
                fi_new = fi_next_v(np.append([i], old_fi), item_history, fu, sfu, lam, a)
                fi[i, :] = fi_new
                RHS = sigma * np.inner(d, (fi_new - old_fi))
                new_cost = Qfi_ones(item_history[i], fu, fi[i], sfu, lam)
                active = new_cost - old_cost > RHS + lsParam
                a *= 0.1

        sfi = np.sum(fi, axis=0)

        # users
        for u in range(1, fu.shape[0]):
            a = 1
            active = True
            d = DQfu_v(user_history[u], fu[u], fi, sfi, lam)
            old_cost = Qfu_ones(user_history[u], fu[u], fi, sfi, lam)
            old_fu = fu[u, :].copy()

            while active:
                fu_new = fu_next_v(np.append([u], old_fu), user_history, fi, sfi, lam, a)
                fu[u, :] = fu_new
                RHS = sigma * np.inner(d, (fu_new - old_fu))
                new_cost = Qfu_ones(user_history[u], fu[u], fi, sfi, lam)
                active = new_cost - old_cost > RHS + lsParam
                a *= 0.1

        sfu = np.sum(fu, axis=0)

    # Computes the probabilities
    np.save(f'fu_{args.dataset}_{args.co_cluster_num}.npy', fu)
    np.save(f'fi_{args.dataset}_{args.co_cluster_num}.npy', fi)
    #
    # fu = np.load(f'fu_{args.co_cluster_num}.npy')

    # print(np.sum(fu > 0.5, axis=-1))
    # print(np.sum(fi > 0.5, axis=-1))
    # prob = 1 - np.exp(-np.dot(fu, fi.T))
    # labels_ = np.zeros((user_num + 1, 1))
    # for i in range(1, user_num + 1):
    #     labels_[i] = fu[i].argmax()
    # labels_cnt = np.zeros((args.co_cluster_num, 1))
    # for i in range(len(labels_)):
    #     labels_cnt[labels_[i + 1]] += 1
    # print(labels_cnt)


