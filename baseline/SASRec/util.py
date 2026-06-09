import sys
import copy
import random
import os
import numpy as np
from collections import defaultdict

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from gradproj_cdsr import GradProjCDSR


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


def data_partition_gradproj(root, data_dir, domain):
    reader = GradProjCDSR(root, data_dir)
    part = reader.single_domain_partition(domain)
    return [
        part["train"],
        part["valid"],
        part["test"],
        part["usernum"],
        part["itemnum"],
        part["valid_prefix"],
        part["test_prefix"],
    ]


def rank_metrics(rank):
    return {
        "MRR": 1.0 / (rank + 1),
        "NDCG@5": 1.0 / np.log2(rank + 2) if rank < 5 else 0.0,
        "NDCG@10": 1.0 / np.log2(rank + 2) if rank < 10 else 0.0,
        "HR@1": 1.0 if rank < 1 else 0.0,
        "HR@5": 1.0 if rank < 5 else 0.0,
        "HR@10": 1.0 if rank < 10 else 0.0,
    }


def empty_metrics():
    return {"MRR": 0.0, "NDCG@5": 0.0, "NDCG@10": 0.0, "HR@1": 0.0, "HR@5": 0.0, "HR@10": 0.0}


def format_metrics(metrics):
    return " ".join("%s=%.4f" % (k, metrics.get(k, 0.0)) for k in ["MRR", "NDCG@5", "NDCG@10", "HR@1", "HR@5", "HR@10"])


def evaluate(model, dataset, args, sess):
    train, valid, test, usernum, itemnum = copy.deepcopy(dataset[:5])
    test_prefix = dataset[6] if len(dataset) > 6 else {}

    sums = empty_metrics()
    valid_user = 0.0

    if usernum>10000:
        users = random.sample(range(1, usernum + 1), 10000)
    else:
        users = range(1, usernum + 1)
    for u in users:

        if len(train[u]) < 1 or len(test[u]) < 1: continue

        prefix = test_prefix.get(u)
        if prefix is None:
            prefix = list(train[u]) + list(valid[u])
        seq = np.zeros([args.maxlen], dtype=np.int32)
        idx = args.maxlen - 1
        for i in reversed(prefix):
            seq[idx] = i
            idx -= 1
            if idx == -1: break
        rated = set(prefix)
        rated.add(0)
        rated.add(test[u][0])
        item_idx = [test[u][0]]
        for _ in range(args.num_negatives):
            t = np.random.randint(1, itemnum + 1)
            while t in rated: t = np.random.randint(1, itemnum + 1)
            item_idx.append(t)

        predictions = -model.predict(sess, [u], [seq], item_idx)
        predictions = predictions[0]

        rank = predictions.argsort().argsort()[0]

        valid_user += 1

        for key, value in rank_metrics(rank).items():
            sums[key] += value

    if valid_user == 0:
        return empty_metrics()
    return {key: value / valid_user for key, value in sums.items()}


def evaluate_valid(model, dataset, args, sess):
    train, valid, test, usernum, itemnum = copy.deepcopy(dataset[:5])
    valid_prefix = dataset[5] if len(dataset) > 5 else {}

    sums = empty_metrics()
    valid_user = 0.0
    if usernum>10000:
        users = random.sample(range(1, usernum + 1), 10000)
    else:
        users = range(1, usernum + 1)
    for u in users:
        if len(train[u]) < 1 or len(valid[u]) < 1: continue

        prefix = valid_prefix.get(u, train[u])
        seq = np.zeros([args.maxlen], dtype=np.int32)
        idx = args.maxlen - 1
        for i in reversed(prefix):
            seq[idx] = i
            idx -= 1
            if idx == -1: break

        rated = set(prefix)
        rated.add(0)
        rated.add(valid[u][0])
        item_idx = [valid[u][0]]
        for _ in range(args.num_negatives):
            t = np.random.randint(1, itemnum + 1)
            while t in rated: t = np.random.randint(1, itemnum + 1)
            item_idx.append(t)

        predictions = -model.predict(sess, [u], [seq], item_idx)
        predictions = predictions[0]

        rank = predictions.argsort().argsort()[0]

        valid_user += 1

        for key, value in rank_metrics(rank).items():
            sums[key] += value

    if valid_user == 0:
        return empty_metrics()
    return {key: value / valid_user for key, value in sums.items()}
