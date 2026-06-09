import argparse
import datetime
import random
from time import time

import numpy as np
import torch

from dataProcessing import DataLoader
from evalMetrics import format_metrics, rank_metrics
from hgn_model import HGN


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def topk_indices(score, topk):
    score = score.detach().cpu().numpy().copy()
    ind = np.argpartition(score, -topk)
    ind = ind[:, -topk:]
    arr_ind = score[np.arange(len(score))[:, None], ind]
    arr_ind_argsort = np.argsort(arr_ind)[np.arange(len(score)), ::-1]
    return ind[np.arange(len(score))[:, None], arr_ind_argsort]


def evaluate(model, data, device, topk=10):
    model.eval()
    pred_list = []
    with torch.no_grad():
        for batch in data:
            seq, ground, position, user, neg = [x.to(device) for x in batch]
            items_to_predict = torch.cat((ground, neg), dim=1)
            score = model(seq, position, user, items_to_predict)
            pred_list.append(topk_indices(score, topk))
    if not pred_list:
        return rank_metrics([], topk=topk)
    return rank_metrics(np.concatenate(pred_list, axis=0), topk=topk)


def train_model(model, optimizer, train_data, valid_data, test_data, conf, device):
    best_valid_mrr = 0.0
    best_test = None

    for epoch_num in range(conf.n_iter):
        t1 = time()
        model.train()
        epoch_loss = 0.0
        num_batches = 0

        for batch in train_data:
            num_batches += 1
            seq, ground, position, user, neg = [x.to(device) for x in batch]
            items_to_predict = torch.cat((ground, neg), dim=1)
            score = model(seq, position, user, items_to_predict)
            pos_score, neg_score = torch.split(score, [ground.size(1), neg.size(1)], dim=1)
            loss = -torch.log(torch.sigmoid(pos_score - neg_score) + 1e-8)
            loss = torch.mean(torch.sum(loss, dim=1))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += float(loss.item())

        epoch_loss /= max(num_batches, 1)
        print("Epoch %d [%.1f s] loss=%.4f" % (epoch_num + 1, time() - t1, epoch_loss))

        if (epoch_num + 1) % conf.eval_interval == 0:
            valid = evaluate(model, valid_data, device, topk=10)
            print("epoch=%d valid[%s]" % (epoch_num + 1, format_metrics(valid)))
            if valid["MRR"] >= best_valid_mrr:
                best_valid_mrr = valid["MRR"]
                best_test = evaluate(model, test_data, device, topk=10)
                print("epoch=%d test[%s]" % (epoch_num + 1, format_metrics(best_test)))

    if best_test is not None:
        print("best_test_by_valid_MRR[%s]" % format_metrics(best_test))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Single-domain HGN using the HGNCDSR dataset protocol.")
    parser.add_argument("--L", type=int, default=15)
    parser.add_argument("--d", type=int, default=256)
    parser.add_argument("--maxlen", type=int, default=15)
    parser.add_argument("--n_iter", type=int, default=100)
    parser.add_argument("--eval_interval", type=int, default=10)
    parser.add_argument("--seed", type=int, default=2040)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--learning_rate", type=float, default=0.0005)
    parser.add_argument("--l2", type=float, default=1e-3)
    parser.add_argument("--neg_samples", type=int, default=99)
    parser.add_argument("--data_dir", type=str, default="Food-Kitchen")
    parser.add_argument("--domain", type=str, default="x", choices=["x", "y", "source", "target", "a", "b"])
    parser.add_argument("--device", type=str, default="cuda")
    config = parser.parse_args()

    set_seed(config.seed)
    device = torch.device(config.device if config.device.startswith("cuda") and torch.cuda.is_available() else "cpu")

    train_data = DataLoader(config.data_dir, config.batch_size, config, evaluation=-1, predict_domain=config.domain)
    valid_data = DataLoader(config.data_dir, config.batch_size, config, evaluation=2, predict_domain=config.domain)
    test_data = DataLoader(config.data_dir, config.batch_size, config, evaluation=1, predict_domain=config.domain)
    config.item_num = config.source_item_num + config.target_item_num + 1

    model = HGN(config, device).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate, weight_decay=config.l2)

    print("recommendation: -------------------------------------------------------------------")
    print(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    print(config)
    print(device)
    print("train_batches=%d valid_batches=%d test_batches=%d" % (len(train_data), len(valid_data), len(test_data)))

    train_model(model, optimizer, train_data, valid_data, test_data, config, device)
