
import os
import math
import pickle
import random
import codecs
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import numpy as np
import torch
from torch.utils.data import Dataset


@dataclass
class SplitRecord:
    user: int
    prefix_items: List[int]
    prefix_times: List[int]
    target: int
    target_domain: int


@dataclass
class UserHistory:
    user: int
    item_ids: List[int]
    timestamps: List[int]


def set_random_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch.backends, 'cudnn'):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


class UserHistoryDataset(Dataset):
    def __init__(self, histories: List[UserHistory], item_num: int,
                 w_min: float, w_max: float,
                 reweight_version: str = 'ExpDecay',
                 exp_beta: float = 3.0,
                 recent_k: int = 30,
                 mode: str = 'long'):
        self.histories = histories
        self.item_num = item_num
        self.w_min = w_min
        self.w_max = w_max
        self.reweight_version = reweight_version
        self.exp_beta = exp_beta
        self.recent_k = recent_k
        self.mode = mode

    def __len__(self):
        return len(self.histories)

    def __getitem__(self, idx):
        hist = self.histories[idx]
        if self.mode == 'recent':
            item_ids = hist.item_ids[-self.recent_k:]
            timestamps = hist.timestamps[-self.recent_k:]
        else:
            item_ids = hist.item_ids
            timestamps = hist.timestamps
        vec = build_weighted_item_vector(
            item_ids=item_ids,
            timestamps=timestamps,
            item_num=self.item_num,
            w_min=self.w_min,
            w_max=self.w_max,
            reweight_version=self.reweight_version,
            exp_beta=self.exp_beta,
        )
        return torch.from_numpy(vec), torch.tensor(hist.user, dtype=torch.long)


class RecordInferDataset(Dataset):
    def __init__(self, records: List[SplitRecord], item_num: int,
                 w_min: float, w_max: float,
                 reweight_version: str = 'ExpDecay',
                 exp_beta: float = 3.0):
        self.records = records
        self.item_num = item_num
        self.w_min = w_min
        self.w_max = w_max
        self.reweight_version = reweight_version
        self.exp_beta = exp_beta

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        rec = self.records[idx]
        vec = build_weighted_item_vector(
            rec.prefix_items,
            rec.prefix_times,
            self.item_num,
            self.w_min,
            self.w_max,
            self.reweight_version,
            self.exp_beta,
        )
        return (
            torch.from_numpy(vec),
            torch.tensor(rec.user, dtype=torch.long),
            torch.tensor(rec.target, dtype=torch.long),
            torch.tensor(rec.target_domain, dtype=torch.long),
        )


class UserHistoryInferDataset(Dataset):
    def __init__(self, histories: List[UserHistory], item_num: int,
                 w_min: float, w_max: float,
                 reweight_version: str = 'ExpDecay',
                 exp_beta: float = 3.0,
                 recent_k: int = 30,
                 mode: str = 'long'):
        self.histories = histories
        self.item_num = item_num
        self.w_min = w_min
        self.w_max = w_max
        self.reweight_version = reweight_version
        self.exp_beta = exp_beta
        self.recent_k = recent_k
        self.mode = mode

    def __len__(self):
        return len(self.histories)

    def __getitem__(self, idx):
        hist = self.histories[idx]
        if self.mode == 'recent':
            item_ids = hist.item_ids[-self.recent_k:]
            timestamps = hist.timestamps[-self.recent_k:]
        else:
            item_ids = hist.item_ids
            timestamps = hist.timestamps
        vec = build_weighted_item_vector(
            item_ids=item_ids,
            timestamps=timestamps,
            item_num=self.item_num,
            w_min=self.w_min,
            w_max=self.w_max,
            reweight_version=self.reweight_version,
            exp_beta=self.exp_beta,
        )
        return torch.from_numpy(vec), torch.tensor(hist.user, dtype=torch.long)


def _read_count_lines(fname: str) -> int:
    count = 0
    with codecs.open(fname, 'r', encoding='utf-8') as fr:
        for _ in fr:
            count += 1
    return count


def get_dataset_meta(data_dir: str) -> Dict[str, int]:
    root = os.path.join('./dataset', data_dir)
    source_item_num = _read_count_lines(os.path.join(root, 'Alist.txt'))
    target_item_num = _read_count_lines(os.path.join(root, 'Blist.txt'))
    user_num = _read_count_lines(os.path.join(root, 'userlist.txt'))
    item_num = source_item_num + target_item_num
    return {
        'source_item_num': source_item_num,
        'target_item_num': target_item_num,
        'user_num': user_num,
        'item_num': item_num,
        'pad_item_id': item_num,
    }


def _parse_interaction_token(token: str, fallback_ts: int) -> Tuple[int, int]:
    fields = token.split('|')
    item_id = int(fields[0])
    ts = None
    for field in reversed(fields[1:]):
        try:
            ts = int(field)
            break
        except Exception:
            continue
    if ts is None:
        ts = fallback_ts
    return item_id, ts


def _parse_line(line: str) -> Tuple[int, List[Tuple[int, int]]]:
    parts = line.strip().split('\t')
    user = int(parts[0])
    pairs: List[Tuple[int, int]] = []
    for idx, token in enumerate(parts[2:]):
        item_id, ts = _parse_interaction_token(token, fallback_ts=idx)
        pairs.append((item_id, ts))
    pairs.sort(key=lambda x: (x[1], x[0]))
    return user, pairs


def read_split_records(file_path: str, source_item_num: int) -> List[SplitRecord]:
    records: List[SplitRecord] = []
    with codecs.open(file_path, 'r', encoding='utf-8') as infile:
        for line in infile:
            user, pairs = _parse_line(line)
            if len(pairs) < 2:
                continue
            prefix = pairs[:-1]
            target_item = pairs[-1][0]
            prefix_items = [x[0] for x in prefix]
            prefix_times = [x[1] for x in prefix]
            target_domain = 1 if target_item >= source_item_num else 0
            records.append(
                SplitRecord(
                    user=user,
                    prefix_items=prefix_items,
                    prefix_times=prefix_times,
                    target=target_item,
                    target_domain=target_domain,
                )
            )
    return records


def aggregate_train_histories(data_dir: str) -> Tuple[Dict[str, int], List[UserHistory], List[SplitRecord], List[SplitRecord]]:
    meta = get_dataset_meta(data_dir)
    root = os.path.join('./dataset', data_dir)
    train_path = os.path.join(root, 'traindata_new.txt')
    valid_path = os.path.join(root, 'validdata_new2.txt')
    test_path = os.path.join(root, 'testdata_new2.txt')

    valid_records = read_split_records(valid_path, meta['source_item_num'])
    test_records = read_split_records(test_path, meta['source_item_num'])

    raw_per_user: List[List[Tuple[int, int]]] = [[] for _ in range(meta['user_num'])]
    with codecs.open(train_path, 'r', encoding='utf-8') as infile:
        for line in infile:
            user, pairs = _parse_line(line)
            if user < 0 or user >= meta['user_num']:
                continue
            raw_per_user[user].extend(pairs)

    histories: List[UserHistory] = []
    for user in range(meta['user_num']):
        pairs = raw_per_user[user]
        if not pairs:
            histories.append(UserHistory(user=user, item_ids=[], timestamps=[]))
            continue
        pairs = sorted(set(pairs), key=lambda x: (x[1], x[0]))
        item_ids = [p[0] for p in pairs]
        timestamps = [p[1] for p in pairs]
        histories.append(UserHistory(user=user, item_ids=item_ids, timestamps=timestamps))

    return meta, histories, valid_records, test_records


def scale_with_time(values: List[int], min_value: float, max_value: float,
                    reweight_version: str = 'ExpDecay', exp_beta: float = 3.0) -> np.ndarray:
    n = len(values)
    if n == 0:
        return np.asarray([], dtype=np.float32)
    if n == 1:
        return np.asarray([max_value], dtype=np.float32)

    if reweight_version == 'AllOne':
        return np.ones(n, dtype=np.float32) * max_value
    if reweight_version == 'AllLinear':
        return np.linspace(min_value, max_value, n, dtype=np.float32)
    if reweight_version == 'MinMax':
        arr = np.asarray(values, dtype=np.float32)
        lo = float(arr.min())
        hi = float(arr.max())
        if hi == lo:
            return np.ones(n, dtype=np.float32) * max_value
        return (min_value + (arr - lo) * (max_value - min_value) / (hi - lo)).astype(np.float32)

    ranks = np.linspace(0.0, 1.0, n, dtype=np.float32)
    if exp_beta <= 1e-8:
        curve = ranks
    else:
        curve = (np.exp(exp_beta * ranks) - 1.0) / (math.exp(exp_beta) - 1.0)
    weights = min_value + (max_value - min_value) * curve
    return weights.astype(np.float32)


def build_weighted_item_vector(item_ids: List[int], timestamps: List[int], item_num: int,
                               w_min: float, w_max: float,
                               reweight_version: str = 'ExpDecay',
                               exp_beta: float = 3.0) -> np.ndarray:
    vec = np.zeros([item_num], dtype=np.float32)
    if len(item_ids) == 0:
        return vec
    weights = scale_with_time(timestamps, w_min, w_max, reweight_version, exp_beta)
    for item_id, weight in zip(item_ids, weights):
        if 0 <= item_id < item_num:
            vec[item_id] = max(vec[item_id], float(weight))
    return vec


@torch.no_grad()
def infer_user_priors(model, diffusion, dataset, device: torch.device,
                      batch_size: int, sampling_steps: int,
                      sampling_noise: bool) -> Tuple[np.ndarray, np.ndarray]:
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)
    all_priors: List[np.ndarray] = []
    all_users: List[np.ndarray] = []
    for batch in loader:
        x_start = batch[0].to(device=device, dtype=torch.float32)
        users = batch[1].cpu().numpy()
        preds = diffusion.p_sample(model, x_start, sampling_steps, sampling_noise).detach().cpu().numpy().astype(np.float32)
        all_priors.append(preds)
        all_users.append(users)
    priors = np.concatenate(all_priors, axis=0) if all_priors else np.zeros((0, dataset.item_num), dtype=np.float32)
    users = np.concatenate(all_users, axis=0) if all_users else np.zeros((0,), dtype=np.int64)
    return priors, users


@torch.no_grad()
def infer_record_priors(model, diffusion, dataset, device: torch.device,
                        batch_size: int, sampling_steps: int,
                        sampling_noise: bool):
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)
    priors, users, targets, domains = [], [], [], []
    for batch in loader:
        x_start = batch[0].to(device=device, dtype=torch.float32)
        pred = diffusion.p_sample(model, x_start, sampling_steps, sampling_noise).detach().cpu().numpy().astype(np.float32)
        priors.append(pred)
        users.append(batch[1].cpu().numpy())
        targets.append(batch[2].cpu().numpy())
        domains.append(batch[3].cpu().numpy())
    return (
        np.concatenate(priors, axis=0) if priors else np.zeros((0, dataset.item_num), dtype=np.float32),
        np.concatenate(users, axis=0) if users else np.zeros((0,), dtype=np.int64),
        np.concatenate(targets, axis=0) if targets else np.zeros((0,), dtype=np.int64),
        np.concatenate(domains, axis=0) if domains else np.zeros((0,), dtype=np.int64),
    )


def mix_priors(long_prior_matrix: np.ndarray, recent_prior_matrix: np.ndarray, eta: float) -> np.ndarray:
    eta = float(max(0.0, min(1.0, eta)))
    return ((1.0 - eta) * long_prior_matrix + eta * recent_prior_matrix).astype(np.float32)


def sample_negative_items_same_domain(target: int, domain: int,
                                      source_item_num: int, target_item_num: int,
                                      n_samples: int, rng: np.random.RandomState) -> np.ndarray:
    negatives: List[int] = []
    if domain == 1:
        low = source_item_num
        high = source_item_num + target_item_num
    else:
        low = 0
        high = source_item_num
    while len(negatives) < n_samples:
        sample = int(rng.randint(low, high))
        if sample != target:
            negatives.append(sample)
    return np.asarray(negatives, dtype=np.int64)


def ranking_metrics_for_candidates(candidate_scores: np.ndarray, pos_index: int = 0) -> Dict[str, float]:
    order = np.argsort(-candidate_scores)
    rank = int(np.where(order == pos_index)[0][0])
    return {
        'MRR': 1.0 / (rank + 1),
        'HR@1': 1.0 if rank < 1 else 0.0,
        'HR@5': 1.0 if rank < 5 else 0.0,
        'HR@10': 1.0 if rank < 10 else 0.0,
        'NDCG@5': (1.0 / math.log2(rank + 2)) if rank < 5 else 0.0,
        'NDCG@10': (1.0 / math.log2(rank + 2)) if rank < 10 else 0.0,
    }


def average_metric_dict(metric_dicts: List[Dict[str, float]]) -> Dict[str, float]:
    if not metric_dicts:
        return {}
    keys = metric_dicts[0].keys()
    return {k: float(np.mean([m[k] for m in metric_dicts])) for k in keys}


def pretty_metric_line(name: str, metrics: Dict[str, float]) -> str:
    if not metrics:
        return f'{name}: empty'
    ordered = ['MRR', 'NDCG@5', 'NDCG@10', 'HR@1', 'HR@5', 'HR@10']
    return f"{name}: " + ', '.join([f'{k}={metrics[k]:.6f}' for k in ordered if k in metrics])


def save_pickle(obj, path: str):
    with open(path, 'wb') as f:
        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)


def load_pickle(path: str):
    with open(path, 'rb') as f:
        return pickle.load(f)


def compute_prior_domain_stats(prior_matrix: np.ndarray, source_item_num: int) -> Dict[str, np.ndarray]:
    x = prior_matrix[:, :source_item_num]
    y = prior_matrix[:, source_item_num:]
    x_mean = x.mean(axis=1).astype(np.float32)
    y_mean = y.mean(axis=1).astype(np.float32)
    x_std = (x.std(axis=1) + 1e-6).astype(np.float32)
    y_std = (y.std(axis=1) + 1e-6).astype(np.float32)
    return {
        'x_mean': x_mean,
        'x_std': x_std,
        'y_mean': y_mean,
        'y_std': y_std,
    }


def build_top_unseen_pools(prior_matrix: np.ndarray,
                           histories: List[UserHistory],
                           source_item_num: int,
                           target_item_num: int,
                           topk_x: int = 200,
                           topk_y: int = 200) -> Dict[str, np.ndarray]:
    user_num, item_num = prior_matrix.shape
    pools_x = np.full((user_num, topk_x), -1, dtype=np.int64)
    pools_y = np.full((user_num, topk_y), -1, dtype=np.int64)
    x_domain = np.arange(0, source_item_num, dtype=np.int64)
    y_domain = np.arange(source_item_num, source_item_num + target_item_num, dtype=np.int64)

    for hist in histories:
        u = hist.user
        row = prior_matrix[u]
        seen = np.zeros(item_num, dtype=bool)
        if hist.item_ids:
            seen[np.asarray(hist.item_ids, dtype=np.int64)] = True

        x_scores = row[:source_item_num].copy()
        x_scores[seen[:source_item_num]] = -1e30
        take_x = min(topk_x, source_item_num)
        if take_x > 0:
            idx = np.argpartition(-x_scores, take_x - 1)[:take_x]
            idx = idx[np.argsort(-x_scores[idx])]
            pools_x[u, :len(idx)] = idx.astype(np.int64)

        y_scores = row[source_item_num:source_item_num + target_item_num].copy()
        y_seen = seen[source_item_num:source_item_num + target_item_num]
        y_scores[y_seen] = -1e30
        take_y = min(topk_y, target_item_num)
        if take_y > 0:
            idx = np.argpartition(-y_scores, take_y - 1)[:take_y]
            idx = idx[np.argsort(-y_scores[idx])]
            pools_y[u, :len(idx)] = (idx + source_item_num).astype(np.int64)

    return {
        'top_unseen_x': pools_x,
        'top_unseen_y': pools_y,
    }


def evaluate_prior_matrix(prior_matrix: np.ndarray,
                          records: List[SplitRecord],
                          meta: Dict[str, int],
                          negatives: int,
                          seed: int) -> Dict[str, float]:
    rng = np.random.RandomState(seed)
    metric_list: List[Dict[str, float]] = []
    for rec in records:
        row = prior_matrix[rec.user]
        target = int(rec.target)
        domain = int(rec.target_domain)
        negatives_arr = sample_negative_items_same_domain(
            target, domain, meta['source_item_num'], meta['target_item_num'], negatives, rng
        )
        candidates = np.concatenate([[target], negatives_arr], axis=0)
        scores = row[candidates].astype(np.float64)
        metric_list.append(ranking_metrics_for_candidates(scores, pos_index=0))
    return average_metric_dict(metric_list)
