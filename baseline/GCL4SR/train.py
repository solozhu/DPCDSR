import argparse
import math
import os
import random
import sys
from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from gradproj_cdsr import GradProjCDSR, pad_left, sample_negative  # noqa: E402
from model import GCL4SR, info_nce  # noqa: E402


class SequenceTrainDataset(Dataset):
    def __init__(self, train, itemnum, maxlen, num_train_negatives=20,
                 crop_ratio=0.8, mask_ratio=0.2, reorder_ratio=0.2):
        self.train = train
        self.itemnum = itemnum
        self.maxlen = maxlen
        self.num_train_negatives = num_train_negatives
        self.crop_ratio = crop_ratio
        self.mask_ratio = mask_ratio
        self.reorder_ratio = reorder_ratio
        self.samples = []
        self.seen = {}
        for uid, seq in train.items():
            seq = list(seq)
            self.seen[uid] = set(seq)
            for idx in range(1, len(seq)):
                self.samples.append((uid, seq[:idx], seq[idx]))

    def __len__(self):
        return len(self.samples)

    def augment(self, prefix):
        seq = list(prefix)
        if len(seq) > 2 and random.random() < 0.5:
            crop_len = max(1, int(len(seq) * self.crop_ratio))
            start = random.randint(0, len(seq) - crop_len)
            seq = seq[start:start + crop_len]
        if len(seq) > 2 and random.random() < 0.5:
            reorder_len = max(2, int(len(seq) * self.reorder_ratio))
            reorder_len = min(reorder_len, len(seq))
            start = random.randint(0, len(seq) - reorder_len)
            span = seq[start:start + reorder_len]
            random.shuffle(span)
            seq = seq[:start] + span + seq[start + reorder_len:]
        if self.mask_ratio > 0:
            seq = [0 if random.random() < self.mask_ratio else item for item in seq]
        return pad_left(seq, self.maxlen)

    def __getitem__(self, index):
        uid, prefix, pos = self.samples[index]
        seen = self.seen.get(uid, set())
        neg = [sample_negative(self.itemnum, seen) for _ in range(self.num_train_negatives)]
        return (
            torch.LongTensor(pad_left(prefix, self.maxlen)),
            torch.LongTensor(self.augment(prefix)),
            torch.LongTensor(self.augment(prefix)),
            torch.LongTensor([pos]),
            torch.LongTensor(neg),
        )


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_witg(train):
    """Build the weighted item transition graph used by GCL4SR."""
    counts = defaultdict(lambda: defaultdict(float))
    for seq in train.values():
        seq = [x for x in seq if x > 0]
        for src, dst in zip(seq[:-1], seq[1:]):
            counts[src][dst] += 1.0
    adjacency = {}
    for src, dst_counts in counts.items():
        neighbors = list(dst_counts.keys())
        weights = list(dst_counts.values())
        total = sum(weights)
        adjacency[src] = (neighbors, [w / total for w in weights])
    return adjacency


def collect_seen(train, records):
    seen = {uid: set(seq) for uid, seq in train.items()}
    prefixes = {}
    for rec in records:
        uid = rec["user"]
        prefixes[uid] = list(rec["prefix"])
        seen.setdefault(uid, set()).update(rec["prefix"])
        seen[uid].add(rec["target"])
    return seen, prefixes


def rank_metrics(rank):
    return {
        "HR@1": 1.0 if rank < 1 else 0.0,
        "HR@5": 1.0 if rank < 5 else 0.0,
        "HR@10": 1.0 if rank < 10 else 0.0,
        "NDCG@5": 1.0 / math.log2(rank + 2) if rank < 5 else 0.0,
        "NDCG@10": 1.0 / math.log2(rank + 2) if rank < 10 else 0.0,
        "MRR": 1.0 / (rank + 1),
    }


@torch.no_grad()
def evaluate(model, records, train, itemnum, maxlen, device, num_negatives=999):
    model.eval()
    seen, _ = collect_seen(train, records)
    sums = defaultdict(float)
    count = 0
    rng = random.Random(2026)
    for rec in records:
        uid = rec["user"]
        prefix = rec["prefix"]
        target = rec["target"]
        if not prefix:
            continue
        sampled = set()
        forbidden = set(seen.get(uid, set()))
        forbidden.add(target)
        while len(sampled) < min(num_negatives, itemnum - len(forbidden)):
            neg = rng.randint(1, itemnum)
            if neg not in forbidden:
                sampled.add(neg)
        candidates = [target] + list(sampled)
        seq = torch.LongTensor([pad_left(prefix, maxlen)]).to(device)
        items = torch.LongTensor([candidates]).to(device)
        user_repr = model.local_repr(seq)
        scores = model.score(user_repr, items).squeeze(0)
        rank = int((scores[1:] > scores[0]).sum().item())
        for key, value in rank_metrics(rank).items():
            sums[key] += value
        count += 1
    if count == 0:
        return {key: 0.0 for key in ["HR@1", "HR@5", "HR@10", "NDCG@5", "NDCG@10", "MRR"]}
    return {key: value / count for key, value in sums.items()}


def format_metrics(metrics):
    keys = ["MRR", "NDCG@5", "NDCG@10", "HR@1", "HR@5", "HR@10"]
    return " ".join("%s=%.4f" % (key, metrics.get(key, 0.0)) for key in keys)


def train(args):
    set_seed(args.seed)
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        try:
            torch.set_float32_matmul_precision("high")
        except Exception:
            pass
    root = os.path.abspath(args.gradproj_root)
    result_dir = args.result_dir
    if result_dir is None:
        result_dir = os.path.join(args.result_root, args.data_dir, "GCL4SR", args.domain)
    os.makedirs(result_dir, exist_ok=True)
    log_path = os.path.join(result_dir, "run.txt")

    def log(message):
        print(message)
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(message + "\n")

    with open(os.path.join(result_dir, "args.txt"), "w", encoding="utf-8") as f:
        for key, value in sorted(vars(args).items()):
            f.write("%s,%s\n" % (key, value))

    reader = GradProjCDSR(root, args.data_dir)
    part = reader.single_domain_partition(args.domain)
    train_data = part["train"]
    itemnum = part["itemnum"]
    train_set = SequenceTrainDataset(
        train_data,
        itemnum,
        args.maxlen,
        num_train_negatives=args.num_train_negatives,
        crop_ratio=args.crop_ratio,
        mask_ratio=args.mask_ratio,
        reorder_ratio=args.reorder_ratio,
    )
    if len(train_set) == 0:
        raise RuntimeError("no training samples were built for domain %s" % args.domain)

    adjacency = build_witg(train_data)
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    model = GCL4SR(
        itemnum,
        adjacency,
        hidden_size=args.hidden_size,
        maxlen=args.maxlen,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        dropout=args.dropout,
        num_neighbors=args.num_neighbors,
    ).to(device)
    pin_memory = device.type == "cuda"
    loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
        persistent_workers=args.num_workers > 0,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    best_valid = -1.0
    best_test = None
    best_epoch = None
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        for seq, seq_aug1, seq_aug2, pos, neg in loader:
            seq = seq.to(device, non_blocking=True)
            seq_aug1 = seq_aug1.to(device, non_blocking=True)
            seq_aug2 = seq_aug2.to(device, non_blocking=True)
            pos = pos.to(device, non_blocking=True)
            neg = neg.to(device, non_blocking=True)
            pos_score, neg_score, local, view1, view2 = model(seq, pos, neg)
            rec_loss = -F.logsigmoid(pos_score).mean() - F.logsigmoid(-neg_score).mean()
            aug_view1, aug_view2 = model.contrastive_views(seq_aug1, seq_aug2)
            graph_loss = info_nce(view1, view2, args.temperature)
            seq_graph_loss = info_nce(local, view1, args.temperature)
            aug_loss = info_nce(aug_view1, aug_view2, args.temperature)
            loss = rec_loss + args.graph_weight * graph_loss + args.cl_weight * (seq_graph_loss + aug_loss)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad)
            optimizer.step()
            total_loss += float(loss.item())

        avg_loss = total_loss / max(len(loader), 1)
        log("epoch=%d loss=%.4f" % (epoch, avg_loss))
        if epoch % args.eval_interval == 0:
            valid = evaluate(model, part["valid_records"], train_data, itemnum, args.maxlen, device, args.num_negatives)
            score = valid["MRR"]
            valid_line = "epoch=%d valid[%s]" % (epoch, format_metrics(valid))
            if score > best_valid:
                best_valid = score
                best_epoch = epoch
                test = evaluate(model, part["test_records"], train_data, itemnum, args.maxlen, device, args.num_negatives)
                best_test = test
                if args.save_path:
                    torch.save({"args": vars(args), "model": model.state_dict(), "itemnum": itemnum}, args.save_path)
                log(valid_line)
                log("epoch=%d test[%s]" % (epoch, format_metrics(test)))
            else:
                log(valid_line)

    if best_test is not None:
        log("best_epoch=%s best_test_by_valid_MRR[%s]" % (best_epoch, format_metrics(best_test)))


def parse_args():
    default_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "DiffCDSR1", "dataset"))
    parser = argparse.ArgumentParser()
    parser.add_argument("--gradproj_root", default=default_root)
    parser.add_argument("--data_dir", default="Food-Kitchen")
    parser.add_argument("--domain", default="x", choices=["x", "y", "a", "b", "source", "target"])
    parser.add_argument("--maxlen", type=int, default=15)
    parser.add_argument("--hidden_size", type=int, default=256)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--num_heads", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--num_neighbors", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--cl_weight", type=float, default=0.2)
    parser.add_argument("--graph_weight", type=float, default=0.2)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--num_train_negatives", type=int, default=20)
    parser.add_argument("--crop_ratio", type=float, default=0.8)
    parser.add_argument("--mask_ratio", type=float, default=0.2)
    parser.add_argument("--reorder_ratio", type=float, default=0.2)
    parser.add_argument("--num_negatives", type=int, default=999)
    parser.add_argument("--eval_interval", type=int, default=10)
    parser.add_argument("--clip_grad", type=float, default=5.0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--save_path", default="")
    parser.add_argument("--result_root", default=os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "results")))
    parser.add_argument("--result_dir", default=None)
    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())
