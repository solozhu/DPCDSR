import math
import numpy as np


def rank_metrics(predicted, topk=10):
    """The ground-truth item is placed at candidate index 0."""
    count = len(predicted)
    if count == 0:
        return {"MRR": 0.0, "NDCG@5": 0.0, "NDCG@10": 0.0, "HR@1": 0.0, "HR@5": 0.0, "HR@10": 0.0}

    mrr = 0.0
    ndcg5 = 0.0
    ndcg10 = 0.0
    hr1 = 0.0
    hr5 = 0.0
    hr10 = 0.0

    for row in predicted:
        row = np.asarray(row)
        pos = np.where(row == 0)[0]
        if len(pos) == 0:
            continue
        rank = int(pos[0])
        mrr += 1.0 / (rank + 1)
        if rank < 1:
            hr1 += 1.0
        if rank < 5:
            hr5 += 1.0
            ndcg5 += 1.0 / math.log2(rank + 2)
        if rank < topk:
            hr10 += 1.0
            ndcg10 += 1.0 / math.log2(rank + 2)

    return {
        "MRR": float(mrr / count),
        "NDCG@5": float(ndcg5 / count),
        "NDCG@10": float(ndcg10 / count),
        "HR@1": float(hr1 / count),
        "HR@5": float(hr5 / count),
        "HR@10": float(hr10 / count),
    }


def format_metrics(metrics):
    keys = ["MRR", "NDCG@5", "NDCG@10", "HR@1", "HR@5", "HR@10"]
    return " ".join("%s=%.4f" % (key, metrics[key]) for key in keys)
