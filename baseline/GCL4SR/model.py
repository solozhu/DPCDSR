import math
import random

import torch
import torch.nn as nn
import torch.nn.functional as F


class SequenceEncoder(nn.Module):
    def __init__(self, item_num, hidden_size, maxlen, num_layers=2, num_heads=1, dropout=0.2):
        super().__init__()
        self.item_embedding = nn.Embedding(item_num + 1, hidden_size, padding_idx=0)
        self.position_embedding = nn.Embedding(maxlen, hidden_size)
        layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=num_heads,
            dim_feedforward=hidden_size * 4,
            dropout=dropout,
            batch_first=True,
            activation='gelu',
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.layer_norm = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.maxlen = maxlen

    def embed_items(self, seq):
        bsz, length = seq.shape
        pos = torch.arange(length, device=seq.device).unsqueeze(0).expand(bsz, length)
        out = self.item_embedding(seq) * math.sqrt(self.item_embedding.embedding_dim)
        out = out + self.position_embedding(pos)
        out = self.dropout(out)
        return out

    def encode_embeds(self, embeds, seq):
        pad_mask = seq.eq(0)
        out = self.encoder(embeds, src_key_padding_mask=pad_mask)
        out = self.layer_norm(out)
        lengths = seq.ne(0).sum(dim=1).clamp(min=1) - 1
        return out[torch.arange(seq.size(0), device=seq.device), lengths]

    def forward(self, seq):
        embeds = self.embed_items(seq)
        return self.encode_embeds(embeds, seq)


class WITGAugmentor(nn.Module):
    """Weighted Item Transition Graph augmentation.

    For each item in a sequence, randomly samples weighted outgoing neighbors
    from WITG and aggregates item-neighbor embeddings to inject global item
    transition context, matching the paper's graph-augmented sequence view.
    """

    def __init__(self, adjacency, hidden_size, num_neighbors=5):
        super().__init__()
        self.adjacency = adjacency
        self.num_neighbors = num_neighbors
        self.proj = nn.Linear(hidden_size * 2, hidden_size)

    def sample_neighbor_tensor(self, seq):
        rows = []
        cpu_seq = seq.detach().cpu().tolist()
        for row in cpu_seq:
            out_row = []
            for item in row:
                if item == 0 or item not in self.adjacency:
                    out_row.append([0] * self.num_neighbors)
                    continue
                neigh, probs = self.adjacency[item]
                sampled = random.choices(neigh, weights=probs, k=self.num_neighbors)
                out_row.append(sampled)
            rows.append(out_row)
        return torch.tensor(rows, dtype=torch.long, device=seq.device)

    def forward(self, seq, item_embedding):
        base = item_embedding(seq)
        neigh_ids = self.sample_neighbor_tensor(seq)
        neigh_emb = item_embedding(neigh_ids).mean(dim=2)
        out = self.proj(torch.cat([base, neigh_emb], dim=-1))
        return out * seq.ne(0).unsqueeze(-1)


class GCL4SR(nn.Module):
    def __init__(self, item_num, adjacency, hidden_size=64, maxlen=50,
                 num_layers=2, num_heads=1, dropout=0.2, num_neighbors=5):
        super().__init__()
        self.encoder = SequenceEncoder(item_num, hidden_size, maxlen, num_layers, num_heads, dropout)
        self.augmentor = WITGAugmentor(adjacency, hidden_size, num_neighbors)
        self.item_bias = nn.Embedding(item_num + 1, 1, padding_idx=0)

    def local_repr(self, seq):
        return self.encoder(seq)

    def graph_repr(self, seq):
        graph_embeds = self.augmentor(seq, self.encoder.item_embedding)
        graph_embeds = graph_embeds * math.sqrt(self.encoder.item_embedding.embedding_dim)
        bsz, length = seq.shape
        pos = torch.arange(length, device=seq.device).unsqueeze(0).expand(bsz, length)
        graph_embeds = graph_embeds + self.encoder.position_embedding(pos)
        return self.encoder.encode_embeds(graph_embeds, seq)

    def score(self, user_repr, items):
        item_emb = self.encoder.item_embedding(items)
        logits = torch.sum(user_repr.unsqueeze(1) * item_emb, dim=-1)
        return logits + self.item_bias(items).squeeze(-1)

    def forward(self, seq, pos, neg):
        local = self.local_repr(seq)
        view1 = self.graph_repr(seq)
        view2 = self.graph_repr(seq)
        pos_score = self.score(local, pos)
        neg_score = self.score(local, neg)
        return pos_score, neg_score, local, view1, view2

    def contrastive_views(self, seq_a, seq_b):
        return self.local_repr(seq_a), self.graph_repr(seq_b)


def info_nce(a, b, temperature=0.2):
    a = F.normalize(a, dim=-1)
    b = F.normalize(b, dim=-1)
    logits = torch.matmul(a, b.t()) / temperature
    labels = torch.arange(a.size(0), device=a.device)
    return (F.cross_entropy(logits, labels) + F.cross_entropy(logits.t(), labels)) * 0.5
