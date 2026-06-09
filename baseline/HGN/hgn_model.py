import torch
import torch.nn as nn


class HGN(nn.Module):
    """Single-domain HGN branch adapted from the HGNCDSR source model."""

    def __init__(self, model_args, device):
        super().__init__()
        self.args = model_args

        dims = model_args.d
        item_num = model_args.item_num
        user_num = model_args.user_num
        max_len = model_args.L

        self.user_embeddings = nn.Embedding(user_num, dims).to(device)
        self.item_embeddings = nn.Embedding(item_num, dims, padding_idx=item_num - 1).to(device)
        self.position_embeddings = nn.Embedding(max_len + 1, dims, padding_idx=0).to(device)

        self.feature_gate_user = nn.Linear(dims, dims).to(device)
        self.feature_gate_item = nn.Linear(dims, dims).to(device)

        self.instance_gate_user = nn.Parameter(torch.empty(dims, max_len, device=device))
        self.instance_gate_position = nn.Parameter(torch.empty(dims, 1, device=device))
        self.instance_gate_item = nn.Parameter(torch.empty(dims, 1, device=device))

        self.W2 = nn.Embedding(item_num, dims, padding_idx=item_num - 1).to(device)
        self.b2 = nn.Embedding(item_num, 1, padding_idx=item_num - 1).to(device)

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.instance_gate_user)
        nn.init.xavier_uniform_(self.instance_gate_position)
        nn.init.xavier_uniform_(self.instance_gate_item)

        self.user_embeddings.weight.data.normal_(0, 1.0 / self.user_embeddings.embedding_dim)
        self.item_embeddings.weight.data.normal_(0, 1.0 / self.item_embeddings.embedding_dim)
        self.position_embeddings.weight.data.normal_(0, 1.0 / self.position_embeddings.embedding_dim)
        self.W2.weight.data.normal_(0, 1.0 / self.W2.embedding_dim)
        self.b2.weight.data.zero_()

    def forward(self, seq, position, user_ids, items_to_predict):
        user_embs = self.user_embeddings(user_ids)
        item_embs = self.item_embeddings(seq)
        position_embs = self.position_embeddings(position)
        w2 = self.W2(items_to_predict)

        gate = torch.sigmoid(
            self.feature_gate_item(item_embs) + self.feature_gate_user(user_embs).unsqueeze(1)
        )
        gated_item = item_embs * gate

        instance_score = torch.sigmoid(
            torch.matmul(gated_item, self.instance_gate_item.unsqueeze(0)).squeeze(-1)
            + user_embs.mm(self.instance_gate_user)
            + torch.matmul(position_embs, self.instance_gate_position.unsqueeze(0)).squeeze(-1)
        )
        instance_score = instance_score.masked_fill(seq.eq(self.args.pad_id), 0.0)
        denom = torch.sum(instance_score, dim=1, keepdim=True).clamp_min(1e-8)
        union_out = torch.sum(gated_item * instance_score.unsqueeze(2), dim=1) / denom

        union_score = torch.bmm(union_out.unsqueeze(1), w2.permute(0, 2, 1)).squeeze(1)
        rel_score = torch.mean(item_embs.bmm(w2.permute(0, 2, 1)), dim=1)
        return union_score + rel_score
