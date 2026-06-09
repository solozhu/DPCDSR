import math
from typing import Dict, Optional, Set

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class HGN_CDR(nn.Module):
    """HGNCDSR backbone with two strictly separated diffusion paths.

    Gate path: keeps the original feature-only diffusion gate from the source project.
    Score path: keeps the uploaded score adapter/calibration module.

    This separation is intentional: with only --use_diffusion_gate, the gate forward path is
    the original feature-gate path; with only --use_diffusion_score, score fusion uses the
    uploaded candidate-level score adapter.
    """

    def __init__(self, model_args, device):
        super().__init__()
        self.args = model_args
        self.device = device

        self.use_diffusion_gate = bool(getattr(self.args, 'use_diffusion_gate', False))
        self.use_diffusion_score = bool(getattr(self.args, 'use_diffusion_score', False))

        L = self.args.L
        dims = self.args.d
        item_num = self.args.item_num
        user_num = self.args.user_num
        self.pad_item_id = item_num - 1

        self.user_embeddings = nn.Embedding(user_num, dims).to(device)
        self.item_embeddings = nn.Embedding(item_num, dims, padding_idx=item_num - 1).to(device)
        self.position_embeddings = nn.Embedding(L + 1, dims, padding_idx=0).to(device)

        self.feature_gate_user = nn.Linear(dims, dims).to(device)
        self.feature_gate_item_src = nn.Linear(dims, dims).to(device)
        self.feature_gate_item_tgt = nn.Linear(dims, dims).to(device)
        self.feature_gate_item = nn.Linear(dims, dims).to(device)

        preserve_score_only_rng = self.use_diffusion_score and not self.use_diffusion_gate
        if preserve_score_only_rng:
            cpu_rng_state = torch.get_rng_state()
        self.diff_feature_proj_src = nn.Linear(1, dims).to(device)
        self.diff_feature_proj_tgt = nn.Linear(1, dims).to(device)
        self.diff_feature_proj = nn.Linear(1, dims).to(device)
        nn.init.zeros_(self.diff_feature_proj_src.weight)
        nn.init.zeros_(self.diff_feature_proj_src.bias)
        nn.init.zeros_(self.diff_feature_proj_tgt.weight)
        nn.init.zeros_(self.diff_feature_proj_tgt.bias)
        nn.init.zeros_(self.diff_feature_proj.weight)
        nn.init.zeros_(self.diff_feature_proj.bias)
        if preserve_score_only_rng:
            torch.set_rng_state(cpu_rng_state)

        self.instance_gate_user = Variable(torch.zeros(dims, L).type(torch.FloatTensor), requires_grad=True).to(device)
        self.instance_gate_position = Variable(torch.zeros(dims, 1).type(torch.FloatTensor), requires_grad=True).to(device)
        self.instance_gate_item_src = Variable(torch.zeros(dims, 1).type(torch.FloatTensor), requires_grad=True).to(device)
        self.instance_gate_item_tgt = Variable(torch.zeros(dims, 1).type(torch.FloatTensor), requires_grad=True).to(device)
        self.instance_gate_item = Variable(torch.zeros(dims, 1).type(torch.FloatTensor), requires_grad=True).to(device)
        self.instance_gate_user = torch.nn.init.xavier_uniform_(self.instance_gate_user)
        self.instance_gate_item_src = torch.nn.init.xavier_uniform_(self.instance_gate_item_src)
        self.instance_gate_item_tgt = torch.nn.init.xavier_uniform_(self.instance_gate_item_tgt)
        self.instance_gate_item = torch.nn.init.xavier_uniform_(self.instance_gate_item)
        self.instance_gate_position = torch.nn.init.xavier_uniform_(self.instance_gate_position)

        self._mix_floor = 1e-4
        self._alpha_floor = 1e-4
        if self.use_diffusion_gate:
            self.plugin_feature_mix_raw = nn.Parameter(self._inv_bounded_sigmoid(
                float(getattr(self.args, 'diffusion_feature_mix_init', 0.08)), self._mix_floor, 0.30
            ))
            self.plugin_feature_seq_raw = nn.Parameter(self._inv_positive_softplus(
                float(getattr(self.args, 'diffusion_feature_init_seq', 0.03)), self._alpha_floor
            ))
            self.plugin_feature_x_raw = nn.Parameter(self._inv_positive_softplus(
                float(getattr(self.args, 'diffusion_feature_init_x', 0.05)), self._alpha_floor
            ))
            self.plugin_feature_y_raw = nn.Parameter(self._inv_positive_softplus(
                float(getattr(self.args, 'diffusion_feature_init_y', 0.05)), self._alpha_floor
            ))

        self.W2 = nn.Embedding(item_num, dims, padding_idx=item_num - 1).to(device)
        self.b2 = nn.Embedding(item_num, 1, padding_idx=item_num - 1).to(device)

        self.user_embeddings.weight.data.normal_(0, 1.0 / self.user_embeddings.embedding_dim)
        self.item_embeddings.weight.data.normal_(0, 1.0 / self.item_embeddings.embedding_dim)
        self.position_embeddings.weight.data.normal_(0, 1.0 / self.position_embeddings.embedding_dim)
        self.W2.weight.data.normal_(0, 1.0 / self.W2.embedding_dim)
        self.b2.weight.data.zero_()

        self._last_score_debug: Dict[str, float] = {}
        if self.use_diffusion_score:
            hidden = int(getattr(self.args, 'diffusion_adapter_hidden', 64))
            dropout = float(getattr(self.args, 'diffusion_adapter_dropout', 0.0))
            self.detach_adapter_context = bool(
                getattr(self.args, 'diffusion_adapter_detach_context', False)
                or getattr(self.args, 'diffusion_controller_detach_context', False)
            )
            self.score_residual_max = float(getattr(self.args, 'diffusion_score_residual_max', 1.0))
            self.diffusion_score_domain_embedding = nn.Embedding(2, dims).to(device)
            self.diffusion_score_context = nn.Sequential(
                nn.Linear(dims * 2, hidden),
                nn.GELU(),
                nn.Dropout(dropout),
            ).to(device)
            self.diffusion_score_feature = nn.Sequential(
                nn.Linear(8, hidden),
                nn.GELU(),
                nn.Dropout(dropout),
            ).to(device)
            self.diffusion_score_adapter = nn.Sequential(
                nn.Linear(hidden, hidden),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden, 1),
            ).to(device)
            self._init_score_adapter()
        else:
            self.detach_adapter_context = False
            self.score_residual_max = 1.0

    @staticmethod
    def _clip01(x: float) -> float:
        return min(max(float(x), 1e-6), 1.0 - 1e-6)

    @classmethod
    def _inv_bounded_sigmoid(cls, target: float, lower: float, upper: float) -> torch.Tensor:
        z = (float(target) - float(lower)) / max(float(upper) - float(lower), 1e-6)
        z = cls._clip01(z)
        return torch.tensor(math.log(z / (1.0 - z)), dtype=torch.float32)

    @staticmethod
    def _inv_positive_softplus(target: float, floor: float) -> torch.Tensor:
        shifted = max(float(target) - float(floor), 1e-8)
        return torch.tensor(math.log(math.exp(shifted) - 1.0), dtype=torch.float32)

    @staticmethod
    def _bounded_sigmoid(raw: torch.Tensor, lower: float, upper: float) -> torch.Tensor:
        return float(lower) + (float(upper) - float(lower)) * torch.sigmoid(raw)

    @staticmethod
    def _positive_softplus(raw: torch.Tensor, floor: float) -> torch.Tensor:
        return float(floor) + F.softplus(raw)

    def get_diffusion_gate_values(self) -> Dict[str, torch.Tensor]:
        if not self.use_diffusion_gate:
            z = next(self.parameters()).new_tensor(0.0)
            return {
                'feature_mix_lambda': z,
                'feature_alpha_seq': z,
                'feature_alpha_x': z,
                'feature_alpha_y': z,
            }
        return {
            'feature_mix_lambda': self._bounded_sigmoid(self.plugin_feature_mix_raw, self._mix_floor, 0.30),
            'feature_alpha_seq': self._positive_softplus(self.plugin_feature_seq_raw, self._alpha_floor),
            'feature_alpha_x': self._positive_softplus(self.plugin_feature_x_raw, self._alpha_floor),
            'feature_alpha_y': self._positive_softplus(self.plugin_feature_y_raw, self._alpha_floor),
        }

    def get_diffusion_gate_values_detached(self) -> Dict[str, float]:
        return {k: float(v.detach().cpu().item()) for k, v in self.get_diffusion_gate_values().items()}

    def _init_score_adapter(self):
        with torch.no_grad():
            self.diffusion_score_domain_embedding.weight.normal_(0.0, 0.02)
            modules = [
                self.diffusion_score_context,
                self.diffusion_score_feature,
                self.diffusion_score_adapter,
            ]
            for module in modules:
                for layer in module:
                    if isinstance(layer, nn.Linear):
                        nn.init.xavier_uniform_(layer.weight)
                        nn.init.zeros_(layer.bias)
            nn.init.zeros_(self.diffusion_score_adapter[-1].weight)
            nn.init.zeros_(self.diffusion_score_adapter[-1].bias)

    def score_adapter_param_names(self) -> Set[str]:
        return {name for name, _ in self.named_parameters() if name.startswith('diffusion_score_')}

    def adapter_param_names(self) -> Set[str]:
        return self.score_adapter_param_names()

    def gate_param_names(self) -> Set[str]:
        if not self.use_diffusion_gate:
            return set()
        return {
            'plugin_feature_mix_raw',
            'plugin_feature_seq_raw',
            'plugin_feature_x_raw',
            'plugin_feature_y_raw',
        }

    def plugin_param_names(self) -> Set[str]:
        return self.gate_param_names() | self.score_adapter_param_names()

    def _adapter_user_embs(self, user_ids: torch.Tensor) -> torch.Tensor:
        user_embs = self.user_embeddings(user_ids)
        return user_embs.detach() if self.detach_adapter_context else user_embs

    def get_adapter_debug_values(self) -> Dict[str, float]:
        return dict(self._last_score_debug)

    def get_learned_score_coeffs_detached(self) -> Dict[str, float]:
        return self.get_adapter_debug_values()

    @staticmethod
    def _masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        denom = mask.float().sum(dim=1, keepdim=True).clamp(min=1.0)
        return (values * mask.float()).sum(dim=1, keepdim=True) / denom

    @staticmethod
    def _masked_std(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        denom = mask.float().sum(dim=1, keepdim=True).clamp(min=1.0)
        mean = (values * mask.float()).sum(dim=1, keepdim=True) / denom
        centered = (values - mean) * mask.float()
        var = (centered * centered).sum(dim=1, keepdim=True) / denom
        return torch.sqrt(var + 1e-6)

    def adapt_diffusion_scores(self, user_ids: torch.Tensor, row_domain: torch.Tensor,
                               long_norm: torch.Tensor, short_norm: torch.Tensor,
                               valid_mask: torch.Tensor) -> Dict[str, torch.Tensor]:
        if not self.use_diffusion_score:
            raise RuntimeError('adapt_diffusion_scores called while use_diffusion_score=False')
        row_domain = row_domain.long().clamp(0, 1)
        valid = valid_mask.float()
        diff = long_norm - short_norm
        long_spread = self._masked_std(long_norm, valid_mask)
        short_spread = self._masked_std(short_norm, valid_mask)
        disagreement = self._masked_mean(diff.abs(), valid_mask)
        bsz, cand_num = long_norm.shape

        features = torch.stack([
            long_norm,
            short_norm,
            diff,
            long_norm.abs(),
            short_norm.abs(),
            long_spread.expand(bsz, cand_num),
            short_spread.expand(bsz, cand_num),
            disagreement.expand(bsz, cand_num),
        ], dim=-1)

        user_ctx = self._adapter_user_embs(user_ids)
        domain_ctx = self.diffusion_score_domain_embedding(row_domain)
        context = self.diffusion_score_context(torch.cat([user_ctx, domain_ctx], dim=-1))
        h = self.diffusion_score_feature(features) + context.unsqueeze(1)
        residual = self.score_residual_max * torch.tanh(self.diffusion_score_adapter(h).squeeze(-1))
        residual = residual * valid
        residual_reg = (residual.pow(2) * valid).sum() / valid.sum().clamp(min=1.0)

        with torch.no_grad():
            total = max(float(valid.sum().detach().cpu().item()), 1.0)
            debug = {
                'score_residual_abs_mean': float((residual.abs() * valid).sum().detach().cpu().item() / total),
                'score_long_spread_mean': float(long_spread.mean().detach().cpu().item()),
                'score_short_spread_mean': float(short_spread.mean().detach().cpu().item()),
                'score_disagreement_mean': float(disagreement.mean().detach().cpu().item()),
            }
            for value, name in [(0, 'x'), (1, 'y')]:
                m = row_domain.eq(value)
                if m.any():
                    v = valid[m]
                    r = residual[m]
                    denom = max(float(v.sum().detach().cpu().item()), 1.0)
                    debug[f'score_residual_{name}_abs_mean'] = float((r.abs() * v).sum().detach().cpu().item() / denom)
            self._last_score_debug = debug

        return {'residual': residual, 'residual_reg': residual_reg, 'valid': valid}

    @staticmethod
    def _mix_gate(base_gate: torch.Tensor, guided_gate: torch.Tensor, mix_lambda: torch.Tensor) -> torch.Tensor:
        return (1.0 - mix_lambda) * base_gate + mix_lambda * guided_gate

    def _feature_gate_with_diffusion(self, item_embs: torch.Tensor, user_embs: torch.Tensor,
                                     item_gate_layer: nn.Linear, diff_feature_proj: nn.Linear,
                                     feature_bias: Optional[torch.Tensor],
                                     feature_mix_lambda: torch.Tensor,
                                     feature_alpha: torch.Tensor) -> torch.Tensor:
        base_logits = item_gate_layer(item_embs) + self.feature_gate_user(user_embs).unsqueeze(1)
        base_gate = torch.sigmoid(base_logits)
        if feature_bias is None:
            return base_gate
        diff_bias = diff_feature_proj(feature_bias.unsqueeze(-1))
        guided_logits = base_logits + feature_alpha * diff_bias
        guided_gate = torch.sigmoid(guided_logits)
        return self._mix_gate(base_gate, guided_gate, feature_mix_lambda)

    def _instance_gate(self, gated_item: torch.Tensor, user_embs: torch.Tensor,
                       position_embs: torch.Tensor, instance_gate_item: torch.Tensor) -> torch.Tensor:
        logits = (
            torch.matmul(gated_item, instance_gate_item.unsqueeze(0)).squeeze()
            + user_embs.mm(self.instance_gate_user)
            + torch.matmul(position_embs, self.instance_gate_position.unsqueeze(0)).squeeze()
        )
        return torch.sigmoid(logits)

    @staticmethod
    def _safe_reduce(gated_item: torch.Tensor, instance_score: torch.Tensor) -> torch.Tensor:
        out = gated_item * instance_score.unsqueeze(2)
        out = torch.sum(out, dim=1)
        denom = torch.sum(instance_score, dim=1, keepdim=True).clamp(min=1e-8)
        return out / denom

    def forward(self, seq, x_seq, y_seq, position, x_position, y_position, user_ids, items_to_predict,
                x_items_to_predict=None, y_items_to_predict=None, for_pred=False, pred_domain='x',
                feature_bias_seq=None, feature_bias_x=None, feature_bias_y=None,
                instance_bias_seq=None, instance_bias_x=None, instance_bias_y=None,
                feature_mix_lambda=None, instance_mix_lambda=None,
                feature_alpha=None, instance_alpha=None,
                return_representations: bool = False,
                **unused_kwargs):
        del instance_bias_seq, instance_bias_x, instance_bias_y, instance_mix_lambda, instance_alpha, unused_kwargs

        user_embs = self.user_embeddings(user_ids)
        w2 = self.W2(items_to_predict)

        item_embs = self.item_embeddings(seq)
        item_embs_src = self.item_embeddings(x_seq)
        item_embs_tgt = self.item_embeddings(y_seq)

        position_embs = self.position_embeddings(position)
        position_embs_src = self.position_embeddings(x_position)
        position_embs_tgt = self.position_embeddings(y_position)

        gate_vals = self.get_diffusion_gate_values()
        feature_mix_lambda = gate_vals['feature_mix_lambda'] if feature_mix_lambda is None else feature_mix_lambda

        if isinstance(feature_alpha, dict):
            feature_alpha_seq = feature_alpha.get('seq', gate_vals['feature_alpha_seq'])
            feature_alpha_x = feature_alpha.get('x', gate_vals['feature_alpha_x'])
            feature_alpha_y = feature_alpha.get('y', gate_vals['feature_alpha_y'])
        elif feature_alpha is None:
            feature_alpha_seq = gate_vals['feature_alpha_seq']
            feature_alpha_x = gate_vals['feature_alpha_x']
            feature_alpha_y = gate_vals['feature_alpha_y']
        else:
            feature_alpha_seq = feature_alpha_x = feature_alpha_y = feature_alpha

        gate_src = self._feature_gate_with_diffusion(
            item_embs_src, user_embs, self.feature_gate_item_src, self.diff_feature_proj_src,
            feature_bias_x, feature_mix_lambda, feature_alpha_x
        )
        gate_tgt = self._feature_gate_with_diffusion(
            item_embs_tgt, user_embs, self.feature_gate_item_tgt, self.diff_feature_proj_tgt,
            feature_bias_y, feature_mix_lambda, feature_alpha_y
        )
        gate = self._feature_gate_with_diffusion(
            item_embs, user_embs, self.feature_gate_item, self.diff_feature_proj,
            feature_bias_seq, feature_mix_lambda, feature_alpha_seq
        )

        gated_item_src = item_embs_src * gate_src
        gated_item_tgt = item_embs_tgt * gate_tgt
        gated_item = item_embs * gate

        instance_score_src = self._instance_gate(gated_item_src, user_embs, position_embs_src, self.instance_gate_item_src)
        instance_score_tgt = self._instance_gate(gated_item_tgt, user_embs, position_embs_tgt, self.instance_gate_item_tgt)
        instance_score = self._instance_gate(gated_item, user_embs, position_embs, self.instance_gate_item)

        union_out_src = self._safe_reduce(gated_item_src, instance_score_src)
        union_out_tgt = self._safe_reduce(gated_item_tgt, instance_score_tgt)
        union_out = self._safe_reduce(gated_item, instance_score)

        if for_pred:
            res = torch.bmm(union_out.unsqueeze(1), w2.permute(0, 2, 1)).squeeze()
            rel_score = torch.mean(item_embs.bmm(w2.permute(0, 2, 1)), dim=1)
            res = res + rel_score
            if pred_domain == 'x':
                res_src = torch.bmm(union_out_src.unsqueeze(1), w2.permute(0, 2, 1)).squeeze()
                rel_score_src = torch.mean(item_embs_src.bmm(w2.permute(0, 2, 1)), dim=1)
                res = res + res_src + rel_score_src
            elif pred_domain == 'y':
                res_tgt = torch.bmm(union_out_tgt.unsqueeze(1), w2.permute(0, 2, 1)).squeeze()
                rel_score_tgt = torch.mean(item_embs_tgt.bmm(w2.permute(0, 2, 1)), dim=1)
                res = res + res_tgt + rel_score_tgt
            if return_representations:
                return res, {'union_out': union_out, 'union_out_src': union_out_src, 'union_out_tgt': union_out_tgt}
            return res

        w2_src = self.W2(x_items_to_predict)
        w2_tgt = self.W2(y_items_to_predict)

        res_src = torch.bmm(union_out_src.unsqueeze(1), w2_src.permute(0, 2, 1)).squeeze()
        res_src = res_src + torch.mean(item_embs_src.bmm(w2_src.permute(0, 2, 1)), dim=1)

        res_tgt = torch.bmm(union_out_tgt.unsqueeze(1), w2_tgt.permute(0, 2, 1)).squeeze()
        res_tgt = res_tgt + torch.mean(item_embs_tgt.bmm(w2_tgt.permute(0, 2, 1)), dim=1)

        res = torch.bmm(union_out.unsqueeze(1), w2.permute(0, 2, 1)).squeeze()
        res = res + torch.mean(item_embs.bmm(w2.permute(0, 2, 1)), dim=1)

        if return_representations:
            return res, res_src, res_tgt, {
                'union_out': union_out,
                'union_out_src': union_out_src,
                'union_out_tgt': union_out_tgt,
            }
        return res, res_src, res_tgt


