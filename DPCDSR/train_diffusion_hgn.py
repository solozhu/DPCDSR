import argparse
import datetime
import hashlib
import os
import pickle
import random
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Dict, List, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader

from DNN import DNN
import gaussian_diffusion as gd
from dataProcessing import DataLoader as HGNDataLoader
from hgn_diffusion_utils import (
    UserHistory,
    UserHistoryDataset,
    UserHistoryInferDataset,
    aggregate_train_histories,
    build_top_unseen_pools,
    compute_prior_domain_stats,
    evaluate_prior_matrix,
    infer_user_priors,
    mix_priors,
    pretty_metric_line,
    save_pickle,
    set_random_seed,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='Food-Kitchen')
    parser.add_argument('--output_dir', type=str, default='./diffusion_hgn_ckpt')
    parser.add_argument('--seed', type=int, default=1708008752)
    parser.add_argument('--device', type=str, default='cuda')

    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--eval_batch_size', type=int, default=256)
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--patience', type=int, default=20)
    parser.add_argument('--lr', type=float, default=5e-5)
    parser.add_argument('--weight_decay', type=float, default=0.0)

    parser.add_argument('--w_min', type=float, default=0.1)
    parser.add_argument('--w_max', type=float, default=1.0)
    parser.add_argument('--reweight_version', type=str, default='ExpDecay',
                        choices=['AllOne', 'AllLinear', 'MinMax', 'ExpDecay'])
    parser.add_argument('--exp_beta', type=float, default=3.0)
    parser.add_argument('--recent_k', type=int, default=30)
    parser.add_argument('--recent_mix_eta', type=float, default=0.6)

    parser.add_argument('--shared_weight_x', type=float, default=0.35)
    parser.add_argument('--shared_weight_y', type=float, default=0.35)

    parser.add_argument('--dims', type=str, default='[1000]')
    parser.add_argument('--norm', action='store_true')
    parser.add_argument('--emb_size', type=int, default=10)
    parser.add_argument('--mean_type', type=str, default='x0', choices=['x0', 'eps'])
    parser.add_argument('--steps', type=int, default=10)
    parser.add_argument('--noise_schedule', type=str, default='linear-var')
    parser.add_argument('--noise_scale', type=float, default=0.01)
    parser.add_argument('--noise_min', type=float, default=0.0005)
    parser.add_argument('--noise_max', type=float, default=0.005)
    parser.add_argument('--sampling_steps', type=int, default=0)
    parser.add_argument('--sampling_noise', action='store_true')
    parser.add_argument('--reweight', action='store_true', default=True)

    parser.add_argument('--valid_negatives', type=int, default=999)
    parser.add_argument('--top_pool_x', type=int, default=200)
    parser.add_argument('--top_pool_y', type=int, default=200)

    parser.add_argument('--skip_plugin_cache', action='store_true')
    parser.add_argument('--plugin_seed', type=int, default=None)
    parser.add_argument('--plugin_batch_size', type=int, default=256)
    parser.add_argument('--plugin_neg_samples', type=int, default=99)
    parser.add_argument('--plugin_cache_dtype', type=str, default='float16', choices=['float16', 'float32'])
    parser.add_argument('--plugin_cache_path', type=str, default='')
    parser.add_argument('--plugin_score_short_k', type=int, default=5)
    return parser.parse_args()


@dataclass
class DiffusionBranch:
    name: str
    model: torch.nn.Module
    diffusion: torch.nn.Module


@dataclass
class FrozenDiffusionBranch:
    model: torch.nn.Module
    diffusion: torch.nn.Module


class FrozenDiffusionPlugin:
    def __init__(self, shared_branch: FrozenDiffusionBranch, x_branch: FrozenDiffusionBranch,
                 y_branch: FrozenDiffusionBranch, args: Dict, meta: Dict, dev: torch.device):
        self.shared = shared_branch
        self.x = x_branch
        self.y = y_branch
        self.args = args
        self.meta = meta
        self.device = dev
        self.item_num = int(meta['item_num'])
        self.source_item_num = int(meta['source_item_num'])
        self.w_min = float(args['w_min'])
        self.w_max = float(args['w_max'])
        self.reweight_version = str(args['reweight_version'])
        self.exp_beta = float(args['exp_beta'])
        self.shared_weight_x = float(args['shared_weight_x'])
        self.shared_weight_y = float(args['shared_weight_y'])
        self.recent_k = int(args['recent_k'])
        self.sampling_steps = int(args['sampling_steps'])
        self.sampling_noise = bool(args['sampling_noise'])

    @staticmethod
    def _build_diffusion(args: Dict, item_num: int, dev: torch.device) -> Tuple[torch.nn.Module, torch.nn.Module]:
        mean_type = gd.ModelMeanType.START_X if args['mean_type'] == 'x0' else gd.ModelMeanType.EPSILON
        diffusion = gd.GaussianDiffusion(
            mean_type,
            args['noise_schedule'],
            args['noise_scale'],
            args['noise_min'],
            args['noise_max'],
            args['steps'],
            str(dev),
        ).to(dev)
        out_dims = eval(args['dims']) + [item_num]
        in_dims = out_dims[::-1]
        model = DNN(in_dims, out_dims, args['emb_size'], time_type='cat', norm=args['norm']).to(dev)
        return model, diffusion

    @classmethod
    def load(cls, ckpt_path: str, conf, dev: torch.device):
        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(f'diffusion checkpoint not found: {ckpt_path}')
        saved = torch.load(ckpt_path, map_location=dev)
        args = saved['args']
        meta = saved['meta']
        if int(meta['item_num']) != conf.item_num - 1:
            raise ValueError(f'diffusion item_num mismatch: diffusion={meta["item_num"]}, hgn={conf.item_num - 1}')

        shared_model, shared_diff = cls._build_diffusion(args, int(meta['item_num']), dev)
        x_model, x_diff = cls._build_diffusion(args, int(meta['item_num']), dev)
        y_model, y_diff = cls._build_diffusion(args, int(meta['item_num']), dev)

        shared_model.load_state_dict(saved['shared_state_dict'])
        x_model.load_state_dict(saved['x_state_dict'])
        y_model.load_state_dict(saved['y_state_dict'])

        for model in (shared_model, x_model, y_model):
            model.eval()
            for p in model.parameters():
                p.requires_grad = False

        return cls(
            FrozenDiffusionBranch(shared_model, shared_diff),
            FrozenDiffusionBranch(x_model, x_diff),
            FrozenDiffusionBranch(y_model, y_diff),
            args,
            meta,
            dev,
        )

    @torch.no_grad()
    def infer_triplet(self, input_shared: torch.Tensor, input_x: torch.Tensor, input_y: torch.Tensor):
        score_shared = self.shared.diffusion.p_sample(self.shared.model, input_shared, self.sampling_steps, self.sampling_noise).detach()
        score_x = self.x.diffusion.p_sample(self.x.model, input_x, self.sampling_steps, self.sampling_noise).detach()
        score_y = self.y.diffusion.p_sample(self.y.model, input_y, self.sampling_steps, self.sampling_noise).detach()
        return score_shared, score_x, score_y


def build_diffusion(args, item_num: int, device: torch.device):
    mean_type = gd.ModelMeanType.START_X if args.mean_type == 'x0' else gd.ModelMeanType.EPSILON
    diffusion = gd.GaussianDiffusion(
        mean_type,
        args.noise_schedule,
        args.noise_scale,
        args.noise_min,
        args.noise_max,
        args.steps,
        str(device),
    ).to(device)
    out_dims = eval(args.dims) + [item_num]
    in_dims = out_dims[::-1]
    model = DNN(in_dims, out_dims, args.emb_size, time_type='cat', norm=args.norm).to(device)
    return model, diffusion


def filter_histories_by_domain(histories: List[UserHistory], source_item_num: int, domain: str) -> List[UserHistory]:
    out = []
    for hist in histories:
        if domain == 'x':
            pairs = [(i, t) for i, t in zip(hist.item_ids, hist.timestamps) if i < source_item_num]
        elif domain == 'y':
            pairs = [(i, t) for i, t in zip(hist.item_ids, hist.timestamps) if i >= source_item_num]
        else:
            raise ValueError(domain)
        out.append(UserHistory(user=hist.user, item_ids=[p[0] for p in pairs], timestamps=[p[1] for p in pairs]))
    return out


def non_empty_histories(histories: List[UserHistory]) -> List[UserHistory]:
    return [h for h in histories if len(h.item_ids) > 0]


def rowwise_tanh_zscore(matrix: np.ndarray, item_slice: slice) -> np.ndarray:
    out = np.zeros_like(matrix, dtype=np.float32)
    sub = matrix[:, item_slice].astype(np.float32)
    if sub.size == 0:
        return out
    mean = sub.mean(axis=1, keepdims=True)
    std = sub.std(axis=1, keepdims=True) + 1e-6
    out[:, item_slice] = np.tanh(np.clip((sub - mean) / std, -3.0, 3.0)).astype(np.float32)
    return out


def build_final_prior(shared_matrix: np.ndarray, x_matrix: np.ndarray, y_matrix: np.ndarray,
                      meta: Dict[str, int], shared_weight_x: float, shared_weight_y: float) -> np.ndarray:
    source_item_num = int(meta['source_item_num'])
    item_num = int(meta['item_num'])
    x_slice = slice(0, source_item_num)
    y_slice = slice(source_item_num, item_num)

    shared_x = rowwise_tanh_zscore(shared_matrix, x_slice)
    shared_y = rowwise_tanh_zscore(shared_matrix, y_slice)
    spec_x = rowwise_tanh_zscore(x_matrix, x_slice)
    spec_y = rowwise_tanh_zscore(y_matrix, y_slice)

    out = np.zeros_like(shared_matrix, dtype=np.float32)
    out[:, x_slice] = (shared_weight_x * shared_x[:, x_slice] + (1.0 - shared_weight_x) * spec_x[:, x_slice]).astype(np.float32)
    out[:, y_slice] = (shared_weight_y * shared_y[:, y_slice] + (1.0 - shared_weight_y) * spec_y[:, y_slice]).astype(np.float32)
    return out


def train_one_epoch(branch: DiffusionBranch, loader: DataLoader, optimizer: torch.optim.Optimizer,
                    reweight: bool, device: torch.device) -> float:
    branch.model.train()
    total_loss = 0.0
    steps = 0
    for x_start, _ in loader:
        x_start = x_start.to(device=device, dtype=torch.float32)
        optimizer.zero_grad()
        losses = branch.diffusion.training_losses(branch.model, x_start, reweight)
        loss = losses['loss'].mean()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        steps += 1
    return total_loss / max(steps, 1)


def infer_matrix(branch: DiffusionBranch, histories: List[UserHistory], args, meta, device: torch.device, mode: str) -> np.ndarray:
    dataset = UserHistoryInferDataset(
        histories, meta['item_num'], args.w_min, args.w_max,
        args.reweight_version, args.exp_beta, args.recent_k, mode=mode
    )
    priors, users = infer_user_priors(branch.model, branch.diffusion, dataset, device,
                                      args.eval_batch_size, args.sampling_steps, args.sampling_noise)
    mat = np.zeros((meta['user_num'], meta['item_num']), dtype=np.float32)
    for row, user in zip(priors, users):
        mat[int(user)] = row
    return mat


def to_storage(arr: torch.Tensor, dtype: str):
    np_arr = arr.detach().cpu().numpy()
    if dtype == 'float16' and np_arr.dtype == np.float32:
        return np_arr.astype(np.float16)
    return np_arr.astype(np.float32) if np_arr.dtype in (np.float16, np.float32) else np_arr


def batch_signature(seq, x_seq, y_seq, ground, user, candidate_items=None):
    sig = {
        'seq': np.asarray(seq.cpu().numpy(), dtype=np.int32),
        'x_seq': np.asarray(x_seq.cpu().numpy(), dtype=np.int32),
        'y_seq': np.asarray(y_seq.cpu().numpy(), dtype=np.int32),
        'ground': np.asarray(ground.cpu().numpy(), dtype=np.int32),
        'user': np.asarray(user.cpu().numpy(), dtype=np.int32),
    }
    if candidate_items is not None:
        sig['candidate_items'] = np.asarray(candidate_items.cpu().numpy(), dtype=np.int32)
    return sig


def rank_weights(length: int, min_value: float, max_value: float,
                 reweight_version: str = 'ExpDecay', exp_beta: float = 3.0) -> torch.Tensor:
    if length <= 0:
        return torch.zeros(0, dtype=torch.float32)
    if length == 1:
        return torch.tensor([max_value], dtype=torch.float32)
    if reweight_version == 'AllOne':
        return torch.ones(length, dtype=torch.float32) * max_value
    if reweight_version in ('AllLinear', 'MinMax'):
        return torch.linspace(min_value, max_value, length, dtype=torch.float32)
    ranks = torch.linspace(0.0, 1.0, length, dtype=torch.float32)
    beta = max(float(exp_beta), 1e-6)
    curve = (torch.exp(beta * ranks) - 1.0) / (float(np.exp(beta)) - 1.0)
    return min_value + (max_value - min_value) * curve


@torch.no_grad()
def build_diffusion_input_from_seq(seq: torch.Tensor, pad_id: int, item_num_wo_pad: int,
                                   w_min: float, w_max: float, reweight_version: str,
                                   exp_beta: float, mode: str = 'long', recent_k: int = 5) -> torch.Tensor:
    bsz, _ = seq.shape
    out = torch.zeros((bsz, item_num_wo_pad), dtype=torch.float32, device=seq.device)
    for i in range(bsz):
        row = seq[i]
        valid = row[row.ne(pad_id)]
        valid = valid[valid.lt(item_num_wo_pad)]
        if mode == 'short' and recent_k > 0 and int(valid.numel()) > recent_k:
            valid = valid[-recent_k:]
        n = int(valid.numel())
        if n == 0:
            continue
        weights = rank_weights(n, w_min, w_max, reweight_version, exp_beta).to(seq.device)
        for item_id, weight in zip(valid.tolist(), weights.tolist()):
            item_id = int(item_id)
            if 0 <= item_id < item_num_wo_pad:
                cur = float(out[i, item_id].item())
                if float(weight) > cur:
                    out[i, item_id] = float(weight)
    return out


def _gather_scores(score_matrix: torch.Tensor, item_ids: torch.Tensor, item_limit: int, pad_id: int) -> torch.Tensor:
    safe_ids = item_ids.clone()
    valid = safe_ids.ne(pad_id) & safe_ids.ge(0) & safe_ids.lt(item_limit)
    safe_ids = torch.where(valid, safe_ids, torch.zeros_like(safe_ids))
    gathered = torch.gather(score_matrix, 1, safe_ids)
    return gathered * valid.float()


def normalize_by_mask(values: torch.Tensor, mask: torch.Tensor, clip_value: float = 3.0) -> torch.Tensor:
    denom = mask.float().sum(dim=1, keepdim=True).clamp(min=1.0)
    mean = (values * mask.float()).sum(dim=1, keepdim=True) / denom
    centered = (values - mean) * mask.float()
    var = (centered * centered).sum(dim=1, keepdim=True) / denom
    std = torch.sqrt(var + 1e-6)
    norm = centered / std
    norm = torch.clamp(norm, -clip_value, clip_value)
    return torch.tanh(norm) * mask.float()


def combine_shared_specific(shared_scores: torch.Tensor, x_scores: torch.Tensor, y_scores: torch.Tensor,
                            source_item_num: int, shared_weight_x: float, shared_weight_y: float) -> Tuple[torch.Tensor, torch.Tensor]:
    item_num = shared_scores.size(1)
    out_x = torch.zeros_like(shared_scores)
    out_y = torch.zeros_like(shared_scores)
    out_x[:, :source_item_num] = shared_weight_x * shared_scores[:, :source_item_num] + (1.0 - shared_weight_x) * x_scores[:, :source_item_num]
    out_y[:, source_item_num:item_num] = shared_weight_y * shared_scores[:, source_item_num:item_num] + (1.0 - shared_weight_y) * y_scores[:, source_item_num:item_num]
    return out_x, out_y


def build_gate_biases(seq: torch.Tensor, x_seq: torch.Tensor, y_seq: torch.Tensor,
                      score_x_combined: torch.Tensor, score_y_combined: torch.Tensor,
                      source_item_num: int, pad_id: int) -> Dict[str, torch.Tensor]:
    item_limit = score_x_combined.size(1)
    x_raw = _gather_scores(score_x_combined, x_seq, item_limit, pad_id)
    y_raw = _gather_scores(score_y_combined, y_seq, item_limit, pad_id)
    seq_x_raw = _gather_scores(score_x_combined, seq, item_limit, pad_id)
    seq_y_raw = _gather_scores(score_y_combined, seq, item_limit, pad_id)
    is_x = (seq < source_item_num) & seq.ne(pad_id)
    is_y = (seq >= source_item_num) & seq.lt(item_limit)
    seq_raw = torch.where(is_x, seq_x_raw, torch.zeros_like(seq_x_raw)) + torch.where(is_y, seq_y_raw, torch.zeros_like(seq_y_raw))

    x_mask = x_seq.ne(pad_id)
    y_mask = y_seq.ne(pad_id)
    seq_mask = seq.ne(pad_id)
    return {
        'feature_bias_x': normalize_by_mask(x_raw, x_mask),
        'feature_bias_y': normalize_by_mask(y_raw, y_mask),
        'feature_bias_seq': normalize_by_mask(seq_raw, seq_mask),
    }


def build_candidate_diffusion_scores(candidate_items: torch.Tensor,
                                     score_x_combined: torch.Tensor,
                                     score_y_combined: torch.Tensor,
                                     source_item_num: int,
                                     pad_id: int) -> torch.Tensor:
    item_limit = score_x_combined.size(1)
    cand_x = _gather_scores(score_x_combined, candidate_items, item_limit, pad_id)
    cand_y = _gather_scores(score_y_combined, candidate_items, item_limit, pad_id)
    is_x = candidate_items < source_item_num
    is_y = (candidate_items >= source_item_num) & candidate_items.lt(item_limit)
    return torch.where(is_x, cand_x, torch.zeros_like(cand_x)) + torch.where(is_y, cand_y, torch.zeros_like(cand_y))


def build_split_cache(split_name: str, dataloader, conf, plugin: FrozenDiffusionPlugin,
                      cache_dtype: str = 'float16', dev: torch.device = None):
    rows: List[Dict] = []
    if dev is None:
        dev = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    pad_id = conf.item_num - 1
    short_k = int(getattr(conf, 'plugin_score_short_k', 5))

    for batch_idx, batch in enumerate(dataloader):
        inputs = [b.to(dev) for b in batch]
        seq = inputs[0]
        x_seq = inputs[1]
        y_seq = inputs[2]
        ground = inputs[3]
        position = inputs[4]
        x_position = inputs[5]
        y_position = inputs[6]
        user = inputs[7] if split_name != 'train' else inputs[9]
        neg = inputs[10] if split_name != 'train' else inputs[12]
        items_to_predict = torch.cat((ground, neg), dim=1)

        input_shared_long = build_diffusion_input_from_seq(seq, pad_id, plugin.item_num, plugin.w_min, plugin.w_max, plugin.reweight_version, plugin.exp_beta, mode='long', recent_k=plugin.recent_k)
        input_x_long = build_diffusion_input_from_seq(x_seq, pad_id, plugin.item_num, plugin.w_min, plugin.w_max, plugin.reweight_version, plugin.exp_beta, mode='long', recent_k=plugin.recent_k)
        input_y_long = build_diffusion_input_from_seq(y_seq, pad_id, plugin.item_num, plugin.w_min, plugin.w_max, plugin.reweight_version, plugin.exp_beta, mode='long', recent_k=plugin.recent_k)
        shared_long, x_long, y_long = plugin.infer_triplet(input_shared_long, input_x_long, input_y_long)

        input_shared_short = build_diffusion_input_from_seq(seq, pad_id, plugin.item_num, plugin.w_min, plugin.w_max, plugin.reweight_version, plugin.exp_beta, mode='short', recent_k=short_k)
        input_x_short = build_diffusion_input_from_seq(x_seq, pad_id, plugin.item_num, plugin.w_min, plugin.w_max, plugin.reweight_version, plugin.exp_beta, mode='short', recent_k=short_k)
        input_y_short = build_diffusion_input_from_seq(y_seq, pad_id, plugin.item_num, plugin.w_min, plugin.w_max, plugin.reweight_version, plugin.exp_beta, mode='short', recent_k=short_k)
        shared_short, x_short, y_short = plugin.infer_triplet(input_shared_short, input_x_short, input_y_short)

        score_x_long, score_y_long = combine_shared_specific(shared_long, x_long, y_long, plugin.source_item_num, plugin.shared_weight_x, plugin.shared_weight_y)
        score_x_short, score_y_short = combine_shared_specific(shared_short, x_short, y_short, plugin.source_item_num, plugin.shared_weight_x, plugin.shared_weight_y)

        gate_biases = build_gate_biases(seq, x_seq, y_seq, score_x_long, score_y_long, plugin.source_item_num, pad_id)
        candidate_scores_long = build_candidate_diffusion_scores(items_to_predict, score_x_long, score_y_long, plugin.source_item_num, pad_id)
        candidate_scores_short = build_candidate_diffusion_scores(items_to_predict, score_x_short, score_y_short, plugin.source_item_num, pad_id)

        rows.append({
            'signature': batch_signature(seq, x_seq, y_seq, ground, user, items_to_predict),
            'feature_bias_seq': to_storage(gate_biases['feature_bias_seq'], cache_dtype),
            'feature_bias_x': to_storage(gate_biases['feature_bias_x'], cache_dtype),
            'feature_bias_y': to_storage(gate_biases['feature_bias_y'], cache_dtype),
            'candidate_scores': to_storage(candidate_scores_long, cache_dtype),
            'candidate_scores_long': to_storage(candidate_scores_long, cache_dtype),
            'candidate_scores_short': to_storage(candidate_scores_short, cache_dtype),
            'position': np.asarray(position.cpu().numpy(), dtype=np.int16),
            'x_position': np.asarray(x_position.cpu().numpy(), dtype=np.int16),
            'y_position': np.asarray(y_position.cpu().numpy(), dtype=np.int16),
        })
        if (batch_idx + 1) % 20 == 0 or (batch_idx + 1) == len(dataloader):
            print(f'[{split_name}] cached batches: {batch_idx + 1}/{len(dataloader)}')
    return rows


def cache_fingerprint(meta: Dict) -> str:
    raw = pickle.dumps(meta, protocol=pickle.HIGHEST_PROTOCOL)
    return hashlib.md5(raw).hexdigest()


def default_diffusion_ckpt(data_dir: str, output_dir: str) -> str:
    return os.path.join(output_dir, f'{data_dir}_diffusion_domainaware_best.pt')


def default_runtime_cache_path(data_dir: str, output_dir: str, seed: int, batch_size: int, neg_samples: int) -> str:
    return os.path.join(output_dir, f'{data_dir}_plugin_cache_seed{seed}_bs{batch_size}_neg{neg_samples}.pkl')


def set_global_random_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch.backends, 'cudnn'):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def build_plugin_cache_after_training(args, dev: torch.device, ckpt_path: str):
    plugin_seed = int(args.plugin_seed if args.plugin_seed is not None else args.seed)
    set_global_random_seed(plugin_seed)

    hgn_conf = SimpleNamespace(L=15, d=256, maxlen=15, neg_samples=int(args.plugin_neg_samples), data_dir=args.data_dir)
    train_data = HGNDataLoader(args.data_dir, int(args.plugin_batch_size), hgn_conf, evaluation=-1)
    valid_data_x = HGNDataLoader(args.data_dir, int(args.plugin_batch_size), hgn_conf, evaluation=2, predict_domain='x')
    valid_data_y = HGNDataLoader(args.data_dir, int(args.plugin_batch_size), hgn_conf, evaluation=2, predict_domain='y')
    test_data_x = HGNDataLoader(args.data_dir, int(args.plugin_batch_size), hgn_conf, evaluation=1, predict_domain='x')
    test_data_y = HGNDataLoader(args.data_dir, int(args.plugin_batch_size), hgn_conf, evaluation=1, predict_domain='y')
    hgn_conf.item_num = hgn_conf.source_item_num + hgn_conf.target_item_num + 1

    plugin_cache_path = args.plugin_cache_path or default_runtime_cache_path(
        args.data_dir, args.output_dir, plugin_seed, int(args.plugin_batch_size), int(args.plugin_neg_samples)
    )

    print('build offline diffusion plugin cache: -----------------------------------------')
    print(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
    print(f'plugin seed = {plugin_seed}')
    plugin = FrozenDiffusionPlugin.load(ckpt_path, hgn_conf, dev)

    cache_meta = {
        'data_dir': args.data_dir,
        'Rseed': plugin_seed,
        'batch_size': int(args.plugin_batch_size),
        'neg_samples': int(args.plugin_neg_samples),
        'source_item_num': int(hgn_conf.source_item_num),
        'target_item_num': int(hgn_conf.target_item_num),
        'item_num': int(hgn_conf.item_num),
        'plugin_item_num': int(plugin.item_num),
        'diffusion_ckpt_path': ckpt_path,
        'cache_dtype': args.plugin_cache_dtype,
        'train_batches': len(train_data),
        'valid_x_batches': len(valid_data_x),
        'valid_y_batches': len(valid_data_y),
        'test_x_batches': len(test_data_x),
        'test_y_batches': len(test_data_y),
    }
    cache_meta['fingerprint'] = cache_fingerprint(cache_meta)

    cache = {
        'meta': cache_meta,
        'train': build_split_cache('train', train_data, hgn_conf, plugin, args.plugin_cache_dtype, dev),
        'valid_x': build_split_cache('valid_x', valid_data_x, hgn_conf, plugin, args.plugin_cache_dtype, dev),
        'valid_y': build_split_cache('valid_y', valid_data_y, hgn_conf, plugin, args.plugin_cache_dtype, dev),
        'test_x': build_split_cache('test_x', test_data_x, hgn_conf, plugin, args.plugin_cache_dtype, dev),
        'test_y': build_split_cache('test_y', test_data_y, hgn_conf, plugin, args.plugin_cache_dtype, dev),
    }

    os.makedirs(os.path.dirname(plugin_cache_path) or '.', exist_ok=True)
    with open(plugin_cache_path, 'wb') as f:
        pickle.dump(cache, f, protocol=pickle.HIGHEST_PROTOCOL)
    size_mb = os.path.getsize(plugin_cache_path) / (1024 * 1024)
    print(f'saved offline plugin cache: {plugin_cache_path}')
    print(f'cache size: {size_mb:.2f} MB')


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    set_random_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

    meta, train_histories, valid_records, test_records = aggregate_train_histories(args.data_dir)
    x_histories = filter_histories_by_domain(train_histories, meta['source_item_num'], 'x')
    y_histories = filter_histories_by_domain(train_histories, meta['source_item_num'], 'y')

    shared_train = non_empty_histories(train_histories)
    x_train = non_empty_histories(x_histories)
    y_train = non_empty_histories(y_histories)

    shared_loader = DataLoader(UserHistoryDataset(shared_train, meta['item_num'], args.w_min, args.w_max,
                                                  args.reweight_version, args.exp_beta, args.recent_k, mode='long'),
                               batch_size=args.batch_size, shuffle=True, num_workers=0)
    x_loader = DataLoader(UserHistoryDataset(x_train, meta['item_num'], args.w_min, args.w_max,
                                             args.reweight_version, args.exp_beta, args.recent_k, mode='long'),
                          batch_size=args.batch_size, shuffle=True, num_workers=0)
    y_loader = DataLoader(UserHistoryDataset(y_train, meta['item_num'], args.w_min, args.w_max,
                                             args.reweight_version, args.exp_beta, args.recent_k, mode='long'),
                          batch_size=args.batch_size, shuffle=True, num_workers=0)

    shared_model, shared_diff = build_diffusion(args, meta['item_num'], device)
    x_model, x_diff = build_diffusion(args, meta['item_num'], device)
    y_model, y_diff = build_diffusion(args, meta['item_num'], device)

    shared_branch = DiffusionBranch('shared', shared_model, shared_diff)
    x_branch = DiffusionBranch('x', x_model, x_diff)
    y_branch = DiffusionBranch('y', y_model, y_diff)

    optimizer = torch.optim.AdamW(
        list(shared_model.parameters()) + list(x_model.parameters()) + list(y_model.parameters()),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.99)

    best_valid_mrr = -1.0
    best_epoch = -1
    wait = 0
    ckpt_path = default_diffusion_ckpt(args.data_dir, args.output_dir)

    print('Start diffusion training')
    print(f'device={device}, users={meta["user_num"]}, items={meta["item_num"]}, shared_train={len(shared_train)}, x_train={len(x_train)}, y_train={len(y_train)}')

    for epoch in range(1, args.epochs + 1):
        shared_loss = train_one_epoch(shared_branch, shared_loader, optimizer, args.reweight, device)
        x_loss = train_one_epoch(x_branch, x_loader, optimizer, args.reweight, device)
        y_loss = train_one_epoch(y_branch, y_loader, optimizer, args.reweight, device)
        scheduler.step()

        shared_long = infer_matrix(shared_branch, train_histories, args, meta, device, mode='long')
        shared_recent = infer_matrix(shared_branch, train_histories, args, meta, device, mode='recent')
        x_long = infer_matrix(x_branch, x_histories, args, meta, device, mode='long')
        x_recent = infer_matrix(x_branch, x_histories, args, meta, device, mode='recent')
        y_long = infer_matrix(y_branch, y_histories, args, meta, device, mode='long')
        y_recent = infer_matrix(y_branch, y_histories, args, meta, device, mode='recent')

        shared_mixed = mix_priors(shared_long, shared_recent, args.recent_mix_eta)
        x_mixed = mix_priors(x_long, x_recent, args.recent_mix_eta)
        y_mixed = mix_priors(y_long, y_recent, args.recent_mix_eta)
        final_prior = build_final_prior(shared_mixed, x_mixed, y_mixed, meta, args.shared_weight_x, args.shared_weight_y)

        valid_metrics = evaluate_prior_matrix(final_prior, valid_records, meta, args.valid_negatives, args.seed + epoch)
        print(
            f'Epoch {epoch:03d} '
            f'shared_loss={shared_loss:.6f} x_loss={x_loss:.6f} y_loss={y_loss:.6f}  '
            f'{pretty_metric_line("valid", valid_metrics)}'
        )

        if valid_metrics.get('MRR', 0.0) > best_valid_mrr:
            best_valid_mrr = valid_metrics['MRR']
            best_epoch = epoch
            wait = 0
            torch.save({
                'shared_state_dict': shared_model.state_dict(),
                'x_state_dict': x_model.state_dict(),
                'y_state_dict': y_model.state_dict(),
                'args': vars(args),
                'meta': meta,
            }, ckpt_path)
        else:
            wait += 1
            if wait >= args.patience:
                print(f'Early stop at epoch {epoch}, best_epoch={best_epoch}, best_valid_mrr={best_valid_mrr:.6f}')
                break

    print(f'Load best checkpoint from epoch {best_epoch}')
    saved = torch.load(ckpt_path, map_location=device)
    args_dict = saved['args']
    shared_model, shared_diff = build_diffusion(args, meta['item_num'], device)
    x_model, x_diff = build_diffusion(args, meta['item_num'], device)
    y_model, y_diff = build_diffusion(args, meta['item_num'], device)
    shared_model.load_state_dict(saved['shared_state_dict'])
    x_model.load_state_dict(saved['x_state_dict'])
    y_model.load_state_dict(saved['y_state_dict'])

    shared_branch = DiffusionBranch('shared', shared_model.eval(), shared_diff)
    x_branch = DiffusionBranch('x', x_model.eval(), x_diff)
    y_branch = DiffusionBranch('y', y_model.eval(), y_diff)

    shared_long = infer_matrix(shared_branch, train_histories, args, meta, device, mode='long')
    shared_recent = infer_matrix(shared_branch, train_histories, args, meta, device, mode='recent')
    x_long = infer_matrix(x_branch, x_histories, args, meta, device, mode='long')
    x_recent = infer_matrix(x_branch, x_histories, args, meta, device, mode='recent')
    y_long = infer_matrix(y_branch, y_histories, args, meta, device, mode='long')
    y_recent = infer_matrix(y_branch, y_histories, args, meta, device, mode='recent')

    shared_mixed = mix_priors(shared_long, shared_recent, args.recent_mix_eta)
    x_mixed = mix_priors(x_long, x_recent, args.recent_mix_eta)
    y_mixed = mix_priors(y_long, y_recent, args.recent_mix_eta)
    final_prior = build_final_prior(shared_mixed, x_mixed, y_mixed, meta, args.shared_weight_x, args.shared_weight_y)

    domain_stats = compute_prior_domain_stats(final_prior, meta['source_item_num'])
    pools = build_top_unseen_pools(
        final_prior,
        train_histories,
        meta['source_item_num'],
        meta['target_item_num'],
        args.top_pool_x,
        args.top_pool_y,
    )
    seen_items = [np.asarray(hist.item_ids, dtype=np.int64) for hist in train_histories]

    valid_metrics = evaluate_prior_matrix(final_prior, valid_records, meta, args.valid_negatives, args.seed + 999)
    test_metrics = evaluate_prior_matrix(final_prior, test_records, meta, args.valid_negatives, args.seed + 1999)

    prior_cache = {
        'meta': meta,
        'args': vars(args),
        'shared_prior_matrix': shared_mixed,
        'x_prior_matrix': x_mixed,
        'y_prior_matrix': y_mixed,
        'final_prior_matrix': final_prior,
        'domain_stats': domain_stats,
        'top_unseen_x': pools['top_unseen_x'],
        'top_unseen_y': pools['top_unseen_y'],
        'train_seen_items': seen_items,
        'valid_metrics': valid_metrics,
        'test_metrics': test_metrics,
    }
    prior_cache_path = os.path.join(args.output_dir, f'{args.data_dir}_diffusion_prior_cache.pkl')
    save_pickle(prior_cache, prior_cache_path)

    print(pretty_metric_line('best valid', valid_metrics))
    print(pretty_metric_line('best test', test_metrics))
    print(f'saved checkpoint: {ckpt_path}')
    print(f'saved prior cache: {prior_cache_path}')

    if not args.skip_plugin_cache and args.plugin_seed is not None:
        build_plugin_cache_after_training(args, device, ckpt_path)


if __name__ == '__main__':
    main()
