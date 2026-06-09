import argparse
import datetime
import hashlib
import os
import pickle
import random
from types import SimpleNamespace
from time import time
from typing import Dict, Optional, Tuple

import numpy as np
import torch

from cdr_model import HGN_CDR
from dataProcessing import DataLoader


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


def resolve_run_seed(seed_arg):
    if seed_arg is not None:
        return int(seed_arg)
    return int.from_bytes(os.urandom(4), byteorder='big') % (2 ** 31 - 1)


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


def default_runtime_cache_path(data_dir: str, seed: int, batch_size: int, neg_samples: int) -> str:
    return f'./diffusion_hgn_ckpt/{data_dir}_plugin_cache_seed{seed}_bs{batch_size}_neg{neg_samples}.pkl'


def ensure_diffusion_plugin_cache(conf, dev: torch.device):
    if os.path.exists(conf.diffusion_cache_path):
        return

    from train_diffusion_hgn import build_plugin_cache_after_training, default_diffusion_ckpt

    output_dir = './diffusion_hgn_ckpt'
    ckpt_path = default_diffusion_ckpt(conf.data_dir, output_dir)
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(
            f'diffusion checkpoint not found: {ckpt_path}. Please run train_diffusion_hgn.py for {conf.data_dir} first.'
        )

    print(f'diffusion plugin cache not found: {conf.diffusion_cache_path}')
    print(f'auto building diffusion plugin cache for batch_size={conf.batch_size}, neg_samples={conf.neg_samples}')

    cache_args = SimpleNamespace(
        data_dir=conf.data_dir,
        output_dir=output_dir,
        seed=int(conf.seed),
        plugin_seed=int(conf.seed),
        plugin_batch_size=int(conf.batch_size),
        plugin_neg_samples=int(conf.neg_samples),
        plugin_cache_dtype='float16',
        plugin_cache_path=conf.diffusion_cache_path,
        plugin_score_short_k=int(getattr(conf, 'diffusion_score_short_k', 5)),
        w_min=0.1,
        w_max=1.0,
        reweight_version='ExpDecay',
        exp_beta=3.0,
        recent_k=30,
        recent_mix_eta=0.6,
        shared_weight_x=0.35,
        shared_weight_y=0.35,
        dims='[1000]',
        norm=False,
        emb_size=10,
        mean_type='x0',
        steps=10,
        noise_schedule='linear-var',
        noise_scale=0.01,
        noise_min=0.0005,
        noise_max=0.005,
        sampling_steps=0,
        sampling_noise=False,
        reweight=True,
        eval_batch_size=256,
        valid_negatives=999,
        top_pool_x=200,
        top_pool_y=200,
    )
    build_plugin_cache_after_training(cache_args, dev, ckpt_path)


def pred_indices_from_scores_topk(prediction_score: np.ndarray, topk: int) -> np.ndarray:
    ind = np.argpartition(prediction_score, -topk)
    ind = ind[:, -topk:]
    arr_ind = prediction_score[np.arange(len(prediction_score))[:, None], ind]
    arr_ind_argsort = np.argsort(arr_ind)[np.arange(len(prediction_score)), ::-1]
    return ind[np.arange(len(prediction_score))[:, None], arr_ind_argsort]


def _single_domain_metric_dict(pred_list: np.ndarray, topk: int = 10) -> dict:
    num = int(pred_list.shape[0])
    if num == 0:
        return {'MRR': 0.0, 'NDCG@5': 0.0, 'NDCG@10': 0.0, 'HR@1': 0.0, 'HR@5': 0.0, 'HR@10': 0.0}
    mrr = hr1 = hr5 = hr10 = ndcg5 = ndcg10 = 0.0
    for row in pred_list:
        row = np.asarray(row)
        pos = np.where(row == 0)[0]
        if pos.size == 0:
            continue
        rank = int(pos[0])
        mrr += 1.0 / float(rank + 1)
        if rank < 1:
            hr1 += 1.0
        if rank < 5:
            hr5 += 1.0
            ndcg5 += 1.0 / np.log2(rank + 2.0)
        if rank < 10:
            hr10 += 1.0
            ndcg10 += 1.0 / np.log2(rank + 2.0)
    denom = float(num)
    return {
        'MRR': mrr / denom,
        'NDCG@5': ndcg5 / denom,
        'NDCG@10': ndcg10 / denom,
        'HR@1': hr1 / denom,
        'HR@5': hr5 / denom,
        'HR@10': hr10 / denom,
    }


def print_domain_metrics(title: str, x_metrics: dict, y_metrics: dict):
    print(title)
    print([
        [x_metrics['MRR'], y_metrics['MRR']],
        [x_metrics['NDCG@5'], y_metrics['NDCG@5']],
        [x_metrics['NDCG@10'], y_metrics['NDCG@10']],
        [x_metrics['HR@1'], y_metrics['HR@1']],
        [x_metrics['HR@5'], y_metrics['HR@5']],
        [x_metrics['HR@10'], y_metrics['HR@10']],
    ])


def cache_fingerprint(meta: Dict) -> str:
    raw = pickle.dumps(meta, protocol=pickle.HIGHEST_PROTOCOL)
    return hashlib.md5(raw).hexdigest()


def _to_torch(arr: np.ndarray, dev: torch.device) -> torch.Tensor:
    if arr.dtype in (np.float16, np.float32, np.float64):
        return torch.from_numpy(arr.astype(np.float32)).to(dev)
    return torch.from_numpy(arr).to(dev)


def _np_equal(a: np.ndarray, b: np.ndarray) -> bool:
    return a.shape == b.shape and np.array_equal(a, b)


class CachedDiffusionPlugin:
    def __init__(self, cache: Dict, dev: torch.device):
        self.cache = cache
        self.device = dev
        self.meta = cache['meta']

    @classmethod
    def load(cls, path: str, conf, dev: torch.device):
        if not os.path.exists(path):
            raise FileNotFoundError(
                f'plugin cache not found: {path}. Please rerun train_diffusion_hgn.py to generate the offline plugin cache first.'
            )
        with open(path, 'rb') as f:
            cache = pickle.load(f)
        meta = cache['meta']
        expected = {
            'data_dir': conf.data_dir,
            'Rseed': int(conf.seed),
            'batch_size': int(conf.batch_size),
            'neg_samples': int(conf.neg_samples),
            'source_item_num': int(conf.source_item_num),
            'target_item_num': int(conf.target_item_num),
            'item_num': int(conf.item_num),
            'plugin_item_num': int(conf.item_num - 1),
            'diffusion_ckpt_path': meta.get('diffusion_ckpt_path', ''),
            'cache_dtype': meta['cache_dtype'],
            'train_batches': meta['train_batches'],
            'valid_x_batches': meta['valid_x_batches'],
            'valid_y_batches': meta['valid_y_batches'],
            'test_x_batches': meta['test_x_batches'],
            'test_y_batches': meta['test_y_batches'],
        }
        if meta.get('fingerprint') != cache_fingerprint(expected):
            raise ValueError('plugin cache meta mismatch. Please rebuild cache with the current seed/batch_size/neg_samples/data_dir.')
        return cls(cache, dev)

    def get_batch(self, split_name: str, batch_idx: int, seq: torch.Tensor, x_seq: torch.Tensor, y_seq: torch.Tensor,
                  ground: torch.Tensor, user: torch.Tensor, candidate_items: torch.Tensor) -> Dict[str, torch.Tensor]:
        rows = self.cache[split_name]
        if batch_idx >= len(rows):
            raise IndexError(f'cache split {split_name} batch_idx out of range: {batch_idx} >= {len(rows)}')
        row = rows[batch_idx]
        sig = row['signature']
        if not _np_equal(sig['seq'], seq.detach().cpu().numpy()):
            raise ValueError(f'plugin cache seq mismatch at split={split_name}, batch={batch_idx}')
        if not _np_equal(sig['x_seq'], x_seq.detach().cpu().numpy()):
            raise ValueError(f'plugin cache x_seq mismatch at split={split_name}, batch={batch_idx}')
        if not _np_equal(sig['y_seq'], y_seq.detach().cpu().numpy()):
            raise ValueError(f'plugin cache y_seq mismatch at split={split_name}, batch={batch_idx}')
        if not _np_equal(sig['ground'], ground.detach().cpu().numpy()):
            raise ValueError(f'plugin cache ground mismatch at split={split_name}, batch={batch_idx}')
        if not _np_equal(sig['user'], user.detach().cpu().numpy()):
            raise ValueError(f'plugin cache user mismatch at split={split_name}, batch={batch_idx}')
        if 'candidate_items' in sig and not _np_equal(sig['candidate_items'], candidate_items.detach().cpu().numpy()):
            raise ValueError(f'plugin cache candidate_items mismatch at split={split_name}, batch={batch_idx}')

        long_scores = row.get('candidate_scores_long', row.get('candidate_scores'))
        short_scores = row.get('candidate_scores_short')
        if short_scores is None and long_scores is not None:
            short_scores = np.zeros_like(long_scores)

        return {
            'feature_bias_seq': _to_torch(row['feature_bias_seq'], self.device),
            'feature_bias_x': _to_torch(row['feature_bias_x'], self.device),
            'feature_bias_y': _to_torch(row['feature_bias_y'], self.device),
            'candidate_scores_long': _to_torch(long_scores, self.device),
            'candidate_scores_short': _to_torch(short_scores, self.device),
        }


def plugin_strength(conf, epoch_num: int) -> float:
    start_epoch = int(getattr(conf, 'diffusion_start_epoch', 0))
    ramp_epochs = max(int(getattr(conf, 'diffusion_ramp_epochs', 1)), 1)
    if epoch_num < start_epoch:
        return 0.0
    return min(float(epoch_num - start_epoch + 1) / float(ramp_epochs), 1.0)


def normalize_by_mask(values: torch.Tensor, mask: torch.Tensor, clip_value: float = 3.0) -> torch.Tensor:
    denom = mask.float().sum(dim=1, keepdim=True).clamp(min=1.0)
    mean = (values * mask.float()).sum(dim=1, keepdim=True) / denom
    centered = (values - mean) * mask.float()
    var = (centered * centered).sum(dim=1, keepdim=True) / denom
    std = torch.sqrt(var + 1e-6)
    norm = centered / std
    norm = torch.clamp(norm, -clip_value, clip_value)
    return torch.tanh(norm) * mask.float()


def _prepare_gate_bias(bias: Optional[torch.Tensor], positive_only: bool) -> Optional[torch.Tensor]:
    if bias is None:
        return None
    return torch.relu(bias) if positive_only else bias


def branch_gate_kwargs(plugin_batch: Optional[Dict[str, torch.Tensor]], conf, ramp: float):
    if plugin_batch is None or ramp <= 0.0 or not conf.use_diffusion_gate:
        return {}
    positive_only = bool(getattr(conf, 'diffusion_gate_positive_only', True))
    return {
        'feature_bias_seq': _prepare_gate_bias(plugin_batch['feature_bias_seq'], positive_only),
        'feature_bias_x': _prepare_gate_bias(plugin_batch['feature_bias_x'], positive_only),
        'feature_bias_y': _prepare_gate_bias(plugin_batch['feature_bias_y'], positive_only),
    }


def maybe_fuse_cross_scores(model: HGN_CDR, base_scores: torch.Tensor, candidate_items: torch.Tensor,
                            plugin_batch: Optional[Dict[str, torch.Tensor]], row_domain: torch.Tensor,
                            user: torch.Tensor, conf, ramp: float) -> Tuple[torch.Tensor, Optional[Dict[str, torch.Tensor]]]:
    if plugin_batch is None or ramp <= 0.0 or not conf.use_diffusion_score:
        return base_scores, None

    valid = candidate_items.ne(conf.item_num - 1)
    long_scores = plugin_batch.get('candidate_scores_long')
    short_scores = plugin_batch.get('candidate_scores_short')
    if long_scores is None and short_scores is None:
        return base_scores, None
    if long_scores is None:
        long_scores = torch.zeros_like(short_scores)
    if short_scores is None:
        short_scores = torch.zeros_like(long_scores)

    long_norm = normalize_by_mask(long_scores, valid, clip_value=conf.diffusion_score_z_clip)
    short_norm = normalize_by_mask(short_scores, valid, clip_value=conf.diffusion_score_z_clip)

    ablation_mode = str(getattr(conf, 'diffusion_score_ablation', 'both'))
    if ablation_mode == 'long_only':
        short_norm = torch.zeros_like(short_norm)
    elif ablation_mode == 'short_only':
        long_norm = torch.zeros_like(long_norm)

    aux = model.adapt_diffusion_scores(user, row_domain, long_norm, short_norm, valid)
    residual = aux['residual']
    fused = base_scores + float(ramp) * residual
    aux['base_scores'] = base_scores
    return fused, aux


def diffusion_safe_loss(score_aux: Optional[Dict[str, torch.Tensor]], pos_width: int = 1) -> torch.Tensor:
    if score_aux is None:
        return torch.tensor(0.0, device=device)
    residual = score_aux['residual']
    valid = score_aux['valid']
    base_scores = score_aux['base_scores']
    pos_base = base_scores[:, :pos_width].mean(dim=1)
    neg_base = base_scores[:, pos_width:]
    neg_valid = valid[:, pos_width:]
    neg_denom = neg_valid.sum(dim=1).clamp(min=1.0)
    neg_mean = (neg_base * neg_valid).sum(dim=1) / neg_denom
    base_margin = pos_base - neg_mean
    safe_weight = torch.sigmoid(base_margin.detach())
    residual_energy = (residual.pow(2) * valid).sum(dim=1) / valid.sum(dim=1).clamp(min=1.0)
    return (safe_weight * residual_energy).mean()


def plugin_coeff_line(model: HGN_CDR) -> str:
    parts = []
    if hasattr(model, 'get_diffusion_gate_values_detached'):
        gate_vals = model.get_diffusion_gate_values_detached()
        if any(abs(v) > 0.0 for v in gate_vals.values()):
            gate_order = ['feature_mix_lambda', 'feature_alpha_seq', 'feature_alpha_x', 'feature_alpha_y']
            parts.append('learned_gate: ' + ', '.join([f'{k}={gate_vals[k]:.4f}' for k in gate_order]))
    if hasattr(model, 'get_adapter_debug_values'):
        vals = model.get_adapter_debug_values()
        if vals:
            keys = sorted(vals.keys())
            parts.append('score_adapter: ' + ', '.join([f'{k}={vals[k]:.4f}' for k in keys]))
    return ' | '.join(parts)


def evaluate_single_domain(model, test_data, split_name: str, pred_domain: str, conf,
                           plugin: Optional[CachedDiffusionPlugin], epoch_num: int, topk: int = 10):
    pred_list = None
    first_batch = True
    model.eval()
    with torch.no_grad():
        for batch_idx, batch in enumerate(test_data):
            inputs = [b.to(device) for b in batch]
            seq = inputs[0]
            x_seq = inputs[1]
            y_seq = inputs[2]
            ground = inputs[3]
            position = inputs[4]
            x_position = inputs[5]
            y_position = inputs[6]
            user = inputs[7]
            x_flag = inputs[8]
            neg = inputs[10]
            items_to_predict = torch.cat((ground, neg), 1)
            ramp = plugin_strength(conf, epoch_num)
            plugin_batch = None
            if plugin is not None and ramp > 0.0:
                plugin_batch = plugin.get_batch(split_name, batch_idx, seq, x_seq, y_seq, ground, user, items_to_predict)
            gate_kwargs = branch_gate_kwargs(plugin_batch, conf, ramp)
            prediction_score = model(
                seq, x_seq, y_seq, position, x_position, y_position, user, items_to_predict,
                for_pred=True, pred_domain=pred_domain, **gate_kwargs
            )
            row_domain = (ground.squeeze(1) >= conf.source_item_num).long()
            prediction_score, _ = maybe_fuse_cross_scores(
                model, prediction_score, items_to_predict, plugin_batch, row_domain, user, conf, ramp
            )
            prediction_score = prediction_score.detach().cpu().numpy().copy()
            batch_pred_list = pred_indices_from_scores_topk(prediction_score, topk)
            if first_batch:
                pred_list = batch_pred_list
                flag = x_flag.cpu().numpy().copy()
                first_batch = False
            else:
                pred_list = np.append(pred_list, batch_pred_list, axis=0)
                flag = np.append(flag, x_flag.cpu().numpy().copy(), axis=0)
    return _single_domain_metric_dict(pred_list, topk=topk)


def train_model(model, optim, train_data, valid_data_x, valid_data_y, test_data_x, test_data_y, conf,
                plugin: Optional[CachedDiffusionPlugin] = None):
    max_MRR_val_X = 0.0
    max_MRR_val_Y = 0.0

    for epoch_num in range(conf.n_iter):
        t1 = time()
        model.train()
        epoch_loss = 0.0
        num_batches = 0

        for batch_idx, batch in enumerate(train_data):
            num_batches += 1
            inputs = [b.to(device) for b in batch]
            seq = inputs[0]
            x_seq = inputs[1]
            y_seq = inputs[2]
            ground = inputs[3]
            x_ground = inputs[4]
            y_ground = inputs[5]
            position = inputs[6]
            x_position = inputs[7]
            y_position = inputs[8]
            user = inputs[9]
            neg = inputs[12]
            x_neg = inputs[13]
            y_neg = inputs[14]

            items_to_predict = torch.cat((ground, neg), 1)
            x_items_to_predict = torch.cat((x_ground, x_neg), 1)
            y_items_to_predict = torch.cat((y_ground, y_neg), 1)

            ramp = plugin_strength(conf, epoch_num + 1)
            plugin_batch = None
            if plugin is not None and ramp > 0.0:
                plugin_batch = plugin.get_batch('train', batch_idx, seq, x_seq, y_seq, ground, user, items_to_predict)
            gate_kwargs = branch_gate_kwargs(plugin_batch, conf, ramp)

            prediction_score, x_prediction_score, y_prediction_score = model(
                seq, x_seq, y_seq, position, x_position, y_position, user,
                items_to_predict, x_items_to_predict, y_items_to_predict, False,
                **gate_kwargs
            )

            row_domain = (ground.squeeze(1) >= conf.source_item_num).long()
            prediction_score, score_aux = maybe_fuse_cross_scores(
                model, prediction_score, items_to_predict, plugin_batch, row_domain, user, conf, ramp
            )

            aims_prediction, negatives_prediction = torch.split(prediction_score, [ground.size(1), neg.size(1)], dim=1)
            x_aims_prediction, x_negatives_prediction = torch.split(x_prediction_score, [x_ground.size(1), x_neg.size(1)], dim=1)
            y_aims_prediction, y_negatives_prediction = torch.split(y_prediction_score, [y_ground.size(1), y_neg.size(1)], dim=1)

            cross_loss_matrix = -torch.log(torch.sigmoid(aims_prediction - negatives_prediction) + 1e-8)
            x_loss_matrix = -torch.log(torch.sigmoid(x_aims_prediction - x_negatives_prediction) + 1e-8)
            y_loss_matrix = -torch.log(torch.sigmoid(y_aims_prediction - y_negatives_prediction) + 1e-8)

            cross_loss = torch.sum(cross_loss_matrix)
            x_loss = torch.sum(x_loss_matrix)
            y_loss = torch.sum(y_loss_matrix)
            loss_all = conf.lamb * cross_loss + (1.0 - conf.lamb) * (x_loss + y_loss)

            if conf.use_diffusion and conf.use_diffusion_score and score_aux is not None:
                score_constraint_scale = float(ground.size(0) * items_to_predict.size(1))
                loss_all = loss_all + float(conf.diffusion_residual_reg) * score_constraint_scale * score_aux['residual_reg']
                loss_all = loss_all + float(conf.diffusion_safe_reg) * score_constraint_scale * diffusion_safe_loss(score_aux, pos_width=ground.size(1))

            if conf.use_diffusion and conf.use_diffusion_gate and plugin_batch is not None and ramp > 0.0:
                gate_vals = model.get_diffusion_gate_values()
                gate_reg = (
                    (gate_vals['feature_mix_lambda'] - float(conf.diffusion_feature_mix_init)) ** 2 +
                    (gate_vals['feature_alpha_seq'] - float(conf.diffusion_feature_init_seq)) ** 2 +
                    (gate_vals['feature_alpha_x'] - float(conf.diffusion_feature_init_x)) ** 2 +
                    (gate_vals['feature_alpha_y'] - float(conf.diffusion_feature_init_y)) ** 2
                )
                loss_all = loss_all + float(conf.diffusion_gate_reg) * gate_reg

            epoch_loss += loss_all.item()
            optim.zero_grad()
            loss_all.backward()
            optim.step()

        epoch_loss /= max(num_batches, 1)
        t2 = time()
        print(f'Epoch {epoch_num + 1:03d} [{t2 - t1:.1f} s]  loss={epoch_loss:.4f}')

        if (epoch_num + 1) % 5 == 0 or (75 <= epoch_num + 1 <= 90):
            valid_x = evaluate_single_domain(model, valid_data_x, 'valid_x', 'x', conf, plugin, epoch_num + 1)
            valid_y = evaluate_single_domain(model, valid_data_y, 'valid_y', 'y', conf, plugin, epoch_num + 1)
            print_domain_metrics('valid data evaluation: ---- MRR, NDCG@5, NDCG@10, HR@1, HR@5, HR@10', valid_x, valid_y)
            if conf.use_diffusion and (conf.use_diffusion_gate or conf.use_diffusion_score):
                line = plugin_coeff_line(model)
                if line:
                    print(line)
            MRR_val_X = valid_x['MRR']
            MRR_val_Y = valid_y['MRR']
            need_test = (MRR_val_X >= max_MRR_val_X) or (MRR_val_Y >= max_MRR_val_Y)
            if need_test:
                test_x = evaluate_single_domain(model, test_data_x, 'test_x', 'x', conf, plugin, epoch_num + 1)
                test_y = evaluate_single_domain(model, test_data_y, 'test_y', 'y', conf, plugin, epoch_num + 1)
            else:
                test_x, test_y = None, None
            if MRR_val_X >= max_MRR_val_X:
                max_MRR_val_X = MRR_val_X
                print('X best! ---- MRR, NDCG@5, NDCG@10, HR@1, HR@5, HR@10')
                print([test_x['MRR'], test_x['NDCG@5'], test_x['NDCG@10'], test_x['HR@1'], test_x['HR@5'], test_x['HR@10']])
            if MRR_val_Y >= max_MRR_val_Y:
                max_MRR_val_Y = MRR_val_Y
                print('Y best! ---- MRR, NDCG@5, NDCG@10, HR@1, HR@5, HR@10')
                print([test_y['MRR'], test_y['NDCG@5'], test_y['NDCG@10'], test_y['HR@1'], test_y['HR@5'], test_y['HR@10']])
    print()


def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--L', type=int, default=15)
    parser.add_argument('--d', type=int, default=256)
    parser.add_argument('--maxlen', type=int, default=15)
    parser.add_argument('--n_iter', type=int, default=200)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--learning_rate', type=float, default=5e-4)
    parser.add_argument('--l2', type=float, default=1e-3)
    parser.add_argument('--neg_samples', type=int, default=99)
    parser.add_argument('--lamb', type=float, default=0.8)
    parser.add_argument('--data_dir', type=str, default='Food-Kitchen')
    parser.add_argument('--Rseed', type=int, default=None)

    parser.add_argument('--use_diffusion', action='store_true')
    parser.add_argument('--use_diffusion_gate', action='store_true')
    parser.add_argument('--use_diffusion_score', action='store_true')
    parser.add_argument('--diffusion_cache_path', type=str, default='')

    parser.add_argument('--diffusion_start_epoch', type=int, default=None)
    parser.add_argument('--diffusion_ramp_epochs', type=int, default=None)

    parser.add_argument('--diffusion_gate_positive_only', dest='diffusion_gate_positive_only', action='store_true')
    parser.add_argument('--diffusion_gate_allow_negative', dest='diffusion_gate_positive_only', action='store_false')
    parser.set_defaults(diffusion_gate_positive_only=True)
    parser.add_argument('--diffusion_feature_mix_init', type=float, default=0.99)
    parser.add_argument('--diffusion_feature_init_seq', type=float, default=4)
    parser.add_argument('--diffusion_feature_init_x', type=float, default=5)
    parser.add_argument('--diffusion_feature_init_y', type=float, default=5)
    parser.add_argument('--diffusion_gate_reg', type=float, default=1e-4)
    parser.add_argument('--diffusion_plugin_lr_ratio', type=float, default=0.3)

    parser.add_argument('--diffusion_adapter_hidden', type=int, default=64)
    parser.add_argument('--diffusion_adapter_dropout', type=float, default=0.0)
    parser.add_argument('--diffusion_adapter_detach_context', action='store_true')
    parser.add_argument('--diffusion_controller_detach_context', dest='diffusion_adapter_detach_context', action='store_true')
    parser.add_argument('--diffusion_adapter_lr_ratio', type=float, default=0.2)
    parser.add_argument('--diffusion_adapter_weight_decay', type=float, default=1e-4)
    parser.add_argument('--diffusion_score_residual_max', type=float, default=1.0)
    parser.add_argument('--diffusion_residual_reg', type=float, default=1e-3)
    parser.add_argument('--diffusion_safe_reg', type=float, default=1e-3)
    parser.add_argument('--diffusion_score_short_k', type=int, default=5)
    parser.add_argument('--diffusion_score_z_clip', type=float, default=2.0)
    parser.add_argument('--diffusion_score_ablation', type=str, default='both', choices=['both', 'long_only', 'short_only'])
    return parser


def main():
    parser = build_parser()
    config = parser.parse_args()
    run_seed = resolve_run_seed(config.Rseed)
    config.seed = run_seed
    set_global_random_seed(run_seed)

    if config.use_diffusion and not (config.use_diffusion_gate or config.use_diffusion_score):
        config.use_diffusion_score = True

    if config.diffusion_start_epoch is None:
        config.diffusion_start_epoch = 5 if (config.use_diffusion_gate and not config.use_diffusion_score) else 20
    if config.diffusion_ramp_epochs is None:
        config.diffusion_ramp_epochs = 5 if (config.use_diffusion_gate and not config.use_diffusion_score) else 20

    train_data = DataLoader(config.data_dir, config.batch_size, config, evaluation=-1)
    valid_data_x = DataLoader(config.data_dir, config.batch_size, config, evaluation=2, predict_domain='x')
    valid_data_y = DataLoader(config.data_dir, config.batch_size, config, evaluation=2, predict_domain='y')
    test_data_x = DataLoader(config.data_dir, config.batch_size, config, evaluation=1, predict_domain='x')
    test_data_y = DataLoader(config.data_dir, config.batch_size, config, evaluation=1, predict_domain='y')

    print('recommendation: -------------------------------------------------------------------')
    print(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
    print('Data loading done!')

    config.item_num = config.source_item_num + config.target_item_num + 1
    model = HGN_CDR(config, device).to(device)

    named_params = dict(model.named_parameters())
    score_names = model.score_adapter_param_names() if hasattr(model, 'score_adapter_param_names') else set()
    gate_names = model.gate_param_names() if hasattr(model, 'gate_param_names') else set()
    special_names = score_names | gate_names
    base_params = [p for n, p in named_params.items() if n not in special_names]
    optimizer_groups = [
        {'params': base_params, 'lr': config.learning_rate, 'weight_decay': config.l2},
    ]
    gate_params = [p for n, p in named_params.items() if n in gate_names]
    if gate_params:
        optimizer_groups.append({
            'params': gate_params,
            'lr': config.learning_rate * float(config.diffusion_plugin_lr_ratio),
            'weight_decay': 0.0,
        })
    score_params = [p for n, p in named_params.items() if n in score_names]
    if score_params:
        optimizer_groups.append({
            'params': score_params,
            'lr': config.learning_rate * float(config.diffusion_adapter_lr_ratio),
            'weight_decay': float(config.diffusion_adapter_weight_decay),
        })
    optimizer = torch.optim.Adam(optimizer_groups)

    plugin = None
    if config.use_diffusion:
        if not config.diffusion_cache_path:
            config.diffusion_cache_path = default_runtime_cache_path(config.data_dir, run_seed, config.batch_size, config.neg_samples)
        ensure_diffusion_plugin_cache(config, device)
        plugin = CachedDiffusionPlugin.load(config.diffusion_cache_path, config, device)
        print(f'loaded diffusion plugin cache: {config.diffusion_cache_path}')
        print(f'diffusion plugin enabled -> gate={config.use_diffusion_gate}, score={config.use_diffusion_score}')

    print(config)
    print(device)
    train_model(model, optimizer, train_data, valid_data_x, valid_data_y, test_data_x, test_data_y, config, plugin=plugin)


if __name__ == '__main__':
    main()
