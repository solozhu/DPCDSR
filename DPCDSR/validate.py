import argparse
import os
from typing import Dict, List, Tuple

import numpy as np
import torch

from DNN import DNN
import gaussian_diffusion as gd
from hgn_diffusion_utils import (
    UserHistoryInferDataset,
    aggregate_train_histories,
    average_metric_dict,
    evaluate_prior_matrix,
    infer_user_priors,
    load_pickle,
    mix_priors,
    pretty_metric_line,
    ranking_metrics_for_candidates,
    sample_negative_items_same_domain,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, required=True)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--seed', type=int, default=18115153)
    parser.add_argument('--negatives', type=int, default=999)

    parser.add_argument('--prior_cache_path', type=str, default='diffusion_hgn_ckpt/Movie-Book_diffusion_prior_cache.pkl')
    parser.add_argument('--checkpoint_path', type=str, default='diffusion_hgn_ckpt/Movie-Book_diffusion_domainaware_best.pt')

    parser.add_argument('--eval_batch_size', type=int, default=256)
    parser.add_argument('--w_min', type=float, default=None)
    parser.add_argument('--w_max', type=float, default=None)
    parser.add_argument('--reweight_version', type=str, default=None)
    parser.add_argument('--exp_beta', type=float, default=None)
    parser.add_argument('--recent_k', type=int, default=None)
    parser.add_argument('--recent_mix_eta', type=float, default=None)
    parser.add_argument('--sampling_steps', type=int, default=None)
    parser.add_argument('--sampling_noise', action='store_true')
    return parser.parse_args()



def build_diffusion_from_saved(saved: Dict, device: torch.device):
    saved_args = saved['args']
    meta = saved['meta']
    mean_type = gd.ModelMeanType.START_X if saved_args['mean_type'] == 'x0' else gd.ModelMeanType.EPSILON
    diffusion = gd.GaussianDiffusion(
        mean_type,
        saved_args['noise_schedule'],
        saved_args['noise_scale'],
        saved_args['noise_min'],
        saved_args['noise_max'],
        saved_args['steps'],
        str(device),
    ).to(device)
    out_dims = eval(saved_args['dims']) + [meta['item_num']]
    in_dims = out_dims[::-1]
    model = DNN(in_dims, out_dims, saved_args['emb_size'], time_type='cat', norm=saved_args['norm']).to(device)
    model.load_state_dict(saved['model_state_dict'])
    model.eval()
    return model, diffusion, saved_args, meta



def build_prior_matrices_from_checkpoint(args, device: torch.device):
    if not args.checkpoint_path:
        raise ValueError('checkpoint_path is required when prior_cache_path is not provided.')
    if not os.path.exists(args.checkpoint_path):
        raise FileNotFoundError(f'checkpoint not found: {args.checkpoint_path}')

    saved = torch.load(args.checkpoint_path, map_location=device)
    model, diffusion, saved_args, meta = build_diffusion_from_saved(saved, device)
    _, train_histories, _, _ = aggregate_train_histories(args.data_dir)

    w_min = float(args.w_min if args.w_min is not None else saved_args['w_min'])
    w_max = float(args.w_max if args.w_max is not None else saved_args['w_max'])
    reweight_version = str(args.reweight_version if args.reweight_version is not None else saved_args['reweight_version'])
    exp_beta = float(args.exp_beta if args.exp_beta is not None else saved_args['exp_beta'])
    recent_k = int(args.recent_k if args.recent_k is not None else saved_args['recent_k'])
    recent_mix_eta = float(args.recent_mix_eta if args.recent_mix_eta is not None else saved_args.get('recent_mix_eta', 0.6))
    sampling_steps = int(args.sampling_steps if args.sampling_steps is not None else saved_args.get('sampling_steps', 0))
    sampling_noise = bool(args.sampling_noise or saved_args.get('sampling_noise', False))

    long_dataset = UserHistoryInferDataset(
        train_histories, meta['item_num'], w_min, w_max,
        reweight_version, exp_beta, recent_k, mode='long'
    )
    recent_dataset = UserHistoryInferDataset(
        train_histories, meta['item_num'], w_min, w_max,
        reweight_version, exp_beta, recent_k, mode='recent'
    )

    long_priors, long_users = infer_user_priors(
        model, diffusion, long_dataset, device,
        args.eval_batch_size, sampling_steps, sampling_noise
    )
    recent_priors, recent_users = infer_user_priors(
        model, diffusion, recent_dataset, device,
        args.eval_batch_size, sampling_steps, sampling_noise
    )

    long_prior_matrix = np.zeros((meta['user_num'], meta['item_num']), dtype=np.float32)
    recent_prior_matrix = np.zeros((meta['user_num'], meta['item_num']), dtype=np.float32)
    for row, user in zip(long_priors, long_users):
        long_prior_matrix[int(user)] = row
    for row, user in zip(recent_priors, recent_users):
        recent_prior_matrix[int(user)] = row
    mixed_prior_matrix = mix_priors(long_prior_matrix, recent_prior_matrix, recent_mix_eta)
    return {
        'meta': meta,
        'long_prior_matrix': long_prior_matrix,
        'recent_prior_matrix': recent_prior_matrix,
        'mixed_prior_matrix': mixed_prior_matrix,
    }



def evaluate_prior_matrix_xyz(prior_matrix: np.ndarray,
                              records,
                              meta: Dict[str, int],
                              negatives: int,
                              seed: int) -> Dict[str, Dict[str, float]]:
    rng = np.random.RandomState(seed)
    x_metric_list: List[Dict[str, float]] = []
    y_metric_list: List[Dict[str, float]] = []
    cross_metric_list: List[Dict[str, float]] = []

    for rec in records:
        row = prior_matrix[rec.user]
        target = int(rec.target)
        domain = int(rec.target_domain)
        negatives_arr = sample_negative_items_same_domain(
            target, domain, meta['source_item_num'], meta['target_item_num'], negatives, rng
        )
        candidates = np.concatenate([[target], negatives_arr], axis=0)
        scores = row[candidates].astype(np.float64)
        metrics = ranking_metrics_for_candidates(scores, pos_index=0)
        cross_metric_list.append(metrics)
        if domain == 0:
            x_metric_list.append(metrics)
        else:
            y_metric_list.append(metrics)

    return {
        'x': average_metric_dict(x_metric_list),
        'y': average_metric_dict(y_metric_list),
        'cross': average_metric_dict(cross_metric_list),
        'count': {
            'x': len(x_metric_list),
            'y': len(y_metric_list),
            'cross': len(cross_metric_list),
        },
    }



def print_xyz_block(name: str, xyz_metrics: Dict[str, Dict[str, float]]):
    print('=' * 90)
    print(f'prior = {name}')
    print(f'record-count: x={xyz_metrics["count"]["x"]}, y={xyz_metrics["count"]["y"]}, cross={xyz_metrics["count"]["cross"]}')
    print(pretty_metric_line(f'{name} test/x', xyz_metrics['x']))
    print(pretty_metric_line(f'{name} test/y', xyz_metrics['y']))
    print(pretty_metric_line(f'{name} test/cross', xyz_metrics['cross']))



def main():
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

    print('validate diffusion prior quality: --------------------------------------------')
    print(f'data_dir = {args.data_dir}')
    print(f'device = {device}')

    meta, train_histories, valid_records, test_records = aggregate_train_histories(args.data_dir)
    print(f'user_num = {meta["user_num"]}, item_num = {meta["item_num"]}, source_item_num = {meta["source_item_num"]}')
    print(f'test_records = {len(test_records)}')

    if args.prior_cache_path:
        if not os.path.exists(args.prior_cache_path):
            raise FileNotFoundError(f'prior cache not found: {args.prior_cache_path}')
        bundle = load_pickle(args.prior_cache_path)
        print(f'prior_cache_path = {args.prior_cache_path}')
    else:
        bundle = build_prior_matrices_from_checkpoint(args, device)
        print(f'checkpoint_path = {args.checkpoint_path}')

    available = []
    for key in ('long_prior_matrix', 'recent_prior_matrix', 'mixed_prior_matrix'):
        if key in bundle and isinstance(bundle[key], np.ndarray):
            available.append((key.replace('_prior_matrix', ''), bundle[key]))

    if not available:
        raise ValueError('No prior matrix found. Expected one of long/recent/mixed_prior_matrix.')

    for idx, (name, prior_matrix) in enumerate(available):
        xyz_metrics = evaluate_prior_matrix_xyz(
            prior_matrix=prior_matrix,
            records=test_records,
            meta=meta,
            negatives=args.negatives,
            seed=args.seed + 1000 + idx,
        )
        print_xyz_block(name, xyz_metrics)

    if 'mixed_prior_matrix' in bundle:
        overall_test = evaluate_prior_matrix(bundle['mixed_prior_matrix'], test_records, meta, args.negatives, args.seed + 1999)
        print('=' * 90)
        print(pretty_metric_line('compat/test_overall', overall_test))


if __name__ == '__main__':
    main()
