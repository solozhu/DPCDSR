import os
import time
import torch
import argparse
try:
    import ipdb
except ImportError:
    ipdb = None

from model import SASRec_V12_time_final
from model import EarlyStopping_onetower
from model import NTXentLoss

from utils import *
import os
import io

try:
    import matplotlib.pyplot as plt
    plt.switch_backend('agg')
except ImportError:
    plt = None

# -*- coding: UTF-8 -*-
np.set_printoptions(suppress=True)
np.set_printoptions(threshold=2000)

if plt is not None:
    from matplotlib.font_manager import FontManager
    fm = FontManager()
    mat_fonts = set(f.name for f in fm.ttflist)


def str2bool(s):
    if s not in {'false', 'true'}:
        raise ValueError('Not a valid boolean string')
    return s == 'true'
    
# load weights
def get_updateModel(model, path_mix, path_source, path_target):

    pretrained_dict_mix = torch.load(path_mix, map_location='cpu') # 68
    pretrained_dict_source = torch.load(path_source, map_location='cpu') # 68
    pretrained_dict_target = torch.load(path_target, map_location='cpu') # 68
    model_dict = model.state_dict() # 68
    
    shared_dict_mix = {k: v for k, v in pretrained_dict_mix.items() if k.startswith('sasrec_embedding_mix')}# 28
    shared_dict_source = {k: v for k, v in pretrained_dict_source.items() if k.startswith('sasrec_embedding_source')}# 28
    shared_dict_target = {k: v for k, v in pretrained_dict_target.items() if k.startswith('sasrec_embedding_target')}# 28

    model_dict.update(shared_dict_mix)
    model_dict.update(shared_dict_source)
    model_dict.update(shared_dict_target)
    
    print("Load the length of mix is:", len(shared_dict_mix.keys()))
    print("Load the length of source is:", len(shared_dict_source.keys()))
    print("Load the length of target is:", len(shared_dict_target.keys()))

    model.load_state_dict(model_dict)
    return model


def performance_line(epoch, total_time, split, perf, extra=''):
    metrics = " ".join("%s=%.4f" % (key, perf.get(key, 0.0)) for key in ["MRR", "NDCG@5", "NDCG@10", "HR@1", "HR@5", "HR@10"])
    return "epoch=%d %s[%s]%s" % (epoch, split, metrics, extra)


def log_to_file(path, filename, line):
    with io.open(os.path.join(path, filename), 'a', encoding='utf-8') as file:
        file.write(line + '\n')
    

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', required=True)
parser.add_argument('--cross_dataset', required=True)
parser.add_argument('--batch_size', default=120, type=int)
parser.add_argument('--lr', default=0.001, type=float)
parser.add_argument('--maxlen', default=200, type=int)
parser.add_argument('--hidden_units', default=64, type=int)
parser.add_argument('--num_blocks', default=2, type=int)
parser.add_argument('--num_epochs', default=1000, type=int)
parser.add_argument('--num_heads', default=1, type=int)
parser.add_argument('--dropout_rate', default=0.2, type=float)
parser.add_argument('--l2_emb', default=0.0, type=float)
parser.add_argument('--device', default='cuda', type=str)
parser.add_argument('--inference_only', default=False, type=str2bool)
parser.add_argument('--state_dict_path', default=None, type=str)
parser.add_argument('--num_samples', default=1000, type=int)
parser.add_argument('--decay', default=4, type=int)
parser.add_argument('--lr_decay_rate', default=0.99, type=float)
parser.add_argument('--index', default=0, type=int)
parser.add_argument('--version', default=None, type=str)
parser.add_argument('--lr_linear', default=0.01, type=float)
parser.add_argument('--start_decay_linear', default=8, type=int)
parser.add_argument('--temperature', default=5, type=float)
parser.add_argument('--seed', default=5, type=int)
parser.add_argument('--lrscheduler', default='ExponentialLR', type=str)
parser.add_argument('--patience', default=10, type=int)
parser.add_argument('--info_NCE_temperature', default=0.1, type=float)
parser.add_argument('--rec_ratio_cl1', default=2.0, type=float)
parser.add_argument('--rate_mix_source', default=1.0, type=float)
parser.add_argument('--rate_mix_target', default=1.0, type=float)
parser.add_argument('--rate_source_target', default=1.0, type=float)
parser.add_argument('--cl_weight', default=1.0, type=float)
parser.add_argument('--triplet_weight', default=1.0, type=float)
parser.add_argument('--triplet_margin', default=1.0, type=float)
parser.add_argument('--gradproj_root', default=None, type=str)
parser.add_argument('--eval_interval', default=10, type=int)
parser.add_argument('--result_root', default=os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'results')), type=str)
parser.add_argument('--result_dir', default=None, type=str)
parser.add_argument('--sampler_workers', default=6, type=int)
parser.add_argument('--log_every', default=50, type=int)

args = parser.parse_args()
if args.gradproj_root is not None:
    os.environ['GRADPROJ_DATA_ROOT'] = args.gradproj_root


SEED = args.seed
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True
    try:
        torch.set_float32_matmul_precision("high")
    except Exception:
        pass




target_domain = 'x' if args.dataset == 'amazon_toy' else 'y'
if args.result_dir is not None:
    result_path = args.result_dir
elif args.version == 'GradProj':
    result_path = os.path.join(args.result_root, args.cross_dataset, 'Tri-CDR', target_domain)
else:
    result_path = './Log_File_' + str(args.dataset) + '/Tri-CDR/'
if not result_path.endswith(os.sep):
    result_path += os.sep
if not os.path.isdir(result_path):
    os.makedirs(result_path)
with open(os.path.join(result_path, 'args.txt'), 'w') as f:
    f.write('\n'.join([str(k) + ',' + str(v) for k, v in sorted(vars(args).items(), key=lambda x: x[0])]))
# f.close()

if args.cross_dataset == 'Book_Movie':
    source_name = 'book'
    target_name = 'movie'
elif args.cross_dataset == 'Toy_Game':
    source_name = 'toy'
    target_name = 'game'
else:
    source_name = 'x'
    target_name = 'y'

rate_for_mix_source = args.rate_mix_source / (args.rate_mix_source + args.rate_mix_target + args.rate_source_target)
rate_for_mix_target = args.rate_mix_target / (args.rate_mix_source + args.rate_mix_target + args.rate_source_target)
rate_for_source_target = args.rate_source_target / (args.rate_mix_source + args.rate_mix_target + args.rate_source_target)

        
if __name__ == '__main__':
    # global dataset
#     ipdb.set_trace()
#     print(os.getcwd())
    dataset = data_partition(args.version, args.dataset, args.cross_dataset, args.maxlen)

    [user_train_mix, user_train_source, user_train_target, user_valid_target, user_test_target, user_train_mix_sequence_for_target, user_train_source_sequence_for_target, usernum, itemnum, interval] = dataset[:10]
#     [user_train_source, user_train_target, user_valid_source, user_valid_target, user_test_source, user_test_target, usernum, itemnum, interval] = dataset
    num_batch = len(user_train_source) // args.batch_size # 908
    cc_source = 0.0
    cc_target = 0.0
    for u in user_train_source:
        cc_source = cc_source + len(user_train_source[u])
        cc_target = cc_target + len(user_train_target[u])

    sampler = WarpSampler(args.version, args.dataset, args.cross_dataset, interval, user_train_mix, user_train_source, user_train_target, user_train_mix_sequence_for_target, user_train_source_sequence_for_target, usernum, itemnum, None, None, SEED, batch_size=args.batch_size, maxlen=args.maxlen, n_workers=args.sampler_workers)
    model = SASRec_V12_time_final(usernum, itemnum, args).to(args.device)
    for name, param in model.named_parameters():
        try:
            torch.nn.init.xavier_normal_(param.data)
        except:
            pass # just ignore those failed init layers
    
    if args.cross_dataset == 'Toy_Game':
        toy_model_path = './Checkpoints/SASRec_checkpoint_Toy.pt'
        game_model_path = './Checkpoints/SASRec_checkpoint_Game.pt'
        if args.dataset == 'amazon_toy':
            mix_model_path = './Checkpoints/SASRec_checkpoint_Toy_Mix.pt'
        elif args.dataset == 'amazon_game':
            mix_model_path = './Checkpoints/SASRec_checkpoint_Game_Mix.pt'
#         ipdb.set_trace()
        get_updateModel(model, mix_model_path, toy_model_path, game_model_path)

    model.train() # enable model training
    
    bce_criterion = torch.nn.BCEWithLogitsLoss() # torch.nn.BCELoss()
    cl_criterion = NTXentLoss(temperature = args.info_NCE_temperature)
    triplet_criterion = torch.nn.TripletMarginLoss(margin=args.triplet_margin, p=2.0, eps=1e-06, swap=False, size_average=None, reduce=None, reduction='mean')

    adam_optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.98))

    # set the early stop
    early_stopping = EarlyStopping_onetower(args.patience, version='SASRec_V3', verbose=False) 

    # set the learning rate scheduler
    if args.lrscheduler == 'Steplr': # 
        learningrate_scheduler = torch.optim.lr_scheduler.StepLR(adam_optimizer, step_size=args.decay, gamma=args.lr_decay_rate)
    elif args.lrscheduler == 'ExponentialLR': # 
        learningrate_scheduler = torch.optim.lr_scheduler.ExponentialLR(adam_optimizer, gamma=args.lr_decay_rate, last_epoch=-1)
    elif args.lrscheduler == 'CosineAnnealingLR':
        learningrate_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(adam_optimizer, T_max=args.num_epochs, eta_min=0, last_epoch=-1)
    
    T = 0.0
    t0 = time.time()
    epoch_list = []
    lr_list = []
    loss_train_rec_list = []
    loss_train_cl_list = []
    loss_train_triplet_list = []
    loss_train_list = []
    loss_test_list = []
    ndcg_list = []
    hr_list = []
    metric_epoch_list = []
    best_valid = -1.0
    best_epoch = None
    best_test = None
    cl_weight_num = args.cl_weight
    triplet_weight_num = args.triplet_weight
    
    for epoch in range(1, args.num_epochs + 1):
        epoch_list.append(epoch)
        lr_list.append(learningrate_scheduler.get_last_lr())

        t1 = time.time()
        loss_mix_source = 0
        loss_mix_target = 0
        loss_source_target = 0
        loss_rec_epoch = 0
        loss_cl1_epoch = 0
        loss_triplet_epoch = 0
        loss_epoch = 0
        distance_mix_source = 0
        distance_mix_target = 0
        distance_source_target = 0
#         lr_scheduler(epoch, args)
        if args.inference_only: break # just to decrease identition
        for step in range(num_batch): # tqdm(range(num_batch), total=num_batch, ncols=70, leave=False, unit='b'):
# #             ipdb.set_trace()
            u, seq_mix, seq_source, seq_target, pos_target, neg_target, user_train_mix_sequence_for_target_indices, user_train_source_sequence_for_target_indices = sampler.next_batch() # tuples to ndarray
            u, seq_mix, seq_source, seq_target, pos_target, neg_target, user_train_mix_sequence_for_target_indices, user_train_source_sequence_for_target_indices = np.array(u), np.array(seq_mix), np.array(seq_source), np.array(seq_target), np.array(pos_target), np.array(neg_target), np.array(user_train_mix_sequence_for_target_indices), np.array(user_train_source_sequence_for_target_indices)        
#             ipdb.set_trace()
            mix_log_feats, source_log_feats, target_log_feats, pos_logits, neg_logits = model(u, seq_mix, seq_source, seq_target, pos_target, neg_target, user_train_mix_sequence_for_target_indices, user_train_source_sequence_for_target_indices)
            pos_labels, neg_labels = torch.ones(pos_logits.shape, device=args.device), torch.zeros(neg_logits.shape, device=args.device)
            adam_optimizer.zero_grad()
            indices = np.where(pos_target != 0)
            loss_rec = bce_criterion(pos_logits[indices], pos_labels[indices])
            loss_rec += bce_criterion(neg_logits[indices], neg_labels[indices])
            
            cl_loss_mix_source = cl_criterion(mix_log_feats, source_log_feats)
            cl_loss_mix_target = cl_criterion(mix_log_feats, target_log_feats)
            cl_loss_source_target = cl_criterion(source_log_feats, target_log_feats)
            loss_cl1 = cl_loss_mix_source * rate_for_mix_source + cl_loss_mix_target * rate_for_mix_target + cl_loss_source_target * rate_for_source_target

            distance_mix_source_batch = torch.dist(mix_log_feats, source_log_feats, p=2)
            distance_mix_target_batch = torch.dist(mix_log_feats, target_log_feats, p=2)
            distance_source_target_batch = torch.dist(source_log_feats, target_log_feats, p=2)
            
            distance_mix_source += distance_mix_source_batch.item()
            distance_mix_target += distance_mix_target_batch.item()
            distance_source_target += distance_source_target_batch.item()

            loss_triplet = triplet_criterion(source_log_feats, mix_log_feats, target_log_feats)
#             ipdb.set_trace()  
            loss = loss_rec + loss_cl1 * cl_weight_num + loss_triplet * triplet_weight_num
            loss_rec_epoch += loss_rec.item()
            loss_cl1_epoch += loss_cl1.item() * cl_weight_num
            loss_triplet_epoch += loss_triplet.item() * triplet_weight_num
            loss_epoch += loss.item()
            
#             for param in model.item_emb.parameters(): loss += args.l2_emb * torch.norm(param)
            loss.backward()
            adam_optimizer.step()
#             ipdb.set_trace()
        loss_train_rec_list.append(loss_rec_epoch / num_batch)
        loss_train_cl_list.append(loss_cl1_epoch / num_batch)
        loss_train_triplet_list.append(loss_triplet_epoch / num_batch)
        loss_train_list.append(loss_epoch / num_batch)
        learningrate_scheduler.step()
        
        epoch_line = "epoch=%d loss=%.4f" % (epoch, loss_epoch / num_batch)
        print(epoch_line)
        log_to_file(result_path, 'train_loss.txt', epoch_line)
            
        if epoch % args.eval_interval == 0:
            model.eval()
            T = time.time() - t0
            t_valid = evaluate_SASRec(model, dataset, args, split='valid')
            valid_line = performance_line(epoch, T, 'valid', t_valid)
            if t_valid["MRR"] > best_valid:
                best_valid = t_valid["MRR"]
                best_epoch = epoch
                t_test = evaluate_SASRec(model, dataset, args, split='test')
                best_test = t_test
                metric_epoch_list.append(epoch)
                ndcg_list.append(t_test["NDCG@10"])
                hr_list.append(t_test["HR@10"])
                loss_test_list.append(0.0)
                test_line = performance_line(epoch, T, 'test', t_test)
                print(valid_line)
                print(test_line)
                log_to_file(result_path, 'valid_performance.txt', valid_line)
                log_to_file(result_path, 'test_performance.txt', test_line)
            else:
                print(valid_line)
                log_to_file(result_path, 'valid_performance.txt', valid_line)
            model.train()

    if best_test is not None:
        best_line = performance_line(best_epoch, time.time() - t0, 'best_test_by_valid_MRR', best_test)
        print(best_line)
        log_to_file(result_path, 'test_performance.txt', best_line)

    sampler.close()

    if plt is not None:
        plt.figure(figsize=(15, 8)) 
        plt.subplots_adjust(left=0.03, bottom=0.03, right=0.97, top=0.97)
        plt.step(epoch_list, lr_list, "green", marker="8", markersize=5, label="lr")
        plt.legend(loc='upper right')
        plt.savefig(result_path + 'lr_plot.png')
        plt.close()

#     ipdb.set_trace()

        plt.figure(figsize=(15, 8))
        plt.subplots_adjust(left=0.03, bottom=0.03, right=0.97, top=0.97)
        plt.plot(epoch_list, loss_train_rec_list, "black", marker="o", markersize=5, label="loss_train_rec")
        plt.plot(epoch_list, loss_train_cl_list, "green", marker="<", markersize=5, label="loss_train_cl")
        plt.plot(epoch_list, loss_train_triplet_list, "red", marker="v", markersize=5, label="loss_train_triplet")
        plt.plot(epoch_list, loss_train_list, "gray", marker=">", markersize=5, label="loss_train_all")
        plt.plot(metric_epoch_list, loss_test_list, "red", marker="v", markersize=5, label="loss_test_rec")
        plt.plot(metric_epoch_list, ndcg_list, "blue", marker="X", markersize=5, label="ndcg_test")
        plt.plot(metric_epoch_list, hr_list, "yellow", marker="s", markersize=5, label="hr_test")
        plt.legend(loc='upper right')
        plt.savefig(result_path + 'performance_plot.png')
        plt.close()
