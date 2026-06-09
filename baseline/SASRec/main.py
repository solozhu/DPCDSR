import os
import time
import argparse
import tensorflow.compat.v1 as tf
from sampler import WarpSampler
from model import Model
from util import *

tf.disable_v2_behavior()


def str2bool(s):
    if s not in {'False', 'True'}:
        raise ValueError('Not a valid boolean string')
    return s == 'True'


parser = argparse.ArgumentParser()
parser.add_argument('--dataset', required=True)
parser.add_argument('--train_dir', required=True)
parser.add_argument('--batch_size', default=128, type=int)
parser.add_argument('--lr', default=0.001, type=float)
parser.add_argument('--maxlen', default=50, type=int)
parser.add_argument('--hidden_units', default=50, type=int)
parser.add_argument('--num_blocks', default=2, type=int)
parser.add_argument('--num_epochs', default=201, type=int)
parser.add_argument('--num_heads', default=1, type=int)
parser.add_argument('--dropout_rate', default=0.5, type=float)
parser.add_argument('--l2_emb', default=0.0, type=float)
parser.add_argument('--gradproj_root', default=None)
parser.add_argument('--domain', default='x')
parser.add_argument('--eval_interval', default=10, type=int)
parser.add_argument('--result_root', default=os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'results')))
parser.add_argument('--result_dir', default=None)
parser.add_argument('--num_negatives', default=999, type=int)

args = parser.parse_args()
if args.result_dir:
    run_dir = args.result_dir
elif args.gradproj_root:
    run_dir = os.path.join(args.result_root, args.dataset, 'SASRec', args.domain)
else:
    run_dir = args.dataset + '_' + args.train_dir
if not os.path.isdir(run_dir):
    os.makedirs(run_dir)
with open(os.path.join(run_dir, 'args.txt'), 'w') as f:
    f.write('\n'.join([str(k) + ',' + str(v) for k, v in sorted(vars(args).items(), key=lambda x: x[0])]))
f.close()
f = open(os.path.join(run_dir, 'run.txt'), 'w')


def log_print(message, end='\n'):
    print(message, end=end)
    f.write(message + end)
    f.flush()

if args.gradproj_root:
    dataset = data_partition_gradproj(args.gradproj_root, args.dataset, args.domain)
else:
    dataset = data_partition(args.dataset)
[user_train, user_valid, user_test, usernum, itemnum] = dataset[:5]
num_batch = len(user_train) // args.batch_size
cc = 0.0
for u in user_train:
    cc += len(user_train[u])

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
config.allow_soft_placement = True
sess = tf.Session(config=config)

sampler = WarpSampler(user_train, usernum, itemnum, batch_size=args.batch_size, maxlen=args.maxlen, n_workers=3)
model = Model(usernum, itemnum, args)
sess.run(tf.global_variables_initializer())

T = 0.0
t0 = time.time()
best_valid = -1.0
best_test = None
best_epoch = None

try:
    for epoch in range(1, args.num_epochs + 1):

        epoch_loss = 0.0
        for step in range(num_batch):
            u, seq, pos, neg = sampler.next_batch()
            auc, loss, _ = sess.run([model.auc, model.loss, model.train_op],
                                    {model.u: u, model.input_seq: seq, model.pos: pos, model.neg: neg,
                                     model.is_training: True})
            epoch_loss += loss
        epoch_loss /= max(num_batch, 1)
        log_print('epoch=%d loss=%.4f' % (epoch, epoch_loss))

        if epoch % args.eval_interval == 0:
            t1 = time.time() - t0
            T += t1
            t_valid = evaluate_valid(model, dataset, args, sess)
            valid_line = 'epoch=%d valid[%s]' % (epoch, format_metrics(t_valid))
            if t_valid["MRR"] > best_valid:
                best_valid = t_valid["MRR"]
                best_epoch = epoch
                t_test = evaluate(model, dataset, args, sess)
                best_test = t_test
                log_print(valid_line)
                log_print('epoch=%d test[%s]' % (epoch, format_metrics(t_test)))
            else:
                log_print(valid_line)

            t0 = time.time()
except:
    sampler.close()
    f.close()
    exit(1)

sampler.close()
if best_test is not None:
    log_print('best_epoch=%d best_test_by_valid_MRR[%s]' % (best_epoch, format_metrics(best_test)))
f.close()
