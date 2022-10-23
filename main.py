import argparse
import torch
import os
from tqdm import tqdm
from dataset import *
from model import TemporalTransformerHawkesGraphModel
import logging
from collections import namedtuple
from torch.utils.data import DataLoader
from utils import set_logger
import pickle
import math

def parse_args(args=None):
    parser = argparse.ArgumentParser(
        description='Training and Testing Temporal Knowledge Graph Reasoning Models',
        usage='main.py [<args>] [-h | --help]'
    )

    parser.add_argument('--data_root', type=str, default='data')
    parser.add_argument('--output_root', type=str, default='output')
    parser.add_argument('--model_name', type=str, default='GHT')
    parser.add_argument('--batch_size', type=int, default=256)

    parser.add_argument('--num_works', type=int, default=8)

    parser.add_argument('--grad_norm', type=float, default=1.0)
    parser.add_argument('--weight_decay', type=float, default=0.00001)

    parser.add_argument('--d_model', default=100, type=int)
    parser.add_argument('--data', default='ICEWS14', type=str)
    parser.add_argument('--max_epochs', default=30, type=int)
    parser.add_argument('--lr', default=0.003, type=float)
    parser.add_argument('--do_train', action='store_true')
    parser.add_argument('--do_test', action='store_true')
    parser.add_argument('--valid_epoch', default=1, type=int)
    parser.add_argument('--history_len', default=10, type=int)
    parser.add_argument('--dropout', default=0.2, type=float)

    parser.add_argument('--seqTransformerLayerNum', default=2, type=int)
    parser.add_argument('--seqTransformerHeadNum', default=2, type=int)

    parser.add_argument('--load_model_path', default='output', type=str)

    parser.add_argument('--history_mode', default='delta_t_windows', type=str)
    parser.add_argument('--nhop', default=1, type=int)
    parser.add_argument('--forecasting_t_win_size', default=1, type=int)

    parser.add_argument('--alpha', default=0.5, type=float)
    parser.add_argument('--beta', default=1.0, type=float)

    parser.add_argument('--time_span', default=24, type=int)
    parser.add_argument('--timestep', default=0.1, type=float)
    parser.add_argument('--hmax', default=5, type=int)
    parser.add_argument('--eps', default=0.1, type=float)
    parser.add_argument('--edge_sample', default='one_hop_conf', type=str)
    parser.add_argument('--desc', default='', type=str)

    parser.add_argument('--warm_up', default=0.0, type=float)

    return parser.parse_args(args)

def test(model, testloader, skip_dict, device):
    model.eval()
    ranks = []
    logs = []
    TimeMSE = 0.
    TimeMAE = 0.
    with torch.no_grad():
        for sub, rel, obj, time, history_graphs, history_times, batch_node_ids in tqdm(testloader):
            sub = sub.to(device, non_blocking=True)
            rel = rel.to(device, non_blocking=True)
            obj = obj.to(device, non_blocking=True)
            time = time.to(device, non_blocking=True)
            history_graphs = history_graphs.to(device, non_blocking=True)
            history_times = history_times.to(device, non_blocking=True)
            batch_node_ids = batch_node_ids.to(device, non_blocking=True)

            scores, estimate_dt, dur_last = model.test_forward(sub, rel, obj, time, history_graphs, history_times, batch_node_ids,
                                                               args.beta)

            mse_loss = torch.nn.MSELoss(reduction='sum')(estimate_dt, dur_last)
            mae_loss = torch.nn.L1Loss(reduction='sum')(estimate_dt, dur_last)

            TimeMSE += mse_loss
            TimeMAE += mae_loss

            _, rank_idx = scores.sort(dim=1, descending=True)
            rank = torch.nonzero(rank_idx == obj.view(-1, 1))[:, 1].view(-1)
            ranks.append(rank)

            for i in range(scores.shape[0]):
                src_i = sub[i].item()
                rel_i = rel[i].item()
                dst_i = obj[i].item()
                time_i = time[i].item()

                predict_score = scores[i].tolist()
                answer_prob = predict_score[dst_i]
                for e in skip_dict[(src_i, rel_i, time_i)]:
                    if e != dst_i:
                        predict_score[e] = -1e6
                predict_score.sort(reverse=True)
                filter_rank = predict_score.index(answer_prob) + 1

                logs.append({
                    'Time-aware Filter MRR': 1.0 / filter_rank,
                    'Time-aware Filter HITS@1': 1.0 if filter_rank <= 1 else 0.0,
                    'Time-aware Filter HITS@3': 1.0 if filter_rank <= 3 else 0.0,
                    'Time-aware Filter HITS@10': 1.0 if filter_rank <= 10 else 0.0,
                })

    metrics = {}
    ranks = torch.cat(ranks)
    ranks += 1
    mrr = torch.mean(1.0 / ranks.float())
    metrics['Raw MRR'] = mrr
    for hit in [1, 3, 10]:
        avg_count = torch.mean((ranks <= hit).float())
        metrics['Raw Hit@{}'.format(hit)] = avg_count

    for metric in logs[0].keys():
        metrics[metric] = sum([log[metric] for log in logs]) / len(logs)
    metrics['Time MSE'] = TimeMSE / len(testloader.dataset)
    metrics['Time MAE'] = TimeMAE / len(testloader.dataset)
    return metrics


def train_epoch(args, model, traindataloader, optimizer, scheduler, device, epoch):
    model.train()
    with tqdm(total=len(traindataloader), unit='ex') as bar:
        bar.set_description('Train')
        total_loss = 0
        total_num = 0
        for sub, rel, obj, time, history_graphs, history_times, batch_node_ids in traindataloader:
            if epoch < args.warm_up:
                scheduler.step()
            sub = sub.to(device, non_blocking=True)
            rel = rel.to(device, non_blocking=True)
            obj = obj.to(device, non_blocking=True)
            time = time.to(device, non_blocking=True)
            history_graphs = history_graphs.to(device, non_blocking=True)
            history_times = history_times.to(device, non_blocking=True)
            batch_node_ids = batch_node_ids.to(device, non_blocking=True)

            lp_loss, tp_loss = model.train_forward(sub, rel, obj, time, history_graphs, history_times, batch_node_ids)
            loss = lp_loss + args.alpha * tp_loss
            # loss = lp_loss
            loss.backward()

            total_loss += loss
            total_num += 1

            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_norm)
            optimizer.step()
            optimizer.zero_grad()

            bar.update(1)
            bar.set_postfix(loss='%.4f' % loss, lp_loss='%.4f' % lp_loss, tp_loss='%.4f' % tp_loss)

        logging.info('Epoch {} Train Loss: {}'.format(epoch, total_loss/total_num))


class WarmUpLR(torch.optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, total_iters, last_epoch=-1):
        self.total_iters = total_iters
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        return [base_lr * self.last_epoch / (self.total_iters + 1e-8) for base_lr in self.base_lrs]


def main(args):
    output_path = os.path.join(args.output_root, '{0}_{1}'.format(args.data, args.model_name))
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    log_file = os.path.join(output_path, 'log.txt')
    set_logger(log_file)

    logging.info(args)

    if torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'

    data_path = os.path.join(args.data_root, args.data)
    trainpath = os.path.join(data_path, 'train.txt')
    validpath = os.path.join(data_path, 'valid.txt')
    testpath = os.path.join(data_path, 'test.txt')
    statpath = os.path.join(data_path, 'stat.txt')
    baseDataset = BaseDataset(trainpath, testpath, statpath, validpath)

    dglGraphDataset = DGLGraphDataset(
        baseDataset.train_snapshots + baseDataset.valid_snapshots + baseDataset.test_snapshots,
        baseDataset.num_e, baseDataset.num_r)

    if args.edge_sample == 'one_hop_conf':
        edges_conf = pickle.load(open(os.path.join(data_path, 'conf.pkl'), 'rb'))
        edges_conf = torch.tensor(edges_conf)
        edge_sample = True
    else:
        edges_conf = None
        edge_sample = False
    trainQuadruples = baseDataset.get_reverse_quadruples_array(baseDataset.trainQuadruples, baseDataset.num_r)
    trainQuadDataset = QuadruplesDataset(trainQuadruples, args.history_len, dglGraphDataset, baseDataset,
                                         args.history_mode, args.nhop, args.forecasting_t_win_size, args.time_span,
                                         edges_conf, edge_sample, 'train')
    trainDataLoader = DataLoader(
        trainQuadDataset,
        shuffle=True,
        batch_size=args.batch_size,
        collate_fn=lambda x: trainQuadDataset.collate_fn(x, baseDataset.num_e),
        num_workers=args.num_works,
        pin_memory=True
    )


    validQuadruples = baseDataset.get_reverse_quadruples_array(baseDataset.validQuadruples, baseDataset.num_r)
    validQuadDataset = QuadruplesDataset(validQuadruples, args.history_len, dglGraphDataset, baseDataset,
                                        args.history_mode, args.nhop, args.forecasting_t_win_size, args.time_span,
                                        edges_conf, edge_sample, 'test')
    validDataLoader = DataLoader(
        validQuadDataset,
        shuffle=False,
        batch_size=args.batch_size,
        collate_fn=lambda x: validQuadDataset.collate_fn(x, baseDataset.num_e),
        num_workers=args.num_works,
        pin_memory=True
    )

    testQuadruples = baseDataset.get_reverse_quadruples_array(baseDataset.testQuadruples, baseDataset.num_r)
    testQuadDataset = QuadruplesDataset(testQuadruples, args.history_len, dglGraphDataset, baseDataset,
                                        args.history_mode, args.nhop, args.forecasting_t_win_size, args.time_span,
                                        edges_conf, edge_sample, 'test')
    testDataLoader = DataLoader(
        testQuadDataset,
        shuffle=False,
        batch_size=args.batch_size,
        collate_fn=lambda x: testQuadDataset.collate_fn(x, baseDataset.num_e),
        num_workers=args.num_works,
        pin_memory=True
    )

    Config = namedtuple('config', ['n_ent', 'd_model', 'n_rel', 'dropout','seqTransformerLayerNum', 'seqTransformerHeadNum'])
    config = Config(n_ent=baseDataset.num_e + 1,
                    n_rel=baseDataset.num_r * 2,
                    d_model=args.d_model,
                    dropout=args.dropout,
                    seqTransformerLayerNum=args.seqTransformerLayerNum,
                    seqTransformerHeadNum=args.seqTransformerHeadNum)
    model = TemporalTransformerHawkesGraphModel(config, args.eps, args.time_span, args.timestep, args.hmax)
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    if args.warm_up > 0:
        warmup_scheduler = WarmUpLR(optimizer, len(trainDataLoader) * args.warm_up)
    else:
        warmup_scheduler = None

    if os.path.isfile(args.load_model_path):
        params = torch.load(args.load_model_path)
        model.load_state_dict(params['model_state_dict'])
        optimizer.load_state_dict(params['optimizer_state_dict'])
        logging.info('Load pretrain model: {}'.format(args.load_model_path))

    if args.do_train:
        logging.info('Start Training......')

        for i in range(args.max_epochs):
            if i % args.valid_epoch == 0 and i != 0:
                model_save_path = os.path.join(output_path, 'model_{}.pth'.format(i))
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                }, model_save_path)

                for delta_t in range(args.forecasting_t_win_size):
                    delta_t = delta_t + 1
                    testDataLoader.dataset.delta_t = delta_t
                    metrics = test(model, validDataLoader, baseDataset.skip_dict, device)

                    for mode in metrics.keys():
                        logging.info('Delta_t {} Valid {} : {}'.format(delta_t, mode, metrics[mode]))

            train_epoch(args, model, trainDataLoader, optimizer, warmup_scheduler, device, i)

    if args.do_test:
        logging.info('Start Testing......')
        for delta_t in range(args.forecasting_t_win_size):
            delta_t = delta_t + 1
            testDataLoader.dataset.delta_t = delta_t
            metrics = test(model, testDataLoader, baseDataset.skip_dict, device)
            for mode in metrics.keys():
                logging.info('Delta_t {} Test {} : {}'.format(delta_t, mode, metrics[mode]))


if __name__ == '__main__':
    args = parse_args()
    main(args)

