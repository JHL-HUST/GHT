from dataset import BaseDataset
import numpy as np
import os
import dgl
import torch
from collections import defaultdict
from tqdm import tqdm
import pickle
import json

class TotalGraph(object):
    def __init__(self, quadruples, num_r, num_e):
        self.quadruples = quadruples 
        self.num_r = num_r
        self.num_e = num_e
        self.global_graph = self.creat_dgl_graph(quadruples, num_e)
        self.rules = json.load(open('data/ICEWS14/rules.json', 'r'))

    def creat_dgl_graph(self, quadruples, num_e):
        src, rel, dst, time = quadruples.transpose()
        g = dgl.DGLGraph()
        g.add_nodes(num_e, {'id': torch.arange(0, num_e, dtype=torch.long)})
        g.add_edges(src, dst)
        g.edata['type'] = torch.LongTensor(rel)
        g.edata['timestamp'] = torch.LongTensor(time)
        return g

    def get_history_subgraph(self, now_time: int):
        history_graph = dgl.edge_subgraph(self.global_graph, self.global_graph.edata['timestamp'] < now_time,
                                          preserve_nodes=True, store_ids=False)
        return history_graph

    def get_history_graphs(self, quad, beam_size=32):
        src, p, dst, time = quad[0], quad[1], quad[2], quad[3]
        sub_graph = self.get_history_subgraph(time)
        based_rules = self.filter_rules(p)
        i = 0
        for edges in dgl.bfs_edges_generator(sub_graph, src, reverse=True):
            e_times = sub_graph.edata['timestamp']
            e_types = sub_graph.edata['type']

    def filter_rules(self, rel, max_len=3, minconf=0.1, body_supp=10):
        if str(rel) not in self.rules.keys():
            return [[rel, 1]]
        based_rules = self.rules[str(rel)]
        rules = []
        for item in based_rules:
            if len(item['body_rels']) <= max_len and item['conf'] >= minconf and item['body_supp'] >= body_supp:
                rules.append([item['body_rels'], item['conf']])
        return rules