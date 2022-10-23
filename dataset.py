import os.path
from collections import defaultdict
from torch.utils.data import Dataset, DataLoader
import numpy as np
import dgl
import torch

class BaseDataset(object):
    def __init__(self, trainpath, testpath, statpath, validpath):
        """base Dataset. Read data files and preprocess.
        Args:
            trainpath: File path of train Data;
            testpath: File path of test data;
            statpath: File path of entities num and relatioins num;
            validpath: File path of valid data
        """
        self.trainQuadruples = self.load_quadruples(trainpath)
        self.testQuadruples = self.load_quadruples(testpath)
        self.validQuadruples = self.load_quadruples(validpath)
        self.allQuadruples = self.trainQuadruples + self.validQuadruples + self.testQuadruples
        self.num_e, self.num_r = self.get_total_number(statpath)  # number of entities, number of relations
        self.skip_dict = self.get_skipdict(self.allQuadruples)

        self.train_snapshots = self.split_by_time(self.trainQuadruples)
        self.valid_snapshots = self.split_by_time(self.validQuadruples)
        self.test_snapshots = self.split_by_time(self.testQuadruples)

        self.time_inverted_index_dict = self.get_time_inverted_index_dict(self.allQuadruples)
        self.reltime2ent_dict = self.get_relation_time_dict(self.allQuadruples)
        self.alltimes = self.get_all_timestamps()

    def get_all_timestamps(self):
        """Get all the timestamps in the dataset.
        return:
            timestamps: a set of timestamps.
        """
        timestamps = set()
        for ex in self.allQuadruples:
            timestamps.add(ex[3])
        return sorted(list(timestamps))

    def get_skipdict(self, quadruples):
        """Used for time-dependent filtered metrics.
        return: a dict [key -> (entity, relation, timestamp),  value -> a set of ground truth entities]
        """
        filters = defaultdict(set)
        for src, rel, dst, time in quadruples:
            filters[(src, rel, time)].add(dst)
            filters[(dst, rel+self.num_r, time)].add(src)
        return filters

    @staticmethod
    def load_quadruples(inpath):
        """train.txt/valid.txt/test.txt reader
        inpath: File path. train.txt, valid.txt or test.txt of a dataset;
        return:
            quadrupleList: A list
            containing all quadruples([subject/headEntity, relation, object/tailEntity, timestamp]) in the file.
        """
        with open(inpath, 'r') as f:
            quadrupleList = []
            for line in f:
                line_split = line.split()
                head = int(line_split[0])
                rel = int(line_split[1])
                tail = int(line_split[2])
                time = int(line_split[3])
                quadrupleList.append([head, rel, tail, time])
        return quadrupleList

    @staticmethod
    def get_total_number(statpath):
        """stat.txt reader
        return:
            (number of entities -> int, number of relations -> int)
        """
        with open(statpath, 'r') as fr:
            for line in fr:
                line_split = line.split()
                return int(line_split[0]), int(line_split[1])

    @staticmethod
    def split_by_time(data):
        snapshot_list = []
        snapshot = []
        latest_t = 0
        for i in range(len(data)):
            t = data[i][3]
            train = data[i]
            if latest_t != t:
                if len(snapshot):
                    snapshot_list.append((np.array(snapshot).copy(), latest_t))
                snapshot = []
                latest_t = t
            snapshot.append(train[:3])
        if len(snapshot) > 0:
            snapshot_list.append((np.array(snapshot).copy(), latest_t))
        return snapshot_list

    @staticmethod
    def get_reverse_quadruples_array(quadruples, num_r):
        quads = np.array(quadruples)
        quads_r = np.zeros_like(quads)
        quads_r[:, 1] = num_r + quads[:, 1]
        quads_r[:, 0] = quads[:, 2]
        quads_r[:, 2] = quads[:, 0]
        quads_r[:, 3] = quads[:, 3]
        return np.concatenate((quads, quads_r))

    def get_time_inverted_index_dict(self, quadruples):
        index_dict = defaultdict(set)
        for quad in quadruples:
            index_dict[quad[0]].add(quad[3])
            index_dict[quad[2]].add(quad[3])
            index_dict[(quad[0], quad[1])].add(quad[3])
            index_dict[(quad[2], quad[1] + self.num_r)].add(quad[3])
        for k, v in index_dict.items():
            index_dict[k] = sorted(list(v))
        return index_dict

    def get_relation_time_dict(self, quadruples):
        dict = defaultdict(list)
        for quad in quadruples:
            dict[(quad[1], quad[3])].append([quad[0], quad[2], quad[3]])
            dict[(quad[1] + self.num_r, quad[3])].append([quad[2], quad[0], quad[3]])
        return dict

class DGLGraphDataset(object):
    def __init__(self, snapshots, n_ent, n_rel):
        self.n_ent = n_ent
        self.n_rel = n_rel
        self.snapshots_num = len(snapshots)
        self.snapshots = snapshots
        self.dgl_graph_dict, self.dgl_graphs = self.get_dglGraph_dict(snapshots)

    def get_dglGraph_dict(self, snapshots):
        dgl_graph_dict = {}
        dgl_graph = []
        for (g, time) in snapshots:
            graph = self.build_sub_graph(self.n_ent, self.n_rel, g, time)
            dgl_graph_dict[time] = graph
            dgl_graph.append(graph)
        PAD_graph = self.build_sub_graph(self.n_ent, self.n_rel, np.array([]), 0)
        dgl_graph_dict[-1] = PAD_graph
        dgl_graph.insert(0, PAD_graph)
        return dgl_graph_dict, dgl_graph

    def build_sub_graph(self, num_nodes, num_rels, triples, time):
        if triples.size != 0:
            src, rel, dst = triples.transpose()
            src, dst = np.concatenate((src, dst)), np.concatenate((dst, src))
            rel = np.concatenate((rel, rel + num_rels))
        else:
            src, rel, dst = np.array([]), np.array([]), np.array([])
        g = dgl.DGLGraph()
        g.add_nodes(num_nodes)
        g.add_edges(src, dst)

        node_id = torch.arange(0, num_nodes, dtype=torch.long).view(-1, 1)
        g.ndata.update({'id': node_id})
        g.edata['type'] = torch.LongTensor(rel)
        g.edata['timestamp'] = torch.LongTensor(torch.ones_like(g.edata['type']) * time)
        return g

    def get_nhop_subgraph(self, time, root_node, n=2):
        g = self.dgl_graph_dict[time]
        # g = dgl.in_subgraph(g, [root_node])
        total_nodes = set()
        total_nodes.add(root_node)
        for i in range(n):
            step_nodes = total_nodes.copy()
            for node in step_nodes:
                neighbor_n, _ = g.in_edges(node)
                neighbor_n = set(neighbor_n.tolist())
                total_nodes |= neighbor_n
        sub_g = g.subgraph(list(total_nodes), store_ids=False)
        # sub_g.ndata['norm'] = self.comp_deg_norm(sub_g).view(-1, 1)
        # sub_g.apply_edges(lambda edges: {'norm': edges.dst['norm'] * edges.src['norm']})
        return sub_g

    def edge_samples(self, root_node, sub_g, conf):
        if sub_g.num_edges() <= 1:
            return sub_g
        edge_type = sub_g.edata['type']
        edge_conf = conf[edge_type]
        chosen = edge_conf > 0.1
        sub_g = dgl.edge_subgraph(sub_g, chosen, store_ids=False)
        if root_node in sub_g.ndata['id'].squeeze(1).tolist():
            return sub_g
        else:
            g = dgl.DGLGraph()
            g.add_nodes(1, {'id': torch.tensor([root_node]).view(-1, 1)})
            g.edata['type'] = torch.LongTensor(np.array([]))
            return g

    def comp_deg_norm(self, g):
        in_deg = g.in_degrees(range(g.number_of_nodes())).float()
        in_deg[torch.nonzero(in_deg == 0).view(-1)] = 1
        norm = 1.0 / in_deg
        return norm


class QuadruplesDataset(Dataset):
    def __init__(self, quadruples, history_len, dglGraphs, baseDataset, history_mode='recent', nhop=2,
                 forecasting_t_windows_size=1, time_span=24, edges_conf=None, edge_sample=False, dataset_type='train'):
        self.quadruples = quadruples
        self.history_len = history_len
        self.dglGraphs = dglGraphs
        self.timeInvDict = baseDataset.time_inverted_index_dict
        self.nhop = nhop
        self.history_mode = history_mode
        self.PAD_TIME = -1
        self.num_r = baseDataset.num_r
        self.forecasting_t_windows_size = forecasting_t_windows_size
        self.time_span = time_span
        self.edges_conf = edges_conf
        self.edge_sample = edge_sample
        self.dataset_type = dataset_type
        self.delta_t = 1

    def __len__(self):
        if self.dataset_type == 'train':
            return len(self.quadruples) * self.forecasting_t_windows_size
        else:
            return len(self.quadruples)

    def __getitem__(self, idx):
        if self.dataset_type == 'train':
            quad_idx = idx // self.forecasting_t_windows_size
            delta_t = idx % self.forecasting_t_windows_size + 1
            quad = self.quadruples[quad_idx]
            head_entity, relation, tail_entity, timestamp = quad[0], quad[1], quad[2], quad[3]
            history_graphs, history_times, head_entity_ids, graphs_node_num = \
                self.get_history_graphs(head_entity, relation, timestamp, self.history_mode, delta_t)
            return head_entity, relation, tail_entity, timestamp, \
                  history_graphs, history_times, head_entity_ids, graphs_node_num
        else:
            quad = self.quadruples[idx]
            head_entity, relation, tail_entity, timestamp = quad[0], quad[1], quad[2], quad[3]
            history_graphs, history_times, head_entity_ids, graphs_node_num = \
                self.get_history_graphs(head_entity, relation, timestamp, self.history_mode, self.delta_t)
            return head_entity, relation, tail_entity, timestamp, \
                   history_graphs, history_times, head_entity_ids, graphs_node_num

    def get_history_graphs(self, head_entity, relation, timestamp, sampled_method='recent', delta_t=1):
        if sampled_method == 'history_copy':
            times = self.timeInvDict[(head_entity, relation)]
            history_times = times[:times.index(timestamp)]
            history_times = history_times[max(-self.history_len, -len(history_times)):]
        elif sampled_method == 'both':
            times1 = self.timeInvDict[(head_entity, relation)]
            times2 = self.timeInvDict[head_entity]
            history_times1 = times1[:times1.index(timestamp)]
            history_times1 = history_times1[max(-(self.history_len // 2), -len(history_times1)):]
            history_times2 = times2[:times2.index(timestamp)]
            history_times2 = history_times2[max(-(self.history_len // 2), -len(history_times2)):]
            history_times = sorted(list(set(history_times1 + history_times2)))
        elif sampled_method == 'delta_t_windows':
            times1 = self.timeInvDict[(head_entity, relation)]
            times2 = self.timeInvDict[head_entity]
            history_times1 = times1[:times1.index(timestamp)]
            history_times1 = list(filter(lambda x: timestamp - x > delta_t * self.time_span, history_times1))
            history_times1 = history_times1[max(-(self.history_len // 2), -len(history_times1)):]
            history_times2 = times2[:times2.index(timestamp)]
            history_times2 = list(filter(lambda x: timestamp - x > delta_t * self.time_span, history_times2))
            history_times2 = history_times2[max(-(self.history_len // 2), -len(history_times2)):]
            last_time = [max([timestamp - self.time_span * delta_t, -1])]
            history_times = sorted(list(set(history_times1 + history_times2 + last_time)))
        else:
            times = self.timeInvDict[head_entity]
            history_times = times[:times.index(timestamp)]
            history_times = history_times[max(-self.history_len, -len(history_times)):]

        history_graphs = []
        head_entity_ids = []
        graphs_node_num = []
        for i, t in enumerate(history_times):
            sub_graph = self.dglGraphs.get_nhop_subgraph(t, head_entity, self.nhop)

            if self.edge_sample:
                sub_graph = self.dglGraphs.edge_samples(head_entity, sub_graph, self.edges_conf[relation])

            sub_graph.edata['query_rel'] = torch.ones_like(sub_graph.edata['type']) * relation
            sub_graph.edata['query_ent'] = torch.ones_like(sub_graph.edata['type']) * head_entity
            # sub_graph.edata['query_time'] = torch.ones_like(sub_graph.edata['type']) * timestamp
            history_graphs.append(sub_graph)
            head_entity_ids.append(sub_graph.ndata['id'].squeeze(1).tolist().index(head_entity))
            graphs_node_num.append(sub_graph.num_nodes())

        return history_graphs, history_times, head_entity_ids, graphs_node_num

    @staticmethod
    def collate_fn(batch, pad_entity):
        batch_data = list(zip(*batch))
        head_entites = batch_data[0]
        relations = batch_data[1]
        tail_entities = batch_data[2]
        timestamps = batch_data[3]

        history_graphs = batch_data[4]  # list
        history_times = batch_data[5]
        head_entity_ids = batch_data[6]
        graphs_node_num = batch_data[7]

        max_history_len = max([len(t) for t in history_times] + [1])
        max_nodes_num = max(sum(graphs_node_num, [1]))

        pad_history_graphs = []
        pad_history_times = []
        pad_history_eids = []
        for i in range(len(history_graphs)):
            hgs = history_graphs[i]
            hts = history_times[i]
            heids = head_entity_ids[i]

            if len(hgs) < max_history_len:
                PAD_G = []
                for j in range(max_history_len - len(hgs)):
                    g = dgl.DGLGraph()
                    g.add_nodes(1, {'id': torch.tensor(head_entites[i]).long().view(-1, 1)})
                    g.edata['type'] = torch.LongTensor(np.array([]))
                    g.edata['query_rel'] = torch.ones_like(g.edata['type'])
                    g.edata['query_ent'] = torch.ones_like(g.edata['type'])
                    # g.edata['query_time'] = torch.ones_like(g.edata['type'])

                    # in_deg = g.in_degrees(range(g.number_of_nodes())).float()
                    # in_deg[torch.nonzero(in_deg == 0).view(-1)] = 1
                    # norm = 1.0 / in_deg
                    # g.ndata['norm'] = norm.view(-1, 1)

                    PAD_G.append(g)

                PAD_HT = [-1 for j in range(max_history_len - len(hgs))]
                PAD_EID = [0 for j in range(max_history_len - len(hgs))]

                hgs.extend(PAD_G)
                hts.extend(PAD_HT)
                heids.extend(PAD_EID)

            for g in hgs:
                node_num = g.num_nodes()
                if node_num < max_nodes_num:
                    g.add_nodes(max_nodes_num - node_num,
                                     {'id': torch.ones(max_nodes_num - node_num, 1).long() * pad_entity})


            pad_history_graphs.extend(hgs)
            pad_history_times.append(hts)
            pad_history_eids.extend(heids)

        pad_history_graphs = dgl.batch(pad_history_graphs)
        batch_node_ids = torch.tensor(pad_history_eids)
        batchgraph_nodes_num = pad_history_graphs.batch_num_nodes()
        graph_num = batchgraph_nodes_num.size(0)
        offset_node_ids = batchgraph_nodes_num.unsqueeze(0).repeat(graph_num, 1)
        offset_mask = torch.tril(torch.ones(graph_num, graph_num), diagonal=-1).long()
        offset_node_ids = offset_node_ids * offset_mask
        offset_node_ids = torch.sum(offset_node_ids, dim=1)
        batch_node_ids += offset_node_ids

        head_entites = torch.tensor(head_entites)  # [bs]
        relations = torch.tensor(relations)  # [bs]
        tail_entities = torch.tensor(tail_entities)  # [bs]
        timestamps = torch.tensor(timestamps)  # [bs]
        pad_history_times = torch.tensor(pad_history_times)  # [bs, history_len]

        return head_entites, relations, tail_entities, timestamps, pad_history_graphs, pad_history_times, batch_node_ids
