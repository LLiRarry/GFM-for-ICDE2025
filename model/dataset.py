import torch
import math
import os
import json
import numpy as np
from texttable import Texttable
from torch.utils.data import random_split, Dataset
from torch_geometric.data import DataLoader, Batch
from torch_geometric.datasets import GEDDataset
from torch_geometric.transforms import OneHotDegree
from torch_geometric.data import Data, InMemoryDataset
from torch_geometric.utils import add_remaining_self_loops, remove_self_loops, dense_to_sparse
from torch_geometric.utils import softmax, degree
from torch_scatter import scatter
from torch_cluster import random_walk
from torch_sparse import spspmm, coalesce


def sort_edge_index(edge_index, edge_attr=None, num_nodes=None):
    r"""Row-wise sorts edge indices :obj:`edge_index`.

    Args:
        edge_index (LongTensor): The edge indices.
        edge_attr (Tensor, optional): Edge weights or multi-dimensional
            edge features. (default: :obj:`None`)
        num_nodes (int, optional): The number of nodes, *i.e.*
            :obj:`max_val + 1` of :attr:`edge_index`. (default: :obj:`None`)

    :rtype: (:class:`LongTensor`, :class:`Tensor`)
    """

    edge_index = edge_index.to(torch.int64)  # Ensure edge_index is of type int64
    idx = edge_index[0] * num_nodes + edge_index[1]
    perm = idx.argsort()

    return edge_index[:, perm], None if edge_attr is None else edge_attr[perm]



class BinaryFuncDataset(InMemoryDataset):
    def __init__(self, root, name, transform=None, pre_transform=None, pre_filter=None):
        self.dir_name = os.path.join(root, name)
        self.number_features = 0
        self.func2graph = dict()
        super(BinaryFuncDataset, self).__init__(root, transform, pre_transform, pre_filter)
        self.func2graph, self.number_features = torch.load(self.processed_paths[0])

    @property
    def processed_file_names(self):
        return ['data.pt']

    def process(self):
        for filename in os.listdir(self.dir_name):
            if filename[-3:] == 'npy' or filename=='processed':
                continue
            print(self.dir_name + '/' + filename,"dir")
            f = open(self.dir_name + '/' + filename, 'r')
            contents = f.readlines()
            f.close()
            for jsline in contents:
                check_dict = dict()
                g = json.loads(jsline)
                funcname = g['fname']  # Type: str
                features = g['features']  # Type: list
                idlist = g['succs']  # Type: list
                n_num = g['n_num']  # Type: int
                # Build graph index
                edge_index = []
                for i in range(n_num):
                    idx = idlist[i]
                    if len(idx) == 0:
                        continue
                    for j in idx:
                        if (i, j) not in check_dict:
                            check_dict[(i, j)] = 1
                            edge_index.append((i, j))
                        if (j, i) not in check_dict:
                            check_dict[(j, i)] = 1
                            edge_index.append((j, i))
                np_edge_index = np.array(edge_index).T
                pt_edge_index = torch.from_numpy(np_edge_index)
                x = np.array(features, dtype=np.float32)
                x = torch.from_numpy(x)
                row, col = pt_edge_index
                cat_row_col = torch.cat((row, col))
                n_nodes = torch.unique(cat_row_col).size(0)
                if n_nodes != x.size(0):
                    continue
                self.number_features = x.size(1)
                pt_edge_index, _ = sort_edge_index(pt_edge_index, num_nodes=x.size(0))
                data = Data(x=x, edge_index=pt_edge_index)
                data.num_nodes = n_num
                if funcname in self.func2graph:
                    self.func2graph[funcname].append(data)
                else:
                    self.func2graph[funcname] = [data]
        torch.save((self.func2graph, self.number_features), self.processed_paths[0])


class GraphClassificationDataset(object):
    def __init__(self,name):
        self.training_funcs = dict()
        self.validation_funcs = dict()
        self.testing_funcs = dict()
        self.number_features = None
        self.id2name = None
        self.func2graph = None
        self.device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.dataset_name=name
        self.batch_size=64
        self.process_dataset()

    def process_dataset(self):
        print('\nPreparing datasets.\n')
        self.dataset = BinaryFuncDataset('hgmn_dataset-master/', self.dataset_name)
        self.number_features = self.dataset.number_features
        self.func2graph = self.dataset.func2graph
        self.id2name = dict()

        cnt = 0
        for k, v in self.func2graph.items():
            self.id2name[cnt] = k
            cnt += 1

        self.train_num = int(len(self.func2graph) * 0.8)
        self.val_num = int(len(self.func2graph) * 0.1)
        self.test_num = int(len(self.func2graph)) - (self.train_num + self.val_num)

        random_idx = np.random.permutation(len(self.func2graph))
        self.train_idx = random_idx[0: self.train_num]
        self.val_idx = random_idx[self.train_num: self.train_num + self.val_num]
        self.test_idx = random_idx[self.train_num + self.val_num:]

        self.training_funcs = self.split_dataset(self.training_funcs, self.train_idx)
        self.validation_funcs = self.split_dataset(self.validation_funcs, self.val_idx)
        self.testing_funcs = self.split_dataset(self.testing_funcs, self.test_idx)

    def split_dataset(self, funcdict, idx):
        for i in idx:
            funcname = self.id2name[i]
            funcdict[funcname] = self.func2graph[funcname]
        return funcdict

    def collate(self, data_list):
        batchS = Batch.from_data_list([data[0] for data in data_list] + [data[0] for data in data_list])
        batchT = Batch.from_data_list([data[1] for data in data_list] + [data[2] for data in data_list])
        batchL = ([1 for data in data_list] + [0 for data in data_list])
        return batchS, batchT, batchL

    def create_batches(self, funcs, collate, shuffle_batch=True):
        data = FuncDataset(funcs)
        loader = torch.utils.data.DataLoader(data, batch_size=self.batch_size, shuffle=shuffle_batch,
                                             collate_fn=collate)

        return loader

    def transform(self, data):
        new_data = dict()

        new_data['g1'] = data[0].to(self.device)
        new_data['g2'] = data[1].to(self.device)
        new_data['target'] = torch.from_numpy(np.array(data[2], dtype=np.float32)).to(self.device)
        return new_data


class FuncDataset(Dataset):
    def __init__(self, funcdict):
        super(FuncDataset, self).__init__()
        self.funcdict = funcdict
        self.id2key = dict()
        cnt = 0
        for k, v in self.funcdict.items():
            self.id2key[cnt] = k
            cnt += 1

    def __len__(self):
        return len(self.funcdict)

    def __getitem__(self, idx):
        graphset = self.funcdict[self.id2key[idx]]
        pos_idx = np.random.choice(range(len(graphset)), size=2, replace=True)
        origin_graph = graphset[pos_idx[0]]
        pos_graph = graphset[pos_idx[1]]
        all_keys = list(self.funcdict.keys())
        neg_key = np.random.choice(range(len(all_keys)))
        while all_keys[neg_key] == self.id2key[idx]:
            neg_key = np.random.choice(range(len(all_keys)))
        neg_data = self.funcdict[all_keys[neg_key]]
        neg_idx = np.random.choice(range(len(neg_data)))
        neg_graph = neg_data[neg_idx]

        return origin_graph, pos_graph, neg_graph, 1, 0

