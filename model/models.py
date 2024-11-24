from torch_geometric.nn import GCNConv, GINConv, GATConv, SAGEConv
from torch_geometric.nn import GCNConv, GINConv, dense_diff_pool

from layers import *


class GMS(torch.nn.Module):

    def __init__(self, number_class):
        """
        :param args: Arguments object.
        :param number_of_labels: Number of node labels.
        """
        super(GMS, self).__init__()
        self.filters_1 = 48
        self.filters_2 = 48
        self.filters_3 = 48
        self.class_num = number_class
        self.gnn_operator = 'sage'
        self.setup_layers()

    def setup_layers(self):
        if self.gnn_operator == 'gcn':
            self.convolution_1 = GCNConv(self.class_num, self.filters_1)
            self.convolution_2 = GCNConv(self.filters_1, self.filters_2)
            self.convolution_3 = GCNConv(self.filters_2, self.filters_3)
        elif self.gnn_operator == 'gin':
            nn1 = torch.nn.Sequential(
                torch.nn.Linear(self.class_num, self.filters_1),
                torch.nn.ReLU(),
                torch.nn.Linear(self.filters_1, self.filters_1),
                torch.nn.BatchNorm1d(self.filters_1))

            nn2 = torch.nn.Sequential(
                torch.nn.Linear(self.filters_1, self.filters_2),
                torch.nn.ReLU(),
                torch.nn.Linear(self.filters_2, self.filters_2),
                torch.nn.BatchNorm1d(self.filters_2))

            nn3 = torch.nn.Sequential(
                torch.nn.Linear(self.filters_2, self.filters_3),
                torch.nn.ReLU(),
                torch.nn.Linear(self.filters_3, self.filters_3),
                torch.nn.BatchNorm1d(self.filters_3))

            self.convolution_1 = GINConv(nn1, train_eps=True)
            self.convolution_2 = GINConv(nn2, train_eps=True)
            self.convolution_3 = GINConv(nn3, train_eps=True)
        elif self.gnn_operator == 'gat':
            self.convolution_1 = GATConv(self.class_num, self.filters_1)
            self.convolution_2 = GATConv(self.filters_1, self.filters_2)
            self.convolution_3 = GATConv(self.filters_2, self.filters_3)
        elif self.gnn_operator == 'sage':
            self.convolution_1 = SAGEConv(self.class_num, self.filters_1)
            self.convolution_2 = SAGEConv(self.filters_1, self.filters_2)
            self.convolution_3 = SAGEConv(self.filters_2, self.filters_3)

        else:
            raise NotImplementedError('Unknown GNN-Operator.')
        nn1 = Sequential(Linear(self.filters_3, self.filters_3 // 2), torch.nn.ReLU(),
                         Linear(self.filters_3 // 2, self.filters_3))
        self.gps = GPSConv(self.filters_3, GINConv(nn1), heads=4)

        self.agg = Combineall(self.filters_3)

        self.mlp = MLPModule(self.filters_3 * 3)

    def forward(self, data):

        edge_index_1 = data["g1"].edge_index.to(torch.int64)
        edge_index_2 = data["g2"].edge_index.to(torch.int64)
        features_1 = data["g1"].x
        features_2 = data["g2"].x

        batch_1 = (
            data["g1"].batch
            if hasattr(data["g1"], "batch")
            else torch.tensor((), dtype=torch.long).new_zeros(data["g1"].num_nodes)
        )
        batch_2 = (
            data["g2"].batch
            if hasattr(data["g2"], "batch")
            else torch.tensor((), dtype=torch.long).new_zeros(data["g2"].num_nodes)
        )

        # layer1

        features_1 = self.convolution_1(features_1, edge_index_1)
        features_2 = self.convolution_1(features_2, edge_index_2)
        features_1 = self.convolution_2(features_1, edge_index_1) + features_1
        features_2 = self.convolution_2(features_2, edge_index_2) + features_2
        features_1 = self.convolution_3(features_1, edge_index_1) + features_1
        features_2 = self.convolution_3(features_2, edge_index_2) + features_2
        # lay2

        features_1, features_2 = self.gps(x1=features_1, x2=features_2, edge_index1=edge_index_1,
                                          edge_index2=edge_index_2, batch1=batch_1, batch2=batch_2)

        scoref = self.agg(features_1, batch_1, features_2, batch_2)

        finalout = self.mlp(scoref)
        return finalout
