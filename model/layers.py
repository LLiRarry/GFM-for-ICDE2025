from torch_geometric.nn.dense.linear import Linear
import inspect
from typing import Any, Dict, Optional
from torch.nn import Dropout, Linear, Sequential
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.inits import reset
from torch_geometric.nn.resolver import (
    activation_resolver,
    normalization_resolver,
)
import torch.nn as nn
from torch_geometric.nn.attention import PerformerAttention
import torch
import torch.nn.functional as F
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
from typing import Optional, Tuple
import torch
from torch import Tensor
from torch_geometric.utils import scatter


class MLPModule(torch.nn.Module):
    def __init__(self, nhid):
        super(MLPModule, self).__init__()
        self.nhid = nhid
        self.dropout = 0.0

        self.lin0 = torch.nn.Linear(self.nhid, self.nhid * 2)
        nn.init.xavier_uniform_(self.lin0.weight.data)
        nn.init.zeros_(self.lin0.bias.data)

        self.lin1 = torch.nn.Linear(self.nhid * 2, self.nhid)
        nn.init.xavier_uniform_(self.lin1.weight.data)
        nn.init.zeros_(self.lin1.bias.data)

        self.lin2 = torch.nn.Linear(self.nhid, self.nhid // 2)
        nn.init.xavier_uniform_(self.lin2.weight.data)
        nn.init.zeros_(self.lin2.bias.data)

        self.lin3 = torch.nn.Linear(self.nhid // 2, 1)
        nn.init.xavier_uniform_(self.lin3.weight.data)
        nn.init.zeros_(self.lin3.bias.data)

    def forward(self, scores):
        scores = self.lin0(scores)
        scores = F.dropout(scores, p=self.dropout, training=self.training)
        scores = F.leaky_relu(self.lin1(scores))
        scores = F.dropout(scores, p=self.dropout, training=self.training)
        scores = F.leaky_relu(self.lin2(scores))
        scores = F.dropout(scores, p=self.dropout, training=self.training)
        scores = torch.sigmoid(self.lin3(scores)).view(-1)

        return scores


class ReadoutModule2(torch.nn.Module):
    def __init__(self, nhid):
        super(ReadoutModule2, self).__init__()
        self.nhid = nhid

        self.weight = torch.nn.Parameter(torch.Tensor(self.nhid, self.nhid))
        nn.init.xavier_uniform_(self.weight.data)

    def forward(self, x, batch):
        mean_pool = global_mean_pool(x, batch)
        transformed_global = torch.tanh(torch.mm(mean_pool, self.weight))
        coefs = torch.sigmoid((x * transformed_global[batch]).sum(dim=1))
        weighted = coefs.unsqueeze(-1) * x

        return global_add_pool(weighted, batch)


class Combineall(nn.Module):
    def __init__(self, dim_size):
        super(Combineall, self).__init__()
        self.dim_size = dim_size
        self.ReadoutModule = ReadoutModule2(dim_size)
        self.ha = VectorSimilarity(dim_size // 2)
        self.Binode = EnhancedBilinearInteraction(dim_size, dim_size, dim_size)
        self.BiDAFWithAttention = BiDAFWithAttention(dim_size, dim_size // 2)

    def forward(self, x1, batch1, x2, batch2):
        embed1_mean = self.ReadoutModule(x1, batch1)
        embed2_mean = self.ReadoutModule(x2, batch2)
        score_mean = self.ha.process_vectors(embed1_mean, embed2_mean)
        h1, _, h2, _, _ = merge_batches_and_process(x1, batch1, x2, batch2)
        scoreh = self.Binode(h1, h2)
        e12h2 = self.BiDAFWithAttention(embed1_mean.unsqueeze(1), h2)
        e22h1 = self.BiDAFWithAttention(embed2_mean.unsqueeze(1), h1)

        return torch.concat((score_mean, scoreh, e12h2, e22h1), dim=1)


#
class AttentionReducer(nn.Module):
    def __init__(self, sequence_dim):
        super(AttentionReducer, self).__init__()
        self.attention_weights = nn.Linear(sequence_dim, 1)

    def forward(self, x):
        # x shape: (batch, seq_len, sequence_dim)
        weights = self.attention_weights(x).squeeze(2)  # (batch, seq_len)
        weights = F.softmax(weights, dim=1)  # Softmax over seq_len dimension to create attention weights
        weighted = torch.bmm(weights.unsqueeze(1), x).squeeze(1)  # (batch, sequence_dim)
        return weighted


class BiDAFWithAttention(nn.Module):
    def __init__(self, embedding_dim, output_dim):
        super(BiDAFWithAttention, self).__init__()
        self.attention_reducer = AttentionReducer(embedding_dim * 2)
        self.output_layer = nn.Linear(embedding_dim * 2, output_dim)

    def forward(self, passage_encodes, question_encodes):
        # Similarity matrix
        sim_matrix = torch.bmm(passage_encodes, question_encodes.transpose(1, 2))

        # Context-to-Question Attention
        c2q_attn = F.softmax(sim_matrix, dim=-1)
        context2question_attn = torch.bmm(c2q_attn, question_encodes)

        # Question-to-Context Attention
        q2c_attn = F.softmax(torch.max(sim_matrix, dim=-1)[0], dim=-1).unsqueeze(1)
        question2context_attn = torch.bmm(q2c_attn, passage_encodes).repeat(1, passage_encodes.size(1), 1)

        # Concatenation of the outputs
        concat_outputs = torch.cat([
            torch.tanh(passage_encodes) * torch.tanh(context2question_attn),
            torch.tanh(passage_encodes) * torch.tanh(question2context_attn)], dim=-1)

        # Attention-based reduction
        reduced_output = self.attention_reducer(concat_outputs)

        # Final output layer
        output = self.output_layer(reduced_output)
        return output


class VectorSimilarity:
    def __init__(self, max_poolings):
        self.max_poolings = max_poolings
        self.kernels = [1, 2, 3, 4, 5, 6, 7, 8]
        self.strides = [1, 2, 3]
        self.convs = nn.ModuleList()
        for kernel in self.kernels:
            for stride in self.strides:
                if len(self.convs) < max_poolings:
                    self.convs.append(
                        nn.Conv1d(in_channels=1, out_channels=1, kernel_size=kernel, stride=stride, bias=False).to(
                            device))
                else:
                    break
            if len(self.convs) >= max_poolings:
                break

    def process_vectors(self, batch_vec1, batch_vec2):
        if batch_vec1.dim() == 2:
            batch_vec1 = batch_vec1.unsqueeze(1)
        if batch_vec2.dim() == 2:
            batch_vec2 = batch_vec2.unsqueeze(1)

        results = []
        for conv in self.convs:
            conv1 = conv(batch_vec1)
            conv2 = conv(batch_vec2)

            if conv1.nelement() != 0 and conv2.nelement() != 0:
                hamming_sim = torch.mean(torch.tanh(conv1) * torch.tanh(conv2), dim=-1)
                cosine_sim = F.cosine_similarity(conv1, conv2, dim=-1)

                results.append(torch.cat((hamming_sim, cosine_sim), dim=-1))

        return torch.cat(results, dim=-1)


#

class EnhancedBilinearInteraction(nn.Module):
    def __init__(self, in_channels, out_channels, hidden_channels):
        super(EnhancedBilinearInteraction, self).__init__()

        self.conv1 = nn.Conv1d(in_channels, hidden_channels, kernel_size=1, padding=0, stride=1)

        self.conv2 = nn.Conv1d(hidden_channels, hidden_channels, kernel_size=1, padding=0, stride=1)

        self.bilinear = BilinearInteraction(hidden_channels, out_channels)

    def forward(self, x, y):
        x = x.transpose(1, 2)
        y = y.transpose(1, 2)

        x = self.conv1(x)
        x = self.conv2(x)

        y = self.conv1(y)
        y = self.conv2(y)

        output = self.bilinear(x, y)

        return output.squeeze(-1)


class BilinearInteraction(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(BilinearInteraction, self).__init__()
        self.out_channels = out_channels

    def forward(self, x, y):
        B, M, L = x.shape
        kernels = torch.tanh(y.view(B * M, 1, L))
        x = torch.tanh(x.view(1, B * M, L))
        result = F.conv1d(x, kernels, groups=B * M)
        result = result.view(B, M, -1)
        return result


def to_dense_batch2(x: Tensor, batch: Optional[Tensor] = None,
                    fill_value: float = 0., max_num_nodes: Optional[int] = None,
                    batch_size: Optional[int] = None) -> Tuple[Tensor, Tensor, int]:
    if batch is None and max_num_nodes is None:
        mask = torch.ones(1, x.size(0), dtype=torch.bool, device=x.device)
        return x.unsqueeze(0), mask

    if batch is None:
        batch = x.new_zeros(x.size(0), dtype=torch.long)

    if batch_size is None:
        batch_size = int(batch.max()) + 1

    num_nodes = scatter(batch.new_ones(x.size(0)), batch, dim=0,
                        dim_size=batch_size, reduce='sum')
    cum_nodes = torch.cat([batch.new_zeros(1), num_nodes.cumsum(dim=0)])

    filter_nodes = False
    if max_num_nodes is None:
        max_num_nodes = int(num_nodes.max())
    elif num_nodes.max() > max_num_nodes:
        filter_nodes = True

    tmp = torch.arange(batch.size(0), device=x.device) - cum_nodes[batch]
    idx = tmp + (batch * max_num_nodes)
    if filter_nodes:
        mask = tmp < max_num_nodes
        x, idx = x[mask], idx[mask]

    size = [batch_size * max_num_nodes] + list(x.size())[1:]
    out = x.new_full(size, fill_value)
    out[idx] = x
    out = out.view([batch_size, max_num_nodes] + list(x.size())[1:])

    mask = torch.zeros(batch_size * max_num_nodes, dtype=torch.bool,
                       device=x.device)
    mask[idx] = 1
    mask = mask.view(batch_size, max_num_nodes)

    return out, mask, max_num_nodes


from torch_geometric.nn import global_mean_pool, global_max_pool, global_add_pool


def merge_batches_and_process(x1, batch1, x2, batch2):
    """Merge two batches and process them into a dense batch with averaging."""
    # Merge x and batch
    x = torch.cat([x1, x2], dim=0)
    batch = torch.cat([batch1, batch2 + batch1.max() + 1])
    batch_size = int(batch1.max()) + 1

    assert int(batch1.max()) == int(batch2.max())

    dense_x, mask, max_num_nodes = to_dense_batch2(x, batch)

    return dense_x[0:batch_size], mask[0:batch_size], dense_x[batch_size:], mask[batch_size:], max_num_nodes


class GPSConv(torch.nn.Module):

    def __init__(
            self,
            channels: int,
            conv: Optional[MessagePassing],
            heads: int = 1,
            dropout: float = 0.0,
            act: str = 'relu',
            act_kwargs: Optional[Dict[str, Any]] = None,
            norm: Optional[str] = 'batch_norm',
            norm_kwargs: Optional[Dict[str, Any]] = None,
            attn_type: str = 'multi-head',
            attn_kwargs: Optional[Dict[str, Any]] = None,
    ):
        super().__init__()

        self.channels = channels
        self.conv = conv
        self.heads = heads
        self.dropout = dropout
        self.attn_type = attn_type

        attn_kwargs = attn_kwargs or {}
        if attn_type == 'multi-head':
            self.attn = torch.nn.MultiheadAttention(
                channels,
                heads,
                batch_first=True,
                **attn_kwargs,
            )
        elif attn_type == 'performer':  # linear attention
            self.attn = PerformerAttention(
                channels=channels,
                heads=heads,
                **attn_kwargs,
            )
        else:
            raise ValueError(f'{attn_type} is not supported')

        self.mlp = Sequential(
            Linear(channels, channels * 2),
            activation_resolver(act, **(act_kwargs or {})),
            Dropout(dropout),
            Linear(channels * 2, channels),
            Dropout(dropout),
        )

        norm_kwargs = norm_kwargs or {}
        self.norm1 = normalization_resolver(norm, channels, **norm_kwargs)
        self.norm2 = normalization_resolver(norm, channels, **norm_kwargs)
        self.norm3 = normalization_resolver(norm, channels, **norm_kwargs)

        self.norm_with_batch = False
        if self.norm1 is not None:
            signature = inspect.signature(self.norm1.forward)
            self.norm_with_batch = 'batch' in signature.parameters

    def reset_parameters(self):
        r"""Resets all learnable parameters of the module."""
        if self.conv is not None:
            self.conv.reset_parameters()
        self.attn._reset_parameters()
        reset(self.mlp)
        if self.norm1 is not None:
            self.norm1.reset_parameters()
        if self.norm2 is not None:
            self.norm2.reset_parameters()
        if self.norm3 is not None:
            self.norm3.reset_parameters()

    def forward(self, x1, x2, edge_index1, edge_index2, batch1, batch2, **kwargs):

        hs1, hs2 = [], []
        if self.conv is not None:  # Local MPNN.
            h1 = self.conv(x1, edge_index1, **kwargs)
            h1 = h1 + x1
            if self.norm1 is not None:
                if self.norm_with_batch:
                    h1 = self.norm1(h1, batch=batch1)
                else:
                    h1 = self.norm1(h1)

            hs1.append(h1)
            h2 = self.conv(x2, edge_index2, **kwargs)
            h2 = h2 + x2
            if self.norm1 is not None:
                if self.norm_with_batch:
                    h2 = self.norm1(h2, batch=batch2)

                else:
                    h2 = self.norm1(h2)

            hs2.append(h2)
        else:
            h1, h2 = x1, x2


        h1, mask1, h2, mask2, max_len = merge_batches_and_process(h1, batch1, h2, batch2)
        # print(h1.shape, h2.shape, 'h1-h2---1')
        h_concat = torch.cat([h1, h2], dim=1)  # h  [B,max_len*2,dim]
        mask_concat = torch.cat([mask1, mask2], dim=1)
        if isinstance(self.attn, torch.nn.MultiheadAttention):
            h_concat, _ = self.attn(h_concat, h_concat, h_concat, key_padding_mask=~mask_concat,
                                    need_weights=False)
        elif isinstance(self.attn, PerformerAttention):
            h_concat = self.attn(h_concat, mask=mask_concat)
        # print(h_concat.shape,'h_concat_0')
        h1, h2 = h_concat[:, :max_len], h_concat[:, max_len:]
        # print(h1.shape,h2.shape,'h1-h2')
        h1 = h1[mask1]
        h2 = h2[mask2]
        #  the third part
        h1 = h1 + x1  # Residual connection.
        if self.norm2 is not None:
            if self.norm_with_batch:
                h1 = self.norm2(h1, batch=batch1)
            else:
                h1 = self.norm2(h1)
        hs1.append(h1)
        out1 = sum(hs1)  # Combine local and global outputs.
        out1 = out1 + self.mlp(out1)
        if self.norm3 is not None:
            if self.norm_with_batch:
                out1 = self.norm3(out1, batch=batch1)
            else:
                out1 = self.norm3(out1)
        #
        h2 = h2 + x2  # Residual connection.
        if self.norm2 is not None:
            if self.norm_with_batch:
                h2 = self.norm2(h2, batch=batch2)
            else:
                h2 = self.norm2(h2)
        hs2.append(h2)
        out2 = sum(hs2)  # Combine local and global outputs.
        out2 = out2 + self.mlp(out2)
        if self.norm3 is not None:
            if self.norm_with_batch:
                out2 = self.norm3(out2, batch=batch2)
            else:
                out2 = self.norm3(out2)

        return out1, out2
