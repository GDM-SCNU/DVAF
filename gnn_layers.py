import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module


def make_quaternion_mul(kernel):
    """" The constructed 'hamilton' W is a modified version of the quaternion representation,
    thus doing tf.matmul(Input,W) is equivalent to W * Inputs. """
    dim = kernel.size(1) // 4
    r, i, j, k = torch.split(kernel, [dim, dim, dim, dim], dim=1)
    r2 = torch.cat([r, -i, -j, -k], dim=0)  # 0, 1, 2, 3
    i2 = torch.cat([i, r, -k, j], dim=0)  # 1, 0, 3, 2
    j2 = torch.cat([j, k, r, -i], dim=0)  # 2, 3, 0, 1
    k2 = torch.cat([k, -j, i, r], dim=0)  # 3, 2, 1, 0
    hamilton = torch.cat([r2, i2, j2, k2], dim=1)
    assert kernel.size(1) == hamilton.size(1)
    return hamilton

''' Quaternion graph neural networks! QGNN layer! https://arxiv.org/abs/2008.05089 '''
class Q4GNNLayer(Module):
    def __init__(self, in_features, out_features, act=torch.tanh):
        super(Q4GNNLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.act = act
        #
        self.weight = Parameter(torch.FloatTensor(self.in_features // 4, self.out_features))

        self.reset_parameters()
        self.bn = torch.nn.BatchNorm1d(out_features)

    def reset_parameters(self):
        stdv = math.sqrt(6.0 / (self.weight.size(0) + self.weight.size(1)))
        self.weight.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        hamilton = make_quaternion_mul(self.weight)
        support = torch.mm(input, hamilton)  # Hamilton product, quaternion multiplication!
        output = torch.spmm(adj, support)
        output = self.bn(output)
        return self.act(output)

''' Simplifying Quaternion graph networks! '''
class SQGNLayer(Module):
    def __init__(self, in_features, out_features, act=torch.tanh, step_k=1):
        super(SQGNLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.act = act
        self.step_k = step_k
        #
        self.weight = Parameter(torch.FloatTensor(self.in_features // 4, self.out_features))

        self.reset_parameters()
        #self.bn = torch.nn.BatchNorm1d(out_features)

    def reset_parameters(self):
        stdv = math.sqrt(6.0 / (self.weight.size(0) + self.weight.size(1)))
        self.weight.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        hamilton = make_quaternion_mul(self.weight)
        new_input = torch.spmm(adj, input)
        if self.step_k > 1:
            for _ in range(self.step_k-1):
                new_input = torch.spmm(adj, new_input)
        output = torch.mm(new_input, hamilton)  # Hamilton product, quaternion multiplication!
        #output = self.bn(output)
        return output

''' Quaternion Graph Isomorphism Networks! QGNN layer! '''
class QGINLayer(Module):
    def __init__(self, in_features, out_features, hid_size, act=torch.tanh):
        super(QGINLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.hid_size = hid_size
        self.act = act
        #
        self.weight1 = Parameter(torch.FloatTensor(self.in_features // 4, self.hid_size))
        self.weight2 = Parameter(torch.FloatTensor(self.hid_size // 4, self.out_features))

        self.reset_parameters()
        self.bn = torch.nn.BatchNorm1d(self.hid_size)
        #self.bn2 = torch.nn.BatchNorm1d(self.out_features)

    def reset_parameters(self):
        stdv1 = math.sqrt(6.0 / (self.weight1.size(0) + self.weight1.size(1)))
        self.weight1.data.uniform_(-stdv1, stdv1)

        stdv2 = math.sqrt(6.0 / (self.weight2.size(0) + self.weight2.size(1)))
        self.weight2.data.uniform_(-stdv2, stdv2)

    def forward(self, input, adj):
        hamilton1 = make_quaternion_mul(self.weight1)
        hamilton2 = make_quaternion_mul(self.weight2)
        new_input = torch.spmm(adj, input)
        output1 = torch.mm(new_input, hamilton1)  # Hamilton product, quaternion multiplication!
        output1 = self.bn(output1)
        output1 = self.act(output1)
        output2 = torch.mm(output1, hamilton2)
        #output2 = self.bn(output2)
        return output2


""" Simple GCN layer, similar to https://arxiv.org/abs/1609.02907 """#简单的gcn层
class GraphConvolution(torch.nn.Module):
    def __init__(self, in_features, out_features, act=torch.relu,  bias=False):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

        self.act = act
        self.bn = torch.nn.BatchNorm1d(out_features)

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        support = torch.mm(input, self.weight)
        output = torch.spmm(adj, support)
        if self.bias is not None:
            output = output + self.bias
        output = self.bn(output)
        return self.act(output)


class Net(torch.nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = SuperGATConv(dataset.num_features, 8, heads=8,
                                  dropout=0.6, attention_type='MX',
                                  edge_sample_ratio=0.8, is_undirected=True)
        self.conv2 = SuperGATConv(8 * 8, dataset.num_classes, heads=8,
                                  concat=False, dropout=0.6,
                                  attention_type='MX', edge_sample_ratio=0.8,
                                  is_undirected=True)

    def forward(self, x, edge_index):
        x = F.dropout(x, p=0.6, training=self.training)
        x = F.elu(self.conv1(x, edge_index))
        att_loss = self.conv1.get_attention_loss()
        x = F.dropout(x, p=0.6, training=self.training)
        x = self.conv2(x, data.edge_index)
        att_loss += self.conv2.get_attention_loss()
        return F.log_softmax(x, dim=-1), att_loss


import copy

import torch
from torch import Tensor

from torch_geometric.nn.conv import MessagePassing


import copy

import torch
from torch import Tensor

from torch_geometric.nn.conv import MessagePassing


class DirGNNConv(torch.nn.Module):
    r"""A generic wrapper for computing graph convolution on directed
    graphs as described in the `"Edge Directionality Improves Learning on
    Heterophilic Graphs" <https://arxiv.org/abs/2305.10498>`_ paper.
    :class:`DirGNNConv` will pass messages both from source nodes to target
    nodes and from target nodes to source nodes.

    Args:
        conv (MessagePassing): The underlying
            :class:`~torch_geometric.nn.conv.MessagePassing` layer to use.
        alpha (float, optional): The alpha coefficient used to weight the
            aggregations of in- and out-edges as part of a convex combination.
            (default: :obj:`0.5`)
        root_weight (bool, optional): If set to :obj:`True`, the layer will add
            transformed root node features to the output.
            (default: :obj:`True`)
    """
    def __init__(
        self,
        conv: MessagePassing,
        alpha: float = 0.5,
        root_weight: bool = True,
    ):
        super().__init__()

        self.alpha = alpha
        self.root_weight = root_weight

        self.conv_in = copy.deepcopy(conv)
        self.conv_out = copy.deepcopy(conv)

        if hasattr(conv, 'add_self_loops'):
            self.conv_in.add_self_loops = False
            self.conv_out.add_self_loops = False
        if hasattr(conv, 'root_weight'):
            self.conv_in.root_weight = False
            self.conv_out.root_weight = False

        if root_weight:
            self.lin = torch.nn.Linear(conv.in_channels, conv.out_channels)
        else:
            self.lin = None

        self.reset_parameters()

    def reset_parameters(self):
        r"""Resets all learnable parameters of the module."""
        self.conv_in.reset_parameters()
        self.conv_out.reset_parameters()
        if self.lin is not None:
            self.lin.reset_parameters()

    def forward(self, x: Tensor, edge_index: Tensor) -> Tensor:

        x_in = self.conv_in(x, edge_index)
        x_out = self.conv_out(x, edge_index.flip([0]))
        out = self.alpha * x_out + (1 - self.alpha) * x_in

        if self.root_weight:
            out = out + self.lin(x)

        return out

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}({self.conv_in}, alpha={self.alpha})'