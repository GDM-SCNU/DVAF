import numpy as np
import torch
from torch.nn.init import xavier_normal_
import torch
from torch.nn import Parameter, Module
import torch.nn.functional as F
import math
import numpy as np
from gnn_layers import *
from load_data import Data
from load_data import Data
import numpy as np
import torch
from collections import defaultdict
import argparse
from utils_WGE import construct_entity_focus_matrix, construct_relation_focus_matrix, get_deg, get_er_vocab, get_batch,get_t_vocab,get_ert_vocab
from model import *
from tqdm import tqdm
from torch.nn import functional as F
from torch.utils.data import DataLoader
torch.manual_seed(1337)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(1337)
    torch.backends.cudnn.deterministic = True
np.random.seed(1337)


def get_quaternion_wise_mul(quaternion):
    size = quaternion.size(1) // 4
    quaternion = quaternion.view(-1, 4, size)
    quaternion = torch.sum(quaternion, 1)
    return quaternion



def vec_vec_wise_multiplication(q, p):  # vector * vector
    normalized_p = normalization(p)  # bs x 4dim
    q_r, q_i, q_j, q_k = make_wise_quaternion(q)  # bs x 4dim

    qp_r = get_quaternion_wise_mul(q_r * normalized_p)  # qrpr−qipi−qjpj−qkpk
    qp_i = get_quaternion_wise_mul(q_i * normalized_p)  # qipr+qrpi−qkpj+qjpk
    qp_j = get_quaternion_wise_mul(q_j * normalized_p)  # qjpr+qkpi+qrpj−qipk
    qp_k = get_quaternion_wise_mul(q_k * normalized_p)  # qkpr−qjpi+qipj+qrpk

    return torch.cat([qp_r, qp_i, qp_j, qp_k], dim=1)



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



''' The re-implementation of Quaternion Knowledge Graph Embeddings (https://arxiv.org/abs/1904.10281), following the 1-N scoring strategy '''
class QuatE(torch.nn.Module):
    def __init__(self, emb_dim, n_entities, n_relations):
        super(QuatE, self).__init__()
        self.n_entities = n_entities
        self.n_relations = n_relations
        self.embeddings = torch.nn.Embedding(self.n_entities + self.n_relations, emb_dim)
        torch.nn.init.xavier_normal_(self.embeddings.weight.data)
        self.loss = torch.nn.BCELoss()

    def forward(self, e1_idx, r_idx, lst_indexes):
        X = self.embeddings(lst_indexes)
        h = X[e1_idx]
        r = X[r_idx + self.n_entities]
        hr = vec_vec_wise_multiplication(h, r)
        hrt = torch.mm(hr, X[:self.n_entities].t())  # following the 1-N scoring strategy in ConvE
        pred = torch.sigmoid(hrt)
        return pred


def make_wise_quaternion(quaternion):  # for vector * vector quaternion element-wise multiplication
    if len(quaternion.size()) == 1:
        quaternion = quaternion.unsqueeze(0)
    size = quaternion.size(1) // 4
    r, i, j, k = torch.split(quaternion, size, dim=1)
    r2 = torch.cat([r, -i, -j, -k], dim=1)  # 0, 1, 2, 3 --> bs x 4dim
    i2 = torch.cat([i, r, -k, j], dim=1)  # 1, 0, 3, 2
    j2 = torch.cat([j, k, r, -i], dim=1)  # 2, 3, 0, 1
    k2 = torch.cat([k, -j, i, r], dim=1)  # 3, 2, 1, 0
    return r2, i2, j2, k2


def normalization(quaternion, split_dim=1):  # vectorized quaternion bs x 4dim
    size = quaternion.size(split_dim) // 4
    quaternion = quaternion.reshape(-1, 4, size)  # bs x 4 x dim
    quaternion = quaternion / torch.sqrt(torch.sum(quaternion ** 2, 1, True))  # quaternion / norm
    quaternion = quaternion.reshape(-1, 4 * size)
    return quaternion


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
        if self.act is not None:
            return self.act(output)
        else:
            return output
""""""

import torch
import torch.nn as nn
from torch_geometric.nn import MessagePassing


''' Quaternion Graph Isomophism Network!'''
class Q4GIN0Layer(Module):
    def __init__(self, in_features, hid_feature, out_features, act=torch.tanh):
        super(Q4GIN0Layer, self).__init__()
        self.in_features = in_features
        self.hid_feature = hid_feature
        self.out_features = out_features
        self.act = act
        #
        self.weight1 = Parameter(torch.FloatTensor(self.in_features // 4, self.hid_feature))
        self.weight2 = Parameter(torch.FloatTensor(self.hid_feature // 4, self.out_features))

        self.reset_parameters()
        self.bn = torch.nn.BatchNorm1d(out_features)

    def reset_parameters(self):
        stdv = math.sqrt(6.0 / (self.weight1.size(0) + self.weight1.size(1)))
        self.weight1.data.uniform_(-stdv, stdv)
        self.weight2.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        hamilton1 = make_quaternion_mul(self.weight1)
        hamilton2 = make_quaternion_mul(self.weight2)
        output1 = torch.mm(input, hamilton1)  # Hamilton product, quaternion multiplication!
        if self.act is not None:
            output1 = self.act(output1)
        output2 = torch.mm(output1, hamilton2)

        output = torch.spmm(adj, output2)
        output = self.bn(output)
        return output


class CompGATv3Layer(MessagePassing):
    def __init__(self, in_channels, out_channels, rel_dim, drop, bias, op, beta):
        super(CompGATv3Layer, self).__init__(aggr="add", node_dim=0)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.rel_dim=rel_dim
        self.op = op
        self.bias = bias
        self.beta = beta

        # self.w_loop = torch.nn.Linear(in_channels, out_channels).cuda()
        self.w_in = torch.nn.Linear(in_channels, out_channels).cuda()
        self.w_out = torch.nn.Linear(in_channels, out_channels).cuda()
        self.w_rel = torch.nn.Linear(in_channels, out_channels).cuda()
        self.w_att = torch.nn.Linear(3 * in_channels, out_channels).cuda()
        self.a = torch.nn.Linear(out_channels, 1, bias=False).cuda()
        # self.loop_rel = torch.nn.Parameter(torch.Tensor(1, in_channels)).cuda()
        # torch.nn.init.xavier_uniform_(self.loop_rel)

        self.drop_ratio = drop
        self.drop = torch.nn.Dropout(drop, inplace=False)
        self.bn = torch.nn.BatchNorm1d(out_channels).to(torch.device("cuda"))

        self.res_w = torch.nn.Linear(in_channels, out_channels, bias=False)
        self.activation = torch.nn.Tanh()  # torch.nn.Tanh()
        self.leaky_relu = torch.nn.LeakyReLU(negative_slope=0.2, inplace=True)

        if bias:
            self.register_parameter("bias_value", torch.nn.Parameter(torch.zeros(out_channels)))

        self.init_weight()

    def init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.fill_(0.0)

    def forward(self, x, edge_index, edge_type, rel_emb, pre_alpha=None):
       # rel_emb = torch.cat([rel_emb, self.loop_rel], dim=0)
        print(x)
        num_ent = x.size(0)
        #loop_index = torch.stack([torch.arange(num_ent), torch.arange(num_ent)]).cuda()
        # loop_type   = torch.full((num_ent,), rel_emb.size(0)-1, dtype=torch.long).cuda()

        # in_res = self.propagate(edge_index=edge_index, x=x, edge_type=edge_type, rel_emb=rel_emb, pre_alpha=pre_alpha)

        messages = self.message(x=x, edge_index=edge_index, edge_type=edge_type, rel_emb=rel_emb, pre_alpha=pre_alpha)

        # Step 2: Aggregate messages
        aggregated_messages = self.aggregate(messages, edge_index)

        # Step 3: Update node features
        in_res = self.update(aggregated_messages, x)



       # Perform any further computations on the messages
    # loop_res = self.propagate(edge_index=loop_index, x=x, edge_type=loop_type, rel_emb=rel_emb, pre_alpha=pre_alpha, mode="loop")
        loop_res = self.res_w(x)
        out = self.drop(in_res) + self.drop(loop_res)

        if self.bias:
            out = out + self.bias_value

        out = self.bn(out)
        out = self.activation(out)

        return out, self.w_rel(rel_emb), self.alpha.detach()

    def message(self, x_i, x_j, edge_type, rel_emb, ptr, index, size_i, pre_alpha):

        rel_emb = torch.index_select(rel_emb, 0, edge_type)
        xj_rel = self.rel_transform(x_j, rel_emb)
        num_edge = xj_rel.size(0) // 2

        in_message = xj_rel[:num_edge]
        out_message = xj_rel[num_edge:]

        trans_in = self.w_in(in_message)
        trans_out = self.w_out(out_message)

        out = torch.cat((trans_in, trans_out), dim=0)

        b = self.leaky_relu(self.w_att(torch.cat((x_i, rel_emb, x_j), dim=1))).cuda()
        b = self.a(b).float()
        alpha = softmax(b, index, ptr, size_i)
        alpha = F.dropout(alpha, p=self.drop_ratio, training=self.training, inplace=False)
        if pre_alpha != None and self.beta != 0:
            self.alpha = alpha * (1 - self.beta) + pre_alpha * (self.beta)
        else:
            self.alpha = alpha
        out = out * alpha.view(-1, 1)

        return out

    def update(self, aggr_out):
        return aggr_out

    def rel_transform(self, ent_embed, rel_emb):

        if self.op == 'corr':
            trans_embed = ccorr(ent_embed, rel_emb)
        elif self.op == 'sub':
            trans_embed = ent_embed - rel_emb
        elif self.op == 'mult':
            trans_embed = ent_embed * rel_emb
        elif self.op == "corr_new":
            trans_embed = ccorr_new(ent_embed, rel_emb)
        elif self.op == "conv":
            trans_embed = cconv(ent_embed, rel_emb)
        elif self.op == "conv_new":
            trans_embed = cconv_new(ent_embed, rel_emb)
        elif self.op == 'cross':
            trans_embed = ent_embed * rel_emb + ent_embed
        elif self.op == "corr_plus":
            trans_embed = ccorr_new(ent_embed, rel_emb) + ent_embed
        elif self.op == "rotate":
            trans_embed = rotate(ent_embed, rel_emb)

        else:
            raise NotImplementedError

        return trans_embed


class CompGATv3_model(torch.nn.Module):
    def __init__(self, args, num_ents, num_rels, adj, adj_r, deg, adj_edge_type,adj_edge_index):
        super(CompGATLayer, self).__init__()
        # self.args = args
        # #self.emb_dim = args.emb_dim
        # #self.emb_dim = args.emb_dim
        # self.n_entities = num_ents
        # self.n_relations = num_rels
        # self.deg = deg
        # self.thetas = nn.Parameter(torch.ones(3))
        # self.all_embeddings = torch.nn.Embedding(self.n_entities + self.n_relations, self.emb_dim)
        # # import pdb; pdb.set_trace()
        # self.adj = adj
        # self.adj_r = adj_r
        self.ent_dim = args.emb_dim
        self.learning_rate = args.cl_lr
        self.batch_size = args.batch_size
        self.op = args.op
        self.ent_emb = torch.nn.Embedding(num_ents, args.init_dim).to(torch.device("cuda"))
        self.rel_emb = torch.nn.Embedding(num_rels, args.init_dim).to(torch.device("cuda"))
        torch.nn.init.xavier_uniform_(self.ent_emb.weight)
        torch.nn.init.xavier_uniform_(self.rel_emb.weight)

        self.ent_dim = args.ent_dim
        self.ent_h = args.ent_height
        self.ent_w = self.ent_dim // self.ent_h
        if args.gcn_layer == 2:
            # self.encoder = Sequential("x, edge_index, edge_type, rel_emb",[
            #     (CompGATv3(in_channels=args.init_dim, out_channels=args.ent_dim, rel_dim=args.rel_dim, drop=args.encoder_drop, bias=True, op=args.op, beta=args.beta), "x, edge_index, edge_type, rel_emb -> x, rel_emb, alpha"),
            #     (, "x -> x"),
            #     (CompGATv3(in_channels=args.ent_dim, out_channels=args.ent_dim, rel_dim=args.rel_dim, drop=args.encoder_drop, bias=True, op=args.op, beta=args.beta), "x, edge_index, edge_type, rel_emb, alpha -> x, rel_emb, alpha"),
            # ]).to(torch.device("cuda"))
            self.layer = 2
            self.gnn1 = CompGATv3(in_channels=args.init_dim, out_channels=args.ent_dim, rel_dim=args.rel_dim, drop=args.encoder_drop, bias=True, op=args.op, beta=args.beta)
            self.hid_drop = torch.nn.Dropout(args.hid_drop)
            self.gnn2 = CompGATv3(in_channels=args.ent_dim, out_channels=args.ent_dim, rel_dim=args.rel_dim, drop=args.encoder_drop, bias=True, op=args.op, beta=args.beta)

        elif args.gcn_layer == 1:
            # self.encoder = Sequential("x, edge_index, edge_type, rel_emb",[
            #     (CompGATv3(in_channels=args.init_dim, out_channels=args.ent_dim, rel_dim=args.rel_dim, drop=args.encoder_drop, bias=True, op=self.op, beta=args.beta), "x, edge_index, edge_type, rel_emb -> x, rel_emb, alpha"),
            # ]).to(torch.device("cuda"))
            self.layer = 1
            self.gnn1 = CompGATv3(in_channels=args.init_dim, out_channels=args.ent_dim, rel_dim=args.rel_dim, drop=args.encoder_drop, bias=True, op=args.op, beta=args.beta)
            self.hid_drop = torch.nn.Dropout(args.hid_drop)

        elif args.gcn_layer == 4:
            self.layer = 4
            self.gnn1 = CompGATv3(in_channels=args.init_dim, out_channels=args.ent_dim, rel_dim=args.rel_dim, drop=args.encoder_drop, bias=True, op=args.op, beta=args.beta)
            self.gnn2 = CompGATv3(in_channels=args.init_dim, out_channels=args.ent_dim, rel_dim=args.rel_dim, drop=args.encoder_drop, bias=True, op=args.op, beta=args.beta)
            self.gnn3 = CompGATv3(in_channels=args.init_dim, out_channels=args.ent_dim, rel_dim=args.rel_dim, drop=args.encoder_drop, bias=True, op=args.op, beta=args.beta)
            self.gnn4 = CompGATv3(in_channels=args.init_dim, out_channels=args.ent_dim, rel_dim=args.rel_dim, drop=args.encoder_drop, bias=True, op=args.op, beta=args.beta)
            self.hid_drop = torch.nn.Dropout(args.hid_drop)

        self.encoder_drop = torch.nn.Dropout(p=args.encoder_hid_drop, inplace=False)
        # self.input_drop = torch.nn.Dropout(args.input_drop, inplace=False)
        self.fea_drop = torch.nn.Dropout(args.fea_drop, inplace=False)
        # self.hid_drop = torch.nn.Dropout(args.hid_drop)

        self.conv1 = torch.nn.Conv2d(1, args.filter_channel, (args.filter_size, args.filter_size), 1, 0, bias=False)
        self.bn0 = torch.nn.BatchNorm2d(1)
        self.bn1 = torch.nn.BatchNorm2d(args.filter_channel)
        self.bn2 = torch.nn.BatchNorm1d(args.ent_dim)
        self.register_parameter('b', torch.nn.Parameter(torch.zeros(num_ents)))

        num_in_features = (args.filter_channel*(2*args.ent_height-args.filter_size+1)*(self.ent_w-args.filter_size+1))
        if args.proj == "mlp":
            self.proj = torch.nn.Sequential(
                torch.nn.Linear(num_in_features, args.ent_dim),
                torch.nn.BatchNorm1d(args.ent_dim),
                torch.nn.PReLU(),
                torch.nn.Linear(args.ent_dim, args.ent_dim),
                torch.nn.BatchNorm1d(args.ent_dim))

        elif args.proj == "linear":
            self.proj = torch.nn.Sequential(
                torch.nn.Linear(num_in_features, args.ent_dim),
                torch.nn.Dropout(args.hid_drop),
                torch.nn.BatchNorm1d(args.ent_dim),
                torch.nn.PReLU())
        else:
            raise ValueError("Type of projection head should be mlp or linear!")

        self.relu = torch.nn.PReLU()
        self.init_weight()

        def init_weight(self):
            for m in self.modules():
                if isinstance(m, nn.Linear):
                    torch.nn.init.xavier_uniform_(m.weight.data)
                    if m.bias is not None:
                        m.bias.data.fill_(0.0)

        def forward(self, edge_index, edge_type, h, r, t, ent_emb, rel_emb, save=False):
            if self.layer == 2:
                ent_emb1, rel_emb1, alpha1 = self.gnn1(ent_emb, edge_index, edge_type, rel_emb)
                ent_emb1 = self.hid_drop(ent_emb1)
                ent_emb2, rel_emb, alpha2 = self.gnn2(ent_emb1, edge_index, edge_type, rel_emb1, alpha1)
                ent_emb = self.hid_drop(ent_emb2)

            elif self.layer == 1:
                ent_emb, rel_emb, alpha1 = self.gnn1(ent_emb, edge_index, edge_type, rel_emb)
                ent_emb = self.hid_drop(ent_emb)

            elif self.layer == 4:
                ent_emb, rel_emb, alpha1 = self.gnn1(ent_emb, edge_index, edge_type, rel_emb)
                ent_emb = self.hid_drop(ent_emb)
                ent_emb, rel_emb, alpha2 = self.gnn2(ent_emb, edge_index, edge_type, rel_emb, alpha1)
                ent_emb = self.hid_drop(ent_emb)
                ent_emb, rel_emb, alpha3 = self.gnn3(ent_emb, edge_index, edge_type, rel_emb, alpha2)
                ent_emb = self.hid_drop(ent_emb)
                ent_emb, rel_emb, alpha4 = self.gnn4(ent_emb, edge_index, edge_type, rel_emb, alpha3)
                ent_emb = self.hid_drop(ent_emb)

            head_emb = torch.index_select(ent_emb, 0, h)
            r_emb = torch.index_select(rel_emb, 0, r).view(-1, 1, self.ent_dim)
            tail_emb = torch.index_select(ent_emb, 0, t)

            x = head_emb.view(-1, 1, self.ent_dim)
            x = torch.cat([x, r_emb], 1)
            x = torch.transpose(x, 2, 1).reshape((-1, 1, 2 * self.ent_h, self.ent_w))

            x = self.bn0(x)
            x = self.conv1(x)
            x = self.bn1(x)
            x = self.relu(x)
            x = self.fea_drop(x)
            x = x.view(x.shape[0], -1)
            x = self.proj(x)
            if save:
                return x, ent_emb
            x = self.bn2(x)
            cl_x = x
            # x = self.hid_drop(x)

            # x = F.relu(x, inplace=True)
            # ent_emb = self.bn2(ent_emb)
            x = torch.mm(x, ent_emb.transpose(1, 0))
            x += self.b.expand_as(x)
            # x = torch.sigmoid(x)
            return cl_x, x, tail_emb, head_emb, ent_emb, rel_emb




        self.num_rels=num_rels
        self.edge_index=edge_index
        self.edge_type=edge_type

        def __init__(self, args):
            super(CLKG_compgatv3_convE, self).__init__()
            self.ent_dim = args.ent_dim
            self.learning_rate = args.cl_lr
            self.batch_size = args.cl_batch_size
            self.op = args.op

            self.ent_emb = torch.nn.Embedding(args.ent_num, args.init_dim).to(torch.device("cuda"))
            self.rel_emb = torch.nn.Embedding(args.rel_num, args.init_dim).to(torch.device("cuda"))
            torch.nn.init.xavier_uniform_(self.ent_emb.weight)
            torch.nn.init.xavier_uniform_(self.rel_emb.weight)

        self.lst_gcn1 = torch.nn.ModuleList()
        self.lst_gcn2 = torch.nn.ModuleList()
        for _layer in range(args.num_layers):
            if self.args.encoder == "qgnn":
                self.lst_gcn1.append(Q4GNNLayer(self.emb_dim, self.emb_dim, act=torch.tanh))
                self.lst_gcn2.append(Q4GNNLayer(self.emb_dim, self.emb_dim, act=torch.tanh))
            else:
                print("This encoder has not been implemented... Existing")
                exit()

        self.linear_ents = torch.nn.ParameterList()
        self.linear_ents_cor = torch.nn.ParameterList()
        
        for _layer in range(args.num_layers):
            self.linear_ents.append(Parameter(torch.FloatTensor(self.emb_dim // 2, self.emb_dim)))
            self.linear_ents_cor.append(Parameter(torch.FloatTensor(self.emb_dim // 4, self.emb_dim)))
        
        # if not self.args.best_model:
        self.reset_parameters()


        xavier_normal_(self.all_embeddings.weight.data)
        # import pdb; pdb.set_trace()

        self.activate = torch.nn.Tanh()

        self.bn1 = nn.BatchNorm1d(self.emb_dim)
        self.bn2 = nn.BatchNorm1d(self.emb_dim)
        self.bn3 = nn.BatchNorm1d(self.emb_dim)
        self.dropout1 = torch.nn.Dropout()
        self.dropout2 = torch.nn.Dropout()
        self.dropout3 = torch.nn.Dropout()
        self.loss = torch.nn.BCELoss()

    def reset_parameters(self):
        for i in range(len(self.linear_ents)):
            weight = self.linear_ents[i]
            stdv = math.sqrt(6.0 / (weight.size(0) + weight.size(1)))
            self.linear_ents[i].data.uniform_(-stdv, stdv)

        for i in range(len(self.linear_ents_cor)):
            weight = self.linear_ents_cor[i]
            stdv = math.sqrt(6.0 / (weight.size(0) + weight.size(1)))
            self.linear_ents_cor[i].data.uniform_(-stdv, stdv)
        

    def forward_normal(self, e1_idx, r_idx, lst_indexes1, lst_indexes2, train=False):
        self.scorer = self.quate
        
        X = self.all_embeddings(lst_indexes1)
        R = self.all_embeddings(lst_indexes2[len(lst_indexes1):])
        h1 = X[e1_idx]
        r1 = R[r_idx]

        hs = [h1]
        rs = [r1]

        #scores = [torch.sigmoid(self.scorer(hs[-1], rs[-1], X, 1))]#头 关系
        scores = [torch.sigmoid(self.scorer(hs[-1], rs[-1], X, 1))]

        for _layer in range(self.args.num_layers):
            XR = torch.cat((X, R), dim=0) # last 
            XRrf = self.lst_gcn2[_layer](XR, self.adj_r) # newX, newR from relational graph
            Xef = self.lst_gcn1[_layer](X, self.adj) # newX2 from original graph 
            Xrf = XRrf[lst_indexes1]
            if self.args.combine_type == "cat":
                size = Xrf.size(1) // 4
                Xef1, Xef2, Xef3, Xef4 = torch.split(Xef, size, dim=1)
                Xrf1, Xrf2, Xrf3, Xrf4 = torch.split(Xrf, size, dim=1)
                X = torch.cat([Xef1, Xrf1, Xef2, Xrf2, Xef3, Xrf3, Xef4, Xrf4], dim=1)
                hamilton = make_quaternion_mul(self.linear_ents[_layer])
                X = torch.mm(X, hamilton)
            elif self.args.combine_type == "sum":
                X = Xef + Xrf
            elif self.args.combine_type == "corr":
                X = Xef * Xrf
            elif self.args.combine_type == "linear_corr":
                hamilton = make_quaternion_mul(self.linear_ents_cor[_layer])
                X = Xef * torch.mm(Xrf, hamilton)

            R = XRrf[lst_indexes2[len(lst_indexes1):]] # newR
            hs.append(X[e1_idx]) # finalX
            rs.append(R[r_idx]) # finalR
            scores.append(torch.sigmoid(self.scorer(hs[-1], rs[-1], X, _layer + 2)))

        return scores


    def quate(self, h, r, X, layer_index=1):
        hr = vec_vec_wise_multiplication(h, r)
        if layer_index == 1:
            hr = self.bn1(hr) 
            hr = self.dropout1(hr) 
        elif layer_index == 2:
            hr = self.bn2(hr)
            hr = self.dropout2(hr)
        else:
            hr = self.bn3(hr)
            hr = self.dropout3(hr)
        hrt = torch.mm(hr, X.t())
        return hrt


    def distmult(self, h, r, X, layer_index=1):
        hr = h * r 
        if layer_index == 1:
            hr = self.bn1(hr) 
            hr = self.dropout1(hr) 
        elif layer_index == 2:
            hr = self.bn2(hr)
            hr = self.dropout2(hr)
        else:
            hr = self.bn3(hr)
            hr = self.dropout3(hr)
        hrt = torch.mm(hr, X.t())
        return hrt 


    def transe(self, h, r, X, layer_index=1):
        hr = h + r 
        hrt = 20 - torch.norm(hr.unsqueeze(1) - X, p=1, dim=2)
        return hrt


    def normalize_embedding(self):
        embed = self.all_embeddings.weight.detach().cpu().numpy()[:self.n_entities]
        rel_emb = self.all_embeddings.weight.detach().cpu().numpy()[self.n_entities:]
        embed = embed / np.sqrt(np.sum(np.square(embed), axis=1, keepdims=True))
        
        self.all_embeddings.weight.data.copy_(torch.from_numpy(np.concatenate((embed, rel_emb), axis=0)))


    def get_hidden_feature(self):
        return self.feat_list


    def regularization(self, dis_loss, margin=1.5):
        return max(0, margin - dis_loss)


    def get_factor(self):
        factor_list = []
        factor_list.append(self.distangle.get_factor())
        return factor_list


    def compute_disentangle_loss(self):
        return self.distangle.compute_disentangle_loss()


    @staticmethod
    def merge_loss(dis_loss):
        return dis_loss


# import torch
# from torch.nn import functional as F
# import numpy as np
# from torch.utils.data import DataLoader
# from model import CLKG_compgatv3_convE, SupConLoss, relation_contrast
# import pytorch_lightning as pl
from load_data import KG_Triples_txt, txt2triples, KG_Triples
from utils_WGE import *
# from torch_geometric.nn import Sequential
# import argparse
# from Evaluation import Evaluator
# from pytorch_lightning.callbacks import ModelCheckpoint
# from os import listdir
# from pytorch_lightning.utilities.seed import seed_everything
# from torch.optim.lr_scheduler import CosineAnnealingLR
# import logging

import torch
from torch_sparse import coalesce
class kgat_model(torch.nn.Module):
    def __init__(self, args, num_ents, num_rels, adj, adj_r, deg):
        super(W_model, self).__init__()
        self.args = args
        self.emb_dim = args.emb_dim
        self.emb_dim = args.emb_dim
        self.n_entities = num_ents
        self.n_relations = num_rels
        self.deg = deg

        self.thetas = nn.Parameter(torch.ones(3))

        # self.learning_rate = args.cl_lr
        # self.evaluator = Evaluator(num_ent=args.ent_num, batch_size=2048)
        # self.augmenter1 = Random_Choice(auglist1, args.num_choice)
        # self.augmenter2 = Random_Choice(auglist2, args.num_choice)
        # self.model = model
        # self.supconloss = SupConLoss(temperature=args.temp1, contrast_mode="all", base_temperature=args.temp1).to(
        #     torch.device("cuda"))
        self.rank_loss = torch.nn.CrossEntropyLoss()
        # self.rel_cl = relation_contrast(args.temp2, args.neg_sample)
        self.lam1 = args.lam1
        self.lam2 = args.lam2
        self.wd = args.weight_decay
        valid_triple, src, rel, dst = txt2triples(args.train_path + "train2id.txt")
        if args.noise_path is not None:
            noise_triple, _, _, _ = txt2triples(args.noise_path + "train2id.txt")
            _, src, rel, dst = add_noise(valid_triple, noise_triple)

        self.edge_type = rel.cuda()
        self.edge_index = torch.stack((src, dst), dim=0).cuda()
        self.label_smoothing = args.label_smoothing
        # self.num_entities = args.ent_num
        # logging.basicConfig(filename="./log/{}.log".format(args.info), filemode="w", level=logging.INFO,
        # #                     format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        # # self.log2 = logging.getLogger(__name__)
        # # self.log2.info(args)

        self.all_embeddings = torch.nn.Embedding(self.n_entities + self.n_relations, self.emb_dim)
        # import pdb; pdb.set_trace()
        self.adj = adj
        self.adj_r = adj_r
        self.num_rels = num_rels
        #self.edge_index = edge_index
        # self.edge_type = edge_type
        self.ent_dim = args.emb_dim
        self.learning_rate = args.cl_lr
        self.batch_size = args.batch_size
        self.op = args.op
        # self.ent_emb = torch.nn.Embedding(num_ents, args.init_dim).to(torch.device("cuda"))
        # self.rel_emb = torch.nn.Embedding(num_rels, args.init_dim).to(torch.device("cuda"))

        self.ent_emb = torch.nn.Embedding(args.ent_num, args.init_dim).to(torch.device("cuda"))
        self.rel_emb = torch.nn.Embedding(args.rel_num, args.init_dim).to(torch.device("cuda"))
        torch.nn.init.xavier_uniform_(self.ent_emb.weight)
        torch.nn.init.xavier_uniform_(self.rel_emb.weight)

        self.ent_dim = args.ent_dim
        self.ent_h = args.ent_height
        self.ent_w = self.ent_dim // self.ent_h

        self.lst_gcn1 = torch.nn.ModuleList()
        self.lst_gcn2 = torch.nn.ModuleList()
        for _layer in range(args.num_layers):
            if self.args.encoder == "kgat":
                self.lst_gcn1.append(CompGATv3Layer(in_channels=args.init_dim, out_channels=args.ent_dim, rel_dim=args.rel_dim,
                              drop=args.encoder_drop, bias=True, op=args.op, beta=args.beta))
                self.lst_gcn2.append(Q4GNNLayer(self.emb_dim, self.emb_dim, act=torch.tanh))
                #self.hid_drop = torch.nn.Dropout(args.hid_drop)
                #self.lst_gcn2.append(CompGATv3(in_channels=args.init_dim, out_channels=args.ent_dim, rel_dim=args.rel_dim,
                              #drop=args.encoder_drop, bias=True, op=args.op, beta=args.beta))

            else:
                print("This encoder has not been implemented... Existing")
                exit()

        self.linear_ents = torch.nn.ParameterList()
        self.linear_ents_cor = torch.nn.ParameterList()

        for _layer in range(args.num_layers):
            self.linear_ents.append(Parameter(torch.FloatTensor(self.emb_dim // 2, self.emb_dim)))
            self.linear_ents_cor.append(Parameter(torch.FloatTensor(self.emb_dim // 4, self.emb_dim)))

        # if not self.args.best_model:
        self.reset_parameters()

        xavier_normal_(self.all_embeddings.weight.data)
        # import pdb; pdb.set_trace()

        self.activate = torch.nn.Tanh()

        self.bn1 = nn.BatchNorm1d(self.emb_dim)
        self.bn2 = nn.BatchNorm1d(self.emb_dim)
        self.bn3 = nn.BatchNorm1d(self.emb_dim)
        self.dropout1 = torch.nn.Dropout()
        self.dropout2 = torch.nn.Dropout()
        self.dropout3 = torch.nn.Dropout()
        self.loss = torch.nn.BCELoss()

    def reset_parameters(self):
        for i in range(len(self.linear_ents)):
            weight = self.linear_ents[i]
            stdv = math.sqrt(6.0 / (weight.size(0) + weight.size(1)))
            self.linear_ents[i].data.uniform_(-stdv, stdv)

        for i in range(len(self.linear_ents_cor)):
            weight = self.linear_ents_cor[i]
            stdv = math.sqrt(6.0 / (weight.size(0) + weight.size(1)))
            self.linear_ents_cor[i].data.uniform_(-stdv, stdv)

    def forward_normal(self, e1_idx, r_idx, lst_indexes1, lst_indexes2, train=False):
        self.scorer = self.quate

        X = self.all_embeddings(lst_indexes1)
        R = self.all_embeddings(lst_indexes2[len(lst_indexes1):])
        h1 = X[e1_idx]
        r1 = R[r_idx]
        hs = [h1]
        rs = [r1]
        ent_emb=self.ent_emb

        rel_emb=self.rel_emb
        edge_index=self.edge_index
        edge_type=self.edge_type

        scores = [torch.sigmoid(self.scorer(hs[-1], rs[-1], X, 1))]

        for _layer in range(self.args.num_layers):
            XR = torch.cat((X, R), dim=0)  # last
            XRrf = self.lst_gcn2[_layer](XR, self.adj_r)  # newX, newR from relational graph
            #Xef = self.lst_gcn1[_layer](X, self.adj)  # newX2 from original graph
            #ent_emb1, rel_emb1, alpha1 = self.gnn1(ent_emb, edge_index, edge_type, rel_emb)#输出实体嵌入、关系嵌入、以及注意力系数
            print(ent_emb)
            # Xef,rel_emb1, alpha1= self.lst_gcn1[_layer](ent_emb, edge_index, edge_type, rel_emb)
            Xef, rel_emb1, alpha1 = self.lst_gcn1[_layer](ent_emb, edge_index, edge_type, rel_emb)

            Xrf = XRrf[lst_indexes1]
            if self.args.combine_type == "cat":
                size = Xrf.size(1) // 4
                Xef1, Xef2, Xef3, Xef4 = torch.split(Xef, size, dim=1)
                Xrf1, Xrf2, Xrf3, Xrf4 = torch.split(Xrf, size, dim=1)
                X = torch.cat([Xef1, Xrf1, Xef2, Xrf2, Xef3, Xrf3, Xef4, Xrf4], dim=1)
                hamilton = make_quaternion_mul(self.linear_ents[_layer])
                X = torch.mm(X, hamilton)
            elif self.args.combine_type == "sum":
                X = Xef + Xrf
            elif self.args.combine_type == "corr":
                X = Xef * Xrf
            elif self.args.combine_type == "linear_corr":
                hamilton = make_quaternion_mul(self.linear_ents_cor[_layer])
                X = Xef * torch.mm(Xrf, hamilton)

            R = XRrf[lst_indexes2[len(lst_indexes1):]]  # newR
            hs.append(X[e1_idx])  # finalX
            rs.append(R[r_idx])  # finalR
            scores.append(torch.sigmoid(self.scorer(hs[-1], rs[-1], X, _layer + 2)))

        return scores

    def quate(self, h, r, X, layer_index=1):
        hr = vec_vec_wise_multiplication(h, r)
        if layer_index == 1:
            hr = self.bn1(hr)
            hr = self.dropout1(hr)
        elif layer_index == 2:
            hr = self.bn2(hr)
            hr = self.dropout2(hr)
        else:
            hr = self.bn3(hr)
            hr = self.dropout3(hr)
        hrt = torch.mm(hr, X.t())
        return hrt

    def predicted_relation(self, h, t, R, layer_index=1):
        ht = vec_vec_wise_multiplication(h, t)
        if layer_index == 1:
            ht = self.bn1(ht)
            ht = self.dropout1(ht)
        elif layer_index == 2:
            ht = self.bn2(ht)
            ht = self.dropout2(ht)
        else:
            ht = self.bn3(ht)
            ht = self.dropout3(ht)
        scores = torch.mm(h, R.t())
        predicted_relation = torch.argmax(scores, dim=1)
        return predicted_relation

    def distmult(self, h, r, X, layer_index=1):
        hr = h * r
        if layer_index == 1:
            hr = self.bn1(hr)
            hr = self.dropout1(hr)
        elif layer_index == 2:
            hr = self.bn2(hr)
            hr = self.dropout2(hr)
        else:
            hr = self.bn3(hr)
            hr = self.dropout3(hr)
        hrt = torch.mm(hr, X.t())
        return hrt

    def transe(self, h, r, X, layer_index=1):
        hr = h + r
        hrt = 20 - torch.norm(hr.unsqueeze(1) - X, p=1, dim=2)
        return hrt

    def normalize_embedding(self):
        embed = self.all_embeddings.weight.detach().cpu().numpy()[:self.n_entities]
        rel_emb = self.all_embeddings.weight.detach().cpu().numpy()[self.n_entities:]
        embed = embed / np.sqrt(np.sum(np.square(embed), axis=1, keepdims=True))

        self.all_embeddings.weight.data.copy_(torch.from_numpy(np.concatenate((embed, rel_emb), axis=0)))

    def get_hidden_feature(self):
        return self.feat_list

    def regularization(self, dis_loss, margin=1.5):
        return max(0, margin - dis_loss)

    def get_factor(self):
        factor_list = []
        factor_list.append(self.distangle.get_factor())
        return factor_list

    def compute_disentangle_loss(self):
        return self.distangle.compute_disentangle_loss()

    @staticmethod
    def merge_loss(dis_loss):
        return dis_loss

from torch.nn import Linear, ReLU, Dropout
from torch_geometric.nn import TransformerConv,RGATConv
from torch_geometric.nn import global_mean_pool

class W_model(torch.nn.Module):
    def __init__(self, args, num_ents, num_rels, adj, adj_r, deg, dir_adj_edge_index, dir_adjr_edge_index):
        super(W_model, self).__init__()
        self.args = args
        self.emb_dim = args.emb_dim
        self.emb_dim = args.emb_dim
        self.n_entities = num_ents
        self.n_relations = num_rels
        self.deg = deg
        self.dir_adj_edge_index=dir_adj_edge_index
        self.dir_adjr_edge_index=dir_adjr_edge_index
        self.thetas = nn.Parameter(torch.ones(3))

        self.all_embeddings = torch.nn.Embedding(self.n_entities + self.n_relations, self.emb_dim)
        print("self.all_embeddings:",self.all_embeddings)
        # import pdb; pdb.set_trace()
        self.adj = adj
        self.adj_r = adj_r
        self.num_rels = num_rels

        self.out_channels = self.emb_dim
        self.num_relations=12
        self.lst_gcn1 = torch.nn.ModuleList()
        self.lst_gcn2 = torch.nn.ModuleList()

        for _layer in range(args.num_layers):
            if self.args.encoder == "111":
              self.lst_gcn1.append(TransformerConv(self.emb_dim, self.emb_dim,heads=4,dropout=0.1,
                                    concat=False, beta=True))
               # self.lst_gcn1.append(RGATConv(64,64,12))
               # self.lst_gcn1.append(TransformerConv(self.emb_dim, self.emb_dim,heads=4,dropout=0,
               #                     concat=False, beta=True))
              self.lst_gcn2.append(Q4GNNLayer(self.emb_dim, self.emb_dim, act=torch.tanh))
               # self.lin = torch.nn.Linear(self.emb_dim, self.emb_dim)
            # elif self.args.encoder == "222":
            #      self.lst_gcn1.append(TransformerConv(self.emb_dim, self.emb_dim,heads=4,dropout=0,
            #                     concat=False, beta=True))
            #      self.lst_gcn2.appennd(TransformerConv(self.emb_dim, self.emb_dim,heads=4,dropout=0,
            #                     concat=False, beta=True))
            elif self.args.encoder == "222":
                self.lst_gcn1.append(Q4GNNLayer(self.emb_dim, self.emb_dim, act=torch.tanh))
                self.lst_gcn2.append(Q4GNNLayer(self.emb_dim, self.emb_dim, act=torch.tanh))
            elif self.args.encoder == "333":
                self.lst_gcn1.append(TransformerConv(self.emb_dim, self.emb_dim, heads=4, dropout=0.2,
                                                     concat=False, beta=True))
                self.lst_gcn2.append(TransformerConv(self.emb_dim, self.emb_dim, heads=4, dropout=0.2,
                                                     concat=False, beta=True))
            elif self.args.encoder == "dir":
                conv=TransformerConv(self.emb_dim, self.emb_dim, heads=1, dropout=0.2,
                                                     concat=False, beta=True)
                self.lst_gcn1.append(DirGNNConv(conv,alpha=0.5, root_weight=True))
                self.lst_gcn2.append(DirGNNConv(conv,alpha=0.5, root_weight=True))

            else:
                print("This encoder has not been implemented... Existing")
                exit()

        self.linear_ents = torch.nn.ParameterList()
        self.linear_ents_cor = torch.nn.ParameterList()



        # 现在，dir_adj_edge_index_with_ids 存储了原始实体id的邻接矩阵的行索引和列索引
                    # 打印转换后的edge_index
        # print(indices)
        # 打印转换后的edge_index
        # print(edge_index1)






        valid_triple, src, rel, dst = txt2triples(args.train_path + "train2id.txt")
        # if args.noise_path is not None:
        #     noise_triple, _, _, _ = txt2triples(args.noise_path + "train2id.txt")
        #     _, src, rel, dst = add_noise(valid_triple, noise_triple)

        self.edge_type = rel.cuda()
        self.edge_index = torch.stack((src, dst), dim=0).cuda()
        for _layer in range(args.num_layers):
            self.linear_ents.append(Parameter(torch.FloatTensor(self.emb_dim // 2, self.emb_dim)))
            self.linear_ents_cor.append(Parameter(torch.FloatTensor(self.emb_dim // 4, self.emb_dim)))

        # if not self.args.best_model:
        self.reset_parameters()

        xavier_normal_(self.all_embeddings.weight.data)
        # import pdb; pdb.set_trace()

        self.activate = torch.nn.Tanh()

        self.bn1 = nn.BatchNorm1d(self.emb_dim)
        self.bn2 = nn.BatchNorm1d(self.emb_dim)
        self.bn3 = nn.BatchNorm1d(self.emb_dim)
        self.dropout1 = torch.nn.Dropout()
        self.dropout2 = torch.nn.Dropout()
        self.dropout3 = torch.nn.Dropout()
        self.loss = torch.nn.BCELoss()

    def reset_parameters(self):
        for i in range(len(self.linear_ents)):
            weight = self.linear_ents[i]
            stdv = math.sqrt(6.0 / (weight.size(0) + weight.size(1)))
            self.linear_ents[i].data.uniform_(-stdv, stdv)

        for i in range(len(self.linear_ents_cor)):
            weight = self.linear_ents_cor[i]
            stdv = math.sqrt(6.0 / (weight.size(0) + weight.size(1)))
            self.linear_ents_cor[i].data.uniform_(-stdv, stdv)



    def forward_normal(self, e1_idx,e2_idx, r_idx, lst_indexes1, lst_indexes2, train=False):
        self.scorer = self.quate
        #self.r_scorer=self.predicted_relation


        X = self.all_embeddings(lst_indexes1)
        # print("X:",X.shape)
        R = self.all_embeddings(lst_indexes2[len(lst_indexes1):])
        h1 = X[e1_idx]
        r1 = R[r_idx]
        t1 = X[e2_idx]
        hs = [h1]
        rs = [r1]
        ts = [t1]
        edge_index=self.edge_index
        # print("edge_index.shape",edge_index.shape)
        edge_type=self.edge_type
        scores = [torch.sigmoid(self.scorer(hs[-1], rs[-1], X, 1))] #尾实体
        #r_scores = [torch.sigmoid(self.r_scorer(hs[-1], ts[-1], R, 1))]#关系



        # # 假设您的稀疏张量是adjacency_matrix
        adjacency_matrix = self.adj_r
        #
        # # 先对稀疏张量进行压缩
        # # 先对稀疏张量进行压缩
        adjacency_matrix = adjacency_matrix.coalesce()
        #
        # # 获取压缩后的稀疏张量的行索引和列索引
        indices = adjacency_matrix.indices()
        edge_index1=indices
        # 打印转换后的edge_index
        # print(indices)
        # 打印转换后的edge_index
        # print(edge_index1)

        dir_adj_edge_index=self.dir_adj_edge_index
        # print("dir_adj_edge_index.shape",dir_adj_edge_index.shape)
        # print("dir_adj_edge_index",dir_adj_edge_index)
        # print("edge_index1",edge_index1)
        # print("edge_index1", edge_index1.shape)

        dir_adjr_edge_index=self.dir_adjr_edge_index



        for _layer in range(self.args.num_layers):
            XR = torch.cat((X, R), dim=0)  # last
            #222
            #XRrf = self.lst_gcn2[_layer](XR, self.adj_r)  # newX, newR from relational graph
            #Xef = self.lst_gcn1[_layer](X, self.adj)  # newX2 from original graph
            #333
            #XRrf = self.lst_gcn2[_layer](XR, edge_index1)
            #Xef = self.lst_gcn1[_layer](X, edge_index)
            #dir
            #Xef = self.lst_gcn1[_layer](X, dir_adj_edge_index)
            #XRrf = self.lst_gcn2[_layer](XR, dir_adjr_edge_index)
            # XRrf = self.lst_gcn2[_layer](XR, self.adj_r)
            #XRrf = self.lst_gcn2[_layer](XR, dir_adjr_edge_index)
            # Xef=self.lst_gcn1[_layer](X,edge_index)#tranormconv
            # # # Xef = self.lst_gcn1[_layer](X, self.adj)
            # #
            # XRrf = self.lst_gcn2[_layer](XR,edge_index1)
            # XRrf = self.lst_gcn2[_layer](XR, self.adj_r)#qgnn
            # print(edge_type)
            # print(edge_index)
            # Xef = self.lst_gcn1[_layer](X, edge_index,edge_type)
            # XRrf = self.lst_gcn2[_layer](XR, self.adj_r)
            #dir-transformer
            # device = torch.device('cuda:0')  # 指定 GPU 设备
            # dir_adj_edge_index = dir_adj_edge_index.to(device)  # 将张量移动到 GPU
            # dir_adjr_edge_index = dir_adjr_edge_index.to(device)  # 将张量移动到 GPU

            dir_adj_edge_index = dir_adj_edge_index.to(device)  # 将张量移动到 GPU
            dir_adjr_edge_index = dir_adjr_edge_index.to(device)  # 将张量移动到 GPU
            # print(dir_adj_edge_index)
            # max_value = torch.max(dir_adj_edge_index).item()
            # min_value = torch.min(dir_adj_edge_index).item()
            #
            # print("Maximum value:", max_value)
            # print("Minimum value:", min_value)
            #dir
            Xef = self.lst_gcn1[_layer](X, dir_adj_edge_index)
            XRrf = self.lst_gcn2[_layer](XR, dir_adjr_edge_index)
            #Xef = self.lst_gcn1[_layer](X, edge_index)
            #XRrf = self.lst_gcn2[_layer](XR, self.adj_r)



            Xrf = XRrf[lst_indexes1]
            if self.args.combine_type == "cat":
                size = Xrf.size(1) // 4
                Xef1, Xef2, Xef3, Xef4 = torch.split(Xef, size, dim=1)
                Xrf1, Xrf2, Xrf3, Xrf4 = torch.split(Xrf, size, dim=1)
                X = torch.cat([Xef1, Xrf1, Xef2, Xrf2, Xef3, Xrf3, Xef4, Xrf4], dim=1)
                hamilton = make_quaternion_mul(self.linear_ents[_layer])
                X = torch.mm(X, hamilton)
            elif self.args.combine_type == "sum":
                X = Xef + Xrf
            elif self.args.combine_type == "corr":
                X = Xef * Xrf
            elif self.args.combine_type == "linear_corr":
                hamilton = make_quaternion_mul(self.linear_ents_cor[_layer])
                X = Xef * torch.mm(Xrf, hamilton)

            R = XRrf[lst_indexes2[len(lst_indexes1):]]  # newR
            hs.append(X[e1_idx])  # finalX
            rs.append(R[r_idx])  # finalR
            ts.append(X[e2_idx])

            # print(ts[-1].shape)
            # x1= self.scorer(hs[-1], rs[-1],X,_layer + 2)
            x1 = rotate(hs[-1], rs[-1])#rotate
            x1 = self.quate_op(hs[-1], rs[-1],X, _layer + 2)
            cl_x = x1
            # print(cl_x.shape)
            scores.append(torch.sigmoid(self.scorer(hs[-1], rs[-1], X, _layer + 2)))
            #r_scores.append(torch.sigmoid(self.r_scorer(hs[-1], rs[-1], X, _layer + 2)))

        #return scores,r_scores,cl_x,ts[-1]
        return scores,cl_x, ts[-1]
    def quate_op(self, h, r, X, layer_index=1):
        hr = vec_vec_wise_multiplication(h, r)
        if layer_index == 1:
            hr = self.bn1(hr)
            hr = self.dropout1(hr)
        elif layer_index == 2:
            hr = self.bn2(hr)
            hr = self.dropout2(hr)
        else:
            hr = self.bn3(hr)
            hr = self.dropout3(hr)
        # hrt = torch.mm(hr, X.t())
        return hr
    def quate(self, h, r, X, layer_index=1):
        hr = vec_vec_wise_multiplication(h, r)
        if layer_index == 1:
            hr = self.bn1(hr)
            hr = self.dropout1(hr)
        elif layer_index == 2:
            hr = self.bn2(hr)
            hr = self.dropout2(hr)
        else:
            hr = self.bn3(hr)
            hr = self.dropout3(hr)
        hrt = torch.mm(hr, X.t())
        return hrt

    def distmult(self, h, r, X, layer_index=1):
        hr = h * r
        if layer_index == 1:
            hr = self.bn1(hr)
            hr = self.dropout1(hr)
        elif layer_index == 2:
            hr = self.bn2(hr)
            hr = self.dropout2(hr)
        else:
            hr = self.bn3(hr)
            hr = self.dropout3(hr)
        hrt = torch.mm(hr, X.t())
        return hrt

    def transe(self, h, r, X, layer_index=1):
        hr = h + r
        hrt = 20 - torch.norm(hr.unsqueeze(1) - X, p=1, dim=2)
        return hrt

    def normalize_embedding(self):
        embed = self.all_embeddings.weight.detach().cpu().numpy()[:self.n_entities]
        rel_emb = self.all_embeddings.weight.detach().cpu().numpy()[self.n_entities:]
        embed = embed / np.sqrt(np.sum(np.square(embed), axis=1, keepdims=True))

        self.all_embeddings.weight.data.copy_(torch.from_numpy(np.concatenate((embed, rel_emb), axis=0)))

    def get_hidden_feature(self):
        return self.feat_list

    def regularization(self, dis_loss, margin=1.5):
        return max(0, margin - dis_loss)

    def get_factor(self):
        factor_list = []
        factor_list.append(self.distangle.get_factor())
        return factor_list

    def compute_disentangle_loss(self):
        return self.distangle.compute_disentangle_loss()

    @staticmethod
    def merge_loss(dis_loss):
        return dis_loss



class SupConLoss(nn.Module):
    """Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.监督对比损失 也支持无监督对比simclr损失
    It also supports the unsupervised contrastive loss in SimCLR"""

    def __init__(self, temperature=0.07, contrast_mode='all',
                 base_temperature=0.07):
        super(SupConLoss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature

    def forward(self, features, labels=None, mask=None):  # 接收特征 标签 mask
        """Compute loss for model. If both `labels` and `mask` are None,
        it degenerates to SimCLR unsupervised loss:
        https://arxiv.org/pdf/2002.05709.pdf计算模型的损失 如果labels和mask都为空，那么用无监督对比simclr损失

        Args:参数
            features: hidden vector of shape [bsz, n_views, ...].特征：隐藏向量的形状为【bsz,n_views,...】
            labels: ground truth of shape [bsz].labels:真实值的形状【bsz】
            mask: contrastive mask of shape [bsz, bsz], mask_{i,j}=1 if sample   对比mask的形状【bsz,bsz】,如果i和j的类别相同，则mask_{i,j}=1
                has the same class as sample i. Can be asymmetric.
        Returns:返回
            A loss scalar.损失值
        """
        device = (torch.device('cuda')
                  if features.is_cuda
                  else torch.device('cpu'))
        # 检查维度
        if len(features.shape) < 3:
            raise ValueError('`features` needs to be [bsz, n_views, ...],'
                             'at least 3 dimensions are required')
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        batch_size = features.shape[0]
        if labels is not None and mask is not None:  # mask和label都不为空
            raise ValueError('Cannot define both `labels` and `mask`')
        elif labels is None and mask is None:  # mask和label都为空
            # SimCLR loss
            mask = torch.eye(batch_size).float().to(device)  # 创建一个对角线mask 为1则说明i和j有相同类别
        elif labels is not None:  # label不为空
            # Supconloss 监督对比损失
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            mask = torch.eq(labels, labels.T).float().to(device)  # 根据标签创建对比掩码
        else:
            mask = mask.float().to(device)  # 有mask

        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        # concat all contrast features at dim 0#拼接dim为0时的所有对比特征
        # 如果对比模式（contrast_mode）为"one"，则选择第一个对比特征作为锚定特征，锚定特征的数量为1
        if self.contrast_mode == 'one':
            anchor_feature = features[:, 0]
            anchor_count = 1  # 根据特征的形状，确定对比特征的数量（contrast_count）和锚定特征（anchor_feature）
            # 如果对比模式为"all"，则将所有对比特征拼接起来作为锚定特征，锚定特征的数量等于对比特征的数量
        elif self.contrast_mode == 'all':
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:  # 如果对比模式既不是"one"也不是"all"，则引发错误
            raise ValueError('Unknown mode: {}'.format(self.contrast_mode))

        # compute logits 计算特征之间的相似度得分（logits）
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),
            self.temperature)  # 首先，计算锚定特征与对比特征的点积，并除以温度参数，得到相似度得分。

        # for numerical stability 数值稳定性
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)  # 然后，通过减去logits的最大值（用于数值稳定性）得到logit矩阵。
        logits = anchor_dot_contrast - logits_max.detach()

        # tile mask
        mask = mask.repeat(anchor_count, contrast_count)  # 接下来，将对比掩码（mask）复制（tile）到与logit矩阵相同的形状。
        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0
        )
        # 最后，使用logits_mask将对角线上的元素（自己与自己的对比）置为0
        mask = mask * logits_mask

        # compute log_prob这部分代码计算对数概率（log_prob）并计算正样本的对数概率的均值

        # negative samples首先，计算logits的指数，并乘以logits_mask。
        # 然后，计算logits的对数概率，通过减去指数的和的对数。
        exp_logits = torch.exp(logits) * logits_mask

        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        # compute mean of log-likelihood over positive接下来，计算每个正样本的对数概率的均值。
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)  # 这行代码计算每个正样本的对数概率的均值。
        # 首先，将log_prob乘以mask，这样只有正样本对应的对数概率才会被保留。然后，对每个样本，将对数概率求和，并除以mask中对应的正样本数量，得到均值。

        # avoid nan loss when there's one sample for a certain class, e.g., 0,1,...1 for bin-cls , this produce nan for 1st in Batch
        # which also results in batch total loss as nan. such row should be dropped
        # 这部分代码是为了避免出现在某个类别只有一个样本的情况下产生nan损失的问题。
        # 首先，计算每个样本的正样本数量，并将其存储在pos_per_sample中。
        # 然后，将pos_per_sample中小于1e-6的值设置为1.0，以避免除以零的情况。最后，将对数概率乘以mask并除以pos_per_sample，
        # 得到每个正样本的对数概率的均值
        pos_per_sample = mask.sum(1)  # B
        pos_per_sample[pos_per_sample < 1e-6] = 1.0
        mean_log_prob_pos = (mask * log_prob).sum(1) / pos_per_sample  # mask.sum(1)

        # mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

        # loss
        loss = - (
                    self.temperature / self.base_temperature) * mean_log_prob_pos  # 这行代码计算损失值。首先，将温度参数除以基础温度参数，然后将其乘以mean_log_prob_pos，得到最终的损失值。
        loss = loss.view(anchor_count, batch_size).mean()  # 在最后一行，损失值进行了一些形状调整和求平均操作，然后返回作为函数的输出。

        return loss



class relation_contrast(torch.nn.Module):
    def __init__(self, temperature, num_neg):
        super(relation_contrast, self).__init__()
        self.temperature = temperature
        self.num_neg = num_neg
        # self.all_pos_triples = all_pos_triple

    def forward(self, pos_scores, neg_scores):
        neg_scores = neg_scores.view(-1, self.num_neg, 1)
        pos = torch.exp(torch.div(pos_scores, self.temperature))
        neg = torch.exp(torch.div(neg_scores, self.temperature)).sum(dim=1)
        loss = -torch.log(torch.div(pos, neg)).mean()
        return loss



class WGE_model(torch.nn.Module):
    def __init__(self, args, num_ents, num_rels, adj, adj_r, deg):
        super(WGE_model, self).__init__()
        self.args = args
        self.emb_dim = args.emb_dim
        self.emb_dim = args.emb_dim
        self.n_entities = num_ents
        self.n_relations = num_rels
        self.deg = deg

        self.thetas = nn.Parameter(torch.ones(3))

        self.all_embeddings = torch.nn.Embedding(self.n_entities + self.n_relations, self.emb_dim)
        # import pdb; pdb.set_trace()
        self.adj = adj
        self.adj_r = adj_r
        self.num_rels = num_rels


        self.lst_gcn1 = torch.nn.ModuleList()
        self.lst_gcn2 = torch.nn.ModuleList()
        for _layer in range(args.num_layers):
            if self.args.encoder == "qgnn":
                self.lst_gcn1.append(Q4GNNLayer(self.emb_dim, self.emb_dim, act=torch.tanh))
                self.lst_gcn2.append(Q4GNNLayer(self.emb_dim, self.emb_dim, act=torch.tanh))
            else:
                print("This encoder has not been implemented... Existing")
                exit()

        self.linear_ents = torch.nn.ParameterList()
        self.linear_ents_cor = torch.nn.ParameterList()

        for _layer in range(args.num_layers):
            self.linear_ents.append(Parameter(torch.FloatTensor(self.emb_dim // 2, self.emb_dim)))
            self.linear_ents_cor.append(Parameter(torch.FloatTensor(self.emb_dim // 4, self.emb_dim)))

        # if not self.args.best_model:
        self.reset_parameters()

        xavier_normal_(self.all_embeddings.weight.data)
        # import pdb; pdb.set_trace()

        self.activate = torch.nn.Tanh()

        self.bn1 = nn.BatchNorm1d(self.emb_dim)
        self.bn2 = nn.BatchNorm1d(self.emb_dim)
        self.bn3 = nn.BatchNorm1d(self.emb_dim)
        self.dropout1 = torch.nn.Dropout()
        self.dropout2 = torch.nn.Dropout()
        self.dropout3 = torch.nn.Dropout()
        self.loss = torch.nn.BCELoss()#计算二分类交叉熵损失

    def reset_parameters(self):
        for i in range(len(self.linear_ents)):
            weight = self.linear_ents[i]
            stdv = math.sqrt(6.0 / (weight.size(0) + weight.size(1)))
            self.linear_ents[i].data.uniform_(-stdv, stdv)

        for i in range(len(self.linear_ents_cor)):
            weight = self.linear_ents_cor[i]
            stdv = math.sqrt(6.0 / (weight.size(0) + weight.size(1)))
            self.linear_ents_cor[i].data.uniform_(-stdv, stdv)

    def forward_normal(self, e1_idx, r_idx, lst_indexes1, lst_indexes2, train=False):
        self.scorer = self.quate

        X = self.all_embeddings(lst_indexes1)
        R = self.all_embeddings(lst_indexes2[len(lst_indexes1):])
        h1 = X[e1_idx]
        r1 = R[r_idx]
        hs = [h1]
        rs = [r1]

        scores = [torch.sigmoid(self.scorer(hs[-1], rs[-1], X, 1))]

        for _layer in range(self.args.num_layers):
            XR = torch.cat((X, R), dim=0)  # last
            XRrf = self.lst_gcn2[_layer](XR, self.adj_r)  # newX, newR from relational graph
            Xef = self.lst_gcn1[_layer](X, self.adj)  # newX2 from original graph
            Xrf = XRrf[lst_indexes1]
            if self.args.combine_type == "cat":
                size = Xrf.size(1) // 4
                Xef1, Xef2, Xef3, Xef4 = torch.split(Xef, size, dim=1)
                Xrf1, Xrf2, Xrf3, Xrf4 = torch.split(Xrf, size, dim=1)
                X = torch.cat([Xef1, Xrf1, Xef2, Xrf2, Xef3, Xrf3, Xef4, Xrf4], dim=1)
                hamilton = make_quaternion_mul(self.linear_ents[_layer])
                X = torch.mm(X, hamilton)
            elif self.args.combine_type == "sum":
                X = Xef + Xrf
            elif self.args.combine_type == "corr":
                X = Xef * Xrf
            elif self.args.combine_type == "linear_corr":
                hamilton = make_quaternion_mul(self.linear_ents_cor[_layer])
                X = Xef * torch.mm(Xrf, hamilton)

            R = XRrf[lst_indexes2[len(lst_indexes1):]]  # newR
            hs.append(X[e1_idx])  # finalX
            rs.append(R[r_idx])  # finalR
            scores.append(torch.sigmoid(self.scorer(hs[-1], rs[-1], X, _layer + 2)))

        return scores

    def quate(self, h, r, X, layer_index=1):
        hr = vec_vec_wise_multiplication(h, r)
        if layer_index == 1:
            hr = self.bn1(hr)
            hr = self.dropout1(hr)
        elif layer_index == 2:
            hr = self.bn2(hr)
            hr = self.dropout2(hr)
        else:
            hr = self.bn3(hr)
            hr = self.dropout3(hr)
        hrt = torch.mm(hr, X.t())
        return hrt

    def distmult(self, h, r, X, layer_index=1):
        hr = h * r
        if layer_index == 1:
            hr = self.bn1(hr)
            hr = self.dropout1(hr)
        elif layer_index == 2:
            hr = self.bn2(hr)
            hr = self.dropout2(hr)
        else:
            hr = self.bn3(hr)
            hr = self.dropout3(hr)
        hrt = torch.mm(hr, X.t())
        return hrt

    def transe(self, h, r, X, layer_index=1):
        hr = h + r
        hrt = 20 - torch.norm(hr.unsqueeze(1) - X, p=1, dim=2)
        return hrt

    def normalize_embedding(self):
        embed = self.all_embeddings.weight.detach().cpu().numpy()[:self.n_entities]
        rel_emb = self.all_embeddings.weight.detach().cpu().numpy()[self.n_entities:]
        embed = embed / np.sqrt(np.sum(np.square(embed), axis=1, keepdims=True))

        self.all_embeddings.weight.data.copy_(torch.from_numpy(np.concatenate((embed, rel_emb), axis=0)))

    def get_hidden_feature(self):
        return self.feat_list

    def regularization(self, dis_loss, margin=1.5):
        return max(0, margin - dis_loss)

    def get_factor(self):
        factor_list = []
        factor_list.append(self.distangle.get_factor())
        return factor_list

    def compute_disentangle_loss(self):
        return self.distangle.compute_disentangle_loss()

    @staticmethod
    def merge_loss(dis_loss):
        return dis_loss


""" Simple GCN layer, similar to https://arxiv.org/abs/1609.02907 """
class GraphConvolution(torch.nn.Module):
    def __init__(self, in_features, out_features, act=torch.tanh,  bias=False):
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

    # def forward_vew2_relation(self, e1_idx, r_idx, lst_indexes1, lst_indexes2, train=False):
    #     self.scorer = self.quate
    #     X = self.all_embeddings(lst_indexes1)
    #     R = self.all_embeddings(lst_indexes2[len(lst_indexes1):])

    #     h1 = X[e1_idx]
    #     r1 = R[r_idx]
    #     hs = [h1]
    #     rs = [r1]

    #     if self.args.use_multiple_layers:
    #         scores = [torch.sigmoid(self.scorer(hs[-1], rs[-1], X, 1))]
    #     else:
    #         scores = []

    #     for _layer in range(self.args.num_layers):
    #         R = self.lst_gcn2[_layer](R, self.adj_r) # newX, newR from relational graph
    #         X = self.lst_gcn1[_layer](X, self.adj) # newX2 from original graph 
    #         hs.append(X[e1_idx]) # finalX
    #         rs.append(R[r_idx]) # finalR
    #         scores.append(torch.sigmoid(self.scorer(hs[-1], rs[-1], X, _layer + 2)))
        
    #     return scores

    # def forward_entity_graph(self, e1_idx, r_idx, lst_indexes1, lst_indexes2, train=False):
    #     self.scorer = self.quate
    #     X = self.all_embeddings(lst_indexes1)
    #     R = self.all_embeddings(lst_indexes2[len(lst_indexes1):])

    #     h1 = X[e1_idx]
    #     r1 = R[r_idx]
    #     hs = [h1]
    #     rs = [r1]

    #     if self.args.use_multiple_layers:
    #         scores = [torch.sigmoid(self.scorer(hs[-1], rs[-1], X, 1))]
    #     else:
    #         scores = []
    #     for _layer in range(self.args.num_layers):
    #         X = self.lst_gcn1[_layer](X, self.adj)
    #         r = r1 
    #         hs.append(X[e1_idx])
    #         rs.append(r)
    #         scores.append(torch.sigmoid(self.scorer(hs[-1], rs[-1], X, _layer + 2)))

    #     return scores