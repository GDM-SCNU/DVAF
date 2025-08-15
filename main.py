# -*- coding: UTF-8 -*-
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


class WGE:
    """ Two-view Graph Neural Networks for Knowledge Graph Completion """
    def __init__(self):
        self.args = args


    """ Functions are adapted from https://github.com/ibalazevic/TuckER for using 1-N scoring strategy """
    def get_data_idxs(self, data):
        data_idxs = [(self.entity_idxs[data[i][0]], self.relation_idxs[data[i][1]], self.entity_idxs[data[i][2]]) for i in range(len(data))]
        return data_idxs

    def evaluate(self, model, data, lst_indexes1, lst_indexes2, er_vocab, test=False, save=False, ep=0):
        model.eval()
        rel_dict = defaultdict(list)
        with torch.no_grad():
            hits = []
            ranks = []
            for i in range(10):
                hits.append([])

            test_data_idxs = self.get_data_idxs(data)
            print("Number of data points: %d" % len(test_data_idxs))
            # import pdb; pdb.set_trace()


            for i in range(0, len(test_data_idxs), 1000):
                data_batch, _ = get_batch(er_vocab, test_data_idxs, i, 1000, d)
                e1_idx = torch.tensor(data_batch[:, 0]).to(device)
                r_idx = torch.tensor(data_batch[:, 1]).to(device)
                e2_idx = torch.tensor(data_batch[:, 2]).to(device)

                forward = model.forward_normal
                
                preds = forward(e1_idx, r_idx, lst_indexes1, lst_indexes2, train=False)

                num_preds = len(preds)
                weights = [args.first_layer_weight, 0, 0]
                for _index in range(num_preds - 1):
                    weights[_index + 1] = (1 - weights[0]) / (num_preds - 1)
                
                pred = 0
                for _layer in range(num_preds):
                    pred += preds[_layer] * weights[_layer]
                    
                pred = pred.detach()

                for j in range(data_batch.shape[0]):
                    this_e1 = data_batch[j][0]
                    this_r = data_batch[j][1]
                    filt = er_vocab[(this_e1, this_r)]
                    target_value = pred[j, e2_idx[j]].item()
                    pred[j, filt] = 0.0
                    pred[j, e2_idx[j]] = target_value

                sort_values, sort_idxs = torch.sort(pred, dim=1, descending=True)
                sort_idxs = sort_idxs.cpu().numpy()

                for j in range(data_batch.shape[0]):
                    this_r = data_batch[j][1]
                    rank = np.where(sort_idxs[j] == e2_idx[j].item())[0][0]
                    ranks.append(rank + 1)
                    rel_dict[self.relation_idx2id[this_r]].append(rank)
                    for hits_level in range(10):
                        if rank <= hits_level:
                            hits[hits_level].append(1.0)
                        else:
                            hits[hits_level].append(0.0)


        hit1 = np.mean(hits[0])*100
        hit3 = np.mean(hits[2])*100
        hit5 = np.mean(hits[4])*100
        hit10 = np.mean(hits[9])*100
        MR = np.mean(ranks)
        MRR = np.mean(1. / np.array(ranks))

        print('Hits @1: {0}'.format(hit1))
        print('Hits @3: {0}'.format(hit3))
        print('Hits @5: {0}'.format(hit5))
        print('Hits @10: {0}'.format(hit10))
        print('MR: {0}'.format(MR))
        print('Mean reciprocal rank: {0}'.format(MRR))

        return hit1, hit3, hit10, MR, MRR


    def prepare_data(self):
        self.entity_idxs = {d.entities[i]: i for i in range(len(d.entities))} # entity ids to index
        self.entity_idx2id = {v:k for k,v in self.entity_idxs.items()}
        self.relation_idxs = {d.relations[i]: i for i in range(len(d.relations))} # relation ids to index
        self.relation_idx2id = {v:k for k,v in self.relation_idxs.items()}
        # import pdb; pdb.set_trace()

        adj_r = construct_relation_focus_matrix(d.train_data, self.entity_idxs, self.relation_idxs, args.beta)
        adj = construct_entity_focus_matrix(d.train_data, self.entity_idxs)

        train_data_idxs = self.get_data_idxs(d.train_data)
        deg = get_deg(train_data_idxs, len(self.entity_idxs)).to(device)

        print("Number of training data points: %d" % len(train_data_idxs))

        model = WGE_model(args, len(self.entity_idxs), len(self.relation_idxs), adj, adj_r, deg=deg).to(device)

        print("Using Adam optimizer")
        opt = torch.optim.Adam(model.parameters(), lr=args.lr)

        lst_indexes1 = torch.LongTensor([i for i in range(len(d.entities))]).to(device)
        lst_indexes2 = torch.LongTensor([i for i in range(len(d.entities) + len(d.relations))]).to(device)
        er_vocab = get_er_vocab(train_data_idxs)
        t_vocab=get_t_vocab(train_data_idxs)
        er_vocab_all = get_er_vocab(self.get_data_idxs(d.data))
        er_vocab_pairs = list(er_vocab.keys())

        return model, opt, lst_indexes1, lst_indexes2, er_vocab, er_vocab_all, er_vocab_pairs,t_vocab


    def train_and_eval(self, model, opt, lst_indexes1, lst_indexes2, er_vocab, er_vocab_all, er_vocab_pairs):
        
        max_valid_mrr, best_hit10_ever, best_mrr_ever, best_epoch = 0.0, 0.0, 0.0, 0
        best_valid_hit10_ever, best_valid_mrr_ever = 0.0, 0.0
        forward = model.forward_normal

        print("Starting training...")
        for it in tqdm(range(1, args.num_iterations + 1)):
            model.train()            
            losses, losses0, losses1, losses2 = [], [], [], []
            np.random.shuffle(er_vocab_pairs)
            for j in range(0, len(er_vocab_pairs), args.batch_size):
                data_batch, targets = get_batch(er_vocab, er_vocab_pairs, j, args.batch_size, d)
                opt.zero_grad()
                e1_idx = torch.tensor(data_batch[:, 0]).to(device)
                r_idx = torch.tensor(data_batch[:, 1]).to(device)
                e2_idx=torch.tensor(data_batch[:, 2]).to(device)
                preds = forward(e1_idx, r_idx, lst_indexes1, lst_indexes2, train=True)
                targets = ((1.0 - args.label_smoothing) * targets) + (1.0 / targets.size(1))

                loss_list = [0, 0, 0]
                loss = 0

                num_preds = len(preds)
                weights = [args.first_layer_weight, 0, 0]
                for _index in range(num_preds - 1):
                    weights[_index + 1] = (1 - weights[0]) / (num_preds - 1)
                
                for _layer in range(num_preds):
                    this_loss = model.loss(preds[_layer], targets)
                    loss_list[_layer] = this_loss.item()
                    loss += this_loss * weights[_layer]
                
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)  # prevent the exploding gradient problem
                opt.step()
                losses.append(loss.item())
                losses0.append(loss_list[0])
                losses1.append(loss_list[1])
                losses2.append(loss_list[2])
            
            print("Epoch: {} --> Loss: {:.4f}, Loss0: {:.4f}, Loss1: {:.4f}, Loss2: {:.4f}".format(it, np.sum(losses), np.sum(losses0), np.sum(losses1), np.sum(losses2)))

            if (it > args.eval_after and it % args.eval_step == 0) or (it == 1):
                print("Valid:")
                hit1, hit3, hit10, MR, tmp_mrr = self.evaluate(model, d.valid_data, lst_indexes1, lst_indexes2, er_vocab_all, test=False, save=True, ep=it)
                if max_valid_mrr < tmp_mrr:
                    max_valid_mrr = tmp_mrr
                    best_epoch = it
                if best_valid_hit10_ever < hit10:
                    best_valid_hit10_ever = hit10 
                
                print("Test:")
                t_hit1, t_hit3, t_hit10, t_MR, t_MRR = self.evaluate(model, d.test_data, lst_indexes1, lst_indexes2, er_vocab_all, test=True, save=True, ep=it)
                if best_hit10_ever < t_hit10:
                    best_hit10_ever = t_hit10
                if best_mrr_ever < t_MRR:
                    best_mrr_ever = t_MRR
                        
                print("Best valid epoch", best_epoch, " --> Final test results: {:.4f}, {:.4f}, {:.4f}, {:.4f}, {:.4f}".format(t_hit1, t_hit3, t_hit10, t_MR, t_MRR))
                #print("x" * 30, "Best test h10 ever: {:.4f}, best test mrr: {:.4f}".format(best_hit10_ever, best_mrr_ever), "x" * 30) 
                #print("x" * 30, "Best valid h10 ever: {:.4f}, best valid mrr: {:.4f}".format(best_valid_hit10_ever, max_valid_mrr), "x" * 30) 


class W:
    """ Two-view Graph Neural Networks for Knowledge Graph Completion """

    def __init__(self):
        self.args = args
        self.supconloss=self.supconloss = SupConLoss(temperature=0.07, contrast_mode="all", base_temperature=0.07).to(torch.device("cuda"))
    """ Functions are adapted from https://github.com/ibalazevic/TuckER for using 1-N scoring strategy """

    def get_data_idxs(self, data):
        data_idxs = [(self.entity_idxs[data[i][0]], self.relation_idxs[data[i][1]], self.entity_idxs[data[i][2]]) for i
                     in range(len(data))]

        #print(self.entity_idxs)
        with open('/output/HhsMath/e.txt', 'w') as file:
            for entity, index in self.entity_idxs.items():  # Assuming entity_idxs is a dictionary
                file.write(f"{entity}\t{index}\n")
        with open('/output/HhsMath/r.txt', 'w') as file:
            for relation, index in self.relation_idxs.items():  # Assuming relation_idxs is a dictionary
                file.write(f"{relation}\t{index}\n")


        return data_idxs

    def evaluate(self, model, data, lst_indexes1, lst_indexes2, er_vocab, test=False, save=False, ep=0):
        model.eval()
        rel_dict = defaultdict(list)
        with torch.no_grad():
            hits = []
            ranks = []
            for i in range(10):
                hits.append([])
            best_mrr = 0  # Track the best MRR
            best_output_format = {}  # To save output format corresponding to the best MRR
            tail_predictions = defaultdict(list)  # To store predictions for each relation
            tail_entity_scores = defaultdict(list)  # To store scores for tail entities

            test_data_idxs = self.get_data_idxs(data)
            print(test_data_idxs)
            print("Number of data points: %d" % len(test_data_idxs))
            # import pdb; pdb.set_trace()

            for i in range(0, len(test_data_idxs), 1000):
                data_batch, _ = get_batch(er_vocab, test_data_idxs, i, 1000, d)
                e1_idx = torch.tensor(data_batch[:, 0]).to(device)
                r_idx = torch.tensor(data_batch[:, 1]).to(device)
                e2_idx = torch.tensor(data_batch[:, 2]).to(device)

                forward = model.forward_normal

                preds,cl_x,t= forward(e1_idx,e2_idx,r_idx, lst_indexes1, lst_indexes2, train=False)

                num_preds = len(preds)
                weights = [args.first_layer_weight, 0, 0]
                for _index in range(num_preds - 1):
                    weights[_index + 1] = (1 - weights[0]) / (num_preds - 1)

                pred = 0
                for _layer in range(num_preds):
                    # weights[_layer] = float(weights[_layer])
                    # print(preds[_layer])
                    # print("&&&&&&&&&&&&&&&")
                    # print( weights[_layer])
                    pred += preds[_layer] * weights[_layer]

                pred = pred.detach()

                for j in range(data_batch.shape[0]):
                    this_e1 = data_batch[j][0]
                    this_r = data_batch[j][1]
                    filt = er_vocab[(this_e1, this_r)]
                    target_value = pred[j, e2_idx[j]].item()
                    pred[j, filt] = 0.0
                    pred[j, e2_idx[j]] = target_value

                sort_values, sort_idxs = torch.sort(pred, dim=1, descending=True)
                sort_idxs = sort_idxs.cpu().numpy()

                for j in range(data_batch.shape[0]):
                    this_e1 = data_batch[j][0]
                    this_r = data_batch[j][1]
                    rank = np.where(sort_idxs[j] == e2_idx[j].item())[0][0]
                    ranks.append(rank + 1)
                    rel_dict[self.relation_idx2id[this_r]].append(rank)
                    for hits_level in range(10):
                        if rank <= hits_level:
                            hits[hits_level].append(1.0)
                        else:
                            hits[hits_level].append(0.0)
                    for tail_idx in sort_idxs[j]:
                        score = pred[j, tail_idx].item()
                        tail_entity_scores[(this_e1, this_r)].append((tail_idx, score))
        print(len(tail_entity_scores))
        # Create the output format
        output_format = {}
        for (e1, r), scores in tail_entity_scores.items():
            # Sort by score in descending order
            sorted_scores = sorted(scores, key=lambda x: x[1], reverse=True)[:30]
            # Prepare the head entity and relation as a string key
            key = f"{e1}\t{r}"
            output_format[key] = [tail_idx for tail_idx, score in sorted_scores]

        hit1 = np.mean(hits[0]) * 100
        hit3 = np.mean(hits[2]) * 100
        hit5 = np.mean(hits[4]) * 100
        hit10 = np.mean(hits[9]) * 100
        MR = np.mean(ranks)
        MRR = np.mean(1. / np.array(ranks))

        print('Hits @1: {0}'.format(hit1))
        print('Hits @3: {0}'.format(hit3))
        print('Hits @5: {0}'.format(hit5))
        print('Hits @10: {0}'.format(hit10))
        print('MR: {0}'.format(MR))
        print('Mean reciprocal rank: {0}'.format(MRR))
        # Check if the current MRR is the best
        if MRR > best_mrr:
            best_mrr = MRR
            best_output_format = output_format
        import json
        # Function to convert non-serializable data types
        def convert_to_serializable(obj):
            if isinstance(obj, np.int64):
                return int(obj)
            if isinstance(obj, list):
                return [convert_to_serializable(i) for i in obj]
            if isinstance(obj, dict):
                return {convert_to_serializable(k): convert_to_serializable(v) for k, v in obj.items()}
            return obj  # Return the object as-is if it's already serializable


        # Save the output format with the highest MRR to a file
        if save and best_output_format:
            serializable_output = convert_to_serializable(best_output_format)
            print(serializable_output)
            with open('2.txt', 'w') as f:
                json.dump(serializable_output, f, indent=0) # Use json.dump for proper JSON formatting

        return hit1, hit3, hit10, MR, MRR

    def prepare_data(self):
        self.entity_idxs = {d.entities[i]: i for i in range(len(d.entities))}  # entity ids to index
        self.entity_idx2id = {v: k for k, v in self.entity_idxs.items()}
        self.relation_idxs = {d.relations[i]: i for i in range(len(d.relations))}  # relation ids to index
        self.relation_idx2id = {v: k for k, v in self.relation_idxs.items()}

        #print("*****************************************************************************")
        #print(self.relation_idxs)
        #print(self.relation_idx2id)
       # print("*****************************************************************************")

        # import pdb; pdb.set_trace()
        #print(self.entity_idxs)
        #print(self.entity_idx2id)
        #print("*****************************************************************************")


        adj_r,adjr_edge_index = construct_relation_focus_matrix(d.train_data, self.entity_idxs, self.relation_idxs, args.beta)
        adj,adj_edge_index = construct_entity_focus_matrix(d.train_data, self.entity_idxs)
        # dir_adj=construct1_entity_focus_matrix(d.train_data, self.entity_idxs)
        # 获取有向的edge_index


        # adj_r = construct_relation_focus_matrix(d.train_data, self.entity_idxs, self.relation_idxs, args.beta)
        # adj = construct_entity_focus_matrix(d.train_data, self.entity_idxs)
        dir_adj,dir_adj_edge_index = dir_construct_entity_focus_matrix(d.train_data, self.entity_idxs)#实体邻接矩阵 不对称的
        # print("实体邻接矩阵：",dir_adj)
        dir_adjr,dir_adjr_edge_index = construct_dir_relation_focus_matrix(d.train_data, self.entity_idxs, self.relation_idxs, args.beta)


        print("adj_r",adj_r.shape)
        print("adj", adj.shape)
        print("dir_adj", dir_adj.shape)
        print("dir_adjr",dir_adjr.shape)

        # 先对稀疏张量进行压缩
        # # dir_adj = dir_adj.coalesce()
        # # dir_adjr = dir_adjr.coalesce()
        # dir_adj = adj.coalesce()
        # dir_adjr = adj_r.coalesce()
        # # 获取压缩后的稀疏张量的行索引和列索引
        # dir_adj_indices = dir_adj.indices()
        # self.dir_adj_edge_index = dir_adj_indices
        # dir_adjr_indices = dir_adjr.indices()
        # self.dir_adjr_edge_index = dir_adjr_indices
        # dir_adj_row_indices = self.dir_adj_edge_index[0]  # 获取压缩稀疏张量的行索引
        # dir_adj_col_indices = self.dir_adj_edge_index[1]  # 获取压缩稀疏张量的列索引
        # dir_adjr_row_indices = self.dir_adjr_edge_index[0]  # 获取压缩稀疏张量的行索引
        # dir_adjr_col_indices = self.dir_adjr_edge_index[1]
        # # missing_keys = set(dir_adjr_row_indices) - set(self.entity_idx2id.keys())
        # # print("Missing keys:", missing_keys)
        # # dir_adj_original_row_ids = [self.entity_idx2id[row_idx.item()] for row_idx in dir_adj_row_indices]
        # # dir_adj_original_col_ids = [self.entity_idx2id[col_idx.item()] for col_idx in dir_adj_col_indices]
        # # dir_adjr_original_row_ids = [self.entity_idx2id[row_idx.item()] for row_idx in dir_adjr_row_indices]
        # # dir_adjr_original_col_ids = [self.entity_idx2id[col_idx.item()] for col_idx in dir_adjr_col_indices]
        #
        # dir_adj_edge_index = torch.stack(
        #     [torch.tensor(dir_adj_row_indices), torch.tensor(dir_adj_col_indices)])  # 实体的有向edge_index
        # dir_adjr_edge_index = torch.stack(
        #     [torch.tensor(dir_adjr_row_indices), torch.tensor(dir_adjr_col_indices)])  # 实体+关系的有向edge_index
        # import torch
        # from scipy.sparse import csr_matrix

        # 将稀疏矩阵转换为CSR格式
        # dir_adj_csr_edge_mat = csr_matrix(dir_adj)
        #
        # # 提取非零元素的行索引和列索引
        # dir_adj_row_ind, dir_adj_col_ind = dir_adj_csr_edge_mat.nonzero()
        #
        # # 创建边索引
        # dir_adj_edge_index = torch.tensor([dir_adj_row_ind, dir_adj_col_ind], dtype=torch.long)
        print("无向图：")
        print("adj_edge_index shape:", adj_edge_index.shape)
        print("adj_edge_index values:", adj_edge_index)
        print("adjr_edge_index shape:", adjr_edge_index.shape)
        print("adjr_edge_index values:", adjr_edge_index)



        # # 打印边索引的形状和数值
        print("有向图：")
        print("dir_adj_edge_index shape:", dir_adj_edge_index.shape)
        print("dir_adj_edge_index values:", dir_adj_edge_index)

        # # 将稀疏矩阵转换为CSR格式
        # dir_adjr_csr_edge_mat = csr_matrix(dir_adjr)
        #
        # # 提取非零元素的行索引和列索引
        # dir_adjr_row_ind, dir_adjr_col_ind = dir_adjr_csr_edge_mat.nonzero()
        #
        # # 创建边索引
        # dir_adjr_edge_index = torch.tensor([dir_adjr_row_ind, dir_adjr_col_ind], dtype=torch.long)
        #
        # 打印边索引的形状和数值
        print("dir_adjr_edge_index shape:", dir_adjr_edge_index.shape)
        print("dir_adjr_edge_index values:", dir_adjr_edge_index)
        # import numpy as np
        # from scipy.sparse import find
        #
        # def sparse_matrix_to_edge_index(sparse_matrix):
        #     # 使用 scipy 的 find 函数获取稀疏矩阵中非零元素的索引
        #     row_ind, col_ind, _ = find(sparse_matrix)
        #
        #     # 将索引转换为 edge_index 格式
        #     edge_index = np.vstack((row_ind, col_ind))
        #
        #     return edge_index
        #
        # dir_adjr_edge_index = sparse_matrix_to_edge_index(dir_adjr)
        # dir_adj_edge_index = sparse_matrix_to_edge_index(dir_adj)




        train_data_idxs = self.get_data_idxs(d.train_data)
        deg = get_deg(train_data_idxs, len(self.entity_idxs)).to(device)

        print("Number of training data points: %d" % len(train_data_idxs))

        model = W_model(args, len(self.entity_idxs), len(self.relation_idxs), adj, adj_r, deg=deg, dir_adj_edge_index=dir_adj_edge_index,dir_adjr_edge_index=dir_adjr_edge_index).to(device)

        print("Using Adam optimizer")
        opt = torch.optim.Adam(model.parameters(), lr=args.lr)

        lst_indexes1 = torch.LongTensor([i for i in range(len(d.entities))]).to(device)
        lst_indexes2 = torch.LongTensor([i for i in range(len(d.entities) + len(d.relations))]).to(device)
        er_vocab = get_er_vocab(train_data_idxs)
        t_vocab=get_t_vocab(train_data_idxs)
        ert_vocab = get_ert_vocab(train_data_idxs)
        er_vocab_all = get_er_vocab(self.get_data_idxs(d.data))
        t_vocab_all=get_t_vocab(self.get_data_idxs(d.data))
        ert_vocab_all=get_ert_vocab(self.get_data_idxs(d.data))
        er_vocab_pairs = list(er_vocab.keys())
        t_vocab_pairs=list(t_vocab.keys())
        ert_vocab_pairs = list(ert_vocab.keys())
        # print("er_vocab:",er_vocab)
        # print("t_vaocab：",er_vocab)
        # print("*******************************")
        # # print("er_vocab_all:",er_vocab_all)
        # print("er_vaocab：", er_vocab_all)
        # print("*******************************")
        # # print("er_vocab_pairs：",er_vocab_pairs)
        # print("t_vocab_pairs：", er_vocab_pairs)


        return model, opt, lst_indexes1, lst_indexes2, er_vocab, er_vocab_all, er_vocab_pairs, t_vocab, t_vocab_all, t_vocab_pairs,ert_vocab, ert_vocab_all, ert_vocab_pairs,dir_adj_edge_index,dir_adjr_edge_index


    def train_and_eval(self, model, opt, lst_indexes1, lst_indexes2, er_vocab, er_vocab_all, er_vocab_pairs,ert_vocab, ert_vocab_all, ert_vocab_pairs,dir_adj_edge_index,dir_adjr_edge_index):
        # train_dataset = KG_Triples(name="train", num_relations=args.rel_num, num_ent=args.ent_num,
        #                            train_path=args.train_path, noise_path=args.noise_path)
        # valid_dataset = KG_Triples(name="valid", num_relations=args.rel_num, num_ent=args.ent_num,
        #                            test_path=args.test_path)
        # test_dataset = KG_Triples(name="test", num_relations=args.rel_num, num_ent=args.ent_num,
        #                           test_path=args.test_path)
        # # 创建数据加载器
        # # cl_train_dataloader: 创建训练数据加载器 DataLoader，用于加载训练数据集 train_dataset。设置批量大小为 args.cl_batch_size，开启数据洗牌，使用 args.num_worker 个工作线程进行数据加载。
        # cl_train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
        #                                  num_workers=args.num_worker)
        # # valid_dataloader: 创建验证数据加载器 DataLoader，用于加载验证数据集 valid_dataset。设置批量大小为 1，不进行数据洗牌
        # valid_dataloader = DataLoader(valid_dataset, batch_size=1)
        # # test_dataloader: 创建测试数据加载器 DataLoader，用于加载测试数据集 test_dataset。设置批量大小为 1，不进行数据洗牌。
        # test_dataloader = DataLoader(test_dataset, batch_size=1)



        max_valid_mrr, best_hit10_ever, best_mrr_ever, best_epoch = 0.0, 0.0, 0.0, 0
        best_hit1_ever,best_hit3_ever,best_hit5_ever,best_mr_ever=0.0, 0.0, 0.0,1000.0
        best_valid_hit10_ever, best_valid_mrr_ever = 0.0, 0.0
        forward = model.forward_normal


        print("Starting training...")
        for it in tqdm(range(1, args.num_iterations + 1)):
            model.train()
            losses, losses0, losses1, losses2 = [], [], [], []
            #np.random.shuffle(er_vocab_pairs)
            np.random.shuffle(ert_vocab_pairs)
            for j in range(0, len(ert_vocab_pairs), args.batch_size):
                data_batch, targets = get_batch(ert_vocab, ert_vocab_pairs, j, args.batch_size, d)
                # t_batch,targets1=get_batch(t_vocab,t_vocab_pairs,j,args.batch_size, d)
                opt.zero_grad()
                device = torch.device('cuda:0')  # 指定 GPU 设备
                e1_idx = torch.tensor(data_batch[:, 0]).to(device)
                r_idx = torch.tensor(data_batch[:, 1]).to(device)
                e2_idx=torch.tensor(data_batch[:, 2]).to(device)

                model.to(device)  # 将模型移动到 GPU
                dir_adj_edge_index = dir_adj_edge_index.to(device)  # 将张量移动到 GPU
                dir_adjr_edge_index = dir_adjr_edge_index.to(device)  # 将张量移动到 GPU
                preds,x1_node,tail_emb = forward(e1_idx, e2_idx,r_idx, lst_indexes1, lst_indexes2, train=True)
                targets = ((1.0 - args.label_smoothing) * targets) + (1.0 / targets.size(1))

                loss_list = [0, 0, 0]
                loss = 0

                tail_emb1 = F.normalize(tail_emb, dim=1)  # 沿维度 1 标准化尾部嵌入。
                x1_node = F.normalize(x1_node, dim=1)  # 沿维度 1 标准化节点嵌入


                # calculate SupCon loss
                features1 = torch.cat((x1_node.unsqueeze(1), tail_emb1.unsqueeze(1)), dim=1)  # 连接标准化节点和尾部嵌入以创建特征张量。
                # features2 = torch.cat((x2_node.unsqueeze(1), tail_emb1.unsqueeze(1)), dim=1)

                # SupCon Loss

                supconloss1 = self.supconloss(features1, labels=e2_idx, mask=None)

                num_preds = len(preds)
                weights = [args.first_layer_weight, 0, 0]
                for _index in  range(num_preds - 1):
                    weights[_index + 1] = (1 - weights[0]) / (num_preds - 1)

                for _layer in range(num_preds):
                    #this_loss= model.loss(preds[_layer], targets)+0.00001*supconloss1
                    this_loss = model.loss(preds[_layer], targets) + 0.00001* supconloss1
                    # print(this_loss1)
                    # this_loss = model.loss(preds[_layer], targets)
                    # print(this_loss)
                    loss_list[_layer] = this_loss.item()
                    loss += this_loss * weights[_layer]

                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)  # prevent the exploding gradient problem
                opt.step()
                losses.append(loss.item())
                losses0.append(loss_list[0])
                losses1.append(loss_list[1])
                losses2.append(loss_list[2])

            print("Epoch: {} --> Loss: {:.4f}, Loss0: {:.4f}, Loss1: {:.4f}, Loss2: {:.4f}".format(it, np.sum(losses),
                                                                                                   np.sum(losses0),
                                                                                                   np.sum(losses1),
                                                                                                   np.sum(losses2)))



            if (it > args.eval_after and it % args.eval_step == 0) or (it == 1):
                print("Valid:")
                hit1, hit3, hit10, MR, tmp_mrr = self.evaluate(model, d.valid_data, lst_indexes1, lst_indexes2,
                                                               er_vocab_all, test=False, save=True, ep=it)
                if max_valid_mrr < tmp_mrr:
                    max_valid_mrr = tmp_mrr
                    best_epoch = it
                if best_valid_hit10_ever < hit10:
                    best_valid_hit10_ever = hit10

                print("Test:")
                t_hit1, t_hit3, t_hit10, t_MR, t_MRR = self.evaluate(model, d.test_data, lst_indexes1, lst_indexes2,
                                                                     er_vocab_all, test=True, save=True, ep=it)
                if best_hit1_ever < t_hit1:
                    best_hit1_ever = t_hit1
                if best_hit3_ever < t_hit3:
                    best_hit3_ever = t_hit3
                if best_hit10_ever < t_hit10:
                    best_hit10_ever = t_hit10
                if best_mr_ever > t_MR:
                    best_mr_ever = t_MR
                if best_mrr_ever < t_MRR:
                    best_mrr_ever = t_MRR




                print("Best valid epoch", best_epoch,
                      " --> test epoch",it,"--> Final test results: {:.4f}, {:.4f}, {:.4f}, {:.4f}, {:.4f}".format(t_hit1, t_hit3, t_hit10,
                                                                                               t_MR, t_MRR))
                with open("result/HhsMath/1/wge.txt", "a") as file:
                    file.write(
                        "Best valid epoch {} -->test epoch {} --> Final test results: {:.4f}, {:.4f}, {:.4f}, {:.4f}, {:.4f}\n".format(
                            best_epoch,it, t_hit1, t_hit3, t_hit10, t_MR, t_MRR))



                print("x" * 30, "Best test h1 ever: {:.4f},best test h3: {:.4f},best test h10{:.4f},best test mr: {:.4f}, best test mrr: {:.4f}".format(best_hit1_ever,best_hit3_ever,best_hit10_ever,best_mr_ever,best_mrr_ever), "x" * 30)
                # print("x" * 30, "Best valid h10 ever: {:.4f}, best valid mrr: {:.4f}".format(best_valid_hit10_ever, max_valid_mrr), "x" * 30)
        with open("result/HhsMath/1/wge.txt", "a") as file:
            file.write(
                "Best test h1 ever: {:.4f}, best test h3: {:.4f}, best test h10: {:.4f}, best test mr: {:.4f}, best test mrr: {:.4f}\n".format(
                    best_hit1_ever, best_hit3_ever, best_hit10_ever, best_mr_ever, best_mrr_ever))


class KGAT:
    """ Two-view Graph Neural Networks for Knowledge Graph Completion """

    def __init__(self):
        self.args = args

    """ Functions are adapted from https://github.com/ibalazevic/TuckER for using 1-N scoring strategy """

    def get_data_idxs(self, data):
        data_idxs = [(self.entity_idxs[data[i][0]], self.relation_idxs[data[i][1]], self.entity_idxs[data[i][2]]) for i
                     in range(len(data))]
        return data_idxs

    def evaluate(self, model, data, lst_indexes1, lst_indexes2, er_vocab, test=False, save=False, ep=0):
        model.eval()
        rel_dict = defaultdict(list)
        with torch.no_grad():
            hits = []
            ranks = []
            for i in range(10):
                hits.append([])

            test_data_idxs = self.get_data_idxs(data)
            print("Number of data points: %d" % len(test_data_idxs))
            # import pdb; pdb.set_trace()

            for i in range(0, len(test_data_idxs), 1000):
                data_batch, _ = get_batch(er_vocab, test_data_idxs, i, 1000, d)
                e1_idx = torch.tensor(data_batch[:, 0]).to(device)
                r_idx = torch.tensor(data_batch[:, 1]).to(device)
                e2_idx = torch.tensor(data_batch[:, 2]).to(device)

                forward = model.forward_normal

                preds = forward(e1_idx, r_idx, lst_indexes1, lst_indexes2, train=False)

                num_preds = len(preds)
                weights = [args.first_layer_weight, 0, 0]
                for _index in range(num_preds - 1):
                    weights[_index + 1] = (1 - weights[0]) / (num_preds - 1)

                pred = 0
                for _layer in range(num_preds):
                    pred += preds[_layer] * weights[_layer]

                pred = pred.detach()

                for j in range(data_batch.shape[0]):
                    this_e1 = data_batch[j][0]
                    this_r = data_batch[j][1]
                    filt = er_vocab[(this_e1, this_r)]
                    target_value = pred[j, e2_idx[j]].item()
                    pred[j, filt] = 0.0
                    pred[j, e2_idx[j]] = target_value

                sort_values, sort_idxs = torch.sort(pred, dim=1, descending=True)
                sort_idxs = sort_idxs.cpu().numpy()

                for j in range(data_batch.shape[0]):
                    this_r = data_batch[j][1]
                    rank = np.where(sort_idxs[j] == e2_idx[j].item())[0][0]
                    ranks.append(rank + 1)
                    rel_dict[self.relation_idx2id[this_r]].append(rank)
                    for hits_level in range(10):
                        if rank <= hits_level:
                            hits[hits_level].append(1.0)
                        else:
                            hits[hits_level].append(0.0)

        hit1 = np.mean(hits[0]) * 100
        hit3 = np.mean(hits[2]) * 100
        hit5 = np.mean(hits[4]) * 100
        hit10 = np.mean(hits[9]) * 100
        MR = np.mean(ranks)
        MRR = np.mean(1. / np.array(ranks))

        print('Hits @1: {0}'.format(hit1))
        print('Hits @3: {0}'.format(hit3))
        print('Hits @5: {0}'.format(hit5))
        print('Hits @10: {0}'.format(hit10))
        print('MR: {0}'.format(MR))
        print('Mean reciprocal rank: {0}'.format(MRR))

        return hit1, hit3, hit10, MR, MRR

    def prepare_data(self):
        self.entity_idxs = {d.entities[i]: i for i in range(len(d.entities))}  # entity ids to index
        self.entity_idx2id = {v: k for k, v in self.entity_idxs.items()}
        self.relation_idxs = {d.relations[i]: i for i in range(len(d.relations))}  # relation ids to index
        self.relation_idx2id = {v: k for k, v in self.relation_idxs.items()}
        # import pdb; pdb.set_trace()

        adj_r = construct_relation_focus_matrix(d.train_data, self.entity_idxs, self.relation_idxs, args.beta)
        adj = construct_entity_focus_matrix(d.train_data, self.entity_idxs)
        # edge_type, edge_index = d.process_data(d.train_data)
        train_data_idxs = self.get_data_idxs(d.train_data)
        deg = get_deg(train_data_idxs, len(self.entity_idxs)).to(device)

        print("Number of training data points: %d" % len(train_data_idxs))

        #model = WGE_model(args, len(self.entity_idxs), len(self.relation_idxs), adj, adj_r, deg=deg).to(device)
        model = KGAT_model(args, len(self.entity_idxs), len(self.relation_idxs), adj, adj_r, deg=deg).to(device)
        print("Using Adam optimizer")
        opt = torch.optim.Adam(model.parameters(), lr=args.lr)

        lst_indexes1 = torch.LongTensor([i for i in range(len(d.entities))]).to(device)
        lst_indexes2 = torch.LongTensor([i for i in range(len(d.entities) + len(d.relations))]).to(device)
        er_vocab = get_er_vocab(train_data_idxs)
        er_vocab_all = get_er_vocab(self.get_data_idxs(d.data))
        er_vocab_pairs = list(er_vocab.keys())




        return model, opt, lst_indexes1, lst_indexes2, er_vocab, er_vocab_all, er_vocab_pairs

    def train_and_eval(self, model, opt, lst_indexes1, lst_indexes2, er_vocab, er_vocab_all, er_vocab_pairs):

        max_valid_mrr, best_hit10_ever, best_mrr_ever, best_epoch = 0.0, 0.0, 0.0, 0
        best_valid_hit10_ever, best_valid_mrr_ever = 0.0, 0.0
        forward = model.forward_normal

        print("Starting training...")
        for it in tqdm(range(1, args.num_iterations + 1)):
            model.train()
            losses, losses0, losses1, losses2 = [], [], [], []
            np.random.shuffle(er_vocab_pairs)
            for j in range(0, len(er_vocab_pairs), args.batch_size):
                data_batch, targets = get_batch(er_vocab, er_vocab_pairs, j, args.batch_size, d)
                opt.zero_grad()
                e1_idx = torch.tensor(data_batch[:, 0]).to(device)
                r_idx = torch.tensor(data_batch[:, 1]).to(device)
                preds = forward(e1_idx, r_idx, lst_indexes1, lst_indexes2, train=True)
                targets = ((1.0 - args.label_smoothing) * targets) + (1.0 / targets.size(1))

                loss_list = [0, 0, 0]


                num_preds = len(preds)
                weights = [args.first_layer_weight, 0, 0]
                for _index in range(num_preds - 1):
                    weights[_index + 1] = (1 - weights[0]) / (num_preds - 1)

                for _layer in range(num_preds):
                    this_loss = model.loss(preds[_layer], targets)
                    loss_list[_layer] = this_loss.item()
                    loss += this_loss * weights[_layer]

                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)  # prevent the exploding gradient problem
                opt.step()
                losses.append(loss.item())
                losses0.append(loss_list[0])
                losses1.append(loss_list[1])
                losses2.append(loss_list[2])

            print("Epoch: {} --> Loss: {:.4f}, Loss0: {:.4f}, Loss1: {:.4f}, Loss2: {:.4f}".format(it, np.sum(losses),
                                                                                                   np.sum(losses0),
                                                                                                   np.sum(losses1),
                                                                                                   np.sum(losses2)))

            if (it > args.eval_after and it % args.eval_step == 0) or (it == 1):
                print("Valid:")
                hit1, hit3, hit10, MR, tmp_mrr = self.evaluate(model, d.valid_data, lst_indexes1, lst_indexes2,
                                                               er_vocab_all, test=False, save=True, ep=it)
                if max_valid_mrr < tmp_mrr:
                    max_valid_mrr = tmp_mrr
                    best_epoch = it
                if best_valid_hit10_ever < hit10:
                    best_valid_hit10_ever = hit10

                print("Test:")
                t_hit1, t_hit3, t_hit10, t_MR, t_MRR = self.evaluate(model, d.test_data, lst_indexes1, lst_indexes2,
                                                                     er_vocab_all, test=True, save=True, ep=it)
                if best_hit10_ever < t_hit10:
                    best_hit10_ever = t_hit10
                if best_mrr_ever < t_MRR:
                    best_mrr_ever = t_MRR

                print("Best valid epoch", best_epoch,
                      " --> Final test results: {:.4f}, {:.4f}, {:.4f}, {:.4f}, {:.4f}".format(t_hit1, t_hit3, t_hit10,
                                                                                               t_MR, t_MRR))
                # print("x" * 30, "Best test h10 ever: {:.4f}, best test mrr: {:.4f}".format(best_hit10_ever, best_mrr_ever), "x" * 30)
                # print("x" * 30, "Best valid h10 ever: {:.4f}, best valid mrr: {:.4f}".format(best_valid_hit10_ever, max_valid_mrr), "x" * 30)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="HhsMath", nargs="?", help="codex-s, codex-m, and codex-l")
    parser.add_argument("--batch_size", type=int, default=528, nargs="?", help="Batch size.")
    parser.add_argument("--emb_dim", type=int, default=256, nargs="?", help="embedding size")
    parser.add_argument("--encoder", type=str, default="dir", nargs="?", help="encoder")
    parser.add_argument("--decoder", type=str, default="quate", nargs="?", help="decoder")
    parser.add_argument("--eval_step", type=int, default=10, nargs="?", help="how often doing eval")
    parser.add_argument("--eval_after", type=int, default=1500, nargs="?", help="only eval after this interation")
    parser.add_argument("--label_smoothing", type=float, default=0.1, help="smoothing constant")

    parser.add_argument("--num_iterations", type=int, default=1500, nargs="?", help="Number of iterations.")
    parser.add_argument("--lr", type=float, default=0.00005, nargs="?", help="Learning rate")
    parser.add_argument("--num_layers", type=int, default=2, help="number of GCN layers")
    parser.add_argument("--first_layer_weight", type=float, default=0.5, help="weight of the first GCN layer") # weight of the first GCN layer!
    parser.add_argument("--beta", type=float, default=0.2, help="triple keeping percentage")##0.2 0.3
    parser.add_argument("--combine_type", type=str, default='corr', help="cat, average, corr, linear_corr") # consider this

    # # parser.add_argument("--cl_lr", type=float, default=1e-3, help="learning rate of contrastive learning")
    # # parser.add_argument("--op", type=str, default="corr_new", help="aggregation opration")
    # # parser.add_argument("--init_emb", type=str, default="random", help="initial embedding")
    # # parser.add_argument("--ent_height", type=int, default=8, help="enttities embedding height after reshaping")
    # # parser.add_argument("--init_dim", type=int, default=256, help="dimension of entities embeddings")
    # # parser.add_argument("--ent_dim", type=int, default=256, help="dimension of entities embeddings")
    # # parser.add_argument("--rel_dim", type=int, default=256, help="dimension of relations embeddings")
    # # parser.add_argument("--encoder_drop", type=float, default=0.1, help="dropout ratio for encoder")
    # # parser.add_argument("--encoder_hid_drop", type=float, default=0.3, help="dropout ratio for encoder")
    # # parser.add_argument("--filter_size", type=int, default=7, help="size of relation specific kernels")
    # # parser.add_argument("--filter_channel", type=int, default=256, help="number of filter channels")
    # # parser.add_argument("--proj", type=str, default="linear", help="projection head type")
    #
    parser.add_argument("--train_path", type=str, default="./data/HhsMath/", help="knowledge graph dataset path")
    parser.add_argument("--test_path", type=str, default="./data/HhsMath/", help="knowledge graph dataset path")
    parser.add_argument("--noise_path", type=str, default=None, help="knowledge graph dataset path")
    #
    parser.add_argument("--rel_num", type=int, default=2*2, help="number of relations in Knowledge Graph")
    parser.add_argument("--ent_num", type=int, default=244, help="number of entites in Knowledge Graph")
    # parser.add_argument("--init_dim", type=int, default=256, help="dimension of entities embeddings")
    # parser.add_argument("--ent_dim", type=int, default=256, help="dimension of entities embeddings")
    # parser.add_argument("--rel_dim", type=int, default=256, help="dimension of relations embeddings")
    # parser.add_argument("--filter_size", type=int, default=7, help="size of relation specific kernels")
    # parser.add_argument("--cl_batch_size", type=int, default=1024, help="training batch size")
    # parser.add_argument("--cl_lr", type=float, default=1e-3, help="learning rate of contrastive learning")
    # parser.add_argument("--decode_batch_size", type=int, default=2048, help="learning rate of decode stage")
    # parser.add_argument("--decode_lr", type=float, default=5e-4, help="learning rate of decoding")
    # parser.add_argument("--decode_epochs", type=int, default=100, help="max epochs of decode training")
    # parser.add_argument("--cl_epochs", type=int, default=1000, help="epochs of contrastive learning")
    # parser.add_argument("--ent_height", type=int, default=8, help="enttities embedding height after reshaping")
    # parser.add_argument("--encoder_drop", type=float, default=0.1, help="dropout ratio for encoder")
    # parser.add_argument("--encoder_hid_drop", type=float, default=0.3, help="dropout ratio for encoder")
    # parser.add_argument("--proj_hid", type=int, default=256, help="hidden dimension of projection head")
    # parser.add_argument("--temp1", type=float, default=0.07, help="temperature of contrastive loss")
    # parser.add_argument("--temp2", type=float, default=0.07, help="temperature of contrastive loss")
    # # parser.add_argument("--label_smoothing", type=float, default=0.0, help="label smoothing value")
    # parser.add_argument("--op", type=str, default="corr_new", help="aggregation opration")
    # parser.add_argument("--init_emb", type=str, default="random", help="initial embedding")
    # parser.add_argument("--gcn_layer", type=int, default=1, help="number of gcn layer")
    # parser.add_argument("--proj", type=str, default="linear", help="projection head type")
    # parser.add_argument("--input_drop", type=float, default=0.2, help="input dropout ratio")
    # parser.add_argument("--fea_drop", type=float, default=0.3, help="feature map dropout ratio")
    # parser.add_argument("--hid_drop", type=float, default=0.3, help="hidden feature dropout ratio")
    # parser.add_argument("--filter_channel", type=int, default=256, help="number of filter channels")
    # parser.add_argument("--bias", type=bool, default=True, help="whether to use bias in convolution opeation")
    # parser.add_argument("--kg_md", type=str, default="conve", help="Knowledge Graph prediction model")
    # #parser.add_argument("--dataset", type=str, default="wn18rr", help="choose the dataset to perform the model")
    # parser.add_argument("--proj_dim", type=int, default=128, help="projection dimension")
    # parser.add_argument("--valid_routine", type=int, default=1, help="valid_routine")
    # parser.add_argument("--random_seed", type=int, default=None, help="random seed")
    # parser.add_argument("--lam1", type=float, default=1, help="weight for two loss function")
    # parser.add_argument("--lam2", type=float, default=1, help="weight for two loss function")
    # parser.add_argument("--info", type=str, default="HhsMath", help="description for experiment")
    # # parser.add_argument("--beta", type=float, default=0, help="description for experiment")
    # parser.add_argument("--save_emb", type=bool, default=False, help="description for experiment")
    # # parser.add_argument("--neg_sample", type=int, default=2, help="description for experiment")
    # parser.add_argument("--p", type=float, default=1.0, help="training data percentage")
    parser.add_argument("--num_worker", type=int, default=4, help="num workers")
    # parser.add_argument("--weight_decay", type=float, default=0, help="num workers")

    args = parser.parse_args()
    print(args)

    # load dataset
    dataset = args.dataset
    data_dir = "data/%s/" % dataset
    d = Data(data_dir=data_dir) 


    #gnnkge = WGE()
    gnnkge=W()
    # 导入txt2triples函数
    from load_data import txt2triples1

    # 假设你的文件路径是 args.train_path + "train2id.txt"

    # 调用txt2triples函数，获取返回的结果

    # model, opt, lst_indexes1, lst_indexes2, er_vocab, er_vocab_all, er_vocab_pairs = gnnkge.prepare_data()
    model, opt, lst_indexes1, lst_indexes2, er_vocab, er_vocab_all, er_vocab_pairs, t_vocab, t_vocab_all, t_vocab_pairs, ert_vocab, ert_vocab_all, ert_vocab_pairs,dir_adj_edge_index,dir_adjr_edge_index=gnnkge.prepare_data()
    # gnnkge.train_and_eval(model, opt, lst_indexes1, lst_indexes2, er_vocab, er_vocab_all, er_vocab_pairs)
    gnnkge.train_and_eval(model, opt, lst_indexes1, lst_indexes2, er_vocab, er_vocab_all, er_vocab_pairs,ert_vocab, ert_vocab_all, ert_vocab_pairs,dir_adj_edge_index,dir_adjr_edge_index)

