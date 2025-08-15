
""" The file is taken from https://github.com/ibalazevic/TuckER"""
import torch
from itertools import chain
import torch
from torch.utils.data import Dataset
import torch
import torch.utils.data as Data
import json
from utils_WGE import add_inverse_triples, add_noise
class Data:

    def __init__(self, data_dir="data/DataStructures/", reverse=True):
        self.train_data = self.load_data(data_dir, "train", reverse=reverse)
        self.valid_data = self.load_data(data_dir, "valid", reverse=reverse)
        self.test_data = self.load_data(data_dir, "test", reverse=reverse)
        self.data = self.train_data + self.valid_data + self.test_data
        self.entities = self.get_entities(self.data)
        self.train_relations = self.get_relations(self.train_data)
        self.valid_relations = self.get_relations(self.valid_data)
        self.test_relations = self.get_relations(self.test_data)
        self.relations = self.train_relations + [i for i in self.valid_relations \
                if i not in self.train_relations] + [i for i in self.test_relations \
                if i not in self.train_relations]
        # self.train_edge_index, self.train_edge_type = self.process_data(self.train_data)
        # self.valid_edge_index, self.valid_edge_type = self.process_data(self.valid_data)

    def load_data(self, data_dir, data_type="train", reverse=False):
        with open("%s%s.txt" % (data_dir, data_type), "r") as f:
            data = f.read().strip().split("\n")
            data = [i.split() for i in data]
            if reverse:
                data += [[i[2], "reverse_" + i[1], i[0]] for i in data]
        return data

    def get_relations(self, data):
        relations = sorted(list(set([d[1] for d in data])))
        return relations

    def get_entities(self, data):
        entities = sorted(list(set([d[0] for d in data]+[d[2] for d in data])))
        return entities

    # def process_data(self, data):
    #     entities = self.get_entities(data)
    #     entity_dict = {e: i for i, e in enumerate(entities)}
    #     edge_index = []
    #     edge_type = []
    #     for triple in data:
    #         src = entity_dict[triple[0]]
    #         dst = entity_dict[triple[2]]
    #         rel = triple[1]
    #         edge_index.append([src, dst])
    #         edge_type.append(rel)
    #     edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    #     edge_type = torch.tensor(edge_type, dtype=torch.long)
    #     return edge_index, edge_type


def txt2triples(path):

    with open(path, 'r') as f:
        data = f.read().split()
        #print(data)
        src = data[1::3]
        dst = data[2::3]
        edge_type = data[3::3]
        #print(src)
        # print("!1!!!!!1!")
        # print(dst)
        src = torch.tensor([int(i) for i in src])
        dst = torch.tensor([int(i) for i in dst])
        rel= torch.tensor([int(i) for i in edge_type])



        # data = add_inverse_triples(torch.stack((src, rel1, dst), dim=1))
        # src, rel, dst = data.t()
        # print(rel1)
        return data, src, rel, dst  # 数据索引 头实体索引 关系索引 为实体索引
def txt2triples1(file_path):
    triples = []
    src = []
    rel = []
    dst = []

    with open(file_path, 'r') as file:
        lines = file.readlines()

        for i, line in enumerate(lines):
            line = line.strip().split()
            if len(line) != 3:
                continue

            try:
                s, r, d = line
                src.append(int(s))
                rel.append(int(r))
                dst.append(int(d))
                triples.append((s, r, d))
            except ValueError as e:
                print(f"Error in line {i + 1}: {line}")
                print(f"Problematic value: {e.args[0]}")

    return triples, src, rel, dst
# def txt2triples(file_path):
#     triples = []
#     src = []
#     rel = []
#     dst = []
#
#     with open(file_path, 'r') as file:
#         lines = file.readlines()
#
#         for i, line in enumerate(lines):
#             line = line.strip().split()
#             if len(line) != 3:
#                 continue
#
#             try:
#                 s, r, d = line
#                 src.append(int(s))
#                 rel.append(int(r))
#                 dst.append(int(d))
#                 triples.append((s, r, d))
#             except ValueError as e:
#                 print(f"Error in line {i + 1}: {line}")
#                 print(f"Problematic value: {e.args[0]}")
#
#     return triples, src, rel, dst

class NELL_txt(Dataset):
    def __init__(self, name, batch_size, path):
        self.name = name
        self.batch_size = batch_size

        if self.name == "train":
            self.train_data, src, rel, dst = txt2triples(path + "train2id.txt")
            self.edge_type = rel
            self.edge_index = torch.stack((src, dst), dim=0)

        if self.name == "valid":
            self.train_data, src, rel, dst = txt2triples(path + "train2id.txt")
            self.edge_type = rel
            self.edge_index = torch.stack((src, dst), dim=0)
            self.valid_data, _, _, _ = txt2triples(path + "valid2id.txt")

        if self.name == "test":
            self.train_data, src, rel, dst = txt2triples(path + "train2id.txt")
            self.edge_type = rel
            self.edge_index = torch.stack((src, dst), dim=0)
            self.valid_data, _, _, _ = txt2triples(path + "valid2id.txt")
            self.test_data, _, _, _ = txt2triples(path + "test2id.txt")

    def __len__(self):
        if self.name == "train":
            return int(self.train_data.size(0) / self.batch_size)
        else:
            return 1

    def __getitem__(self, index):
        if self.name == "train":
            return self.edge_index, self.edge_type, self.train_data
        elif self.name == "valid":
            return self.train_data, self.valid_data, self.edge_index, self.edge_type
        elif self.name == "test":
            return self.train_data, self.valid_data, self.test_data


class KG_Triples_txt(Dataset):
    def __init__(self, name, batch_size, path="/data/HhsMath"):
        self.name = name
        self.batch_size = batch_size

        if self.name == "train":
            self.train_data, src, rel, dst = txt2triples(path + "train2id.txt")
            self.edge_type = rel
            self.edge_index = torch.stack((src, dst), dim=0)

        if self.name == "valid":
            self.train_data, src, rel, dst = txt2triples(path + "train2id.txt")
            self.edge_type = rel
            self.edge_index = torch.stack((src, dst), dim=0)
            self.valid_data, _, _, _ = txt2triples(path + "valid2id.txt")

        if self.name == "test":
            self.train_data, src, rel, dst = txt2triples(path + "train2id.txt")
            self.edge_type = rel
            self.edge_index = torch.stack((src, dst), dim=0)
            self.valid_data, _, _, _ = txt2triples(path + "valid2id.txt")
            self.test_data, _, _, _ = txt2triples(path + "test2id.txt")

    def __len__(self):
        if self.name == "train":
            return int(self.train_data.size(0) / self.batch_size)
        else:
            return 1

    def __getitem__(self, index):
        if self.name == "train":
            return self.edge_index, self.edge_type, self.train_data
        elif self.name == "valid":
            return self.train_data, self.valid_data, self.edge_index, self.edge_type
        elif self.name == "test":
            return self.train_data, self.valid_data, self.test_data


class Triple_Category(Dataset):
    def __init__(self, name, batch_size, path):
        self.name = name
        self.batch_size = batch_size

        if self.name == "train":
            self.train_data, src, rel, dst = txt2triples(path + "train2id.txt")
            self.edge_type = rel
            self.edge_index = torch.stack((src, dst), dim=0)

        if self.name == "valid":
            self.train_data, src, rel, dst = txt2triples(path + "train2id.txt")
            self.edge_type = rel
            self.edge_index = torch.stack((src, dst), dim=0)
            self.valid_data, _, _, _ = txt2triples(path + "valid2id.txt")

        if self.name == "test":
            self.train_data, src, rel, dst = txt2triples(path + "train2id.txt")
            self.edge_type = rel
            self.edge_index = torch.stack((src, dst), dim=0)
            self.valid_data, _, _, _ = txt2triples(path + "valid2id.txt")
            self.relation1to1, _, _, _ = txt2triples(path + "1-1.txt")
            self.relation1ton, _, _, _ = txt2triples(path + "1-n.txt")
            self.relationnto1, _, _, _ = txt2triples(path + "n-1.txt")
            self.relationnton, _, _, _ = txt2triples(path + "n-n.txt")

    def __len__(self):
        if self.name == "train":
            return int(self.train_data.size(0) / self.batch_size)
        else:
            return 1

    def __getitem__(self, index):
        if self.name == "train":
            return self.edge_index, self.edge_type, self.train_data
        elif self.name == "valid":
            return self.train_data, self.valid_data, self.edge_index, self.edge_type
        elif self.name == "test":
            return self.train_data, self.valid_data, self.relation1to1, self.relation1ton, self.relationnto1, self.relationnton


class KG_Triples(Dataset):
    def __init__(self, name, num_relations, num_ent, train_path="./data/HhsMath/", test_path="./data/HhsMath/",
                 noise_path=None, num_negs_per_pos=5):
        self.name = name
        self.num_relations = num_relations
        self.num_entities = num_ent
        self.num_negs_per_pos = num_negs_per_pos
        self.noise_path = noise_path
        self.train_data, _, _, _ = txt2triples(train_path + "train2id.txt")

        if self.name == "train":
            self.train_data, _, _, _ = txt2triples(train_path + "train2id.txt")
            if self.noise_path is not None:
                noise_data, _, _, _ = txt2triples(self.noise_path + "train2id.txt")
                self.train_data, _, _, _ = add_noise(self.train_data, noise_data)

        if self.name == "valid":
            self.train_data, _, _, _ = txt2triples(test_path + "train2id.txt")
            # self.edge_type = rel
            # self.edge_index = torch.stack((src, dst), dim=0)
            self.valid_data, _, _, _ = txt2triples(test_path + "valid2id.txt")
            self.test_data, _, _, _ = txt2triples(test_path + "test2id.txt")

        if self.name == "test":
            self.train_data, _, _, _ = txt2triples(test_path + "train2id.txt")
            # self.edge_type = rel
            # self.edge_index = torch.stack((src, dst), dim=0)
            self.valid_data, _, _, _ = txt2triples(test_path + "valid2id.txt")
            self.test_data, _, _, _ = txt2triples(test_path + "test2id.txt")

    # 这段corrupt_batch函数的注释描述了该函数的功能和参数。该函数用于生成负样本批次，通过对正样本进行随机破坏。
    #
    # 参数
    # positive_batch
    # 是一个形状为(B, 3)
    # 的张量，表示正样本批次，其中
    # B
    # 是批次大小。
    #
    # 函数首先获取批次的形状
    # batch_shape。
    #
    # 接着，通过
    # positive_batch.clone()
    # 创建一个
    # negative_batch，作为负样本批次的初始副本。同时，使用
    # unsqueeze
    # 和
    # repeat
    # 操作将
    # negative_batch
    # 复制为形状为(*batch_shape, self.num_negs_per_pos, 3)
    # 的张量，以生成重复的副本。
    #
    # 然后，使用
    # torch.randint
    # 生成
    # corruption_index，形状为(*batch_shape, self.num_negs_per_pos)，其中每个元素是随机选择的
    # 0
    # 或
    # 1。这些值将用于确定哪些位置进行破坏。
    #
    # 接下来，通过
    # torch.randint
    # 生成
    # negative_indices，形状为(mask.sum().item(), )，其中
    # mask
    # 是
    # corruption_index
    # 中值为
    # 1
    # 的位置的布尔掩码。negative_indices
    # 是从
    # 0
    # 到
    # index_max
    # 之间随机选择的整数，用作负样本的替代索引。
    #
    # 为了确保不替换正样本的值，函数使用
    # shift
    # 计算出偏移量，即大于等于正样本值的位置需要向上移动一位。然后，将
    # shift
    # 加到
    # negative_indices
    # 中。
    #
    # 最后，将生成的负样本索引
    # negative_indices
    # 写入
    # negative_batch
    # 中，通过使用掩码
    # mask
    # 选择需要替换的位置，并将负样本索引写入这些位置。
    #
    # 函数返回生成的负样本批次negative_batch，其形状与输入positive_batch相同。
    def corrupt_batch(self, positive_batch: torch.LongTensor) -> torch.LongTensor:  # noqa: D102

        batch_shape = positive_batch.shape[:-1]
        # Copy positive batch for corruption.
        # Do not detach, as no gradients should flow into the indices.
        negative_batch = positive_batch.clone()
        negative_batch = negative_batch.unsqueeze(dim=-2).repeat(*(1 for _ in batch_shape), self.num_negs_per_pos, 1)

        corruption_index = torch.randint(1, size=(*batch_shape, self.num_negs_per_pos))

        index_max = self.num_relations
        mask = corruption_index == 1
        # To make sure we don't replace the {head, relation, tail} by the
        # original value we shift all values greater or equal than the original value by one up
        # for that reason we choose the random value from [0, num_{heads, relations, tails} -1]
        negative_indices = torch.randint(
            high=index_max,
            size=(mask.sum().item(),),
            device=positive_batch.device,
        )

        # determine shift *before* writing the negative indices
        shift = (negative_indices >= negative_batch[mask][:, 1]).long()
        negative_indices += shift

        # write the negative indices
        negative_batch[
            mask.unsqueeze(dim=-1) & (torch.arange(3) == 1).view(*(1 for _ in batch_shape), 1, 3)
            ] = negative_indices

        return negative_batch

    def __len__(self):
        if self.name == "train":
            # return self.train_data.size(0)
            return len(self.train_data)
        else:
            return 1

    def __getitem__(self, index):
        if self.name == "train":
            pos_batch = self.train_data[index]
            neg_batch = self.corrupt_batch(pos_batch)
            return pos_batch, neg_batch  # 返回正样本 负样本

        elif self.name == "valid":
            return self.train_data, self.valid_data, self.test_data

        elif self.name == "test":
            return self.train_data, self.valid_data, self.test_data
