from torch.utils.data import Dataset
import torch
import json
import random

class DataSet(Dataset):
    def __init__(self, file_path):
        self.len = 0
        self.head = []
        self.rel = []
        self.tail = []
        self.triple = []
        self.negative = []
        self.ent2id = torch.load('data/vocabulary/ent2id.pt')
        self.id2ent = torch.load('data/vocabulary/id2ent.pt')
        self.rel2id = torch.load('data/vocabulary/rel2id.pt')
        self.id2rel = torch.load('data/vocabulary/id2rel.pt')
        self.ent_tot = len(self.id2ent)
        self.rel_tot = len(self.id2rel)
        with open(file_path) as f:
            for line in f:
                line = json.loads(line)
                self.len += 1
                self.head.append(self.ent2id[line['src']])
                self.rel.append(self.rel2id[line['dstProperty']])
                self.tail.append(self.ent2id[line['dst']])
                self.negative.append(self.id2ent.index(random.sample(self.id2ent, 1)[0]))

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        return self.head[idx], self.rel[idx], self.tail[idx], self.negative[idx]