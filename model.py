import torch
import torch.nn as nn
import torch.nn.functional as F

class TransE(nn.Module):
    def __init__(self, opts, ent_tot, rel_tot):
        super(TransE, self).__init__()
        self.ent_embeddings = nn.Embedding(ent_tot, opts.dim)
        self.rel_embeddings = nn.Embedding(rel_tot, opts.dim)
        nn.init.xavier_uniform_(self.ent_embeddings.weight.data)
        nn.init.xavier_uniform_(self.rel_embeddings.weight.data)

    def calc_score_head(self, h, r, t):
        h = F.normalize(h, 2, -1)
        t = F.normalize(t, 2, -1)
        score = (h + r) - t
        score = -1 * torch.norm(score, 2, -1)
        return score

    def calc_score_tail(self, h, r, t):
        h = F.normalize(h, 2, -1)
        t = F.normalize(t, 2, -1)
        score = h + (r - t)
        score = -1 * torch.norm(score, 2, -1)
        return score

    def calc_dist(self, h, t):
        dist = h-t
        return torch.sum(torch.norm(dist, 2, -1))

    def forward(self, batch_head, batch_rel, batch_tail, batch_negative):
        h = self.ent_embeddings(batch_head)
        r = self.rel_embeddings(batch_rel)
        t = self.ent_embeddings(batch_tail)
        n = self.ent_embeddings(batch_negative)
        pos_score_1 = self.calc_score_head(h, r, t)
        pos_score_2 = self.calc_score_tail(h, r, t)
        neg_score_1 = self.calc_score_head(n, r ,t)
        neg_score_2 = self.calc_score_tail(h, r, n)
        pos_score = torch.cat([pos_score_1, pos_score_2], -1)
        neg_score = torch.cat([neg_score_1, neg_score_2], -1)
        dist = self.calc_dist(h,t)
        return pos_score, neg_score, dist

    def entity_normalize(self):
        self.ent_embeddings.weight.data = F.normalize(self.ent_embeddings.weight.data, 2, -1)

    def relation_normalize(self):
        self.rel_embeddings.weight.data = F.normalize(self.rel_embeddings.weight.data, 2, -1)

class DistMult(nn.Module):
    def __init__(self, opts, ent_tot, rel_tot):
        super(DistMult, self).__init__()
        self.ent_embeddings = nn.Embedding(ent_tot, opts.dim)
        self.rel_embeddings = nn.Embedding(rel_tot, opts.dim)
        nn.init.xavier_uniform_(self.ent_embeddings.weight.data)
        nn.init.xavier_uniform_(self.rel_embeddings.weight.data)

    def calc_score_head(self, h, r, t):
        h = F.normalize(h, 2, -1)
        t = F.normalize(t, 2, -1)
        score = (h * r) * t
        score = -1 * score.sum(dim=-1)
        return score

    def calc_score_tail(self, h, r, t):
        h = F.normalize(h, 2, -1)
        t = F.normalize(t, 2, -1)
        score = h * (r * t)
        score = -1 * score.norm(dim=-1)
        return score

    def forward(self, batch_head, batch_rel, batch_tail, batch_negative):
        h = self.ent_embeddings(batch_head)
        r = self.rel_embeddings(batch_rel)
        t = self.ent_embeddings(batch_tail)
        n = self.ent_embeddings(batch_negative)
        pos_score_1 = self.calc_score_head(h, r, t)
        pos_score_2 = self.calc_score_tail(h, r, t)
        neg_score_1 = self.calc_score_head(n, r ,t)
        neg_score_2 = self.calc_score_tail(h, r, n)
        pos_score = torch.cat([pos_score_1, pos_score_2], -1)
        neg_score = torch.cat([neg_score_1, neg_score_2], -1)
        return pos_score, neg_score

    def entity_normalize(self):
        self.ent_embeddings.weight.data = F.normalize(self.ent_embeddings.weight.data, 2, -1)

    def relation_normalize(self):
        self.rel_embeddings.weight.data = F.normalize(self.rel_embeddings.weight.data, 2, -1)