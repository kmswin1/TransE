import torch
import json

def main():
    ent2id = dict()
    id2ent = set()
    rel2id = dict()
    id2rel = set()
    with open('data/modified_triples.txt', 'r') as f:
        for line in f:
            line = json.loads(line)
            id2ent.add(line['src'])
            id2rel.add(line['dstProperty'])
            id2ent.add(line['dst'])

    id2ent = sorted(list(id2ent))
    id2rel = sorted(list(id2rel))

    for i,meta in enumerate(id2ent):
        ent2id[meta] = i

    for i,meta in enumerate(id2rel):
        rel2id[meta] = i

    torch.save(ent2id, 'data/vocabulary/ent2id.pt')
    torch.save(id2ent, 'data/vocabulary/id2ent.pt')
    torch.save(rel2id, 'data/vocabulary/rel2id.pt')
    torch.save(id2rel, 'data/vocabulary/id2rel.pt')

if __name__ == "__main__":
    main()