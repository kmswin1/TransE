import torch
from torch.utils.data import DataLoader
from model import TransE, DistMult
from dataset import DataSet
from args import get_train_args
import torch.optim as optim
import time, math, json, os

def save_embeddings(model, opts, id2ent, id2rel, num_splits=1):
    total_ent_cnt = len(model.ent_embeddings.weight.data)
    split_size = math.ceil(total_ent_cnt / num_splits)
    for n in range(num_splits):
        with open('embeddings/embeddings.json', 'w') as o:
            for i, embedding in enumerate(model.ent_embeddings.weight.data[n * split_size: (n+1) * split_size]):
                d = {
                    'cruise_id': id2ent[i],
                    'embedding': embedding.tolist()
                }
                o.write(json.dumps(d) + '\n')
            # All relation embeddings are written on the last json split.
            if n == num_splits - 1:
                for i, embedding in enumerate(model.rel_embeddings.weight.data):
                    d = {
                        'cruise_id': id2rel[i],
                        'embedding': embedding.tolist()
                    }
                    o.write(json.dumps(d) + '\n')
        with open('embeddings/embeddings.txt', 'w') as o:
            for i, embedding in enumerate(model.ent_embeddings.weight.data[n * split_size: (n+1) * split_size]):
                o.write(str(id2ent[i]) + ' ' + str(embedding.tolist()) + '\n')
            # All relation embeddings are written on the last json split.
            if n == num_splits - 1:
                for i, embedding in enumerate(model.rel_embeddings.weight.data):
                    o.write(str(id2rel[i]) + ' ' + str(embedding.tolist()) + '\n')

def main():
    opts = get_train_args()
    print ("load data ...")
    data = DataSet('data/modified_triples.txt')
    dataloader = DataLoader(data, shuffle=True, batch_size=opts.batch_size)
    print ("load model ...")
    if opts.model_type == 'transe':
        model = TransE(opts, data.ent_tot, data.rel_tot)
    elif opts.model_type =="distmult":
        model = DistMult(opts, data.ent_tot, data.rel_tot)
    if opts.optimizer == 'Adam':
        optimizer = optim.Adam(model.parameters(), lr=opts.lr)
    elif opts.optimizer == 'SGD':
        optimizer = optim.SGD(model.parameters(), lr=opts.lr)
    model.cuda()
    model.relation_normalize()
    loss = torch.nn.MarginRankingLoss(margin=opts.margin)

    print ("start training")
    for epoch in range(1, opts.epochs + 1):
        print ("epoch : " + str(epoch))
        model.train()
        epoch_start = time.time()
        epoch_loss = 0
        tot = 0
        cnt = 0
        for i, batch_data in enumerate(dataloader):
            optimizer.zero_grad()
            batch_h, batch_r, batch_t, batch_n = batch_data
            batch_h = torch.LongTensor(batch_h).cuda()
            batch_r = torch.LongTensor(batch_r).cuda()
            batch_t = torch.LongTensor(batch_t).cuda()
            batch_n = torch.LongTensor(batch_n).cuda()
            pos_score, neg_score, dist = model.forward(batch_h, batch_r, batch_t, batch_n)
            pos_score = pos_score.cpu()
            neg_score = neg_score.cpu()
            dist = dist.cpu()
            train_loss = loss(pos_score, neg_score, torch.ones(pos_score.size(-1))) + dist
            train_loss.backward()
            optimizer.step()
            batch_loss = torch.sum(train_loss)
            epoch_loss += batch_loss
            batch_size = batch_h.size(0)
            tot += batch_size
            cnt += 1
            print ('\r{:>10} epoch {} progress {} loss: {}\n'.format('', epoch, tot/data.__len__(), train_loss), end='')
        end = time.time()
        time_used = end - epoch_start
        epoch_loss /= cnt
        print ('one epoch time: {} minutes'.format(time_used/60))
        print ('{} epochs'.format(epoch))
        print ('epoch {} loss: {}'.format(epoch, epoch_loss))

        if epoch % opts.save_step == 0:
            print ("save model...")
            model.entity_normalize()
            torch.save(model.state_dict(), 'model.pt')

    print("save model...")
    model.entity_normalize()
    torch.save(model.state_dict(), 'model.pt')
    print("[Saving embeddings of whole entities & relations...]")
    save_embeddings(model, opts, data.id2ent, data.id2rel)
    print("[Embedding results are saved successfully.]")

if __name__ == '__main__':
    main()