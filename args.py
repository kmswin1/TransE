from argparse import ArgumentParser


def get_train_args():
    parser = ArgumentParser()
    parser.add_argument('-epochs', '--epochs', type=int, default=2000, help='the number of maximum training epoch')
    parser.add_argument('-model_type', '--model_type', type=str, default='transe', help='lower-cased model name')
    parser.add_argument('-optimizer', '--optimizer', type=str, default='Adam', help='lower-cased optimizer string')
    parser.add_argument('-lr', '--lr', type=float, default=0.001, help='initial learning rate of optimizer')
    parser.add_argument('-dim', '--dim', type=int, default=25, help='dimension of KG representation vectors')
    parser.add_argument('-order', '--order', type=int, default=2, help='order of norm (1 or 2)')
    parser.add_argument('-margin', '--margin', type=float, default=1.0, help='margin for RankingLoss')
    parser.add_argument('-num_processes', '--num_processes', type=int, default=10, help='the number of processes')
    parser.add_argument('-seed', '--seed', type=int, default=1234, help='random seed')
    parser.add_argument('-batch_size', '--batch_size', type=int, default=1000, help='size of batch in mini-batch training')
    parser.add_argument('-save_step', '--save_step', type=int, default=10, help='step size of save check point of model')
    parser.add_argument('-data_path', '--data_path', type=str, help='path of preprocessed triple data')
    parser.add_argument('-model_path', '--model_path', type=str, help='path to save trained model')
    parser.add_argument('-embedding_path', '--embedding_path', type=str, help='path to save trained embeddings')
    parser.add_argument('-alpha', '--alpha', type=float, default=0.5, help='ratio of relation(or attribute) loss in joint loss')
    args = parser.parse_args()
    return args