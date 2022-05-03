import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='Train classification network')
    parser.add_argument('--lr', default=1e-3, type=float, help='Initial learning rate')
    parser.add_argument('--momentum', default=0.9, type=float, help='Momentum')
    parser.add_argument('--wd', default=0, type=float, help='Weight decay')
    parser.add_argument('--optim', default='sgd', help='Optimizer: sgd|adam|adagrad|adadelta|rmsprop')
    parser.add_argument('--net', default='Baseline', help='Network to train')
    parser.add_argument('--train', default=1, type=int, help='0: training mode, 1: evaluation mode')
    parser.add_argument('--resume', default=0, type=int, help='resume training')
    parser.add_argument('--eval', default=0, type=int, help='evaluate model')
    parser.add_argument('--save_nth_epoch', default=1, type=int, help='Save model each n-th epoch during training')
    parser.add_argument('--test_nth_epoch', default=1, type=int, help='Test each n-th epoch during training')
    parser.add_argument('--batch_size', default=32, type=int, help='batch_size')
    parser.add_argument('--seed', default=1, type=int, help='random seed')
    parser.add_argument('--dataset', default='PVG', help='Dataset')
    parser.add_argument('--epochs', default=300, type=int, help='number of epochs')
    parser.add_argument('--nworkers', default=4, type=int, help='number of workers')

    args = parser.parse_args()

    return args

if __name__ == "__main__":
    args = parse_args()
    print("learning rate: %f", args.lr)
    print("training mode: %d" % (args.train))