import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='Train classification network')
    parser.add_argument('--lr', default=1e-3, type=float, help='Initial learning rate')
    parser.add_argument('--train', default=1, type=int, help='0: training mode, 1: evaluation mode')

    args = parser.parse_args()

    return args

if __name__ == "__main__":
    args = parse_args()
    print("learning rate: %f", args.lr)
    print("training mode: %d" % (args.train))