import sys
import numpy as np
import json
import matplotlib.pyplot as plt


def plot_losses(epochs, trainloss, valloss):

    fig = plt.figure()
    plt.plot(epochs, trainloss, label="trainloss")
    epochs_val = [1]
    increment = len(epochs) / len(valloss)
    for i in range(1, len(valloss)):
        epochs_val.append(int(i * increment))

    plt.plot(epochs_val, valloss, label="valloss")
    plt.legend()
    plt.show()

    

if __name__=="__main__":
    log_file = sys.argv[1]
    stats = json.loads(open(log_file).read())
    valloss = [k['loss_val'] for k in stats if 'loss_val' in k.keys()]
    trainloss = [k['loss_train'] for k in stats if 'loss_train' in k.keys()]
    epochs = [k['epoch'] for k in stats if 'epoch' in k.keys()]
    plot_losses(epochs, trainloss, valloss)
    #plot_accuracy()

