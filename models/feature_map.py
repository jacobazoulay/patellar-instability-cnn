import torch
import random
import torchvision.transforms as T
import numpy as np
from scipy.ndimage.filters import gaussian_filter1d
import matplotlib.pyplot as plt
import os
from models import AlexNet
from models import common
import argparse
import torchvision.models as models
import torch
import torch.nn as nn
import torch._utils
from torch.autograd import Variable
import json
import pandas as pd
import pingouin as pg


def main():
    VGGNet = True
    Alexnet = False

    if VGGNet:
        PATH = "/Users/jacobazoulay/Repos/Total Results/results/CDI/VGGNet_b8/models/model_600.pth.tar"
        vgg16 = models.vgg16(pretrained=True)
        vgg16.classifier[6] = nn.Linear(4096, 6)
        model = vgg16

    if Alexnet:
        PATH = os.path.join(os.getcwd(), "results/CDI/cache/AlexNet_b16WD1e3/models/model_100.pth.tar")
        parser = argparse.ArgumentParser(description='Train classification network')
        parser.add_argument('--num_classes', default=6, type=int, help='Num labels')
        args = parser.parse_args()
        model = AlexNet.AlexNet(args)

    # Load model parameters
    checkpoint = torch.load(PATH, map_location=torch.device('cpu'))
    model_state = checkpoint['state_dict']
    model_params = model_state.copy()

    for key in model_state.keys():
        model_params[key.replace('module.model.', '')] = model_state[key]
        del model_params[key]

    model.load_state_dict(model_params)
    model.eval()

    dtype = torch.FloatTensor
    model.type(dtype)


    # Load random image data and labels
    dtype = torch.FloatTensor
    model.type(dtype)

    test_dir = "/Users/jacobazoulay/Repos/CS231N_Project/data/CDI/test/"
    pathlabel = "/Users/jacobazoulay/Repos/CS231N_Project/data/CDI/labels.json"
    mean_impath = "/Users/jacobazoulay/Repos/CS231N_Project/data/CDI/cache/im_mean.npy"
    std_impath = "/Users/jacobazoulay/Repos/CS231N_Project/data/CDI/cache/im_std.npy"

    im_mean = np.load(mean_impath)
    im_std = np.load(std_impath)

    labels = json.load(open(pathlabel))

    test_list = os.listdir(test_dir)
    test_list = [name[:-4] for name in test_list]

    imgs = torch.empty((len(test_list), 3, 128, 128))
    targets = torch.empty((len(test_list), 6))
    for i, name in enumerate(test_list):
        img = np.load((test_dir + name + '.npy'))
        img = torch.from_numpy(img)
        imgs[i] = img
        # print(imgs[i].shape)

        target = labels[name]
        target = torch.tensor(target)
        targets[i] = target


    # we will save the conv layer weights in this list
    model_weights = []
    # we will save the 49 conv layers in this list
    conv_layers = []
    # get all the model children as list
    model_children = list(model.children())

    # counter to keep count of the conv layers
    counter = 0
    # append all the conv layers and their respective wights to the list

    for idx in [0, 2]:
        for i in range(len(model_children[idx])):
            print(type(model_children[idx][i]))
            if type(model_children[idx][i]) == nn.Conv2d:
                counter += 1
                model_weights.append(model_children[idx][i].weight)
                conv_layers.append(model_children[idx][i])
            elif type(model_children[idx][i]) == nn.Sequential:
                for j in range(len(model_children[idx][i])):
                    for child in model_children[idx][i][j].children():
                        if type(child) == nn.Conv2d:
                            counter += 1
                            model_weights.append(child.weight)
                            conv_layers.append(child)
    print(f"Total convolution layers: {counter}")
    print("conv_layers")

    image = imgs[0]

    outputs = []
    names = []
    for layer in conv_layers[0:]:
        image = layer(image)
        outputs.append(image)
        names.append(str(layer))
    print(len(outputs))
    # print feature_maps
    for feature_map in outputs:
        print(feature_map.shape)

    processed = []
    for feature_map in outputs:
        feature_map = feature_map.squeeze(0)
        gray_scale = torch.sum(feature_map, 0)
        gray_scale = gray_scale / feature_map.shape[0]
        processed.append(gray_scale.data.numpy())
    for fm in processed:
        print(fm.shape)

    fig = plt.figure() #figsize=(30, 50))
    for i in range(12): #range(len(processed)):
        a = fig.add_subplot(5, 4, i + 1)
        imgplot = plt.imshow(processed[i])
        a.axis("off")
        a.set_title(names[i].split('(')[0] + ' ' + str(i + 1))
    plt.savefig(str('feature_maps.jpg'), bbox_inches='tight')
    plt.show()


if __name__ == "__main__":
    main()