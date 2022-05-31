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


def show_saliency_maps(X, y, model, n=6):
    # Convert X and y from numpy arrays to Torch Tensors
    # X_tensor = torch.cat([x for x in X], dim=0)
    # y_tensor = torch.tensor(y)

    # Compute saliency maps for images in X
    saliency = compute_saliency_maps(X, y, model)

    # Convert the saliency map from Torch Tensor to numpy array and show images
    # and saliency maps together.
    saliency = saliency.numpy()
    N = X.shape[0]
    N=n
    for i in range(N):
        plt.subplot(2, N, i + 1)
        show = X[i].data.clone().cpu()
        plt.imshow(show[0], cmap='bone')
        plt.axis('off')
        plt.title('Sample ' + str(i+1))
        plt.subplot(2, N, N + i + 1)
        plt.imshow(saliency[i], cmap=plt.cm.hot)
        plt.axis('off')
        plt.gcf().set_size_inches(12, 5)
    plt.show()


def compute_saliency_maps(X, y, model):
    """
    Compute a class saliency map using the model for images X and labels y.

    Input:
    - X: Input images; Tensor of shape (N, 3, H, W)
    - y: Labels for X; LongTensor of shape (N,)
    - model: A pretrained CNN that will be used to compute the saliency map.

    Returns:
    - saliency: A Tensor of shape (N, H, W) giving the saliency maps for the input
    images.
    """
    # Make sure the model is in "model_params" mode
    model.eval()

    # Make input tensor require gradient
    X.requires_grad_()

    out = model(X)
    out_correct = torch.sum(torch.square(out - y), dim=1)

    out_correct.backward(torch.ones(out_correct.shape))

    saliency = X.grad
    saliency = saliency.abs()
    saliency, _ = torch.max(saliency, dim=1)

    return saliency


def main():
    VGGNet = True

    if VGGNet:
        PATH = "/Users/jacobazoulay/Repos/Total Results/results/CDI/VGGNet_b8/models/model_600.pth.tar"

        vgg16 = models.vgg16(pretrained=True)
        vgg16.classifier[6] = nn.Linear(4096, 6)
        model = vgg16


    # Load model parameters
    checkpoint = torch.load(PATH, map_location=torch.device('cpu'))
    model_state = checkpoint['state_dict']
    model_params = model_state.copy()

    for key in model_state.keys():
        model_params[key.replace('module.model.', '')] = model_state[key]
        del model_params[key]

    model.load_state_dict(model_params)
    model.eval()


    # Load random image data and labels
    dtype = torch.FloatTensor
    model.type(dtype)

    img_odir = "/Users/jacobazoulay/Repos/CS231N_Project/data/CDI/train/"
    pathlabel = "/Users/jacobazoulay/Repos/CS231N_Project/data/CDI/labels.json"
    labels = json.load(open(pathlabel))
    label_no_aug = [key for key in list(labels.keys()) if 'aug' not in key]
    img_names = random.choices(label_no_aug, k=16)

    imgs = torch.empty((16, 3, 128, 128))
    targets = torch.empty((16, 6))
    for i, name in enumerate(img_names):
        try:
            img = np.load(img_odir + name + '.npy')
        except:
            try:
                img = np.load((img_odir + name + '.npy').replace('train', 'test'))
            except:
                img = np.load((img_odir + name + '.npy').replace('train', 'val'))

        img = torch.from_numpy(img)
        imgs[i] = img

        target = labels[name]
        target = torch.tensor(target)
        targets[i] = target

    show_saliency_maps(imgs, targets, model)


if __name__ == "__main__":
    main()