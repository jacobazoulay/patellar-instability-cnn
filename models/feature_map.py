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


def class_visualization_update_step(img, model, target_y, l2_reg, learning_rate):
    out = model.forward(img)

    target = torch.sum(torch.square(out - target_y), dim=1) - l2_reg*torch.linalg.norm(img, dim=(1, 2, 3)).square()

    target.backward(torch.ones(target.shape))

    g = img.grad
    dX = learning_rate * g
    with torch.no_grad():
        img += dX
    img.grad.zero_()


def create_class_visualization(target_y, model, dtype, **kwargs):
    """
    Generate an image to maximize the score of target_y under a pretrained model.

    Inputs:
    - target_y: Integer in the range [0, 1000) giving the index of the class
    - model: A pretrained CNN that will be used to generate the image
    - dtype: Torch datatype to use for computations

    Keyword arguments:
    - l2_reg: Strength of L2 regularization on the image
    - learning_rate: How big of a step to take
    - num_iterations: How many iterations to use
    - blur_every: How often to blur the image as an implicit regularizer
    - max_jitter: How much to gjitter the image as an implicit regularizer
    - show_every: How often to show the intermediate result
    """
    model.type(dtype)
    l2_reg = kwargs.pop('l2_reg', 1e-3)
    learning_rate = kwargs.pop('learning_rate', 25)
    num_iterations = kwargs.pop('num_iterations', 1000)
    blur_every = kwargs.pop('blur_every', 10)
    max_jitter = kwargs.pop('max_jitter', 16)
    show_every = kwargs.pop('show_every', 250)

    # Randomly initialize the image as a PyTorch Tensor, and make it requires gradient.
    img = torch.rand((16, 3, 128, 128)).type(dtype)
    img = img * 2 - 1
    img = img.requires_grad_()

    for t in range(num_iterations):
        print(t, ' / ', num_iterations)
        # Randomly jitter the image a bit; this gives slightly nicer results
        ox, oy = random.randint(0, max_jitter), random.randint(0, max_jitter)
        img.data.copy_(jitter(img.data, ox, oy))
        class_visualization_update_step(img, model, target_y, l2_reg, learning_rate)
        # Undo the random jitter
        img.data.copy_(jitter(img.data, -ox, -oy))

        # As regularizer, clamp and periodically blur the image
        # for c in range(3):
        #     lo = float(-SQUEEZENET_MEAN[c] / SQUEEZENET_STD[c])
        #     hi = float((1.0 - SQUEEZENET_MEAN[c]) / SQUEEZENET_STD[c])
        #     img.data[:, c].clamp_(min=lo, max=hi)
        if t % blur_every == 0:
            blur_image(img.data, sigma=0.5)

        # Periodically show the image
        if t == 0 or (t + 1) % show_every == 0 or t == num_iterations - 1:
            N = img.shape[0]
            # N = 6

            show = img.data.clone().cpu()
            show = (show + 1) * 64
            show = torch.mean(show, dim=1)

            lab_show = target_y.data.clone().cpu()
            lab_show = (lab_show + 1) * 64

            for i in range(N):
                plt.subplot(2, N // 2, i + 1)
                plt.imshow(show[i], cmap='bone')
                plt.scatter(lab_show[i, 0], lab_show[i, 3], c='red', s=5)  # superior patella loc
                plt.scatter(lab_show[i, 1], lab_show[i, 4], c='red', s=5)  # inferior patella loc
                plt.scatter(lab_show[i, 2], lab_show[i, 5], c='red', s=5)  # tibial_plateau loc
                plt.axis('off')
                plt.title('Sample ' + str(i + 1))
                plt.gcf().set_size_inches(12, 5)
            plt.show()

    return (img.data.cpu() + 1) * 64


def blur_image(X, sigma=1.0):
    X_np = X.cpu().clone().numpy()
    X_np = gaussian_filter1d(X_np, sigma, axis=2)
    X_np = gaussian_filter1d(X_np, sigma, axis=3)
    X.copy_(torch.Tensor(X_np).type_as(X))
    return X


def jitter(X, ox, oy):
    """
    Helper function to randomly jitter an image.

    Inputs
    - X: PyTorch Tensor of shape (N, C, H, W)
    - ox, oy: Integers giving number of pixels to jitter along W and H axes

    Returns: A new PyTorch Tensor of shape (N, C, H, W)
    """
    if ox != 0:
        left = X[:, :, :, :-ox]
        right = X[:, :, :, -ox:]
        X = torch.cat([right, left], dim=3)
    if oy != 0:
        top = X[:, :, :-oy]
        bottom = X[:, :, -oy:]
        X = torch.cat([bottom, top], dim=2)
    return X


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
        print(img.shape)
        imgs[i] = img
        # print(imgs[i].shape)

        target = labels[name]
        target = torch.tensor(target)
        targets[i] = target

    print(targets.shape)
    print(imgs.shape)

    out = model.forward(imgs)

    print(out.shape)

    targets = (targets + 1) * 64
    out = (out + 1) * 64

    import pandas as pd

    gts_np = targets.detach().numpy()  # convert to Numpy array
    df = pd.DataFrame(gts_np)  # convert to a dataframe
    df.to_csv("gts", index=False)  # save to file

    preds_np = out.detach().numpy()  # convert to Numpy array
    df = pd.DataFrame(preds_np)  # convert to a dataframe
    df.to_csv("preds", index=False)  # save to file

    df = pd.DataFrame({"data_ids": list(range(0, len(gts_np))) + list(range(0, len(gts_np))),
                       "judge":  ['gt']*len(gts_np) + ['pred']*len(gts_np),
                       "CDI": gts_np.tolist() + preds_np.tolist()})
    # print(df)
    # iccs = pg.intraclass_corr(data=df, targets='data_ids', raters='judge', ratings='CDI')
    # icc = iccs['ICC'][5] # ICC2 Single random raters (absolute distance between cdis), -ve values indicate poor reliability
    # print(iccs)

    imgs = imgs * im_std + im_mean

    # for i, im in enumerate([45, 50, 40, 32]):
    for i, im in enumerate([0, 5, 48, 54]):
        plt.subplot(2, 2, i + 1)
        plt.imshow(imgs[im, 0], cmap='bone')
        plt.scatter(gts_np[im, 0], gts_np[im, 3], c='green', s=8)  # superior patella loc
        plt.scatter(gts_np[im, 1], gts_np[im, 4], c='green', s=8)  # inferior patella loc
        plt.scatter(gts_np[im, 2], gts_np[im, 5], c='green', s=8)  # tibial_plateau loc

        plt.scatter(preds_np[im, 0], preds_np[im, 3], c='red', s=8)  # superior patella loc
        plt.scatter(preds_np[im, 1], preds_np[im, 4], c='red', s=8)  # inferior patella loc
        plt.scatter(preds_np[im, 2], preds_np[im, 5], c='red', s=8)  # tibial_plateau loc

        d1 = np.sqrt(np.square(gts_np[im, 0] - preds_np[im, 0]) + np.square(gts_np[im, 1] - preds_np[im, 1]))
        d2 = np.sqrt(np.square(gts_np[im, 2] - preds_np[im, 2]) + np.square(gts_np[im, 3] - preds_np[im, 3]))
        d3 = np.sqrt(np.square(gts_np[im, 4] - preds_np[im, 4]) + np.square(gts_np[im, 5] - preds_np[im, 5]))
        out = np.mean((d1 + d2 + d3) / 3)
        out = 100 * (out / 128)

        plt.title('Avg Pixel Error: ' + str("%.2f" % round(float(out), 2)) + '%')
        plt.axis('off')
    plt.show()

    # for i, im in enumerate([59, 29, 2, 23]):
    for i, im in enumerate([15, 30, 53, 42]):
        plt.subplot(2, 2, i + 1)
        plt.imshow(imgs[im, 0], cmap='bone')
        plt.scatter(gts_np[im, 0], gts_np[im, 3], c='green', s=8)  # superior patella loc
        plt.scatter(gts_np[im, 1], gts_np[im, 4], c='green', s=8)  # inferior patella loc
        plt.scatter(gts_np[im, 2], gts_np[im, 5], c='green', s=8)  # tibial_plateau loc

        plt.scatter(preds_np[im, 0], preds_np[im, 3], c='red', s=8)  # superior patella loc
        plt.scatter(preds_np[im, 1], preds_np[im, 4], c='red', s=8)  # inferior patella loc
        plt.scatter(preds_np[im, 2], preds_np[im, 5], c='red', s=8)  # tibial_plateau loc

        d1 = np.sqrt(np.square(gts_np[im, 0] - preds_np[im, 0]) + np.square(gts_np[im, 1] - preds_np[im, 1]))
        d2 = np.sqrt(np.square(gts_np[im, 2] - preds_np[im, 2]) + np.square(gts_np[im, 3] - preds_np[im, 3]))
        d3 = np.sqrt(np.square(gts_np[im, 4] - preds_np[im, 4]) + np.square(gts_np[im, 5] - preds_np[im, 5]))
        out = np.mean((d1 + d2 + d3) / 3)
        out = 100 * (out / 128)

        plt.title('Avg Pixel Error: ' + str(round(float(out), 2)) + '%')
        plt.axis('off')
    plt.show()



if __name__ == "__main__":
    main()