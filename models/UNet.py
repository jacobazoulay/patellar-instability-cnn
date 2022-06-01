# ------------------------------------------------------------------------------
#Copyright 2021, Micael Tchapmi, All rights reserved
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
import torch._utils
from torch.autograd import Variable
from torch.nn import functional as F
import numpy as np
import os
import json
import pingouin as pg
import pandas as pd
from shared.data_utils import un_standard_avg_key_dist
from shared.data_utils import compute_CDI

from common import weights_init, FullModel


def UNet_setup(args):
    #setup path to save experiment results
    args.odir = 'results/%s/%s' % (args.dataset, args.net)
    args.odir += '_b%d' % args.batch_size


def UNet_create_model(args):
    """ Creates model """
    model = UNet(args)
    args.gpus = list(range(torch.cuda.device_count()))
    args.nparams = sum([p.numel() for p in model.parameters()])
    print('Total number of parameters: {}'.format(args.nparams))
    if len(args.gpus) > 0:
        model = model.cuda()
    return model


def UNet_step(args, item):
    #compute forward pass
    imgs, gt, meta = item
    n,x,y = imgs.shape
    imgs = imgs.reshape(n,x,y,1)
    n,x,y,c = imgs.shape
    args.gpus = list(range(torch.cuda.device_count()))
    device = 'cpu'
    if len(args.gpus) > 0:
        targets = Variable(torch.from_numpy(gt), requires_grad=False).float().cuda()
        inp = Variable(torch.from_numpy(imgs), requires_grad=False).float().cuda()
        device = 'cuda'
    else:
        targets = Variable(torch.from_numpy(gt), requires_grad=False).float()
        inp = Variable(torch.from_numpy(imgs), requires_grad=False).float()
    targets = targets.contiguous()

    inp = inp.transpose(3,2).transpose(2,1)
    code, out_img, pred = args.model.forward(inp)

    criterion = nn.MSELoss()
    reconstruction_loss = criterion(out_img, inp)
    pred_loss = criterion(pred, targets)
    loss = (reconstruction_loss + pred_loss) / 2.0

    #compute metrics
    avg_keypoint_dist = un_norm_avg_key_dist(args.label_mean, args.label_std, targets, pred, device)
    avg_keypoint_dist = un_standard_avg_key_dist(targets, pred)
    if len(args.gpus) > 0:
        avg_keypoint_dist = avg_keypoint_dist.detach().cpu().numpy()

    #un_normalize pred and targets
    gt_unorm = (targets + 1) * 64
    pred_unorm = (pred + 1) * 64

    # compute CDI
    pred_superior_patella_x, pred_inferior_patella_x, pred_tibial_plateau_x, pred_superior_patella_y, pred_inferior_patella_y, pred_tibial_plateau_y = pred_unorm[:,0], pred_unorm[:,1], pred_unorm[:,2], pred_unorm[:,3], pred_unorm[:,4], pred_unorm[:,5]
    gt_superior_patella_x, gt_inferior_patella_x, gt_tibial_plateau_x, gt_superior_patella_y, gt_inferior_patella_y, gt_tibial_plateau_y = gt_unorm[:,0], gt_unorm[:,1], gt_unorm[:,2], gt_unorm[:,3], gt_unorm[:,4], gt_unorm[:,5]
    pred_CDI = compute_CDI(pred_superior_patella_x, pred_inferior_patella_x, pred_tibial_plateau_x, pred_superior_patella_y, pred_inferior_patella_y, pred_tibial_plateau_y)
    gt_CDI = compute_CDI(gt_superior_patella_x, gt_inferior_patella_x, gt_tibial_plateau_x, gt_superior_patella_y, gt_inferior_patella_y, gt_tibial_plateau_y)
    # mean absolute CDI distance btwn gt and pred CDI
    mean_abs_CDI_dist = torch.sum(torch.abs(gt_CDI - pred_CDI)) / len(pred_CDI)

    # compute CDI Intra class correlation(ICC) coefficient
    # ICC can tell you how close 2 sets of labels (e.g., human vs human or model vs human) are to a y=x fit. Ideally, ICC=1.
    # Between two human raters (benchmark values that you're trying to beat): CDI ICC - 0.644
    pred_CDI_np = pred_CDI
    gt_CDI_np = gt_CDI
    if len(args.gpus) > 0:
        pred_CDI_np = pred_CDI.detach().cpu().numpy()
        gt_CDI_np = gt_CDI.detach().cpu().numpy()
    df = pd.DataFrame({"data_ids": list(range(0, len(inp))) + list(range(0, len(inp))),
                       "judge":  ['gt']*len(inp) + ['pred']*len(inp),
                       "CDI": gt_CDI_np.tolist() + pred_CDI_np.tolist()})
    iccs = pg.intraclass_corr(data=df, targets='data_ids', raters='judge', ratings='CDI')
    icc = iccs['ICC'][2] # Single fixed raters (absolute distance between cdis), -ve values indicate poor reliability

    mean_abs_CDI_dist_np = mean_abs_CDI_dist
    if len(args.gpus) > 0:
        mean_abs_CDI_dist_np = mean_abs_CDI_dist.detach().cpu().numpy()

    losses = [loss, avg_keypoint_dist, icc, mean_abs_CDI_dist_np, reconstruction_loss, pred_loss]
    outputs = [pred]
    loss_names = ['loss','avg_keypt_dist', 'avg_abs_icc', 'avg_abs_CDI_dist', 'rec_loss', 'kpt_loss']
    return loss_names, losses, outputs


# UNet architecture
class UNet(nn.Module):
    def __init__(self, args, n_channels=1, n_classes=1, bilinear=False):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)
        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc = OutConv(64, n_classes)

        self.classifier = nn.Sequential(
            nn.Linear(1024 * 8 * 8, 4096),
            nn.ReLU(),
            nn.Linear(4096, 1024),
            nn.ReLU(),
            nn.Linear(1024, 256),
            nn.ReLU(),
            nn.Linear(256, args.num_classes),
        )

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)

        # use embedding to predict keypoints
        pred = self.classifier(x5.flatten(start_dim=1))

        return x5, logits, pred


#UNet Parts
class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)
