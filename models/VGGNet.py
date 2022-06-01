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
import torchvision.models as models
import os
import json
import pingouin as pg
import pandas as pd
from shared.data_utils import un_norm_avg_key_dist
from shared.data_utils import un_standard_avg_key_dist
from shared.data_utils import compute_CDI

from common import weights_init, FullModel


def VGGNet_setup(args):
    #setup path to save experiment results
    args.odir = 'results/%s/%s' % (args.dataset, args.net)
    args.odir += '_b%d' % args.batch_size


def VGGNet_create_model(args):
    """ Creates model """
    vgg16 = models.vgg16(pretrained=True)
    vgg16.classifier[6] = nn.Linear(4096, args.num_classes)
    model = vgg16

    args.gpus = list(range(torch.cuda.device_count()))
    args.nparams = sum([p.numel() for p in model.parameters()])
    print('Total number of parameters: {}'.format(args.nparams))
    criterion = nn.MSELoss()
    model = FullModel(model, criterion)
    if len(args.gpus) > 0:
        model = nn.DataParallel(model, device_ids=args.gpus).cuda()
    else:
        model = nn.DataParallel(model, device_ids=args.gpus)
    #model.apply(weights_init)
    return model


def VGGNet_step(args, item):
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
    #repeat input along channel dimensions since pre-trained VGG takes RGB
    inp = inp.repeat([1,3,1,1])
    loss, pred = args.model.forward(inp, targets)

    loss = loss.mean()

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

    losses = [loss, avg_keypoint_dist, icc, mean_abs_CDI_dist_np]
    outputs = [pred]
    loss_names = ['loss','avg_keypt_dist', 'avg_abs_icc', 'avg_abs_CDI_dist']
    return loss_names, losses, outputs

