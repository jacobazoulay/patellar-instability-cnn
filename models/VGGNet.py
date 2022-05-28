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
from shared.data_utils import un_norm_avg_key_dist
from shared.data_utils import un_standard_avg_key_dist

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
    if len(args.gpus) > 0:
        targets = Variable(torch.from_numpy(gt), requires_grad=False).float().cuda()
        inp = Variable(torch.from_numpy(imgs), requires_grad=False).float().cuda()
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
    # label_cache_stats = json.load(open(os.path.join(os.getcwd(), "data/CDI/cache/label_stats.json")))
    # label_mean = label_cache_stats['label_mean']
    # label_std = label_cache_stats['label_std']
    # avg_keypoint_dist = un_norm_avg_key_dist(label_mean, label_std, targets, pred)
    avg_keypoint_dist = un_standard_avg_key_dist(targets, pred)
    if len(args.gpus) > 0:
        avg_keypoint_dist = avg_keypoint_dist.detach().cpu().numpy()

    losses = [loss, avg_keypoint_dist]
    outputs = [pred]
    loss_names = ['loss','avg_keypt_dist']
    return loss_names, losses, outputs

