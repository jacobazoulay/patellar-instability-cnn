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

from common import weights_init, FullModel
#from Model import KeypointModel as CDINet 
from ModelPreTrained import KeypointPretrainedModel as CDINet #modifying this to use pretrained model



def CDINet_setup(args):
    #setup path to save experiment results
    args.odir = 'results/%s/%s' % (args.dataset, args.net)
    args.odir += '_b%d' % args.batch_size


def CDINet_create_model(args):
    """ Creates model """
    model = CDINet(args)
    args.gpus = list(range(torch.cuda.device_count()))
    args.nparams = sum([p.numel() for p in model.parameters()])
    print('Total number of parameters: {}'.format(args.nparams))
    criterion = nn.MSELoss()
    model = FullModel(model, criterion)
    """
    if len(args.gpus) > 0:
        model = nn.DataParallel(model, device_ids=args.gpus).cuda()
    else:
        model = nn.DataParallel(model, device_ids=args.gpus)
    """
    model.apply(weights_init)
    return model


def CDINet_step(args, item):
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
    loss, pred = args.model.forward(inp, targets)
    loss = loss.mean()

    #compute metrics
    sqrt_dist = torch.sum((targets - pred)**2, axis=0)
    avg_keypoint_dist = torch.sqrt(torch.mean(sqrt_dist))

    losses = [loss, avg_keypoint_dist]
    outputs = [pred]
    loss_names = ['loss','avg_keypt_dist']
    return loss_names, losses, outputs
