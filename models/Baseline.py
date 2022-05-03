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


def BaseNet_setup(args):
    args.odir = 'results/%s/%s' % (args.dataset, args.net)
    args.odir += '_b%d' % args.batch_size


def BaseNet_create_model(args):
    """ Creates model """
    model = BaseNet(args)
    args.gpus = list(range(torch.cuda.device_count()))
    args.nparams = sum([p.numel() for p in model.parameters()])
    print('Total number of parameters: {}'.format(args.nparams))
    criterion = nn.CrossEntropyLoss()
    model = FullModel(model, criterion)
    """
    if len(args.gpus) > 0:
        model = nn.DataParallel(model, device_ids=args.gpus).cuda()
    else:
        model = nn.DataParallel(model, device_ids=args.gpus)
    """
    model.apply(weights_init)
    return model


def BaseNet_step(args, item):
    imgs, gt, meta = item
    n,x,y,C = imgs.shape
    args.gpus = list(range(torch.cuda.device_count()))
    if len(args.gpus) > 0:
        targets = Variable(torch.from_numpy(gt), requires_grad=False).long().cuda()
        inp = Variable(torch.from_numpy(imgs), requires_grad=False).float().cuda()
    else:
        targets = Variable(torch.from_numpy(gt), requires_grad=False).long()
        inp = Variable(torch.from_numpy(imgs), requires_grad=False).float()

    targets = targets.contiguous()
    inp = inp.transpose(3,2).transpose(2,1)/255.0
    loss, pred = args.model.forward(inp,targets)
    loss = loss.mean()
    acc = np.mean(torch.argmax(pred,1).detach().cpu().numpy()==gt) * 100
    losses = [loss, acc]
    outputs = [pred]
    loss_names = ['loss','acc']
    return loss_names, losses, outputs

class BaseNet(nn.Module):
    def __init__(self, args):
        super(BaseNet, self).__init__()
        #self.conv1 = nn.Conv2d(3, 8, kernel_size=3, stride=2)
        #self.conv2 = nn.Conv2d(8, 16, kernel_size=3, stride=2)
        #self.conv3 = nn.Conv2d(16, 8, kernel_size=3, stride=2)
        self.args = args
        self.fc1 = nn.Linear(3*64*64, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, args.num_classes)

    def forward(self, x):
        """
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        
        x = x.view(-1, 8*31*31)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = self.fc5(x)
        """
       
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        return F.softmax(x, dim=1)