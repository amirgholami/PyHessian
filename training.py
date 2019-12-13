#*
# @file Different utility functions
# Copyright (c) Zhewei Yao, Amir Gholami
# All rights reserved.
# This file is part of PyHessian library.
#
# PyHessian is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# PyHessian is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with PyHessian.  If not, see <http://www.gnu.org/licenses/>.
#*

from __future__ import print_function
import logging
import os
import sys

import numpy as np
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable

from utils import *
from models.resnet import resnet
from tqdm import tqdm, trange

# Training settings
parser = argparse.ArgumentParser(description='Training on Cifar10')

parser.add_argument('--batch-size',
                    type=int,
                    default=128,
                    metavar='N',
                    help='input batch size for training (default: 128)')
parser.add_argument('--test-batch-size',
                    type=int,
                    default=256,
                    metavar='N',
                    help='input batch size for testing (default: 256)')
parser.add_argument('--epochs',
                    type=int,
                    default=180,
                    metavar='N',
                    help='number of epochs to train (default: 10)')

parser.add_argument('--lr',
                    type=float,
                    default=0.1,
                    metavar='LR',
                    help='learning rate (default: 0.01)')
parser.add_argument('--lr-decay',
                    type=float,
                    default=0.1,
                    help='learning rate ratio')
parser.add_argument('--lr-decay-epoch',
                    type=int,
                    nargs='+',
                    default=[80, 120],
                    help='Decrease learning rate at these epochs.')

parser.add_argument('--seed',
                    type=int,
                    default=1,
                    metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--weight-decay',
                    default=5e-4,
                    type=float,
                    metavar='W',
                    help='weight decay (default: 1e-4)')
parser.add_argument('--batch-norm',
                    action='store_false',
                    help='do we need batch norm or not')
parser.add_argument('--residual',
                    action='store_false',
                    help='do we need residula connect or not')

parser.add_argument('--cuda',
                    action='store_false',
                    help='do we use gpu or not')
parser.add_argument('--saving-folder',
                    type=str,
                    default='checkpoints/',
                    help='choose saving name')

args = parser.parse_args()

# set random seed to reproduce the work
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

for arg in vars(args):
    print(arg, getattr(args, arg))

# get dataset
train_loader, test_loader = getData(name='cifar10',
                                    train_bs=args.batch_size,
                                    test_bs=args.test_batch_size)

# get model and optimizer
model = resnet(num_classes=10,
               depth=20,
               residual_not=args.residual,
               batch_norm_not=args.batch_norm)
if args.cuda:
    model = model.cuda()
model = torch.nn.DataParallel(model)

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(),
                      lr=args.lr,
                      momentum=0.9,
                      weight_decay=args.weight_decay)
lr_scheduler = optim.lr_scheduler.MultiStepLR(optimizer,
                                              args.lr_decay_epoch,
                                              gamma=args.lr_decay)

if not os.path.isdir(args.saving_folder):
    os.makedirs(args.saving_folder)

for epoch in range(1, args.epochs + 1):
    print('Current Epoch: ', epoch)
    train_loss = 0.
    total_num = 0
    correct = 0
    with tqdm(total=len(train_loader.dataset)) as progressbar:

        for batch_idx, (data, target) in enumerate(train_loader):

            model.train()
            if args.cuda:
                data, target = data.cuda(), target.cuda()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            train_loss += loss.item() * target.size()[0]
            total_num += target.size()[0]
            _, predicted = output.max(1)
            correct += predicted.eq(target).sum().item()
            optimizer.step()
            optimizer.zero_grad()

            progressbar.set_postfix(loss=train_loss / total_num,
                                    acc=100. * correct / total_num)

            progressbar.update(target.size(0))

    acc = test(model, test_loader)
    lr_scheduler.step()

torch.save(model.state_dict(), args.saving_folder + 'net.pkl')
