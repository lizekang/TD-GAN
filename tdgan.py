import argparse
import os
import numpy as np
import time
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.autograd import Variable

import torchvision.transforms as transforms

from tensorboardX import SummaryWriter


parser = argparse.ArgumentParser()
parser.add_argument('--train_step', type=int, default=100000, help='epoch to start training from')
parser.add_argument('--test_step', type=int, default=10, help='epoch to start training from')
parser.add_argument('--data_path', type=str, default="3dletter_list.txt", help='number of epochs of training')
parser.add_argument('--batch_size', type=int, default=8, help='size of the batches')
parser.add_argument('--retrain', type=bool, default=False, help='if retrain')
parser.add_argument('--lr', type=float, default=0.0002, help='adam: learning rate')
parser.add_argument('--decay_epoch', type=int, default=100, help='epoch from which to start lr decay')
parser.add_argument('--n_cpu', type=int, default=8, help='number of cpu threads to use during batch generation')
parser.add_argument('--img_height', type=int, default=32, help='size of image height')
parser.add_argument('--img_width', type=int, default=32, help='size of image width')
parser.add_argument('--channels', type=int, default=27, help='number of image channels')
parser.add_argument('--sample_interval', type=int, default=100,
                    help='interval between sampling of images from generators')
parser.add_argument('--checkpoint_interval', type=int, default=100, help='interval between model checkpoints')
parser.add_argument('--summary_path', type=str, default='./summary', help='path for summary')
parser.add_argument('--n_critic', type=int, default=2, help='number of training iterations for WGAN discriminator')
opt = parser.parse_args()
print(opt)

writer = SummaryWriter(opt.summary_path)
