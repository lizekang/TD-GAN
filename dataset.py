import torch
import torch.nn as nn
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import os


class TDGANDataset(Dataset):
    def __init__(self, data_path=None, transforms=None):
        if transforms is not None:
            self.transforms = transforms

    def __getitem__(self, item):
        pass

    def __len__(self):
        pass


