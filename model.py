import torch
import torch.nn as nn


class DisentanglingNetR(nn.Module):
    def __init__(self, img_shape=(3, 128, 128)):
        super(DisentanglingNetR, self).__init__()
        self.channel = img_shape[0]
        self.img_height = img_shape[1]
        self.img_width = img_shape[2]

        self.convs = nn.Sequential([
            nn.Conv2d(self.channel, 64, (3, 3), stride=2),
            nn.LeakyReLU(inplace=True),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 128, (3, 3), stride=2),
            nn.LeakyReLU(inplace=True),
            nn.BatchNorm2d(128),
            nn.Conv2d(128, 256, (3, 3), stride=2),
            nn.LeakyReLU(inplace=True),
            nn.BatchNorm2d(256),
            nn.Conv2d(256, 256, (3, 3), stride=1),
            nn.LeakyReLU(inplace=True),
            nn.BatchNorm2d(256),
            nn.Conv2d(256, 256, (3, 3), stride=1),
            nn.LeakyReLU(inplace=True),
            nn.BatchNorm2d(256),
            nn.Conv2d(256, 512, (3, 3), stride=2),
            nn.LeakyReLU(inplace=True),
            nn.BatchNorm2d(512),
            nn.Conv2d(512, 512, (3, 3), stride=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(512)
        ])

        self.fc1 = nn.Sequential([
            nn.Linear(32768, 5120),
            nn.LeakyReLU(inplace=True)
        ])
        self.identity = nn.Sequential([
            nn.Linear(5120, 1024),
            nn.Tanh()
        ])
        self.viewpoint = nn.Sequential([
            nn.Linear(5120, 512),
            nn.Tanh()
        ])

    def forward(self, img):
        x = self.convs(img)
        x = x.view(x.size(0), 1)
        x = self.fc1(x)
        view = self.viewpoint(x)
        identity = self.identity(x)
        return [view, identity]


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.fc = nn.Sequential([
            nn.Linear(1536, 16384),
            nn.ReLU(inplace=True)
        ])
        self.bn = nn.BatchNorm2d(1024)
        self.deconvs = nn.Sequential([
            nn.ConvTranspose2d(1024, 512, (4, 4), stride=2),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(512),
            nn.ConvTranspose2d(512, 256, (4, 4), stride=2),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(256),
            nn.ConvTranspose2d(256, 128, (4, 4), stride=2),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(128),
            nn.ConvTranspose2d(128, 64, (4, 4), stride=2),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(64),
            nn.ConvTranspose2d(64, 3, (4, 4), stride=2),
            nn.Tanh()
        ])

    def forward(self, representations):
        x = torch.cat(representations, 1)
        x = self.fc(x)
        x.view(x.size(0), 1024, 4, 4)
        x = self.bn(x)
        x = self.deconvs(x)
        return x


class Discriminator(nn.Module):
    def __init__(self, img_shape=(3, 128, 128)):
        super(Discriminator, self).__init__()
        self.channel = img_shape[0]
        self.img_height = img_shape[1]
        self.img_width = img_shape[2]

        self.convs = nn.Sequential([
            nn.Conv2d(self.channel, 64, (3, 3), stride=2),
            nn.LeakyReLU(inplace=True),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 128, (3, 3), stride=2),
            nn.LeakyReLU(inplace=True),
            nn.BatchNorm2d(128),
            nn.Conv2d(128, 256, (3, 3), stride=2),
            nn.LeakyReLU(inplace=True),
            nn.BatchNorm2d(256),
            nn.Conv2d(256, 256, (3, 3), stride=1),
            nn.LeakyReLU(inplace=True),
            nn.BatchNorm2d(256),
            nn.Conv2d(256, 256, (3, 3), stride=1),
            nn.LeakyReLU(inplace=True),
            nn.BatchNorm2d(256),
            nn.Conv2d(256, 512, (3, 3), stride=2),
            nn.LeakyReLU(inplace=True),
            nn.BatchNorm2d(512),
            nn.Conv2d(512, 512, (3, 3), stride=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(512)
        ])
        self.fc1 = nn.Sequential([
            nn.Linear(32768, 2560),
            nn.LeakyReLU(inplace=True)
        ])
        self.fc2 = nn.Sequential([
            nn.Linear(2560, 1)
        ])

    def forward(self, img):
        x = self.convs(img)
        x = x.view(x.size(0), 1)
        x = self.fc1(x)
        logits = self.fc2(x)
        return logits


class TagMappingNet(nn.Module):
    def __init__(self):
        super(TagMappingNet, self).__init__()
        self.fc_identity = nn.Sequential([
            nn.Linear(500, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 1024),
            nn.Tanh()
        ])
        self.fc_view = nn.Sequential([
            nn.Linear(31, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 512),
            nn.Tanh()
        ])

    def forward(self, identity, view):
        identity = self.fc_identity(identity)
        view = self.fc_view(view)
        return [identity, view]

