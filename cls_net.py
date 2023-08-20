import torch
import torch.nn as nn
import torch.optim as optim
import argparse
from torch.utils.data import Dataset,DataLoader
from torch.autograd import Variable
from tensorboardX import SummaryWriter
import numpy as np
import math
import os
import glob
import torch.nn.functional as F
import torch
import cv2
from torch.utils.data import Dataset,DataLoader
from torch.autograd import Variable
from tensorboardX import SummaryWriter
import math
import os
import torchvision.models as models
import basic_net as basic_net
from ResCBAM import *
import torch.cuda

device_ids = [0]

"----------------CLS Net--------------------------------------"
class clsNet(nn.Module):

    def __init__(self, classes_num=5, basic_model='resnet',pretrain=True):
        super(clsNet, self).__init__()
        self.basic_model=basic_model
        if self.basic_model == 'resnet':
            self.resNet1 = basic_net.resnet34(pretrained=pretrain)
            self.resNet = list(self.resNet1.children())[:-2]
            self.features = nn.Sequential(*self.resNet)
            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
            self.fc = nn.Linear(512, classes_num)
        elif self.basic_model == 'inception':
            self.inception1 = basic_net.inception_v3(pretrained=pretrain)
            self.inception = list(self.inception1.children())
            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
            self.fc = nn.Linear(2048, classes_num)
        elif self.basic_model == 'densenet':
            self.densenet1 = basic_net.densenet201(pretrained=pretrain)
            self.densenet = list(self.densenet1.children())
            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
            self.fc = nn.Linear(1920, classes_num)
        elif self.basic_model == 'resnet_cbam':
            self.resNet1 = resnet34_cbam(pretrained=pretrain)
            self.resNet = list(self.resNet1.children())[:-2]
            self.features = nn.Sequential(*self.resNet)
            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
            self.fc = nn.Linear(10752, classes_num)
        elif self.basic_model == 'ECA':
            self.resNet1 = basic_net.ECA_ResNet50(3, classes_num)
            self.resNet = list(self.resNet1.children())[:-2]
            self.features = nn.Sequential(*self.resNet)
            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
            self.fc = nn.Linear(2048, classes_num)


    def embed(self, x):
        x = self.features(x)
        x = self.spatial_pyramid_pool(x, x.size(0), [int(x.size(2)),int(x.size(3))], [4,2,1])
        x = F.normalize(x)
        return x.squeeze()

    def spatial_pyramid_pool(self, previous_conv, num_sample, previous_conv_size, out_pool_size):

        for i in range(len(out_pool_size)):
            h_wid = int(math.ceil(previous_conv_size[0] / out_pool_size[i]))
            w_wid = int(math.ceil(previous_conv_size[1] / out_pool_size[i]))
            h_pad = int((h_wid * out_pool_size[i] - previous_conv_size[0] + 1) / 2)
            w_pad = int((w_wid * out_pool_size[i] - previous_conv_size[1] + 1) / 2)
            maxpool = nn.MaxPool2d((h_wid, w_wid), stride=(h_wid, w_wid), padding=(h_pad, w_pad))
            x = maxpool(previous_conv)
            if (i == 0):
                spp = x.view(num_sample, -1)
            else:
                spp = torch.cat((spp, x.view(num_sample, -1)), 1)
        return spp

    def forward(self, x):
        if self.basic_model == 'resnet':
            x = self.features(x)  # 512
            x = self.avgpool(x)
            x = x.view(x.size(0), -1)
            x = self.fc(x)
            return x
        elif self.basic_model == 'inception':
            x = self.inception[0](x)
            x = self.inception[1](x)
            x = self.inception[2](x)
            x = F.max_pool2d(x, kernel_size=3, stride=2)
            x = self.inception[3](x)
            x = self.inception[4](x)
            x = F.max_pool2d(x, kernel_size=3, stride=2)
            x = self.inception[5](x)
            x = self.inception[6](x)
            x = self.inception[7](x)
            x = self.inception[8](x)
            x = self.inception[9](x)
            x = self.inception[10](x)
            x = self.inception[11](x)
            x = self.inception[12](x)
            x = self.inception[14](x)
            x = self.inception[15](x)
            x = self.inception[16](x)
            x = F.adaptive_avg_pool2d(x, (1, 1))
            x = F.dropout(x, training=self.training)
            x = x.view(x.size(0), -1)
            x = self.fc(x)
            return x
        elif self.basic_model == 'densenet':
            features = self.densenet[0](x)  # 512
            out = F.relu(features, inplace=True)
            out = F.adaptive_avg_pool2d(out, (1, 1)).view(features.size(0), -1)
            out = self.fc(out)
            return out
        elif self.basic_model == 'resnet_cbam':
            x = self.features(x)
            x = self.spatial_pyramid_pool(x, x.size(0), [int(x.size(2)),int(x.size(3))], [4,2,1])
            x = x.view(x.size(0), -1)
            x = self.fc(x)
            return x
        elif self.basic_model == 'ECA':
            x = self.features(x)  # 512
            x = self.avgpool(x)
            x = x.view(x.size(0), -1)
            x = self.fc(x)
            return x



if __name__ == "__main__":

    x = torch.randn((4, 3, 512, 512))
    net = clsNet(classes_num=5, basic_model='resnet_cbam', pretrain=False)
    print(net(x).shape)

