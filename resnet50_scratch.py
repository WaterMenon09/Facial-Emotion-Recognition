import torch
import torch.nn as nn
import random

class Resnet_scratch(nn.Module):
    def __init__(self):
        super(Resnet_scratch,self).__init__()
        self.meta = {
            "mean": [131.0912,103.8827,91.4953],
            "std": [1,1,1],
            "imageSize": [244,244,3],
        }
        self.conv1_7x7_s2 = nn.Conv2d(
            in_channels=3,out_channels=64,
            kernel_size=[7,7],
            stride=(2,2),
            padding=(3,3),
            bias=False
        )
        self.conv1_7x7_bn = nn.BatchNorm2d(
            num_features=64, 
            eps=0.00001,
            momentum=0.1,
            affine=True, 
            track_running_stats=True
        )
        self.conv1_relu_7x7_s2 = nn.ReLU()
        self.pool1_3x3_s2 = nn.MaxPool2d(
            kernel_size=[3,3],
            stride=[2,2],
            padding=(0,0),
            dilation=1,
            ceil_mode=True
        )
