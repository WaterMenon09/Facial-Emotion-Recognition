import torch
import torch.nn as nn
import random

class Resnet_scratch(nn.Module):
    def __init__(self):
        super(Resnet_scratch,self).__init__()
        #image metadata
        self.meta = {
            "mean": [131.0912,103.8827,91.4953],
            "std": [1,1,1],
            "imageSize": [244,244,3],
        }
        #2x2 convolution
        self.conv1_7x7_2 = nn.Conv2d(
            in_channels=3,out_channels=64,
            kernel_size=[7,7],
            stride=(2,2),
            padding=(3,3),
            bias=False
        )
        #Batch Normalization
        self.conv1_7x7_bn = nn.BatchNorm2d(
            num_features=64, 
            eps=0.00001,
            momentum=0.1,
            affine=True, 
            track_running_stats=True
        )
        #Relu
        self.conv1_relu_7x7_2 = nn.ReLU()
        #Max Pooling
        self.pool1_3x3_2 = nn.MaxPool2d(
            kernel_size=[3,3],
            stride=[2,2],
            padding=(0,0),
            dilation=1,
            ceil_mode=True
        )

        #1x1 convo
        self.convo2_1_1x1_reduce = nn.Conv2d(
            in_channels=64,
            out_channels=64,
            kernel_size=[1,1],
            stride=(1,1),
            bias=False
        )
        self.convo2_1_1x1_reduce__bn = nn.BatchNorm2d(
            num_features=64,
            eps=0.00001,
            momentum=0.1,
            affine=True, 
            track_running_stats=True
        )
        self.conv2_1_1x1_reduce_relu = nn.ReLU()

        #3x3 kernel dim Convo
        self.conv2_1_3x3 = nn.Conv2d(
            in_channels=64,
            out_channels=64,
            kernel_size=[3,3],
            stride=(1,1),
            bias=False
        )
        self.conv2_1_3x3_bn = nn.BatchNorm2d(
            num_features=64,
            eps=0.00001,
            momentum=0.1,
            affine=True, 
            track_running_stats=True
        )
        self.conv2_1_3x3_relu = nn.ReLU()

        #1x1 higher output convo
        self.conv2_1_1x1_increase = nn.Conv2d(
            in_channels=64,
            out_channels=256,
            kernel_size=[1,1],
            stride=(1,1),
            bias=False
        )
        self.conv2_1_1x1_increase_bn = nn.BatchNorm2d(
            num_features=256,
            eps=0.00001,
            momentum=0.1,
            affine=True, 
            track_running_stats=True
        )
        