import torch
import torch.nn as nn

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
        self.conv2_1_1x1_reduce = nn.Conv2d(
            in_channels=64,
            out_channels=64,
            kernel_size=[1,1],
            stride=(1,1),
            bias=False
        )
        self.conv2_1_1x1_reduce_bn = nn.BatchNorm2d(
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

        #increase
        self.conv2_1_1x1_proj = nn.Conv2d(
            in_channels=64,
            out_channels=256,
            kernel_size=[1,1],
            stride=(1,1),
            bias=False
        )
        self.conv2_1_1x1_proj_bn = nn.BatchNorm2d(
            num_features=256,
            eps=0.00001,
            momentum=0.1,
            affine=True, 
            track_running_stats=True
        )
        self.conv2_1_relu = nn.ReLU()

        #reduce
        self.conv2_2_1x1_reduce = nn.Conv2d(
            in_channels=256,
            out_channels=64,
            kernel_size=[1,1],
            stride=(1,1),
            bias=False
        )
        self.conv2_2_1x1_reduce_bn = nn.BatchNorm2d(
            num_features=64,
            eps=0.00001,
            momentum=0.1,
            affine=True, 
            track_running_stats=True
        )
        self.conv2_2_1x1_reduce_relu = nn.ReLU()

        self.conv2_2_3x3 = nn.Conv2d(
            in_channels=64,
            out_channels= 64, 
            kernel_size=[3, 3], 
            stride=(1, 1), 
            padding=(1, 1), 
            bias=False
        )
        self.conv2_2_3x3_bn = nn.BatchNorm2d(
            num_features=64, 
            eps=1e-05, 
            momentum=0.1, 
            affine=True, 
            track_running_stats=True
        )
        self.conv2_2_3x3_relu = nn.ReLU()

        self.conv2_2_1x1_increase = nn.Conv2d(
            in_channels=64, 
            out_channels=256, 
            kernel_size=[1, 1], 
            stride=(1, 1), 
            bias=False
        )
        self.conv2_2_1x1_increase_bn = nn.BatchNorm2d(
            num_features=256, 
            eps=1e-05, 
            momentum=0.1, 
            affine=True, 
            track_running_stats=True
        )
        self.conv2_2_relu = nn.ReLU()

        self.conv2_3_1x1_reduce = nn.Conv2d(
            in_channels=256, 
            out_channels=64, 
            kernel_size=[1, 1], 
            stride=(1, 1), 
            bias=False
        )
        self.conv2_3_1x1_reduce_bn = nn.BatchNorm2d(
            num_features=64, 
            eps=1e-05, 
            momentum=0.1, 
            affine=True, 
            track_running_stats=True
        )
        self.conv2_3_1x1_reduce_relu = nn.ReLU()

        self.conv2_3_3x3 = nn.Conv2d(
            in_channels=64, 
            out_channels=64, 
            kernel_size=[3, 3], 
            stride=(1, 1), 
            padding=(1, 1), 
            bias=False
        )
        self.conv2_3_3x3_bn = nn.BatchNorm2d(
            num_features=64, 
            eps=1e-05, 
            momentum=0.1, 
            affine=True, 
            track_running_stats=True
        )
        self.conv2_3_3x3_relu = nn.ReLU()

        self.conv2_3_1x1_increase = nn.Conv2d(
            in_channels=64, 
            out_channels=256, 
            kernel_size=[1, 1], 
            stride=(1, 1), 
            bias=False
        )
        self.conv2_3_1x1_increase_bn = nn.BatchNorm2d(
            num_features=256, 
            eps=1e-05, 
            momentum=0.1, 
            affine=True, 
            track_running_stats=True
        )
        self.conv2_3_relu = nn.ReLU()

        self.conv3_1_1x1_reduce = nn.Conv2d(
            in_channels=256, 
            out_channels=128, 
            kernel_size=[1, 1], 
            stride=(2, 2), 
            bias=False
        )
        self.conv3_1_1x1_reduce_bn = nn.BatchNorm2d(
            num_features=128, 
            eps=1e-05, 
            momentum=0.1, 
            affine=True, 
            track_running_stats=True
        )
        self.conv3_1_1x1_reduce_relu = nn.ReLU()
        self.conv3_1_3x3 = nn.Conv2d(
            in_channels=128, 
            out_channels=128, 
            kernel_size=[3, 3], 
            stride=(1, 1), 
            padding=(1, 1), 
            bias=False
        )
        self.conv3_1_3x3_bn = nn.BatchNorm2d(
            num_features=128, 
            eps=1e-05, 
            momentum=0.1, 
            affine=True, 
            track_running_stats=True
        )
        self.conv3_1_3x3_relu = nn.ReLU()

        self.conv3_1_1x1_increase = nn.Conv2d(
            in_channels=128, 
            out_channels=512, 
            kernel_size=[1, 1], 
            stride=(1, 1), 
            bias=False
        )
        self.conv3_1_1x1_increase_bn = nn.BatchNorm2d(
            num_features=512, 
            eps=1e-05,
            momentum=0.1, 
            affine=True, 
            track_running_stats=True
        )
        self.conv3_1_1x1_proj = nn.Conv2d(
            in_channels=256, 
            out_channels=512, 
            kernel_size=[1, 1], 
            stride=(2, 2), 
            bias=False
        )
        self.conv3_1_1x1_proj_bn = nn.BatchNorm2d(
            num_features=512, 
            eps=1e-05, 
            momentum=0.1, 
            affine=True, 
            track_running_stats=True
        )
        self.conv3_1_relu = nn.ReLU()

        self.conv3_2_1x1_reduce = nn.Conv2d(
            in_channels=512, 
            out_channels=128, 
            kernel_size=[1, 1], 
            stride=(1, 1), 
            bias=False
        )
        self.conv3_2_1x1_reduce_bn = nn.BatchNorm2d(
            num_features=128, 
            eps=1e-05, 
            momentum=0.1, 
            affine=True, 
            track_running_stats=True
        )
        self.conv3_2_1x1_reduce_relu = nn.ReLU()

        self.conv3_2_3x3 = nn.Conv2d(
            in_channels=128, 
            out_channels=128, 
            kernel_size=[3, 3], 
            stride=(1, 1), 
            padding=(1, 1), 
            bias=False
        )
        self.conv3_2_3x3_bn = nn.BatchNorm2d(
            num_features=128, 
            eps=1e-05, 
            momentum=0.1, 
            affine=True, 
            track_running_stats=True
        )
        self.conv3_2_3x3_relu = nn.ReLU()

        self.conv3_2_1x1_increase = nn.Conv2d(
            128, 512, kernel_size=[1, 1], stride=(1, 1), bias=False
        )
        self.conv3_2_1x1_increase_bn = nn.BatchNorm2d(
            512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True
        )
        self.conv3_2_relu = nn.ReLU()

        self.conv3_3_1x1_reduce = nn.Conv2d(
            512, 128, kernel_size=[1, 1], stride=(1, 1), bias=False
        )
        self.conv3_3_1x1_reduce_bn = nn.BatchNorm2d(
            128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True
        )
        self.conv3_3_1x1_reduce_relu = nn.ReLU()

        self.conv3_3_3x3 = nn.Conv2d(
            128, 128, kernel_size=[3, 3], stride=(1, 1), padding=(1, 1), bias=False
        )
        self.conv3_3_3x3_bn = nn.BatchNorm2d(
            128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True
        )
        self.conv3_3_3x3_relu = nn.ReLU()

        self.conv3_3_1x1_increase = nn.Conv2d(
            128, 512, kernel_size=[1, 1], stride=(1, 1), bias=False
        )
        self.conv3_3_1x1_increase_bn = nn.BatchNorm2d(
            512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True
        )
        self.conv3_3_relu = nn.ReLU()

        self.conv3_4_1x1_reduce = nn.Conv2d(
            512, 128, kernel_size=[1, 1], stride=(1, 1), bias=False
        )
        self.conv3_4_1x1_reduce_bn = nn.BatchNorm2d(
            128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True
        )
        self.conv3_4_1x1_reduce_relu = nn.ReLU()

        self.conv3_4_3x3 = nn.Conv2d(
            128, 128, kernel_size=[3, 3], stride=(1, 1), padding=(1, 1), bias=False
        )
        self.conv3_4_3x3_bn = nn.BatchNorm2d(
            128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True
        )
        self.conv3_4_3x3_relu = nn.ReLU()

        self.conv3_4_1x1_increase = nn.Conv2d(
            128, 512, kernel_size=[1, 1], stride=(1, 1), bias=False
        )
        self.conv3_4_1x1_increase_bn = nn.BatchNorm2d(
            512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True
        )
        self.conv3_4_relu = nn.ReLU()

        self.conv4_1_1x1_reduce = nn.Conv2d(
            512, 256, kernel_size=[1, 1], stride=(2, 2), bias=False
        )
        self.conv4_1_1x1_reduce_bn = nn.BatchNorm2d(
            256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True
        )
        self.conv4_1_1x1_reduce_relu = nn.ReLU()

        self.conv4_1_3x3 = nn.Conv2d(
            256, 256, kernel_size=[3, 3], stride=(1, 1), padding=(1, 1), bias=False
        )
        self.conv4_1_3x3_bn = nn.BatchNorm2d(
            256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True
        )
        self.conv4_1_3x3_relu = nn.ReLU()

        self.conv4_1_1x1_increase = nn.Conv2d(
            256, 1024, kernel_size=[1, 1], stride=(1, 1), bias=False
        )
        self.conv4_1_1x1_increase_bn = nn.BatchNorm2d(
            1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True
        )

        self.conv4_1_1x1_proj = nn.Conv2d(
            512, 1024, kernel_size=[1, 1], stride=(2, 2), bias=False
        )
        self.conv4_1_1x1_proj_bn = nn.BatchNorm2d(
            1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True
        )
        self.conv4_1_relu = nn.ReLU()

        self.conv4_2_1x1_reduce = nn.Conv2d(
            1024, 256, kernel_size=[1, 1], stride=(1, 1), bias=False
        )
        self.conv4_2_1x1_reduce_bn = nn.BatchNorm2d(
            256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True
        )
        self.conv4_2_1x1_reduce_relu = nn.ReLU()

        self.conv4_2_3x3 = nn.Conv2d(
            256, 256, kernel_size=[3, 3], stride=(1, 1), padding=(1, 1), bias=False
        )
        self.conv4_2_3x3_bn = nn.BatchNorm2d(
            256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True
        )
        self.conv4_2_3x3_relu = nn.ReLU()

        self.conv4_2_1x1_increase = nn.Conv2d(
            256, 1024, kernel_size=[1, 1], stride=(1, 1), bias=False
        )
        self.conv4_2_1x1_increase_bn = nn.BatchNorm2d(
            1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True
        )
        self.conv4_2_relu = nn.ReLU()

        self.conv4_3_1x1_reduce = nn.Conv2d(
            1024, 256, kernel_size=[1, 1], stride=(1, 1), bias=False
        )
        self.conv4_3_1x1_reduce_bn = nn.BatchNorm2d(
            256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True
        )
        self.conv4_3_1x1_reduce_relu = nn.ReLU()

        self.conv4_3_3x3 = nn.Conv2d(
            256, 256, kernel_size=[3, 3], stride=(1, 1), padding=(1, 1), bias=False
        )
        self.conv4_3_3x3_bn = nn.BatchNorm2d(
            256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True
        )
        self.conv4_3_3x3_relu = nn.ReLU()

        self.conv4_3_1x1_increase = nn.Conv2d(
            256, 1024, kernel_size=[1, 1], stride=(1, 1), bias=False
        )
        self.conv4_3_1x1_increase_bn = nn.BatchNorm2d(
            1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True
        )
        self.conv4_3_relu = nn.ReLU()

        self.conv4_4_1x1_reduce = nn.Conv2d(
            1024, 256, kernel_size=[1, 1], stride=(1, 1), bias=False
        )
        self.conv4_4_1x1_reduce_bn = nn.BatchNorm2d(
            256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True
        )
        self.conv4_4_1x1_reduce_relu = nn.ReLU()

        self.conv4_4_3x3 = nn.Conv2d(
            256, 256, kernel_size=[3, 3], stride=(1, 1), padding=(1, 1), bias=False
        )
        self.conv4_4_3x3_bn = nn.BatchNorm2d(
            256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True
        )
        self.conv4_4_3x3_relu = nn.ReLU()

        self.conv4_4_1x1_increase = nn.Conv2d(
            256, 1024, kernel_size=[1, 1], stride=(1, 1), bias=False
        )
        self.conv4_4_1x1_increase_bn = nn.BatchNorm2d(
            1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True
        )
        self.conv4_4_relu = nn.ReLU()

        self.conv4_5_1x1_reduce = nn.Conv2d(
            1024, 256, kernel_size=[1, 1], stride=(1, 1), bias=False
        )
        self.conv4_5_1x1_reduce_bn = nn.BatchNorm2d(
            256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True
        )
        self.conv4_5_1x1_reduce_relu = nn.ReLU()

        self.conv4_5_3x3 = nn.Conv2d(
            256, 256, kernel_size=[3, 3], stride=(1, 1), padding=(1, 1), bias=False
        )
        self.conv4_5_3x3_bn = nn.BatchNorm2d(
            256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True
        )
        self.conv4_5_3x3_relu = nn.ReLU()

        self.conv4_5_1x1_increase = nn.Conv2d(
            256, 1024, kernel_size=[1, 1], stride=(1, 1), bias=False
        )
        self.conv4_5_1x1_increase_bn = nn.BatchNorm2d(
            1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True
        )
        self.conv4_5_relu = nn.ReLU()

        self.conv4_6_1x1_reduce = nn.Conv2d(
            1024, 256, kernel_size=[1, 1], stride=(1, 1), bias=False
        )
        self.conv4_6_1x1_reduce_bn = nn.BatchNorm2d(
            256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True
        )
        self.conv4_6_1x1_reduce_relu = nn.ReLU()

        self.conv4_6_3x3 = nn.Conv2d(
            256, 256, kernel_size=[3, 3], stride=(1, 1), padding=(1, 1), bias=False
        )
        self.conv4_6_3x3_bn = nn.BatchNorm2d(
            256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True
        )
        self.conv4_6_3x3_relu = nn.ReLU()

        self.conv4_6_1x1_increase = nn.Conv2d(
            256, 1024, kernel_size=[1, 1], stride=(1, 1), bias=False
        )
        self.conv4_6_1x1_increase_bn = nn.BatchNorm2d(
            1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True
        )
        self.conv4_6_relu = nn.ReLU()

        self.conv5_1_1x1_reduce = nn.Conv2d(
            1024, 512, kernel_size=[1, 1], stride=(2, 2), bias=False
        )
        self.conv5_1_1x1_reduce_bn = nn.BatchNorm2d(
            512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True
        )
        self.conv5_1_1x1_reduce_relu = nn.ReLU()

        self.conv5_1_3x3 = nn.Conv2d(
            512, 512, kernel_size=[3, 3], stride=(1, 1), padding=(1, 1), bias=False
        )
        self.conv5_1_3x3_bn = nn.BatchNorm2d(
            512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True
        )
        self.conv5_1_3x3_relu = nn.ReLU()

        self.conv5_1_1x1_increase = nn.Conv2d(
            512, 2048, kernel_size=[1, 1], stride=(1, 1), bias=False
        )
        self.conv5_1_1x1_increase_bn = nn.BatchNorm2d(
            2048, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True
        )

        self.conv5_1_1x1_proj = nn.Conv2d(
            1024, 2048, kernel_size=[1, 1], stride=(2, 2), bias=False
        )
        self.conv5_1_1x1_proj_bn = nn.BatchNorm2d(
            2048, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True
        )
        self.conv5_1_relu = nn.ReLU()

        self.conv5_2_1x1_reduce = nn.Conv2d(
            2048, 512, kernel_size=[1, 1], stride=(1, 1), bias=False
        )
        self.conv5_2_1x1_reduce_bn = nn.BatchNorm2d(
            512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True
        )
        self.conv5_2_1x1_reduce_relu = nn.ReLU()

        self.conv5_2_3x3 = nn.Conv2d(
            512, 512, kernel_size=[3, 3], stride=(1, 1), padding=(1, 1), bias=False
        )
        self.conv5_2_3x3_bn = nn.BatchNorm2d(
            512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True
        )
        self.conv5_2_3x3_relu = nn.ReLU()

        self.conv5_2_1x1_increase = nn.Conv2d(
            512, 2048, kernel_size=[1, 1], stride=(1, 1), bias=False
        )
        self.conv5_2_1x1_increase_bn = nn.BatchNorm2d(
            2048, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True
        )
        self.conv5_2_relu = nn.ReLU()

        self.conv5_3_1x1_reduce = nn.Conv2d(
            2048, 512, kernel_size=[1, 1], stride=(1, 1), bias=False
        )
        self.conv5_3_1x1_reduce_bn = nn.BatchNorm2d(
            512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True
        )
        self.conv5_3_1x1_reduce_relu = nn.ReLU()

        self.conv5_3_3x3 = nn.Conv2d(
            512, 512, kernel_size=[3, 3], stride=(1, 1), padding=(1, 1), bias=False
        )
        self.conv5_3_3x3_bn = nn.BatchNorm2d(
            512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True
        )
        self.conv5_3_3x3_relu = nn.ReLU()

        self.conv5_3_1x1_increase = nn.Conv2d(
            512, 2048, kernel_size=[1, 1], stride=(1, 1), bias=False
        )
        self.conv5_3_1x1_increase_bn = nn.BatchNorm2d(
            2048, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True
        )
        self.conv5_3_relu = nn.ReLU()
        self.pool5_7x7_s1 = nn.AvgPool2d(kernel_size=[7, 7], stride=[1, 1], padding=0)
        self.classifier = nn.Conv2d(2048, 8631, kernel_size=[1, 1], stride=(1, 1))
    
    def forward(self, data):
        conv1_7x7_s2 = self.conv1_7x7_2(data)
        conv1_7x7_s2_bn = self.conv1_7x7_bn(conv1_7x7_s2)
        conv1_7x7_s2_bnxx = self.conv1_relu_7x7_2(conv1_7x7_s2_bn)
        pool1_3x3_s2 = self.pool1_3x3_2(conv1_7x7_s2_bnxx)
        conv2_1_1x1_reduce = self.conv2_1_1x1_reduce(pool1_3x3_s2)
        conv2_1_1x1_reduce_bn = self.conv2_1_1x1_reduce_bn(conv2_1_1x1_reduce)
        conv2_1_1x1_reduce_bnxx = self.conv2_1_1x1_reduce_relu(conv2_1_1x1_reduce_bn)
        conv2_1_3x3 = self.conv2_1_3x3(conv2_1_1x1_reduce_bnxx)
        conv2_1_3x3_bn = self.conv2_1_3x3_bn(conv2_1_3x3)
        conv2_1_3x3_bnxx = self.conv2_1_3x3_relu(conv2_1_3x3_bn)
        conv2_1_1x1_increase = self.conv2_1_1x1_increase(conv2_1_3x3_bnxx)
        conv2_1_1x1_increase_bn = self.conv2_1_1x1_increase_bn(conv2_1_1x1_increase)
        conv2_1_1x1_proj = self.conv2_1_1x1_proj(pool1_3x3_s2)
        conv2_1_1x1_proj_bn = self.conv2_1_1x1_proj_bn(conv2_1_1x1_proj)
        conv2_1 = torch.add(conv2_1_1x1_proj_bn, 1, conv2_1_1x1_increase_bn)
        conv2_1x = self.conv2_1_relu(conv2_1)
        conv2_2_1x1_reduce = self.conv2_2_1x1_reduce(conv2_1x)
        conv2_2_1x1_reduce_bn = self.conv2_2_1x1_reduce_bn(conv2_2_1x1_reduce)
        conv2_2_1x1_reduce_bnxx = self.conv2_2_1x1_reduce_relu(conv2_2_1x1_reduce_bn)
        conv2_2_3x3 = self.conv2_2_3x3(conv2_2_1x1_reduce_bnxx)
        conv2_2_3x3_bn = self.conv2_2_3x3_bn(conv2_2_3x3)
        conv2_2_3x3_bnxx = self.conv2_2_3x3_relu(conv2_2_3x3_bn)
        conv2_2_1x1_increase = self.conv2_2_1x1_increase(conv2_2_3x3_bnxx)
        conv2_2_1x1_increase_bn = self.conv2_2_1x1_increase_bn(conv2_2_1x1_increase)
        conv2_2 = torch.add(conv2_1x, 1, conv2_2_1x1_increase_bn)
        conv2_2x = self.conv2_2_relu(conv2_2)
        conv2_3_1x1_reduce = self.conv2_3_1x1_reduce(conv2_2x)
        conv2_3_1x1_reduce_bn = self.conv2_3_1x1_reduce_bn(conv2_3_1x1_reduce)
        conv2_3_1x1_reduce_bnxx = self.conv2_3_1x1_reduce_relu(conv2_3_1x1_reduce_bn)
        conv2_3_3x3 = self.conv2_3_3x3(conv2_3_1x1_reduce_bnxx)
        conv2_3_3x3_bn = self.conv2_3_3x3_bn(conv2_3_3x3)
        conv2_3_3x3_bnxx = self.conv2_3_3x3_relu(conv2_3_3x3_bn)
        conv2_3_1x1_increase = self.conv2_3_1x1_increase(conv2_3_3x3_bnxx)
        conv2_3_1x1_increase_bn = self.conv2_3_1x1_increase_bn(conv2_3_1x1_increase)
        conv2_3 = torch.add(conv2_2x, 1, conv2_3_1x1_increase_bn)
        conv2_3x = self.conv2_3_relu(conv2_3)
        conv3_1_1x1_reduce = self.conv3_1_1x1_reduce(conv2_3x)
        conv3_1_1x1_reduce_bn = self.conv3_1_1x1_reduce_bn(conv3_1_1x1_reduce)
        conv3_1_1x1_reduce_bnxx = self.conv3_1_1x1_reduce_relu(conv3_1_1x1_reduce_bn)
        conv3_1_3x3 = self.conv3_1_3x3(conv3_1_1x1_reduce_bnxx)
        conv3_1_3x3_bn = self.conv3_1_3x3_bn(conv3_1_3x3)
        conv3_1_3x3_bnxx = self.conv3_1_3x3_relu(conv3_1_3x3_bn)
        conv3_1_1x1_increase = self.conv3_1_1x1_increase(conv3_1_3x3_bnxx)
        conv3_1_1x1_increase_bn = self.conv3_1_1x1_increase_bn(conv3_1_1x1_increase)
        conv3_1_1x1_proj = self.conv3_1_1x1_proj(conv2_3x)
        conv3_1_1x1_proj_bn = self.conv3_1_1x1_proj_bn(conv3_1_1x1_proj)
        conv3_1 = torch.add(conv3_1_1x1_proj_bn, 1, conv3_1_1x1_increase_bn)
        conv3_1x = self.conv3_1_relu(conv3_1)
        conv3_2_1x1_reduce = self.conv3_2_1x1_reduce(conv3_1x)
        conv3_2_1x1_reduce_bn = self.conv3_2_1x1_reduce_bn(conv3_2_1x1_reduce)
        conv3_2_1x1_reduce_bnxx = self.conv3_2_1x1_reduce_relu(conv3_2_1x1_reduce_bn)
        conv3_2_3x3 = self.conv3_2_3x3(conv3_2_1x1_reduce_bnxx)
        conv3_2_3x3_bn = self.conv3_2_3x3_bn(conv3_2_3x3)
        conv3_2_3x3_bnxx = self.conv3_2_3x3_relu(conv3_2_3x3_bn)
        conv3_2_1x1_increase = self.conv3_2_1x1_increase(conv3_2_3x3_bnxx)
        conv3_2_1x1_increase_bn = self.conv3_2_1x1_increase_bn(conv3_2_1x1_increase)
        conv3_2 = torch.add(conv3_1x, 1, conv3_2_1x1_increase_bn)
        conv3_2x = self.conv3_2_relu(conv3_2)
        conv3_3_1x1_reduce = self.conv3_3_1x1_reduce(conv3_2x)
        conv3_3_1x1_reduce_bn = self.conv3_3_1x1_reduce_bn(conv3_3_1x1_reduce)
        conv3_3_1x1_reduce_bnxx = self.conv3_3_1x1_reduce_relu(conv3_3_1x1_reduce_bn)
        conv3_3_3x3 = self.conv3_3_3x3(conv3_3_1x1_reduce_bnxx)
        conv3_3_3x3_bn = self.conv3_3_3x3_bn(conv3_3_3x3)
        conv3_3_3x3_bnxx = self.conv3_3_3x3_relu(conv3_3_3x3_bn)
        conv3_3_1x1_increase = self.conv3_3_1x1_increase(conv3_3_3x3_bnxx)
        conv3_3_1x1_increase_bn = self.conv3_3_1x1_increase_bn(conv3_3_1x1_increase)
        conv3_3 = torch.add(conv3_2x, 1, conv3_3_1x1_increase_bn)
        conv3_3x = self.conv3_3_relu(conv3_3)
        conv3_4_1x1_reduce = self.conv3_4_1x1_reduce(conv3_3x)
        conv3_4_1x1_reduce_bn = self.conv3_4_1x1_reduce_bn(conv3_4_1x1_reduce)
        conv3_4_1x1_reduce_bnxx = self.conv3_4_1x1_reduce_relu(conv3_4_1x1_reduce_bn)
        conv3_4_3x3 = self.conv3_4_3x3(conv3_4_1x1_reduce_bnxx)
        conv3_4_3x3_bn = self.conv3_4_3x3_bn(conv3_4_3x3)
        conv3_4_3x3_bnxx = self.conv3_4_3x3_relu(conv3_4_3x3_bn)
        conv3_4_1x1_increase = self.conv3_4_1x1_increase(conv3_4_3x3_bnxx)
        conv3_4_1x1_increase_bn = self.conv3_4_1x1_increase_bn(conv3_4_1x1_increase)
        conv3_4 = torch.add(conv3_3x, 1, conv3_4_1x1_increase_bn)
        conv3_4x = self.conv3_4_relu(conv3_4)
        conv4_1_1x1_reduce = self.conv4_1_1x1_reduce(conv3_4x)
        conv4_1_1x1_reduce_bn = self.conv4_1_1x1_reduce_bn(conv4_1_1x1_reduce)
        conv4_1_1x1_reduce_bnxx = self.conv4_1_1x1_reduce_relu(conv4_1_1x1_reduce_bn)
        conv4_1_3x3 = self.conv4_1_3x3(conv4_1_1x1_reduce_bnxx)
        conv4_1_3x3_bn = self.conv4_1_3x3_bn(conv4_1_3x3)
        conv4_1_3x3_bnxx = self.conv4_1_3x3_relu(conv4_1_3x3_bn)
        conv4_1_1x1_increase = self.conv4_1_1x1_increase(conv4_1_3x3_bnxx)
        conv4_1_1x1_increase_bn = self.conv4_1_1x1_increase_bn(conv4_1_1x1_increase)
        conv4_1_1x1_proj = self.conv4_1_1x1_proj(conv3_4x)
        conv4_1_1x1_proj_bn = self.conv4_1_1x1_proj_bn(conv4_1_1x1_proj)
        conv4_1 = torch.add(conv4_1_1x1_proj_bn, 1, conv4_1_1x1_increase_bn)
        conv4_1x = self.conv4_1_relu(conv4_1)
        conv4_2_1x1_reduce = self.conv4_2_1x1_reduce(conv4_1x)
        conv4_2_1x1_reduce_bn = self.conv4_2_1x1_reduce_bn(conv4_2_1x1_reduce)
        conv4_2_1x1_reduce_bnxx = self.conv4_2_1x1_reduce_relu(conv4_2_1x1_reduce_bn)
        conv4_2_3x3 = self.conv4_2_3x3(conv4_2_1x1_reduce_bnxx)
        conv4_2_3x3_bn = self.conv4_2_3x3_bn(conv4_2_3x3)
        conv4_2_3x3_bnxx = self.conv4_2_3x3_relu(conv4_2_3x3_bn)
        conv4_2_1x1_increase = self.conv4_2_1x1_increase(conv4_2_3x3_bnxx)
        conv4_2_1x1_increase_bn = self.conv4_2_1x1_increase_bn(conv4_2_1x1_increase)
        conv4_2 = torch.add(conv4_1x, 1, conv4_2_1x1_increase_bn)
        conv4_2x = self.conv4_2_relu(conv4_2)
        conv4_3_1x1_reduce = self.conv4_3_1x1_reduce(conv4_2x)
        conv4_3_1x1_reduce_bn = self.conv4_3_1x1_reduce_bn(conv4_3_1x1_reduce)
        conv4_3_1x1_reduce_bnxx = self.conv4_3_1x1_reduce_relu(conv4_3_1x1_reduce_bn)
        conv4_3_3x3 = self.conv4_3_3x3(conv4_3_1x1_reduce_bnxx)
        conv4_3_3x3_bn = self.conv4_3_3x3_bn(conv4_3_3x3)
        conv4_3_3x3_bnxx = self.conv4_3_3x3_relu(conv4_3_3x3_bn)
        conv4_3_1x1_increase = self.conv4_3_1x1_increase(conv4_3_3x3_bnxx)
        conv4_3_1x1_increase_bn = self.conv4_3_1x1_increase_bn(conv4_3_1x1_increase)
        conv4_3 = torch.add(conv4_2x, 1, conv4_3_1x1_increase_bn)
        conv4_3x = self.conv4_3_relu(conv4_3)
        conv4_4_1x1_reduce = self.conv4_4_1x1_reduce(conv4_3x)
        conv4_4_1x1_reduce_bn = self.conv4_4_1x1_reduce_bn(conv4_4_1x1_reduce)
        conv4_4_1x1_reduce_bnxx = self.conv4_4_1x1_reduce_relu(conv4_4_1x1_reduce_bn)
        conv4_4_3x3 = self.conv4_4_3x3(conv4_4_1x1_reduce_bnxx)
        conv4_4_3x3_bn = self.conv4_4_3x3_bn(conv4_4_3x3)
        conv4_4_3x3_bnxx = self.conv4_4_3x3_relu(conv4_4_3x3_bn)
        conv4_4_1x1_increase = self.conv4_4_1x1_increase(conv4_4_3x3_bnxx)
        conv4_4_1x1_increase_bn = self.conv4_4_1x1_increase_bn(conv4_4_1x1_increase)
        conv4_4 = torch.add(conv4_3x, 1, conv4_4_1x1_increase_bn)
        conv4_4x = self.conv4_4_relu(conv4_4)
        conv4_5_1x1_reduce = self.conv4_5_1x1_reduce(conv4_4x)
        conv4_5_1x1_reduce_bn = self.conv4_5_1x1_reduce_bn(conv4_5_1x1_reduce)
        conv4_5_1x1_reduce_bnxx = self.conv4_5_1x1_reduce_relu(conv4_5_1x1_reduce_bn)
        conv4_5_3x3 = self.conv4_5_3x3(conv4_5_1x1_reduce_bnxx)
        conv4_5_3x3_bn = self.conv4_5_3x3_bn(conv4_5_3x3)
        conv4_5_3x3_bnxx = self.conv4_5_3x3_relu(conv4_5_3x3_bn)
        conv4_5_1x1_increase = self.conv4_5_1x1_increase(conv4_5_3x3_bnxx)
        conv4_5_1x1_increase_bn = self.conv4_5_1x1_increase_bn(conv4_5_1x1_increase)
        conv4_5 = torch.add(conv4_4x, 1, conv4_5_1x1_increase_bn)
        conv4_5x = self.conv4_5_relu(conv4_5)
        conv4_6_1x1_reduce = self.conv4_6_1x1_reduce(conv4_5x)
        conv4_6_1x1_reduce_bn = self.conv4_6_1x1_reduce_bn(conv4_6_1x1_reduce)
        conv4_6_1x1_reduce_bnxx = self.conv4_6_1x1_reduce_relu(conv4_6_1x1_reduce_bn)
        conv4_6_3x3 = self.conv4_6_3x3(conv4_6_1x1_reduce_bnxx)
        conv4_6_3x3_bn = self.conv4_6_3x3_bn(conv4_6_3x3)
        conv4_6_3x3_bnxx = self.conv4_6_3x3_relu(conv4_6_3x3_bn)
        conv4_6_1x1_increase = self.conv4_6_1x1_increase(conv4_6_3x3_bnxx)
        conv4_6_1x1_increase_bn = self.conv4_6_1x1_increase_bn(conv4_6_1x1_increase)
        conv4_6 = torch.add(conv4_5x, 1, conv4_6_1x1_increase_bn)
        conv4_6x = self.conv4_6_relu(conv4_6)
        conv5_1_1x1_reduce = self.conv5_1_1x1_reduce(conv4_6x)
        conv5_1_1x1_reduce_bn = self.conv5_1_1x1_reduce_bn(conv5_1_1x1_reduce)
        conv5_1_1x1_reduce_bnxx = self.conv5_1_1x1_reduce_relu(conv5_1_1x1_reduce_bn)
        conv5_1_3x3 = self.conv5_1_3x3(conv5_1_1x1_reduce_bnxx)
        conv5_1_3x3_bn = self.conv5_1_3x3_bn(conv5_1_3x3)
        conv5_1_3x3_bnxx = self.conv5_1_3x3_relu(conv5_1_3x3_bn)
        conv5_1_1x1_increase = self.conv5_1_1x1_increase(conv5_1_3x3_bnxx)
        conv5_1_1x1_increase_bn = self.conv5_1_1x1_increase_bn(conv5_1_1x1_increase)
        conv5_1_1x1_proj = self.conv5_1_1x1_proj(conv4_6x)
        conv5_1_1x1_proj_bn = self.conv5_1_1x1_proj_bn(conv5_1_1x1_proj)
        conv5_1 = torch.add(conv5_1_1x1_proj_bn, 1, conv5_1_1x1_increase_bn)
        conv5_1x = self.conv5_1_relu(conv5_1)
        conv5_2_1x1_reduce = self.conv5_2_1x1_reduce(conv5_1x)
        conv5_2_1x1_reduce_bn = self.conv5_2_1x1_reduce_bn(conv5_2_1x1_reduce)
        conv5_2_1x1_reduce_bnxx = self.conv5_2_1x1_reduce_relu(conv5_2_1x1_reduce_bn)
        conv5_2_3x3 = self.conv5_2_3x3(conv5_2_1x1_reduce_bnxx)
        conv5_2_3x3_bn = self.conv5_2_3x3_bn(conv5_2_3x3)
        conv5_2_3x3_bnxx = self.conv5_2_3x3_relu(conv5_2_3x3_bn)
        conv5_2_1x1_increase = self.conv5_2_1x1_increase(conv5_2_3x3_bnxx)
        conv5_2_1x1_increase_bn = self.conv5_2_1x1_increase_bn(conv5_2_1x1_increase)
        conv5_2 = torch.add(conv5_1x, 1, conv5_2_1x1_increase_bn)
        conv5_2x = self.conv5_2_relu(conv5_2)
        conv5_3_1x1_reduce = self.conv5_3_1x1_reduce(conv5_2x)
        conv5_3_1x1_reduce_bn = self.conv5_3_1x1_reduce_bn(conv5_3_1x1_reduce)
        conv5_3_1x1_reduce_bnxx = self.conv5_3_1x1_reduce_relu(conv5_3_1x1_reduce_bn)
        conv5_3_3x3 = self.conv5_3_3x3(conv5_3_1x1_reduce_bnxx)
        conv5_3_3x3_bn = self.conv5_3_3x3_bn(conv5_3_3x3)
        conv5_3_3x3_bnxx = self.conv5_3_3x3_relu(conv5_3_3x3_bn)
        conv5_3_1x1_increase = self.conv5_3_1x1_increase(conv5_3_3x3_bnxx)
        conv5_3_1x1_increase_bn = self.conv5_3_1x1_increase_bn(conv5_3_1x1_increase)
        conv5_3 = torch.add(conv5_2x, 1, conv5_3_1x1_increase_bn)
        conv5_3x = self.conv5_3_relu(conv5_3)
        pool5_7x7_s1 = self.pool5_7x7_s1(conv5_3x)  #classifier
        classifier_preflatten = self.classifier(pool5_7x7_s1)
        classifier = classifier_preflatten.view(classifier_preflatten.size(0), -1)
        return classifier
    
class Resnet50_pretrained_vgg(Resnet_scratch):
    def __init__(self):
        super(Resnet50_pretrained_vgg, self).__init__()
        # state_dict = torch.load('./saved/pretrained/resnet50_scratch_dims_2048.pth')
        # self.load_state_dict(state_dict)

        self.classifier = nn.Conv2d(2048, 7, kernel_size=[1, 1], stride=(1, 1))

    def forward(self, data):
        conv1_7x7_s2 = self.conv1_7x7_s2(data)
        conv1_7x7_s2_bn = self.conv1_7x7_s2_bn(conv1_7x7_s2)
        conv1_7x7_s2_bnxx = self.conv1_relu_7x7_s2(conv1_7x7_s2_bn)
        pool1_3x3_s2 = self.pool1_3x3_s2(conv1_7x7_s2_bnxx)
        conv2_1_1x1_reduce = self.conv2_1_1x1_reduce(pool1_3x3_s2)
        conv2_1_1x1_reduce_bn = self.conv2_1_1x1_reduce_bn(conv2_1_1x1_reduce)
        conv2_1_1x1_reduce_bnxx = self.conv2_1_1x1_reduce_relu(conv2_1_1x1_reduce_bn)
        conv2_1_3x3 = self.conv2_1_3x3(conv2_1_1x1_reduce_bnxx)
        conv2_1_3x3_bn = self.conv2_1_3x3_bn(conv2_1_3x3)
        conv2_1_3x3_bnxx = self.conv2_1_3x3_relu(conv2_1_3x3_bn)
        conv2_1_1x1_increase = self.conv2_1_1x1_increase(conv2_1_3x3_bnxx)
        conv2_1_1x1_increase_bn = self.conv2_1_1x1_increase_bn(conv2_1_1x1_increase)
        conv2_1_1x1_proj = self.conv2_1_1x1_proj(pool1_3x3_s2)
        conv2_1_1x1_proj_bn = self.conv2_1_1x1_proj_bn(conv2_1_1x1_proj)
        conv2_1 = torch.add(conv2_1_1x1_proj_bn, 1, conv2_1_1x1_increase_bn)
        conv2_1x = self.conv2_1_relu(conv2_1)
        conv2_2_1x1_reduce = self.conv2_2_1x1_reduce(conv2_1x)
        conv2_2_1x1_reduce_bn = self.conv2_2_1x1_reduce_bn(conv2_2_1x1_reduce)
        conv2_2_1x1_reduce_bnxx = self.conv2_2_1x1_reduce_relu(conv2_2_1x1_reduce_bn)
        conv2_2_3x3 = self.conv2_2_3x3(conv2_2_1x1_reduce_bnxx)
        conv2_2_3x3_bn = self.conv2_2_3x3_bn(conv2_2_3x3)
        conv2_2_3x3_bnxx = self.conv2_2_3x3_relu(conv2_2_3x3_bn)
        conv2_2_1x1_increase = self.conv2_2_1x1_increase(conv2_2_3x3_bnxx)
        conv2_2_1x1_increase_bn = self.conv2_2_1x1_increase_bn(conv2_2_1x1_increase)
        conv2_2 = torch.add(conv2_1x, 1, conv2_2_1x1_increase_bn)
        conv2_2x = self.conv2_2_relu(conv2_2)
        conv2_3_1x1_reduce = self.conv2_3_1x1_reduce(conv2_2x)
        conv2_3_1x1_reduce_bn = self.conv2_3_1x1_reduce_bn(conv2_3_1x1_reduce)
        conv2_3_1x1_reduce_bnxx = self.conv2_3_1x1_reduce_relu(conv2_3_1x1_reduce_bn)
        conv2_3_3x3 = self.conv2_3_3x3(conv2_3_1x1_reduce_bnxx)
        conv2_3_3x3_bn = self.conv2_3_3x3_bn(conv2_3_3x3)
        conv2_3_3x3_bnxx = self.conv2_3_3x3_relu(conv2_3_3x3_bn)
        conv2_3_1x1_increase = self.conv2_3_1x1_increase(conv2_3_3x3_bnxx)
        conv2_3_1x1_increase_bn = self.conv2_3_1x1_increase_bn(conv2_3_1x1_increase)
        conv2_3 = torch.add(conv2_2x, 1, conv2_3_1x1_increase_bn)
        conv2_3x = self.conv2_3_relu(conv2_3)
        conv3_1_1x1_reduce = self.conv3_1_1x1_reduce(conv2_3x)
        conv3_1_1x1_reduce_bn = self.conv3_1_1x1_reduce_bn(conv3_1_1x1_reduce)
        conv3_1_1x1_reduce_bnxx = self.conv3_1_1x1_reduce_relu(conv3_1_1x1_reduce_bn)
        conv3_1_3x3 = self.conv3_1_3x3(conv3_1_1x1_reduce_bnxx)
        conv3_1_3x3_bn = self.conv3_1_3x3_bn(conv3_1_3x3)
        conv3_1_3x3_bnxx = self.conv3_1_3x3_relu(conv3_1_3x3_bn)
        conv3_1_1x1_increase = self.conv3_1_1x1_increase(conv3_1_3x3_bnxx)
        conv3_1_1x1_increase_bn = self.conv3_1_1x1_increase_bn(conv3_1_1x1_increase)
        conv3_1_1x1_proj = self.conv3_1_1x1_proj(conv2_3x)
        conv3_1_1x1_proj_bn = self.conv3_1_1x1_proj_bn(conv3_1_1x1_proj)
        conv3_1 = torch.add(conv3_1_1x1_proj_bn, 1, conv3_1_1x1_increase_bn)
        conv3_1x = self.conv3_1_relu(conv3_1)
        conv3_2_1x1_reduce = self.conv3_2_1x1_reduce(conv3_1x)
        conv3_2_1x1_reduce_bn = self.conv3_2_1x1_reduce_bn(conv3_2_1x1_reduce)
        conv3_2_1x1_reduce_bnxx = self.conv3_2_1x1_reduce_relu(conv3_2_1x1_reduce_bn)
        conv3_2_3x3 = self.conv3_2_3x3(conv3_2_1x1_reduce_bnxx)
        conv3_2_3x3_bn = self.conv3_2_3x3_bn(conv3_2_3x3)
        conv3_2_3x3_bnxx = self.conv3_2_3x3_relu(conv3_2_3x3_bn)
        conv3_2_1x1_increase = self.conv3_2_1x1_increase(conv3_2_3x3_bnxx)
        conv3_2_1x1_increase_bn = self.conv3_2_1x1_increase_bn(conv3_2_1x1_increase)
        conv3_2 = torch.add(conv3_1x, 1, conv3_2_1x1_increase_bn)
        conv3_2x = self.conv3_2_relu(conv3_2)
        conv3_3_1x1_reduce = self.conv3_3_1x1_reduce(conv3_2x)
        conv3_3_1x1_reduce_bn = self.conv3_3_1x1_reduce_bn(conv3_3_1x1_reduce)
        conv3_3_1x1_reduce_bnxx = self.conv3_3_1x1_reduce_relu(conv3_3_1x1_reduce_bn)
        conv3_3_3x3 = self.conv3_3_3x3(conv3_3_1x1_reduce_bnxx)
        conv3_3_3x3_bn = self.conv3_3_3x3_bn(conv3_3_3x3)
        conv3_3_3x3_bnxx = self.conv3_3_3x3_relu(conv3_3_3x3_bn)
        conv3_3_1x1_increase = self.conv3_3_1x1_increase(conv3_3_3x3_bnxx)
        conv3_3_1x1_increase_bn = self.conv3_3_1x1_increase_bn(conv3_3_1x1_increase)
        conv3_3 = torch.add(conv3_2x, 1, conv3_3_1x1_increase_bn)
        conv3_3x = self.conv3_3_relu(conv3_3)
        conv3_4_1x1_reduce = self.conv3_4_1x1_reduce(conv3_3x)
        conv3_4_1x1_reduce_bn = self.conv3_4_1x1_reduce_bn(conv3_4_1x1_reduce)
        conv3_4_1x1_reduce_bnxx = self.conv3_4_1x1_reduce_relu(conv3_4_1x1_reduce_bn)
        conv3_4_3x3 = self.conv3_4_3x3(conv3_4_1x1_reduce_bnxx)
        conv3_4_3x3_bn = self.conv3_4_3x3_bn(conv3_4_3x3)
        conv3_4_3x3_bnxx = self.conv3_4_3x3_relu(conv3_4_3x3_bn)
        conv3_4_1x1_increase = self.conv3_4_1x1_increase(conv3_4_3x3_bnxx)
        conv3_4_1x1_increase_bn = self.conv3_4_1x1_increase_bn(conv3_4_1x1_increase)
        conv3_4 = torch.add(conv3_3x, 1, conv3_4_1x1_increase_bn)
        conv3_4x = self.conv3_4_relu(conv3_4)
        conv4_1_1x1_reduce = self.conv4_1_1x1_reduce(conv3_4x)
        conv4_1_1x1_reduce_bn = self.conv4_1_1x1_reduce_bn(conv4_1_1x1_reduce)
        conv4_1_1x1_reduce_bnxx = self.conv4_1_1x1_reduce_relu(conv4_1_1x1_reduce_bn)
        conv4_1_3x3 = self.conv4_1_3x3(conv4_1_1x1_reduce_bnxx)
        conv4_1_3x3_bn = self.conv4_1_3x3_bn(conv4_1_3x3)
        conv4_1_3x3_bnxx = self.conv4_1_3x3_relu(conv4_1_3x3_bn)
        conv4_1_1x1_increase = self.conv4_1_1x1_increase(conv4_1_3x3_bnxx)
        conv4_1_1x1_increase_bn = self.conv4_1_1x1_increase_bn(conv4_1_1x1_increase)
        conv4_1_1x1_proj = self.conv4_1_1x1_proj(conv3_4x)
        conv4_1_1x1_proj_bn = self.conv4_1_1x1_proj_bn(conv4_1_1x1_proj)
        conv4_1 = torch.add(conv4_1_1x1_proj_bn, 1, conv4_1_1x1_increase_bn)
        conv4_1x = self.conv4_1_relu(conv4_1)
        conv4_2_1x1_reduce = self.conv4_2_1x1_reduce(conv4_1x)
        conv4_2_1x1_reduce_bn = self.conv4_2_1x1_reduce_bn(conv4_2_1x1_reduce)
        conv4_2_1x1_reduce_bnxx = self.conv4_2_1x1_reduce_relu(conv4_2_1x1_reduce_bn)
        conv4_2_3x3 = self.conv4_2_3x3(conv4_2_1x1_reduce_bnxx)
        conv4_2_3x3_bn = self.conv4_2_3x3_bn(conv4_2_3x3)
        conv4_2_3x3_bnxx = self.conv4_2_3x3_relu(conv4_2_3x3_bn)
        conv4_2_1x1_increase = self.conv4_2_1x1_increase(conv4_2_3x3_bnxx)
        conv4_2_1x1_increase_bn = self.conv4_2_1x1_increase_bn(conv4_2_1x1_increase)
        conv4_2 = torch.add(conv4_1x, 1, conv4_2_1x1_increase_bn)
        conv4_2x = self.conv4_2_relu(conv4_2)
        conv4_3_1x1_reduce = self.conv4_3_1x1_reduce(conv4_2x)
        conv4_3_1x1_reduce_bn = self.conv4_3_1x1_reduce_bn(conv4_3_1x1_reduce)
        conv4_3_1x1_reduce_bnxx = self.conv4_3_1x1_reduce_relu(conv4_3_1x1_reduce_bn)
        conv4_3_3x3 = self.conv4_3_3x3(conv4_3_1x1_reduce_bnxx)
        conv4_3_3x3_bn = self.conv4_3_3x3_bn(conv4_3_3x3)
        conv4_3_3x3_bnxx = self.conv4_3_3x3_relu(conv4_3_3x3_bn)
        conv4_3_1x1_increase = self.conv4_3_1x1_increase(conv4_3_3x3_bnxx)
        conv4_3_1x1_increase_bn = self.conv4_3_1x1_increase_bn(conv4_3_1x1_increase)
        conv4_3 = torch.add(conv4_2x, 1, conv4_3_1x1_increase_bn)
        conv4_3x = self.conv4_3_relu(conv4_3)
        conv4_4_1x1_reduce = self.conv4_4_1x1_reduce(conv4_3x)
        conv4_4_1x1_reduce_bn = self.conv4_4_1x1_reduce_bn(conv4_4_1x1_reduce)
        conv4_4_1x1_reduce_bnxx = self.conv4_4_1x1_reduce_relu(conv4_4_1x1_reduce_bn)
        conv4_4_3x3 = self.conv4_4_3x3(conv4_4_1x1_reduce_bnxx)
        conv4_4_3x3_bn = self.conv4_4_3x3_bn(conv4_4_3x3)
        conv4_4_3x3_bnxx = self.conv4_4_3x3_relu(conv4_4_3x3_bn)
        conv4_4_1x1_increase = self.conv4_4_1x1_increase(conv4_4_3x3_bnxx)
        conv4_4_1x1_increase_bn = self.conv4_4_1x1_increase_bn(conv4_4_1x1_increase)
        conv4_4 = torch.add(conv4_3x, 1, conv4_4_1x1_increase_bn)
        conv4_4x = self.conv4_4_relu(conv4_4)
        conv4_5_1x1_reduce = self.conv4_5_1x1_reduce(conv4_4x)
        conv4_5_1x1_reduce_bn = self.conv4_5_1x1_reduce_bn(conv4_5_1x1_reduce)
        conv4_5_1x1_reduce_bnxx = self.conv4_5_1x1_reduce_relu(conv4_5_1x1_reduce_bn)
        conv4_5_3x3 = self.conv4_5_3x3(conv4_5_1x1_reduce_bnxx)
        conv4_5_3x3_bn = self.conv4_5_3x3_bn(conv4_5_3x3)
        conv4_5_3x3_bnxx = self.conv4_5_3x3_relu(conv4_5_3x3_bn)
        conv4_5_1x1_increase = self.conv4_5_1x1_increase(conv4_5_3x3_bnxx)
        conv4_5_1x1_increase_bn = self.conv4_5_1x1_increase_bn(conv4_5_1x1_increase)
        conv4_5 = torch.add(conv4_4x, 1, conv4_5_1x1_increase_bn)
        conv4_5x = self.conv4_5_relu(conv4_5)
        conv4_6_1x1_reduce = self.conv4_6_1x1_reduce(conv4_5x)
        conv4_6_1x1_reduce_bn = self.conv4_6_1x1_reduce_bn(conv4_6_1x1_reduce)
        conv4_6_1x1_reduce_bnxx = self.conv4_6_1x1_reduce_relu(conv4_6_1x1_reduce_bn)
        conv4_6_3x3 = self.conv4_6_3x3(conv4_6_1x1_reduce_bnxx)
        conv4_6_3x3_bn = self.conv4_6_3x3_bn(conv4_6_3x3)
        conv4_6_3x3_bnxx = self.conv4_6_3x3_relu(conv4_6_3x3_bn)
        conv4_6_1x1_increase = self.conv4_6_1x1_increase(conv4_6_3x3_bnxx)
        conv4_6_1x1_increase_bn = self.conv4_6_1x1_increase_bn(conv4_6_1x1_increase)
        conv4_6 = torch.add(conv4_5x, 1, conv4_6_1x1_increase_bn)
        conv4_6x = self.conv4_6_relu(conv4_6)
        conv5_1_1x1_reduce = self.conv5_1_1x1_reduce(conv4_6x)
        conv5_1_1x1_reduce_bn = self.conv5_1_1x1_reduce_bn(conv5_1_1x1_reduce)
        conv5_1_1x1_reduce_bnxx = self.conv5_1_1x1_reduce_relu(conv5_1_1x1_reduce_bn)
        conv5_1_3x3 = self.conv5_1_3x3(conv5_1_1x1_reduce_bnxx)
        conv5_1_3x3_bn = self.conv5_1_3x3_bn(conv5_1_3x3)
        conv5_1_3x3_bnxx = self.conv5_1_3x3_relu(conv5_1_3x3_bn)
        conv5_1_1x1_increase = self.conv5_1_1x1_increase(conv5_1_3x3_bnxx)
        conv5_1_1x1_increase_bn = self.conv5_1_1x1_increase_bn(conv5_1_1x1_increase)
        conv5_1_1x1_proj = self.conv5_1_1x1_proj(conv4_6x)
        conv5_1_1x1_proj_bn = self.conv5_1_1x1_proj_bn(conv5_1_1x1_proj)
        conv5_1 = torch.add(conv5_1_1x1_proj_bn, 1, conv5_1_1x1_increase_bn)
        conv5_1x = self.conv5_1_relu(conv5_1)
        conv5_2_1x1_reduce = self.conv5_2_1x1_reduce(conv5_1x)
        conv5_2_1x1_reduce_bn = self.conv5_2_1x1_reduce_bn(conv5_2_1x1_reduce)
        conv5_2_1x1_reduce_bnxx = self.conv5_2_1x1_reduce_relu(conv5_2_1x1_reduce_bn)
        conv5_2_3x3 = self.conv5_2_3x3(conv5_2_1x1_reduce_bnxx)
        conv5_2_3x3_bn = self.conv5_2_3x3_bn(conv5_2_3x3)
        conv5_2_3x3_bnxx = self.conv5_2_3x3_relu(conv5_2_3x3_bn)
        conv5_2_1x1_increase = self.conv5_2_1x1_increase(conv5_2_3x3_bnxx)
        conv5_2_1x1_increase_bn = self.conv5_2_1x1_increase_bn(conv5_2_1x1_increase)
        conv5_2 = torch.add(conv5_1x, 1, conv5_2_1x1_increase_bn)
        conv5_2x = self.conv5_2_relu(conv5_2)
        conv5_3_1x1_reduce = self.conv5_3_1x1_reduce(conv5_2x)
        conv5_3_1x1_reduce_bn = self.conv5_3_1x1_reduce_bn(conv5_3_1x1_reduce)
        conv5_3_1x1_reduce_bnxx = self.conv5_3_1x1_reduce_relu(conv5_3_1x1_reduce_bn)
        conv5_3_3x3 = self.conv5_3_3x3(conv5_3_1x1_reduce_bnxx)
        conv5_3_3x3_bn = self.conv5_3_3x3_bn(conv5_3_3x3)
        conv5_3_3x3_bnxx = self.conv5_3_3x3_relu(conv5_3_3x3_bn)
        conv5_3_1x1_increase = self.conv5_3_1x1_increase(conv5_3_3x3_bnxx)
        conv5_3_1x1_increase_bn = self.conv5_3_1x1_increase_bn(conv5_3_1x1_increase)
        conv5_3 = torch.add(conv5_2x, 1, conv5_3_1x1_increase_bn)
        conv5_3x = self.conv5_3_relu(conv5_3)
        pool5_7x7_s1 = self.pool5_7x7_s1(conv5_3x)
        classifier_preflatten = self.classifier(pool5_7x7_s1)
        classifier = classifier_preflatten.view(classifier_preflatten.size(0), -1)

        # return classifier, pool5_7x7_s1
        return classifier


def resnet50_pretrained_vgg(pretrained=True, progress=True, **kwargs):
    return resnet50_pretrained_vgg()