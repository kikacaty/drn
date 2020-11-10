from torch import nn
from torchvision import models
import math
import torch


class DUC(nn.Module):
    def __init__(self, in_dim, label_num, aspp_num, aspp_stride, down_factor):
        super(DUC, self).__init__()
        self.all_conv_layers = nn.ModuleList()
        self.pixel_shuffle = nn.PixelShuffle(down_factor)
        for i in range(aspp_num):
            pad = (i + 1) * aspp_stride
            dilate = pad
            self.all_conv_layers.append(
                nn.Conv2d(in_dim, (down_factor ** 2) * label_num, kernel_size=3,
                          padding=pad, dilation=dilate)
            )
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))

    def forward(self, x):
        aspp_list = list()
        for conv_layer in self.all_conv_layers:
            aspp_list.append(conv_layer(x))
        aspp_sum = sum(aspp_list)
        out = self.pixel_shuffle(aspp_sum)
        return out


class ResNetDUCHDC(nn.Module):
    def __init__(self, in_dim=2048, label_num=19, aspp_num=4, aspp_stride=6,
                 down_factor=8, mode='bigger'):
        super(ResNetDUCHDC, self).__init__()
        resnet = models.resnet101()
        resnet.load_state_dict(torch.load('resnet101-5d3b4d8f.pth'))
        # resnet = models.resnet50()
        # resnet.load_state_dict(torch.load('resnet50-19c8e357.pth'))
        # self.group1 = nn.Sequential(nn.Conv2d(3, 64, kernel_size=3,
        #                                       padding=1, stride=2),
        #                             nn.BatchNorm2d(64),
        #                             nn.ReLU(inplace=True),
        #                             nn.Conv2d(64, 64, kernel_size=3, padding=1),
        #                             nn.BatchNorm2d(64),
        #                             nn.ReLU(inplace=True),
        #                             nn.Conv2d(64, 64, kernel_size=3,
        #                                       padding=1),
        #                             nn.BatchNorm2d(64),
        #                             nn.ReLU(inplace=True),
        #                             resnet.maxpool
        #                             )
        self.group1= nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool)
        self.group2 = resnet.layer1
        self.group3 = resnet.layer2
        self.group4 = resnet.layer3
        self.group5 = resnet.layer4
        self.softmax = nn.LogSoftmax()

        self.group4[0].conv2.stride = (1, 1)
        self.group4[0].downsample[0].stride = (1, 1)
        self.group5[0].conv2.stride = (1, 1)
        self.group5[0].downsample[0].stride = (1, 1)

        if mode == 'duc_hdc_bigger':
            group4_dilation = [1, 2, 5, 9]
            for idx in range(len(self.group4)):
                dilation = (group4_dilation[idx % len(group4_dilation)],
                            group4_dilation[idx % len(group4_dilation)])
                padding = dilation
                self.group4[idx].conv2.dilation = dilation
                self.group4[idx].conv2.padding = padding

            group5_dilation = [5, 9, 17]
            for idx in range(len(self.group5)):
                dilation = (group5_dilation[idx % len(group5_dilation)],
                            group5_dilation[idx % len(group5_dilation)])
                padding = dilation
                self.group5[idx].conv2.dilation = dilation
                self.group5[idx].conv2.padding = padding
        elif mode == 'duc_hdc_rf':
            group4_dilation = [1, 2, 3]
            for idx in range(len(self.group4)):
                dilation = (group4_dilation[idx % len(group4_dilation)],
                            group4_dilation[idx % len(group4_dilation)])
                padding = dilation
                self.group4[idx].conv2.dilation = dilation
                self.group4[idx].conv2.padding = padding
            self.group4[-2].conv2.dilation = 2
            self.group4[-2].conv2.padding = 2

            group5_dilation = [3, 4, 5]
            for idx in range(len(self.group5)):
                dilation = (group5_dilation[idx % len(group5_dilation)],
                            group5_dilation[idx % len(group5_dilation)])
                padding = dilation
                self.group5[idx].conv2.dilation = dilation
                self.group5[idx].conv2.padding = padding
        elif mode == 'duc_hdc_no':
            pass
        else:
            assert mode in ['duc_hdc_bigger', 'duc_hdc_rf', 'duc_hdc_no']


        self.duc = DUC(in_dim, label_num, aspp_num, aspp_stride, down_factor)


    def forward(self, x):
        x = self.group1(x)
        x = self.group2(x)
        x = self.group3(x)
        x = self.group4(x)
        x = self.group5(x)
        # return self.duc(x)
        x_ = self.duc(x)
        return self.softmax(x_), x

    def optim_parameters(self, memo=None):
        for param in self.group1.parameters():
            yield param
        for param in self.group2.parameters():
            yield param
        for param in self.group3.parameters():
            yield param
        for param in self.group4.parameters():
            yield param
        for param in self.group5.parameters():
            yield param
        for param in self.duc.parameters():
            yield param

