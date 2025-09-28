import math

import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader


class Block(nn.Module):
    """模型核心模块设计类"""
    def __init__(self, in_channels, conv_channels):
        """
        :param in_channels: 输入通道数
        :param conv_channels: 输出通道数
        """
        super().__init__()

        # （3，3，3）大小的卷积核，填充一个像素，需要偏置
        self.conv1 = nn.Conv3d(in_channels, conv_channels, kernel_size=3, padding=1, bias=True)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv3d(conv_channels, conv_channels, kernel_size=3, padding=1, bias=True)
        self.relu2 = nn.ReLU(inplace=True)

        # （2，2，2）大小的最大池化层，步幅为2
        self.maxpool = nn.MaxPool3d(2, 2)

    def forward(self, input):

        out = self.conv1(input)
        out = self.relu1(out)
        out = self.conv2(out)
        out = self.relu2(out)

        return self.maxpool(out)



class LunaModel(nn.Module):

    def __init__(self, in_channels=1, conv_channels=8):
        super().__init__()

        self.batchnorm = nn.BatchNorm3d(in_channels)

        self.block1 = Block(in_channels, conv_channels)
        self.block2 = Block(conv_channels, conv_channels * 2)
        self.block3 = Block(conv_channels * 2, conv_channels * 4)
        self.block4 = Block(conv_channels * 4, conv_channels * 8)

        self.linear = nn.Linear(1152, 2)
        self.softmax = nn.Softmax(dim=1)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                if type(m) in {nn.Linear, nn.Conv3d}:
                    nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_out', nonlinearity='relu')
                    if m.bias is not None:
                        fan_in , fan_out = nn.init._calculate_fan_in_and_fan_out(m.weight.data)
                        bound = 1 / math.sqrt(fan_out)
                        nn.init.uniform_(m.bias, -bound, bound)



    def forward(self, input):

        out = self.batchnorm(input)

        out = self.block1(out)
        out = self.block2(out)
        out = self.block3(out)
        out = self.block4(out)

        linear_input = out.view(out.size(0), -1)

        linear_out = self.linear(linear_input)
        output = self.softmax(linear_out)

        return linear_out, output







