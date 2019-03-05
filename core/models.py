# -*- coding: utf-8 -*-
# @Time    : 2019/2/27 16:43
# @Author  : liuqixuan_i
# @Email   : liuqixuan_i@didiglobal.com
# @File    : models.py
# @Software: PyCharm

import torch as t
from torch import nn
from torchvision import models as CNNs
from torch.nn import Module, functional as F, Parameter as P
from core.config import Config


class ActionNet(Module):
    def __init__(self, config: Config):
        self.config = config
        self.cnn = CNNs.vgg13_bn(pretrained=True, num_classes=1000)
        self.rnn = nn.LSTM(config.image_resize[0] * config.image_resize[1] * 3, 30, 2)

    def forward(self, input,h0=None):
        output,hn = self.rnn(input,h0)
        return output,hn



if __name__ == "__main__":
    pass
