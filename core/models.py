# -*- coding: utf-8 -*-
# @Time    : 2019/2/27 16:43
# @Author  : liuqixuan_i
# @Email   : liuqixuan_i@didiglobal.com
# @File    : models.py
# @Software: PyCharm

import torch as t
from . import vgg as CNNs
from torch import nn
from torch.nn import Module, functional as F, Parameter as P
from .config import Config


class ActionNet(Module):
    def __init__(self, config: Config):
        super(ActionNet, self).__init__()
        self.config = config
        self.cnn = CNNs.vgg13_bn(pretrained=True, num_classes=1000)
        self.rnn = nn.LSTM(
            # input_size=config.image_resize[0] * config.image_resize[1] * 3,
            input_size=1000,
            hidden_size=config.hidden_size,
            num_layers=config.lstm_layers,
            batch_first=True,
            dropout=config.dropout)
        self.classifier = nn.Linear(config.hidden_size, config.num_classes)

    def forward(self, input, h_state=None):
        # type:(t.Tensor,t.Tensor)->t.Tensor
        bc, seq, c, h, w = input.shape
        input = input.view(bc * seq, c, h, w)
        input = self.cnn(input)
        input = input.view(bc, seq, -1)
        output, hn = self.rnn(input, h_state)
        output = self.classifier(output[:, -1, :])
        return output
