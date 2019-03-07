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
from config import Config


class ActionNet(Module):
    def __init__(self, config: Config):
        super(ActionNet, self).__init__()
        self.config = config
        # self.cnn = CNNs.vgg13_bn(pretrained=False, num_classes=1000)
        self.rnn = nn.LSTM(
            input_size=config.image_resize[0] * config.image_resize[1] * 3,
            hidden_size=config.hidden_size,
            num_layers=config.lstm_layers,
            batch_first=True,
            dropout=config.dropout)
        self.classifier = nn.Linear(config.hidden_size, 1)

    def forward(self, input, h0=None):
        output, hn = self.rnn(input, h0)
        output = self.classifier(output)
        return output, hn


if __name__ == "__main__":
    from random import randint
    from dataset import ActionDataset

    config = Config("train")
    ds = ActionDataset(config.train_data_path, config.classes, config.image_resize)
    label, data = ds[randint(0, len(ds))]
    data.unsqueeze_(0)
    model = ActionNet(config)
    if config.use_gpu:
        data = data.cuda()
        model = model.cuda()
    bs, seq, c, h, w = data.shape
    data = data.view(bs, seq, c * h * w)
    ret, hn = model(data)
    ret = ret.detach().cpu().numpy()
    h0 = hn[0].detach().cpu().numpy()
    print(config.classes[label], ret, h0)
