# -*- coding: utf-8 -*-
# @Time    : 2019/3/1 17:07
# @Author  : liuqixuan_i
# @Email   : liuqixuan_i@didiglobal.com
# @File    : dataset.py
# @Software: PyCharm

import os
import csv
import cv2
import numpy as np
import torch as t
from torch.utils.data import Dataset, DataLoader
from torch.nn import functional as F
from config import Config


class ActionDataset(Dataset):
    def __init__(self, path: str, classes: list, resize: tuple):
        """
        :param path: datalist_file file inited by prepare_data.prepare
        :param classes: classes in config.Config
        :param resize: Height * Width tuple
        """
        assert os.path.isfile(path), "file %s not exist" % path
        self.class_dict = {}
        for i, cls in enumerate(classes):
            self.class_dict[cls] = i
        self.resize = resize
        self.data = []
        with open(path, "r") as f:
            reader = csv.reader(f)
            for row in reader:
                self.data.append([self.class_dict[row[0]], row[1]])

    def __getitem__(self, idx):
        """
        get data item
        :param idx: int index
        :return: (label, data) label is an int, data is a [sequence, channel, height, width] tensor
        """
        label, path = self.data[idx]
        data = np.load(path)
        data = data['arr_0']
        data = t.from_numpy(data).to(t.float).permute(0, 3, 1, 2)
        data = F.interpolate(data, size=self.resize, mode='nearest')
        return label, data

    def __len__(self):
        return len(self.data)


def ActionDataloader(data_type, config):
    # type:(str,Config)->DataLoader
    assert data_type in ["train", "training", "val", "validation", "inference"]
    if data_type in ["train", "training"]:
        data_path = config.train_data_path
        shuffle = config.shuffle_train
        drop_last = config.drop_last_train
    elif data_type in ["val", "validation", "inference"]:
        data_path = config.val_data_path
        shuffle = config.shuffle_val
        drop_last = config.drop_last_val
    dataset = ActionDataset(data_path, config.classes)
    assert len(dataset) > config.batch_size
    return DataLoader(dataset, config.batch_size, shuffle, num_workers=config.num_data_workers,
                      pin_memory=config.pin_memory, drop_last=drop_last, timeout=config.time_out)


if __name__ == "__main__":
    from random import randint

    config = Config("train")
    ds = ActionDataset(config.train_data_path, config.classes, (720, 1280))
    label, data = ds[randint(0, len(ds))]
    for i in range(data.shape[0] // 3):
        frame = data[i, ...].permute(1, 2, 0).numpy()
        cv2.imshow(str(label), frame)
        cv2.waitKey(66)
    print(config.classes[label], data.shape)
